"""Continuous audio drone detection.

Opens the default microphone at 16 kHz mono via sounddevice, keeps a rolling
1-second ring buffer, runs CRNN inference every AUDIO_HOP_SECONDS, and calls
on_detection() with the drone probability when it crosses AUDIO_THRESHOLD.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio

import config as C
from audio_model import load_audio_model

log = logging.getLogger("audio")


class AudioDetector(threading.Thread):
    def __init__(self, on_detection: Callable[[float], None], device: str = "cpu"):
        super().__init__(daemon=True, name="audio-detector")
        self.on_detection = on_detection
        self.device = device
        self._stop = threading.Event()

        self.window_samples = int(C.AUDIO_WINDOW_SECONDS * C.AUDIO_SAMPLE_RATE)
        self.hop_samples = int(C.AUDIO_HOP_SECONDS * C.AUDIO_SAMPLE_RATE)
        self.block_samples = int(C.AUDIO_BLOCK_SECONDS * C.AUDIO_SAMPLE_RATE)

        # Ring buffer for the last ``window_samples`` of mono audio.
        self._buffer = np.zeros(self.window_samples, dtype=np.float32)
        self._lock = threading.Lock()

        self.model = load_audio_model(device=self.device)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=C.AUDIO_SAMPLE_RATE,
            n_fft=C.AUDIO_N_FFT,
            hop_length=C.AUDIO_HOP_LENGTH,
            n_mels=C.AUDIO_N_MELS,
            f_min=C.AUDIO_F_MIN,
            f_max=C.AUDIO_F_MAX,
            power=2.0,
        ).to(self.device)
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0).to(self.device)

    # -- audio callback --------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            log.warning("sounddevice status: %s", status)
        # indata shape: (frames, channels). We want mono float32.
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata.mean(axis=1)
        else:
            mono = indata[:, 0] if indata.ndim == 2 else indata
        mono = mono.astype(np.float32, copy=False)
        with self._lock:
            n = mono.shape[0]
            if n >= self.window_samples:
                self._buffer[:] = mono[-self.window_samples :]
            else:
                self._buffer = np.roll(self._buffer, -n)
                self._buffer[-n:] = mono

    # -- inference helpers -----------------------------------------------
    def _snapshot(self) -> np.ndarray:
        with self._lock:
            return self._buffer.copy()

    @torch.no_grad()
    def _classify(self, waveform: np.ndarray) -> float:
        wav = torch.from_numpy(waveform).to(self.device)
        spec = self.melspec(wav)
        spec = self.amp_to_db(spec)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        spec = spec.unsqueeze(0).unsqueeze(0)   # (1, 1, n_mels, T)
        logits = self.model(spec)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        return float(probs[1])  # index 1 = yes_drone

    # -- main loop -------------------------------------------------------
    def run(self):
        log.info("starting input stream @ %d Hz mono", C.AUDIO_SAMPLE_RATE)
        try:
            with sd.InputStream(
                samplerate=C.AUDIO_SAMPLE_RATE,
                channels=C.AUDIO_CHANNELS,
                blocksize=self.block_samples,
                dtype="float32",
                device=C.AUDIO_DEVICE,
                callback=self._audio_callback,
            ):
                # Warm-up: let the buffer fill before classifying.
                time.sleep(C.AUDIO_WINDOW_SECONDS)
                while not self._stop.is_set():
                    waveform = self._snapshot()
                    prob = self._classify(waveform)
                    if prob >= C.AUDIO_THRESHOLD:
                        log.info("DRONE detected (prob=%.3f)", prob)
                        self.on_detection(prob)
                    else:
                        log.debug("no drone (prob=%.3f)", prob)
                    self._stop.wait(C.AUDIO_HOP_SECONDS)
        except Exception:
            log.exception("audio detector crashed")
            raise

    def stop(self):
        self._stop.set()
