"""Record audio from the mic and show what the CRNN model thinks of it.

Captures N seconds (default 60) at the same 16 kHz mono format the model
was trained on, saves the recording as a .wav, then runs sliding-window
inference and writes a per-window drone-probability CSV so you can see
exactly what the model is hearing.

Default output goes to recordings/<timestamp>/:
    recording.wav     — the raw audio you can play back on any computer
    predictions.csv   — t_start, t_end, drone_prob per window
    summary.txt       — one-paragraph stats

Stop the systemd service first — it holds the mic exclusively:

    sudo systemctl stop drone-fusion.service
    source venv/bin/activate
    python scripts/record_audio_sample.py
    # play recording.wav on your machine, look at predictions.csv
    sudo systemctl start drone-fusion.service

Usage:
    python scripts/record_audio_sample.py
    python scripts/record_audio_sample.py --seconds 30 --hop 0.25
    python scripts/record_audio_sample.py --output-dir /tmp/my-rec
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio

# Make project modules importable when running from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config as C  # noqa: E402
from audio_model import load_audio_model  # noqa: E402


def record(seconds: float) -> np.ndarray:
    n_samples = int(seconds * C.AUDIO_SAMPLE_RATE)
    print(f"recording {seconds}s @ {C.AUDIO_SAMPLE_RATE} Hz mono ...",
          flush=True)
    t0 = time.time()
    audio = sd.rec(n_samples, samplerate=C.AUDIO_SAMPLE_RATE,
                   channels=1, dtype="float32", device=C.AUDIO_DEVICE)
    sd.wait()
    print(f"  done in {time.time() - t0:.1f}s", flush=True)
    return audio[:, 0]


def make_melspec_fn(device):
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=C.AUDIO_SAMPLE_RATE,
        n_fft=C.AUDIO_N_FFT,
        hop_length=C.AUDIO_HOP_LENGTH,
        n_mels=C.AUDIO_N_MELS,
        f_min=C.AUDIO_F_MIN,
        f_max=C.AUDIO_F_MAX,
        power=2.0,
    ).to(device)
    amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0).to(device)

    def transform(window: torch.Tensor) -> torch.Tensor:
        spec = melspec(window)
        spec = amp_to_db(spec)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        return spec.unsqueeze(0).unsqueeze(0)   # (1, 1, n_mels, T)

    return transform


def classify_windows(audio: np.ndarray, hop_seconds: float, device: str):
    """Slide a 1-second window with the given hop; return (starts_s, probs)."""
    model = load_audio_model(device=device)
    transform = make_melspec_fn(device)

    win = int(C.AUDIO_WINDOW_SECONDS * C.AUDIO_SAMPLE_RATE)
    hop = max(1, int(hop_seconds * C.AUDIO_SAMPLE_RATE))
    starts = list(range(0, max(1, audio.shape[0] - win + 1), hop))
    probs = np.empty(len(starts), dtype=np.float32)

    with torch.no_grad():
        for i, s in enumerate(starts):
            window = audio[s : s + win]
            wav = torch.from_numpy(window).to(device)
            spec = transform(wav)
            logits = model(spec)
            p = F.softmax(logits, dim=1).cpu().numpy()[0]
            probs[i] = p[1]   # index 1 = yes_drone
    starts_s = np.array(starts, dtype=np.float64) / C.AUDIO_SAMPLE_RATE
    return starts_s, probs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="Recording length (default 60)")
    ap.add_argument("--hop", type=float, default=C.AUDIO_HOP_SECONDS,
                    help=f"Seconds between inference windows (default {C.AUDIO_HOP_SECONDS})")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write outputs (default: recordings/<timestamp>/)")
    ap.add_argument("--device", default="cpu",
                    help="Inference device for the model (default: cpu)")
    args = ap.parse_args()

    # 1. Record
    audio = record(args.seconds)

    # 2. Save WAV
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = PROJECT_ROOT / "recordings" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path = out_dir / "recording.wav"
    sf.write(str(wav_path), audio, C.AUDIO_SAMPLE_RATE, subtype="PCM_16")
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    print(f"saved {wav_path}  (RMS={rms:.4f}, peak={peak:.4f})")
    if rms < 1e-4:
        print("  ! mic level is essentially silent — check the input device "
              "and gain before trusting predictions")

    # 3. Classify with the CRNN
    print(f"running model inference (hop={args.hop}s)...")
    starts_s, probs = classify_windows(audio, args.hop, device=args.device)

    # 4. Save predictions CSV
    csv_path = out_dir / "predictions.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_start_s", "t_end_s", "drone_prob"])
        for s, p in zip(starts_s, probs):
            w.writerow([f"{s:.3f}", f"{s + C.AUDIO_WINDOW_SECONDS:.3f}",
                        f"{p:.4f}"])
    print(f"saved {csv_path}  ({len(probs)} windows)")

    # 5. Summary
    n_above = int((probs >= C.AUDIO_THRESHOLD).sum())
    pct = 100.0 * n_above / len(probs) if len(probs) else 0
    summary = (
        f"recording      : {wav_path.name}\n"
        f"duration       : {args.seconds:.1f}s\n"
        f"sample rate    : {C.AUDIO_SAMPLE_RATE} Hz mono\n"
        f"input RMS/peak : {rms:.4f} / {peak:.4f}\n"
        f"windows        : {len(probs)} @ {C.AUDIO_WINDOW_SECONDS}s "
        f"with {args.hop}s hop\n"
        f"drone_prob max : {probs.max():.4f}\n"
        f"drone_prob mean: {probs.mean():.4f}\n"
        f"drone_prob med : {np.median(probs):.4f}\n"
        f"above threshold {C.AUDIO_THRESHOLD}: "
        f"{n_above}/{len(probs)} windows ({pct:.1f}%)\n"
    )
    print("\n" + summary)
    (out_dir / "summary.txt").write_text(summary)

    # 6. Quick highlights — top 5 most drone-like moments
    print("top 5 highest-probability windows:")
    order = np.argsort(-probs)[:5]
    for rank, idx in enumerate(order, 1):
        t = starts_s[idx]
        mm, ss = int(t // 60), t - 60 * int(t // 60)
        print(f"  #{rank}  t={t:6.2f}s  ({mm}:{ss:05.2f})  "
              f"prob={probs[idx]:.4f}")

    print(f"\nplay {wav_path} on your laptop and compare against {csv_path}")


if __name__ == "__main__":
    main()
