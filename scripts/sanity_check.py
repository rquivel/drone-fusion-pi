"""Quick hardware sanity check: mic, camera, GPIO.

Run on the Pi after pip-installing requirements. Each section is independent;
failures are reported but don't abort the others.

    python scripts/sanity_check.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make ``config`` importable when running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C


def check_audio():
    print("== audio ==")
    try:
        import sounddevice as sd
        import numpy as np
        print("default input :", sd.query_devices(kind="input")["name"])
        print(f"recording {C.AUDIO_WINDOW_SECONDS}s @ {C.AUDIO_SAMPLE_RATE} Hz...")
        rec = sd.rec(int(C.AUDIO_WINDOW_SECONDS * C.AUDIO_SAMPLE_RATE),
                     samplerate=C.AUDIO_SAMPLE_RATE,
                     channels=1, dtype="float32",
                     device=C.AUDIO_DEVICE)
        sd.wait()
        rms = float(np.sqrt(np.mean(rec ** 2)))
        peak = float(np.max(np.abs(rec)))
        print(f"  RMS={rms:.4f}  peak={peak:.4f}  ({'silence?' if rms < 1e-4 else 'OK'})")
    except Exception as e:
        print(f"  FAIL: {e}")


def check_camera():
    print("== camera ==")
    try:
        import cv2
        cap = cv2.VideoCapture(C.CAMERA_INDEX)
        if not cap.isOpened():
            print(f"  FAIL: cv2.VideoCapture({C.CAMERA_INDEX}) didn't open")
            return
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print("  FAIL: read() returned no frame")
            return
        h, w = frame.shape[:2]
        out_path = Path(__file__).parent / "test_frame.jpg"
        cv2.imwrite(str(out_path), frame)
        print(f"  captured {w}x{h} frame -> {out_path}")
    except Exception as e:
        print(f"  FAIL: {e}")


def check_gpio():
    print("== GPIO ==")
    try:
        from gpiozero import LED
        led = LED(C.GPIO_AUDIO)
        led.on(); time.sleep(0.2); led.off()
        led.close()
        print(f"  GPIO{C.GPIO_AUDIO} blink OK")
    except Exception as e:
        print(f"  FAIL: {e}")


def check_models():
    print("== model weights ==")
    for name, path in [("audio CRNN", C.AUDIO_WEIGHTS),
                       ("YOLOv8", C.YOLO_WEIGHTS)]:
        exists = Path(path).exists()
        size = Path(path).stat().st_size if exists else 0
        status = f"{size/1e6:.1f} MB" if exists else "MISSING"
        print(f"  {name:<11} {path}  [{status}]")


if __name__ == "__main__":
    check_models()
    check_audio()
    check_camera()
    check_gpio()
