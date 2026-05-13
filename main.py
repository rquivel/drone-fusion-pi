"""drone-fusion-pi entry point.

Starts the audio + image detector threads and the GPIO controller, then
blocks until SIGINT/SIGTERM. On shutdown, all GPIOs are driven low and
hardware resources are released.

Usage:
    python main.py
    python main.py --mock-gpio              # bench-test on a non-Pi machine
    python main.py --no-image               # audio only (e.g. while debugging)
    python main.py --no-audio               # image only
    python main.py --stream-port 8000       # live MJPEG of annotated frames at
                                            # http://<pi>:8000/  (debug only)
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

import config as C
from gpio_controller import GpioController


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, C.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock-gpio", action="store_true",
                    help="Don't touch real GPIO pins; just log transitions.")
    ap.add_argument("--no-audio", action="store_true",
                    help="Disable the audio detector (image only).")
    ap.add_argument("--no-image", action="store_true",
                    help="Disable the image detector (audio only).")
    ap.add_argument("--device", default="cpu",
                    help="Inference device: cpu | cuda | mps (default: cpu).")
    ap.add_argument("--stream-port", type=int, default=None,
                    help="If set, also serve annotated camera frames over MJPEG "
                         "at http://<pi>:<port>/stream.mjpg . Off by default "
                         "(adds ~30%% CPU; do not leave on in production).")
    args = ap.parse_args()

    setup_logging()
    log = logging.getLogger("main")

    if args.no_audio and args.no_image:
        log.error("nothing to do — both detectors disabled")
        return 2

    log.info("audio threshold: %.2f   image threshold: %.2f   hold: %.1fs",
             C.AUDIO_THRESHOLD, C.IMAGE_THRESHOLD, C.HOLD_SECONDS)
    log.info("GPIO pins (BCM): audio=%d image=%d fused=%d  hold=%.1fs",
             C.GPIO_AUDIO, C.GPIO_IMAGE, C.GPIO_FUSED, C.HOLD_SECONDS)

    gpio = GpioController(mock=args.mock_gpio)
    gpio.start()

    # Optional debug video stream
    stream = None
    if args.stream_port is not None and not args.no_image:
        from debug_stream import DebugStream
        stream = DebugStream(port=args.stream_port)
        stream.start()

    detectors = []
    if not args.no_audio:
        from audio_detector import AudioDetector
        ad = AudioDetector(on_detection=gpio.report_audio, device=args.device)
        ad.start()
        detectors.append(ad)
    if not args.no_image:
        from image_detector import ImageDetector
        idt = ImageDetector(
            on_detection=gpio.report_image,
            device=args.device,
            on_frame=(stream.push_jpeg if stream is not None else None),
        )
        idt.start()
        detectors.append(idt)

    stop = {"flag": False}

    def _on_signal(signum, frame):
        log.info("received signal %d, shutting down", signum)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        while not stop["flag"]:
            time.sleep(0.5)
            for d in detectors:
                if not d.is_alive():
                    log.error("%s thread died; exiting", d.name)
                    stop["flag"] = True
                    break
    finally:
        log.info("stopping detectors...")
        for d in detectors:
            d.stop()
        for d in detectors:
            d.join(timeout=3.0)
        if stream is not None:
            log.info("stopping debug stream...")
            stream.stop()
        log.info("stopping GPIO controller...")
        gpio.stop()
        log.info("bye")

    return 0


if __name__ == "__main__":
    sys.exit(main())
