"""Flash the three GPIO-driven LEDs in a sequence to verify wiring.

Walks through:
    1. all off                  -> baseline (none should light up)
    2. audio LED only           -> GPIO17 (BCM)
    3. image LED only           -> GPIO27 (BCM)
    4. fused LED only           -> GPIO22 (BCM)
    5. all three on             -> confirms simultaneous drive

Then repeats. Hit Ctrl-C to stop; the script drives all three pins low and
releases the GPIO resources on the way out.

Usage:
    python scripts/flash_leds.py
    python scripts/flash_leds.py --interval 0.5 --cycles 5
    python scripts/flash_leds.py --mock-gpio       # bench-test on a Mac
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make the project's ``config`` importable when running from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config as C  # noqa: E402


log = logging.getLogger("flash")


# --- GPIO setup -----------------------------------------------------------

class _MockOut:
    """Stand-in for gpiozero.LED on systems without GPIO support."""
    def __init__(self, pin: int, label: str):
        self.pin = pin
        self.label = label

    def on(self):
        log.info("[mock] %-6s GPIO%-2d HIGH", self.label, self.pin)

    def off(self):
        log.info("[mock] %-6s GPIO%-2d LOW", self.label, self.pin)

    def close(self):
        pass


def make_led(pin: int, label: str, mock: bool):
    if mock:
        return _MockOut(pin, label)
    try:
        from gpiozero import LED
        from gpiozero.pins.lgpio import LGPIOFactory
        from gpiozero import Device
        if not isinstance(Device.pin_factory, LGPIOFactory):
            Device.pin_factory = LGPIOFactory()
        led = LED(pin, active_high=True, initial_value=False)
        log.info("opened hardware LED on GPIO%-2d (%s)", pin, label)
        return led
    except Exception as e:
        log.warning("falling back to mock GPIO for %s (reason: %s)", label, e)
        return _MockOut(pin, label)


# --- Test sequence --------------------------------------------------------

def run_sequence(audio, image, fused, interval: float, cycles: int | None):
    steps = [
        ("all off",     [],                       "baseline; all LEDs dark"),
        ("audio only",  [audio],                  f"GPIO{C.GPIO_AUDIO} HIGH"),
        ("image only",  [image],                  f"GPIO{C.GPIO_IMAGE} HIGH"),
        ("fused only",  [fused],                  f"GPIO{C.GPIO_FUSED} HIGH"),
        ("all on",      [audio, image, fused],    "all three HIGH"),
    ]

    cycle = 0
    while cycles is None or cycle < cycles:
        cycle += 1
        log.info("--- cycle %d ---", cycle)
        for name, on_now, hint in steps:
            for led in (audio, image, fused):
                if led in on_now:
                    led.on()
                else:
                    led.off()
            log.info("  %-10s %s", name, hint)
            time.sleep(interval)


def all_off(*leds):
    for led in leds:
        try:
            led.off()
        except Exception:
            pass


# --- Entry point ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=1.0,
                    help="Seconds per step (default 1.0)")
    ap.add_argument("--cycles", type=int, default=None,
                    help="How many full cycles to run (default: forever)")
    ap.add_argument("--mock-gpio", action="store_true",
                    help="Don't touch real GPIO; log transitions instead")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("pins (BCM): audio=%d  image=%d  fused=%d",
             C.GPIO_AUDIO, C.GPIO_IMAGE, C.GPIO_FUSED)
    audio = make_led(C.GPIO_AUDIO, "audio", args.mock_gpio)
    image = make_led(C.GPIO_IMAGE, "image", args.mock_gpio)
    fused = make_led(C.GPIO_FUSED, "fused", args.mock_gpio)

    try:
        run_sequence(audio, image, fused, args.interval, args.cycles)
    except KeyboardInterrupt:
        log.info("interrupted; turning all LEDs off")
    finally:
        all_off(audio, image, fused)
        for led in (audio, image, fused):
            if hasattr(led, "close"):
                led.close()
        log.info("done")


if __name__ == "__main__":
    main()
