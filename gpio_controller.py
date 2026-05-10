"""GPIO output controller with hold-time and fusion.

State model:
    - Each detector calls report_audio() / report_image() whenever it sees a drone.
    - update() runs continuously and drives the three pins:
        AUDIO pin = high if last audio detection was less than HOLD_SECONDS ago
        IMAGE pin = high if last image detection was less than HOLD_SECONDS ago
        FUSED pin = AUDIO pin AND IMAGE pin

Pi 5 / DietPi: gpiozero with the lgpio backend is the only thing that works
out of the box. ``RPi.GPIO`` does not work on the Pi 5 because it lacks
``/dev/gpiomem``. This module gracefully falls back to mock mode if no GPIO
backend is available, so audio + image pipelines can be tested on a dev box.
"""
from __future__ import annotations

import logging
import threading
import time

import config as C

log = logging.getLogger("gpio")


class _MockOut:
    """Stand-in for gpiozero.LED on systems without GPIO support."""
    def __init__(self, pin: int):
        self.pin = pin
        self.value = 0

    def on(self):
        if self.value != 1:
            log.info("[mock] GPIO%d -> HIGH", self.pin)
        self.value = 1

    def off(self):
        if self.value != 0:
            log.info("[mock] GPIO%d -> LOW", self.pin)
        self.value = 0

    def close(self):
        self.value = 0


def _make_output(pin: int, mock: bool):
    if mock:
        log.info("creating MOCK output on GPIO%d", pin)
        return _MockOut(pin)
    try:
        # Importing here so the module loads on dev machines without gpiozero.
        from gpiozero import LED
        from gpiozero.pins.lgpio import LGPIOFactory
        from gpiozero import Device
        if not isinstance(Device.pin_factory, LGPIOFactory):
            Device.pin_factory = LGPIOFactory()
        led = LED(pin, active_high=True, initial_value=False)
        log.info("created hardware output on GPIO%d", pin)
        return led
    except Exception as e:
        log.warning("falling back to mock GPIO (reason: %s)", e)
        return _MockOut(pin)


class GpioController:
    def __init__(self, mock: bool = False):
        self.audio_out = _make_output(C.GPIO_AUDIO, mock)
        self.image_out = _make_output(C.GPIO_IMAGE, mock)
        self.fused_out = _make_output(C.GPIO_FUSED, mock)
        self.mock = mock

        # monotonic timestamps of the last detection from each detector.
        self._last_audio = 0.0
        self._last_image = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- detector callbacks ---------------------------------------------
    def report_audio(self, prob: float = 1.0):
        with self._lock:
            self._last_audio = time.monotonic()

    def report_image(self, conf: float = 1.0):
        with self._lock:
            self._last_image = time.monotonic()

    # -- update loop ----------------------------------------------------
    def _step(self) -> tuple[bool, bool, bool]:
        now = time.monotonic()
        with self._lock:
            audio_active = (now - self._last_audio) < C.HOLD_SECONDS
            image_active = (now - self._last_image) < C.HOLD_SECONDS
        fused_active = audio_active and image_active

        (self.audio_out.on if audio_active else self.audio_out.off)()
        (self.image_out.on if image_active else self.image_out.off)()
        (self.fused_out.on if fused_active else self.fused_out.off)()
        return audio_active, image_active, fused_active

    def _run(self):
        while not self._stop.is_set():
            try:
                self._step()
            except Exception:
                log.exception("GPIO update step failed")
            self._stop.wait(C.GPIO_UPDATE_INTERVAL)

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run,
                                        name="gpio-controller",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        for out in (self.audio_out, self.image_out, self.fused_out):
            try:
                out.off()
                if hasattr(out, "close"):
                    out.close()
            except Exception:
                pass
