"""Continuous image drone detection using YOLOv8.

Opens the USB camera, runs the trained YOLO model every IMAGE_FRAME_STRIDE
frames, and calls on_detection(conf) when a 'drone' class box is seen at or
above IMAGE_THRESHOLD.

If on_frame is provided, every analyzed frame's annotated JPEG bytes are
pushed there (used by the optional debug stream).
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import cv2
from ultralytics import YOLO

import config as C

log = logging.getLogger("image")


class ImageDetector(threading.Thread):
    def __init__(self, on_detection: Callable[[float], None],
                 device: str = "cpu",
                 on_frame: Optional[Callable[[bytes], None]] = None):
        super().__init__(daemon=True, name="image-detector")
        self.on_detection = on_detection
        self.on_frame = on_frame
        self.device = device
        self._stop = threading.Event()

        log.info("loading YOLO weights from %s", C.YOLO_WEIGHTS)
        self.model = YOLO(str(C.YOLO_WEIGHTS))

        # Resolve which class indices we should treat as "drone".
        names = self.model.names if isinstance(self.model.names, dict) \
                else dict(enumerate(self.model.names))
        self.drone_class_ids = {
            idx for idx, name in names.items()
            if name in C.IMAGE_DRONE_CLASS_NAMES
        }
        if not self.drone_class_ids:
            raise RuntimeError(
                f"None of {C.IMAGE_DRONE_CLASS_NAMES!r} found in YOLO classes {names!r}"
            )
        log.info("drone class ids: %s (from %s)", self.drone_class_ids, names)

    def run(self):
        cap = cv2.VideoCapture(C.CAMERA_INDEX)
        if C.CAMERA_WIDTH:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAMERA_WIDTH)
        if C.CAMERA_HEIGHT:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAMERA_HEIGHT)
        if not cap.isOpened():
            log.error("could not open camera index %s", C.CAMERA_INDEX)
            return

        log.info("camera opened: index=%s %dx%d",
                 C.CAMERA_INDEX,
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        frame_idx = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    log.warning("camera read failed; retrying")
                    self._stop.wait(0.1)
                    continue
                frame_idx += 1
                if frame_idx % C.IMAGE_FRAME_STRIDE != 0:
                    continue

                results = self.model.predict(
                    source=frame,
                    conf=C.IMAGE_THRESHOLD,
                    imgsz=C.YOLO_IMG_SIZE,
                    device=self.device,
                    verbose=False,
                )
                best_drone_conf = 0.0
                for r in results:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    for cls_id, conf in zip(
                        r.boxes.cls.cpu().numpy().astype(int),
                        r.boxes.conf.cpu().numpy().astype(float),
                    ):
                        if cls_id in self.drone_class_ids and conf > best_drone_conf:
                            best_drone_conf = conf

                if best_drone_conf >= C.IMAGE_THRESHOLD:
                    log.info("DRONE detected (conf=%.3f)", best_drone_conf)
                    self.on_detection(best_drone_conf)
                else:
                    log.debug("no drone in frame")

                # Debug stream: encode + push the annotated frame.
                if self.on_frame is not None and results:
                    annotated = results[0].plot()
                    ok, jpeg = cv2.imencode(
                        ".jpg", annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, 75],
                    )
                    if ok:
                        try:
                            self.on_frame(jpeg.tobytes())
                        except Exception:
                            log.debug("on_frame callback raised", exc_info=True)
        except Exception:
            log.exception("image detector crashed")
            raise
        finally:
            cap.release()

    def stop(self):
        self._stop.set()
