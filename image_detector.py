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

        # Track-confirmation state. Maps each YOLO track id (int) to the
        # number of consecutive analyzed frames it has been seen on. A track
        # has to clear IMAGE_MIN_TRACK_FRAMES before it's reported.
        self._track_seen: dict[int, int] = {}
        self._track_missing: dict[int, int] = {}

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

                if C.IMAGE_USE_TRACKING:
                    results = self.model.track(
                        source=frame,
                        conf=C.IMAGE_THRESHOLD,
                        imgsz=C.YOLO_IMG_SIZE,
                        device=self.device,
                        persist=True,
                        verbose=False,
                    )
                else:
                    results = self.model.predict(
                        source=frame,
                        conf=C.IMAGE_THRESHOLD,
                        imgsz=C.YOLO_IMG_SIZE,
                        device=self.device,
                        verbose=False,
                    )

                # Collect (track_id_or_None, conf) for every drone detection
                # in this frame so we can both report best-confidence and
                # update track-confirmation state.
                frame_drone_dets: list[tuple[int | None, float]] = []
                for r in results:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    cls_arr = r.boxes.cls.cpu().numpy().astype(int)
                    conf_arr = r.boxes.conf.cpu().numpy().astype(float)
                    if r.boxes.id is not None:
                        id_arr = r.boxes.id.cpu().numpy().astype(int)
                    else:
                        id_arr = [None] * len(cls_arr)
                    for cls_id, conf, tid in zip(cls_arr, conf_arr, id_arr):
                        if cls_id in self.drone_class_ids:
                            frame_drone_dets.append(
                                (None if tid is None else int(tid), float(conf))
                            )

                # Track confirmation: only fire on_detection when a track
                # has persisted for IMAGE_MIN_TRACK_FRAMES analyzed frames.
                if C.IMAGE_USE_TRACKING:
                    seen_ids_this_frame = {tid for tid, _ in frame_drone_dets
                                           if tid is not None}

                    # Increment seen counter for tracks present this frame.
                    for tid in seen_ids_this_frame:
                        self._track_seen[tid] = self._track_seen.get(tid, 0) + 1
                        self._track_missing.pop(tid, None)

                    # Increment missing counter for tracks that didn't show
                    # this frame, and forget tracks missing for too long.
                    for tid in list(self._track_seen):
                        if tid in seen_ids_this_frame:
                            continue
                        self._track_missing[tid] = self._track_missing.get(tid, 0) + 1
                        if self._track_missing[tid] >= C.IMAGE_TRACK_FORGET_FRAMES:
                            self._track_seen.pop(tid, None)
                            self._track_missing.pop(tid, None)

                    # Confirmed = at least MIN frames seen. Pick best confidence
                    # among confirmed tracks present this frame.
                    confirmed_conf = 0.0
                    confirmed_count = 0
                    for tid, conf in frame_drone_dets:
                        if tid is None:
                            continue
                        if self._track_seen.get(tid, 0) >= C.IMAGE_MIN_TRACK_FRAMES:
                            confirmed_count += 1
                            if conf > confirmed_conf:
                                confirmed_conf = conf

                    if confirmed_count > 0:
                        log.info("DRONE confirmed (conf=%.3f, tracks=%d)",
                                 confirmed_conf, confirmed_count)
                        self.on_detection(confirmed_conf)
                    else:
                        pending = sum(
                            1 for tid, _ in frame_drone_dets
                            if tid is not None
                            and 0 < self._track_seen.get(tid, 0) < C.IMAGE_MIN_TRACK_FRAMES
                        )
                        if pending:
                            log.debug("drone pending confirmation (%d track(s))", pending)
                        else:
                            log.debug("no drone in frame")
                else:
                    # No tracking: legacy behavior — fire on any single-frame hit.
                    best_drone_conf = max((c for _, c in frame_drone_dets), default=0.0)
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
