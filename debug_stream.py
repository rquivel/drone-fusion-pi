"""Tiny MJPEG-over-HTTP debug stream.

Off by default. Enable from main.py with --stream-port. When running, the
image_detector pushes each annotated frame in here; HTTP clients see a
live multipart/x-mixed-replace stream from /stream.mjpg, or a single
snapshot from /snapshot.jpg.

This is meant for ad-hoc testing on a local network from a laptop
browser. There's no auth — don't expose it to the open internet.
"""
from __future__ import annotations

import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

log = logging.getLogger("stream")

BOUNDARY = "frame"


class _Handler(BaseHTTPRequestHandler):
    """Per-request handler. Resolves the stream instance via server.stream_ref."""

    def log_message(self, fmt, *args):  # silence the default access log
        pass

    def _send_jpeg(self, jpeg: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(jpeg)

    def do_GET(self):
        stream: DebugStream = self.server.stream_ref  # type: ignore[attr-defined]

        if self.path in ("/", "/index.html"):
            body = (f"<html><body style='background:#000;color:#fff;font:14px sans-serif'>"
                    f"<h2>drone-fusion-pi debug stream</h2>"
                    f"<p><a style='color:#0af' href='/stream.mjpg'>live stream</a> "
                    f"&middot; <a style='color:#0af' href='/snapshot.jpg'>snapshot</a></p>"
                    f"<img src='/stream.mjpg' style='max-width:100%'></body></html>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/snapshot.jpg":
            jpeg = stream.snapshot()
            if jpeg is None:
                self.send_error(503, "no frame yet")
                return
            self._send_jpeg(jpeg)
            return

        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.send_header("Content-Type",
                             f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.end_headers()
            try:
                last_id = -1
                while not stream.is_closed():
                    jpeg, fid = stream.wait_for_new_frame(last_id, timeout=2.0)
                    if jpeg is None:
                        continue
                    last_id = fid
                    self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
            return

        self.send_error(404)


class DebugStream:
    """A tiny multipart-JPEG server. Single latest-frame slot, broadcast to
    however many HTTP clients happen to be connected."""

    def __init__(self, port: int = 8000):
        self.port = port
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest: Optional[bytes] = None
        self._frame_id = 0
        self._closed = False
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ---- producer side ------------------------------------------------
    def push_jpeg(self, jpeg: bytes):
        with self._cond:
            self._latest = jpeg
            self._frame_id += 1
            self._cond.notify_all()

    # ---- consumer side (used by HTTP handler) -------------------------
    def snapshot(self) -> Optional[bytes]:
        with self._lock:
            return self._latest

    def wait_for_new_frame(self, last_id: int, timeout: float = 2.0):
        """Block until a frame newer than last_id arrives (or timeout)."""
        with self._cond:
            if self._frame_id <= last_id:
                self._cond.wait(timeout=timeout)
            if self._frame_id <= last_id:
                return None, self._frame_id
            return self._latest, self._frame_id

    def is_closed(self) -> bool:
        return self._closed

    # ---- lifecycle ----------------------------------------------------
    def start(self):
        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), _Handler)
        self._server.stream_ref = self          # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="debug-stream",
            daemon=True,
        )
        self._thread.start()
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            local_ip = "<pi-ip>"
        log.info("debug stream listening on http://%s:%d/  "
                 "(stream.mjpg / snapshot.jpg)", local_ip, self.port)

    def stop(self):
        self._closed = True
        with self._cond:
            self._cond.notify_all()
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
