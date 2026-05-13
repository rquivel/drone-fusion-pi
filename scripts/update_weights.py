"""Refresh model weights from the sibling project folders.

By default this script reads from:
  - ../DroneAudioDataset/checkpoints/best.pt        -> models/audio_best.pt
  - ../detectfpvdrones/runs/detect/train_v3/weights/best.pt -> models/yolo_best.pt

That layout matches the dev machine where all three projects live in /var/www.

It reports each source/destination's size, modification time, and SHA-256;
skips copies when the source matches the destination already; and writes
atomically via a temporary file + rename so a crash mid-copy can never
leave a half-written weight file in models/.

Usage:
    python scripts/update_weights.py                       # both
    python scripts/update_weights.py --only audio
    python scripts/update_weights.py --force               # copy even if unchanged
    python scripts/update_weights.py --dry-run
    python scripts/update_weights.py \\
        --audio-src /path/to/best.pt \\
        --yolo-src  /path/to/best.pt
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import shutil
import sys
from pathlib import Path

# Make the project's ``config`` importable when running from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config as C   # noqa: E402


# -- Default source paths (sibling folders to this project) ----------------
DEFAULT_AUDIO_SRC = PROJECT_ROOT.parent / "DroneAudioDataset" / "checkpoints" / "best.pt"

_YOLO_RUNS_DIR = PROJECT_ROOT.parent / "detectfpvdrones" / "runs" / "detect"


def _latest_yolo_best() -> Path:
    """Return the most recently modified best.pt across runs/detect/*/weights/.

    Falls back to the train_v3 path if no runs are found yet, so first-time
    setups still have a reasonable default.
    """
    fallback = _YOLO_RUNS_DIR / "train_v3" / "weights" / "best.pt"
    if not _YOLO_RUNS_DIR.is_dir():
        return fallback
    candidates = [
        p for p in _YOLO_RUNS_DIR.glob("*/weights/best.pt")
        if p.is_file()
    ]
    if not candidates:
        return fallback
    return max(candidates, key=lambda p: p.stat().st_mtime)


DEFAULT_YOLO_SRC = _latest_yolo_best()


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def fmt_mtime(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def update_one(name: str, src: Path, dst: Path,
               force: bool, dry_run: bool) -> bool:
    print(f"\n== {name} ==")
    if not src.exists():
        print(f"  source missing: {src}")
        return False

    src_size = src.stat().st_size
    src_hash = sha256(src)
    print(f"  source: {src}")
    print(f"          {fmt_size(src_size)}   "
          f"mtime={fmt_mtime(src)}   "
          f"sha256={src_hash[:12]}…")

    if dst.exists():
        dst_size = dst.stat().st_size
        dst_hash = sha256(dst)
        print(f"  dest  : {dst}")
        print(f"          {fmt_size(dst_size)}   "
              f"mtime={fmt_mtime(dst)}   "
              f"sha256={dst_hash[:12]}…")
        if dst_hash == src_hash and not force:
            print("  -> up to date, skipping (pass --force to overwrite anyway)")
            return False
    else:
        print(f"  dest  : {dst}  (does not exist yet)")

    if dry_run:
        print("  -> dry-run; would copy")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.copy2(src, tmp)            # preserves mtime
    os.replace(tmp, dst)              # atomic on POSIX
    new_hash = sha256(dst)
    if new_hash != src_hash:
        print(f"  !! hash mismatch after copy: {new_hash[:12]}… vs {src_hash[:12]}…")
        return False
    print(f"  -> copied OK ({fmt_size(src_size)})")
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--audio-src", type=Path, default=DEFAULT_AUDIO_SRC)
    ap.add_argument("--yolo-src", type=Path, default=DEFAULT_YOLO_SRC)
    ap.add_argument("--audio-dst", type=Path, default=Path(C.AUDIO_WEIGHTS))
    ap.add_argument("--yolo-dst", type=Path, default=Path(C.YOLO_WEIGHTS))
    ap.add_argument("--only", choices=("audio", "yolo"),
                    help="Update just one of the two")
    ap.add_argument("--force", action="store_true",
                    help="Copy even if source and dest hashes match")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would happen without writing")
    args = ap.parse_args()

    print(f"Project root: {PROJECT_ROOT}")
    targets = []
    if args.only in (None, "audio"):
        targets.append(("audio CRNN", args.audio_src, args.audio_dst))
    if args.only in (None, "yolo"):
        targets.append(("YOLOv8", args.yolo_src, args.yolo_dst))

    n_copied = 0
    for name, src, dst in targets:
        if update_one(name, src, dst, force=args.force, dry_run=args.dry_run):
            n_copied += 1

    print(f"\n{n_copied} file(s) updated"
          + (" (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
