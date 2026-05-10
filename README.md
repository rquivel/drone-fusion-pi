# drone-fusion-pi

Combines an audio CRNN drone detector and a YOLOv8 image drone detector on a
Raspberry Pi 5 running DietPi. Three GPIO outputs report state:

| Pin (BCM) | High when                                     |
|-----------|-----------------------------------------------|
| 17        | audio model has heard a drone in last 2s      |
| 27        | image model has seen a drone in last 2s       |
| 22        | both of the above are currently high (fusion) |

## Project layout

```
drone-fusion-pi/
├── config.py              # GPIO pins, thresholds, paths, timing
├── main.py                # entry point; starts threads, drives GPIO
├── audio_model.py         # CRNN architecture (mirror of upstream model)
├── audio_detector.py      # mic -> ring buffer -> CRNN -> callback
├── image_detector.py      # camera -> YOLOv8 -> callback
├── gpio_controller.py     # debounced fusion + hold-time logic
├── models/                # drop trained weights here (not committed)
│   ├── audio_best.pt
│   └── yolo_best.pt
├── scripts/
│   └── sanity_check.py    # mic / camera / GPIO / weights smoke test
├── systemd/
│   └── drone-fusion.service
├── requirements.txt
└── README.md
```

## One-time setup on the Pi (DietPi)

```bash
# system packages
sudo apt update
sudo apt install -y python3-venv python3-pip \
    portaudio19-dev libsndfile1 libgl1 libglib2.0-0 \
    python3-lgpio
# (libgl1/libglib2.0-0 are needed by opencv-python-headless / ultralytics deps)

# clone the project
sudo mkdir -p /var/www && sudo chown "$USER" /var/www
cd /var/www
git clone <your-remote-url> drone-fusion-pi
cd drone-fusion-pi

# python env
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

## Copy in the trained weights

If the three projects sit side by side under `/var/www/` (the default layout),
just run the deploy script — it copies the latest weights from the sibling
folders into `models/`, atomically and skipping unchanged files:

```bash
python scripts/update_weights.py            # both
python scripts/update_weights.py --only audio
python scripts/update_weights.py --force    # overwrite even if hash matches
python scripts/update_weights.py --dry-run  # show plan, don't write
```

It compares SHA-256 hashes so re-running it is a no-op when nothing changed.
After retraining either upstream model, run it again to refresh.

If the projects don't sit side by side (e.g. you're pulling weights from a
different machine), pass paths explicitly:

```bash
python scripts/update_weights.py \
    --audio-src /elsewhere/best.pt \
    --yolo-src  /elsewhere/best.pt
```

Or, if you'd rather copy from a dev machine straight onto the Pi:

```bash
scp /var/www/DroneAudioDataset/checkpoints/best.pt \
    pi@<host>:/var/www/drone-fusion-pi/models/audio_best.pt
scp /var/www/detectfpvdrones/runs/detect/train_v3/weights/best.pt \
    pi@<host>:/var/www/drone-fusion-pi/models/yolo_best.pt
```

## Sanity check before running the full service

```bash
source .venv/bin/activate
python scripts/sanity_check.py
```

Expected output covers all four sections (model weights, audio, camera,
GPIO). Any FAIL line is something to fix before continuing.

## Run it

Foreground (good for first run / watching logs):

```bash
source .venv/bin/activate
python main.py
```

Useful flags while debugging:

| Flag           | What it does                                   |
|----------------|------------------------------------------------|
| `--mock-gpio`  | Skip real GPIO, just log transitions           |
| `--no-image`   | Audio detector only (e.g. while testing mic)   |
| `--no-audio`   | Image detector only (e.g. while testing camera)|
| `--device cpu` | Force CPU inference (default; Pi 5 has no CUDA)|

## Run as a service (auto-start on boot)

```bash
sudo cp systemd/drone-fusion.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now drone-fusion.service

# tail logs
journalctl -u drone-fusion.service -f
```

The unit file assumes `User=dietpi` and `WorkingDirectory=/var/www/drone-fusion-pi`.
Edit those if your DietPi user is different.

## Tuning

Most tweaks live in `config.py`:

- `AUDIO_THRESHOLD` (default 0.5) and `IMAGE_THRESHOLD` (0.30) — raise for fewer
  false alarms, lower for higher recall.
- `HOLD_SECONDS` (2.0) — how long a pin stays high after the last detection.
  Lower for snappier release, higher to ride out brief gaps.
- `IMAGE_FRAME_STRIDE` (3) — run YOLO every Nth frame. Bigger numbers = lower
  CPU use, slower reaction.
- `YOLO_IMG_SIZE` (416) — smaller is faster on Pi 5 at the cost of small-object
  recall. 320, 416, 640 are common.
- `AUDIO_DEVICE` (`None` = system default) — set to a device index from
  `python -m sounddevice` if the wrong mic is being used.

## Wiring (BCM)

| Function     | BCM | Header pin |
|--------------|-----|------------|
| audio out    | 17  | 11         |
| image out    | 27  | 13         |
| fused out    | 22  | 15         |
| GND          | -   | 9          |

All outputs are active-high. Drive a logic-level transistor or opto-isolator
if you're switching anything bigger than an LED + resistor.

## Pi 5 GPIO note

This project uses `gpiozero` with the `lgpio` backend. Older `RPi.GPIO`
is **not** supported on the Pi 5 (no `/dev/gpiomem`). The included
`requirements.txt` and the apt step above install everything you need.
