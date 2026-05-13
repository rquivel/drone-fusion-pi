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

Tested on a Raspberry Pi 5 running DietPi (Debian 13 / Python 3.13). Steps
look long because DietPi ships minimal — once these are done, redeploys
are just `git pull` + `pip install -r requirements.txt`.

### 1. System packages

```bash
sudo apt update
sudo apt install -y \
    python3-venv python3-pip python3-dev \
    build-essential swig liblgpio-dev \
    portaudio19-dev libsndfile1 \
    libgl1 libglib2.0-0
```

What each one is for:

- `python3-venv` / `python3-dev`: virtualenv tooling + C headers for native extensions
- `build-essential`: gcc, make, libc-dev — needed because `lgpio` has no
  prebuilt wheel for Python 3.13 / aarch64 and pip compiles it from source
- `swig`: `lgpio`'s build step generates its Python bindings with SWIG
- `liblgpio-dev`: the underlying C library that the Python `lgpio` package
  wraps — without it the linker fails with `cannot find -llgpio`
- `portaudio19-dev` + `libsndfile1`: required by `sounddevice` / `soundfile`
- `libgl1` + `libglib2.0-0`: shared libraries pulled in by `opencv-python`

### 2. Clone the project

```bash
sudo mkdir -p /var/www && sudo chown "$USER" /var/www
cd /var/www
git clone https://github.com/<user>/drone-fusion-pi.git
cd drone-fusion-pi
```

Use the **HTTPS** URL (not `git@github.com:...`), unless you've already
added an SSH key to your GitHub account. SSH requires authentication even
for public repos; HTTPS doesn't for read-only clones.

### 3. Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel
```

If your shell prompt ever loses the `(venv)` prefix (new terminal, etc.),
just call the binaries directly — `./venv/bin/python` and `./venv/bin/pip`
work the same whether or not `activate` has been sourced. This avoids the
`error: externally-managed-environment` confusion if you forget to activate.

### 4. Install PyTorch (CPU-only build)

**Important:** PyTorch's `torch>=2.7` aarch64 wheels are Jetson-targeted
and pull ~2.5 GB of NVIDIA CUDA libraries that won't run on a Pi 5. We need
the older CPU-only build. The last release that supports Python 3.13 and
has CPU-only aarch64 wheels is 2.6.x.

```bash
pip install "torch==2.6.*" "torchaudio==2.6.*" "torchvision==0.21.*"
```

Verify it actually imports without trying to load CUDA libraries:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
# Expected: torch 2.6.0 cuda: False
```

If you get `OSError: libcudart.so.13: cannot open shared object file`, you
ended up with a `>=2.7` build by mistake — `pip uninstall -y torch torchaudio
torchvision` and rerun the pinned install above.

### 5. Install the rest

```bash
pip install -r requirements.txt
```

Most of this comes from prebuilt aarch64 wheels and is fast. `lgpio` is
compiled from source (~30 seconds). If you see
`error: command 'aarch64-linux-gnu-gcc' failed: No such file or directory`,
step 1's `build-essential` wasn't installed.

### 6. Sanity check before continuing

```bash
python scripts/sanity_check.py
python scripts/flash_leds.py --cycles 2
```

`sanity_check.py` covers four sections (model weights, audio, camera, GPIO).
`flash_leds.py` cycles all three outputs so you can visually confirm each
LED + resistor is wired to the right pin. Watch the startup log lines —
you want to see `opened hardware LED on GPIO17 (audio)`, not `[mock]`.

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

## Run it

Foreground (good for first run / watching logs):

```bash
source venv/bin/activate
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

### Optional: LED self-test on every boot

`systemd/drone-fusion-selftest.service` is a one-shot unit that runs
`scripts/flash_leds.py --cycles 1` during boot — about 5 seconds of LED
cycling so you can visually confirm the hardware is alive before the
main detector takes over. The main service has `After=drone-fusion-selftest.service`
so it waits for the test to finish before claiming the GPIO pins.

```bash
sudo cp systemd/drone-fusion-selftest.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable drone-fusion-selftest.service
```

To disable later: `sudo systemctl disable drone-fusion-selftest.service`.
The main service still works whether or not the self-test unit is installed.

### Manually checking the LEDs after the service is running

GPIO pins are exclusive — while `drone-fusion.service` is up it owns
GPIO 17/27/22 and a manual `flash_leds.py` will fall back to mock mode.
To run the LED test by hand:

```bash
sudo systemctl stop drone-fusion.service
source venv/bin/activate
python scripts/flash_leds.py --cycles 2
sudo systemctl start drone-fusion.service
```

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

## Troubleshooting

Things you may hit on a fresh DietPi install, with the fix that worked:

**`ModuleNotFoundError: No module named 'cv2'` after `source venv/bin/activate`.**
A common cause is a venv that was created in a different folder and then
moved — `activate` hard-codes the absolute path. Recreate the venv from
scratch (`rm -rf venv && python3 -m venv venv && source venv/bin/activate
&& pip install -r requirements.txt`), or invoke `./venv/bin/python` directly.

**`error: externally-managed-environment` from pip.**
You're not inside the venv. Either run `source venv/bin/activate` first or
call `./venv/bin/pip ...` directly.

**Pip downloads ~2.5 GB of `nvidia-*` / `cuda-toolkit` packages.**
You ended up on a torch >=2.7 aarch64 wheel, which is a Jetson build with
mandatory CUDA dependencies. Uninstall and pin to 2.6:
```bash
pip uninstall -y torch torchaudio torchvision
pip install "torch==2.6.*" "torchaudio==2.6.*" "torchvision==0.21.*"
```

**`error: command 'swig' failed: No such file or directory`** while building
`lgpio`. Run `sudo apt install -y swig python3-dev`.

**`error: command 'aarch64-linux-gnu-gcc' failed: No such file or directory`**
while building `lgpio`. Run `sudo apt install -y build-essential`.

**`/usr/bin/ld: cannot find -llgpio`** while building `lgpio`. The Python
package is just a SWIG wrapper; the C library needs to be installed
separately. Run `sudo apt install -y liblgpio-dev`.

**`flash_leds.py` runs but no LEDs light up.**
Look at the startup log. If you see `falling back to mock GPIO for audio
(reason: ...)`, the reason tells you why `gpiozero` couldn't open the
hardware. Common causes: `lgpio` not installed, not running as root (or
not in the `gpio` group), or another process is holding `/dev/gpiochip4`.
