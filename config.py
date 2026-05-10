"""Central configuration for the drone-fusion-pi service.

All timing, thresholds, GPIO pins, and model paths live here so deployment
tweaks happen in one file.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
AUDIO_WEIGHTS = MODELS_DIR / "audio_best.pt"   # CRNN weights from DroneAudioDataset
YOLO_WEIGHTS = MODELS_DIR / "yolo_best.pt"     # YOLOv8 weights from detectfpvdrones

# ---------------------------------------------------------------------------
# GPIO (BCM numbering)
# ---------------------------------------------------------------------------
GPIO_AUDIO = 17    # high when audio model has heard a drone recently
GPIO_IMAGE = 27    # high when image model has seen a drone recently
GPIO_FUSED = 22    # high only when both audio and image are currently high

# Hold each output high for this many seconds after the most recent detection.
# Prevents rapid on/off flicker between consecutive frames / audio windows.
HOLD_SECONDS = 2.0

# ---------------------------------------------------------------------------
# Audio detection
# ---------------------------------------------------------------------------
AUDIO_DEVICE = None              # None = default input device (use `python -m sounddevice` to list)
AUDIO_SAMPLE_RATE = 16_000       # CRNN was trained at 16 kHz
AUDIO_CHANNELS = 1
AUDIO_WINDOW_SECONDS = 1.0       # 1-second clip per inference (matches training)
AUDIO_HOP_SECONDS = 0.5          # run inference twice per second
AUDIO_THRESHOLD = 0.5            # drone-class softmax probability
AUDIO_BLOCK_SECONDS = 0.1        # callback chunk size

# Mel spectrogram parameters (must match dataset.py from DroneAudioDataset)
AUDIO_N_FFT = 512
AUDIO_HOP_LENGTH = 160
AUDIO_N_MELS = 64
AUDIO_F_MIN = 20
AUDIO_F_MAX = AUDIO_SAMPLE_RATE // 2

# ---------------------------------------------------------------------------
# Image detection
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0                 # USB camera (matches your test_camera())
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
IMAGE_FRAME_STRIDE = 3           # run YOLO every Nth frame to keep CPU sane
IMAGE_THRESHOLD = 0.30           # 30% per the spec
IMAGE_DRONE_CLASS_NAMES = ("drone",)   # YOLO model has classes drone/bird/person
YOLO_IMG_SIZE = 416              # smaller = faster on Pi 5; 640 is YOLO default

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
GPIO_UPDATE_INTERVAL = 0.1       # how often the GPIO controller re-evaluates state
LOG_LEVEL = "INFO"
