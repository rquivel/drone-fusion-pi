"""CRNN architecture — copy of /var/www/DroneAudioDataset/src/model.py so the
Pi project is self-contained and can be deployed without cloning the audio
repo. If you retrain the upstream model, copy the new weights to
models/audio_best.pt; the architecture rarely changes.
"""
import torch
import torch.nn as nn

import config as C


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool=(2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.block(x)


class CRNN(nn.Module):
    def __init__(self, num_classes: int = 2, gru_hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBlock(1, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 128),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.gru = nn.GRU(
            input_size=128, hidden_size=gru_hidden,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_hidden * 2, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.freq_pool(x).squeeze(2).transpose(1, 2)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


def load_audio_model(weights_path=None, device="cpu"):
    weights_path = weights_path or C.AUDIO_WEIGHTS
    model = CRNN().to(device)
    ckpt = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model
