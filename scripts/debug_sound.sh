#!/usr/bin/env bash
sudo systemctl stop drone-fusion.service     # mic is held by the service
cd /root/drone-fusion-pi
source venv/bin/activate
python3 scripts/record_audio_sample.py        # 60s by default
# defaults to recordings/<timestamp>/
sudo systemctl start drone-fusion.service