#!/bin/sh
cd /root/drone-fusion-pi
git pull                          # after you push from the Mac
sudo systemctl stop drone-fusion.service
source venv/bin/activate
python main.py --stream-port 8000