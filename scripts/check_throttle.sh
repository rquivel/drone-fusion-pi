#!/usr/bin/env bash
sudo vcgencmd get_throttled
journalctl -k --since "5 minutes ago" | grep -i voltage | wc -l