#!/usr/bin/env bash
python3 -c "
import csv, sys
with open('predictions.csv') as f:
    rows = list(csv.DictReader(f))
for r in rows:
    t = float(r['t_start_s']); p = float(r['drone_prob'])
    bar = '#' * int(p * 50)
    print(f'{t:6.2f}s |{bar:<50}| {p:.4f}')
"