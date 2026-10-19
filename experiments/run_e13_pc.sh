#!/usr/bin/env bash
# E13 full run on the home PC: Part B then Part A for k562, norman, rpe1.
# Norman uses e13_power_sign.py (tol 1e-8); K562 and RPE1 use e13_rebuilt_data.py
# (amendment 1, tol 1e-6). Launch:
#   nohup bash experiments/run_e13_pc.sh > results/e13_power_sign/run_all.log 2>&1 &
set -e
cd ~/grd-framework
export GRD_DATA_ROOT=$HOME/grd_data

for ds in k562 norman rpe1; do
  if [ "$ds" = norman ]; then
    script=experiments/e13_power_sign.py
  else
    script=experiments/e13_rebuilt_data.py
  fi
  for part in B A; do
    echo "$(date -u +%FT%TZ) start $ds part $part ($script)"
    .venv/bin/python -u "$script" --dataset "$ds" --part "$part" --workers 24 > "results/e13_power_sign/${ds}_part${part}.log" 2>&1
    echo "$(date -u +%FT%TZ) done $ds part $part"
  done
done
echo "$(date -u +%FT%TZ) ALL DONE"
