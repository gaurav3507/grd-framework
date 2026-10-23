#!/usr/bin/env bash
# E15 full run on the home PC (pre-registration: notes/e15_prereg.md), at most 16 workers.
# Launch: nohup bash experiments/run_e15_pc.sh > results/e15/run_all.log 2>&1 &
# Each step logs to results/e15/logs/<step>.log.
set -e
cd ~/grd-framework
export GRD_DATA_ROOT=$HOME/grd_data
PY=.venv/bin/python
S=experiments/e15_onesided_gate.py
LOG=results/e15/logs
mkdir -p "$LOG"

step() {
  local name=$1; shift
  echo "$(date -u +%FT%TZ) start $name"
  "$@" > "$LOG/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) done $name"
}

step selftest $PY -u $S --selftest
step smoke $PY -u $S --smoke --workers 16
step synthetic $PY -u $S --synthetic --workers 16
for ds in k562 norman rpe1; do
  for s in 0 1 2; do
    step real_${ds}_seed${s} $PY -u $S --real --dataset $ds --seed $s --workers 16
  done
done
for ds in k562 norman rpe1; do
  step planted_${ds} $PY -u $S --planted --dataset $ds --workers 16
done
step aggregate $PY -u $S --aggregate
echo "$(date -u +%FT%TZ) ALL DONE"
