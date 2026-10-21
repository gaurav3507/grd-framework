#!/usr/bin/env bash
# D4 full run on the home PC (pre-registration: notes/d4_prereg.md).
# Launch: nohup bash experiments/run_d4_pc.sh > results/d4_heldout/run_all.log 2>&1 &
# Each step logs to results/d4_heldout/logs/<step>.log.
set -e
cd ~/grd-framework
export GRD_DATA_ROOT=$HOME/grd_data
PY=.venv/bin/python
LOG=results/d4_heldout/logs
mkdir -p "$LOG"

step() {
  local name=$1; shift
  echo "$(date -u +%FT%TZ) start $name"
  "$@" > "$LOG/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) done $name"
}

step common_selftest $PY -u experiments/d4_common.py --selftest
step alignment_selftest $PY -u experiments/d4_alignment.py --selftest
for ds in k562 norman rpe1; do
  for s in 0 1 2; do
    step alignment_${ds}_seed${s} $PY -u experiments/d4_alignment.py --dataset $ds --seed $s
  done
done
step decisions_export $PY -u experiments/d4_decisions_export.py

step ilcs_smoke $PY -u experiments/d4_ilcs_heldout.py --dataset k562 --seed 0 --smoke
for ds in k562 norman rpe1; do
  for s in 0 1 2; do
    step ilcs_${ds}_seed${s} $PY -u experiments/d4_ilcs_heldout.py --dataset $ds --seed $s
  done
done

# Attribution: the 8 (rpe1, norman) x (0, 1, 2, union) jobs concurrently; Norman 8
# threads, RPE1 2 threads (go-ahead 21 Oct: at most 32 threads in total).
# A job whose output already exists is skipped (run_dataset is deterministic; the
# Step 3 timing run of norman seed 0 produced its output).
pids=()
for ds in rpe1 norman; do
  th=$([ "$ds" = norman ] && echo 8 || echo 2)
  for s in 0 1 2 union; do
    tag=$([ "$s" = union ] && echo union || echo seed$s)
    if [ -f results/d4_heldout/attribution/${ds}_${tag}.json ]; then
      echo "$(date -u +%FT%TZ) skip attribution_${ds}_${tag} (output exists)"
      continue
    fi
    echo "$(date -u +%FT%TZ) start attribution_${ds}_${tag}"
    $PY -u experiments/d4_attribution.py --dataset $ds --seed $s --threads $th \
      > "$LOG/attribution_${ds}_${tag}.log" 2>&1 &
    pids+=($!)
  done
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "$(date -u +%FT%TZ) done attribution (rpe1, norman)"
for s in 0 1 2 union; do
  step attribution_k562_$([ "$s" = union ] && echo union || echo seed$s) \
    $PY -u experiments/d4_attribution.py --dataset k562 --seed $s --threads 1
done

step aggregate $PY -u experiments/d4_aggregate.py
echo "$(date -u +%FT%TZ) ALL DONE"
