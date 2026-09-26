# E4: second backbone (Tier 2, Experiment A)

Backbone B replaces the precision-difference statistic with a covariance-difference
one everywhere it is used: the gate reads `T_e = lambda_max(Cov(Y_0) - Cov(Y_e))` and
the recovery rule takes `w_i = v_max(Cov(Y_0) - Cov(Y_e(i)))`, `Z_i = Y_0 w_i`. The
size-matched null, BH, `d_rec`, the verdict rule and every experiment design are
unchanged and are reached through the same code with `readout="covariance"`
(`src/gate/precision_readout.py`, `src/recover/backbone.py`). It is run on the four E2
arms and the E2c random-subset starvation series (10 seeds, B=500), and as a gate
only on the K562 / RPE1 / Norman panel (corrected disjoint null, BH within family,
NMIN=200, d_proj=10, five gate seeds).

Produced by `experiments/e4_second_backbone.py` (launcher:
`experiments/run_tier2_a100.sh`).

## JSON keys

`e4_calibration_report.json` (same schema as `results/e2/e2_calibration_report.json`)
- `arms.<arm>.levels[]`: per-severity gate counts, verdicts, naive MCC, certified recovery, `per_seed`.
- `arms.<arm>.gate_behavior`, `crossover_*`, `silent_failure`: the E2 classification, unchanged.
- `status`, `central_claim`: E2 meaning; a FAIL is a finding about Backbone B, not a script error.
- `precision_same_code`, `comparison`: the precision readout on the same code in the same job.
  Compare against these, not `results/e2`, which predates the size-matched null fix.
- `population_decomposition`: MCC of each row rule with exact population covariances versus
  sample covariances (separates a property of the rule from an estimation problem).
- `preregistered_expectation`: the three Tier 2 expectations and whether each held.

`e4_starvation_report.json` (same schema as `results/e2c/starvation_report.json`)
- `aggregate.levels[].random_subset_control`: MCC mean and seed SD, gate cap, verdict counts.
- `aggregate.crossover_gate_first_restricts`, `crossover_random_subset_mcc_below_0p90`,
  `gate_restricts_at_or_before_mcc_crossover`.
- `precision_reference`: crossovers of the committed precision run for comparison.

`e4_real_panel.json`
- `datasets.<K562|RPE1|Norman>.n_environments`, `raw_count_per_seed`, `bh_count_per_seed`.
- `datasets.<name>.bh_selected_ids_per_seed`: BH-selected environment ids for Jaccard.
- `datasets.<name>.per_seed[].jaccard_bh_vs_precision`: against `results/e3_decisions` (seed 0)
  and `results/e3_stability` (same seed).
- `datasets.<name>.negative_control_*_count_per_seed`: the 50 random pure-control environments.
- `datasets.<name>.per_seed[].per_perturbation`: per-environment signal, threshold, p-value, decisions.
