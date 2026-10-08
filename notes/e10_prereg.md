# E10 pre-registration (8 Oct 2026)

Unit: one (arm, level, seed, intervened direction) of the four E2 arms, precision backbone.
Ground truth: a unit is GOOD if the recovered column's best-match |corr| with a true latent > 0.90.
Coverage = certified units / all units. Risk = certified BAD units / certified units.
Gates: G0 none; G1 raw per-environment alpha; G2 GRD (BH at level q over the supplied
environments of that arm-level-seed); G3 sample size n_e >= n*; G4 uncalibrated statistic
T_e > c; G5 bootstrap direction stability; G7 oracle. Knob grids are fixed in the E10 script.
Primary metric: risk at matched coverage (each gate's risk at the coverage GRD reaches with
q = 0.05), pooled over arms A, B, D; secondary: area under the risk-coverage curve (AURC) over each
gate's covered range. Arm C (measurement contamination) is reported separately.

Predictions
P1. At q = 0.05, GRD's risk on arms A, B, D pooled is <= 0.10.
P2. On A, B, D pooled, GRD's risk at its q = 0.05 coverage is lower than G0's risk and lower
    than G3's risk at matched coverage (G3 only varies on arm B).
P3. G4 uses the same statistic as GRD without the size-matched null. On the arms with one sample
    size (A and D, n_e = 2000) its AURC is within 0.02 of GRD's. On arm B, where n_e varies from
    20 to 4000, its AURC is higher than GRD's by more than 0.02, because a single cutoff cannot
    adapt to the null's dependence on n_e.
P4. G5 (bootstrap stability) is competitive with GRD on A, B, D: its risk at GRD's q = 0.05
    coverage is within 0.02 of GRD's, or lower.
P5. On arm C every non-oracle gate has risk > 0.30 at every operating point with coverage >= 0.5
    (the declared failure class); G5 does not escape it, because contaminated directions are
    stable but wrong.
P6. Transfer: the G4 threshold c that gives GRD's q = 0.05 coverage on arm B, applied unchanged
    to arm D, gives a risk at least 0.05 further from GRD's arm-D risk at q = 0.05 than GRD's own
    arm-B-to-arm-D change.

Decision rule written in advance: if G5's risk at GRD's q = 0.05 coverage on A, B, D is lower
than GRD's by more than 0.02,
the manuscript reports that and states GRD's advantage as calibration, estimator coupling and
theory, not as a better risk-coverage trade-off.
