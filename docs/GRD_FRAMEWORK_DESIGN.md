# Gate-Recover-Discover (GRD): A Precondition-Gated Framework for Causal Representation Learning and Causal Discovery

Design document v0.1, 1 Sep 2026.
Author: Gaurav Goyal (TIET, reg. 902503013). Status: DESIGN ONLY, no code exists yet.
Repo (to be created): github.com/gaurav3507/grd-framework. Do NOT reuse ccrl-newidea2.

Style rule: no em dashes anywhere in this project (papers, code, comments, docs).

---

## 1. One-line claim

GRD is a CRL+CD pipeline that first tests whether the data satisfies the identifiability
preconditions of causal representation learning, recovers latent causal variables only from
environments the test certifies, and reports the latent causal graph only at the resolution
that the recovery ambiguity class permits, abstaining on everything else.

The integrating claim: a pipeline that knows and states what it can and cannot identify,
instead of returning a confident graph regardless of data adequacy.

## 2. Why this exists (motivation, in the paper's voice)

Current CRL+CD methods always return latents and a graph, even when the data could never
support them. On a physical benchmark with known ground truth, a leading method returned a
different wrong graph on every run and never signalled failure (Gamella, Bing, Runge, ICML
2025). Separately, our own audits show that widely used multi-environment benchmarks fail
the identifiability preconditions the methods assume (Paper 2, JBHI submission). GRD turns
that diagnostic knowledge into a method: the audit becomes the front end of a recovery
pipeline, and honesty about resolution becomes the output format.

Positioning sentence for page 1 (mandatory, see Section 10): precondition gating and
admissibility-first abstention have recently proven valuable in observed-variable causal
discovery (Uehara 2026, arXiv 2605.27477; Dang et al. 2026, arXiv 2602.08340). Both assume
the causal variables are observed. GRD brings this discipline to the latent setting, where
the preconditions concern identifiability of a latent SCM under an unknown mixing, and the
abstention is over the resolution of the latent graph.

## 3. Scope and non-goals

In scope, v1:
- Linear-mixing regime end to end (backbone and gate share one exact theory).
- Multi-environment i.i.d.-per-environment data. No time series in v1.
- Known-ish latent dimension handled as a gate output (estimated bound), not assumed.
- Validation on synthetic, semi-synthetic, and the real datasets already in hand
  (Perturb-seq: Norman, Replogle K562/RPE1, Frangieh arms; fMRI: HCP, ABIDE-derived).

Non-goals, v1 (explicitly deferred, each is a later chapter, not scope creep):
- Nonlinear mixing backbone (discrepancy-VAE) and the curvature-robust gate (Path B).
- New identifiability theory. We cite Bing CLeaR 2024, Varici NeurIPS 2024, Jin and
  Syrgkanis for every guarantee. Zero new theorems are claimed in v1.
- Temporal CRL, cyclic latent graphs, latent confounding among latents.
- Beating any method on recovery accuracy. The claim is calibrated honesty, not SOTA MCC.

## 4. Architecture

Three modules, strict one-way dataflow, each module a standalone script with a JSON
contract so any module can be run and tested alone.

    raw multi-env data X_e (e = 1..m)
        |
        v
    [ MODULE 1: GATE ]  ->  gate_certificate.json
        |
        v  (only certified environments + gate metadata)
    [ MODULE 2: RECOVER ]  ->  latents Z_hat, mixing B_hat, targets I_hat, recover_report.json
        |
        v
    [ MODULE 3: DISCOVER ]  ->  annotated graph G_hat with per-edge status, discover_report.json

### 4.1 Module 1: GATE

Input: per-environment data matrices X_e in R^(n_e x D), environment labels, declared
intervention type if known (hard/soft/unknown).

Tests, in order. Each returns PASS / FAIL / CAPPED plus numbers.

P1. Environment count. Compare m against the requirement of the chosen backbone and the
    general bound (Rajendran et al. NeurIPS 2024, Remark 2: m >= n+1 environments for n
    latents; iVAE-style needs 2n+1). Output: max latent dimension n_max_env supportable by
    the environment count alone.

P2. Per-environment power. Compare each n_e against the measured detection floors from
    Paper 2: hard interventions need n_e above the 125-cell floor, soft above the 8000
    crossing (both are bounds, not point values; keep that wording). Output: the subset of
    environments with adequate power, and the count surviving.

P3. Mechanism vs measurement. For data with a nuisance grouping (site, batch, session),
    run the Paper 1 VAR diagnostic: fit per-level VAR, test coefficient stability, and
    partition instability into diagonal (autocorrelation, measurement-type) vs off-diagonal
    (cross-variable coupling, mechanism-type). Environments whose distinguishing shift is
    diagonal-dominated are flagged MEASUREMENT and excluded or down-weighted. For
    Perturb-seq (no time axis) P3 falls back to the randomised-vs-confounded nuisance check
    (batch randomisation fraction), and this fallback is stated openly.
    NOTE: P3's VAR form applies to time-series-like data (fMRI). Its exact form per data
    type is a design decision recorded in Section 8 risks.

P4. Recoverable rank. Chen-Fang LFC bootstrap rank test on covariance differences between
    certified environments (H0: rank(Delta) <= r, alpha 0.05, B = 500, statistic = sum of
    trailing squared singular values, zero tuning constants). Known structural fact from the
    ranktest lane: a k=1 hard intervention yields rank <= 2 change and sits inside the
    rank-2 null by construction; the test's power regime is k >= 3, and power decays with
    dimension (k-sweep, commit 4e36230). The gate therefore reports the number of latent
    directions with detectable rank signal, k_hat, with the caveat wording fixed in Paper 2:
    "exceeds the envelope in which THIS TEST is exact", never "outside the regime where CRL
    is identifiable".

Output: gate_certificate.json with schema (draft):

    {
      "m_total": int, "m_certified": int, "environments_certified": [ids],
      "environments_excluded": [{"id":, "reason": "power|measurement"}],
      "n_max_env": int,            // from P1
      "k_hat_rank": int,           // from P4
      "n_recoverable": int,        // min(n_max_env, k_hat_rank)
      "intervention_type_assumed": "hard|soft|unknown",
      "verdict": "PROCEED|PROCEED_CAPPED|ABSTAIN",
      "provenance": {"input_paths": [], "sha256": [], "code_commit": ""}
    }

Gate verdict rules:
- ABSTAIN if m_certified < backbone minimum or n_recoverable == 0.
- PROCEED_CAPPED if recovery is possible but only for n_recoverable < intended latent dim.
- PROCEED otherwise.

The gate never silently drops data: every exclusion carries a machine-readable reason.

### 4.2 Module 2: RECOVER

Backbone, v1: linear-mixing multi-node interventional CRL, the Bing et al. CLeaR 2024 and
Varici et al. NeurIPS 2024 lineage (unknown multi-node targets; hard gives perfect
identifiability, soft gives ancestral). Chosen because the Chen-Fang rank test in P4 is
exact in exactly this regime, so the gate's promise and the estimator's guarantee are in
the same mathematical language and the headline claim (gate predicts recovery) is
verifiable rather than approximate.

Implementation plan: implement the published estimator(s), or adapt released code if the
licence permits, inside our repo with our oracle harness around it. We claim zero method
novelty in this module and say so; novelty lives in Modules 1 and 3 and the integration.

Input: certified environments only, plus n_recoverable from the certificate.
Output: Z_hat (n_recoverable latents), B_hat (mixing), I_hat (estimated targets),
ambiguity_class in {"perfect", "ancestral", "SNA"} recorded explicitly, recover_report.json
with the same provenance block.

Real-data note: raw pixels and raw counts are not linear in the latents. For v1 real-data
runs, the framework operates on a linearised representation (PCA projection fitted on
control/reference data, following the Paper 2 pipeline), and the paper states plainly that
linear-exactness claims apply to the projected representation. The nonlinear chapter (v2)
removes this caveat.

### 4.3 Module 3: DISCOVER

Input: Z_hat, ambiguity_class, I_hat.
Step 1: run CD on Z_hat. v1 default: the graph estimate that the backbone itself produces
(these estimators recover the graph jointly); a secondary check with a standard CD
algorithm (PC or GES) on Z_hat for agreement.
Step 2: abstention filter. Map ambiguity_class to per-edge decidability:
- perfect (hard multi-node, conditions met): all edges reportable.
- ancestral (soft): only ancestor relations are identified; edges within the same
  transitive-closure equivalence are marked UNDECIDED. Report the transitive closure, not
  an arbitrary DAG member.
- SNA (general environments, Jin and Syrgkanis): edges involving surrounded nodes are
  marked UNDECIDED with the SNA code.
Step 3: consistency check against the gate. If the recovered rank or target count
contradicts the certificate (e.g. estimator returns structure in directions the gate said
were undetectable), flag DISAGREEMENT loudly; this is a headline diagnostic, not an error
to hide.

Output: G_hat where every edge carries a status code:
    EDGE_DECIDED_PRESENT | EDGE_DECIDED_ABSENT | EDGE_UNDECIDED_<reason>
plus discover_report.json. The per-edge status-code format deliberately parallels Uehara's
certificates, and we cite that parallel; our codes concern latent-recovery ambiguity, not
pairwise noise-model identifiability.

## 5. Validation ladder (Objective 3, built in)

All experiments follow the standing disciplines: oracle harness first (Lesson 1), leak
gate on any ceiling or split (Lesson 10), no trust in seed 0 (Lesson 9), batch runners
assert N artefacts for N jobs with PID lockfiles (Lessons 13-14), every artefact records
resolved input path + content fingerprint (Lesson 21), fresh-clone regeneration test before
any writeup (Lesson 17).

Simulator: reuse and extend the Paper 2 simulator (81_ranktest_oracle lineage: d_latent,
D = 200, edge_prob 0.4, weights and noise U(0.5, 1.5), linear mixing for v1; hard = zero
incoming edges + noise var U(2,4), soft = noise var x U(2.5, 4.0)). Port it into the new
repo as fresh code (no artefact migration, Lesson 20); byte-check its gate0 against the
precondition-audit oracle numbers once, then treat as independent.

E0 (oracle gate, blocking): feed a known-answer configuration through every module readout
before any experiment. Gate must return the constructed k; Recover must return MCC ~= 1 on
clean data at correct n; Discover must return the true graph with all edges DECIDED under
hard multi-node conditions. A rotationally blind baseline (MSE autoencoder) must FAIL the
recovery gate (Lesson 8). If E0 fails, everything stops.

E1 (synthetic, clean): all preconditions satisfied by construction. Expected: gate
PROCEED, recovery succeeds (report MCC and SHD), all edges decided. Establishes the
pipeline works when it should.

E2 (semi-synthetic violation injection, the headline experiment): start from E1 and inject
one violation at a time, sweeping severity:
  a. environment starvation: reduce m below n+1.
  b. power starvation: shrink n_e through the 125 floor.
  c. measurement contamination: replace a fraction of interventional environments with
     measurement-only shifts (diagonal/gain perturbations of the mixing, not the SCM).
  d. rank starvation: reduce intervened-node count k through the k >= 3 power boundary.
Expected pattern, which IS the paper's central figure: gate verdict degrades in lockstep
with recovery quality of the naive (ungated) backbone, and the gated pipeline either
abstains or returns the capped-but-correct subgraph while the naive run returns confident
garbage. Metric: calibration of gate verdict vs realised recovery error (e.g. verdict-
stratified MCC distributions), plus abstention correctness (did UNDECIDED edges actually
correspond to unidentifiable structure).

E3 (real data): run gated vs naive (same backbone, no gate) on the in-hand datasets.
Honest expectation from the five-lane wall: the gate will often say ABSTAIN or
PROCEED_CAPPED on real data. The paper frames this as the finding: the framework correctly
refuses where Paper 2 showed refusal is warranted, and quantifies exactly how many
directions and which resolution each dataset supports. The comparison that must be shown:
what the naive backbone confidently (and unstably) returns on the same data, seed-to-seed
graph instability as the symptom (mirroring the Gamella CCRL instability result).

E4 (optional stretch, only if E1-E3 are done and budget remains): light-tunnel data as an
external real benchmark for the gate's P3/P4 verdicts (not for recovery claims; the mixing
is nonlinear so v1 recovery does not apply there). Reuse nothing from the abandoned
ccrl-newidea2 env.

Decision rule per experiment: pre-register the expected direction before running; a result
that contradicts the gate's prediction is a finding about the gate and must be reported,
not tuned away.

## 6. Baselines and metrics

Baselines: (1) the same backbone run naively on all environments (the essential ablation;
the only difference is the gate); (2) backbone on randomly selected environment subsets of
the same size as the certified set (shows the gate's selection is informative, not just
smaller-m regularisation); (3) for Discover, the backbone's raw graph without abstention.

Metrics: MCC for latent recovery; SHD and per-edge precision/recall computed ONLY over
DECIDED edges plus an abstention-quality pair (coverage = fraction of edges decided,
correctness among decided); gate calibration curves (verdict vs realised error); run-to-run
graph stability (SHD between seeds) as the honesty metric.

## 7. What is and is not claimed (novelty ledger, keep in the paper)

Not claimed: any single component. Rank tests, VAR diagnostics, linear multi-node CRL
estimators, CD algorithms, and the abstention concept all exist and are cited.

Claimed: (a) the first precondition-gated environment-selection front end for LATENT
recovery (no located prior work gates CRL recovery on data-side identifiability tests);
(b) ambiguity-aware abstention over the LATENT graph, the latent analogue of
observed-variable admissibility gating; (c) the empirical characterisation, across
synthetic, semi-synthetic, and eight real datasets, of exactly where identification is and
is not achievable, which is Objective 2's second clause fulfilled with numbers.

## 8. Risks and open design decisions

R1. P3's form outside time series. The VAR diagnostic needs a temporal axis. For
Perturb-seq the mechanism-vs-measurement split must be operationalised differently
(batch-randomisation check; possibly control-vs-control covariance shifts). Decide before
coding Module 1; record the decision here.

R2. Backbone implementation cost. If no usable reference implementation of the
Bing/Varici estimators exists, implementing from the papers is the single largest time
item. Timebox: if a faithful implementation cannot pass E0 within the timebox, fall back
to the simplest provable linear estimator (Squires-style with known/estimated targets) and
say so; the framework claim survives a simpler backbone.

R3. Gate-recovery disagreement on real data. Possible that the estimator succeeds where
the gate says CAPPED (test underpowered) or fails where the gate says PROCEED. This is
reportable signal either way, but the writeup must pre-commit to that framing (Section 5
decision rule) to avoid looking post hoc.

R4. Reviewer pattern-match to observed-variable gating. Mitigated by the page-1
distinction paragraph (Section 10) and the per-edge status codes being explicitly about
recovery ambiguity, not noise-model preconditions.

R5. Ceiling honesty. This is an 8/10 thesis framework and a 6-7/10 mid-tier journal
paper; it is 4-5/10 at CLeaR/UAI/NeurIPS as-is. The upgrade path is Path B (curvature-
robust rank test) slotting in as a novel Module 1 later. Do not oversell v1.

Kill criteria (pre-registered): if E2 shows the gate verdict does NOT track naive-backbone
failure (calibration flat), the central claim is false and the framework reverts to an
audit, which we already have; stop and reassess rather than adding epicycles.

## 9. Infrastructure and repo plan (standing hygiene rules apply)

- New repo github.com/gaurav3507/grd-framework, created at start, own Mac folder
  ~/Downloads/grd-framework, own /workspace/grd-framework on the A100,
  nothing nested under or migrated from any prior project. Track sheet from day one.
- .gitignore day one, ignore contents not directories, verify with git add -An probes
  (Lesson 17; do not repeat the ccrl-newidea2 dead-negation bug).
- Git identity on both machines before first commit: gaurav3507 <ggoyal_phd25@thapar.edu>.
- v1 is CPU-friendly (linear algebra + bootstrap); E1/E2 run on the Mac or A100 CPU.
  nohup + disown + PID lockfiles for anything long. No GPU dependency in v1.
- Module contracts are JSON on disk; every artefact carries resolved input path + sha256 +
  code commit (Lesson 21). literature/ holds the novelty-sweep records already gathered
  (Uehara, Dang, Bing, Varici, 2604.23800, lineage-study shortlist).

## 10. Must-cite and distinguish (page-1 obligations)

1. Uehara 2026 (arXiv 2605.27477) and Dang et al. 2026 (arXiv 2602.08340): the
   observed-variable precedent for gating and abstention; we do the latent setting, they
   have no recover module and no latent preconditions.
2. Gamella, Bing, Runge ICML 2025 (arXiv 2502.20099): the motivating failure; also the
   source of the instability-as-symptom metric.
3. Bing et al. CLeaR 2024; Varici et al. NeurIPS 2024 (arXiv 2406.05937): the backbone's
   identifiability guarantees; we add nothing to them.
4. Jin and Syrgkanis (arXiv 2311.12267): SNA, the source of one abstention code.
5. Rajendran et al. NeurIPS 2024 Remark 2: the environment-count precondition P1 tests.
6. Econometric weak-identification pre-testing ancestry: Staiger and Stock 1997, Stock and
   Yogo 2005, Andrews and Cheng 2012. Citing this converts "just pre-testing" into owned
   framing.
7. Paper 1 (NetN, CMPB) and Paper 2 (JBHI): the diagnostic machinery Modules 1 reuses;
   the framework is their constructive continuation, not a repackaging (state this).
8. Lee, Jin, Aragam 2026 (arXiv 2603.25796) and CRL-from-general-environments 2026 (arXiv
   2604.23800): the current frontier the linear backbone sits inside; cite to show the
   regime choice is deliberate, not ignorant.

## 11. Milestones (order fixed, dates indicative, no external deadline)

M0 Repo + scaffolding + this doc committed. Half a day.
M1 Simulator port + E0 oracle harness green across all three modules. This gates
   everything (Lesson 1). ~1 week.
M2 Module 1 complete on synthetic (P1, P2, P4; P3 decision resolved per R1). ~1-2 weeks.
M3 Module 2 backbone passing E0/E1 (R2 timebox applies). ~2-4 weeks, the risk item.
M4 Module 3 + E1 end to end green. ~1 week.
M5 E2 violation-injection suite, the headline figure. ~2 weeks.
M6 E3 real-data runs (data already in hand, paths known). ~2 weeks.
M7 Writeup: thesis chapter first, then journal manuscript (venue decision after results;
   candidates Neurocomputing / KBS / Pattern Recognition / TNNLS, no APC).

First concrete build step after this doc is approved: M0 repo scaffolding prompt for
Claude Code, then the M1 simulator-port handoff. Nothing runs on the A100 until M1's
oracle is green on the Mac.
