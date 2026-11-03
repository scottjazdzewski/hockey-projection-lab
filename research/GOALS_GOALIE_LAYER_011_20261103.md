# Goals Model EXP-0029 — Goalie Quality and Workload Layer

**Model ID:** `HOCKEY-LAB-GOALS-GOALIE-LAYER-011-RESEARCH`  
**Status:** CANDIDATE for a timestamped pregame-starter validation; not production eligible  
**Run date:** 2026-11-03  
**Champion protection:** Hockey Lab Game v1.0.0 remains frozen. No website export or deployment.

## Question and data limit

Does the opposing starter's established shot-quality performance improve the fixed-60-date goals model? The warehouse contains goalie-level PBP features for 20,820 games and starter rows with PBP shot data coverage of 99.98%. It does not contain an announcement or retrieval timestamp for the same-game starter identity. Therefore this is a retrospective upper-bound experiment: goalie identity is taken from the postgame goalie box, while goalie performance and workload features use only games on strictly earlier dates. It is not a pregame availability test.

The roster and scratch tables also lack timestamps and line combinations. No lineup feature is included.

## Design

For each target season S (2022–2026), train through S−2, tune Poisson regularization on S−1, refit through S−1, and score S. The baseline is the fixed 60-date team/opponent hockey-process model. Tested goalie layers are prior-20-start GSAx/save quality, career-shrunk GSAx/save quality, workload/rest, and quality plus workload. GSAx and workload are strictly lagged by full game date. For a clean feature ablation, each fold's score-law family and parameter are chosen on S−1 using the no-goalie baseline and held fixed across goalie variants.

All 6,560 out-of-fold games have a retrospective actual-starter identity. The score-distribution export contains the career-shrunk candidate's full regulation and final score PMFs.

## Results

| Variant | Mean home/away goal-count log loss | Regulation score log loss | Regulation 3-way Brier | Home / away regulation-goal MAE | Final-total log loss | OT Brier |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 60 dates, no goalie | 1.92155 | 3.83180 | 0.62306 | 1.39548 / 1.34878 | 2.17814 | 0.17308 |
| Opposing starter, prior-20-start quality | 1.92136 | 3.83147 | 0.62273 | 1.39577 / 1.34732 | 2.17837 | 0.17307 |
| Opposing starter, career-shrunk quality | **1.92083** | **3.83019** | **0.62268** | 1.39562 / 1.34809 | **2.17695** | 0.17309 |
| Opposing starter, workload/rest | 1.92148 | 3.83163 | 0.62313 | 1.39609 / 1.34864 | 2.17791 | 0.17308 |
| Opposing starter, quality plus workload | 1.92134 | 3.83142 | 0.62282 | 1.39609 / 1.34734 | 2.17821 | **0.17307** |

The career-shrunk quality layer reduces regulation score log loss by 0.00161 and three-way Brier by 0.00038 versus the no-goalie baseline. Regulation score log loss improves in all five target seasons; three-way Brier improves in four of five. A five-season block bootstrap gives a 95% interval of −0.00236 to −0.00077 for the score-log-loss difference (candidate minus baseline); this is exploratory, based on only five season blocks, and has no multiple-comparison adjustment.

Goal-mean effects are mixed: home-goal MAE worsens by 0.00014 while away-goal MAE improves by 0.00070. Overtime Brier is essentially unchanged. The pooled improvement is real but small, and the feature set was selected from several predeclared ablations.

## Decision

Keep the career-shrunk goalie layer at **CANDIDATE for timestamped pregame-source validation only**. Do not promote or publish it as a pregame forecast: its same-game goalie identity is retrospective and has no recorded announcement time. The earlier win-model goalie battery also found only small, noisy win-probability gains, so the goals-model result is a reason to build and test a proper pregame starter path, not a reason to treat the retrospective identity as live truth. Hockey Lab Game v1.0.0 remains unchanged.

## Next work

1. Identify a free, legally accessible source that records expected/confirmed starter identity and the time it became available; preserve goalie changes before puck drop.
2. Re-run this exact candidate with timestamped starter snapshots, scoring only what was known at each prediction cutoff.
3. Measure starter changes, goalie rest/workload, and lineup availability separately; keep lineup features out until a comparable pregame snapshot exists.
4. Continue calibration and scoring-environment work; the current goals model still underpredicts total goals slightly.

## Artifacts

The experiment package includes the model code, requirements, manifest, fold metrics, OOF predictions, summary/calibration metrics, and full score PMFs for the career-shrunk goalie candidate. The PMF files store float32 probabilities with Zstandard compression; the score ledger retains float64 forecast summaries.
