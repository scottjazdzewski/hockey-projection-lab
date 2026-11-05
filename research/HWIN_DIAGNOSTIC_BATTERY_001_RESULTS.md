# HWIN Diagnostic Battery 001 — Verified First Pass

**Status:** Research-only. No production probabilities, weights, or scorer files changed.

**Data:** 6,993 frozen chronological holdout games from Schedule Context 007 (2022–2026), scoring the existing `baseline_b2b` predictions. I ran the updated branch script against that exact frozen Parquet input and verified that all 1,000 bootstrap replicates completed.

**Initial workflow run:** [GitHub Actions run 42045215658](https://github.com/scottjazdzewski/hockey-projection-lab/actions/runs/42045215658)  
**Research PR:** [#3](https://github.com/scottjazdzewski/hockey-projection-lab/pull/3)  
**Verified interval table:** [hwin_diagnostic_battery_001_schedule_cluster_bootstrap.csv](hwin_diagnostic_battery_001_schedule_cluster_bootstrap.csv)

## Overall probability quality

- Brier score: **0.23456**; log loss: **0.66119**; accuracy: **60.00%**.
- The model averaged **54.08%** home-win probability; home teams won **53.65%**. That is about **0.42 percentage points of aggregate home-probability overprediction**.
- Calibration slope was **1.002**, so the aggregate issue does not look like a general “probabilities too extreme” problem.
- The error changed direction by season: home wins were overpredicted by about **2.1–2.2 points** in 2023 and 2026, and underpredicted by about **2.4 points** in 2025. That is a regime stability question, not a reason to apply one global correction.

## Schedule-state diagnostics

Intervals below are for the difference in calibration residual (observed minus predicted home-win rate) between flagged and unflagged games. Negative values mean the flagged games were more overpredicted. The bootstrap resampled dates within seasons and applied shared team-cluster weights for both home and away roles.

| Slice | Flagged vs unflagged residual gap | 95% interval | Read |
|---|---:|---:|---|
| Either team in 3-in-4 | −0.87 pp | −7.62 to +5.54 pp | Includes zero |
| Either team in 4-in-6 | −0.03 pp | −6.52 to +5.98 pp | Essentially no raw gap |
| Either team in 5-in-8 | −0.11 pp | −6.84 to +5.40 pp | Includes zero |
| Away road-trip finale proxy | −1.08 pp | −8.44 to +5.71 pp | Includes zero |

These multiway intervals are broad. They do not support promoting any of these schedule flags as a probability adjustment. The road-trip finale remains a plausible hypothesis to investigate with a sharper, predeclared test, not a confirmed edge.

## Other calibration slices

- Home probabilities from **35–40%** were low by **3.37 points**; **55–60%** were high by **2.20 points**; **70%+** were low by **1.86 points**. The changing direction across bands argues against a blanket offset. Those bins still need uncertainty-adjusted, season-aware confirmation.
- Corrected favorite-confidence diagnostics also move in both directions: favorites at **60–65%** were high by **2.35 points**, while **70–75%** were low by **2.58 points**. The **80%+** band has only 65 games and is too small for a confident read.

## Team-season signal worth checking

Home-team slices are noisy when pooled over changing roster eras. On a complete 32-team by five-season scan (about 41–54 home games per team-season), Philadelphia was the only team whose home residual had the same sign in every season: its home-win probability was overpredicted by **13.1 points** in 2022, **1.1** in 2023, **5.4** in 2024, **0.7** in 2025, and **0.4** in 2026. The pooled gap is about **4.2 points across 210 games**, but its magnitude is mostly driven by 2022. Because Philadelphia emerged from scanning all teams, this is a post-hoc lead requiring a multiple-comparison correction and forward-season validation before any team-specific feature is considered.

## Decision and next test

**No production change.** The diagnostics do not show a stable rest-state or getaway-game correction. The next useful modeling test is a preregistered forward-validation comparison for season/context drift and team-era strength, with block-aware uncertainty and correction for searching across teams and probability bins. Only a candidate that improves held-out Brier and log loss and persists in later seasons should advance.

Market odds remain benchmark-only and are not used as model inputs.
