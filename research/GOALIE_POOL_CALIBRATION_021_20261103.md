# EXP-0035 — Calibrate the Full Goalie Pool

**Status:** RESEARCH; no promotion or public export  
**Model:** `HOCKEY-LAB-GOALIE-POOL-CALIBRATION-021-RESEARCH`  
**Run date:** 2026-11-03

## Question

Can the model distinguish a named goalie’s repeatable shot-quality performance from PBP xG baseline drift, and how much individual goalie signal persists into later starts?

## Method

From 41,633 historical starter rows and 289 goalies, each goalie’s GSAx is centered against a league baseline estimated before that game date. Candidate baselines (expanding, rolling 365 days, rolling 1,095 days), goalie memories (career, prior 20 or 40 starts), and pseudo-shot shrinkage were selected through rolling-origin validation. Same-date starts are excluded from history features. The model is fit to future goalie GSAx relative to the prior-date pool mean, not team wins.

## Results

- 22,100 out-of-fold goalie starts across target seasons 2018–2026; 204 goalies represented in the test folds.
- Shot-weighted RMSE of future GSAx per 30 shots: **1.8174** for a neutral pool prior and **1.8032** for the calibrated goalie rating, a **1.56% weighted-MSE reduction**.
- Correlation between the calibrated rating and future goalie outcome: **0.083**. This is evidence of a small persistent individual signal, not a high-confidence talent ranking.
- The rating improved in 8 of 9 test seasons, but the 2024 fold was marginally worse; the largest gains were in 2025 and 2026. The pooled gain is modest.
- Pool calibration slope for the pooled test is 0.782, showing that remaining goalie differences should continue to be shrunk.

## Decision

Keep this as a goalie-pool calibration layer for research. It does not by itself establish a win-probability improvement or justify public goalie rankings. Current target starter IDs come from postgame goalie boxes, so these historical tests are not live-eligible until timestamped pregame starter identity is available.

The next experiment (EXP-0036) applies the calibrated named-goalie skill to the team expected-goal rates and evaluates the resulting full-game win probabilities.

## Artifacts

- Reproducible script: `research/hockey_goalie_pool_calibration_021.py`
- OOF goalie-start ratings and season-level tests: archived with EXP-0035 artifacts
