# HWIN Diagnostic Battery 001 — First Pass

**Status:** Research-only. No production probabilities, weights, or scorer files were changed.

**Data:** 6,993 frozen chronological holdout games from Schedule Context 007 (2022–2026). The diagnostics score the existing `baseline_b2b` probabilities.

**Run:** [GitHub Actions run 42045215658](https://github.com/scottjazdzewski/hockey-projection-lab/actions/runs/42045215658)  
**Research PR:** [#3](https://github.com/scottjazdzewski/hockey-projection-lab/pull/3)

## First-pass findings

- Overall Brier score: **0.23456**; log loss: **0.66119**; accuracy: **60.00%**.
- Mean predicted home win probability was **54.08%**, against a **53.65%** observed home-win rate: about **0.42 percentage points of aggregate home-probability overprediction**. The global calibration slope was **1.002**, so this does not point to a broad “probabilities too extreme” problem.
- Season residuals changed direction. Home-win probability was overpredicted by about **2.1–2.2 points** in 2023 and 2026, while it was underpredicted by about **2.4 points** in 2025. That pattern argues for checking season/context drift before considering a global recalibration.
- Rest flags by themselves did not isolate a clean 4-in-6 or 5-in-8 penalty. In 3-in-4 games, the model overpredicted home wins by **1.04 points**, compared with **0.17 points** in unflagged games. In 4-in-6 games, the residuals were nearly identical in flagged and unflagged games (**−0.44 vs −0.42 points**). These are raw slice gaps and are not yet uncertainty-adjusted.
- The away road-trip-finale proxy had a **−1.22-point** calibration residual (predicted minus observed home win probability) versus **−0.14 points** in unflagged games. This suggests the away team may be stronger than the model recognizes in this context, but the earlier team/date-bootstrap interval crossed zero; it is a lead to verify, not evidence for promotion.
- Initial probability bands showed local miscalibration: predicted home probabilities of **35–40%** were low by **3.37 points**; **55–60%** were high by **2.20 points**; and **70%+** were low by **1.86 points**. These bands need confidence intervals and season consistency checks before any modeling response.
- Team-level pooled residuals were uneven, but pooled team gaps alone can be confounded by roster eras and schedule mix. They are not interpreted as evidence that any team needs a bespoke adjustment.

## Validation note

A first workflow run succeeded and produced the results above. While reviewing the run, I found a bug in the favorite-confidence calculation and fixed it on the research branch. That run predates the correction, so its favorite-confidence results are excluded here. I also added team-by-season slices.

I have now added a season-stratified date/team-cluster bootstrap for the schedule-state residual gaps. The interval calculation detects the available date and away-team columns and records a clear skip reason if those fields are missing. The updated code is committed to the research branch, but its new workflow output has not yet been independently verified. Treat the first-pass values above as descriptive; do not use them as evidence for promotion until the corrected run and intervals are available.

## Next tests

1. Verify the rerun of corrected favorite calibration, team-by-season cuts, and clustered uncertainty intervals.
2. Require schedule and probability-band residuals to persist across seasons rather than selecting the largest pooled gap.
3. For any persistent schedule signal, test incremental value against the frozen baseline with a small, predeclared feature family (rest-day difference, travel/rest interaction, trip game number, and 3-in-4 / 4-in-6 state). Use rolling/forward validation and compare Brier and log loss with paired uncertainty intervals.
4. Only propose a production change if it improves held-out proper scores, is calibrated, survives season-block checks, and does not depend on market data. Otherwise retain the baseline.

**Interpretation:** The current evidence points to regime/season and specific probability-range checks as the strongest next diagnostics. Raw back-to-back flags alone do not show a stable enough improvement target in this pass.
