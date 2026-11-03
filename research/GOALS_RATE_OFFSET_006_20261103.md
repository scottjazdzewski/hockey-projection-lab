# Hockey Goals Rate Offset 006 — Wider Shrinkage Grid

**Experiment:** EXP-0024  
**Model ID:** `HOCKEY-LAB-GOALS-RATE-OFFSET-006-RESEARCH`  
**Status:** RESEARCH  
**Run date:** 2026-11-03  
**Champion protection:** Hockey Lab Game v1.0.0 remains frozen. No HWIN or market inputs. No website export.

## Question and design

EXP-0023 selected the maximum tested common offset strength (`λ=1`) in all five validation folds. EXP-0024 repeats the same rolling-origin pipeline with the wider, predeclared grid `{0, .5, 1, 1.25, 1.5, 1.75, 2}`. For each target season S, side-specific log-rate residuals and λ are estimated on S−1 only; models are refit through S−1 and scored on S. All target outcomes remain out of selection.

## Result

The wider search selected **λ=1 in all five folds**. This confirms that the EXP-0023 choice was not an artifact of the grid's upper boundary. Since the selected correction, model features, and target predictions match EXP-0023, the out-of-sample metrics are the same:

- Final-total mean error improves from **−0.186 to −0.045 goals/game**.
- Home/away regulation-goal MAE changes from **1.398 / 1.351** to **1.402 / 1.353**, a small regression.
- Final exact-score log loss improves from **3.7136 to 3.7104**; final-total log loss improves from **2.17984 to 2.17723**.
- Regulation 3-way Brier changes from **0.62564 to 0.62582**; OT Brier from **0.17320 to 0.17323**.

The bias and likelihood gains do not improve point accuracy or probability Brier. EXP-0024 therefore adds evidence about the offset's selected strength, but no new case for promotion.

## Decision and next step

Keep at **RESEARCH** and do not publish. Hockey Lab Game v1.0.0 remains the frozen win-probability champion. The next rate-layer test is a structural comparison of the current pooled home/away Poisson regression against separately fitted home-goal and away-goal regressions using the same lagged hockey features. This tests whether distinct home/road scoring relationships improve mean accuracy without a global rate correction.
