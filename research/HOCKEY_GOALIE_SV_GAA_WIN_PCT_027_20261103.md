# EXP-0042 — Named Goalie Save Percentage and GAA on Win Probability

**Model:** `HOCKEY-LAB-HWIN-NAMED-GOALIE-SV-GAA-027-RESEARCH`  
**Run date:** 2026-11-03  
**Status:** RESEARCH; no promotion or public export  
**Champion:** Hockey Lab Game v1.0.0 remains frozen.

## Question

Does the actual named goalie’s individual save percentage and goals-against rate add useful information to the frozen team win percentage? Goalie win-loss record is excluded because it is a team outcome.

## Method

The experiment uses 3,936 regular-season games from 2024–2026 and compares the frozen Team Strength 004 + back-to-back probabilities with a named-goalie challenger. It uses the actual home and away starters in each game, rates each goalie from his own earlier starts, and adjusts the baseline home-win logit using the difference between the two goalies’ predicted save percentages and goals against per 60 minutes.

For each target season, rolling-origin selection chooses a league-centered goalie-rate estimate and shrinkage from the preceding season, fits the win adjustment only on earlier seasons, selects the feature family and ridge strength on the preceding validation season, refits through that season, and evaluates the next season. A no-goalie forecast is retained when validation does not support an adjustment. League reference rates use the prior 365 days, with a strictly prior-date expanding fallback. Career, last-20-start, and last-40-start goalie histories are compared with pseudo-exposure shrinkage.

The goalie history and league rates exclude the target game date. **Starter identity comes from the postgame goalie box**, so the test is retrospective upper-bound evidence; it does not establish that the same starter was known before puck drop. No team-goalie W–L feature, market data, or production/site output is used.

## Results

| Metric | Frozen HWIN v1.0.0 | SV% + GAA challenger | Change |
|---|---:|---:|---:|
| Home-win Brier | 0.238209 | 0.238248 | +0.000039, worse |
| Home-win log loss | 0.668622 | 0.668658 | +0.000036, worse |
| Winner accuracy | 58.10% | 58.10% | unchanged |
| Mean absolute probability shift | — | 1.23 percentage points | — |

The paired 14-day block-bootstrap 95% intervals include no change for both Brier (**−0.000371 to +0.000437**) and log loss (**−0.000843 to +0.000879**). Results varied by season: the challenger lost ground in 2024, improved in 2025, and lost ground again in 2026.

The full goalie-pool calibration produced a small future-performance signal. Across the held-out starts, save-percentage performance was essentially flat to the league baseline (shot-weighted log-loss change **+0.0000044** and Brier change **+0.0000004**); it improved in 2025 and 2026 but was slightly worse in 2024. Calibrated GAA-rate forecasts improved Poisson loss in all three test seasons (weighted change **−0.00197 per goalie-hour**), but the downstream win model assigned GAA a zero coefficient in each season; the selected win model retained save percentage alone. The GAA result must be interpreted cautiously because GAA also reflects team defense and shot quality.

Selected win-probability challenger by test season:

| Test season | Selected feature | Mean absolute win-probability shift | Brier change | Log-loss change |
|---|---|---:|---:|---:|
| 2024 | Save percentage | 0.86 pp | +0.000276 | +0.000548 |
| 2025 | Save percentage; GAA coefficient selected as zero | 1.62 pp | −0.000526 | −0.001217 |
| 2026 | Save percentage; GAA coefficient selected as zero | 1.20 pp | +0.000368 | +0.000776 |

## Decision

**Do not promote this adjustment.** The actual named goalie changes the research probabilities, but the average pooled scores are marginally worse and the season results are inconsistent. This is not evidence that goalie identity has no effect; it means this save-percentage/GAA formulation has not yet shown reliable incremental win-prediction value. Hockey Lab Game v1.0.0 and the live site remain unchanged.

## Next step

Run the same individual-goalie comparison from timestamped pregame starter snapshots so the model uses the goalie actually announced before puck drop. Keep save percentage and GAA as transparent baseline goalie measures, then compare them with the existing calibrated GSAx/30 estimate and, when a reliable shot-quality history is available, goals saved above expected. Add prospective forecasts to an immutable ledger and score them only after results are final.

## Reproducibility

The independent rerun reproduced the summary JSON and all 3,936 prediction rows exactly. See `research/hwin_goalie_sv_gaa_027.py`, `research/exports/HWIN_GOALIE_SV_GAA_027_20261103/summary.json`, and `research/exports/HWIN_GOALIE_SV_GAA_027_20261103/oof_goalie_sv_gaa_win_pct.parquet`.
