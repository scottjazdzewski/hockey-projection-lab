# HWIN Schedule Context 007 — Results

Run: GitHub Actions workflow run 41937808076 (2026-11-04/05)
Evidence artifact: `hwin-schedule-007` (90-day retention)

## Design

Five chronological rolling-origin test seasons (2022–2026), 6,993 held-out games. For each target season, the prior season was used for validation and earlier seasons for training. Every candidate was compared with the same research baseline containing the B2B indicators. The reported scores are weighted over the five held-out folds.

Schedule states are calculated from games played before the current game. A 3-in-4 flag means at least two prior games during the preceding three days; 4-in-6 means at least three prior games during the preceding five days; 5-in-8 means at least four prior games during the preceding seven days.

The road-trip-finale proxy is true only when the current game is the second or later consecutive away game and the team’s next scheduled game is at home. It uses the published schedule and does not measure actual travel, flight timing, distance, or departure time.

## Overall results

Research baseline: Brier 0.234560; log loss 0.661189; accuracy 60.00%.

| Candidate over B2B baseline | Delta Brier | Delta log loss | Seasons with better Brier |
|---|---:|---:|---:|
| 3-in-4 | -0.0000043 | -0.0000087 | 3/5 |
| 4-in-6 | -0.0000115 | -0.0000223 | 3/5 |
| 3-in-4 + 4-in-6 | +0.0000134 | +0.0000246 | 3/5 |
| 5-in-8 | +0.0000131 | +0.0000205 | 3/5 |
| Road-trip-finale proxy | -0.0000241 | -0.0000446 | 4/5 |
| All schedule context together | +0.0000613 | +0.0001203 | 2/5 |

The road-trip-finale proxy produced the largest isolated gain, but it is still very small. In the 1,856 held-out finale-proxy games, conditional Brier was 0.235358 versus 0.235438 for the baseline. The 3-in-4 and 4-in-6 lifts are also tiny; combining the indicators did not help.

## Decision

No production change. Keep the road-trip-finale proxy as a research candidate only. Do not promote 3-in-4, 4-in-6, 5-in-8, or the combined schedule bundle from this result. A follow-up should test stability with paired per-game uncertainty and a prospective shadow ledger before considering promotion.

The comparison baseline and metrics are for this research experiment and should not be read as a direct rerating of the live production probabilities.
