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

## Paired uncertainty check

A paired game bootstrap resampled predictions within each target season (2,000 draws). The 95% intervals for every overall candidate included zero. For the road-trip-finale proxy, the overall Brier change was -0.0000241 (95% interval -0.000076 to +0.000030) and log-loss change was -0.0000446 (95% interval -0.000154 to +0.000067). Inside the 1,856 proxy games, Brier change was -0.000080 (95% interval -0.000252 to +0.000096) and log-loss change was -0.000142 (95% interval -0.000499 to +0.000213).

The bootstrap is paired at the game level and stratified by target season. It does not cluster games by team or date, so it is a first uncertainty check rather than a final dependence-aware test.

## Team/date cluster check

A second paired bootstrap resampled team nodes (pigeonhole weights by both teams in each matchup) and game dates, separately within each target season (2,000 draws). Every 95% interval still included zero.

| Road-trip-finale proxy comparison | Cluster | Games | Delta Brier | 95% interval | Delta log loss | 95% interval |
|---|---|---:|---:|---:|---:|---:|
| All held-out games | Team | 6,993 | -0.000024 | -0.000114 to +0.000075 | -0.000045 | -0.000232 to +0.000164 |
| All held-out games | Date | 6,993 | -0.000024 | -0.000077 to +0.000029 | -0.000045 | -0.000157 to +0.000066 |
| Finale-proxy games only | Team | 1,856 | -0.000080 | -0.000371 to +0.000212 | -0.000142 | -0.000746 to +0.000485 |
| Finale-proxy games only | Date | 1,856 | -0.000080 | -0.000246 to +0.000097 | -0.000142 | -0.000487 to +0.000230 |

The result does not hold up as a statistically clear gain under either resampling scheme.

## Forward-ledger check

The latest published forward-ledger summary is generated 2026-11-04 and contains 10 settled games for `team-strength-004+b2b-007` (Brier 0.254175; log loss 0.705264). It contains no parallel road-trip-finale shadow probabilities, so it cannot yet compare this candidate against the active model. Ten games would also be too few to decide on a signal this small. A prospective comparison would require freezing a separate shadow probability alongside each production forecast before puck drop.

## Decision

No production change. Keep the road-trip-finale proxy and compressed-schedule features research-only. The historical lift is small and all game-, team-, and date-resampled 95% intervals include zero. If the proxy is pursued further, first add a separate immutable shadow prediction to the prospective ledger, then wait for a useful sample before assessing it.

The comparison baseline and metrics are for this research experiment and should not be read as a direct rerating of live production probabilities. The workflow artifact `hwin-schedule-007` contains the fold ledger, subgroup scores, per-game predictions, and all bootstrap outputs; run 42042884528 retains it through 2027-02-03.
