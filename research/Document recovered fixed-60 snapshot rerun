# Goals Model Fixed-60 Snapshot Rerun — 2026-11-04

**Purpose:** Recover and rerun the strongest saved pregame-eligible goals benchmark, then diagnose remaining total-goal and overtime error.  
**Status:** Research only. No production, Hockey Lab win-model, or Game Card changes.

## Recovery and snapshot identity

The connected GitHub integration was sufficient; no GitHub browser login was needed. Two previously successful read-only Actions runs supplied the data artifacts:

| Dataset | Workflow run | Artifact | Artifact ZIP SHA-256 |
|---|---:|---|---|
| Play-by-play team features | 41766609805 | hockey-pbp-features-2010-2027-20261102 | `3e1cbbaf4b7916ddd125e3dfa5281c8032589de90d2fdbeb17109c30d40c6148` |
| Canonical game spine | 41766612176 | hockey-canonical-spine-v3-20261102 | `35da1f914e494f871e3bc9ad3c151e70c9340e4afb7a8c01d5fc51ce43d06eb6` |

These downloads are valid, but their Parquet hashes differ from the frozen EXP-0027/0028 input hashes. The recovered files are therefore identified as a **new 2026-11-04 snapshot**, not represented as byte-exact recovery of the November 2 inputs.

| Input | Frozen EXP-0027/0028 SHA-256 | Recovered SHA-256 | Match |
|---|---|---|---|
| `game_team_event_features.parquet` | `8fc51d8de6db68fca1ea9ab850627c044354d60364b63d6e8531f18110c6f3ef` | `d8871553d43ee42433687de5c1ee9fc421949f796b0725e79976d66836d7c0e6` | No |
| `games.parquet` | `e3b6a3cb9784a91ec6e396e1a7e72330451abee055ed0ba54e0408e37f40a0e3` | `bc1459150f6f055711b96855433b8c254bd2dab118fd637f23b86e86bb5989cf` | No |
| PBP source manifest | `3c4ab0fc8fbc56ec2591c6722e89fa36f721fad2379e4aa9d61460a6cddd5b2c` | `5ae5ff5cd03b9a49451686e1d7e14a2e349446bc0d1ff64ac66f8f2a26e064f1` | No |
| Spine source manifest | `8b19be878cf291c7969f0cbd67447c977cd5b3929dfa00261ce40a891dd53e66` | `259860c366079c8050447da8674a6de7f1ed08f62aab798e2a039d22181bcafa` | No |

## Rerun

Reran the EXP-0027 fixed-60 model from the research branch on the recovered snapshot. The rolling-origin design and predictors were unchanged: train through season (S-2), tune on (S-1), refit through (S-1), and score (S). Markets, HWIN probabilities, same-game goalie identity, and lineups remain excluded.

All five target seasons contain 1,312 regular-season games (6,560 total). Regulation-tie/overtime consistency and canonical winner checks passed. The model excluded 605 historical rows for label-quality reasons, including eight from 2016 onward.

The recovered run reproduced the stored EXP-0027 headline metric artifacts byte-for-byte:

- `fold_metrics.csv`: `12dc00127361370e7823d922b8c940257c94f76fd062a71595198a9c22cdd9e2`
- `context_metrics.parquet`: `d97c9957455786a168a56d1a41dd4554d488d03cd7e377f1cdc9d56bf24cff37`
- `reliability_bins.parquet`: `7ec6e825cf25b31d52aaa4d94a78b171aa0ee72cbcd7a119c23f81f63712c54f`

The new OOF predictions are retained with this run, but their file hash differs from the old manifest because the run metadata and input snapshot differ. The older OOF file itself is not present in the repository export, so this is exact parity for those saved metric artifacts and headline scores—not a claim of exact input or OOF-file reproduction.

## Results and error diagnosis

| Metric | Recovered fixed-60 result |
|---|---:|
| Home regulation goals: MAE / mean forecast error | 1.39548 / −0.06216 |
| Away regulation goals: MAE / mean forecast error | 1.34878 / −0.09893 |
| Final-total goals: MAE / mean forecast error | 1.85170 / −0.16836 |
| Final-total log loss | 2.17814 |
| Regulation score log loss | 3.83180 |
| Regulation three-way Brier | 0.62306 |
| Overtime Brier | 0.17308 |
| Predicted / observed overtime rate | 21.51% / 22.24% |
| Predicted / observed over 5.5 rate | 53.53% / 57.16% |
| Predicted / observed over 6.5 rate | 43.08% / 46.17% |

The total-goal shortfall is spread across seasons: mean forecast error ranged from −0.047 to −0.256 goals/game in each of the five target seasons. A five-season cluster bootstrap gives 95% intervals of −0.244 to −0.092 for final-total error, −0.114 to −0.010 for home regulation-goal error, and −0.136 to −0.065 for away regulation-goal error. These intervals use only five season clusters and should be read as coarse development diagnostics.

Across predicted-total deciles, nine of ten deciles underpredicted the observed final total; errors ranged from −0.045 to −0.298 goals/game. The highest predicted-total decile overpredicted by 0.093. Team-goal deciles were also below actual scoring throughout, with the largest shortfalls in the middle-to-upper range (up to 0.145 goals/team-game). This supports a broad rate-level deficit with nonuniform calibration, rather than an error confined to only the highest-scoring games. It does not establish a safe fixed correction: the previous prior-season offset experiment reduced average bias but worsened MAE and probability scores.

OT probabilities have little separation: decile mean forecasts ranged only from 18.4% to 24.4%, while observed rates were nonmonotonic. Overall the model underpredicted OT by 0.73 percentage points and its Brier score remains effectively at the constant-rate benchmark. The OOF export stores aggregate regulation-tie probability, not individual diagonal score-cell probabilities, so exact 0–0/1–1/2–2 tie-mass calibration remains unmeasured here.

The model already includes rest days, back-to-back status, and games in the prior three and five days. Its saved context export separates neither team on B2B, home-only, away-only, and both. The pooled B2B group has 1,722 games versus 4,838 neither-B2B games; the B2B group has worse regulation score log loss (3.849 vs. 3.826) but lower regulation-outcome Brier (0.615 vs. 0.626). This does not yet test 3-in-4, 4-in-6, or getaway games.

## Decision and next work

The fixed-60 benchmark remains **RESEARCH / CANDIDATE**, not promoted. The rerun confirms its saved headline metrics on the rebuilt snapshot while leaving the exact frozen source-input mismatch documented. The next goal-model tests should fit shared and side-specific log-rate intercepts/slopes using only the preceding validation season, then score the next season; separately audit exact regulation-tie score cells and test 3-in-4, 4-in-6, and getaway states. Preserve a prospective 2026–27 shadow ledger before any promotion decision.

No production files, HWIN model logic, public exports, or Game Cards were changed.
