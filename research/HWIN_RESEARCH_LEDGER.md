# HWIN Research Ledger

Last updated: 2026-11-03 (Score State 009)

## Status

**No production HWIN model has been promoted yet.**

Current status: Team Strength 004 + B2B remains the benchmark candidate. Score State 009 tied-state xG is **RESEARCH** because the effect is small and its paired interval includes zero.

Current research backbone:
- dynamic Elo-style team strength
- rolling 20-game 5v5 shots-on-goal share
- rolling 20-game 5v5 expected-goal share
- rolling recent non-empty-net scoring
- rest-days difference

This backbone is the benchmark that new layers must beat incrementally.

## Data authority

See `research/DATA_AUTHORITY.md`.

Winner target:
- goalie box-score `W` decision
- 22,139 verified regular-season/playoff results in canonical spine V3
- zero winner-team identity mismatches

Canonical normalized spine:
- 24,014 scheduled games
- 22,139 verified results
- 20,611 regular season
- 1,528 playoffs

Canonical full-PBP HWIN event population:
- 20,819 games
- 41,638 team-game rows
- exactly two sides per game
- 0 duplicate game-side keys
- 0 unresolved winner labels
- 2014-15 full PBP unavailable in audited archive

## Signal Audit 002

Development validation: source season 2025.
Development test: source season 2026.
Current source season 2027 remains prospective shadow.

Best simple event carry-forward signal in 2026 development test:
- rolling 20-game 5v5 SOG share difference
- Brier 0.245176
- log loss 0.683296
- accuracy 55.02%

Event multivariate benchmark:
- 2025 Brier 0.241244
- 2026 Brier 0.245639

## Elo Baseline 001

Best initial training-grid parameters:
- K = 10
- home advantage = 20 Elo points
- offseason retention = 0.70

2025:
- Brier 0.238835

2026:
- Brier 0.247660

Interpretation: Elo and event form carry different information.

## Fusion 001

Elo + broader event form.

2025:
- Brier 0.237479

2026:
- Brier 0.246175

Promising, not promoted.

## Rolling Backtest 001

Nested rolling-origin evaluation for target seasons 2022-2026.
Each target season tunes only on the preceding season and trains on earlier history.

Total games across folds: 6,993.

Weighted results:

| Model | Brier | Log loss | Accuracy |
|---|---:|---:|---:|
| Constant home | 0.248723 | 0.690592 | 53.65% |
| Elo | 0.237896 | 0.668352 | 58.49% |
| Event backbone | 0.239300 | 0.671283 | 58.26% |
| Fusion backbone | **0.236064** | **0.664431** | **59.32%** |

Fusion Brier by target season:
- 2022: 0.228961
- 2023: 0.232873
- 2024: 0.234206
- 2025: 0.237392
- 2026: 0.246943

Event backbone beat fusion in 2026 Brier, but fusion beat event backbone in 2022-2025 and is best overall across all five folds.

Conclusion:
- dynamic team strength is durable
- 5v5 territorial form adds independent value
- fusion is the current research benchmark
- no single 2026 result should determine feature promotion

## Goalie Layer

Goalie feature ETL 001:
- 20,820 games
- 83,283 goalie-game rows
- 41,640 starter rows
- exactly two starters per game
- starter PBP shot coverage: 99.983%
- zero duplicate goalie-game keys
- cross-era comparable xG metric: SOG-based GSAx
- richer unblocked-shot attribution available in modern seasons

Confirmed-starter HWIN Layer 001:
- broad goalie bundle did **not** improve the 2026 development test
- backbone Brier 0.246230
- backbone + broad goalie bundle Brier 0.246556
- broad goalie bundle is not promoted

Goalie Signal Audit 002:
- best single incremental signal so far: 20-game starter save percentage
  - 2026 Brier 0.245949
  - delta vs same backbone: -0.000281
- second: 20-game SOG-based GSAx
  - 2026 Brier 0.246009
  - delta vs backbone: -0.000221
- effects are small and require rolling-origin confirmation before inclusion
- short/medium/long goalie features were highly correlated and unstable when bundled

Rolling-origin goalie test is required before any goalie feature is promoted.

## Schedule Increment 007 — No-Goalie

Five nested rolling-origin target seasons (2022–2026), 6,993 games. The experiment froze the fold-specific Elo/opponent-adjusted choices from Team Strength 004 and tuned logistic regularization on each prior validation season. The Team Strength 004 baseline Brier reproduced exactly in every fold. All variants exclude goalie inputs and use schedule features from prior dates only; ambiguous same-date location sequence is missing.

| Variant | Brier | Log loss | Accuracy | Brier folds better than baseline |
|---|---:|---:|---:|---:|
| Team Strength 004 baseline | 0.235382 | 0.662976 | 59.47% | — |
| + back-to-back | **0.234726** | **0.661540** | 59.92% | 5/5 |
| + schedule density | 0.235248 | 0.662693 | 59.13% | 5/5 |
| + location sequence | 0.235420 | 0.663052 | 59.97% | 3/5 |
| + all schedule groups | 0.234800 | 0.661700 | 59.65% | 5/5 |

**Best schedule signal so far: back-to-back indicators.** Versus Team Strength 004, B2B lowered Brier by 0.000655, lowered log loss by 0.001437, and raised accuracy by 0.44 percentage points. The full schedule bundle was weaker than B2B alone; location sequence worsened Brier/log loss despite higher accuracy. This suggests a specific fatigue signal may help, while adding loosely related schedule features can dilute the probability improvement.

Pooled calibration for Team Strength 004 + B2B: intercept -0.0174, slope 0.9979; mean home probability 0.5407 versus actual home win rate 0.5365. Reliability deciles are included in the frozen artifact. Calibration was measured diagnostically and not applied to the predictions.

Status: B2B remains **CANDIDATE**, not promoted. The historical test seasons were already inspected and the early 2026–27 shadow slice has been viewed, so there is no untouched historical holdout. A genuine prospective forward test is still required. Output bundle and source manifest: `HWIN_SCHEDULE_007_NO_GOALIE_20261103.zip` and `HWIN_SCHEDULE_007_MANIFEST_20261103.json` in Drive `04_VALIDATION`.

## Promotion rules

A candidate layer must:
1. be pregame-only and leakage-safe
2. improve Brier/log loss, not only accuracy
3. survive multiple rolling-origin seasons
4. add value after existing backbone features
5. have a live-data path that can reproduce the same feature before puck drop
6. be versioned and archived before promotion

Market odds remain comparison data, not an input to the independent HWIN baseline.


## Goalie Rolling Backtest 003 — Invalid Pregame Comparison

**Status: REJECTED for pregame use.** Although this backtest reports five rolling-origin seasons, it selects each game's actual starter from a postgame goalie box. That starter identity was not known at the historical pregame cutoff, so the performance below is retrospective and cannot support a pregame goalie feature or production decision. Schedule 006 has the same starter-selection leakage.

Archived retrospective results:
- shrunk prior starter save percentage: Brier 0.235900, log loss 0.664074, accuracy 59.57%
- raw 20-game starter save percentage: Brier 0.236004
- 20-game SOG-based GSAx: Brier 0.236018

These values are retained only as descriptive diagnostics. A valid goalie experiment requires a timestamped expected/confirmed starter source or a model that predicts starter identity using only pregame information.

## Team Strength 004 — Recency / Opponent Adjustment

Nested rolling-origin targets 2022-2026, 6,993 games.

Flat-20 backbone:
- Brier 0.236064
- log loss 0.664431
- accuracy 59.32%

Exponentially weighted recent form:
- Brier 0.235665
- log loss 0.663555
- accuracy 59.62%
- delta Brier vs flat-20: -0.000399

Pure opponent-adjusted SOG/xG ratings:
- Brier 0.235509
- log loss 0.663242
- accuracy 59.57%
- delta Brier: -0.000556

**Flat-20 + opponent-adjusted SOG/xG hybrid:**
- Brier **0.235382**
- log loss **0.662976**
- accuracy 59.47%
- delta Brier: **-0.000683**
- Brier improved vs flat-20 in 2023, 2024, 2025 and 2026; 2022 was slightly worse

Opponent-adjusted offense/defense components separately:
- Brier 0.235411
- log loss 0.663031
- accuracy 59.53%
- slightly worse Brier than the simpler aggregate opponent-adjusted hybrid

Interpretation:
- schedule/opponent quality materially improves the team-strength representation
- recency decay also helps, but opponent adjustment adds more
- the simpler aggregate opponent-adjusted SOG/xG signal is preferred over four separate offense/defense component terms
- **new research backbone candidate = Elo + flat-20 5v5 form + opponent-adjusted 5v5 SOG/xG + recent scoring + rest**
- still research-only; not production HWIN


## Special Teams Increment 008 — No-Goalie

Five nested rolling-origin target seasons (2022–2026), 6,993 games; train through T-2, tune on T-1, test T. Team Strength 004 + B2B is the comparison baseline.

| Variant | Brier | Log loss | Accuracy | Brier folds better than B2B |
|---|---:|---:|---:|---:|
| Team Strength 004 + B2B | **0.234726** | **0.661540** | **59.92%** | — |
| + PP xG quality | 0.234812 | 0.661708 | 59.63% | 2/5 |
| + PP xG, exposure proxy | 0.234810 | 0.661705 | 59.75% | 2/5 |
| + PP xG, exposure, and finishing | 0.234863 | 0.661813 | 59.85% | 1/5 |

Features use a 20-game-half-life exponentially weighted mean from earlier dates only. Same-date results never update a feature before puck drop; actual starter identity is excluded. The PBP package lacks official power-play opportunity counts, so opponent penalty events proxy PP exposure and team penalty events proxy PK exposure. This denominator is noisy. The PP layer slightly worsened pooled Brier/log loss and did not show a repeatable lift in seasons or the highest pregame penalty-exposure quartile.

**Decision: REJECTED as an incremental probability feature.** The candidate adds no production signal now. Do not recalibrate it after the fact. An experiment with official PP opportunity counts would need a free, legally usable source and timestamped pregame availability.

Reproducible outputs: GitHub Actions run 41571326250 on isolated branch `codex/hwin-special-teams-008-no-goalie`; artifact `hwin-special-teams-008-no-goalie-20261103`. Store the artifact bundle and manifest under Drive `04_VALIDATION`. The permanent registry ID is EXP-0014.


## Score State Increment 009 — Tied/Close 5v5 xG, No-Goalie

Five nested rolling-origin target seasons (2022–2026), 6,993 games. The comparison baseline is Team Strength 004 + B2B. Each feature uses a rolling share of 5v5 xG for over xG for plus against, from a team's prior 20 games only (minimum 8). No current-game event, goalie, market, or postgame starter input is used.

| Variant | Brier | Log loss | Accuracy | Brier folds better than B2B |
|---|---:|---:|---:|---:|
| Team Strength 004 + B2B | 0.234726 | 0.661540 | 59.92% | — |
| + tied-state xG share | **0.234630** | **0.661329** | **59.96%** | 4/5 |
| + close-state xG share | 0.234742 | 0.661575 | 59.83% | 2/5 |
| + tied and close xG shares | 0.234663 | 0.661412 | 59.89% | 4/5 |

The tied-state model's pooled calibration intercept/slope are -0.01786/1.00090, compared with -0.01739/0.99789 for the baseline. Its Brier gain is 0.000097 and log-loss gain is 0.000211. The 2022 Brier was fractionally worse (0.228259 vs 0.228251); 2023–2026 improved. The paired-game bootstrap (5,000 draws, seed 9009) gives mean ΔBrier -0.0000966 and a 95% interval of [-0.0002232, +0.0000291]. This interval crosses zero and is not clustered by repeated team or season, so uncertainty may be understated.

Same-day repeated team-game ordering was checked against canonical start times; rolling features use shift(1); the Team Strength 004 + B2B fold Brier values reproduced; home/away probabilities sum to one. Feature availability in target OOF rows was 99.9% for tied-state share on both sides. Close-only was worse overall, and adding close share to tied share diluted the tied-only gain.

**Decision: RESEARCH.** This small pooled improvement is hypothesis-generating, not promotion evidence. No calibration change or public prediction export is authorized by this result. Next test a predeclared 10/20/40-game tied-state window set and a limited shrinkage option, selected only on each chronological validation season. Preserve the prospective 2026–27 shadow ledger separately.

Reproducible artifacts: `HWIN_SCORE_STATE_009_NO_GOALIE_20261103.zip` and `HWIN_SCORE_STATE_009_MANIFEST_20261103.json` in Drive `04_VALIDATION`. Code is on isolated branch `codex/hwin-score-state-009`; dependencies are pinned in `research/requirements-game-model.txt`. Registry ID: EXP-0015.
