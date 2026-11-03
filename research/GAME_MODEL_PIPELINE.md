# Game Model Research Pipeline

Last updated: 2026-11-03 (Score State 009)

This is the canonical research path for independent NHL pregame win probabilities. `HWIN` is an internal engineering label only; it is not a public model name. No production model is currently promoted.

## Authority and target

- Canonical game identity and schedule: normalized game spine V3, sourced from per-season SportsDataverse schedules. Do not use the master schedule as a key.
- Winner target: the verified goalie-box `W` decision, cross-checked against PBP home/away winner labels. Never use historical schedule score fields.
- Canonical historical hockey features: the normalized SportsDataverse PBP team-game event dataset. Apply the documented release-to-season mapping in `DATA_AUTHORITY.md`.
- Markets are comparison-only. They are not training labels or feature inputs for the independent model.
- The current pregame-eligible comparison backbone is Team Strength 004. Schedule 007's B2B-only variant remains a research candidate; it is not promoted. Score State 009's tied-state xG feature has a slightly better pooled score but uncertain paired evidence and remains RESEARCH. Goalie Rolling 003, Goalie Layer 001, and Schedule 006 use current-game actual starter identity from a postgame box and are not eligible pregame comparisons.

## Feature contract

For game *G*, every feature must use information available before *G* begins. State updates occur after a game's prediction is recorded. Rolling team metrics are shifted by one game; Elo and opponent-adjusted team ratings are recorded before the game result updates them; schedule features use only earlier dates. When date-only data cannot establish the order of multiple same-day games, suppress ambiguous location-sequence features.

Any new layer must have a pregame source path as well as historical backtest value. This applies to expected/confirmed goalie identity, scratches, injuries, rosters, usage, lines, and pairings. A postgame actual starter is not a safe substitute for a pregame starter snapshot.

## Validation protocol

Use nested chronological rolling-origin folds. For target source season `T`:

1. Fit candidate models on completed seasons `<= T-2`.
2. Select hyperparameters/features on season `T-1`.
3. Fit the selected model on train plus validation.
4. Save probabilities for season `T` before using those games to update any state.

Current development target seasons are 2022–2026. They are not an untouched final test: source season 2026 was inspected in prior experiments, and early 2026–27 shadow results have also been viewed. The next final forward test must freeze timestamped pregame predictions prospectively.

At minimum report Brier score, log loss, accuracy, calibration intercept/slope, and reliability bins. Preserve the per-game out-of-fold predictions in Parquet; aggregate metrics alone are insufficient. Break out season and probability band first, then month, team, home/away, goalie state, rest/B2B, schedule density, expected-goal environment, and strength difference when coverage permits. Calibration diagnostics do not authorize post-hoc adjustments on the same held-out predictions.

## Current benchmark

Team Strength 004 combines dynamic Elo, lagged 20-game 5v5 SOG/xG share and non-empty-net scoring, rest, and online opponent-adjusted 5v5 SOG/xG. Its 2022–2026 rolling-origin result is 6,993 games, Brier 0.235382, log loss 0.662976, accuracy 59.47%. It is the no-goalie comparison baseline.

Schedule 007 reuses Team Strength 004's fold-specific Elo/opponent-adjusted choices and adds strictly lagged schedule features. The B2B-only variant scored Brier 0.234726, log loss 0.661540, accuracy 59.92%; Brier improved in all five target seasons. Pooled calibration intercept/slope are -0.0174/0.9979, with decile reliability data saved. Schedule density added a smaller score gain, location-sequence features worsened Brier/log loss, and the full schedule bundle was weaker than B2B alone. Treat B2B as a research candidate only: source season 2026 and early 2026–27 shadow games were previously inspected, and no untouched final test remains.

Special Teams 008 tested lagged PP xG generation, PK xG allowed, penalty-event exposure, and finishing residuals on top of Team Strength 004 + B2B. PP xG quality scored Brier 0.234812, log loss 0.661708, accuracy 59.63%, versus the B2B model at 0.234726 / 0.661540 / 59.92%; Brier improved in only 2/5 seasons. Adding exposure and finishing did not help (Brier 0.234863). This feature layer is **REJECTED** for now: the canonical features lack official power-play opportunity counts, and penalty-event counts are a noisy proxy. The results do not justify a PP/PK probability adjustment.

Score State 009 tested lagged tied- and close-state 5v5 xG share on top of Team Strength 004 + B2B, across the same five chronological folds and 6,993 held-out games. Tied-state xG lowered Brier from 0.234726 to 0.234630 and log loss from 0.661540 to 0.661329; accuracy moved from 59.92% to 59.96%, and Brier improved in 4/5 seasons. The paired-game bootstrap 95% interval for ΔBrier was [-0.000223, +0.000029], crossing zero. Close-only worsened pooled Brier, and tied+close was weaker than tied-only. The same-day team-game order guard passed. Decision: **RESEARCH**, not candidate or promoted; do not recalibrate or publish this layer.

## Canonical code and artifacts

- `hwin_team_strength_004.py`: benchmark feature construction and nested rolling-origin research.
- `hwin_schedule_007.py`: leakage-safe schedule increment study with no goalie inputs; it writes game-level OOF Parquet, fold metrics, reliability bins, and a JSON summary.
- `hwin_special_teams_008.py`: leakage-safe incremental test of lagged PP/PK features; it writes game-level OOF Parquet, fold metrics, reliability bins, and a provenance manifest. Official PP opportunities are not available; penalty-event counts are labeled as a proxy.
- `hwin_score_state_009.py`: tied/close 5v5 xG state-share ablation on the frozen Team Strength 004 + B2B comparison; writes per-game OOF Parquet, folds, reliability, subgroup diagnostics, and a checksum-bearing manifest.
- `requirements-game-model.txt`: exact Python package pins used in Score State 009.
- `.github/workflows/hwin-schedule-007-no-goalie.yml`: research run referencing audited historical input artifact IDs.
- `.github/workflows/hwin-special-teams-008-no-goalie.yml`: repeatable, pinned-dependency run on branch `codex/hwin-special-teams-008-no-goalie`; run 41571326250 succeeded and writes a checksum-bearing manifest. It references the same source Actions artifacts, which expire after one day; durable reruns still need a Drive-to-runner handoff or a versioned rebuild from immutable releases.
- Experiment registry: the existing Google Sheet `NHL Prediction Model — EXPERIMENT REGISTRY` is canonical for candidate status and decisions; Schedule 007 variants are EXP-0010 through EXP-0013, Special Teams 008 is EXP-0014, and Score State 009 is EXP-0015.
- Canonical large-data/artifact warehouse: Google Drive `HOCKEY_PREDICTION_MODEL`; store versioned research results in `04_VALIDATION`, promoted models in `03_MODELS/GAME_MODEL`, and raw source releases immutably under `00_SOURCE_DATA`.
- Reproducibility: Score State 009 pins Python dependencies and records Drive input/output checksums; its frozen result bundle and manifest are in Drive `04_VALIDATION`. It ran locally from retained Drive source bundles. Earlier workflow artifacts still expire after one day, and an automated Drive-to-runner handoff remains open.
- Website exports: only small production JSON files in `public/data/`, after promotion. No model calculations or research database belong in the frontend.

## Next research priorities

1. Retest tied-state xG with a predeclared 10/20/40 prior-game window set and a small fixed shrinkage option. Select only on each fold's validation season and report window-by-season stability plus team/season-clustered uncertainty.
2. Test a few predeclared B2B interactions (home/away, team-strength difference, and density state) without expanding the search on the inspected 2026 target season.
3. Freeze a prospective 2026–27 shadow ledger with prediction timestamp, input cutoff, model/feature version, goalie and lineup state, and immutable pregame probabilities; append outcomes only after final results.
4. Investigate trusted free pregame goalie/lineup sources and timestamped availability before building those layers.
5. Revisit PP/PK only if a free, legally reusable source supplies a defensible opportunity denominator or time-on-advantage.

No candidate is ready for production or a public site export.

## Promotion gate

Promote only after leakage review, reproducible inputs/dependencies, meaningful out-of-sample improvement over the current benchmark, acceptable calibration, stability across seasons, diagnostics explaining the gain, and a named model version with rollback preserved. Until then, keep work in `RESEARCH` or `CANDIDATE`, with no public prediction export.
