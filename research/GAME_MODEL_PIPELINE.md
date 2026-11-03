# Game Model Research Pipeline

Last updated: 2026-11-03

This is the canonical research path for independent NHL pregame win probabilities. `HWIN` is an internal engineering label only; it is not a public model name. No production model is currently promoted.

## Authority and target

- Canonical game identity and schedule: normalized game spine V3, sourced from per-season SportsDataverse schedules. Do not use the master schedule as a key.
- Winner target: the verified goalie-box `W` decision, cross-checked against PBP home/away winner labels. Never use historical schedule score fields.
- Canonical historical hockey features: the normalized SportsDataverse PBP team-game event dataset. Apply the documented release-to-season mapping in `DATA_AUTHORITY.md`.
- Markets are comparison-only. They are not training labels or feature inputs for the independent model.
- The current pregame-eligible comparison model is Team Strength 004. Goalie Rolling 003, Goalie Layer 001, and Schedule 006 use current-game actual starter identity from a postgame box and are not eligible pregame comparisons.

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

Team Strength 004 combines dynamic Elo, lagged 20-game 5v5 SOG/xG share and non-empty-net scoring, rest, and online opponent-adjusted 5v5 SOG/xG. Its 2022–2026 rolling-origin result is 6,993 games, Brier 0.235382, log loss 0.662976, accuracy 59.47%. It is the current research candidate, not a promoted champion.

## Canonical code and artifacts

- `hwin_team_strength_004.py`: benchmark feature construction and nested rolling-origin research.
- `hwin_schedule_007.py`: leakage-safe schedule increment study with no goalie inputs; it writes game-level OOF Parquet, fold metrics, reliability bins, and a JSON summary.
- `.github/workflows/hwin-schedule-007-no-goalie.yml`: research run referencing the audited historical input artifact run IDs. It runs on the isolated research branch, not `main`. The source Actions artifacts currently expire after one day, so durable reruns need a Drive-to-runner handoff or a versioned rebuild from immutable source releases; the current workflow is not yet durable on its own.
- Experiment registry: the existing Google Sheet `NHL Prediction Model — EXPERIMENT REGISTRY` is canonical for candidate status and decisions.
- Canonical large-data/artifact warehouse: Google Drive `HOCKEY_PREDICTION_MODEL`; store versioned research results in `04_VALIDATION`, promoted models in `03_MODELS/GAME_MODEL`, and raw source releases immutably under `00_SOURCE_DATA`.
- Reproducibility gap: lock exact Python dependency versions and record the manifest/source checksums for each run. Current experiment workflows install unpinned packages and some source artifacts have one-day retention.
- Website exports: only small production JSON files in `public/data/`, after promotion. No model calculations or research database belong in the frontend.

## Promotion gate

Promote only after leakage review, reproducible inputs/dependencies, meaningful out-of-sample improvement over the current benchmark, acceptable calibration, stability across seasons, diagnostics explaining the gain, and a named model version with rollback preserved. Until then, keep work in `RESEARCH` or `CANDIDATE`, with no public prediction export.
