# Hockey Prediction Lab engineering instructions

## Goalie-conditional projections

Before changing goalie ratings, game scoring, live-context handling, projection exports, validators, or game-card rendering, read:

- `docs/GOALIE_CONDITIONAL_WIN_PROJECTIONS.md` (the source of truth)
- `docs/PRODUCTION_HANDOFF.md`
- `docs/LIVE_CONTEXT_FEEDS.md`

Follow these requirements:

- A game-card win probability is conditional on the goalie currently expected to start. If the selected starter changes before puck drop, rescore the game and update its win probabilities and fair moneylines.
- Estimate goalie quality from performance measures such as save percentage and goals-against average, with the documented limits on GAA. Do not treat goalie wins or losses as goalie-quality inputs; wins are team outcomes. Add expected-goals-against and goals-saved-above-expected measures when reliable inputs are available.
- Learn aging and performance curves from prior-only data. Shrink goalie estimates according to calibrated exposure, including starts and available shots or minutes; do not use unshrunk small-sample rates or hand-coded aging assumptions.
- Preserve the team-only baseline alongside the adjusted projection, goalie identity and status, timestamps, model version, percentage-point adjustment, and adjusted fair moneyline. When identity or a calibrated rating is unavailable, use the team-only baseline and label the goalie adjustment unavailable.
- Keep market odds out of model inputs.
- Store pregame snapshots as immutable observations after puck drop; later goalie-feed updates must not rewrite a game's historical pregame forecast.

## Current goalie calculation state

The calculation hook is in `src/production/goalie_adjustment.py`; the production scorer, projection schema/validator, Game Card, forward ledger, and goalie-history workflow are wired to carry the adjusted result and its audit fields. The model artifact `artifacts/models/goalie-adjustment-model.json` is still CANDIDATE with no fitted parameters. Do not activate or publish goalie effects until the age curve, starts/shots/minutes shrinkage, and probability coefficient have passed rolling-origin and prospective validation.

## Production promotion guard

The active production model is team-only while its feature version is `team-strength-004+b2b-007` and its output carries the `no_goalie_layer` flag. Do not remove that flag, change the feature version, or present production percentages as goalie-adjusted until the goalie model has passed the validation and promotion gate in the source-of-truth document.

When implementing the layer, update the scorer, output schema and validator, live-context rescore trigger, game-card presentation, and forward snapshot ledger as one versioned change. Keep the existing rescore trigger gated on the explicit `goalie-adjustment-` feature-version marker so live-feed changes cannot silently alter team-only probabilities.
