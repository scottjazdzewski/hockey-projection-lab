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

The calculation hook is in `src/production/goalie_adjustment.py`; the production scorer, projection schema/validator, Game Card, forward ledger, and goalie-history workflow are wired to carry the adjusted result and its audit fields. The model artifact `artifacts/models/goalie-adjustment-model.json` is promoted as `prospective-0.3.0` against `team-strength-004+b2b-007`. Its rolling-origin lift is small and uneven (3 of 5 seasons improved; 2026 worsened), so keep the prospective ledger under review. A promoted artifact must set `baseline_feature_version` to the exact active team-only `FEATURE_VERSION`; the scorer rejects calibrations fitted against another baseline.

## Production rollout and prospective monitoring

Model `prospective-0.3.0` is promoted against the exact active baseline and is eligible for pregame projections. The public projection file may retain the old team-only version until the next pregame scoring run; never rewrite forecasts after puck drop. The scorer now runs daily before the slate, uses the projected starter, and the live-context workflow rescores a changed or newly confirmed starter once the active projection has the `goalie-adjustment-` marker.

Continue comparing goalie-adjusted probabilities with the frozen team-only baseline in the forward ledger. The rolling lift is small and uneven (3 of 5 seasons improved, while 2026 worsened), so future model changes require prospective results and a fresh versioned review.
