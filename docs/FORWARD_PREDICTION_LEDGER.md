# Hockey Lab Forward Prediction Ledger

The forward ledger is the production audit trail for pregame forecasts.

## Immutability rule

Each NHL game receives one record at `artifacts/forward/games/<game_id>.json`.
The `forecast` block is written only once, before puck drop, and is never
rewritten. Postgame automation may update only `result` and `evaluation`.

This prevents historical rescoring, corrected lineups, later goalie news,
postgame statistics, or future model versions from leaking into the forecast
that was actually available before the game.

## Freeze timing

The live-context workflow performs its final goalie/lineup check roughly
15 minutes before each scheduled game. The forward-ledger workflow is offset
from that refresh and normally freezes the production snapshot roughly
10 minutes before :00/:30 starts. It refuses to create a new forecast record
after puck drop.

The frozen evidence includes:

- model ID, version and feature version
- generated-at and data-cutoff timestamps
- away/home win probability and fair price
- expected goals, regulation win probabilities and overtime probability
  when a promoted model publishes them
- today's rest/load state
- goalie names, statuses and source evidence
- the current projected lineup objects
- a reserved market-benchmark block; market data is never a model input

## Settlement

Open records are checked against the NHL score feed. Once a game is final,
the ledger stores the final score, winner, OT/SO status and settlement time.
It then calculates forward-only winner accuracy, Brier score and log loss.
When expected goals are present it also calculates team-goal MAE, total-goals
absolute error and goal-differential absolute error.

## Summary

`public/data/forward-ledger-summary.json` is a compact derived surface for
future Model Performance UI work. It contains overall forward metrics,
per-model-version metrics and one summary row per frozen game.

The per-game files remain the authority.
