# Pregame Goalie Snapshot Contract v1

## Purpose

Build a timestamped, append-only record of which goalie Hockey Lab expected for each team before puck drop, and when that expectation changed to confirmed. This is research input for a future starter-specific expected-goal adjustment. It does not alter the public cards or production projections.

## Required capture behavior

- Capture every game/team state when the pregame source is polled and whenever its goalie assignment changes.
- Store each observation as a separate JSONL row. Never overwrite an earlier projection with a later confirmation.
- Record capture time in UTC, source-reported time when available, source/provider, game ID, team, goalie ID, assignment state, and forecast/model version.
- Preserve “no assignment” and unavailable-source states. Do not silently substitute the eventual actual starter.
- A later actual starter, saves, goals allowed, or game result belongs in a separate outcome table. The snapshot input must not contain those fields.
- Keep market prices out of this input and out of the independent model.

## Assignment states

- \`projected\`: source listed a likely starter before confirmation.
- \`confirmed\`: source explicitly confirmed the starter at the recorded time.
- \`unassigned\`: source responded successfully but had no goalie assignment.
- \`unavailable\`: source could not be queried or returned an error.

## Storage

The current site is static and its \`today-games.json\` file is replaced during refreshes. It cannot serve as the historical ledger. The live collector must append the schema-valid rows to durable storage before updating the card. Keep a retention-safe export so each research run can pin the exact ledger checksum.

## Model evaluation gate

Use only snapshots with \`captured_at_utc < scheduled_start_utc\`. For each forecast cutoff, select the latest eligible row for that game/team. Train goalie value from strictly prior completed games, shrink by workload, and tune all choices within the chronological training/validation folds. Compare projected-goalie forecasts with the existing 60-game reference on team-goal MAE and team count log loss; report final-score log loss separately. Keep expected and confirmed starter results separate. Never use the actual starter reconstructed after puck drop as the predictor.

## Current blocker

This repository has no persistent backend binding or historical pregame starter archive. The schema and validator can be integrated with the capture source once its durable write path is available. Until then, no goalie-specific feature is eligible for promotion.
