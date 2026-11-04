# Goalie-Conditional Win Projections

**Status:** Required engineering behavior; research and production integration are not complete.

This document is the source of truth for adding named-goalie effects to Hockey Lab Game Cards. A card's projected win percentage is conditional on the goalie currently expected to start. If the pregame feed later identifies a different starter, Hockey Lab must recalculate the projection for that goalie and refresh the card.

## Required card behavior

For each game:

1. Begin with the promoted team-only probability, which holds team strength, schedule, venue, and opponent constant.
2. When the live context feed names an expected goalie, calculate and display the probability conditional on that goalie. Keep the goalie name and state visible as PROJECTED or CONFIRMED.
3. If a later feed update changes the expected goalie or confirms a different goalie, rerun the scorer from the same team-only baseline with the newly selected goalie. Update win probabilities and fair moneylines together.
4. Preserve the source timestamp, Hockey Lab observation timestamp, goalie identity, and model version used for each pregame snapshot.
5. If no goalie can be resolved, display the team-only baseline and mark the goalie adjustment unavailable. Never invent an identity or rating.
6. Freeze every published pregame snapshot after puck drop. Later starter or result updates may create a new record, but must not overwrite the historical forecast.

The card should make the adjustment easy to read: adjusted win percentage as the main projection; expected/confirmed goalie status; and the goalie effect in percentage points versus the team-only baseline. The baseline can appear as a secondary figure.

## Goalie rating and probability adjustment

Goalie wins and losses are excluded as a goalie-skill feature because game results depend on the team. The candidate goalie rating is built from the individual goalie's performance before the target game. Start with SV% and GAA as transparent measures. Evaluate shot-quality measures such as GSAx per 30 shots when coverage supports them.

The rating must account for:

- **Age:** estimate the goalie aging curve from historical, prior-only goalie performance. Fit the curve on earlier seasons and select its shape only within chronological validation. Do not assume a fixed peak age or impose a hand-picked decline.
- **Experience and sample size:** measure prior starts, shots faced, and minutes. Calibrate shrinkage so limited samples stay closer to an age-adjusted league prior and larger samples contribute more individual evidence. Starts alone are not an adequate exposure measure.
- **Recency:** test career, recent-start, and decayed histories in rolling-origin validation. Select the memory window and decay using training data only.
- **Team and shot context:** treat GAA cautiously because it includes team defense and shot quality. Prefer GSAx when available and validated. Avoid counting the same performance signal twice.

The probability conversion is trained from historical games with goalie ratings available before puck drop. A candidate form is:

P(home win | expected goalies) = logistic(logit(P_team-only) + beta × (home goalie rating − away goalie rating))

The rating scale, coefficient, shrinkage, age curve, and any fallback must be learned and versioned. Do not hand-set a percentage-point effect. Sportsbook odds and closing-market probabilities remain benchmark-only and must never enter this calculation.

## Pregame identity and updates

The live-context feed supplies goalie identity, provider status, source timestamp, source URL, and Hockey Lab observation time. Use the provider player ID to join to goalie history where possible; use a name-only match only when it resolves uniquely.

A PROJECTED goalie is the current expected-starter input and may drive a clearly labeled projected probability. A CONFIRMED update for the same goalie changes status and provenance. A new goalie identity changes the input and must trigger a fresh score. Keep each score tied to the goalie snapshot that produced it.

The update workflow must detect both:
- a changed PROJECTED goalie, including a change before confirmation; and
- a newly CONFIRMED goalie that differs from the goalie used in the current projection.

The scorer, rather than the browser or live-context updater, recalculates probabilities. The browser renders the newest valid projection export. The fair moneyline must be recomputed from the same adjusted probability.

## Projection export contract

Extend the game-projection export and its JSON schema with enough information to audit and render the adjustment. At minimum each team side needs:

- team-only baseline win probability;
- goalie-adjusted win probability;
- signed goalie adjustment in percentage points;
- fair moneyline derived from the adjusted probability.

The game record must preserve the away and home goalie IDs/names, PROJECTED or CONFIRMED status, provider timestamp, Hockey Lab observation timestamp, and goalie-adjustment model version. The top-level model feature version must include the explicit `goalie-adjustment-` marker to identify a promoted goalie layer and activate the live rescore trigger. A projection must not claim goalie adjustment when either rating is missing; use the documented baseline fallback and show the adjustment as unavailable.

## Calibration and promotion gate

The goalie layer remains RESEARCH or CANDIDATE until it passes all of the following:

1. Strict prior-date feature construction, with leakage checks.
2. Rolling-origin tests across multiple seasons against the same frozen team-only baseline.
3. Brier score, log loss, calibration, and probability-shift diagnostics; assess uncertainty by season and goalie exposure.
4. Age-curve and sample-shrinkage selection performed only within the training/validation window.
5. A prospective ledger using timestamped pregame goalie identities and immutable probability snapshots.
6. Validation that a goalie change updates the projected percentage and fair line on the card without changing the team-only baseline.
7. Explicit promotion and model-version change before public probabilities use the layer.

A positive goalie-to-goalie scenario calculation alone does not pass this gate. It shows how the candidate model maps a goalie difference into probability, not that the effect is calibrated or predictive.

## Current implementation state

The live goalie feed and card already carry goalie names/statuses. The promoted Game v1.0.0 scorer is still team-only and marks outputs with a no-goalie-layer flag. The live-context workflow now compares selected goalie identity and status, including PROJECTED changes, and requests a scorer rerun only when the active feature version includes the explicit `goalie-adjustment-` marker. The current feature version is team-only, so goalie-feed changes update the context display but do not yet change public probabilities. The goalie rating, production scorer integration, export schema, and adjusted card rendering still require research, implementation, and validation before the public win percentage becomes goalie-conditional.

## Change-control checklist

When implementing this layer, update together:

- research feature builder and rolling-origin experiment;
- age/exposure-calibrated goalie rating export;
- production scorer and model feature version;
- live-context rescore trigger for projected and confirmed identity changes;
- game-projection schema and validator;
- Game Card rendering and fair-moneyline output;
- forward-ledger snapshot and audit fields;
- this document and the production handoff.

Do not let the browser compute the model, let postgame starter identity enter a pregame rating, or rewrite a frozen snapshot.
