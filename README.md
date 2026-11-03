# Hockey Projection Lab

Standalone NHL modeling and product repository.

This project is intentionally isolated from Rail & Rye Racing. Do not bind Rail & Rye routes, domains, databases, storage, secrets, or production assets to this repository.

## Live development site

https://hockey-projection-lab.scottjazdzewski.workers.dev/

Cloudflare deploys the hockey repository automatically.

## Operating lanes

### Research / engineering

The Work tab owns model research, historical data engineering, feature development, validation, model comparison, and promotion decisions.

Research code belongs outside `public/`.

Do not use website files as a research database.

### Production / site

The website consumes small validated exports from `public/data/`.

Current canonical exports:

- `today-games.json`
- `game-projections.json`
- `model-performance.json`
- `team-ratings.json`
- `player-ratings.json`
- `fantasy-rankings.json`

The frontend must never reconstruct the predictive model.

## Production contract

Read:

`docs/PRODUCTION_HANDOFF.md`

Schemas:

`artifacts/schemas/`

Validation:

`node src/production/validate-public-data.mjs`

GitHub also runs `.github/workflows/validate-production-data.yml` whenever production data changes.

## Identity layer

Current NHL team identity crosswalk:

`src/data/nhl-team-crosswalk.json`

Use this as the canonical current-team abbreviation/name mapping for joins between model outputs and the site.

Historical franchise aliases should be handled separately in research code.

## Model publication rule

A model may populate public Game Cards only when it has been explicitly promoted and its export passes production integrity checks.

Until then, the live site must show blank/pending values.

Never fabricate a probability.

Never silently rewrite a frozen pregame projection after puck drop.

## Public model name

The public-facing model name is intentionally not hard-coded. Internal research labels may change without breaking the production contract.

## Protection

- Do not touch Rail & Rye.
- Do not introduce paid data without approval.
- Do not use future information in pregame features.
- Do not manually type model performance statistics into the site.
- Do not create duplicate production exports for the same surface.
