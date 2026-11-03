# Hockey Lab — Production Data Handoff

This repository has two separate jobs:

1. **Research/modeling** produces validated hockey intelligence.
2. **The website** displays small, frozen production exports.

The browser must never reproduce the model calculation.

## Canonical public exports

| File | Owner | Purpose |
|---|---|---|
| `public/data/today-games.json` | schedule updater | Real NHL slate only |
| `public/data/game-projections.json` | promoted game model | Pregame probabilities and game context |
| `public/data/model-performance.json` | validation pipeline | Frozen-model performance metrics |
| `public/data/team-ratings.json` | team layer | Team profile ratings |
| `public/data/player-ratings.json` | player layer | Player profile ratings |
| `public/data/fantasy-rankings.json` | fantasy layer | Season-long fantasy outputs |

Schemas live in `artifacts/schemas/`.

## Promotion gate

A game projection may be published only when all are true:

- model status is `PROMOTED`
- game exists in the real schedule ledger
- projection was generated before puck drop
- `data_cutoff <= generated_at <= game start`
- away + home win probability equals 1 within tolerance
- probabilities are in [0,1]
- no required field contains NaN/Infinity
- model ID/version/feature version are present
- original pregame export is immutable after puck drop

If any check fails, the site must show **Model pending** or **Withheld**. It must never invent a value.

## Work-tab handoff

When the engineering Work tab promotes a model:

1. Generate `public/data/game-projections.json` from the promoted pipeline.
2. Run `node src/production/validate-public-data.mjs`.
3. Commit only after validation passes.
4. Cloudflare deploys the GitHub commit automatically.
5. Verify the live Game Cards match the committed JSON.
6. Freeze that pregame export in the research ledger before games start.
7. Append outcomes later; never rewrite the original prediction.

## Model naming

The public model name is intentionally not hard-coded in the contract. Use `model.model_id` and `model.version`. The current internal engineering label can change without breaking the site.

## Failure behavior

Missing file, stale file, invalid schema, unknown game ID, or non-promoted model:
**display no probability**.

The frontend is a renderer, not a fallback calculator.


## Production manifest

The machine-readable inventory is:

`public/data/production-manifest.json`

A surface is:
- `EMPTY` when the contract exists but no promoted output is available.
- `READY` only when its export is promoted, validated, and safe for the site.
- `WITHHELD` when an expected export is intentionally blocked because a production gate failed.

The manifest does not override the validator. A surface marked READY with invalid underlying data must still fail validation.


## Website publication switch

The browser requires **two independent conditions** before displaying model-derived values:

1. the underlying export must satisfy its contract (for example, a `PROMOTED` game or team model), and
2. the matching surface in `public/data/production-manifest.json` must be `READY`.

This prevents a research or candidate file from appearing simply because it was copied into `public/data/`.

For a promotion, update the export and its manifest surface together, run the validator, then commit.


## Promotion utility

To prepare a validated export for publication without hand-editing manifest metadata:

```bash
node src/production/promote-surface.mjs game_projections
```

The default is a **dry run**.

If the dry run is correct:

```bash
node src/production/promote-surface.mjs game_projections --apply
node src/production/validate-public-data.mjs
node src/production/validate-site-contract.mjs
```

Supported surfaces:

- `game_projections`
- `model_performance`
- `team_ratings`
- `player_ratings`
- `fantasy_rankings`

The utility refuses to mark a model surface READY unless its export is already promoted and non-empty. Team ratings additionally require all 32 current NHL teams.
