# Independent NHL Market Benchmark on Game Cards

## Display contract

Each Game Card has two completely independent layers:
- Hockey Lab forecast, produced by the promoted game model and gated by the production manifest.
- Market benchmark, loaded by the website from public/data/market-benchmark.json. It is NEVER read by prediction code, research training, feature engineering, or production scoring.

The browser joins market rows by NHL game ID AND both team abbreviations. Wrong-day snapshots and wrong-team rows do not display. Missing market data uses a placeholder; it never hides the prediction.

## Live provider / one-time setup

Provider: The Odds API: GET https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/?regions=us&markets=h2h&oddsFormat=american

The provider advertises a free 500-credit monthly plan supporting NHL moneylines across multiple sportsbooks. Price and quota may change.

ONE-TIME USER ACTION: obtain an API key at https://the-odds-api.com/ and save it as the GitHub Actions repository secret THE_ODDS_API_KEY. Never write the key in the public repo, site JSON, or front-end JavaScript. Without that secret the workflow skips the live fetch and retains the previous market file.

The independently scheduled workflow .github/workflows/update-market-benchmark.yml collects and commits ONLY the market data file and dated research archive. It neither runs nor changes the prediction scorer.

## Consensus calculation

Per sportsbook, require a complete 2-way full-game (including OT/SO) head-to-head moneyline, timestamped pre-start, current within 90 minutes of collection, and matched to both teams and the scheduled game time.

For each book, American -A implies A/(A+100); +A implies 100/(A+100). Calculate bookmaker away no-vig as P(away)/(P(away)+P(home)); home=1-away. Combine the per-book away no-vig probabilities with an equal-book median. The consensus is retained internally as de-vigged percentages for benchmarking; Game Cards now show only the MEDIAN RAW BOOK MONEYLINE for each side. Median book moneylines retain overround and are not converted from fair probability. One book is a reference, NOT consensus.

Research-only disagreement remains Hockey Lab win probability minus market no-vig win probability, in percentage points. This calculation is not displayed on Game Cards or used by the model.

## Capture, freshness and freeze

- On NHL game days, capture at **12:20 PM ET**, then **6:50, 7:20, 7:50, 8:50, and 9:50 PM ET**. These six scheduled checks include a daytime reference and near-puck checks for typical 7:00, 7:30, 8:00, 9:00, and 10:00 PM games. Actual GitHub Actions start times can be delayed and cannot guarantee an exact final closing tick.
- At most **6 automatic requests per game day**, or **180 credits in a 30-day month / 186 credits in a 31-day month**, assuming the provider charges one credit for the single US-region h2h market request. Days without any upcoming NHL games, late scheduled runs outside the 24-minute grace window, and finished slates skip paid API calls. Manual dispatch and other uses of the same API key could add credits, so monitor the provider dashboard.
- An already-captured scheduled slot is skipped to reduce duplicate requests. The bookmaker feed can move more often than the collection schedule; the website shows the latest saved snapshot, not every intra-interval tick.
- Open Game Cards re-fetch the independent market JSON every five minutes and when the browser window regains focus, updating only their market strips without rerendering the promoted Hockey Lab probabilities.
- Every saved row includes when the request was captured, the participating books, and each book's source quote timestamp.
- Prices captured after scheduled start are NEVER substituted into existing pregame records.
- When the start passes, a snapshot saved at most 25 minutes before start is labeled CLOSING (meaning last locally captured pregame snapshot, not an exchange-certified final tick); older rows are LAST_PREGAME.
- On cards, an unfrozen quote older than 60 minutes is visibly marked as not current.
- Permanent daily research copies are written to research/market_benchmarks/live/YYYY-MM-DD.json. The existing immutable model forecasts are never rewritten.
- November 3 began with a 7:27PM StatMuse site-reported reference; its underlying book count and pregame timestamp are not independently verified. It is clearly distinguished from verified multi-book consensus. Future quotes supersede it only for games that have not started.

## Validation

Run: python src/market/update_market_benchmark.py --self-test

Run: node src/production/validate-public-data.mjs

No live API key is required to test calculations or the pregame freeze.
