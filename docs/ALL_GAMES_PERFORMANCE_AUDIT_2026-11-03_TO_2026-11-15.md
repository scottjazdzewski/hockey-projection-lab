# All Hockey Lab NHL games since the production model launched
**Audit date:** November 16, 2026  
**Scope:** Nov 3–15, 2026; 95 completed NHL games, 95 verified GitHub morning predictions. Nov 16 scheduled games excluded because they were not completed when auditing.  
**Primary per-game output:** [95-game morning audit CSV](../artifacts/audits/nhl_morning_predictions_2026-11-03_to_2026-11-15.csv).

## Method (do not conflate with forward-freeze ledger)
The repository was created November 2; the first **promoted production forecast** was posted November 3. For each of the 13 NHL slate dates, recover the earliest day's GitHub commit that explicitly published `HOCKEY-LAB-GAME` and contained the complete morning slate. Check the commit's time was **strictly before the start of every selected game**, check `generated_at` before game start and `data_cutoff` not later than generation, and pair games by date/away/home to externally reported finals. No post-puck probability recalculation or replay is allowed. Outcome sources are saved on each CSV row (WebPuck's NHL-feed scoreboard for Nov 3–11, 13–14; Hockey-Reference for Nov 12; CBS Sports for Nov 15). Audit uses morning snapshots, **not** subsequent pregame goalie-confirmation scores. The raw 18 immutable forward-ledger freezes remain a separate cohort and are not overwritten. The exact source commit for each record is included in CSV.

### Summary
- **55/95 correct picks = 57.9%**, 40 misses. This is *all 95 verified morning game predictions* between Nov 3 and Nov 15.
- **Brier 0.2375; log loss 0.6674**. A constant 50/50 prediction has Brier 0.2500 and log loss 0.6931 on any full-game binary sample.
- Home-team selections: **30/58** (51.7%); away-team selections: **25/37** (67.6%). Both are descriptive small subsets.
- 84 games have paired published goalie-adjusted and team-only probabilities; goalie-adjusted Brier **0.2358** vs **0.2347** team-only, delta **+0.0010 (worse)**. No model promotion conclusion should be made from this short window.
- No live pregame expected-goals forecasts for these 95 games; xG scoring is **not available**.
- Historical 6,993-game rolling-origin model-performance page reports 60.05% historical accuracy, but that is a separate evaluation population, not the record below.
- Model's research feature input was observed to be stale (as far back as Nov 2 at end of audit window), potentially limiting current-form responsiveness. This merits independent repair/investigation, not a post-hoc recalibration.

### Per-day results
| Date | Morning games | Correct | Accuracy | Brier | Model | Source feature cutoff |
|---|---:|---:|---:|---:|---|---|
| 2026-11-03 | 11 | 5 | 45.5% | 0.2504 | 1.0.0 | 2026-11-01 |
| 2026-11-04 | 4 | 3 | 75.0% | 0.2103 | prospective-0.3.0 | 2026-11-02 |
| 2026-11-05 | 11 | 8 | 72.7% | 0.2227 | prospective-0.3.0 | 2026-11-02 |
| 2026-11-06 | 3 | 2 | 66.7% | 0.1792 | prospective-0.3.0 | 2026-11-02 |
| 2026-11-07 | 12 | 9 | 75.0% | 0.1821 | prospective-0.3.0 | 2026-11-02 |
| 2026-11-08 | 6 | 4 | 66.7% | 0.2118 | prospective-0.3.0 | 2026-11-02 |
| 2026-11-09 | 3 | 2 | 66.7% | 0.2442 | prospective-0.3.0 | 2026-11-02 |
| 2026-11-10 | 7 | 2 | 28.6% | 0.2656 | prospective-0.3.0 | 2026-11-02 |
| 2026-11-11 | 6 | 4 | 66.7% | 0.2463 | prospective-0.3.0 | 2026-11-02 |
| 2026-11-12 | 11 | 5 | 45.5% | 0.2520 | prospective-0.3.0 | 2026-11-02 |
| 2026-11-13 | 2 | 1 | 50.0% | 0.2790 | prospective-0.3.0 | 2026-11-02 |
| 2026-11-14 | 14 | 7 | 50.0% | 0.2614 | prospective-0.3.0 | 2026-11-02 |
| 2026-11-15 | 5 | 3 | 60.0% | 0.2923 | prospective-0.3.0 | 2026-11-02 |
| **Total** | **95** | **55** | **57.9%** | **0.2375** | Mixed | — |

### Confidence calibration by five-point pick probability band
| Forecast probability | Games | Correct | Actual hit rate | Average model pick probability |
|---|---:|---:|---:|---:|
| 50–55% | 39 | 19 | 48.7% | 52.3% |
| 55–60% | 24 | 15 | 62.5% | 57.0% |
| 60–65% | 15 | 11 | 73.3% | 62.6% |
| 65–70% | 7 | 3 | 42.9% | 66.3% |
| 70–75% | 4 | 2 | 50.0% | 71.6% |
| 75–80% | 5 | 4 | 80.0% | 77.4% |
| 80–85% | 1 | 1 | 100.0% | 80.8% |

**Caution:** one-point 54%, 55%, 56% buckets are badly undersampled in only 95 games. These descriptive proportions cannot establish permanent per-point calibration adjustments. Longer-term one-point reliability should be scored from the complete rolling-origin per-game out-of-fold predictions, not the aggregate model file.

### Early vs later
- Nov 4–9: **28/39 = 71.8%**
- Nov 10–15: **22/45 = 48.9%**
- Nov 3 launch baseline: **5/11 = 45.5%**

The drop coincides with increasing source-feature lag. Coincidence is not causal evidence, and these windows are too small to infer a trend with precision.

### Highest-confidence misses
| Date | Game | Selected pick | Model pick probability | Actual winner |
|---|---|---|---:|---|
| 2026-11-15 | VAN @ VGK | VGK | 77.7% | VAN |
| 2026-11-03 | CAR @ NSH | CAR | 72.2% | NSH |
| 2026-11-05 | NSH @ COL | COL | 70.0% | NSH |
| 2026-11-14 | SJS @ UTA | UTA | 68.2% | SJS |
| 2026-11-12 | WPG @ DAL | DAL | 67.2% | WPG |
| 2026-11-13 | NSH @ VGK | VGK | 65.9% | NSH |
| 2026-11-11 | NSH @ ANA | ANA | 65.1% | NSH |
| 2026-11-03 | WPG @ MTL | MTL | 63.9% | WPG |
| 2026-11-05 | ANA @ EDM | EDM | 62.1% | ANA |
| 2026-11-14 | DET @ TBL | TBL | 62.0% | DET |
| 2026-11-14 | PHI @ CBJ | CBJ | 61.5% | PHI |
| 2026-11-09 | NYI @ SJS | NYI | 59.8% | SJS |

## Operational findings
- The legacy freeze ledger recorded only **18/95** games; the missing 77 can be retrospectively **audited as historic published morning picks**, but they must not be falsely labeled as near-puck immutable freeze records.
- The scorer workflow had repeated failures rejecting after-puck projections, even though the validation rule is correct. Fix the staggered-start publishing mechanism without weakening pregame integrity.
- Verify starters change the actual probabilities when the confirmed goalie differs. Two Nov 15 immutable records had CONFIRMED current context but PROJECTED adjustment status.
- Add immutable per-game morning snapshots and near-start freeze coverage monitoring, with permanent per-game provenance and missing-game alerts.
- Address stale source data and build a separate independently promoted xG forecast layer and its scoring ledger.

**Audit creation does not alter any model weights, goalie model, result, production predictions, or frozen forward records.** This is a post-hoc *evaluation* of independently archived pregame predictions.
