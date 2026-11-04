#!/usr/bin/env python3
"""Export game-level Hockey Lab chronological out-of-sample forecasts.

This is a research replay of the production Game V1 selection/fitting policy.
For each target source-season, model choices use the prior season for
validation, and only earlier seasons for fitting. It does not read market data
and does not modify production projections or the forward ledger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "production"))
sys.path.insert(0, str(ROOT / "research"))

import hockey_lab_game_v1 as production
import hwin_team_strength_004 as ts


def _summary(y: pd.Series, p: np.ndarray) -> dict:
    actual = pd.to_numeric(y, errors="coerce").astype(int).to_numpy()
    prob = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "games": int(len(actual)),
        "accuracy": float(accuracy_score(actual, prob >= 0.5)),
        "brier": float(brier_score_loss(actual, prob)),
        "log_loss": float(log_loss(actual, prob, labels=[0, 1])),
        "mean_home_probability": float(prob.mean()),
        "home_win_rate": float(actual.mean()),
    }


def export(pbp_path: Path, games_path: Path, out_path: Path,
           first_season: int = 2022, last_season: int = 2026) -> dict:
    pbp = pd.read_parquet(pbp_path)
    spine = pd.read_parquet(games_path)
    spine = spine[
        spine.result_available.fillna(False)
        & spine.game_type.astype("string").isin(["R", "P"])
    ].copy()
    spine["game_date"] = pd.to_datetime(spine.game_date, errors="coerce")
    spine["source_season"] = pd.to_numeric(spine.source_season, errors="coerce")
    spine = spine.dropna(subset=["game_date", "source_season"])

    base_games = ts.game_event_table(pbp)
    features = ts.add_flat_and_ewm_features(base_games)
    features = features.merge(production.build_schedule_features(spine), on="game_id", how="left")

    elo = {}
    for k, ha, retain in ts.ELO_GRID:
        key = f"k{k}_h{ha}_r{retain}"
        elo[key] = production.elo_with_future(spine, k, ha, retain)

    all_features = features.copy()
    oa_tags = []
    for alpha, retain in ts.OA_GRID:
        z, tag = ts.opponent_adjusted_features(base_games, alpha, retain)
        all_features = all_features.merge(z, on="game_id", how="left")
        oa_tags.append(tag)

    rows = []
    fold_metrics = {}
    for target in range(first_season, last_season + 1):
        val_season = target - 1
        if not (all_features.source_season.eq(target).any()
                and all_features.source_season.eq(val_season).any()):
            continue

        elo_choice = None
        for key, values in elo.items():
            val_elo = all_features.loc[
                all_features.source_season.eq(val_season), ["game_id", "home_win"]
            ].merge(
                values.loc[values.source_season.eq(val_season), ["game_id", "elo_p"]],
                on="game_id", how="inner",
            ).dropna(subset=["home_win", "elo_p"])
            if len(val_elo) < 100:
                continue
            score = log_loss(
                val_elo.home_win.astype(int),
                np.clip(val_elo.elo_p, 1e-6, 1 - 1e-6), labels=[0, 1],
            )
            if elo_choice is None or score < elo_choice[0]:
                elo_choice = (score, key)
        if elo_choice is None:
            raise RuntimeError(f"No eligible Elo validation candidate for source_season={target}")
        selected_elo = elo_choice[1]

        x = all_features.merge(elo[selected_elo][["game_id", "elo_p"]], on="game_id", how="left")
        x["elo_logit"] = np.log(
            np.clip(x.elo_p, 1e-6, 1 - 1e-6)
            / np.clip(1 - x.elo_p, 1e-6, 1 - 1e-6)
        )
        train = x[x.source_season.le(target - 2) & x.home_win.notna()].copy()
        val = x[x.source_season.eq(val_season) & x.home_win.notna()].copy()
        test = x[x.source_season.eq(target) & x.home_win.notna()].copy()
        if min(len(train), len(val), len(test)) == 0:
            continue

        selected = None
        for tag in oa_tags:
            cols = [
                "elo_logit", "d_sog_share_r20", "d_xg_share_r20", "d_goals_r20",
                "d_rest_days", f"oa_sog_diff_{tag}", f"oa_xg_diff_{tag}",
                "h_sched_b2b", "a_sched_b2b", "d_sched_b2b",
            ]
            c_value, val_loss = production.tune(train, val, cols)
            if selected is None or val_loss < selected[0]:
                selected = (val_loss, tag, c_value, cols)
        assert selected is not None
        _, selected_oa, c_value, model_features = selected

        prior_data = pd.concat([train, val], ignore_index=True)
        model = production.fit_pipeline(prior_data, model_features, c_value)
        home_p = model.predict_proba(test[model_features])[:, 1]
        fold_metrics[str(target)] = _summary(test.home_win, home_p)

        start_col = next((c for c in [
            "scheduled_start_utc", "scheduled_start", "start_time_utc",
            "game_start_utc", "game_time",
        ] if c in spine.columns), None)
        if start_col is None:
            raise ValueError("Canonical games file needs game_time or scheduled_start_utc")
        game_meta = spine[[
            "game_id", "game_date", "source_season", "home_team_abbr", "away_team_abbr", start_col,
        ]].drop_duplicates("game_id").rename(columns={
            "home_team_abbr": "home_abbr", "away_team_abbr": "away_abbr",
            start_col: "scheduled_start_utc",
        })
        game_meta["scheduled_start_utc"] = pd.to_datetime(
            game_meta.scheduled_start_utc, utc=True, errors="coerce"
        )
        cutoff_game_date = pd.Timestamp(prior_data.game_date.max()).date().isoformat()
        scored = test[[
            "game_id", "source_season", "h_sched_rest_days", "a_sched_rest_days",
            "h_sched_b2b", "a_sched_b2b", "h_sched_3in4", "a_sched_3in4",
            "h_sched_4in6", "a_sched_4in6", "h_sched_5in8", "a_sched_5in8",
        ]].copy()
        scored["home_win_probability"] = home_p
        scored["away_win_probability"] = 1.0 - home_p
        scored["model_id"] = production.MODEL_ID
        scored["model_version"] = production.MODEL_VERSION
        scored["feature_version"] = production.FEATURE_VERSION
        scored["forecast_origin"] = "CHRONOLOGICAL_OOS"
        scored["training_max_source_season"] = val_season
        scored["training_cutoff_game_date"] = cutoff_game_date
        scored["validation_source_season"] = val_season
        scored["selected_elo_candidate"] = selected_elo
        scored["selected_opponent_adjustment"] = selected_oa
        scored["selected_logistic_c"] = float(c_value)
        scored["oos_replay_notice"] = "Chronological research replay; not a forecast actually frozen before this game."
        def rest_label(value):
            if pd.isna(value):
                return None
            gap = int(round(float(value)))
            if gap <= 1:
                return "B2B"
            days_off = max(0, gap - 1)
            unit = "day" if days_off == 1 else "days"
            return f"{days_off} {unit} rest"

        def rest_severity(value):
            if pd.isna(value):
                return "UNKNOWN"
            gap = int(round(float(value)))
            return "B2B" if gap <= 1 else ("SHORT" if gap == 2 else "RESTED")

        def load_flags(row, side):
            flags = []
            for suffix, label in [("3in4", "3-IN-4"), ("4in6", "4-IN-6"), ("5in8", "5-IN-8")]:
                value = row.get(f"{side}_sched_{suffix}")
                if pd.notna(value) and float(value) >= 1:
                    flags.append(label)
            return flags

        rest_records = []
        for row in scored.itertuples(index=False):
            away_gap = getattr(row, "a_sched_rest_days")
            home_gap = getattr(row, "h_sched_rest_days")
            away_flags = load_flags(row._asdict(), "a")
            home_flags = load_flags(row._asdict(), "h")
            rest_records.append({
                "away": rest_label(away_gap), "home": rest_label(home_gap),
                "away_severity": rest_severity(away_gap), "home_severity": rest_severity(home_gap),
                "away_flags": away_flags, "home_flags": home_flags,
            })
        scored["rest_state"] = rest_records
        scored["goalie_state"] = "UNKNOWN"
        scored = scored.merge(game_meta, on=["game_id", "source_season"], how="left", validate="one_to_one")
        rows.append(scored)

    if not rows:
        raise RuntimeError("No chronological OOS forecast rows were produced")
    output = pd.concat(rows, ignore_index=True)
    output["game_id"] = pd.to_numeric(output.game_id, errors="raise").astype("int64")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(out_path, index=False)

    # Aggregate the replay rows against their evaluation labels only in this
    # sidecar. Labels are not copied into the forecast artifact.
    all_targets = all_features[["game_id", "source_season", "home_win"]].copy()
    scored_metrics = output[["game_id", "source_season", "home_win_probability"]].merge(
        all_targets, on=["game_id", "source_season"], how="left", validate="one_to_one"
    )
    summary = {
        "model_id": production.MODEL_ID,
        "model_version": production.MODEL_VERSION,
        "feature_version": production.FEATURE_VERSION,
        "forecast_origin": "CHRONOLOGICAL_OOS",
        "historical_replay_not_frozen_forecasts": True,
        "source_season_range": [first_season, last_season],
        "nhl_season_range": [f"{first_season - 1}-{str(first_season)[-2:]}",
                              f"{last_season - 1}-{str(last_season)[-2:] if last_season != 2000 else '00'}"],
        "games": int(len(output)),
        "overall": _summary(scored_metrics.home_win, scored_metrics.home_win_probability),
        "by_source_season": fold_metrics,
        "stored_production_performance": {
            "games": production.PERFORMANCE["games_scored"],
            "accuracy": production.PERFORMANCE["accuracy"],
            "brier": production.PERFORMANCE["brier"],
            "log_loss": production.PERFORMANCE["log_loss"],
        },
        "replay_minus_stored_performance": {
            "accuracy": float(_summary(scored_metrics.home_win, scored_metrics.home_win_probability)["accuracy"] - production.PERFORMANCE["accuracy"]),
            "brier": float(_summary(scored_metrics.home_win, scored_metrics.home_win_probability)["brier"] - production.PERFORMANCE["brier"]),
            "log_loss": float(_summary(scored_metrics.home_win, scored_metrics.home_win_probability)["log_loss"] - production.PERFORMANCE["log_loss"]),
        },
        "forecast_file": str(out_path),
    }
    sidecar = out_path.with_name("chronological-oos-forecast-summary.json")
    sidecar.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbp", required=True, help="Lagged NHL play-by-play feature parquet")
    parser.add_argument("--games", required=True, help="Canonical NHL games parquet")
    parser.add_argument("--out", default=str(ROOT / "artifacts/market/staging/chronological-oos-forecasts.parquet"))
    parser.add_argument("--start-season", type=int, default=2022,
                        help="First source_season code (2022 = 2021-22 NHL season)")
    parser.add_argument("--end-season", type=int, default=2026,
                        help="Last source_season code (2026 = 2025-26 NHL season)")
    args = parser.parse_args()
    summary = export(Path(args.pbp), Path(args.games), Path(args.out),
                     args.start_season, args.end_season)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
