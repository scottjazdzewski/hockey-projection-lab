"""No-goalie chronological schedule increment test.

This experiment compares the HWIN Team Strength 004 pregame feature set with
schedule-derived additions. It deliberately excludes all goalie fields because
the existing historical goalie feed identifies the actual starter after the
game. Every model is tuned on the previous season and evaluated on the next
season, with predictions saved per game before any pooled scoring.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "research")
import hwin_team_strength_004 as ts

MODEL_VERSION = "HWIN-SCHEDULE-007-NOGOALIE-RESEARCH"
FEATURE_VERSION = "pregame-team-strength-004-plus-schedule-001"
SOURCE_VERSION = "sportsdataverse-pbp-features-20261102+canonical-spine-v3-20261102"
RETRIEVAL_DATE = "2026-11-02"
TARGET_SEASONS = [2022, 2023, 2024, 2025, 2026]
C_GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]


def build_schedule_features(spine: pd.DataFrame) -> pd.DataFrame:
    """Create strictly lagged schedule features from games on earlier dates."""
    s = spine[["game_id", "game_date", "source_season", "home_team_abbr", "away_team_abbr"]].copy()
    s["game_date"] = pd.to_datetime(s.game_date, errors="coerce").dt.normalize()
    h = s.rename(columns={"home_team_abbr": "team", "away_team_abbr": "opp"}).assign(home_away="home")
    a = s.rename(columns={"away_team_abbr": "team", "home_team_abbr": "opp"}).assign(home_away="away")
    long = pd.concat([h, a], ignore_index=True)
    long = long.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)

    rows = []
    for team, group in long.groupby("team", sort=False):
        history_dates = []
        history_locations = []
        for r in group.itertuples(index=False):
            d = r.game_date
            # Exclude same-date games: the source has date-only timestamps, so
            # their before/after-puck-drop order cannot be established safely.
            prior_dates = [x for x in history_dates if pd.notna(d) and x < d]
            prev_date = prior_dates[-1] if prior_dates else pd.NaT
            rest = (d - prev_date).days if pd.notna(prev_date) and pd.notna(d) else np.nan
            prior3 = sum(1 for x in prior_dates if (d - x).days <= 3)
            prior5 = sum(1 for x in prior_dates if (d - x).days <= 5)
            prior7 = sum(1 for x in prior_dates if (d - x).days <= 7)

            # Location history is likewise restricted to earlier dates. When
            # a team has multiple games on one prior date, date-only data cannot
            # establish their sequence, so location streak features are null.
            prior_unique_dates = sorted(set(prior_dates), reverse=True)
            indices_by_date = {
                prior_date: [i for i, x in enumerate(history_dates) if x == prior_date]
                for prior_date in prior_unique_dates
            }
            latest_group = indices_by_date.get(prior_unique_dates[0], []) if prior_unique_dates else []
            prev_away = (
                float(history_locations[latest_group[0]] == "away")
                if len(latest_group) == 1 else np.nan
            )
            latest_two = [indices_by_date[x] for x in prior_unique_dates[:2]]
            prev2_away = (
                float(
                    len(latest_two) == 2
                    and len(latest_two[0]) == 1
                    and len(latest_two[1]) == 1
                    and history_locations[latest_two[0][0]] == "away"
                    and history_locations[latest_two[1][0]] == "away"
                )
                if len(latest_two) == 2 and all(len(x) == 1 for x in latest_two)
                else np.nan
            )
            current_home = r.home_away == "home"
            trip_no = 0.0 if current_home else 1.0
            if not current_home:
                if prior_unique_dates and len(latest_group) != 1:
                    trip_no = np.nan
                else:
                    for prior_date in prior_unique_dates:
                        indices = indices_by_date[prior_date]
                        if len(indices) != 1:
                            trip_no = np.nan
                            break
                        if history_locations[indices[0]] != "away":
                            break
                        trip_no += 1

            rows.append({
                "game_id": r.game_id,
                "team": team,
                "home_away": r.home_away,
                "sched_rest_days": rest,
                "sched_b2b": float(pd.notna(rest) and rest <= 1),
                "sched_3in4": float(prior3 >= 2),
                "sched_4in6": float(prior5 >= 3),
                "sched_5in8": float(prior7 >= 4),
                "sched_prior3_games": float(prior3),
                "sched_prior5_games": float(prior5),
                "sched_prior7_games": float(prior7),
                "sched_prev_away": prev_away,
                "sched_home_after_road": float(current_home and prev_away) if pd.notna(prev_away) else np.nan,
                "sched_home_after_2road": float(current_home and prev2_away) if pd.notna(prev2_away) else np.nan,
                "sched_away_trip_no": float(trip_no) if pd.notna(trip_no) else np.nan,
            })
            history_dates.append(d)
            history_locations.append(r.home_away)

    z = pd.DataFrame(rows)
    sched_cols = [c for c in z.columns if c.startswith("sched_")]
    hh = z[z.home_away.eq("home")][["game_id"] + sched_cols].rename(
        columns={c: f"h_{c}" for c in sched_cols}
    )
    aa = z[z.home_away.eq("away")][["game_id"] + sched_cols].rename(
        columns={c: f"a_{c}" for c in sched_cols}
    )
    out = hh.merge(aa, on="game_id", how="inner", validate="one_to_one")
    for c in sched_cols:
        out[f"d_{c}"] = pd.to_numeric(out[f"h_{c}"], errors="coerce") - pd.to_numeric(
            out[f"a_{c}"], errors="coerce"
        )
    return out


def fit_pipeline(train: pd.DataFrame, features: list[str], C: float) -> Pipeline:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=C, max_iter=2000)),
    ])
    model.fit(train[features], train.home_win.astype(int))
    return model


def tune_C(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    best = None
    for C in C_GRID:
        model = fit_pipeline(train, features, C)
        p = np.clip(model.predict_proba(val[features])[:, 1], 1e-6, 1 - 1e-6)
        score = log_loss(val.home_win.astype(int), p, labels=[0, 1])
        if best is None or score < best[0]:
            best = (score, C)
    return float(best[1]), float(best[0])


def score_predictions(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "games": int(len(y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "mean_home_probability": float(np.mean(p)),
        "home_win_rate": float(np.mean(y)),
    }


def calibration_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    # The large C approximates unpenalized logistic recalibration. This is a
    # diagnostic only; these coefficients are not applied to predictions.
    cal = LogisticRegression(C=1e8, max_iter=5000).fit(logit_p, y)
    return {
        "calibration_intercept": float(cal.intercept_[0]),
        "calibration_slope": float(cal.coef_[0, 0]),
    }


def reliability_table(predictions: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for model_name, group in predictions.groupby("model_variant", sort=True):
        g = group.copy()
        g["probability_bin"] = pd.qcut(g.p_home, q=10, labels=False, duplicates="drop") + 1
        for bin_id, b in g.groupby("probability_bin", dropna=True):
            outputs.append({
                "model_variant": model_name,
                "probability_bin": int(bin_id),
                "games": int(len(b)),
                "mean_home_probability": float(b.p_home.mean()),
                "home_win_rate": float(b.home_win.mean()),
                "min_probability": float(b.p_home.min()),
                "max_probability": float(b.p_home.max()),
            })
    return pd.DataFrame(outputs)


def main(pbp_path: str, spine_path: str, outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pbp = pd.read_parquet(pbp_path)
    spine = pd.read_parquet(spine_path)
    spine = spine[
        spine.result_available.fillna(False)
        & spine.game_type.astype("string").isin(["R", "P"])
    ].copy()
    if spine.game_id.duplicated().any():
        raise ValueError("Canonical result spine contains duplicate game_id values")
    spine["game_date"] = pd.to_datetime(spine.game_date, errors="coerce").dt.normalize()
    spine["source_season"] = pd.to_numeric(spine.source_season, errors="coerce")

    games = ts.add_flat_and_ewm_features(ts.game_event_table(pbp))
    metadata_cols = ["game_id", "game_type"]
    for source_name in ["home_team_id", "away_team_id"]:
        if source_name in spine.columns:
            metadata_cols.append(source_name)
    metadata = spine[[c for c in metadata_cols if c in spine.columns]]
    games = games.merge(metadata, on="game_id", how="inner", validate="one_to_one")
    schedule = build_schedule_features(spine)
    games = games.merge(schedule, on="game_id", how="left", validate="one_to_one")

    # Candidate Elo streams and opponent-adjusted ratings are generated online:
    # each row is recorded before its result updates the corresponding rating.
    elo = {}
    for k, ha, retain in ts.ELO_GRID:
        key = f"k{k}_h{ha}_r{retain}"
        elo[key] = ts.elo_all(spine, k, ha, retain)
    oa = {}
    for alpha, retain in ts.OA_GRID:
        z, tag = ts.opponent_adjusted_features(games, alpha, retain)
        oa[tag] = z
        games = games.merge(z, on="game_id", how="left", validate="one_to_one")

    schedule_groups = {
        "plus_b2b": ["h_sched_b2b", "a_sched_b2b", "d_sched_b2b"],
        "plus_density": [
            "d_sched_3in4", "d_sched_4in6", "d_sched_5in8",
            "d_sched_prior3_games", "d_sched_prior5_games", "d_sched_prior7_games",
        ],
        "plus_location_sequence": [
            "h_sched_home_after_road", "h_sched_home_after_2road",
            "a_sched_prev_away", "a_sched_away_trip_no",
        ],
    }
    schedule_groups["plus_schedule_all"] = (
        schedule_groups["plus_b2b"]
        + schedule_groups["plus_density"]
        + schedule_groups["plus_location_sequence"]
    )

    prediction_rows = []
    fold_rows = []
    for target in TARGET_SEASONS:
        val_season = target - 1

        # Select Elo hyperparameters using the immediately prior season only.
        elo_choice = None
        for key, pred in elo.items():
            valid = games.loc[games.source_season.eq(val_season), ["game_id", "home_win"]].merge(
                pred.loc[pred.source_season.eq(val_season), ["game_id", "elo_p"]],
                on="game_id", how="inner", validate="one_to_one",
            )
            if len(valid) < 100:
                continue
            score = log_loss(valid.home_win.astype(int), np.clip(valid.elo_p, 1e-6, 1 - 1e-6), labels=[0, 1])
            if elo_choice is None or score < elo_choice[0]:
                elo_choice = (score, key)
        if elo_choice is None:
            raise RuntimeError(f"No usable Elo validation data for target season {target}")
        elo_key = elo_choice[1]
        frame = games.merge(
            elo[elo_key][["game_id", "elo_p"]], on="game_id", how="left", validate="one_to_one"
        )
        frame["elo_logit"] = np.log(
            np.clip(frame.elo_p, 1e-6, 1 - 1e-6) / np.clip(1 - frame.elo_p, 1e-6, 1 - 1e-6)
        )

        train = frame.loc[frame.source_season.le(target - 2)].copy()
        val = frame.loc[frame.source_season.eq(val_season)].copy()
        test = frame.loc[frame.source_season.eq(target)].copy()
        if len(test) == 0 or len(val) < 100:
            raise RuntimeError(f"Insufficient chronological data for target season {target}")

        # Match Team Strength 004's validation-only OA selection procedure.
        oa_choice = None
        for alpha, retain in ts.OA_GRID:
            tag = f"a{alpha:.2f}_r{retain:.2f}".replace(".", "p")
            features = [
                "elo_logit", f"oa_sog_diff_{tag}", f"oa_xg_diff_{tag}",
                "d_goals_r20", "d_rest_days",
            ]
            for C in C_GRID:
                model = fit_pipeline(train, features, C)
                p = np.clip(model.predict_proba(val[features])[:, 1], 1e-6, 1 - 1e-6)
                score = log_loss(val.home_win.astype(int), p, labels=[0, 1])
                if oa_choice is None or score < oa_choice[0]:
                    oa_choice = (score, tag)
        if oa_choice is None:
            raise RuntimeError(f"No usable OA validation data for target season {target}")
        oa_tag = oa_choice[1]

        base_features = [
            "elo_logit", "d_sog_share_r20", "d_xg_share_r20", "d_goals_r20", "d_rest_days",
            f"oa_sog_diff_{oa_tag}", f"oa_xg_diff_{oa_tag}",
        ]
        specifications = {"team_strength_004": base_features}
        for name, additions in schedule_groups.items():
            specifications[name] = base_features + additions

        for model_name, features in specifications.items():
            C, validation_log_loss = tune_C(train, val, features)
            fit = pd.concat([train, val], ignore_index=True)
            model = fit_pipeline(fit, features, C)
            p_home = model.predict_proba(test[features])[:, 1]
            fold_metric = score_predictions(test.home_win, p_home)
            fold_rows.append({
                "model_variant": model_name,
                "target_season": target,
                "validation_season": val_season,
                "training_through_season": target - 2,
                "elo_choice": elo_key,
                "opponent_adjusted_choice": oa_tag,
                "regularization_C": C,
                "validation_log_loss": validation_log_loss,
                **fold_metric,
            })
            for game, probability in zip(test.itertuples(index=False), p_home):
                record = {
                    "game_id": game.game_id,
                    "game_date": pd.Timestamp(game.game_date).date().isoformat(),
                    "source_season": int(game.source_season),
                    "home_team_abbr": game.home_team,
                    "away_team_abbr": game.away_team,
                    "game_type": getattr(game, "game_type", None),
                    "home_win": int(game.home_win),
                    "p_home": float(probability),
                    "p_away": float(1 - probability),
                    "model_variant": model_name,
                    "model_version": MODEL_VERSION,
                    "feature_version": FEATURE_VERSION,
                    "source_version": SOURCE_VERSION,
                    "retrieval_date": RETRIEVAL_DATE,
                    "feature_cutoff_date": pd.Timestamp(game.game_date).date().isoformat(),
                    "data_quality_flag": "verified_winner_target;strict_prior_date_schedule",
                    "target_season": target,
                    "training_through_season": target - 2,
                    "validation_season": val_season,
                    "elo_choice": elo_key,
                    "opponent_adjusted_choice": oa_tag,
                    "regularization_C": C,
                }
                for id_col in ["home_team_id", "away_team_id"]:
                    value = getattr(game, id_col, None)
                    if value is not None and not pd.isna(value):
                        record[id_col] = value
                prediction_rows.append(record)

    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    probability_totals = predictions.p_home + predictions.p_away
    if not np.allclose(probability_totals.to_numpy(), 1.0, atol=1e-12):
        raise ValueError("Home and away probabilities do not sum to one")
    predictions.to_parquet(out / "oof_predictions.parquet", index=False)
    folds.to_parquet(out / "fold_metrics.parquet", index=False)

    summary = {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "source_version": SOURCE_VERSION,
        "retrieval_date": RETRIEVAL_DATE,
        "target_seasons": TARGET_SEASONS,
        "validation_design": "rolling_origin; train<=target-2; tune on target-1; test target",
        "total_test_games_per_variant": int(predictions.game_id.nunique()),
        "models": {},
    }
    for model_name, group in predictions.groupby("model_variant", sort=True):
        scores = score_predictions(group.home_win, group.p_home)
        summary["models"][model_name] = {
            **scores,
            **calibration_metrics(group.home_win, group.p_home),
            "season_brier": {
                str(int(season)): float(brier_score_loss(g.home_win, g.p_home))
                for season, g in group.groupby("target_season")
            },
        }
    baseline_brier = summary["models"]["team_strength_004"]["brier"]
    for model_name, metrics in summary["models"].items():
        metrics["delta_brier_vs_team_strength_004"] = metrics["brier"] - baseline_brier
        if model_name != "team_strength_004":
            base_by_season = summary["models"]["team_strength_004"]["season_brier"]
            metrics["seasons_better_than_team_strength_004"] = sum(
                metrics["season_brier"][season] < base_by_season[season]
                for season in base_by_season
            )
    reliability_table(predictions).to_parquet(out / "reliability_bins.parquet", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pbp", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.pbp, args.spine, args.out)
