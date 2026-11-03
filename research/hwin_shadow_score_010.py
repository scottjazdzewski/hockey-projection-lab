"""Create frozen research-only pregame probabilities for the current slate.

The model is the no-goalie Team Strength 004 + B2B candidate. Hyperparameters
are selected on completed source season 2026, the model is refit through 2026,
and only the current slate is scored. Historical PBP gaps are preserved as
missing observations; they are never filled from markets or guessed outcomes.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import hwin_schedule_007 as sched
import hwin_score_state_009 as score_state
import hwin_team_strength_004 as ts


MODEL_VERSION = "HWIN-SHADOW-010-SCHEDULE007-B2B-RESEARCH"
FEATURE_VERSION = "team-strength-004-plus-b2b-shadow-2027-001"
SOURCE_VERSION = "sportsdataverse-pbp-20261102+canonical-spine-v3-20261102"
TRAIN_THROUGH_SEASON = 2026
VALIDATION_SEASON = 2026


def parse_utc(value: str) -> pd.Timestamp:
    result = pd.to_datetime(value, utc=True, errors="raise")
    if pd.isna(result):
        raise ValueError(f"Invalid UTC timestamp: {value}")
    return result


def verified_spine(spine: pd.DataFrame, latest_pbp_date: pd.Timestamp) -> pd.DataFrame:
    d = spine.loc[
        spine.result_available.fillna(False)
        & spine.game_type.astype("string").isin(["R", "P"])
    ].copy()
    d["game_date"] = pd.to_datetime(d.game_date, errors="coerce")
    d["game_time"] = pd.to_datetime(d.game_time, utc=True, errors="coerce")
    d["source_season"] = pd.to_numeric(d.source_season, errors="coerce")
    d["home_win"] = pd.to_numeric(d.home_win, errors="coerce")
    d = d.loc[d.game_date.dt.normalize().le(latest_pbp_date.normalize())].copy()
    if d.game_id.duplicated().any() or d.home_win.isna().any():
        raise ValueError("Verified result spine has duplicate games or missing winner labels")
    return d


def append_schedule_stubs(
    event_games: pd.DataFrame,
    spine: pd.DataFrame,
    target_date: pd.Timestamp,
    latest_pbp_date: pd.Timestamp,
) -> pd.DataFrame:
    """Retain scheduled dates after the PBP cutoff so rest remains correct."""
    missing = spine.loc[
        pd.to_numeric(spine.source_season, errors="coerce").eq(2027)
        & pd.to_datetime(spine.game_date, errors="coerce").dt.normalize().gt(latest_pbp_date.normalize())
        & pd.to_datetime(spine.game_date, errors="coerce").dt.normalize().le(target_date.normalize())
    ].copy()
    missing = missing.loc[~missing.game_id.isin(event_games.game_id)].copy()
    if missing.empty:
        return event_games
    stubs = pd.DataFrame({
        "game_id": missing.game_id,
        "source_season": pd.to_numeric(missing.source_season, errors="coerce"),
        "game_date": pd.to_datetime(missing.game_date, errors="coerce"),
        "home_team": missing.home_team_abbr.astype(str),
        "away_team": missing.away_team_abbr.astype(str),
        "home_win": np.nan,
        "h_sog": np.nan, "a_sog": np.nan,
        "h_xg": np.nan, "a_xg": np.nan,
        "h_sog_share": np.nan, "a_sog_share": np.nan,
        "h_xg_share": np.nan, "a_xg_share": np.nan,
        "h_goals": np.nan, "a_goals": np.nan,
    })
    return pd.concat([event_games, stubs], ignore_index=True, sort=False)


def current_elo_probabilities(
    result_spine: pd.DataFrame,
    targets: pd.DataFrame,
    k: int,
    home_advantage: int,
    retention: float,
) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    previous_season = None
    ordered = result_spine.sort_values(["game_date", "game_id"])
    for r in ordered.itertuples(index=False):
        season = int(r.source_season)
        if previous_season is None:
            previous_season = season
        elif season != previous_season:
            ratings = {team: 1500.0 + retention * (rating - 1500.0)
                       for team, rating in ratings.items()}
            previous_season = season
        home, away = str(r.home_team_abbr), str(r.away_team_abbr)
        p_home = ts.elo_prob(ratings.get(home, 1500.0), ratings.get(away, 1500.0), home_advantage)
        delta = k * (float(r.home_win) - p_home)
        ratings[home] = ratings.get(home, 1500.0) + delta
        ratings[away] = ratings.get(away, 1500.0) - delta

    season = int(targets.source_season.iloc[0])
    if previous_season is not None and season != previous_season:
        ratings = {team: 1500.0 + retention * (rating - 1500.0)
                   for team, rating in ratings.items()}
    rows = []
    for r in targets.itertuples(index=False):
        home, away = str(r.home_team_abbr), str(r.away_team_abbr)
        rows.append({
            "game_id": r.game_id,
            "elo_p": ts.elo_prob(ratings.get(home, 1500.0), ratings.get(away, 1500.0), home_advantage),
        })
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args) -> None:
    as_of = parse_utc(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    data_cutoff = parse_utc(args.data_cutoff)
    if data_cutoff > as_of:
        raise ValueError("Input data cutoff cannot be after prediction timestamp")

    pbp_path, spine_path, out = Path(args.pbp), Path(args.spine), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen shadow output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    pbp = pd.read_parquet(pbp_path)
    spine_all = pd.read_parquet(spine_path)
    spine_all["game_date"] = pd.to_datetime(spine_all.game_date, errors="coerce")
    spine_all["game_time"] = pd.to_datetime(spine_all.game_time, utc=True, errors="coerce")
    spine_all["source_season"] = pd.to_numeric(spine_all.source_season, errors="coerce")
    if spine_all.game_id.duplicated().any():
        raise ValueError("Canonical game spine contains duplicate IDs")

    pbp["game_date"] = pd.to_datetime(pbp.game_date, errors="coerce")
    latest_pbp_date = pbp.game_date.max().normalize()
    if pd.isna(latest_pbp_date):
        raise ValueError("PBP source has no valid game date")
    score_state.check_same_day_event_order(
        pbp,
        spine_all.loc[spine_all.game_type.astype("string").isin(["R", "P"])].copy(),
    )

    local_date = as_of.tz_convert(ZoneInfo("America/New_York")).date()
    day_spine = spine_all.loc[
        spine_all.source_season.eq(2027)
        & spine_all.game_type.astype("string").eq("R")
        & spine_all.game_date.dt.date.eq(local_date)
        & spine_all.game_time.gt(as_of)
        & ~spine_all.result_available.fillna(False)
    ].copy()
    if day_spine.empty:
        raise RuntimeError(f"No unstarted canonical-spine games found for local slate date {local_date}")
    if pd.concat([
        day_spine.home_team_abbr.rename("team"),
        day_spine.away_team_abbr.rename("team"),
    ]).duplicated().any():
        raise ValueError("Slate contains a same-day team doubleheader; current rating snapshot is ambiguous")

    result_spine = verified_spine(spine_all, latest_pbp_date)
    event_games = ts.game_event_table(pbp)
    labels = result_spine[["game_id", "home_win"]].copy()
    event_games = event_games.merge(labels, on="game_id", how="inner", suffixes=("_pbp", ""), validate="one_to_one")
    mismatch = event_games.home_win_pbp.astype(int).ne(event_games.home_win.astype(int))
    if mismatch.any():
        raise ValueError(f"PBP winner disagrees with canonical spine in {int(mismatch.sum())} games")
    event_games = event_games.drop(columns=["home_win_pbp"])

    max_target_date = day_spine.game_date.max().normalize()
    event_games = append_schedule_stubs(event_games, spine_all, max_target_date, latest_pbp_date)
    feature_games = ts.add_flat_and_ewm_features(event_games)
    schedule_spine = spine_all.loc[spine_all.game_date.dt.normalize().le(max_target_date)].copy()
    schedule_features = sched.build_schedule_features(schedule_spine)
    feature_games = feature_games.merge(schedule_features, on="game_id", how="left", validate="one_to_one")

    train_mask = feature_games.source_season.le(2025) & feature_games.home_win.notna()
    val_mask = feature_games.source_season.eq(VALIDATION_SEASON) & feature_games.home_win.notna()
    common = ["d_sog_share_r20", "d_xg_share_r20", "d_goals_r20", "d_rest_days"]
    b2b = ["h_sched_b2b", "a_sched_b2b", "d_sched_b2b"]
    val_labels = feature_games.loc[val_mask, ["game_id", "home_win"]].copy()
    if len(val_labels) < 100:
        raise RuntimeError("Insufficient 2026 validation games for live shadow model selection")

    best_elo = None
    elo_predictions = {}
    for k, home_advantage, retention in ts.ELO_GRID:
        key = (k, home_advantage, retention)
        ep = ts.elo_all(result_spine, k, home_advantage, retention)
        joined = val_labels.merge(ep, on="game_id", validate="one_to_one")
        score = log_loss(joined.home_win, np.clip(joined.elo_p, 1e-6, 1 - 1e-6), labels=[0, 1])
        elo_predictions[key] = ep
        if best_elo is None or score < best_elo[0]:
            best_elo = (score, key)
    elo_val_log_loss, elo_choice = best_elo
    k, home_advantage, elo_retention = elo_choice
    historical_elo = elo_predictions[elo_choice]
    target_elo = current_elo_probabilities(result_spine, day_spine, k, home_advantage, elo_retention)
    elo_all_rows = pd.concat([historical_elo, target_elo.assign(source_season=2027)], ignore_index=True)
    elo_all_rows = elo_all_rows.drop_duplicates("game_id", keep="last")

    selected = None
    for alpha, oa_retention in ts.OA_GRID:
        oa, oa_tag = ts.opponent_adjusted_features(event_games, alpha, oa_retention)
        frame = feature_games.merge(oa, on="game_id", how="left", validate="one_to_one")
        frame = frame.merge(elo_all_rows[["game_id", "elo_p"]], on="game_id", how="left", validate="one_to_one")
        frame["elo_logit"] = np.log(
            np.clip(frame.elo_p, 1e-6, 1 - 1e-6) / np.clip(1 - frame.elo_p, 1e-6, 1 - 1e-6)
        )
        oa_features = [f"oa_sog_diff_{oa_tag}", f"oa_xg_diff_{oa_tag}"]
        features = ["elo_logit"] + common + oa_features + b2b
        train = frame.loc[train_mask].copy()
        val = frame.loc[val_mask].copy()
        target = frame.loc[frame.game_id.isin(day_spine.game_id)].copy()
        for c in features:
            if c not in frame.columns:
                raise ValueError(f"Missing feature: {c}")
        for C in ts.C_GRID:
            model = sched.fit_pipeline(train, features, C)
            p_val = model.predict_proba(val[features])[:, 1]
            score = log_loss(val.home_win, np.clip(p_val, 1e-6, 1 - 1e-6), labels=[0, 1])
            choice = (score, alpha, oa_retention, C, oa_tag, features, model, target)
            if selected is None or score < selected[0]:
                selected = choice

    val_log_loss, alpha, oa_retention, C, oa_tag, features, _, target = selected
    oa, _ = ts.opponent_adjusted_features(event_games, alpha, oa_retention)
    frame = feature_games.merge(oa, on="game_id", how="left", validate="one_to_one")
    frame = frame.merge(elo_all_rows[["game_id", "elo_p"]], on="game_id", how="left", validate="one_to_one")
    frame["elo_logit"] = np.log(
        np.clip(frame.elo_p, 1e-6, 1 - 1e-6) / np.clip(1 - frame.elo_p, 1e-6, 1 - 1e-6)
    )
    train = frame.loc[train_mask].copy()
    val = frame.loc[val_mask].copy()
    target = frame.loc[frame.game_id.isin(day_spine.game_id)].copy()
    final_model = sched.fit_pipeline(pd.concat([train, val], ignore_index=True), features, C)
    p_home = final_model.predict_proba(target[features])[:, 1]
    if not np.isfinite(p_home).all():
        raise ValueError("Non-finite shadow probability")

    details = day_spine[["game_id", "game_date", "game_time", "home_team_abbr", "away_team_abbr"]].copy()
    details = details.merge(target[["game_id"]], on="game_id", validate="one_to_one")
    latest_pbp_date_s = latest_pbp_date.date().isoformat()
    rows = []
    for g, p in zip(details.itertuples(index=False), p_home):
        lag = (pd.Timestamp(g.game_date).normalize() - latest_pbp_date).days
        rows.append({
            "game_id": int(g.game_id),
            "game_date": pd.Timestamp(g.game_date).date().isoformat(),
            "game_start_timestamp": pd.Timestamp(g.game_time).isoformat(),
            "home_team": str(g.home_team_abbr),
            "away_team": str(g.away_team_abbr),
            "prediction_timestamp": as_of.isoformat(),
            "input_data_cutoff_timestamp": data_cutoff.isoformat(),
            "latest_pbp_game_date": latest_pbp_date_s,
            "input_lag_days": int(lag),
            "model_version": MODEL_VERSION,
            "model_status": "RESEARCH_CANDIDATE_SHADOW",
            "feature_version": FEATURE_VERSION,
            "source_version": SOURCE_VERSION,
            "training_through_season": TRAIN_THROUGH_SEASON,
            "parameter_selection_validation_season": VALIDATION_SEASON,
            "home_win_probability": float(p),
            "away_win_probability": float(1.0 - p),
            "expected_home_goals": None,
            "expected_away_goals": None,
            "regulation_home_win_probability": None,
            "overtime_probability": None,
            "goalie_state": "unavailable_in_snapshot",
            "lineup_state": "unavailable_in_snapshot",
            "market_input_used": False,
            "model_flags": [
                f"PBP_LATEST_GAME_DATE_{latest_pbp_date_s}",
                f"PBP_INPUT_LAG_DAYS_{lag}",
                "GOALIE_SNAPSHOT_UNAVAILABLE",
                "LINEUP_SNAPSHOT_UNAVAILABLE",
                "RESEARCH_SHADOW_ONLY_NOT_SITE_EXPORT",
            ],
            "hyperparameters": {
                "elo_k": int(k), "elo_home_advantage": int(home_advantage),
                "elo_offseason_retention": float(elo_retention),
                "opponent_adjustment_alpha": float(alpha),
                "opponent_adjustment_retention": float(oa_retention),
                "logistic_C": float(C),
            },
            "input_manifest_sha256": args.input_manifest_sha256,
        })
    pred = pd.DataFrame(rows).sort_values(["game_start_timestamp", "game_id"]).reset_index(drop=True)
    if not np.allclose(pred.home_win_probability + pred.away_win_probability, 1.0, atol=1e-12):
        raise ValueError("Home and away probabilities do not sum to one")
    if not (pd.to_datetime(pred.prediction_timestamp, utc=True) < pd.to_datetime(pred.game_start_timestamp, utc=True)).all():
        raise ValueError("Refusing to freeze a prediction at or after puck drop")

    prediction_path = out / "pregame_predictions.parquet"
    pred.to_parquet(prediction_path, index=False)
    summary = {
        "model_version": MODEL_VERSION,
        "model_status": "RESEARCH_CANDIDATE_SHADOW",
        "prediction_date_et": str(local_date),
        "prediction_timestamp": as_of.isoformat(),
        "input_data_cutoff_timestamp": data_cutoff.isoformat(),
        "latest_pbp_game_date": latest_pbp_date_s,
        "games_frozen": int(len(pred)),
        "historical_oof_baseline": {
            "model": "Team Strength 004 + B2B",
            "games": 6993,
            "brier": 0.23472630331662248,
            "log_loss": 0.6615400452176158,
            "accuracy": 0.5991705991705991,
            "status": "candidate; historical targets previously inspected",
        },
        "selection": {
            "train_through_season": 2025,
            "validation_season": VALIDATION_SEASON,
            "final_fit_through_season": TRAIN_THROUGH_SEASON,
            "validation_log_loss": float(val_log_loss),
            "elo_validation_log_loss": float(elo_val_log_loss),
            "elo_choice": {"k": int(k), "home_advantage": int(home_advantage), "retention": float(elo_retention)},
            "opponent_adjusted_choice": {"alpha": float(alpha), "retention": float(oa_retention), "tag": oa_tag},
            "logistic_C": float(C),
            "features": features,
        },
        "data_notes": [
            "Canonical spine schedule used; site today-games.json was not used because its published date lagged the current local date.",
            "Available PBP snapshot ends 2026-11-01; outcomes after that date are not included in team event form.",
            "Schedule stubs preserve rest-day features after the PBP cutoff; missing event performance is left missing.",
            "No odds, current-game results, actual starters, or postgame lineup fields enter the model.",
            "These are research-only prospective predictions; no website export or production promotion.",
        ],
        "outputs": {prediction_path.name: {"sha256": sha256(prediction_path), "bytes": prediction_path.stat().st_size}},
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "pyarrow"]},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    print(pred[["game_id", "away_team", "home_team", "home_win_probability", "away_win_probability", "input_lag_days"]].to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pbp", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--data-cutoff", required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    main(parser.parse_args())
