"""Calibrate individual-goalie skill across the full historical goalie pool.

The rating is a named-goalie estimate of goals saved above the league pool's
historical expected-goal baseline per 30 shots. It uses prior goalie starts
only, centers each historical performance against the league mean known before
that game date, and tests career/recent shrinkage with rolling-origin seasons.

This is a research diagnostic: actual target-game goalie identity is from the
postgame goalie box. It does not create public predictions or alter Game v1.0.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ID = "HOCKEY-LAB-GOALIE-POOL-CALIBRATION-021-RESEARCH"
TARGET_SEASONS = list(range(2018, 2027))
CAREER_K = [100, 500, 1000, 2500, 5000]
RECENT_K = [100, 500, 1000, 2500, 5000]
CONTEXTS = ["expanding", "roll365", "roll1095"]
MEMORIES = ["career", "r20", "r40"]
SHRINKAGE_SHOTS = [100, 1000, 5000]
AS_OF = pd.Timestamp("2026-11-02")  # includes completed games through Nov. 1


def normalize_game_id(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric.astype("Int64").astype("string")
    return values.astype("string")


def build_pool(goalie_path: str, spine_path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    goalie = pd.read_parquet(goalie_path)
    spine = pd.read_parquet(spine_path)
    goalie = goalie.loc[goalie.starter_bool.fillna(False).astype(bool)].copy()
    goalie = goalie[[
        "game_id", "goalie_id_key", "home_away", "starter_bool",
        "shots_against_pbp", "gsax", "player_name", "team_abbrev",
    ]].copy()
    meta = spine[["game_id", "game_date", "source_season", "game_type"]].copy()
    meta["game_id"] = normalize_game_id(meta.game_id)
    meta["game_date"] = pd.to_datetime(meta.game_date, errors="coerce").dt.normalize()
    goalie["game_id"] = normalize_game_id(goalie.game_id)
    goalie["goalie_id_key"] = goalie.goalie_id_key.astype("string")
    rows = goalie.merge(meta, on="game_id", how="inner", validate="many_to_one", suffixes=("_goalie", "_spine"))
    if rows.game_date.isna().any() or rows.source_season.isna().any():
        raise ValueError("Every goalie start must have canonical game date and source season")
    for col in ["shots_against_pbp", "gsax"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows.loc[
        rows.shots_against_pbp.gt(0) & rows.gsax.notna()
    ].copy()
    rows["shots"] = rows.shots_against_pbp.astype(float)
    rows["gsax"] = rows.gsax.astype(float)
    rows["source_season"] = rows.source_season.astype(int)
    rows["game_date"] = pd.to_datetime(rows.game_date).dt.normalize()
    if rows.duplicated(["game_id", "home_away"]).any():
        raise ValueError("Duplicate goalie starter rows for the same game-side")

    # Estimate the changing league-wide PBP-xG residual using completed dates
    # strictly before the date being rated.
    league = rows.groupby("game_date", as_index=False).agg(
        league_shots=("shots", "sum"), league_gsax=("gsax", "sum")
    ).sort_values("game_date")
    league["prior_league_shots"] = league.league_shots.cumsum().shift(1).fillna(0.0)
    league["prior_league_gsax"] = league.league_gsax.cumsum().shift(1).fillna(0.0)
    league["league_mean_expanding"] = np.divide(
        league.prior_league_gsax,
        league.prior_league_shots,
        out=np.zeros(len(league), dtype=float),
        where=league.prior_league_shots.to_numpy() > 0,
    )
    indexed = league.set_index("game_date")
    for window, name in [(365, "roll365"), (1095, "roll1095")]:
        rolling = indexed[["league_shots", "league_gsax"]].rolling(
            f"{window}D", closed="left", min_periods=1
        ).sum()
        shots = rolling.league_shots.to_numpy(dtype=float)
        gsax = rolling.league_gsax.to_numpy(dtype=float)
        rate = np.divide(
            gsax, shots, out=np.full(len(shots), np.nan, dtype=float), where=shots > 0
        )
        # Fall back to the expanding prior when a recent window is too small.
        rate[(shots < 30000) | ~np.isfinite(rate)] = league.loc[
            (shots < 30000) | ~np.isfinite(rate), "league_mean_expanding"
        ].to_numpy(dtype=float)
        league[f"league_mean_{name}"] = rate
    rows = rows.merge(
        league[["game_date"] + [f"league_mean_{c}" for c in CONTEXTS]],
        on="game_date", how="left", validate="many_to_one",
    )
    for context in CONTEXTS:
        rows[f"league_mean_{context}"] = rows[f"league_mean_{context}"].fillna(0.0)
        rows[f"actual_relative_{context}_per30"] = 30.0 * (
            rows.gsax / rows.shots - rows[f"league_mean_{context}"]
        )

    # Collapse by goalie/date before creating features; same-date outcomes are
    # never visible to any prediction made that date.
    daily = rows.groupby(
        ["goalie_id_key", "game_date"], as_index=False, sort=True
    ).agg(
        shots=("shots", "sum"), gsax=("gsax", "sum"),
        **{f"league_mean_{context}": (f"league_mean_{context}", "first") for context in CONTEXTS},
        source_season=("source_season", "first"),
        game_type=("game_type", "first"),
        player_name=("player_name", "last"), team_abbrev=("team_abbrev", "last"),
        start_rows=("game_id", "count"),
    ).sort_values(["goalie_id_key", "game_date"], kind="stable")
    for context in CONTEXTS:
        daily[f"centered_{context}"] = daily.gsax - daily[f"league_mean_{context}"] * daily.shots
    features = []
    for goalie_id, group in daily.groupby("goalie_id_key", sort=False):
        g = group.copy()
        g["prior_starts"] = np.arange(len(g), dtype=float)
        g["prior_last_start_date"] = g.game_date.shift(1)
        g["prior_shots_career"] = g.shots.cumsum().shift(1).fillna(0.0)
        for context in CONTEXTS:
            g[f"prior_resid_{context}_career"] = (
                g[f"centered_{context}"].cumsum().shift(1).fillna(0.0)
            )
            for window in (20, 40):
                g[f"prior_shots_r{window}"] = (
                    g.shots.rolling(window, min_periods=1).sum().shift(1).fillna(0.0)
                )
                g[f"prior_resid_{context}_r{window}"] = (
                    g[f"centered_{context}"].rolling(window, min_periods=1).sum().shift(1).fillna(0.0)
                )
            for memory in MEMORIES:
                for k in SHRINKAGE_SHOTS:
                    if memory == "career":
                        numerator = g[f"prior_resid_{context}_career"]
                        denominator = g.prior_shots_career
                    else:
                        window = int(memory[1:])
                        numerator = g[f"prior_resid_{context}_r{window}"]
                        denominator = g[f"prior_shots_r{window}"]
                    g[f"{context}_{memory}_k{k}"] = 30.0 * numerator / (denominator + k)
        features.append(g)
    daily_features = pd.concat(features, ignore_index=True)
    out = rows.merge(
        daily_features.drop(columns=[
            "shots", "gsax", *[f"league_mean_{c}" for c in CONTEXTS], "source_season", "game_type",
            "player_name", "team_abbrev", "start_rows", *[f"centered_{c}" for c in CONTEXTS],
        ]),
        on=["goalie_id_key", "game_date"], how="left", validate="many_to_one",
    )
    if (out.prior_last_start_date.notna() & (out.prior_last_start_date >= out.game_date)).any():
        raise ValueError("Leakage check failed: a goalie feature includes a same/future-date start")
    diagnostics = {
        "starter_rows": int(len(rows)),
        "unique_goalies": int(rows.goalie_id_key.nunique()),
        "date_span": [rows.game_date.min().date().isoformat(), rows.game_date.max().date().isoformat()],
        "target_game_sides_with_positive_shots": int(len(rows)),
        "starter_identity_source": "postgame goalie-box starter flag; retrospective only",
        "all_rating_features_cutoff": "strictly before current game_date; same-date starts aggregated and excluded",
    }
    return out, daily_features, diagnostics


def weighted_mse(y: np.ndarray, pred: np.ndarray, shots: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    w = np.asarray(shots, dtype=float)
    return float(np.sum(w * (y - pred) ** 2) / np.sum(w))


def rating_candidates(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        f"{context}_{memory}_k{k}": frame[f"{context}_{memory}_k{k}"].to_numpy(dtype=float)
        for context in CONTEXTS for memory in MEMORIES for k in SHRINKAGE_SHOTS
    }


def select_and_calibrate(validation: pd.DataFrame) -> dict:
    shots = validation.shots.to_numpy(dtype=float)
    candidates = rating_candidates(validation)
    scores = {}
    baselines = {}
    targets = {}
    for name, pred in candidates.items():
        context = name.split("_")[0]
        y = validation[f"actual_relative_{context}_per30"].to_numpy(dtype=float)
        baseline = weighted_mse(y, np.zeros(len(y)), shots)
        baselines[name] = baseline
        targets[name] = y
        # Select using proportional improvement versus that global-baseline
        # window; this keeps 365-day, 3-year, and expanding contexts comparable.
        scores[name] = weighted_mse(y, pred, shots) / baseline
    best = min(scores, key=scores.get)
    if scores[best] >= 1.0:
        context = best.split("_")[0]
        y = targets[best]
        baseline_mse = baselines[best]
        return {
            "selected_feature": "neutral_pool_mean",
            "selected_context": context,
            "selected_memory": "neutral",
            "pseudo_shots": None,
            "selected_validation_weighted_mse": baseline_mse,
            "neutral_validation_weighted_mse": baseline_mse,
            "relative_weighted_mse_by_candidate": scores,
            "calibration_slope": 0.0,
            "validation_calibrated_weighted_mse": baseline_mse,
            "validation_games": int(len(validation)),
        }
    raw = candidates[best]
    context, memory, shrink_text = best.rsplit("_", 2)
    k = int(shrink_text[1:])
    y = targets[best]
    baseline_mse = baselines[best]
    # Pool-wide calibration: estimate how much of an individual rating
    # persists into the next season. The slope is trained only on validation
    # games and is applied to the later target season.
    denominator = float(np.sum(shots * raw * raw))
    slope = 0.0 if denominator <= 1e-12 else float(np.sum(shots * raw * y) / denominator)
    slope = float(np.clip(slope, 0.0, 2.0))
    return {
        "selected_feature": best,
        "selected_context": context,
        "selected_memory": memory,
        "pseudo_shots": k,
        "selected_validation_weighted_mse": float(scores[best] * baseline_mse),
        "neutral_validation_weighted_mse": baseline_mse,
        "relative_weighted_mse_by_candidate": scores,
        "calibration_slope": slope,
        "validation_calibrated_weighted_mse": weighted_mse(y, slope * raw, shots),
        "validation_games": int(len(validation)),
    }


def season_metrics(frame: pd.DataFrame, pred_col: str) -> dict:
    y = frame.actual_relative_gsax_per30.to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    shots = frame.shots.to_numpy(dtype=float)
    baseline = weighted_mse(y, np.zeros(len(y)), shots)
    model = weighted_mse(y, pred, shots)
    denom = float(np.sum(shots * pred * pred))
    slope = 0.0 if denom <= 1e-12 else float(np.sum(shots * pred * y) / denom)
    corr = np.corrcoef(pred, y)[0, 1] if np.std(pred) > 1e-12 and np.std(y) > 1e-12 else 0.0
    return {
        "goalie_starts": int(len(frame)),
        "goalies": int(frame.goalie_id_key.nunique()),
        "shots_against": float(shots.sum()),
        "neutral_pool_weighted_rmse_goals_per30": float(np.sqrt(baseline)),
        "calibrated_pool_weighted_rmse_goals_per30": float(np.sqrt(model)),
        "relative_weighted_mse_improvement_vs_neutral": float((baseline - model) / baseline),
        "test_calibration_slope_through_origin": slope,
        "correlation": float(corr),
        "mean_predicted_skill_delta_per30": float(np.average(pred, weights=shots)),
        "mean_actual_skill_delta_per30": float(np.average(y, weights=shots)),
    }


def build_season_opening_cohorts(rows: pd.DataFrame) -> pd.DataFrame:
    """One pre-season goalie rating paired with that goalie's full-season result."""
    regular = rows.loc[rows.game_type.eq("R")].copy()
    first_dates = regular.groupby(
        ["source_season", "goalie_id_key"], as_index=False
    ).game_date.min().rename(columns={"game_date": "season_first_start_date"})
    openings = regular.merge(
        first_dates, on=["source_season", "goalie_id_key"], how="inner",
        validate="many_to_one",
    )
    openings = openings.loc[openings.game_date.eq(openings.season_first_start_date)].copy()
    # If a goalie has two same-day records, retain the one actual starter row.
    openings = openings.sort_values(["source_season", "goalie_id_key", "game_id"]).drop_duplicates(
        ["source_season", "goalie_id_key"], keep="first"
    )
    outcomes = regular.groupby(["source_season", "goalie_id_key"], as_index=False).agg(
        season_shots=("shots", "sum"), season_gsax=("gsax", "sum"),
        season_starts=("game_id", "size"),
    )
    season_pool = regular.groupby("source_season", as_index=False).agg(
        pool_shots=("shots", "sum"), pool_gsax=("gsax", "sum")
    )
    season_pool["season_pool_gsax_per30"] = 30.0 * season_pool.pool_gsax / season_pool.pool_shots
    cohort = openings.merge(
        outcomes, on=["source_season", "goalie_id_key"], how="inner", validate="one_to_one"
    ).merge(season_pool, on="source_season", how="left", validate="many_to_one")
    cohort["actual_relative_gsax_per30"] = (
        30.0 * cohort.season_gsax / cohort.season_shots - cohort.season_pool_gsax_per30
    )
    return cohort


def select_and_calibrate_season(validation: pd.DataFrame) -> dict:
    y = validation.actual_relative_gsax_per30.to_numpy(dtype=float)
    shots = validation.season_shots.to_numpy(dtype=float)
    candidates = rating_candidates(validation)
    baseline = weighted_mse(y, np.zeros(len(y)), shots)
    scores = {name: weighted_mse(y, pred, shots) / baseline for name, pred in candidates.items()}
    best = min(scores, key=scores.get)
    if scores[best] >= 1.0:
        return {
            "selected_feature": "neutral_pool_mean", "selected_context": best.split("_")[0],
            "selected_memory": "neutral", "pseudo_shots": None,
            "validation_relative_mse_ratio": 1.0, "calibration_slope": 0.0,
            "validation_goalies": int(len(validation)),
        }
    raw = candidates[best]
    context, memory, shrink_text = best.rsplit("_", 2)
    k = int(shrink_text[1:])
    denom = float(np.sum(shots * raw * raw))
    slope = 0.0 if denom <= 1e-12 else float(np.sum(shots * raw * y) / denom)
    slope = float(np.clip(slope, 0.0, 2.0))
    return {
        "selected_feature": best, "selected_context": context,
        "selected_memory": memory, "pseudo_shots": k,
        "validation_relative_mse_ratio": float(scores[best]),
        "calibration_slope": slope,
        "validation_goalies": int(len(validation)),
        "candidate_relative_mse_ratio": scores,
    }


def make_reliability(rows: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for season, group in rows.groupby("target_season", sort=True):
        group = group.copy()
        if group.predicted_skill_delta_per30.nunique() < 2:
            continue
        group["rating_band"] = pd.qcut(
            group.predicted_skill_delta_per30.rank(method="first"),
            q=min(10, group.predicted_skill_delta_per30.nunique()),
            labels=False,
            duplicates="drop",
        )
        for band, g in group.groupby("rating_band", sort=True):
            pieces.append({
                "target_season": int(season), "rating_band": int(band),
                "goalie_starts": int(len(g)), "goalies": int(g.goalie_id_key.nunique()),
                "shots_against": float(g.shots.sum()),
                "predicted_skill_delta_per30": float(np.average(g.predicted_skill_delta_per30, weights=g.shots)),
                "observed_skill_delta_per30": float(np.average(g.actual_relative_gsax_per30, weights=g.shots)),
            })
    return pd.DataFrame(pieces)


def latest_pool(daily: pd.DataFrame, selected: dict, outpath: Path) -> pd.DataFrame:
    feature = selected["selected_feature"]
    context = selected["selected_context"]
    memory = selected["selected_memory"]
    k = int(selected["pseudo_shots"] or 0)
    rows = daily.loc[daily.game_date < AS_OF].copy()
    if rows.empty:
        raise ValueError("No goalie games exist before the requested rating date")
    current = rows.sort_values(["goalie_id_key", "game_date"]).groupby("goalie_id_key", as_index=False).tail(1).copy()
    current["starts_asof"] = current.prior_starts + 1.0
    current["shots_asof"] = current.prior_shots_career + current.shots
    if feature == "neutral_pool_mean":
        raw = np.zeros(len(current), dtype=float)
    else:
        centered = current[f"centered_{context}"]
        if memory == "career":
            numerator = current[f"prior_resid_{context}_career"] + centered
            denominator = current.prior_shots_career + current.shots
        else:
            window = int(memory[1:])
            numerator = current[f"prior_resid_{context}_r{window}"] + centered
            denominator = current[f"prior_shots_r{window}"] + current.shots
        raw = 30.0 * numerator / (denominator + k)
    slope = float(selected["calibration_slope"])
    current["raw_pool_skill_delta_per30"] = raw
    current["calibrated_pool_skill_delta_per30"] = slope * raw
    if context == "expanding":
        baseline_rows = rows
    else:
        days = 365 if context == "roll365" else 1095
        baseline_rows = rows.loc[rows.game_date >= AS_OF - pd.Timedelta(days=days)]
        if baseline_rows.shots.sum() < 30000:
            baseline_rows = rows
    asof_shots = float(baseline_rows.shots.sum())
    asof_gsax = float(baseline_rows.gsax.sum())
    current["league_mean_gsax_per30_asof"] = 30.0 * asof_gsax / asof_shots
    current["estimated_gsax_per30_asof"] = (
        current.league_mean_gsax_per30_asof + current.calibrated_pool_skill_delta_per30
    )
    current["as_of_date"] = AS_OF.date().isoformat()
    current["active_in_source_season_2027"] = current.source_season.eq(2027)
    current["calibration_feature"] = feature
    current["calibration_context"] = context
    current["calibration_memory"] = memory
    current["calibration_slope"] = slope
    current["rating_status"] = "RESEARCH; named-goalie estimate; not live pregame confirmation"
    current = current.rename(columns={"goalie_id_key": "goalie_id"})
    current = current.sort_values("calibrated_pool_skill_delta_per30", ascending=False)
    current.to_parquet(outpath, index=False)
    return current


def main(goalie_path: str, spine_path: str, outdir: str) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows, daily, diagnostics = build_pool(goalie_path, spine_path)
    predictions = []
    folds = {}
    for target in TARGET_SEASONS:
        validation_season = target - 1
        validation = rows.loc[
            rows.source_season.eq(validation_season) & rows.game_type.eq("R")
        ].copy()
        test = rows.loc[
            rows.source_season.eq(target) & rows.game_type.eq("R")
        ].copy()
        if len(validation) < 1000 or len(test) < 1000:
            continue
        selected = select_and_calibrate(validation)
        feature = selected["selected_feature"]
        if feature == "neutral_pool_mean":
            raw = np.zeros(len(test), dtype=float)
        else:
            raw = test[feature].to_numpy(dtype=float)
        test = test.copy()
        context = selected["selected_context"]
        test["actual_relative_gsax_per30"] = test[f"actual_relative_{context}_per30"]
        test["target_season"] = target
        test["validation_season"] = validation_season
        test["selected_feature"] = feature
        test["selected_context"] = context
        test["calibration_slope"] = selected["calibration_slope"]
        test["raw_skill_delta_per30"] = raw
        test["predicted_skill_delta_per30"] = selected["calibration_slope"] * raw
        folds[str(target)] = {
            "validation_season": validation_season,
            "validation_selection": selected,
            "test_metrics": season_metrics(test, "predicted_skill_delta_per30"),
        }
        predictions.append(test)

    if not predictions:
        raise RuntimeError("No complete rolling-origin goalie calibration folds were scored")
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_parquet(out / "goalie_pool_oof_calibration.parquet", index=False)
    reliability = make_reliability(oof)
    reliability.to_parquet(out / "reliability_by_rating_band.parquet", index=False)

    # Also evaluate each individual goalie's preseason rating against that
    # goalie's whole next-season performance. This reduces the noise of a
    # single-game GSAx target and is closer to calibrating latent goalie ability.
    cohorts = build_season_opening_cohorts(rows)
    season_oof = []
    season_folds = {}
    for target in TARGET_SEASONS:
        validation_season = target - 1
        validation = cohorts.loc[cohorts.source_season.eq(validation_season)].copy()
        test = cohorts.loc[cohorts.source_season.eq(target)].copy()
        if len(validation) < 50 or len(test) < 50:
            continue
        selected = select_and_calibrate_season(validation)
        feature = selected["selected_feature"]
        raw = np.zeros(len(test), dtype=float) if feature == "neutral_pool_mean" else test[feature].to_numpy(dtype=float)
        test["target_season"] = target
        test["validation_season"] = validation_season
        test["selected_feature"] = feature
        test["calibration_slope"] = selected["calibration_slope"]
        test["raw_skill_delta_per30"] = raw
        test["predicted_skill_delta_per30"] = selected["calibration_slope"] * raw
        test["shots"] = test.season_shots
        season_folds[str(target)] = {
            "validation_selection": selected,
            "test_metrics": season_metrics(test, "predicted_skill_delta_per30"),
        }
        season_oof.append(test)
    if not season_oof:
        raise RuntimeError("No season-level goalie-pool folds could be scored")
    season_oof = pd.concat(season_oof, ignore_index=True)
    season_oof.to_parquet(out / "goalie_pool_season_oof.parquet", index=False)
    season_reliability = make_reliability(season_oof)
    season_reliability.to_parquet(out / "season_reliability_by_rating_band.parquet", index=False)

    # Fit the latest individual-goalie calibration from the 2026 goalie pool,
    # then update each named rating using starts strictly before Nov. 2.
    validation_2026 = cohorts.loc[cohorts.source_season.eq(2026)].copy()
    selected_2026 = select_and_calibrate_season(validation_2026)
    pool = latest_pool(daily, selected_2026, out / "pool_ratings_asof_20261102.parquet")
    summary = {
        "experiment_id": "EXP-0035",
        "model_id": MODEL_ID,
        "status": "RESEARCH; no promotion, public export, or site update",
        "as_of_rating_date": AS_OF.date().isoformat(),
        "starter_identity_caveat": "Historical target starter ID is from postgame goalie box; features are strict prior-date.",
        "calibration_target": "Per-goalie GSAx relative to a pregame league-average GSAx baseline, in goals per 30 shots.",
        "why_centered": "The goalie PBP GSAx baseline shifts by season; a lifetime zero-centered prior can confuse xG-model/environment drift with an individual goalie's ability.",
        "validation_design": "For each target source season 2018-2026, choose pool-centering window, goalie-memory window, and shrinkage on the immediately preceding regular season; fit the calibration slope there; then test on target regular season.",
        "candidate_features": "Pregame league baseline: expanding, prior 365 days, or prior 1095 days; goalie history: career, prior 20 starts, or prior 40 starts; pseudo-shot shrinkage: 100, 1,000, or 5,000.",
        "pool_diagnostics": diagnostics,
        "weighted_gsax_per30_by_source_season": {
            str(int(season)): float(30.0 * g.gsax.sum() / g.shots.sum())
            for season, g in rows.groupby("source_season", sort=True)
        },
        "rolling_origin_folds": folds,
        "pooled_test_metrics": season_metrics(oof, "predicted_skill_delta_per30"),
        "season_level_validation_design": "Rate each goalie before its first regular-season start; evaluate against its full regular-season GSAx relative to that season's full-pool mean; fit calibration on the prior season and score the next season.",
        "season_level_rolling_origin_folds": season_folds,
        "season_level_pooled_test_metrics": season_metrics(season_oof, "predicted_skill_delta_per30"),
        "latest_pool_rating_selection": selected_2026,
        "latest_pool_counts": {
            "goalies_rated": int(len(pool)),
            "active_in_source_season_2027": int(pool.active_in_source_season_2027.sum()),
            "top_10_current_pool": pool.loc[pool.active_in_source_season_2027].head(10)[
                ["goalie_id", "player_name", "team_abbrev", "starts_asof", "shots_asof", "calibrated_pool_skill_delta_per30"]
            ].to_dict(orient="records"),
        },
        "decision": "Use this calibrated individual-goalie skill rating as the named-goalie input candidate. Do not force its coefficient into public win probabilities until the next nested goals/HWIN test confirms lift.",
        "artifacts": {
            "goalie_pool_oof_calibration": "goalie_pool_oof_calibration.parquet",
            "reliability_by_rating_band": "reliability_by_rating_band.parquet",
            "goalie_pool_season_oof": "goalie_pool_season_oof.parquet",
            "season_reliability_by_rating_band": "season_reliability_by_rating_band.parquet",
            "pool_ratings_asof_20261102": "pool_ratings_asof_20261102.parquet",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--goalie",
        default="research/.tmp_goalie_probability_016/goalie/data/goalie_feature_output/goalie_game_features.parquet",
    )
    parser.add_argument(
        "--spine",
        default="research/.tmp_goalie_probability_016/spine/data/spine_v3/games.parquet",
    )
    parser.add_argument(
        "--out",
        default="research/.tmp_goalie_probability_016/out/goalie_pool_calibration_021",
    )
    args = parser.parse_args()
    main(args.goalie, args.spine, args.out)
