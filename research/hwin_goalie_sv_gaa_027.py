"""Test named-goalie save percentage and goals-against skill on HWIN.

The feature is each game's two named starters' prior-only, league-centered
save-percentage and goals-against-rate estimates. A nested rolling-origin test
first selects/shrinks goalie-pool ratings on the preceding season, then tests
whether those ratings improve the frozen HWIN Game v1.0.0 win probabilities.

Target-game starter identity comes from postgame goalie records, so results are
retrospective upper-bound evidence until timestamped pregame starter snapshots
are used. No market inputs or production/site changes are made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit


EXPERIMENT_ID = "EXP-0042"
MODEL_ID = "HOCKEY-LAB-HWIN-NAMED-GOALIE-SV-GAA-027-RESEARCH"
TARGET_SEASONS = [2024, 2025, 2026]
MEMORIES = ["career", "r20", "r40"]
SV_PSEUDO_SHOTS = [100, 300, 600, 1200]
GAA_PSEUDO_MINUTES = [300, 900, 1800, 3600]
PENALTIES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 4203


def norm_ids(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"\.0$", "", regex=True)


def rate_loss_binomial(frame: pd.DataFrame, rating: np.ndarray, alpha: float) -> float:
    base = np.clip(frame.base_sv.to_numpy(float), 1e-5, 1 - 1e-5)
    p = expit(logit(base) + alpha * rating)
    shots = frame.shots.to_numpy(float)
    saves = frame.saves.to_numpy(float)
    nll = -(saves * np.log(p) + (shots - saves) * np.log1p(-p))
    return float(nll.sum() / shots.sum())


def rate_loss_poisson(frame: pd.DataFrame, rating: np.ndarray, alpha: float) -> float:
    base = np.clip(frame.base_gaa.to_numpy(float), 1e-5, None)
    rate = np.clip(base * np.exp(-alpha * rating), 1e-5, None)
    hours = frame.toi_seconds.to_numpy(float) / 3600.0
    goals = frame.goals_against.to_numpy(float)
    lam = np.clip(hours * rate, 1e-8, None)
    return float(np.sum(lam - goals * np.log(lam)) / hours.sum())


def fit_pool_alpha(frame: pd.DataFrame, rating_col: str, metric: str) -> tuple[float, float]:
    rating = frame[rating_col].to_numpy(float)
    loss = rate_loss_binomial if metric == "sv" else rate_loss_poisson
    base_loss = loss(frame, np.zeros(len(frame)), 0.0)
    fit = minimize_scalar(
        lambda a: loss(frame, rating, float(a)), bounds=(0.0, 3.0), method="bounded",
        options={"xatol": 1e-10},
    )
    if not fit.success or not np.isfinite(fit.fun) or fit.fun >= base_loss:
        return 0.0, base_loss
    return float(fit.x), float(fit.fun)


def pool_candidate_columns(metric: str) -> list[str]:
    if metric == "sv":
        return [f"sv_skill_{m}_k{k}" for m in MEMORIES for k in SV_PSEUDO_SHOTS]
    return [f"gaa_skill_{m}_m{k}" for m in MEMORIES for k in GAA_PSEUDO_MINUTES]


def select_pool_rating(validation: pd.DataFrame, metric: str) -> dict:
    loss = rate_loss_binomial if metric == "sv" else rate_loss_poisson
    base_loss = loss(validation, np.zeros(len(validation)), 0.0)
    best = {"rating_col": "neutral_pool_baseline", "alpha": 0.0, "validation_loss": base_loss}
    for col in pool_candidate_columns(metric):
        alpha, score = fit_pool_alpha(validation, col, metric)
        if score < best["validation_loss"]:
            best = {"rating_col": col, "alpha": alpha, "validation_loss": score}
    best["validation_baseline_loss"] = base_loss
    return best


def pool_metrics(frame: pd.DataFrame, metric: str, config: dict) -> dict:
    if metric == "sv":
        base = frame.base_sv.to_numpy(float)
        shots = frame.shots.to_numpy(float)
        saves = frame.saves.to_numpy(float)
        rating = np.zeros(len(frame)) if config["rating_col"] == "neutral_pool_baseline" else frame[config["rating_col"]].to_numpy(float)
        pred = expit(logit(np.clip(base, 1e-5, 1 - 1e-5)) + config["alpha"] * rating)
        baseline_nll = rate_loss_binomial(frame, np.zeros(len(frame)), 0.0)
        candidate_nll = rate_loss_binomial(frame, rating, config["alpha"])
        baseline_brier = float(np.sum(shots * base**2 - 2 * base * saves + saves) / shots.sum())
        candidate_brier = float(np.sum(shots * pred**2 - 2 * pred * saves + saves) / shots.sum())
        return {"goalie_starts": int(len(frame)), "shots_against": int(shots.sum()),
                "baseline_save_pct_logloss_per_shot": baseline_nll,
                "candidate_save_pct_logloss_per_shot": candidate_nll,
                "save_pct_logloss_delta": candidate_nll - baseline_nll,
                "baseline_save_pct_brier_per_shot": baseline_brier,
                "candidate_save_pct_brier_per_shot": candidate_brier,
                "save_pct_brier_delta": candidate_brier - baseline_brier,
                "mean_predicted_save_pct": float(np.average(pred, weights=shots)),
                "mean_observed_save_pct": float(saves.sum() / shots.sum())}
    base = frame.base_gaa.to_numpy(float)
    hours = frame.toi_seconds.to_numpy(float) / 3600.0
    goals = frame.goals_against.to_numpy(float)
    rating = np.zeros(len(frame)) if config["rating_col"] == "neutral_pool_baseline" else frame[config["rating_col"]].to_numpy(float)
    pred = base * np.exp(-config["alpha"] * rating)
    baseline_loss = rate_loss_poisson(frame, np.zeros(len(frame)), 0.0)
    candidate_loss = rate_loss_poisson(frame, rating, config["alpha"])
    return {"goalie_starts": int(len(frame)), "goalie_minutes": float(hours.sum() * 60.0),
            "baseline_goals_against_poisson_nll_per_hour": baseline_loss,
            "candidate_goals_against_poisson_nll_per_hour": candidate_loss,
            "gaa_nll_delta": candidate_loss - baseline_loss,
            "mean_predicted_gaa": float(np.average(pred, weights=hours)),
            "mean_observed_gaa": float(goals.sum() / hours.sum())}


def build_goalie_features(goalie_path: str, spine_path: str) -> tuple[pd.DataFrame, dict]:
    goalie = pd.read_parquet(goalie_path)
    spine = pd.read_parquet(spine_path)
    meta = spine[["game_id", "game_date", "source_season", "game_type"]].copy()
    meta["game_id"] = norm_ids(meta.game_id)
    meta["game_date"] = pd.to_datetime(meta.game_date, errors="coerce").dt.normalize()
    goalie = goalie.loc[goalie.starter_bool.fillna(False).astype(bool), [
        "game_id", "goalie_id_key", "home_away", "team_abbrev", "player_name",
        "shots_against", "saves", "goals_against", "toi_seconds_box",
    ]].copy()
    goalie["game_id"] = norm_ids(goalie.game_id)
    goalie["goalie_id_key"] = goalie.goalie_id_key.astype("string")
    rows = goalie.merge(meta, on="game_id", how="inner", validate="many_to_one")
    rows = rows.loc[rows.game_type.eq("R")].copy()
    for col in ["shots_against", "saves", "goals_against", "toi_seconds_box"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows.loc[
        rows.goalie_id_key.notna() & rows.game_date.notna() & rows.toi_seconds_box.gt(0)
    ].copy()
    rows = rows.rename(columns={
        "shots_against": "shots", "toi_seconds_box": "toi_seconds",
    })
    if rows.duplicated(["game_id", "home_away"]).any():
        raise ValueError("Duplicate starter rows for a game-side")

    # League reference rates use only regular-season starter results strictly
    # before each date; rolling 365-day rates fall back to expanding history.
    league = rows.groupby("game_date", as_index=False).agg(
        shots=("shots", "sum"), saves=("saves", "sum"),
        goals_against=("goals_against", "sum"), toi_seconds=("toi_seconds", "sum"),
    ).sort_values("game_date")
    league["exp_shots"] = league.shots.cumsum().shift(1).fillna(0.0)
    league["exp_saves"] = league.saves.cumsum().shift(1).fillna(0.0)
    league["exp_ga"] = league.goals_against.cumsum().shift(1).fillna(0.0)
    league["exp_toi"] = league.toi_seconds.cumsum().shift(1).fillna(0.0)
    indexed = league.set_index("game_date")
    rolling = indexed[["shots", "saves", "goals_against", "toi_seconds"]].rolling(
        "365D", closed="left", min_periods=1
    ).sum()
    league["roll_shots"] = rolling.shots.to_numpy(float)
    league["roll_saves"] = rolling.saves.to_numpy(float)
    league["roll_ga"] = rolling.goals_against.to_numpy(float)
    league["roll_toi"] = rolling.toi_seconds.to_numpy(float)
    use_roll_sv = league.roll_shots.ge(50000)
    use_roll_gaa = league.roll_toi.ge(50000 * 60)
    league["base_sv"] = np.where(
        use_roll_sv, league.roll_saves / league.roll_shots,
        league.exp_saves / league.exp_shots.replace(0, np.nan),
    )
    league["base_gaa"] = np.where(
        use_roll_gaa, league.roll_ga * 3600.0 / league.roll_toi,
        league.exp_ga * 3600.0 / league.exp_toi.replace(0, np.nan),
    )
    league["base_sv"] = league.base_sv.fillna(0.90).clip(0.75, 0.99)
    league["base_gaa"] = league.base_gaa.fillna(2.9).clip(0.25, 8.0)
    rows = rows.merge(league[["game_date", "base_sv", "base_gaa"]], on="game_date", how="left", validate="many_to_one")

    daily = rows.groupby(["goalie_id_key", "game_date"], as_index=False, sort=True).agg(
        shots=("shots", "sum"), saves=("saves", "sum"),
        goals_against=("goals_against", "sum"), toi_seconds=("toi_seconds", "sum"),
    ).sort_values(["goalie_id_key", "game_date"], kind="stable")
    pieces = []
    for _, g in daily.groupby("goalie_id_key", sort=False):
        g = g.copy()
        g["prior_start_dates"] = np.arange(len(g), dtype=float)
        for col in ["shots", "saves", "goals_against", "toi_seconds"]:
            g[f"career_{col}"] = g[col].cumsum().shift(1).fillna(0.0)
            for w in [20, 40]:
                g[f"r{w}_{col}"] = g[col].rolling(w, min_periods=1).sum().shift(1).fillna(0.0)
        pieces.append(g)
    history = pd.concat(pieces, ignore_index=True)
    prior_cols = [c for c in history.columns if c.startswith("career_") or c.startswith("r20_") or c.startswith("r40_")]
    rows = rows.merge(history[["goalie_id_key", "game_date", "prior_start_dates"] + prior_cols],
                      on=["goalie_id_key", "game_date"], how="left", validate="many_to_one")
    if rows.prior_start_dates.isna().any():
        raise ValueError("Missing prior-start feature rows")

    for mem in MEMORIES:
        prefix = "career" if mem == "career" else mem
        pshots = rows[f"{prefix}_shots"].to_numpy(float)
        psaves = rows[f"{prefix}_saves"].to_numpy(float)
        pga = rows[f"{prefix}_goals_against"].to_numpy(float)
        ptoi_hours = rows[f"{prefix}_toi_seconds"].to_numpy(float) / 3600.0
        base_sv = rows.base_sv.to_numpy(float)
        base_gaa = rows.base_gaa.to_numpy(float)
        for k in SV_PSEUDO_SHOTS:
            est = (psaves + k * base_sv) / np.maximum(pshots + k, 1.0)
            rows[f"sv_skill_{mem}_k{k}"] = logit(np.clip(est, 1e-5, 1 - 1e-5)) - logit(np.clip(base_sv, 1e-5, 1 - 1e-5))
        for k in GAA_PSEUDO_MINUTES:
            est = (pga + base_gaa * (k / 60.0)) / np.maximum(ptoi_hours + k / 60.0, 1e-6)
            rows[f"gaa_skill_{mem}_m{k}"] = np.log(np.clip(base_gaa / np.clip(est, 1e-5, None), 1e-5, 1e5))

    if (rows.prior_start_dates < 0).any():
        raise ValueError("Negative prior goalie start count")
    diagnostics = {
        "regular_season_starter_rows": int(len(rows)),
        "regular_season_games": int(rows.game_id.nunique()),
        "unique_goalies": int(rows.goalie_id_key.nunique()),
        "date_span": [rows.game_date.min().date().isoformat(), rows.game_date.max().date().isoformat()],
        "history_cutoff": "All goalie performance and league reference statistics use game dates strictly before the target date.",
        "target_identity_source": "Postgame goalie box starter flag; retrospective until timestamped pregame identity is joined.",
        "league_reference": "Rolling 365-day prior-date league rates, with expanding prior-date fallback when exposure is thin.",
    }
    return rows, diagnostics


def make_win_data(base_path: str, goalie_rows: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_parquet(base_path)
    base = base.loc[
        base.model_variant.eq("team_strength_004_plus_b2b") & base.game_type.eq("R")
        & base.source_season.between(2022, 2026),
        ["game_id", "game_date", "source_season", "home_team", "away_team", "home_win", "p_home"],
    ].copy()
    base["game_id"] = norm_ids(base.game_id)
    base["game_date"] = pd.to_datetime(base.game_date, errors="coerce").dt.normalize()
    base["home_win"] = base.home_win.astype(int)
    base["p_base"] = base.p_home.astype(float).clip(1e-8, 1 - 1e-8)
    if len(base) != 5 * 1312 or base.game_id.duplicated().any():
        raise ValueError("Frozen HWIN regular-season ledger has unexpected count or duplicate games")
    starters = goalie_rows.loc[
        goalie_rows.game_id.isin(base.game_id),
        ["game_id", "goalie_id_key", "home_away", "team_abbrev", "player_name",
         "source_season", "game_date", "base_sv", "base_gaa", "prior_start_dates"]
        + pool_candidate_columns("sv") + pool_candidate_columns("gaa"),
    ].copy()
    if starters.duplicated(["game_id", "home_away"]).any() or starters.groupby("game_id").size().ne(2).any():
        raise ValueError("Each HWIN game must map to exactly two named starters")
    games = base.merge(
        starters.loc[starters.home_away.eq("home")].rename(columns={
            "goalie_id_key": "home_goalie_id", "team_abbrev": "home_goalie_team", "player_name": "home_goalie_name",
            "prior_start_dates": "home_prior_start_dates", "base_sv": "home_base_sv", "base_gaa": "home_base_gaa",
            **{c: f"home_{c}" for c in pool_candidate_columns("sv") + pool_candidate_columns("gaa")},
        }).drop(columns=["home_away", "source_season", "game_date"]), on="game_id", how="inner", validate="one_to_one")
    games = games.merge(
        starters.loc[starters.home_away.eq("away")].rename(columns={
            "goalie_id_key": "away_goalie_id", "team_abbrev": "away_goalie_team", "player_name": "away_goalie_name",
            "prior_start_dates": "away_prior_start_dates", "base_sv": "away_base_sv", "base_gaa": "away_base_gaa",
            **{c: f"away_{c}" for c in pool_candidate_columns("sv") + pool_candidate_columns("gaa")},
        }).drop(columns=["home_away", "source_season", "game_date"]), on="game_id", how="inner", validate="one_to_one")
    if len(games) != len(base):
        raise ValueError("Starter matching did not preserve the frozen HWIN game population")
    if not games.home_team.astype(str).eq(games.home_goalie_team.astype(str)).all():
        raise ValueError("Home goalie team does not match HWIN home team")
    if not games.away_team.astype(str).eq(games.away_goalie_team.astype(str)).all():
        raise ValueError("Away goalie team does not match HWIN away team")
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def apply_pool_rating(rows: pd.DataFrame, metric: str, config: dict, side: str = "") -> np.ndarray:
    if config["rating_col"] == "neutral_pool_baseline":
        rating = np.zeros(len(rows), dtype=float)
    else:
        col = f"{side}{config['rating_col']}" if side else config["rating_col"]
        rating = rows[col].to_numpy(float)
    if metric == "sv":
        base_col = f"{side}base_sv" if side else "base_sv"
        return expit(logit(np.clip(rows[base_col].to_numpy(float), 1e-5, 1 - 1e-5)) + config["alpha"] * rating)
    base_col = f"{side}base_gaa" if side else "base_gaa"
    return rows[base_col].to_numpy(float) * np.exp(-config["alpha"] * rating)


def fit_win_beta(frame: pd.DataFrame, cols: list[str], penalty: float) -> np.ndarray:
    y = frame.home_win.to_numpy(float)
    offset = logit(frame.p_base.to_numpy(float))
    x = frame[cols].to_numpy(float)
    def objective(beta: np.ndarray) -> float:
        eta = np.clip(offset + x @ beta, -25.0, 25.0)
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta) + 0.5 * penalty * np.sum(beta**2))
    fit = minimize(objective, np.zeros(len(cols)), method="L-BFGS-B", bounds=[(0.0, 25.0)] * len(cols),
                   options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000})
    if not fit.success and not np.isfinite(fit.fun):
        raise RuntimeError(f"Win adjustment fit failed: {fit.message}")
    return np.asarray(fit.x, dtype=float)


def probability(frame: pd.DataFrame, cols: list[str], beta: np.ndarray) -> np.ndarray:
    p = expit(np.clip(logit(frame.p_base.to_numpy(float)) + frame[cols].to_numpy(float) @ beta, -25.0, 25.0))
    return p


def calibration(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, float)
    z = logit(np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8))
    def objective(theta: np.ndarray) -> float:
        eta = np.clip(theta[0] + theta[1] * z, -25.0, 25.0)
        return float(np.mean(np.logaddexp(0, eta) - y * eta))
    fit = minimize(objective, np.array([0.0, 1.0]), method="L-BFGS-B")
    return {"intercept": float(fit.x[0]), "slope": float(fit.x[1]), "converged": bool(fit.success)}


def win_metrics(frame: pd.DataFrame, p: np.ndarray) -> dict:
    y = frame.home_win.to_numpy(float)
    p = np.clip(np.asarray(p, float), 1e-10, 1 - 1e-10)
    ll = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return {"games": int(len(y)), "log_loss": float(ll.mean()), "brier": float(np.mean((p - y) ** 2)),
            "accuracy": float(np.mean((p >= 0.5) == y.astype(bool))),
            "mean_home_probability": float(p.mean()), "home_win_rate": float(y.mean()),
            "calibration": calibration(y, p)}


def paired_ci(pred: pd.DataFrame) -> dict:
    y = pred.home_win.to_numpy(float)
    p0 = np.clip(pred.p_base.to_numpy(float), 1e-10, 1 - 1e-10)
    p1 = np.clip(pred.p_candidate.to_numpy(float), 1e-10, 1 - 1e-10)
    ll0 = -(y * np.log(p0) + (1 - y) * np.log1p(-p0))
    ll1 = -(y * np.log(p1) + (1 - y) * np.log1p(-p1))
    b0, b1 = (p0 - y) ** 2, (p1 - y) ** 2
    dates = pd.to_datetime(pred.game_date).dt.normalize()
    block = ((dates - dates.min()).dt.days // 14).to_numpy()
    members = [np.flatnonzero(block == b) for b in np.unique(block)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    dl, db = np.empty(BOOTSTRAP_REPS), np.empty(BOOTSTRAP_REPS)
    for i in range(BOOTSTRAP_REPS):
        ix = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        dl[i] = (ll1[ix] - ll0[ix]).mean()
        db[i] = (b1[ix] - b0[ix]).mean()
    return {"log_loss_delta_candidate_minus_base": float((ll1 - ll0).mean()),
            "log_loss_95pct_14_day_block_ci": [float(x) for x in np.quantile(dl, [0.025, 0.975])],
            "brier_delta_candidate_minus_base": float((b1 - b0).mean()),
            "brier_95pct_14_day_block_ci": [float(x) for x in np.quantile(db, [0.025, 0.975])],
            "bootstrap_replicates": BOOTSTRAP_REPS, "calendar_blocks": int(len(members))}


def run(base_path: str, goalie_path: str, spine_path: str, outdir: str) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    goalie_rows, goalie_diag = build_goalie_features(goalie_path, spine_path)
    games = make_win_data(base_path, goalie_rows)
    preds, folds = [], {}
    for target in TARGET_SEASONS:
        validation_season = target - 1
        pool_val = goalie_rows.loc[goalie_rows.source_season.eq(validation_season)].copy()
        pool_test = goalie_rows.loc[goalie_rows.source_season.eq(target)].copy()
        if pool_val.empty or pool_test.empty:
            raise ValueError(f"Missing goalie pool calibration season {validation_season} or {target}")
        sv_cfg = select_pool_rating(pool_val, "sv")
        gaa_cfg = select_pool_rating(pool_val, "gaa")
        pool_sv_metrics = pool_metrics(pool_test, "sv", sv_cfg)
        pool_gaa_metrics = pool_metrics(pool_test, "gaa", gaa_cfg)
        all_games = games.copy()
        all_games["home_pred_sv"] = apply_pool_rating(all_games, "sv", sv_cfg, "home_")
        all_games["away_pred_sv"] = apply_pool_rating(all_games, "sv", sv_cfg, "away_")
        all_games["home_pred_gaa"] = apply_pool_rating(all_games, "gaa", gaa_cfg, "home_")
        all_games["away_pred_gaa"] = apply_pool_rating(all_games, "gaa", gaa_cfg, "away_")
        all_games["sv_gap_pp"] = 100.0 * (all_games.home_pred_sv - all_games.away_pred_sv)
        all_games["gaa_gap"] = all_games.away_pred_gaa - all_games.home_pred_gaa
        fold_games = all_games.loc[all_games.source_season.le(target)].copy()
        train = fold_games.loc[fold_games.source_season.le(target - 2)].copy()
        val = fold_games.loc[fold_games.source_season.eq(validation_season)].copy()
        test = fold_games.loc[fold_games.source_season.eq(target)].copy()
        if len(train) < 1000 or len(val) != 1312 or len(test) != 1312:
            raise ValueError(f"Incomplete win-probability fold {target}: {len(train)}, {len(val)}, {len(test)}")
        cols_by_family = {
            "sv_only": ["sv_gap_pp"], "gaa_only": ["gaa_gap"],
            "save_pct_plus_gaa": ["sv_gap_pp", "gaa_gap"],
        }
        baseline_val = win_metrics(val, val.p_base.to_numpy(float))
        choices = [{"family": "baseline_no_goalie", "cols": [], "penalty": None,
                    "beta": np.array([], dtype=float), "validation_log_loss": baseline_val["log_loss"]}]
        for family, cols in cols_by_family.items():
            for penalty in PENALTIES:
                beta = fit_win_beta(train, cols, penalty)
                pval = probability(val, cols, beta)
                choices.append({"family": family, "cols": cols, "penalty": penalty, "beta": beta,
                                "validation_log_loss": win_metrics(val, pval)["log_loss"]})
        selected = min(choices, key=lambda z: z["validation_log_loss"])
        if selected["family"] == "baseline_no_goalie" or selected["validation_log_loss"] >= baseline_val["log_loss"]:
            family, cols, penalty, beta = "baseline_no_goalie", [], None, np.array([], dtype=float)
        else:
            family, cols, penalty = selected["family"], selected["cols"], float(selected["penalty"])
            beta = fit_win_beta(pd.concat([train, val], ignore_index=True), cols, penalty)
        pbase = test.p_base.to_numpy(float)
        pcandidate = pbase.copy() if not cols else probability(test, cols, beta)
        result = test[["game_id", "game_date", "source_season", "home_team", "away_team", "home_win",
                       "p_base", "home_goalie_id", "away_goalie_id", "home_goalie_name", "away_goalie_name",
                       "home_prior_start_dates", "away_prior_start_dates", "home_pred_sv", "away_pred_sv",
                       "home_pred_gaa", "away_pred_gaa", "sv_gap_pp", "gaa_gap"]].copy()
        result["p_candidate"] = pcandidate
        result["p_away_base"] = 1 - pbase
        result["p_away_candidate"] = 1 - pcandidate
        result["selected_family"] = family
        result["selected_penalty"] = penalty
        result["selected_sv_rating"] = sv_cfg["rating_col"]
        result["selected_sv_alpha"] = sv_cfg["alpha"]
        result["selected_gaa_rating"] = gaa_cfg["rating_col"]
        result["selected_gaa_alpha"] = gaa_cfg["alpha"]
        preds.append(result)
        folds[str(target)] = {
            "training_seasons": sorted(int(x) for x in train.source_season.unique()),
            "validation_season": validation_season, "test_season": target, "games": int(len(test)),
            "pool_sv_rating_selected_on_validation": sv_cfg,
            "pool_sv_test_metrics": pool_sv_metrics,
            "pool_gaa_rating_selected_on_validation": gaa_cfg,
            "pool_gaa_test_metrics": pool_gaa_metrics,
            "win_adjustment_family": family, "win_adjustment_penalty": penalty,
            "win_adjustment_coefficients": {c: float(b) for c, b in zip(cols, beta)},
            "validation_baseline": baseline_val,
            "validation_selected_log_loss": float(selected["validation_log_loss"]),
            "test_baseline": win_metrics(test, pbase),
            "test_candidate": win_metrics(test, pcandidate),
            "mean_absolute_probability_shift_pp": float(100 * np.mean(np.abs(pcandidate - pbase))),
            "mean_signed_probability_shift_pp": float(100 * np.mean(pcandidate - pbase)),
        }
    pred = pd.concat(preds, ignore_index=True).sort_values(["game_date", "game_id"]).reset_index(drop=True)
    summary = {
        "experiment_id": EXPERIMENT_ID, "model_id": MODEL_ID, "status": "RESEARCH; no promotion or public export",
        "question": "Do the named starters' individual prior save percentage and goals-against rate improve HWIN win probabilities?",
        "baseline": "Frozen Hockey Lab Game v1.0.0 Team Strength 004 + B2B out-of-fold probabilities",
        "target_seasons": TARGET_SEASONS, "target_games": int(len(pred)),
        "goalie_pool_diagnostics": goalie_diag,
        "pool_calibration_method": "On the immediately preceding season, select a prior-only career/last-20/last-40 rate with pseudo-exposure shrinkage and a nonnegative calibration slope; test it on the next season.",
        "win_method": "For each target season S, fit the goalie logit adjustment through S-2, select feature family and ridge strength on S-1, refit through S-1, and test on S. Baseline wins ties.",
        "individual_goalie_features": "Calibrated prior-only save percentage difference and calibrated prior-only goals-against-per-60 difference between the exact home and away starters.",
        "starter_identity_caveat": "Target starter IDs are taken from postgame goalie boxes. Skill histories and league baselines use only dates strictly before the target game, but starter selection itself remains retrospective pending timestamped pregame snapshots.",
        "team_record_excluded": True, "market_inputs": [], "production_or_site_changes": False,
        "folds": folds, "pooled_baseline": win_metrics(pred, pred.p_base.to_numpy(float)),
        "pooled_candidate": win_metrics(pred, pred.p_candidate.to_numpy(float)),
        "pooled_mean_absolute_probability_shift_pp": float(100 * np.mean(np.abs(pred.p_candidate - pred.p_base))),
        "pooled_mean_signed_probability_shift_pp": float(100 * np.mean(pred.p_candidate - pred.p_base)),
        "paired_uncertainty": paired_ci(pred),
    }
    pred.to_parquet(out / "oof_goalie_sv_gaa_win_pct.parquet", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--goalie", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.base, args.goalie, args.spine, args.outdir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
