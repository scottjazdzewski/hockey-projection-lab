"""Pregame goalie-probability layer for the NHL score-distribution model.

Experiment 018 uses goalie starter-choice probabilities from experiment 016,
calibrated only on older OOF seasons using experiment 017's temperature plus
unknown-goalie intercept. It tests prior goalie skill and workload signals,
probability-weighted across candidate starters. Baseline goal rates are offsets;
goalie coefficients are fitted on old seasons, checked on the preceding season,
then evaluated forward. The official Game v1.0.0 win model is not touched.
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hockey_goals_team_form_60_009 as goals


VARIANT = "hockey_process_plus_league_scoring_validation_selected_distribution"
TARGET_SOURCE_SEASONS = [2025, 2026]
GOALIE_L2_PENALTY = 2.0
BOOTSTRAP_REPLICATES = 2000
RNG_SEED = 1803
GOALIE_FEATURE_SETS = {
    "career_gsax": ["quality_career_gsax30"],
    "recent_gsax_r20": ["quality_gsax_r20"],
    "established_gsax_r40": ["quality_gsax_r40"],
    "fivev5_gsax_r20": ["quality_fivev5_gsax_r20"],
    "save_pct_r20": ["save_pct_r20_delta"],
    "recent_vs_established": ["quality_recent10_vs_r40"],
    "workload_rest": ["days_since_start", "starts_7d", "starts_14d"],
    "quality_plus_workload": ["quality_gsax_r40", "quality_recent10_vs_r40",
                              "days_since_start", "starts_7d", "starts_14d"],
}


def grouped_softmax(frame: pd.DataFrame, temperature: float, unknown_shift: float) -> np.ndarray:
    frame = frame.reset_index(drop=True)
    z = frame.logit.to_numpy(float) / temperature
    z += frame.candidate.eq("__UNKNOWN__").to_numpy(float) * unknown_shift
    ids = frame.choice_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(frame)])
    maxima = np.maximum.reduceat(z, starts)
    sums = np.add.reduceat(np.exp(z - np.repeat(maxima, sizes)), starts)
    return np.exp(z - np.repeat(maxima + np.log(sums), sizes))


def calibrate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply rolling full-season calibration to each later candidate season."""
    out = []
    for season in sorted(candidates.season.unique()):
        test = candidates.loc[candidates.season.eq(season)].copy().reset_index(drop=True)
        earlier = candidates.loc[candidates.season.lt(season)].copy().reset_index(drop=True)
        if earlier.empty:
            test["calibrated_probability"] = test.model_probability.astype(float)
            test["calibration_temperature"] = 1.0
            test["calibration_unknown_shift"] = 0.0
            out.append(test)
            continue
        ids = earlier.choice_id.to_numpy()
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
        sizes = np.diff(np.r_[starts, len(earlier)])
        labels = earlier.label.to_numpy(float)
        logits = earlier.logit.to_numpy(float)
        unknown = earlier.candidate.eq("__UNKNOWN__").to_numpy(float)
        n_groups = earlier.choice_id.nunique()

        def objective(theta: np.ndarray) -> float:
            temp = float(np.exp(theta[0]))
            shift = float(theta[1])
            z = logits / temp + shift * unknown
            maxima = np.maximum.reduceat(z, starts)
            sums = np.add.reduceat(np.exp(z - np.repeat(maxima, sizes)), starts)
            log_norm = np.repeat(maxima + np.log(sums), sizes)
            return float(-np.sum(labels * (z - log_norm)) / n_groups)

        fit = minimize(objective, np.zeros(2), method="L-BFGS-B", bounds=[(-1, 1), (-5, 5)])
        temperature, shift = float(np.exp(fit.x[0])), float(fit.x[1])
        test["calibrated_probability"] = grouped_softmax(test, temperature, shift)
        test["calibration_temperature"] = temperature
        test["calibration_unknown_shift"] = shift
        out.append(test)
    return pd.concat(out, ignore_index=True)


def prior_goalie_features(goalies: pd.DataFrame, spine: pd.DataFrame,
                          candidate_rows: pd.DataFrame) -> pd.DataFrame:
    """Attach pregame skill/workload features using starts strictly before date."""
    meta = spine[["game_id", "game_date"]].copy()
    meta["game_id"] = meta.game_id.astype(str)
    meta["game_date"] = pd.to_datetime(meta.game_date, errors="coerce").dt.normalize()
    starts = goalies.loc[goalies.starter_bool.fillna(False).astype(bool)].copy()
    starts = starts.drop(columns=["game_date"], errors="ignore")
    starts["game_id"] = starts.game_id.astype(str)
    starts = starts.merge(meta, on="game_id", how="inner", validate="many_to_one")
    starts["goalie_id_key"] = starts.goalie_id_key.astype(str)
    starts["shots_against_pbp"] = pd.to_numeric(starts.shots_against_pbp, errors="coerce").clip(lower=0).fillna(0)
    starts["saves_pbp"] = pd.to_numeric(starts.saves_pbp, errors="coerce").clip(lower=0).fillna(0)
    starts["gsax"] = pd.to_numeric(starts.gsax, errors="coerce").fillna(0)
    starts["fivev5_shots_against"] = pd.to_numeric(starts.fivev5_shots_against, errors="coerce").clip(lower=0).fillna(0)
    starts["fivev5_gsax"] = pd.to_numeric(starts.fivev5_gsax, errors="coerce").fillna(0)
    daily = starts.groupby(["goalie_id_key", "game_date"], as_index=False, sort=True).agg(
        shots=("shots_against_pbp", "sum"), saves=("saves_pbp", "sum"),
        gsax=("gsax", "sum"), fivev5_shots=("fivev5_shots_against", "sum"),
        fivev5_gsax=("fivev5_gsax", "sum"))
    daily = daily.sort_values(["goalie_id_key", "game_date"], kind="stable")
    daily["gsax_per30"] = np.where(daily.shots.gt(0), 30.0*daily.gsax/daily.shots, 0.0)
    daily["fivev5_gsax_per30"] = np.where(daily.fivev5_shots.gt(0), 30.0*daily.fivev5_gsax/daily.fivev5_shots, 0.0)
    daily["save_pct"] = np.where(daily.shots.gt(0), daily.saves/daily.shots, 0.90)
    history_parts = []
    start_dates_by_goalie = {}
    for goalie_id, g in daily.groupby("goalie_id_key", sort=False):
        g = g.sort_values("game_date", kind="stable").copy()
        dates = g.game_date.to_numpy(dtype="datetime64[ns]")
        g["quality_career_gsax30"] = 30.0*(g.gsax.cumsum())/(g.shots.cumsum()+100.0)
        g["quality_gsax_r20"] = g.gsax_per30.rolling(20, min_periods=5).mean()
        g["quality_gsax_r40"] = g.gsax_per30.rolling(40, min_periods=8).mean()
        g["quality_fivev5_gsax_r20"] = g.fivev5_gsax_per30.rolling(20, min_periods=5).mean()
        g["save_pct_r20_delta"] = g.save_pct.rolling(20, min_periods=5).mean() - 0.90
        g["quality_recent10_vs_r40"] = (
            g.gsax_per30.rolling(10, min_periods=5).mean()
            - g.gsax_per30.rolling(40, min_periods=8).mean()
        )
        g["history_date"] = g.game_date
        history_parts.append(g[["goalie_id_key", "history_date", "quality_career_gsax30",
                                "quality_gsax_r20", "quality_gsax_r40",
                                "quality_fivev5_gsax_r20", "save_pct_r20_delta",
                                "quality_recent10_vs_r40"]])
        start_dates_by_goalie[str(goalie_id)] = dates
    history = pd.concat(history_parts, ignore_index=True)

    req = candidate_rows.copy().reset_index(drop=True)
    req["candidate"] = req.candidate.astype(str)
    req["game_date"] = pd.to_datetime(req.game_date, errors="coerce").dt.normalize()
    known = req.loc[req.candidate.ne("__UNKNOWN__")].copy()
    known["goalie_id_key"] = known.candidate
    known = known.sort_values(["game_date", "goalie_id_key"], kind="stable")
    history = history.sort_values(["history_date", "goalie_id_key"], kind="stable")
    matched = pd.merge_asof(
        known, history, by="goalie_id_key", left_on="game_date", right_on="history_date",
        direction="backward", allow_exact_matches=False,
    )
    feature_names = ["quality_career_gsax30", "quality_gsax_r20", "quality_gsax_r40",
                     "quality_fivev5_gsax_r20", "save_pct_r20_delta",
                     "quality_recent10_vs_r40"]
    for feature in feature_names:
        known[feature] = pd.to_numeric(matched[feature], errors="coerce").fillna(0.0).to_numpy(float)
    known["days_since_start"] = (known.game_date - matched.history_date).dt.days.fillna(4).clip(lower=0, upper=30)
    workload = []
    for r in known[["goalie_id_key", "game_date"]].itertuples(index=False):
        dates = start_dates_by_goalie.get(str(r.goalie_id_key), np.array([], dtype="datetime64[ns]"))
        d = np.datetime64(pd.Timestamp(r.game_date).to_datetime64())
        hi = int(np.searchsorted(dates, d, side="left"))
        lo7 = int(np.searchsorted(dates, d-np.timedelta64(7, "D"), side="left"))
        lo14 = int(np.searchsorted(dates, d-np.timedelta64(14, "D"), side="left"))
        workload.append((hi-lo7, hi-lo14))
    known[["starts_7d", "starts_14d"]] = np.asarray(workload, dtype=float)
    unknown = req.loc[req.candidate.eq("__UNKNOWN__")].copy()
    for feature in feature_names:
        unknown[feature] = 0.0
    unknown["days_since_start"] = 4.0
    unknown["starts_7d"] = 0.0
    unknown["starts_14d"] = 0.0
    result = pd.concat([known, unknown], ignore_index=True)
    result = result.sort_values(["season", "choice_id", "candidate"], kind="stable").reset_index(drop=True)
    result["quality_source"] = "prior_starter_history;strict_date_lt"
    return result


def load_baseline(zip_path: str) -> pd.DataFrame:
    inner = "research/exports/GOALS_TEAM_FORM_60_009_VALIDATION/oof_predictions.parquet"
    with zipfile.ZipFile(zip_path) as zf:
        data = pd.read_parquet(BytesIO(zf.read(inner)))
    data = data.loc[data.model_variant.eq(VARIANT) & data.game_type.eq("R")].copy()
    data["game_id"] = data.game_id.astype(str)
    return data.sort_values(["source_season", "game_id"], kind="stable").reset_index(drop=True)


def make_game_quality(candidates: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    meta = scores[["game_id", "home_team", "away_team"]].drop_duplicates("game_id")
    x = candidates.merge(meta, on="game_id", how="inner", validate="many_to_one")
    x["side_match"] = np.where(x.team.eq(x.home_team), "home",
                        np.where(x.team.eq(x.away_team), "away", "mismatch"))
    if x.side_match.eq("mismatch").any():
        raise ValueError("Goalie candidate team does not match the canonical game teams")
    quality_features = sorted({feature for features in GOALIE_FEATURE_SETS.values() for feature in features})
    agg_spec = {
        "probability_sum": ("calibrated_probability", "sum"),
        "candidate_count": ("candidate", "size"),
        "max_candidate_probability": ("calibrated_probability", "max"),
    }
    for feature in quality_features:
        weighted = f"weighted__{feature}"
        x[weighted] = x.calibrated_probability * pd.to_numeric(x[feature], errors="coerce").fillna(0)
        agg_spec[feature] = (weighted, "sum")
    team_q = x.groupby(["game_id", "team"], as_index=False, sort=False).agg(**agg_spec)
    team_q = team_q.merge(meta.melt(id_vars="game_id", value_vars=["home_team", "away_team"],
                                    var_name="home_away", value_name="team"),
                          on=["game_id", "team"], how="left", validate="one_to_one")
    team_q["home_away"] = team_q.home_away.map({"home_team": "home", "away_team": "away"})
    if team_q.home_away.isna().any() or not np.allclose(team_q.probability_sum, 1.0, atol=1e-8):
        raise ValueError("Missing game-side goalie probabilities or probability totals differ from one")
    quality = None
    for feature in quality_features:
        pivot = team_q.pivot(index="game_id", columns="home_away", values=feature).reset_index()
        pivot = pivot.rename(columns={"home": f"home_{feature}", "away": f"away_{feature}"})
        quality = pivot if quality is None else quality.merge(pivot, on="game_id", validate="one_to_one")
    diagnostics = team_q.groupby("home_away").agg(
        mean_top_probability=("max_candidate_probability", "mean"),
        mean_candidate_count=("candidate_count", "mean"),
    ).to_dict(orient="index")
    return quality, diagnostics


def side_rows(games: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    home = games[["game_id", "source_season", "reg_home_goals", "rate_mu_home"]].copy()
    home = home.rename(columns={"reg_home_goals": "goals", "rate_mu_home": "base_mu"})
    for feature in features:
        home[feature] = games[f"away_{feature}"].to_numpy(float)
    home["side"] = "home"
    away = games[["game_id", "source_season", "reg_away_goals", "rate_mu_away"]].copy()
    away = away.rename(columns={"reg_away_goals": "goals", "rate_mu_away": "base_mu"})
    for feature in features:
        away[feature] = games[f"home_{feature}"].to_numpy(float)
    away["side"] = "away"
    return pd.concat([home, away], ignore_index=True)


def poisson_nll(rows: pd.DataFrame, features: list[str], model: dict | None) -> float:
    log_rate = np.log(np.clip(rows.base_mu.to_numpy(float), 1e-5, 20))
    if model is not None:
        z = (rows[features].to_numpy(float) - model["mean"]) / model["scale"]
        log_rate = log_rate + z @ model["beta"]
    mu = np.clip(np.exp(np.clip(log_rate, -8, 4)), 1e-5, 20)
    y = rows.goals.to_numpy(float)
    return float(np.mean(mu - y * np.log(mu) + goals.gammaln(y + 1)))


def fit_model(rows: pd.DataFrame, features: list[str]) -> dict:
    x = rows[features].to_numpy(float)
    mean = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    x = np.nan_to_num((x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    offset = np.log(np.clip(rows.base_mu.to_numpy(float), 1e-5, 20))
    y = rows.goals.to_numpy(float)
    n = max(len(y), 1)

    def objective(beta: np.ndarray) -> float:
        log_rate = np.clip(offset + x @ beta, -8, 4)
        mu = np.exp(log_rate)
        nll = np.sum(mu - y*log_rate + goals.gammaln(y+1))
        return float(nll + 0.5*GOALIE_L2_PENALTY*np.dot(beta, beta))

    result = minimize(objective, np.zeros(len(features)), method="L-BFGS-B", options={"maxiter":500,"ftol":1e-10})
    return {"features": features, "mean": mean, "scale": scale, "beta": result.x,
            "converged": bool(result.success), "train_nll": float(result.fun), "n_train_sides": int(n)}


def score_predictions(games: pd.DataFrame, model: dict | None, variant: str) -> pd.DataFrame:
    out = []
    for r in games.itertuples(index=False):
        adjust_h = adjust_a = 0.0
        if model is not None:
            features = model["features"]
            hvec = np.array([getattr(r, f"away_{f}") for f in features], dtype=float)
            avec = np.array([getattr(r, f"home_{f}") for f in features], dtype=float)
            adjust_h = float(((hvec-model["mean"])/model["scale"]) @ model["beta"])
            adjust_a = float(((avec-model["mean"])/model["scale"]) @ model["beta"])
        mu_h = float(np.clip(r.rate_mu_home * np.exp(np.clip(adjust_h,-3,3)), 1e-4, 20.0))
        mu_a = float(np.clip(r.rate_mu_away * np.exp(np.clip(adjust_a,-3,3)), 1e-4, 20.0))
        matrix = goals.joint_score_pmf(mu_h, mu_a, r.distribution_family, r.distribution_parameter)
        ph, pa, pt = goals.outcome_probs(matrix)
        final = goals.final_score_pmf(matrix, r.p_home_ot_winner)
        fwin_h = ph + pt * r.p_home_ot_winner
        fh, fa = int(r.official_final_home_goals), int(r.official_final_away_goals)
        rh, ra = int(r.reg_home_goals), int(r.reg_away_goals)
        p_full = max(float(final[fh, fa]), 1e-15)
        onehot_class = 2 if rh > ra else (0 if rh < ra else 1)
        p_class = [pa, pt, ph]
        p_true = max(p_class[onehot_class], 1e-15)
        out.append({
            "game_id": r.game_id, "source_season": int(r.source_season), "model_variant": variant,
            "reg_home_goals": rh, "reg_away_goals": ra,
            "official_final_home_goals": fh, "official_final_away_goals": fa,
            "went_ot": int(r.went_ot), "base_mu_home": float(r.rate_mu_home), "base_mu_away": float(r.rate_mu_away),
            "mu_home": mu_h, "mu_away": mu_a,
            "home_goalie_adjustment_log_rate": adjust_h, "away_goalie_adjustment_log_rate": adjust_a,
            "exp_home_goals": float(sum(i * matrix[i, :].sum() for i in range(matrix.shape[0]))),
            "exp_away_goals": float(sum(j * matrix[:, j].sum() for j in range(matrix.shape[1]))),
            "p_reg_home": ph, "p_reg_away": pa, "p_reg_tie": pt, "p_overtime": pt,
            "reg_score_logloss": float(-np.log(max(matrix[rh, ra], 1e-15))),
            "reg_outcome_logloss": float(-np.log(p_true)),
            "reg_outcome_brier": float(sum((p_class[k] - float(k == onehot_class)) ** 2 for k in range(3))),
            "overtime_logloss": float(-np.log(max(pt if r.went_ot else 1-pt, 1e-15))),
            "overtime_brier": float((pt - r.went_ot) ** 2),
            "p_full_home": fwin_h,
            "full_score_logloss": float(-np.log(p_full)),
            "home_full_win": int(fh > fa),
            "full_home_brier": float((fwin_h - int(fh > fa)) ** 2),
        })
    return pd.DataFrame(out)


def calibrate_binary(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    y = np.asarray(y, float)
    x = np.column_stack([np.ones(len(p)), logit(p)])
    def loss(b):
        z = np.clip(x @ b, -30, 30)
        return float(np.mean(np.logaddexp(0, z) - y*z))
    fit = minimize(loss, np.array([0.0, 1.0]), method="BFGS")
    return {"intercept": float(fit.x[0]), "slope": float(fit.x[1]), "converged": bool(fit.success)}


def summarize(rows: pd.DataFrame) -> dict:
    return {
        "games": int(len(rows)),
        "home_goal_mae": float(np.mean(np.abs(rows.exp_home_goals-rows.reg_home_goals))),
        "away_goal_mae": float(np.mean(np.abs(rows.exp_away_goals-rows.reg_away_goals))),
        "mean_home_goal_error_pred_minus_actual": float(np.mean(rows.exp_home_goals-rows.reg_home_goals)),
        "mean_away_goal_error_pred_minus_actual": float(np.mean(rows.exp_away_goals-rows.reg_away_goals)),
        "regulation_score_logloss": float(rows.reg_score_logloss.mean()),
        "regulation_outcome_logloss": float(rows.reg_outcome_logloss.mean()),
        "regulation_outcome_brier": float(rows.reg_outcome_brier.mean()),
        "overtime_logloss": float(rows.overtime_logloss.mean()),
        "overtime_brier": float(rows.overtime_brier.mean()),
        "overtime_predicted_rate": float(rows.p_overtime.mean()),
        "overtime_observed_rate": float(rows.went_ot.mean()),
        "regulation_home_calibration": calibrate_binary(
            rows.loc[rows.reg_home_goals.ne(rows.reg_away_goals), "reg_home_goals"].to_numpy() >
            rows.loc[rows.reg_home_goals.ne(rows.reg_away_goals), "reg_away_goals"].to_numpy(),
            rows.loc[rows.reg_home_goals.ne(rows.reg_away_goals), "p_reg_home"].to_numpy() /
            np.clip(1-rows.loc[rows.reg_home_goals.ne(rows.reg_away_goals), "p_reg_tie"].to_numpy(), 1e-8, 1),
        ),
        "full_score_logloss": float(rows.full_score_logloss.mean()),
        "full_game_home_win_brier": float(rows.full_home_brier.mean()),
        "full_game_home_win_calibration": calibrate_binary(rows.home_full_win.to_numpy(), rows.p_full_home.to_numpy()),
    }


def paired_bootstrap(base: pd.DataFrame, candidate: pd.DataFrame, metric: str) -> list[float]:
    d = base[["game_id", metric]].merge(candidate[["game_id", metric]], on="game_id", validate="one_to_one", suffixes=("_base", "_candidate"))
    delta = d[f"{metric}_candidate"].to_numpy() - d[f"{metric}_base"].to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))
    vals = delta[draws].mean(axis=1)
    return [float(x) for x in np.quantile(vals, [0.025, 0.975])]


def main(baseline_zip: str, candidates_path: str, games_path: str,
         goalies_path: str, spine_path: str, outdir: str) -> dict:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(baseline_zip)
    # The candidate-choice OOF file begins at canonical game season 2022, which
    # aligns by game_id with goals source_season 2023; restrict both layers to
    # their complete overlap rather than imputing starter projections backward.
    baseline = baseline.loc[baseline.source_season.between(2023, 2026)].copy()
    c = pd.read_parquet(candidates_path)
    c["game_id"] = c.game_id.astype(str)
    games = pd.read_parquet(games_path)
    games["game_id"] = games.game_id.astype(str)
    c = calibrate_candidates(c)
    c = c.merge(games[["game_id", "game_date"]].drop_duplicates("game_id"), on="game_id", how="left", validate="many_to_one")
    c["game_date"] = pd.to_datetime(c.game_date, errors="coerce")
    goalies = pd.read_parquet(goalies_path)
    spine = pd.read_parquet(spine_path)
    c = prior_goalie_features(goalies, spine, c)
    c = c.loc[c.game_id.isin(baseline.game_id)].copy()
    quality, projection_diag = make_game_quality(c, baseline)
    full = baseline.merge(quality, on="game_id", how="inner", validate="one_to_one")
    quality_features = sorted({feature for features in GOALIE_FEATURE_SETS.values() for feature in features})
    for feature in quality_features:
        full[f"home_{feature}"] = pd.to_numeric(full[f"home_{feature}"], errors="coerce").fillna(0)
        full[f"away_{feature}"] = pd.to_numeric(full[f"away_{feature}"], errors="coerce").fillna(0)
    if len(full) != len(baseline):
        raise ValueError(f"Starter candidates cover {len(full)}/{len(baseline)} baseline games")

    # Reproduction guard: the selected distribution and stored means must
    # reproduce the canonical baseline score log loss before any adjustment.
    checks = score_predictions(full.head(30), model=None, variant="recomputed_base")
    expected = full.head(30).logp_reg_score.to_numpy(float)
    actual = -checks.reg_score_logloss.to_numpy(float)
    if not np.allclose(actual, expected, atol=2e-6):
        raise ValueError("Recomputed no-goalie score PMF differs from frozen baseline OOF probabilities")

    # Precompute model outcomes for the unadjusted model and retrospective
    # actual-starter identity diagnostic. Actual same-game identity is never
    # used in the candidate model or in coefficient fitting.
    actual = goalies.loc[goalies.starter_bool.fillna(False).astype(bool),
                         ["game_id", "team_abbrev", "goalie_id_key"]].copy()
    actual["game_id"] = actual.game_id.astype(str)
    actual["team_abbrev"] = actual.team_abbrev.astype(str)
    actual["goalie_id_key"] = actual.goalie_id_key.astype(str)
    # The actual goalie quality is derived with the same strict prior-date state.
    actual_req = actual.rename(columns={"goalie_id_key": "candidate", "team_abbrev": "team"}).copy()
    meta = spine[["game_id", "game_date"]].copy(); meta["game_id"] = meta.game_id.astype(str)
    actual_req = actual_req.merge(meta, on="game_id", how="inner", validate="many_to_one")
    actual_req["season"] = 0; actual_req["choice_id"] = actual_req.game_id+":"+actual_req.team
    actual_req["candidate"] = actual_req.candidate.astype(str)
    actual_q = prior_goalie_features(goalies, spine, actual_req)
    hteams = full[["game_id", "home_team", "away_team"]].drop_duplicates("game_id")
    actual_q = actual_q.merge(hteams.melt(id_vars="game_id", value_vars=["home_team", "away_team"],
                                         var_name="side_name", value_name="team"),
                              on=["game_id", "team"], how="inner", validate="one_to_one")
    actual_q["side"] = actual_q.side_name.map({"home_team": "home", "away_team": "away"})
    actual_wide = None
    for feature in quality_features:
        pivot = actual_q.pivot(index="game_id", columns="side", values=feature).reset_index()
        pivot = pivot.rename(columns={"home": f"home_actual__{feature}", "away": f"away_actual__{feature}"})
        actual_wide = pivot if actual_wide is None else actual_wide.merge(pivot, on="game_id", validate="one_to_one")
    full = full.merge(actual_wide, on="game_id", how="left", validate="one_to_one")
    for feature in quality_features:
        full[f"home_actual__{feature}"] = pd.to_numeric(full[f"home_actual__{feature}"], errors="coerce").fillna(0)
        full[f"away_actual__{feature}"] = pd.to_numeric(full[f"away_actual__{feature}"], errors="coerce").fillna(0)

    # The actual-starter diagnostic uses prior skill only; same-game starter
    # identity is retrospective and explicitly prohibited from promotion.
    oracle_games = full.copy()
    for feature in quality_features:
        oracle_games[f"home_{feature}"] = oracle_games[f"home_actual__{feature}"]
        oracle_games[f"away_{feature}"] = oracle_games[f"away_actual__{feature}"]

    all_metrics = {}
    all_fold_rows = []
    all_prediction_rows = []
    for target in TARGET_SOURCE_SEASONS:
        fold = full.loc[full.source_season.eq(target)].copy()
        train = full.loc[full.source_season.le(target-2) & full.source_season.ge(2023)].copy()
        validation = full.loc[full.source_season.eq(target-1)].copy()
        if len(fold) < 1000 or len(validation) < 1000 or len(train) < 1000:
            continue
        base_validation = side_rows(validation, [])
        ll_base_val = poisson_nll(base_validation, [], None)
        candidate_fits = {}
        for feature_set, features in GOALIE_FEATURE_SETS.items():
            train_sides = side_rows(train, features)
            validation_sides = side_rows(validation, features)
            fitted = fit_model(train_sides, features)
            ll = poisson_nll(validation_sides, features, fitted)
            candidate_fits[feature_set] = {"features": features, "fit": fitted,
                                           "validation_count_logloss": ll,
                                           "validation_delta_vs_baseline": ll-ll_base_val}
        selected_feature_set = min(candidate_fits, key=lambda name: candidate_fits[name]["validation_count_logloss"])
        if candidate_fits[selected_feature_set]["validation_count_logloss"] >= ll_base_val:
            selected_feature_set = None
            fitted_model = None
        else:
            selected_features = candidate_fits[selected_feature_set]["features"]
            refit_sides = side_rows(pd.concat([train, validation], ignore_index=True), selected_features)
            fitted_model = fit_model(refit_sides, selected_features)

        eval_variants = {
            "no_goalie": (fold, None),
            "pregame_probability_weighted": (fold, fitted_model),
            "retrospective_actual_starter_diagnostic": (
                oracle_games.loc[oracle_games.source_season.eq(target)].copy(), fitted_model),
        }
        fold_results = {}
        outputs = {}
        for name, (data, beta) in eval_variants.items():
            prediction = score_predictions(data, beta, name)
            outputs[name] = prediction
            fold_results[name] = summarize(prediction)
            all_prediction_rows.extend(prediction.to_dict(orient="records"))

        # Ensure all variants use the same game set and report paired changes.
        base_pred = outputs["no_goalie"]
        deltas = {}
        for name in ["pregame_probability_weighted", "retrospective_actual_starter_diagnostic"]:
            cand = outputs[name]
            deltas[name] = {
                "delta_score_logloss_candidate_minus_base": float(cand.reg_score_logloss.mean()-base_pred.reg_score_logloss.mean()),
                "paired_game_bootstrap_95pct_score_logloss_delta": paired_bootstrap(base_pred, cand, "reg_score_logloss"),
                "delta_home_goal_mae": float(np.mean(np.abs(cand.exp_home_goals-cand.reg_home_goals))-np.mean(np.abs(base_pred.exp_home_goals-base_pred.reg_home_goals))),
                "delta_away_goal_mae": float(np.mean(np.abs(cand.exp_away_goals-cand.reg_away_goals))-np.mean(np.abs(base_pred.exp_away_goals-base_pred.reg_away_goals))),
                "delta_regulation_brier": float(cand.reg_outcome_brier.mean()-base_pred.reg_outcome_brier.mean()),
                "delta_overtime_brier": float(cand.overtime_brier.mean()-base_pred.overtime_brier.mean()),
            }
        all_metrics[str(target)] = {
            "train_source_seasons": sorted(int(s) for s in train.source_season.unique()),
            "validation_source_season": int(target-1), "test_source_season": int(target),
            "beta_penalty": GOALIE_L2_PENALTY,
            "validation_count_logloss_no_goalie": ll_base_val,
            "validation_by_feature_set": {
                name: {"features": value["features"],
                       "beta_standardized": value["fit"]["beta"].tolist(),
                       "count_logloss": value["validation_count_logloss"],
                       "delta_vs_baseline": value["validation_delta_vs_baseline"]}
                for name, value in candidate_fits.items()},
            "selected_feature_set": selected_feature_set,
            "decision": "refit validation-selected goalie features on train+validation" if fitted_model is not None else "hold all goalie coefficients at zero; validation did not improve",
            "test_model": None if fitted_model is None else {
                "features": fitted_model["features"], "beta_standardized": fitted_model["beta"].tolist(),
                "train_converged": fitted_model["converged"]},
            "test_metrics": fold_results,
            "paired_test_deltas": deltas,
            "projection_diagnostics": projection_diag,
        }
        all_fold_rows.extend([{ "target_source_season": target, "model_variant": k, **v}
                              for k, v in fold_results.items()])

    predictions = pd.DataFrame(all_prediction_rows)
    predictions.to_parquet(outdir / "oof_predictions.parquet", index=False)
    folds = pd.DataFrame(all_fold_rows)
    folds.to_json(outdir / "fold_metrics.json", orient="records", indent=2)
    summary = {
        "experiment_id": "HOCKEY-LAB-GOALS-PREDICTED-GOALIE-018",
        "status": "RESEARCH; no production export; Game v1.0.0 frozen",
        "input_score_model": VARIANT,
        "input_starter_model": "HOCKEY-LAB-GOALIE-STARTER-PROBABILITY-016 with rolling calibration-017",
        "target_source_seasons": TARGET_SOURCE_SEASONS,
        "overlap_note": "Source season 2022 is excluded because starter-choice OOF begins at game_id season 2022 (goals source season 2023); evaluation uses the complete game_id-aligned overlap 2023-2026.",
        "design": "OOF goals baseline; fit one shrinkage-controlled log-goal-rate coefficient on earlier seasons; require count-logloss improvement on immediately prior validation season before refitting and scoring the next season",
        "goalie_predictor": "starter-probability-weighted features for career/r20/r40 GSAx per 30, recent-vs-established GSAx, 5v5 goalie GSAx, and starter workload/rest; unknown alternative carries neutral skill and four rest days",
        "coefficient": "Poisson log-rate adjustment with frozen baseline log rate as offset; standardized goalie covariates; ridge penalty=2; selected on prior validation only",
        "quality_cutoff": "goalie performance state and workload use starts strictly before target date; same-date results excluded",
        "starter_identity": "candidate probabilities are pregame projections from prior team starts; same-game starter identity is label/diagnostic only",
        "distribution": "hold each OOF baseline game's validation-selected score-family and parameter fixed; recompute full score PMF from adjusted score rates",
        "paired_bootstrap": {"unit": "game", "replicates": BOOTSTRAP_REPLICATES, "seed": RNG_SEED},
        "calibration_by_target_source_season": all_metrics,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-zip", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--games", required=True)
    ap.add_argument("--goalies", required=True)
    ap.add_argument("--spine", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    main(a.baseline_zip, a.candidates, a.games, a.goalies, a.spine, a.outdir)
