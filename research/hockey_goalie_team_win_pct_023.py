"""Test whether team win history with the named goalie adds pregame value.

Uses the frozen Goals 009 score-distribution model as the base probability.
For each team-goalie pairing, the candidate feature is the strictly prior
team win residual (actual team wins minus the base model's expected wins),
shrunk by a prior-start equivalent. Raw win percentage is retained for
diagnostics, but is not used unadjusted as a skill estimate because it is
confounded by team strength. Target-game starter IDs are retrospective, so
this remains an upper-bound research test until timestamped pregame starter
snapshots are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit


MODEL_ID = "HOCKEY-LAB-GOALIE-TEAM-WIN-PCT-023-RESEARCH"
GOAL_VARIANT = "hockey_process_plus_league_scoring_validation_selected_distribution"
TARGET_SEASONS = [2024, 2025, 2026]
PRIOR_START_EQUIV = [5, 15, 30, 60]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 2307


def norm_ids(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True)


def load_game_ledger(goals_path: str, starter_path: str) -> pd.DataFrame:
    goals = pd.read_parquet(goals_path)
    goals = goals.loc[
        goals.model_variant.eq(GOAL_VARIANT) & goals.game_type.eq("R"),
        [
            "game_id", "game_date", "source_season", "home_team", "away_team",
            "official_final_home_goals", "official_final_away_goals", "p_full_home_win",
        ],
    ].copy()
    goals["game_id"] = norm_ids(goals.game_id)
    goals["game_date"] = pd.to_datetime(goals.game_date, errors="coerce").dt.normalize()
    goals["source_season"] = goals.source_season.astype(int)
    goals["y_home_win"] = (
        goals.official_final_home_goals > goals.official_final_away_goals
    ).astype(float)
    goals["p_home_base"] = goals.p_full_home_win.astype(float)

    starters = pd.read_parquet(starter_path)
    starters = starters.loc[
        starters.starter_bool.fillna(False).astype(bool),
        ["game_id", "goalie_id_key", "home_away", "team_abbrev", "player_name"],
    ].copy()
    starters["game_id"] = norm_ids(starters.game_id)
    starters["goalie_id_key"] = starters.goalie_id_key.astype("string")
    starters["home_away"] = starters.home_away.astype("string")
    if starters.duplicated(["game_id", "home_away"]).any():
        raise ValueError("Multiple actual starters on one game side")

    home = starters.loc[starters.home_away.eq("home")].rename(
        columns={
            "goalie_id_key": "home_goalie_id", "team_abbrev": "home_goalie_team",
            "player_name": "home_goalie_name",
        }
    )[["game_id", "home_goalie_id", "home_goalie_team", "home_goalie_name"]]
    away = starters.loc[starters.home_away.eq("away")].rename(
        columns={
            "goalie_id_key": "away_goalie_id", "team_abbrev": "away_goalie_team",
            "player_name": "away_goalie_name",
        }
    )[["game_id", "away_goalie_id", "away_goalie_team", "away_goalie_name"]]
    games = goals.merge(home, on="game_id", how="inner", validate="one_to_one")
    games = games.merge(away, on="game_id", how="inner", validate="one_to_one")
    if len(games) != 5 * 1312:
        raise ValueError(f"Expected 6,560 fully matched regular-season games; found {len(games)}")
    if not (games.home_team.astype(str) == games.home_goalie_team.astype(str)).all():
        raise ValueError("Home goalie team does not match game ledger")
    if not (games.away_team.astype(str) == games.away_goalie_team.astype(str)).all():
        raise ValueError("Away goalie team does not match game ledger")
    if games.duplicated(["home_team", "game_date"]).any() or games.duplicated(
        ["away_team", "game_date"]
    ).any():
        raise ValueError("Team has multiple regular-season games on one date; date-level lag required")
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def build_goalie_team_history(games: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame({
        "game_id": games.game_id,
        "game_date": games.game_date,
        "source_season": games.source_season,
        "team": games.home_team.astype("string"),
        "goalie_id": games.home_goalie_id.astype("string"),
        "goalie_name": games.home_goalie_name.astype("string"),
        "team_win": games.y_home_win.astype(float),
        "team_expected_win": games.p_home_base.astype(float),
        "side": "home",
    })
    away = pd.DataFrame({
        "game_id": games.game_id,
        "game_date": games.game_date,
        "source_season": games.source_season,
        "team": games.away_team.astype("string"),
        "goalie_id": games.away_goalie_id.astype("string"),
        "goalie_name": games.away_goalie_name.astype("string"),
        "team_win": 1.0 - games.y_home_win.astype(float),
        "team_expected_win": 1.0 - games.p_home_base.astype(float),
        "side": "away",
    })
    long = pd.concat([home, away], ignore_index=True).sort_values(
        ["game_date", "game_id", "side"]
    ).reset_index(drop=True)
    long["win_residual"] = long.team_win - long.team_expected_win

    # Each current team's feature uses only its history with the exact named
    # goalie. Shift all cumulative/windowed terms by one game before merging.
    pieces: list[pd.DataFrame] = []
    for _, group in long.groupby(["team", "goalie_id"], sort=False, dropna=False):
        x = group.sort_values(["game_date", "game_id"]).copy()
        x["prior_start_count"] = np.arange(len(x), dtype=float)
        x["prior_team_wins"] = x.team_win.cumsum().shift(1).fillna(0.0)
        x["prior_expected_wins"] = x.team_expected_win.cumsum().shift(1).fillna(0.0)
        x["prior_win_residual"] = x.win_residual.cumsum().shift(1).fillna(0.0)
        for window in (5, 10, 20):
            x[f"prior_wins_r{window}"] = (
                x.team_win.rolling(window, min_periods=1).sum().shift(1).fillna(0.0)
            )
            x[f"prior_expected_r{window}"] = (
                x.team_expected_win.rolling(window, min_periods=1).sum().shift(1).fillna(0.0)
            )
            x[f"prior_residual_r{window}"] = (
                x.win_residual.rolling(window, min_periods=1).sum().shift(1).fillna(0.0)
            )
        for k in PRIOR_START_EQUIV:
            # Smoothed goalie-team win percentage relative to the base model's
            # expected win rate. Equivalent to a prior centered on zero excess.
            x[f"win_excess_career_k{k}"] = x.prior_win_residual / (x.prior_start_count + k)
            for window in (5, 10, 20):
                n = np.minimum(x.prior_start_count, window)
                x[f"win_excess_r{window}_k{k}"] = x[f"prior_residual_r{window}"] / (n + k)
        x["raw_prior_win_pct"] = np.divide(
            x.prior_team_wins, x.prior_start_count,
            out=np.full(len(x), np.nan, dtype=float), where=x.prior_start_count > 0,
        )
        x["prior_expected_win_pct"] = np.divide(
            x.prior_expected_wins, x.prior_start_count,
            out=np.full(len(x), np.nan, dtype=float), where=x.prior_start_count > 0,
        )
        pieces.append(x)
    features = pd.concat(pieces, ignore_index=True)
    if (features.prior_start_count.gt(0) & features.game_date.isna()).any():
        raise ValueError("Missing game date in goalie-team history")
    # Within a pairing, the current date must not appear in any prior feature.
    for _, group in features.groupby(["team", "goalie_id"], sort=False):
        ordered = group.sort_values(["game_date", "game_id"])
        if not np.array_equal(ordered.prior_start_count.to_numpy(), np.arange(len(ordered))):
            raise ValueError("Goaltender-team history ordering check failed")
    return features


def select_features(frame: pd.DataFrame, family: str) -> dict[str, np.ndarray]:
    if family == "career":
        return {
            f"career_k{k}": (
                frame[f"home_win_excess_career_k{k}"].to_numpy(float)
                - frame[f"away_win_excess_career_k{k}"].to_numpy(float)
            )
            for k in PRIOR_START_EQUIV
        }
    return {
        f"r{window}_k{k}": (
            frame[f"home_win_excess_r{window}_k{k}"].to_numpy(float)
            - frame[f"away_win_excess_r{window}_k{k}"].to_numpy(float)
        )
        for window in (5, 10, 20) for k in PRIOR_START_EQUIV
    }


def fit_beta(base_p: np.ndarray, x: np.ndarray, y: np.ndarray, alpha: float) -> float:
    z = logit(np.clip(base_p, 1e-6, 1 - 1e-6))

    def objective(beta: float) -> float:
        p = np.clip(expit(z + beta * x), 1e-12, 1 - 1e-12)
        nll = -np.sum(y * np.log(p) + (1 - y) * np.log1p(-p))
        return float(nll + 0.5 * alpha * beta * beta)

    result = minimize_scalar(objective, bounds=(-8.0, 8.0), method="bounded")
    candidates = [(float(result.fun), float(result.x)), (objective(0.0), 0.0)]
    return min(candidates, key=lambda z: z[0])[1]


def metric_rows(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    ll = float(np.mean(-(y * np.log(p) + (1 - y) * np.log1p(-p))))
    # Calibration regression: observed outcome ~ intercept + slope*logit(p).
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    design = np.column_stack([np.ones(len(z)), z])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    return {
        "games": int(len(y)), "home_win_brier": brier, "home_win_log_loss": ll,
        "home_win_accuracy": float(np.mean((p >= 0.5) == y.astype(bool))),
        "calibration_intercept": float(coef[0]), "calibration_slope": float(coef[1]),
        "mean_predicted_home_win_pct": float(p.mean()),
        "observed_home_win_pct": float(y.mean()),
    }


def paired_block_ci(rows: pd.DataFrame, candidate: str, baseline: str, metric: str) -> dict:
    y = rows.y_home_win.to_numpy(dtype=float)
    p1 = np.clip(rows[candidate].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    p0 = np.clip(rows[baseline].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    if metric == "brier":
        delta = (p1-y) ** 2 - (p0-y) ** 2
    elif metric == "log_loss":
        delta = -(y*np.log(p1) + (1-y)*np.log1p(-p1))
        delta -= -(y*np.log(p0) + (1-y)*np.log1p(-p0))
    else:
        raise ValueError(f"Unknown metric {metric}")
    dates = pd.to_datetime(rows.game_date, errors="coerce")
    if dates.isna().any():
        raise ValueError("Block bootstrap requires a valid game date")
    block_ids = ((dates - dates.min()).dt.days // 14).to_numpy()
    blocks = np.unique(block_ids)
    indices_by_block = [np.flatnonzero(block_ids == b) for b in blocks]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPS, dtype=float)
    for i in range(BOOTSTRAP_REPS):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        ix = np.concatenate([indices_by_block[j] for j in chosen])
        draws[i] = float(delta[ix].mean())
    return {
        "candidate_minus_baseline_mean_loss": float(delta.mean()),
        "paired_14_day_block_bootstrap_95pct_ci": [
            float(x) for x in np.quantile(draws, [0.025, 0.975])
        ],
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "calendar_blocks": int(len(blocks)),
    }


def main(goals_path: str, starter_path: str, goalie_skill_scores_path: str | None, outdir: str) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    games = load_game_ledger(goals_path, starter_path)
    history = build_goalie_team_history(games)
    paired = history.pivot(
        index=["game_id", "source_season", "game_date"], columns="side",
        values=["goalie_id", "goalie_name", "team", "team_win", "team_expected_win",
                "prior_start_count", "raw_prior_win_pct", "prior_expected_win_pct",
                *[f"win_excess_career_k{k}" for k in PRIOR_START_EQUIV],
                *[f"win_excess_r{w}_k{k}" for w in (5, 10, 20) for k in PRIOR_START_EQUIV]],
    ).reset_index()
    paired.columns = [
        "_".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else str(c)
        for c in paired.columns
    ]
    rename = {
        "goalie_id_home": "home_goalie_id", "goalie_id_away": "away_goalie_id",
        "goalie_name_home": "home_goalie_name", "goalie_name_away": "away_goalie_name",
        "team_home": "home_team", "team_away": "away_team",
        "team_win_home": "home_win", "team_win_away": "away_win",
        "team_expected_win_home": "home_expected_win", "team_expected_win_away": "away_expected_win",
        "prior_start_count_home": "home_goalie_prior_starts", "prior_start_count_away": "away_goalie_prior_starts",
        "raw_prior_win_pct_home": "home_goalie_prior_raw_win_pct", "raw_prior_win_pct_away": "away_goalie_prior_raw_win_pct",
        "prior_expected_win_pct_home": "home_goalie_prior_expected_win_pct", "prior_expected_win_pct_away": "away_goalie_prior_expected_win_pct",
    }
    for k in PRIOR_START_EQUIV:
        rename[f"win_excess_career_k{k}_home"] = f"home_win_excess_career_k{k}"
        rename[f"win_excess_career_k{k}_away"] = f"away_win_excess_career_k{k}"
    for w in (5, 10, 20):
        for k in PRIOR_START_EQUIV:
            rename[f"win_excess_r{w}_k{k}_home"] = f"home_win_excess_r{w}_k{k}"
            rename[f"win_excess_r{w}_k{k}_away"] = f"away_win_excess_r{w}_k{k}"
    paired = paired.rename(columns=rename)
    paired = paired.merge(
        games[["game_id", "home_team", "away_team", "y_home_win", "p_home_base"]],
        on="game_id", how="left", validate="one_to_one", suffixes=("", "_ledger"),
    )
    # For each feature, missing prior goalie-team history is a neutral zero.
    feature_columns = [c for c in paired.columns if "win_excess_" in c]
    paired[feature_columns] = paired[feature_columns].fillna(0.0)
    paired["home_wpct_history_advantage"] = (
        paired.home_goalie_prior_raw_win_pct - paired.away_goalie_prior_raw_win_pct
    )
    paired["home_wpct_resid_advantage"] = (
        (paired.home_goalie_prior_raw_win_pct - paired.home_goalie_prior_expected_win_pct)
        - (paired.away_goalie_prior_raw_win_pct - paired.away_goalie_prior_expected_win_pct)
    )

    # Retain the exact-goalie score-model probability as a separately reported
    # reference. It recalculates goals and all derived win probabilities.
    if goalie_skill_scores_path:
        exact = pd.read_parquet(goalie_skill_scores_path)
        exact = exact.loc[exact.model_variant.eq("calibrated_named_goalie"), [
            "game_id", "source_season", "p_full_home", "exp_home_goals", "exp_away_goals",
        ]].copy()
        exact["game_id"] = norm_ids(exact.game_id)
        exact = exact.rename(columns={
            "p_full_home": "p_home_goalie_score", "exp_home_goals": "home_xg_goalie_score",
            "exp_away_goals": "away_xg_goalie_score",
        })
        paired = paired.merge(exact, on=["game_id", "source_season"], how="left", validate="one_to_one")
        if paired.loc[paired.source_season.isin(TARGET_SEASONS), "p_home_goalie_score"].isna().any():
            raise ValueError("Missing exact-goalie score-model comparison for a target game")

    # Scale the explicit goalie-team residual feature on chronological folds.
    preds, fold_results = [], {}
    for target in TARGET_SEASONS:
        train = paired.loc[paired.source_season.le(target - 2)].copy()
        val = paired.loc[paired.source_season.eq(target - 1)].copy()
        test = paired.loc[paired.source_season.eq(target)].copy()
        if min(len(train), len(val), len(test)) < 1000:
            raise ValueError(f"Incomplete fold for {target}: {len(train)}, {len(val)}, {len(test)}")
        best = None
        selection: dict[str, float] = {}
        for family in ["career", "recent"]:
            features = select_features(val, family)
            for name, xv in features.items():
                for alpha in RIDGE_ALPHAS:
                    xt = select_features(train, family)[name]
                    beta_train = fit_beta(train.p_home_base.to_numpy(float), xt,
                                          train.y_home_win.to_numpy(float), alpha)
                    pval = expit(logit(np.clip(val.p_home_base.to_numpy(float), 1e-6, 1 - 1e-6)) + beta_train * xv)
                    score = metric_rows(val.y_home_win.to_numpy(float), pval)["home_win_log_loss"]
                    key = f"{family}:{name}:alpha={alpha}"
                    selection[key] = score
                    if best is None or score < best[0]:
                        best = (score, family, name, alpha, beta_train)
        assert best is not None
        _, family, name, alpha, val_beta = best
        xtrain = select_features(train, family)[name]
        xval = select_features(val, family)[name]
        x_test = select_features(test, family)[name]
        refit = pd.concat([train, val], ignore_index=True)
        x_refit = np.concatenate([xtrain, xval])
        beta = fit_beta(refit.p_home_base.to_numpy(float), x_refit,
                        refit.y_home_win.to_numpy(float), alpha)
        pbase = test.p_home_base.to_numpy(float)
        pgoalie_wpct = expit(logit(np.clip(pbase, 1e-6, 1 - 1e-6)) + beta * x_test)
        result = {
            "training_through_season": int(target - 2), "validation_season": int(target - 1),
            "test_season": int(target), "games": int(len(test)),
            "selected_feature": f"{family}:{name}", "selected_ridge_alpha": float(alpha),
            "validation_beta": float(val_beta), "refit_beta": float(beta),
            "validation_log_loss_by_candidate": selection,
            "baseline": metric_rows(test.y_home_win.to_numpy(float), pbase),
            "team_goalie_win_history": metric_rows(test.y_home_win.to_numpy(float), pgoalie_wpct),
            "mean_absolute_home_probability_shift_pp": float(100 * np.mean(np.abs(pgoalie_wpct-pbase))),
            "mean_absolute_goalie_specific_shift_pp": float(100 * np.mean(np.abs(pgoalie_wpct-pbase))),
            "goalie_history_n_available_pct": float(np.mean(
                (test.home_goalie_prior_starts.to_numpy(float) > 0)
                & (test.away_goalie_prior_starts.to_numpy(float) > 0)
            )),
        }
        if "p_home_goalie_score" in test:
            result["named_goalie_adjusted_score_distribution"] = metric_rows(
                test.y_home_win.to_numpy(float), test.p_home_goalie_score.to_numpy(float)
            )
            result["named_goalie_score_distribution_mean_abs_shift_pp"] = float(
                100 * np.mean(np.abs(test.p_home_goalie_score.to_numpy(float)-pbase))
            )
        fold_results[str(target)] = result
        pred = test[["game_id", "source_season", "game_date", "home_goalie_name", "away_goalie_name",
                     "home_goalie_prior_starts", "away_goalie_prior_starts",
                     "home_goalie_prior_raw_win_pct", "away_goalie_prior_raw_win_pct",
                     "home_goalie_prior_expected_win_pct", "away_goalie_prior_expected_win_pct",
                     "y_home_win", "p_home_base"]].copy()
        pred["selected_feature"] = f"{family}:{name}"
        pred["selected_beta"] = beta
        pred["home_win_excess_adjustment"] = x_test
        pred["p_home_goalie_team_win_history"] = pgoalie_wpct
        if "p_home_goalie_score" in test:
            pred["p_home_goalie_score_distribution"] = test.p_home_goalie_score.to_numpy(float)
            pred["home_xg_goalie_score"] = test.home_xg_goalie_score.to_numpy(float)
            pred["away_xg_goalie_score"] = test.away_xg_goalie_score.to_numpy(float)
        preds.append(pred)

    pred_all = pd.concat(preds, ignore_index=True)
    pred_all.to_parquet(out / "oof_team_goalie_win_pct.parquet", index=False)
    raw_history_summary = {
        "goalie_team_games": int(len(history)),
        "goalie_team_pairs": int(history.groupby(["team", "goalie_id"]).ngroups),
        "starts_with_prior_history": int(history.prior_start_count.gt(0).sum()),
        "prior_history_win_pct_mean": float(history.loc[history.prior_start_count.gt(0), "raw_prior_win_pct"].mean()),
        "prior_history_expected_win_pct_mean": float(history.loc[history.prior_start_count.gt(0), "prior_expected_win_pct"].mean()),
        "strict_pregame_lag": "All cumulative and rolling goalie-team features are shifted one start; same-day team repeats are asserted absent.",
    }
    summary = {
        "experiment_id": "EXP-0037",
        "model_id": MODEL_ID,
        "status": "RESEARCH; no promotion or public export",
        "goal_baseline": "Frozen Goals 009 validation-selected score distribution, out-of-fold",
        "target_seasons": TARGET_SEASONS,
        "starter_identity_caveat": "Actual target starter comes from postgame goalie-box identity; this is an upper-bound diagnostic until a timestamped pregame starter snapshot is available.",
        "winning_percentage_method": "Team-goalie raw prior W% is reported descriptively. The tested predictor is prior team-goalie wins minus the frozen model's expected wins, shrunk by prior-start equivalents; this reduces confounding from team strength.",
        "selection_method": "For each target season, fit on seasons through S-2, select residual-history window/shrinkage and ridge penalty on S-1, refit through S-1, test on S. No betting-market inputs.",
        "feature_history": raw_history_summary,
        "folds": fold_results,
        "pooled": {},
        "decision": "No promotion from this experiment alone. Compare the team's goalie-specific win history with the named-goalie score-distribution layer and frozen baseline; retain only if consistent out-of-sample gains survive follow-up checks.",
        "artifacts": {"predictions": "oof_team_goalie_win_pct.parquet"},
    }
    for col, label in [("p_home_base", "frozen_goal_baseline"), ("p_home_goalie_team_win_history", "goalie_team_win_history")]:
        summary["pooled"][label] = metric_rows(pred_all.y_home_win.to_numpy(float), pred_all[col].to_numpy(float))
    summary["pooled_paired_loss_deltas"] = {
        metric: paired_block_ci(
            pred_all, "p_home_goalie_team_win_history", "p_home_base", metric
        )
        for metric in ("brier", "log_loss")
    }
    if "p_home_goalie_score_distribution" in pred_all:
        summary["pooled"]["named_goalie_score_distribution"] = metric_rows(
            pred_all.y_home_win.to_numpy(float), pred_all.p_home_goalie_score_distribution.to_numpy(float)
        )
        summary["pooled_paired_loss_deltas"]["named_goalie_score_distribution_vs_baseline"] = {
            metric: paired_block_ci(
                pred_all, "p_home_goalie_score_distribution", "p_home_base", metric
            )
            for metric in ("brier", "log_loss")
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--goals",
        default="research/exports/GOALS_TEAM_FORM_60_009_VALIDATION/oof_predictions.parquet",
    )
    parser.add_argument(
        "--starters",
        default="research/.tmp_goalie_probability_016/goalie/data/goalie_feature_output/goalie_game_features.parquet",
    )
    parser.add_argument(
        "--goalie-skill-scores",
        default="research/.tmp_goalie_probability_016/out/goalie_calibration_to_goals_022/oof_goalie_adjusted_scores.parquet",
    )
    parser.add_argument(
        "--out",
        default="research/.tmp_goalie_probability_016/out/goalie_team_win_pct_023",
    )
    args = parser.parse_args()
    main(args.goals, args.starters, args.goalie_skill_scores, args.out)
