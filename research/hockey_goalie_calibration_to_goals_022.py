"""Apply full-pool, calibrated named-goalie ratings to the goals model.

The goal baseline is frozen out-of-fold Goals 009. The goalie skill is
calibrated across the full historical goalie pool by EXP-0035 and uses only
goalie starts before the target game date. The same-game starter identity is
still retrospective, so this is an upper-bound research test, not a live model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import hockey_goals_nested_window_selection_010 as goals


MODEL_ID = "HOCKEY-LAB-GOALIE-CALIBRATION-TO-GOALS-022-RESEARCH"
GOAL_VARIANT = "hockey_process_plus_league_scoring_validation_selected_distribution"
GOALIE_VARIANT = "calibrated_pool_skill"
TARGET_SEASONS = [2024, 2025, 2026]
ALPHAS = [0.01, 0.1, 1.0, 10.0]


def normalize_game_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True)


def load_matched(goals_path: str, goalie_path: str, goalie_source_path: str) -> pd.DataFrame:
    g = pd.read_parquet(goals_path)
    g = g.loc[
        g.model_variant.eq(GOAL_VARIANT) & g.game_type.eq("R"),
        [
            "game_id", "game_date", "source_season", "home_team", "away_team",
            "reg_home_goals", "reg_away_goals", "official_final_home_goals",
            "official_final_away_goals", "went_ot", "rate_mu_home", "rate_mu_away",
            "distribution_family", "distribution_parameter", "p_home_ot_winner",
            "p_reg_home_win", "p_reg_away_win", "p_reg_tie", "p_overtime",
            "p_full_home_win", "exp_reg_home_goals", "exp_reg_away_goals",
        ],
    ].copy()
    g["game_id"] = normalize_game_id(g.game_id)
    goalie = pd.read_parquet(goalie_path)
    all_goalie_starts = goalie.copy()
    goalie = goalie.loc[
        goalie.game_type.eq("R"),
        [
            "game_id", "source_season", "home_away", "goalie_id_key",
            "player_name", "shots", "predicted_skill_delta_per30",
            "selected_feature", "validation_season",
        ],
    ].copy()
    goalie["game_id"] = normalize_game_id(goalie.game_id)
    goalie["home_away"] = goalie.home_away.astype("string")
    if goalie.duplicated(["game_id", "home_away"]).any():
        raise ValueError("Calibrated goalie pool has duplicate game-side ratings")
    wide = goalie.pivot(
        index=["game_id", "source_season"],
        columns="home_away",
        values=["goalie_id_key", "player_name", "predicted_skill_delta_per30", "selected_feature"],
    ).reset_index()
    wide.columns = [
        "_".join(str(x) for x in col if str(x) != "") if isinstance(col, tuple) else str(col)
        for col in wide.columns
    ]
    wide = wide.rename(
        columns={
            "goalie_id_key_home": "home_goalie_id",
            "goalie_id_key_away": "away_goalie_id",
            "player_name_home": "home_goalie_name",
            "player_name_away": "away_goalie_name",
            "predicted_skill_delta_per30_home": "home_goalie_skill",
            "predicted_skill_delta_per30_away": "away_goalie_skill",
            "selected_feature_home": "home_skill_rating_spec",
            "selected_feature_away": "away_skill_rating_spec",
        }
    )
    out = g.merge(wide, on=["game_id", "source_season"], how="left", validate="one_to_one")
    if len(out) != 5 * 1312:
        raise ValueError(f"Expected 6,560 matched games, found {len(out)}")

    # A rare starter can leave after facing zero shots. Such a goalie still
    # has a prior named-goalie rating; reconstruct it from that fold's selected
    # rating rule and the latest earlier positive-shot start.
    raw = pd.read_parquet(goalie_source_path)
    raw = raw.loc[raw.starter_bool.fillna(False).astype(bool), [
        "game_id", "goalie_id_key", "home_away", "player_name", "source_season",
    ]].copy()
    raw["game_id"] = normalize_game_id(raw.game_id)
    raw["goalie_id_key"] = raw.goalie_id_key.astype("string")
    all_goalie_starts["game_id"] = normalize_game_id(all_goalie_starts.game_id)
    all_goalie_starts["goalie_id_key"] = all_goalie_starts.goalie_id_key.astype("string")
    all_goalie_starts["game_date"] = pd.to_datetime(all_goalie_starts.game_date, errors="coerce")

    def reconstruct_skill(goalie_id: str, game_date: pd.Timestamp, season: int) -> tuple[float, str]:
        fold = all_goalie_starts.loc[all_goalie_starts.source_season.eq(season)]
        if fold.empty:
            return 0.0, "neutral_no_fold_rating"
        spec = str(fold.selected_feature.iloc[0])
        beta = float(fold.calibration_slope.iloc[0])
        if spec == "neutral_pool_mean":
            return 0.0, spec
        context, memory, shrink_text = spec.rsplit("_", 2)
        k = int(shrink_text[1:])
        history = all_goalie_starts.loc[
            all_goalie_starts.goalie_id_key.eq(goalie_id)
            & all_goalie_starts.game_date.lt(game_date)
        ].sort_values("game_date")
        if history.empty:
            return 0.0, spec
        prior = history.iloc[-1]
        centered = float(prior.gsax) - float(prior[f"league_mean_{context}"]) * float(prior.shots)
        if memory == "career":
            numerator = float(prior[f"prior_resid_{context}_career"]) + centered
            denominator = float(prior.prior_shots_career) + float(prior.shots) + k
        else:
            window = int(memory[1:])
            numerator = float(prior[f"prior_resid_{context}_r{window}"]) + centered
            denominator = float(prior[f"prior_shots_r{window}"]) + float(prior.shots) + k
        return float(beta * 30.0 * numerator / denominator), spec

    missing = out.loc[out.home_goalie_skill.isna() | out.away_goalie_skill.isna()]
    for idx, row in missing.iterrows():
        missing_side = "home" if pd.isna(row.home_goalie_skill) else "away"
        actual = raw.loc[
            raw.game_id.eq(row.game_id)
            & raw.source_season.eq(row.source_season)
            & raw.home_away.eq(missing_side)
        ]
        if actual.empty:
            raise ValueError(f"Missing actual {missing_side} starter ID for {row.game_id}")
        starter = actual.iloc[0]
        skill, spec = reconstruct_skill(
            str(starter.goalie_id_key), pd.Timestamp(row.game_date), int(row.source_season)
        )
        out.loc[idx, f"{missing_side}_goalie_skill"] = skill
        out.loc[idx, f"{missing_side}_goalie_id"] = str(starter.goalie_id_key)
        out.loc[idx, f"{missing_side}_goalie_name"] = starter.player_name
        out.loc[idx, f"{missing_side}_skill_rating_spec"] = spec
    if out[["home_goalie_skill", "away_goalie_skill"]].isna().any().any():
        raise ValueError("Missing named-goalie skill ratings after the zero-shot fallback")
    if (out.home_goalie_id.astype(str) == out.away_goalie_id.astype(str)).any():
        raise ValueError("Same goalie ID mapped to both teams in a game")
    return out.sort_values(["source_season", "game_date", "game_id"]).reset_index(drop=True)


def long_sides(games: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame({
        "game_id": games.game_id,
        "source_season": games.source_season,
        "goals": games.reg_home_goals.astype(float),
        "base_mu": games.rate_mu_home.astype(float),
        "opponent_goalie_skill": games.away_goalie_skill.astype(float),
    })
    away = pd.DataFrame({
        "game_id": games.game_id,
        "source_season": games.source_season,
        "goals": games.reg_away_goals.astype(float),
        "base_mu": games.rate_mu_away.astype(float),
        "opponent_goalie_skill": games.home_goalie_skill.astype(float),
    })
    return pd.concat([home, away], ignore_index=True)


def fit_beta(sides: pd.DataFrame, alpha: float) -> float:
    y = sides.goals.to_numpy(dtype=float)
    offset = np.log(np.clip(sides.base_mu.to_numpy(dtype=float), 1e-5, 20.0))
    skill = sides.opponent_goalie_skill.to_numpy(dtype=float)

    def objective(beta: float) -> float:
        log_rate = np.clip(offset + beta * skill, -8, 4)
        mu = np.exp(log_rate)
        nll = np.sum(mu - y * log_rate + goals.gammaln(y + 1))
        return float(nll + 0.5 * alpha * beta * beta)

    result = minimize_scalar(objective, bounds=(-3.0, 0.0), method="bounded")
    # Include the no-goalie endpoint explicitly: if the fitted effect is not
    # better under the training objective, let the rolling selection reject it.
    candidates = [(float(result.fun), float(result.x)), (objective(0.0), 0.0)]
    return min(candidates, key=lambda x: x[0])[1]


def count_logloss(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-8, 50.0)
    return float(np.mean(mu - y * np.log(mu) + goals.gammaln(y + 1)))


def score_rates(games: pd.DataFrame, beta: float) -> pd.DataFrame:
    out = games.copy()
    out["mu_home_goalie"] = np.exp(
        np.log(np.clip(out.rate_mu_home.to_numpy(float), 1e-5, 20.0))
        + beta * out.away_goalie_skill.to_numpy(float)
    )
    out["mu_away_goalie"] = np.exp(
        np.log(np.clip(out.rate_mu_away.to_numpy(float), 1e-5, 20.0))
        + beta * out.home_goalie_skill.to_numpy(float)
    )
    return out


def score_distributions(rows: pd.DataFrame, use_goalie: bool) -> pd.DataFrame:
    result = []
    for r in rows.itertuples(index=False):
        mu_home = r.mu_home_goalie if use_goalie else r.rate_mu_home
        mu_away = r.mu_away_goalie if use_goalie else r.rate_mu_away
        mat = goals.joint_score_pmf(
            float(mu_home), float(mu_away), r.distribution_family,
            float(r.distribution_parameter),
        )
        p_hw, p_aw, p_tie = goals.outcome_probs(mat)
        p_full_home = p_hw + p_tie * float(r.p_home_ot_winner)
        y_full_home = int(r.official_final_home_goals > r.official_final_away_goals)
        y_reg = int(
            2 if r.reg_home_goals > r.reg_away_goals
            else 0 if r.reg_home_goals < r.reg_away_goals else 1
        )
        p_reg = [p_aw, p_tie, p_hw]
        exp_h = float(sum(i * mat[i, :].sum() for i in range(mat.shape[0])))
        exp_a = float(sum(j * mat[:, j].sum() for j in range(mat.shape[1])))
        result.append({
            "game_id": r.game_id, "source_season": int(r.source_season),
            "reg_home_goals": int(r.reg_home_goals), "reg_away_goals": int(r.reg_away_goals),
            "official_final_home_goals": int(r.official_final_home_goals),
            "official_final_away_goals": int(r.official_final_away_goals),
            "went_ot": int(bool(r.went_ot)), "y_reg_class": y_reg,
            "p_reg_away": p_aw, "p_reg_tie": p_tie, "p_reg_home": p_hw,
            "p_overtime": p_tie, "p_full_home": p_full_home,
            "y_full_home": y_full_home, "reg_score_ll": -np.log(max(float(mat[int(r.reg_home_goals), int(r.reg_away_goals)]), 1e-15)),
            "exp_home_goals": exp_h, "exp_away_goals": exp_a,
        })
    return pd.DataFrame(result)


def probability_metrics(rows: pd.DataFrame) -> dict:
    y = rows.y_full_home.to_numpy(dtype=float)
    p = np.clip(rows.p_full_home.to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    yreg = rows.y_reg_class.to_numpy(dtype=int)
    p3 = rows[["p_reg_away", "p_reg_tie", "p_reg_home"]].to_numpy(dtype=float)
    onehot = np.eye(3)[yreg]
    yot = rows.went_ot.to_numpy(dtype=float)
    pot = np.clip(rows.p_overtime.to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    return {
        "games": int(len(rows)),
        "reg_home_goal_mae": float(np.mean(np.abs(rows.exp_home_goals - rows.reg_home_goals))),
        "reg_away_goal_mae": float(np.mean(np.abs(rows.exp_away_goals - rows.reg_away_goals))),
        "regulation_score_log_loss": float(rows.reg_score_ll.mean()),
        "regulation_3way_brier": float(np.mean(np.sum((p3 - onehot) ** 2, axis=1))),
        "full_game_home_win_log_loss": float(np.mean(-(y * np.log(p) + (1-y) * np.log1p(-p)))),
        "full_game_home_win_brier": float(np.mean((p-y)**2)),
        "full_game_home_win_accuracy": float(np.mean((p >= .5) == y.astype(bool))),
        "overtime_brier": float(np.mean((pot-yot)**2)),
        "p_home_mean_abs_shift_pp": float(100 * np.mean(np.abs(p-rows.p_full_home_baseline.to_numpy(dtype=float)))) if "p_full_home_baseline" in rows else None,
    }


def main(goals_path: str, goalie_path: str, goalie_source_path: str, outdir: str) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    games = load_matched(goals_path, goalie_path, goalie_source_path)
    sides = long_sides(games)
    predictions = []
    folds = {}
    for target in TARGET_SEASONS:
        train = sides.loc[sides.source_season.le(target - 2)].copy()
        validation = sides.loc[sides.source_season.eq(target - 1)].copy()
        test_games = games.loc[games.source_season.eq(target)].copy()
        if len(train) < 1000 or len(validation) < 1000 or len(test_games) != 1312:
            raise ValueError(f"Incomplete rolling fold for source season {target}")
        val_game_ids = set(validation.game_id.astype(str))
        val_games = games.loc[games.game_id.astype(str).isin(val_game_ids)].copy()
        alpha_scores = {}
        beta_by_alpha = {}
        for alpha in ALPHAS:
            beta_train = fit_beta(train, alpha)
            val_scored = score_rates(val_games, beta_train)
            val_side = long_sides(val_scored)
            val_side["mu"] = np.exp(
                np.log(np.clip(val_side.base_mu, 1e-5, 20.0))
                + beta_train * val_side.opponent_goalie_skill
            )
            score = count_logloss(val_side.goals.to_numpy(), val_side.mu.to_numpy())
            alpha_scores[str(alpha)] = score
            beta_by_alpha[str(alpha)] = beta_train
        alpha = min(ALPHAS, key=lambda x: alpha_scores[str(x)])
        train_full = pd.concat([train, validation], ignore_index=True)
        beta = fit_beta(train_full, alpha)
        test_rates = score_rates(test_games, beta)
        baseline_scored = score_distributions(test_rates, use_goalie=False)
        goalie_scored = score_distributions(test_rates, use_goalie=True)
        base_ll = count_logloss(
            np.concatenate([test_games.reg_home_goals, test_games.reg_away_goals]),
            np.concatenate([test_games.rate_mu_home, test_games.rate_mu_away]),
        )
        cand_ll = count_logloss(
            np.concatenate([test_games.reg_home_goals, test_games.reg_away_goals]),
            np.concatenate([test_rates.mu_home_goalie, test_rates.mu_away_goalie]),
        )
        baseline_scored["p_full_home_baseline"] = baseline_scored.p_full_home
        goalie_scored["p_full_home_baseline"] = baseline_scored.p_full_home.to_numpy()
        metric_base = probability_metrics(baseline_scored)
        metric_cand = probability_metrics(goalie_scored)
        metric_base["goal_count_log_loss_mean_sides"] = base_ll
        metric_cand["goal_count_log_loss_mean_sides"] = cand_ll
        folds[str(target)] = {
            "train_through_season": int(target - 2),
            "validation_season": int(target - 1),
            "test_games": int(len(test_games)),
            "selected_alpha": float(alpha),
            "validation_goal_count_logloss_by_alpha": alpha_scores,
            "refit_beta_on_train_plus_validation": float(beta),
            "baseline_metrics": metric_base,
            "calibrated_named_goalie_metrics": metric_cand,
        }
        for frame, name in [(baseline_scored, "no_goalie"), (goalie_scored, "calibrated_named_goalie")]:
            frame["model_variant"] = name
            frame["selected_beta"] = beta if name != "no_goalie" else 0.0
            predictions.append(frame)

    all_scores = pd.concat(predictions, ignore_index=True)
    all_scores.to_parquet(out / "oof_goalie_adjusted_scores.parquet", index=False)
    summary = {
        "experiment_id": "EXP-0036",
        "model_id": MODEL_ID,
        "status": "RESEARCH; no promotion or public export",
        "goal_baseline": "Goals Team Form 60 009, validation-selected distribution, OOF, no goalie input",
        "goalie_rating": "EXP-0035 calibrated named-goalie GSAx/30 relative to the prior-date goalie-pool mean",
        "identity_caveat": "The actual starter ID is from postgame goalie-box identity; goalie skill is strictly prior-date. Upper-bound diagnostic only until time-stamped pregame starter snapshots exist.",
        "validation_design": "Train coefficient through S-2; select ridge penalty on S-1; refit through S-1; test S. The first scored target is 2024 because the Goals 009 OOF ledger begins in 2022.",
        "adjustment": "One shared nonpositive log-rate coefficient: log(team goal rate)=log(frozen team-only rate)+beta*opposing-goalie skill. Positive goalie skill lowers opponent expected goals; below-average skill raises them.",
        "target_seasons": TARGET_SEASONS,
        "folds": folds,
        "pooled_test_comparison": {},
        "decision": "Do not promote yet. This experiment tests whether a calibrated named-goalie rating adds to the team-only goal distribution and derived win probabilities; all reported results remain retrospective until a timestamped starter source is available.",
        "artifacts": {"oof_goalie_adjusted_scores": "oof_goalie_adjusted_scores.parquet"},
    }
    for variant in ["no_goalie", "calibrated_named_goalie"]:
        group = all_scores.loc[all_scores.model_variant.eq(variant)].copy()
        summary["pooled_test_comparison"][variant] = {
            "games": int(len(group)),
            "full_game_home_win_brier": float(np.mean((group.p_full_home - group.y_full_home) ** 2)),
            "full_game_home_win_log_loss": float(np.mean(-(
                group.y_full_home * np.log(np.clip(group.p_full_home,1e-15,1))
                + (1-group.y_full_home)*np.log(np.clip(1-group.p_full_home,1e-15,1))
            ))),
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
        "--goalie",
        default="research/.tmp_goalie_probability_016/out/goalie_pool_calibration_021/goalie_pool_oof_calibration.parquet",
    )
    parser.add_argument(
        "--goalie-source",
        default="research/.tmp_goalie_probability_016/goalie/data/goalie_feature_output/goalie_game_features.parquet",
    )
    parser.add_argument(
        "--out",
        default="research/.tmp_goalie_probability_016/out/goalie_calibration_to_goals_022",
    )
    args = parser.parse_args()
    main(args.goals, args.goalie, args.goalie_source, args.out)
