"""Retrospective starter-quality ablation for the NHL goals model.

This research experiment asks whether prior-date goalie skill and workload
improve the fixed-60-date team goals model. Historical starter identity comes
from the postgame goalie box, so this is an upper-bound diagnostic rather than
a timestamped pregame test. No public export is produced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import hockey_goals_nested_window_selection_010 as goals
from hwin_previous_season_battery_011 import build_goalie_pregame_features


MODEL_VERSION = "HOCKEY-LAB-GOALS-GOALIE-LAYER-011-RESEARCH"
FEATURE_VERSION = "pregame-goals-process-team60-plus-retrospective-starter-quality-workload-011"
TEAM_HISTORY_WINDOW = 60
TARGET_SEASONS = [2022, 2023, 2024, 2025, 2026]

GOALIE_FEATURES = {
    "fixed60_no_goalie": [],
    "opponent_goalie_r20_quality": [
        "opp_goalie_gsax_per30_r20",
        "opp_goalie_fivev5_gsax_per30_r20",
        "opp_goalie_save_pct_r20",
    ],
    "opponent_goalie_career_shrunk": [
        "opp_goalie_gsax_per30_shrunk",
        "opp_goalie_save_pct_shrunk",
    ],
    "opponent_goalie_workload": [
        "opp_goalie_days_since_start",
        "opp_goalie_starts_7d",
        "opp_goalie_starts_14d",
    ],
    "opponent_goalie_quality_plus_workload": [
        "opp_goalie_gsax_per30_r20",
        "opp_goalie_fivev5_gsax_per30_r20",
        "opp_goalie_save_pct_r20",
        "opp_goalie_days_since_start",
        "opp_goalie_starts_7d",
        "opp_goalie_starts_14d",
    ],
}


def attach_retrospective_starter_features(
    side: pd.DataFrame, goalies: pd.DataFrame, spine: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    goalie_spine = spine.copy()
    goalie_spine["game_id"] = pd.to_numeric(goalie_spine.game_id, errors="coerce").astype("Int64").astype("string")
    wide, coverage = build_goalie_pregame_features(goalies, goalie_spine)
    wide["game_id"] = pd.to_numeric(wide.game_id, errors="coerce").astype("Int64")
    out = side.merge(wide, on="game_id", how="left", validate="many_to_one")
    goalie_metrics = [
        "gsax_per30_r20", "fivev5_gsax_per30_r20", "save_pct_r20",
        "gsax_per30_shrunk", "save_pct_shrunk", "days_since_start",
        "starts_7d", "starts_14d",
    ]
    for metric in goalie_metrics:
        away_goalie = pd.to_numeric(out[f"a_goalie_{metric}"], errors="coerce")
        home_goalie = pd.to_numeric(out[f"h_goalie_{metric}"], errors="coerce")
        out[f"opp_goalie_{metric}"] = np.where(out.is_home.eq(1), away_goalie, home_goalie)
    out["opp_goalie_prior_starts_count"] = np.where(
        out.is_home.eq(1),
        pd.to_numeric(out.a_goalie_prior_starts_count, errors="coerce"),
        pd.to_numeric(out.h_goalie_prior_starts_count, errors="coerce"),
    )

    # Every modeled team-game needs a corresponding realized starter for this
    # diagnostic. Missing identity is reported and never silently imputed.
    for col in [f"opp_goalie_{c}" for c in goalie_metrics]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    coverage["target_2022_2026_sides_with_opponent_starter"] = int(
        out.source_season.isin(TARGET_SEASONS).sum()
    )
    coverage["target_2022_2026_sides_missing_opponent_starter"] = int(
        out.loc[out.source_season.isin(TARGET_SEASONS), "opp_goalie_prior_starts_count"].isna().sum()
    )
    coverage["target_2022_2026_sides_missing_prior_goalie_quality"] = int(
        out.loc[out.source_season.isin(TARGET_SEASONS), "opp_goalie_gsax_per30_shrunk"].isna().sum()
    )
    return out, coverage


def pivot_goal_rates(rows: pd.DataFrame, model: object, features: list[str]) -> pd.DataFrame:
    pred = rows[["game_id", "is_home"]].copy()
    pred["mu"] = np.clip(model.predict(rows[features]), 1e-4, 20.0)
    wide = pred.pivot(index="game_id", columns="is_home", values="mu").reset_index()
    return wide.rename(columns={0: "mu_away", 1: "mu_home"})


def count_logloss_by_side(rows: pd.DataFrame) -> dict:
    return {
        "home": goals.count_log_loss(rows.reg_home_goals.to_numpy(), rows.rate_mu_home.to_numpy()),
        "away": goals.count_log_loss(rows.reg_away_goals.to_numpy(), rows.rate_mu_away.to_numpy()),
        "mean_side_logloss": float(np.mean([
            goals.count_log_loss(rows.reg_home_goals.to_numpy(), rows.rate_mu_home.to_numpy()),
            goals.count_log_loss(rows.reg_away_goals.to_numpy(), rows.rate_mu_away.to_numpy()),
        ])),
    }


def main(pbp_path: str, spine_path: str, goalie_path: str, outdir: str) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pbp = pd.read_parquet(pbp_path)
    spine = pd.read_parquet(spine_path)
    goalies = pd.read_parquet(goalie_path)
    labels = goals.derive_game_targets(pbp, spine)
    labels = labels.loc[labels.source_season.between(goals.MIN_SOURCE_SEASON, 2026)].copy()

    side = goals.build_side_data(pbp, spine, labels, rolling_window=TEAM_HISTORY_WINDOW)
    side, goalie_coverage = attach_retrospective_starter_features(side, goalies, spine)
    regular_side = side.loc[side.game_type.eq("R")].copy()
    regular_games = labels.loc[
        labels.game_type.eq("R") & labels.source_season.between(goals.MIN_SOURCE_SEASON, 2026)
    ].copy()

    feature_sets = {
        name: goals.FEATURE_SETS[goals.CORE_FEATURE_SET] + extras
        for name, extras in GOALIE_FEATURES.items()
    }
    all_predictions, fold_rows, fold_choices = [], [], {}
    for target in TARGET_SEASONS:
        validation_season = target - 1
        train = regular_side.loc[regular_side.source_season.le(target - 2)].copy()
        val = regular_side.loc[regular_side.source_season.eq(validation_season)].copy()
        test = regular_side.loc[regular_side.source_season.eq(target)].copy()
        train_games = regular_games.loc[regular_games.source_season.le(target - 2)].copy()
        val_games = regular_games.loc[regular_games.source_season.eq(validation_season)].copy()
        test_games = regular_games.loc[regular_games.source_season.eq(target)].copy()
        if len(test_games) < 1000 or len(val_games) < 500:
            raise RuntimeError(f"Unexpected game coverage for target season {target}")

        fold_models = {}
        for name, features in feature_sets.items():
            alpha, val_count_ll = goals.choose_alpha(train, val, features)
            validation_model = goals.fit_count(train, features, alpha)
            val_mu = pivot_goal_rates(val, validation_model, features)
            fold_models[name] = {
                "alpha": alpha,
                "validation_count_logloss": val_count_ll,
                "validation_model": validation_model,
                "val_mu": val_mu,
            }

        # Select the score law once from the no-goalie baseline on S-1, then
        # hold it fixed across goalie variants to isolate the mean-rate change.
        base = fold_models["fixed60_no_goalie"]
        family, parameter, distribution_scores = goals.validation_distribution_choice(
            val_games, base["val_mu"]
        )
        fold_choices[str(target)] = {
            "validation_season": validation_season,
            "score_distribution_family": family,
            "score_distribution_parameter": parameter,
            "score_distribution_scores": distribution_scores,
            "alpha_by_variant": {
                name: value["alpha"] for name, value in fold_models.items()
            },
        }

        p_ot_final = goals.fit_ot_home_rate(pd.concat([train_games, val_games], ignore_index=True))
        variant_rows = {}
        for name, features in feature_sets.items():
            model_data = pd.concat([train, val], ignore_index=True)
            model = goals.fit_count(model_data, features, fold_models[name]["alpha"])
            test_mu = pivot_goal_rates(test, model, features)
            rows = goals.make_prediction_rows(
                test_games, test_mu, family, parameter, p_ot_final, name, target,
            )
            for row in rows:
                row["model_version"] = MODEL_VERSION
                row["feature_version"] = FEATURE_VERSION
                row["model_variant"] = name
                row["goalie_identity_state"] = (
                    "not_used_baseline" if name == "fixed60_no_goalie"
                    else "retrospective_actual_starter_from_postgame_goalie_box"
                )
                row["goalie_skill_cutoff"] = (
                    "not_used_baseline" if name == "fixed60_no_goalie"
                    else "strictly_prior_game_date; all same-date results excluded"
                )
                row["team_history_window"] = TEAM_HISTORY_WINDOW
                row["data_quality_flag"] = (
                    "verified_regulation_target;strict_prior_date_team_features;no_goalie_predictors"
                    if name == "fixed60_no_goalie" else
                    "verified_regulation_target;strict_prior_date_team_and_goalie_skill;retrospective_starter_identity;not_pregame_snapshot"
                )
            variant_rows[name] = pd.DataFrame(rows)
            all_predictions.extend(rows)

            fold_metric = goals.metrics_for_oof(variant_rows[name])
            fold_rows.append({
                "target_season": target,
                "validation_season": validation_season,
                "train_through_season": target - 2,
                "team_history_window": TEAM_HISTORY_WINDOW,
                "model_variant": name,
                "poisson_alpha": fold_models[name]["alpha"],
                "validation_count_logloss": fold_models[name]["validation_count_logloss"],
                "validation_regulation_score_logloss_base_selected": distribution_scores[family]["validation_reg_score_logloss"],
                "distribution_family": family,
                "distribution_parameter": parameter,
                "goal_count_logloss": count_logloss_by_side(variant_rows[name]),
                "regulation_score_logloss": fold_metric["reg_score_logloss"],
                "regulation_outcome_brier": fold_metric["reg_outcome_multiclass_brier"],
                "overtime_brier": fold_metric["overtime_brier"],
            })

    predictions = pd.DataFrame(all_predictions)
    if predictions.duplicated(["game_id", "model_variant"]).any():
        raise ValueError("Duplicate goal predictions")
    if not np.allclose(
        predictions.p_reg_home_win + predictions.p_reg_away_win + predictions.p_reg_tie,
        1.0, atol=1e-8,
    ):
        raise ValueError("Regulation probabilities do not sum to one")
    if not np.allclose(predictions.p_reg_tie, predictions.p_overtime, atol=1e-12):
        raise ValueError("Overtime probability must equal regulation-tie probability")
    predictions.to_parquet(out / "oof_predictions.parquet", index=False)
    pmf_rows = predictions.loc[
        predictions.model_variant.eq("opponent_goalie_career_shrunk")
    ].copy()
    goals.write_score_distribution_table(pmf_rows, out / "score_distribution_pmf.parquet")
    folds = pd.DataFrame(fold_rows)
    folds.to_parquet(out / "fold_metrics.parquet", index=False)

    models = {}
    for name, group in predictions.groupby("model_variant", sort=True):
        models[name] = {
            "pooled": goals.metrics_for_oof(group),
            "goal_count_logloss": count_logloss_by_side(group),
            "season_metrics": {
                str(int(season)): {
                    **goals.metrics_for_oof(g),
                    "goal_count_logloss": count_logloss_by_side(g),
                }
                for season, g in group.groupby("source_season", sort=True)
            },
        }
    baseline = predictions.loc[predictions.model_variant.eq("fixed60_no_goalie")]
    paired = {}
    for name in GOALIE_FEATURES:
        if name == "fixed60_no_goalie":
            continue
        candidate = predictions.loc[predictions.model_variant.eq(name)]
        keep = ["game_id", "reg_home_goals", "reg_away_goals", "went_ot",
                "p_reg_away_win", "p_reg_tie", "p_reg_home_win", "p_overtime", "logp_reg_score"]
        joined = baseline[keep].merge(
            candidate[keep], on="game_id", suffixes=("_baseline", "_candidate"), validate="one_to_one",
        )
        cls = np.where(joined.reg_home_goals_baseline.gt(joined.reg_away_goals_baseline), 2,
                       np.where(joined.reg_home_goals_baseline.lt(joined.reg_away_goals_baseline), 0, 1))
        p_base = joined[["p_reg_away_win_baseline", "p_reg_tie_baseline", "p_reg_home_win_baseline"]].to_numpy()
        p_cand = joined[["p_reg_away_win_candidate", "p_reg_tie_candidate", "p_reg_home_win_candidate"]].to_numpy()
        onehot = np.eye(3)[cls]
        y_ot = joined.went_ot_baseline.to_numpy(dtype=float)
        paired[name] = {
            "games": int(len(joined)),
            "delta_regulation_score_logloss_candidate_minus_baseline": float(
                (-joined.logp_reg_score_candidate + joined.logp_reg_score_baseline).mean()
            ),
            "delta_regulation_brier_candidate_minus_baseline": float(
                (np.sum((p_cand - onehot) ** 2, axis=1) - np.sum((p_base - onehot) ** 2, axis=1)).mean()
            ),
            "delta_overtime_brier_candidate_minus_baseline": float(
                ((joined.p_overtime_candidate - y_ot) ** 2 - (joined.p_overtime_baseline - y_ot) ** 2).mean()
            ),
        }

    summary = {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "status": "RESEARCH; retrospective starter upper-bound only; not production eligible",
        "validation_design": "rolling origin; train<=S-2; alpha/distribution chosen on S-1; test S; regular season",
        "target_seasons": TARGET_SEASONS,
        "target_season_games": {str(s): int(regular_games.source_season.eq(s).sum()) for s in TARGET_SEASONS},
        "team_history_window": TEAM_HISTORY_WINDOW,
        "feature_sets": feature_sets,
        "goalie_identity_caveat": "Same-game actual starter identity is from the postgame goalie box; no timestamped pregame snapshot. This is retrospective and cannot be promoted or published as a pregame model without an availability-time source.",
        "goalie_skill_cutoff": "Prior full game date only; all results on target date excluded.",
        "goalie_feature_coverage": goalie_coverage,
        "distribution_policy": "For each fold choose score law/parameter on S-1 using the no-goalie baseline, then hold it fixed across goalie variants to isolate goal-rate inputs.",
        "score_distribution_export": {
            "file": "score_distribution_pmf.parquet",
            "variant": "opponent_goalie_career_shrunk",
            "rows": "one per target game",
            "regulation_grid": "26x26; row-major home goals by away goals",
            "final_grid": "27x27; includes training-only overtime winner split",
            "probability_dtype": "float32, zstd-compressed; forecast summaries remain float64",
        },
        "window_and_distribution_choices": fold_choices,
        "paired_candidate_deltas": paired,
        "win_probability_or_market_inputs": "none",
        "models": models,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbp", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--goalie", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.pbp, args.spine, args.goalie, args.out)
