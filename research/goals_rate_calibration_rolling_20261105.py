"""Rolling prior-season rate calibration diagnostics for the fixed-60 goals OOF.

Fits candidate Poisson log-rate mappings on season S-1 OOF forecasts and scores
them only on season S. This is development research; it does not modify the
production model or select a promotion candidate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from hockey_goals_team_form_60_009 import (
    final_score_pmf,
    joint_score_pmf,
    outcome_probs,
    score_total_pmf,
)

ROOT = Path("/workspace/scratch/2369a7a4286e")
OOF_PATH = ROOT / "run_009_new_snapshot/oof_predictions.parquet"
OUT = ROOT / "run_009_new_snapshot/rate_calibration"
CORE_VARIANT = "hockey_process_plus_league_scoring_validation_selected_distribution"
TARGET_SEASONS = [2023, 2024, 2025, 2026]


def fit_poisson_line(mu: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Poisson MLE for log(mean) = intercept + slope * log(base_mean)."""
    x = np.log(np.clip(np.asarray(mu, float), 1e-8, None))
    y = np.asarray(y, float)

    def objective(theta: np.ndarray) -> float:
        eta = np.clip(theta[0] + theta[1] * x, -8.0, 8.0)
        return float(np.sum(np.exp(eta) - y * eta + gammaln(y + 1)))

    fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    if not fit.success and np.linalg.norm(fit.jac) > 1e-4:
        raise RuntimeError(f"Poisson calibration fit failed: {fit.message}")
    return float(fit.x[0]), float(fit.x[1])


def fit_calibrators(prior: pd.DataFrame) -> dict[str, object]:
    hmu = prior.rate_mu_home.to_numpy(float)
    amu = prior.rate_mu_away.to_numpy(float)
    hy = prior.reg_home_goals.to_numpy(float)
    ay = prior.reg_away_goals.to_numpy(float)
    all_mu = np.concatenate([hmu, amu])
    all_y = np.concatenate([hy, ay])
    common_intercept = float(np.log(all_y.sum() / all_mu.sum()))
    home_intercept = float(np.log(hy.sum() / hmu.sum()))
    away_intercept = float(np.log(ay.sum() / amu.sum()))
    shared_line = fit_poisson_line(all_mu, all_y)
    home_line = fit_poisson_line(hmu, hy)
    away_line = fit_poisson_line(amu, ay)
    return {
        "shared_intercept": common_intercept,
        "side_intercepts": [home_intercept, away_intercept],
        "shared_affine_log_rate": list(shared_line),
        "side_affine_log_rate": [list(home_line), list(away_line)],
    }


def apply_rate_model(rows: pd.DataFrame, method: str, pars: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    h = rows.rate_mu_home.to_numpy(float)
    a = rows.rate_mu_away.to_numpy(float)
    if method == "no_correction":
        return h, a
    if method == "shared_intercept":
        c = float(pars["shared_intercept"])
        return h * np.exp(c), a * np.exp(c)
    if method == "side_intercepts":
        ch, ca = pars["side_intercepts"]
        return h * np.exp(ch), a * np.exp(ca)
    if method == "shared_affine_log_rate":
        b0, b1 = pars["shared_affine_log_rate"]
        return np.exp(b0 + b1 * np.log(h)), np.exp(b0 + b1 * np.log(a))
    if method == "side_affine_log_rate":
        (bh0, bh1), (ba0, ba1) = pars["side_affine_log_rate"]
        return np.exp(bh0 + bh1 * np.log(h)), np.exp(ba0 + ba1 * np.log(a))
    raise ValueError(method)


def score(rows: pd.DataFrame, mu_home: np.ndarray, mu_away: np.ndarray) -> dict[str, float]:
    preds = []
    for i, r in enumerate(rows.itertuples(index=False)):
        reg = joint_score_pmf(
            float(mu_home[i]),
            float(mu_away[i]),
            str(r.distribution_family),
            float(r.distribution_parameter),
        )
        ph, pa, pt = outcome_probs(reg)
        full = final_score_pmf(reg, float(r.p_home_ot_winner))
        total = score_total_pmf(full)
        eh = float(np.dot(np.arange(reg.shape[0]), reg.sum(axis=1)))
        ea = float(np.dot(np.arange(reg.shape[1]), reg.sum(axis=0)))
        efh = float(np.dot(np.arange(full.shape[0]), full.sum(axis=1)))
        efa = float(np.dot(np.arange(full.shape[1]), full.sum(axis=0)))
        rh, ra = int(r.reg_home_goals), int(r.reg_away_goals)
        fh, fa = int(r.official_final_home_goals), int(r.official_final_away_goals)
        actual_total = fh + fa
        observed_class = 0 if rh > ra else 1 if rh == ra else 2
        p = np.array([ph, pt, pa])
        brier3 = float(np.sum((p - np.eye(3)[observed_class]) ** 2))
        preds.append({
            "eh": eh, "ea": ea, "efh": efh, "efa": efa,
            "p_ot": pt, "brier3": brier3,
            "reg_ll": -np.log(max(float(reg[rh, ra]), 1e-15)),
            "final_ll": -np.log(max(float(full[fh, fa]), 1e-15)),
            "total_ll": -np.log(max(float(total[actual_total]), 1e-15)) if actual_total < len(total) else -np.log(1e-15),
            "p_over55": float(total[6:].sum()),
        })
    p = pd.DataFrame(preds)
    home = rows.reg_home_goals.to_numpy(float)
    away = rows.reg_away_goals.to_numpy(float)
    total_actual = (rows.official_final_home_goals + rows.official_final_away_goals).to_numpy(float)
    final_total_pred = p.efh.to_numpy() + p.efa.to_numpy()
    return {
        "games": int(len(rows)),
        "home_reg_mean_error": float(np.mean(p.eh.to_numpy() - home)),
        "home_reg_mae": float(np.mean(np.abs(p.eh.to_numpy() - home))),
        "away_reg_mean_error": float(np.mean(p.ea.to_numpy() - away)),
        "away_reg_mae": float(np.mean(np.abs(p.ea.to_numpy() - away))),
        "final_total_mean_error": float(np.mean(final_total_pred - total_actual)),
        "final_total_mae": float(np.mean(np.abs(final_total_pred - total_actual))),
        "reg_score_logloss": float(p.reg_ll.mean()),
        "final_score_logloss": float(p.final_ll.mean()),
        "final_total_logloss": float(p.total_ll.mean()),
        "reg_three_way_brier": float(p.brier3.mean()),
        "ot_brier": float(np.mean((p.p_ot.to_numpy() - rows.went_ot.to_numpy(float)) ** 2)),
        "ot_predicted_rate": float(p.p_ot.mean()),
        "ot_observed_rate": float(rows.went_ot.mean()),
        "over55_predicted_rate": float(p.p_over55.mean()),
        "over55_observed_rate": float(np.mean(total_actual > 5.5)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(OOF_PATH)
    d = d.loc[d.model_variant.eq(CORE_VARIANT)].copy()
    methods = [
        "no_correction",
        "shared_intercept",
        "side_intercepts",
        "shared_affine_log_rate",
        "side_affine_log_rate",
    ]
    metric_rows = []
    parameter_rows = []
    for target in TARGET_SEASONS:
        prior = d.loc[d.source_season.eq(target - 1)]
        test = d.loc[d.source_season.eq(target)]
        if prior.empty or len(test) != 1312:
            raise ValueError(f"Missing prior OOF or target rows for {target}: prior={len(prior)} target={len(test)}")
        pars = fit_calibrators(prior)
        parameter_rows.append({
            "target_season": target,
            "calibration_season": target - 1,
            "prior_games": int(len(prior)),
            "parameters_json": json.dumps(pars, sort_keys=True),
        })
        for method in methods:
            h, a = apply_rate_model(test, method, pars)
            row = {"target_season": target, "method": method, **score(test, h, a)}
            metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    params = pd.DataFrame(parameter_rows)
    metrics.to_csv(OUT / "metrics_by_season.csv", index=False)
    params.to_csv(OUT / "calibration_parameters.csv", index=False)
    pooled = (
        metrics.groupby("method", sort=False)
        .apply(lambda g: pd.Series({
            col: float(np.average(g[col], weights=g.games))
            for col in metrics.columns
            if col not in {"target_season", "method", "games"}
        }), include_groups=False)
        .reset_index()
    )
    pooled.insert(1, "target_seasons", "2023-2026")
    pooled.insert(2, "games", 5248)
    pooled.to_csv(OUT / "metrics_pooled.csv", index=False)
    manifest = {
        "run_id": "GOALS-RATE-CALIBRATION-ROLLING-20261105",
        "scope": "fixed-60 goals-only research; OOF calibration fit on S-1 and scored on S",
        "target_seasons": TARGET_SEASONS,
        "excluded_target_2022_reason": "No 2021 OOF forecast row is present in the recovered EXP-0027 export.",
        "calibration_methods": methods,
        "calibration_candidates": "shared/side-specific log-rate intercepts and shared/side-specific affine maps on log rate",
        "selection_policy": "No target-season metrics used to fit or select parameters; development-only, no promotion",
        "predictor_policy": "No markets, HWIN, same-game goalie identity, or lineup inputs",
        "source_snapshot": "2026-11-04 rebuilt artifacts; frozen EXP-0027 input hashes differ",
        "oof_input_sha256": __import__("hashlib").sha256(OOF_PATH.read_bytes()).hexdigest(),
        "results": "metrics_by_season.csv and metrics_pooled.csv",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("POOLED METRICS")
    print(pooled.to_string(index=False))
    print("\nSEASON METRICS")
    print(metrics.to_string(index=False))
    print("\nCALIBRATION PARAMETERS")
    print(params.to_string(index=False))


if __name__ == "__main__":
    main()
