"""Test exact-starter goalie quality as a pregame win-probability increment.

The starter identity is the game's recorded starter, used as a retrospective
proxy for a pregame-confirmed goalie. Every goalie skill value is calculated
from earlier game dates only. The baseline is the frozen OOF Team Strength
004 + B2B probability; no-goalie scores are never altered in place.

Experiment 019 asks whether a named starter's prior career-shrunk save
quality adds to the baseline, and whether the effect grows for unusually
large goalie-quality gaps. All selection is rolling-origin and uses only
previous seasons. This remains research; no site export is produced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hwin_previous_season_battery_011 import build_goalie_pregame_features


BASE_VARIANT = "team_strength_004_plus_b2b"
TARGET_SEASONS = [2024, 2025, 2026]
PENALTIES = [0.01, 0.1, 1.0, 10.0, 100.0]
BOOTSTRAP_REPLICATES = 4000
RNG_SEED = 1903


def model_matrix(delta: np.ndarray, kind: str, threshold: float | None = None) -> np.ndarray:
    """Scale save-percentage gaps in percentage points and add a tail hinge."""
    z = np.asarray(delta, dtype=float) / 0.01
    if kind == "linear":
        return z[:, None]
    if kind == "high_gap":
        if threshold is None:
            raise ValueError("high_gap requires a training-only threshold")
        threshold_z = threshold / 0.01
        tail = np.sign(z) * np.maximum(np.abs(z) - threshold_z, 0.0)
        return np.column_stack([z, tail])
    raise ValueError(f"Unknown feature family: {kind}")


def fit_offset_logit(y: np.ndarray, offset: np.ndarray, x: np.ndarray,
                     penalty: float) -> np.ndarray:
    """Fit nonnegative goalie coefficients with baseline logits as an offset."""
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    x = np.asarray(x, dtype=float)

    def objective(beta: np.ndarray) -> float:
        eta = np.clip(offset + x @ beta, -25, 25)
        nll = np.logaddexp(0.0, eta) - y * eta
        return float(nll.mean() + 0.5 * penalty * np.sum(beta ** 2))

    res = minimize(objective, np.zeros(x.shape[1]), method="L-BFGS-B",
                   bounds=[(0.0, 20.0)] * x.shape[1])
    if not res.success and not np.isfinite(res.fun):
        raise RuntimeError(f"Goaltending coefficient fit failed: {res.message}")
    return np.asarray(res.x, dtype=float)


def predict(frame: pd.DataFrame, kind: str | None,
            beta: np.ndarray | None, threshold: float | None) -> np.ndarray:
    offset = logit(np.clip(frame.p_base.to_numpy(float), 1e-8, 1 - 1e-8))
    if kind is None or beta is None:
        return expit(offset)
    x = model_matrix(frame.d_goalie_save_pct_shrunk.to_numpy(float), kind, threshold)
    return expit(np.clip(offset + x @ beta, -25, 25))


def calibration(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    x = logit(np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8))

    def objective(theta: np.ndarray) -> float:
        eta = np.clip(theta[0] + theta[1] * x, -25, 25)
        return float((np.logaddexp(0.0, eta) - y * eta).mean())

    fit = minimize(objective, np.array([0.0, 1.0]), method="L-BFGS-B")
    return {
        "intercept": float(fit.x[0]),
        "slope": float(fit.x[1]),
        "converged": bool(fit.success),
    }


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-10, 1 - 1e-10)
    ll = -y * np.log(p) - (1 - y) * np.log(1 - p)
    br = (p - y) ** 2
    return {
        "games": int(len(y)),
        "log_loss": float(ll.mean()),
        "brier": float(br.mean()),
        "accuracy": float(((p >= 0.5) == y.astype(bool)).mean()),
        "mean_home_probability": float(p.mean()),
        "home_win_rate": float(y.mean()),
        "calibration_intercept_slope": calibration(y, p),
    }


def reliability_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    frame["bin"] = pd.qcut(frame.p, q=bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        games=("y", "size"), mean_probability=("p", "mean"), observed_rate=("y", "mean"),
        min_probability=("p", "min"), max_probability=("p", "max"),
    ).reset_index()
    grouped["bin"] = grouped.bin.astype(str)
    return grouped.to_dict(orient="records")


def bootstrap_delta(base_loss: np.ndarray, candidate_loss: np.ndarray,
                    seed_offset: int = 0) -> list[float]:
    delta = np.asarray(candidate_loss) - np.asarray(base_loss)
    rng = np.random.default_rng(RNG_SEED + seed_offset)
    draws = rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))
    values = delta[draws].mean(axis=1)
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def loss_vector(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-10, 1 - 1e-10)
    return -y * np.log(p) - (1 - y) * np.log(1 - p), (p - y) ** 2


def main(oof_path: str, goalie_path: str, spine_path: str, outdir: str) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    oof = pd.read_parquet(oof_path)
    oof = oof.loc[oof.model_variant.eq(BASE_VARIANT)].copy()
    oof["game_id"] = pd.to_numeric(oof.game_id, errors="coerce").astype("Int64").astype("string")
    oof["source_season"] = pd.to_numeric(oof.source_season, errors="coerce").astype(int)
    oof["p_base"] = pd.to_numeric(oof.p_home, errors="coerce").clip(1e-8, 1 - 1e-8)
    oof["home_win"] = pd.to_numeric(oof.home_win, errors="coerce").astype(int)

    spine = pd.read_parquet(spine_path)
    spine["game_id"] = pd.to_numeric(spine.game_id, errors="coerce").astype("Int64").astype("string")
    goalie = pd.read_parquet(goalie_path)
    goalie_features, coverage = build_goalie_pregame_features(goalie, spine)
    goalie_features["game_id"] = pd.to_numeric(goalie_features.game_id, errors="coerce").astype("Int64").astype("string")
    full = oof.merge(
        goalie_features[["game_id", "d_goalie_save_pct_shrunk"]],
        on="game_id", how="inner", validate="one_to_one",
    )
    full["d_goalie_save_pct_shrunk"] = pd.to_numeric(
        full.d_goalie_save_pct_shrunk, errors="coerce"
    )
    full = full.dropna(subset=["p_base", "home_win", "d_goalie_save_pct_shrunk"])
    if len(full) != len(oof):
        raise ValueError(f"Exact-starter feature coverage {len(full)}/{len(oof)}")

    test_predictions: list[pd.DataFrame] = []
    fold_summary: dict[str, dict] = {}
    for target in TARGET_SEASONS:
        train = full.loc[full.source_season.le(target - 2)].copy()
        validation = full.loc[full.source_season.eq(target - 1)].copy()
        test = full.loc[full.source_season.eq(target)].copy()
        if min(len(train), len(validation), len(test)) < 1000:
            raise ValueError(f"Insufficient rolling-origin data for {target}")

        y_train = train.home_win.to_numpy(float)
        y_val = validation.home_win.to_numpy(float)
        offset_train = logit(train.p_base.to_numpy(float))
        offset_val = logit(validation.p_base.to_numpy(float))
        base_val = metrics(y_val, expit(offset_val))
        choices = [{"kind": None, "penalty": None, "threshold": None,
                    "validation_log_loss": base_val["log_loss"], "beta": None}]

        for kind in ["linear", "high_gap"]:
            threshold = (float(train.d_goalie_save_pct_shrunk.abs().quantile(0.75))
                         if kind == "high_gap" else None)
            x_train = model_matrix(train.d_goalie_save_pct_shrunk.to_numpy(float), kind, threshold)
            x_val = model_matrix(validation.d_goalie_save_pct_shrunk.to_numpy(float), kind, threshold)
            for penalty in PENALTIES:
                beta = fit_offset_logit(y_train, offset_train, x_train, penalty)
                p_val = expit(np.clip(offset_val + x_val @ beta, -25, 25))
                choices.append({
                    "kind": kind, "penalty": penalty, "threshold": threshold,
                    "validation_log_loss": metrics(y_val, p_val)["log_loss"],
                    "beta_train": beta.tolist(),
                })

        selected = min(choices, key=lambda row: row["validation_log_loss"])
        # Require improvement on the prior validation season; otherwise this
        # fold sends the frozen baseline through unchanged.
        if selected["kind"] is None or selected["validation_log_loss"] >= base_val["log_loss"]:
            selected_kind, selected_penalty, threshold, beta = None, None, None, None
        else:
            selected_kind = selected["kind"]
            selected_penalty = float(selected["penalty"])
            fit_data = pd.concat([train, validation], ignore_index=True)
            threshold = (float(fit_data.d_goalie_save_pct_shrunk.abs().quantile(0.75))
                         if selected_kind == "high_gap" else None)
            x_fit = model_matrix(fit_data.d_goalie_save_pct_shrunk.to_numpy(float), selected_kind, threshold)
            beta = fit_offset_logit(
                fit_data.home_win.to_numpy(float),
                logit(fit_data.p_base.to_numpy(float)),
                x_fit, selected_penalty,
            )

        p_test = predict(test, selected_kind, beta, threshold)
        p_base = test.p_base.to_numpy(float)
        y_test = test.home_win.to_numpy(float)
        ll_base, br_base = loss_vector(y_test, p_base)
        ll_candidate, br_candidate = loss_vector(y_test, p_test)
        score = test[["game_id", "source_season", "home_win", "p_base"]].copy()
        score["p_exact_starter_candidate"] = p_test
        score["selected_family"] = selected_kind or "baseline_no_goalie"
        score["goalie_identity_state"] = "retrospective_actual_starter_from_postgame_box"
        test_predictions.append(score)
        fold_summary[str(target)] = {
            "train_source_seasons": sorted(int(s) for s in train.source_season.unique()),
            "validation_source_season": target - 1,
            "test_source_season": target,
            "games": int(len(test)),
            "selected_family": selected_kind or "baseline_no_goalie",
            "selected_penalty": selected_penalty,
            "high_gap_threshold_save_pct": threshold,
            "coefficients": None if beta is None else beta.tolist(),
            "validation_no_goalie_log_loss": base_val["log_loss"],
            "validation_selected_log_loss": float(selected["validation_log_loss"]),
            "test_baseline": metrics(y_test, p_base),
            "test_candidate": metrics(y_test, p_test),
            "test_delta_log_loss": float(ll_candidate.mean() - ll_base.mean()),
            "test_delta_brier": float(br_candidate.mean() - br_base.mean()),
            "test_mean_absolute_probability_shift": float(np.abs(p_test - p_base).mean()),
        }

    predictions = pd.concat(test_predictions, ignore_index=True)
    p = predictions.p_exact_starter_candidate.to_numpy(float)
    y = predictions.home_win.to_numpy(float)
    p0 = predictions.p_base.to_numpy(float)
    ll0, br0 = loss_vector(y, p0)
    ll1, br1 = loss_vector(y, p)
    pooled = {
        "baseline": metrics(y, p0),
        "exact_starter_candidate": metrics(y, p),
        "delta_log_loss": float((ll1 - ll0).mean()),
        "delta_brier": float((br1 - br0).mean()),
        "paired_game_bootstrap_95pct_log_loss_delta": bootstrap_delta(ll0, ll1),
        "paired_game_bootstrap_95pct_brier_delta": bootstrap_delta(br0, br1, 1),
        "mean_absolute_probability_shift": float(np.abs(p - p0).mean()),
        "reliability_bins_baseline": reliability_bins(y, p0),
        "reliability_bins_candidate": reliability_bins(y, p),
    }
    predictions.to_parquet(out / "oof_predictions.parquet", index=False)
    result = {
        "experiment_id": "HOCKEY-LAB-HWIN-EXACT-STARTER-INCREMENT-019",
        "status": "RESEARCH; retrospective exact starter identity; not a timestamped pregame validation",
        "baseline": "Team Strength 004 + Schedule 007 B2B OOF probabilities; no-goalie probabilities held as logit offset",
        "starter_identity": "Exact starter from postgame goalie box, treated as a retrospective proxy for a pregame-confirmed starter",
        "goalie_quality": "Career-shrunk save percentage differential; all same-date and same-game goalie performance excluded",
        "nonlinearity": "Validation may choose linear effect or a signed high-gap hinge; high-gap cutoff is training-only 75th percentile",
        "selection": "For each target season, fit on seasons <= target-2, select feature family and L2 penalty on target-1, refit through target-1, score target",
        "target_seasons": TARGET_SEASONS,
        "source_coverage": coverage,
        "folds": fold_summary,
        "pooled_test": pooled,
        "bootstrap_note": "Game-level paired bootstrap is exploratory and does not account for team or season clustering; only three target seasons are available to this overlay.",
        "constraints": ["no market inputs", "no same-game goalie statistics", "no live export", "Hockey Lab Game v1.0.0 untouched"],
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True)
    parser.add_argument("--goalies", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.oof, args.goalies, args.spine, args.out)
