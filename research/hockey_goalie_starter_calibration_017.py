"""Chronological calibration check for goalie starter choice probabilities.

Consumes the frozen out-of-fold candidate logits produced by starter model 016.
For each target season, calibrators are fitted only on earlier-season OOF rows.
Includes an unknown/returning-goalie intercept and temperature scaling; neither
starter labels from the target season nor future dates are used for calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


def grouped_probabilities(frame: pd.DataFrame, temperature: float, unknown_shift: float) -> pd.Series:
    z = frame.logit.to_numpy(float) / temperature
    z = z + frame.candidate.eq("__UNKNOWN__").to_numpy(float) * unknown_shift
    choice = frame.choice_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(choice[1:] != choice[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(frame)])
    maxima = np.maximum.reduceat(z, starts)
    sums = np.add.reduceat(np.exp(z - np.repeat(maxima, sizes)), starts)
    out = np.exp(z - np.repeat(maxima + np.log(sums), sizes))
    return pd.Series(out, index=frame.index)


def fit_calibrator(train: pd.DataFrame, allow_unknown_shift: bool) -> dict:
    labels = train.label.to_numpy(float)
    n_groups = train.choice_id.nunique()
    choice = train.choice_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(choice[1:] != choice[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(train)])
    unknown = train.candidate.eq("__UNKNOWN__").to_numpy(float)
    logits = train.logit.to_numpy(float)

    def loss(theta: np.ndarray) -> float:
        temp = float(np.exp(theta[0]))
        shift = float(theta[1]) if allow_unknown_shift else 0.0
        z = logits / temp + unknown * shift
        maxima = np.maximum.reduceat(z, starts)
        sums = np.add.reduceat(np.exp(z - np.repeat(maxima, sizes)), starts)
        log_norm = np.repeat(maxima + np.log(sums), sizes)
        return float(-np.sum(labels * (z - log_norm)) / n_groups)

    x0 = np.array([0.0, 0.0] if allow_unknown_shift else [0.0])
    bounds = [(-1.0, 1.0)] + ([(-5.0, 5.0)] if allow_unknown_shift else [])
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
    return {
        "temperature": float(np.exp(res.x[0])),
        "unknown_logit_shift": float(res.x[1]) if allow_unknown_shift else 0.0,
        "train_groups": int(n_groups),
        "converged": bool(res.success),
        "train_logloss": float(res.fun),
    }


def metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict:
    p = np.clip(np.asarray(probability, float), 1e-15, 1.0)
    y = frame.label.to_numpy(float)
    work = frame[["choice_id", "candidate", "label"]].copy()
    work["p"] = p
    result = []
    for cid, g in work.groupby("choice_id", sort=False):
        actual = g.label.to_numpy().argmax()
        probs = g.p.to_numpy()
        result.append({
            "choice_id": cid,
            "logloss": float(-np.log(max(probs[actual], 1e-15))),
            "brier": float(np.sum((probs - np.eye(1, len(g), actual)[0]) ** 2)),
            "correct": int(np.argmax(probs) == actual),
            "confidence": float(np.max(probs)),
            "unknown_actual": int(g.candidate.iloc[actual] == "__UNKNOWN__"),
            "unknown_probability": float(probs[g.candidate.eq("__UNKNOWN__").to_numpy()][0]),
        })
    m = pd.DataFrame(result)
    return {
        "choices": int(len(m)),
        "logloss": float(m.logloss.mean()),
        "multiclass_brier": float(m.brier.mean()),
        "top1_accuracy": float(m.correct.mean()),
        "unknown_actual_rate": float(m.unknown_actual.mean()),
        "mean_unknown_probability": float(m.unknown_probability.mean()),
        "top_choice_calibration_bins": calibration_bins(m),
    }


def calibration_bins(m: pd.DataFrame) -> list[dict]:
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
    labels = ["[0,.4)", "[.4,.5)", "[.5,.6)", "[.6,.7)", "[.7,.8)", "[.8,.9)", "[.9,1]"]
    x = m.copy()
    x["bin"] = pd.cut(x.confidence, bins=bins, labels=labels, right=False)
    out = []
    for name, g in x.groupby("bin", observed=False):
        if len(g):
            out.append({"bin": str(name), "n": int(len(g)), "mean_confidence": float(g.confidence.mean()),
                        "accuracy": float(g.correct.mean())})
    return out


def main(candidates_path: str, games_path: str, out_path: str) -> dict:
    c = pd.read_parquet(candidates_path)
    games = pd.read_parquet(games_path, columns=["game_id", "season", "game_date"])
    games["game_id"] = games.game_id.astype(str)
    games["game_date"] = pd.to_datetime(games.game_date, errors="coerce")
    dates = games.drop_duplicates("game_id").set_index("game_id").game_date
    c["game_date"] = pd.to_datetime(c.game_id.map(dates), errors="coerce")
    c["season"] = pd.to_numeric(c.season, errors="coerce").astype(int)
    c = c.loc[c.game_date.notna()].copy()
    season_first = c.groupby("season").game_date.min().to_dict()
    c["day_from_season_start"] = [int((d - season_first[s]).days) for s, d in zip(c.season, c.game_date)]
    seasons = sorted(c.season.unique().tolist())
    results = {}
    candidate_outputs = []
    for target in seasons:
        test = c.loc[c.season.eq(target)].copy()
        earlier = c.loc[c.season.lt(target)].copy()
        if earlier.empty:
            continue
        full_temp = fit_calibrator(earlier, allow_unknown_shift=False)
        full_both = fit_calibrator(earlier, allow_unknown_shift=True)
        p_raw = test.model_probability.to_numpy(float)
        p_temp = grouped_probabilities(test, full_temp["temperature"], 0.0).to_numpy()
        p_both = grouped_probabilities(test, full_both["temperature"], full_both["unknown_logit_shift"]).to_numpy()
        test["full_calibrated_probability"] = p_both
        test["full_calibration_temperature"] = full_both["temperature"]
        test["full_calibration_unknown_logit_shift"] = full_both["unknown_logit_shift"]
        test["early_phase_calibrated_probability"] = np.nan
        test["early_phase_calibration_temperature"] = np.nan
        test["early_phase_calibration_unknown_logit_shift"] = np.nan
        results[str(int(target))] = {
            "test_window": "full available target season",
            "raw": metrics(test, p_raw),
            "temperature_only": {"calibrator": full_temp, "metrics": metrics(test, p_temp)},
            "temperature_plus_unknown_shift": {"calibrator": full_both, "metrics": metrics(test, p_both)},
            "frequency_baseline": metrics(test, test.frequency_probability.to_numpy(float)),
        }

        # A 34-day early-season diagnostic compares like seasonal phases. The
        # calibrator sees only the same early window from strictly older seasons.
        early_train = earlier.loc[earlier.day_from_season_start.le(33)].copy()
        early_test = test.loc[test.day_from_season_start.le(33)].copy()
        if not early_train.empty and not early_test.empty:
            early_cal = fit_calibrator(early_train, allow_unknown_shift=True)
            p_early_raw = early_test.model_probability.to_numpy(float)
            p_early_cal = grouped_probabilities(
                early_test, early_cal["temperature"], early_cal["unknown_logit_shift"]
            ).to_numpy()
            test.loc[early_test.index, "early_phase_calibrated_probability"] = p_early_cal
            test.loc[early_test.index, "early_phase_calibration_temperature"] = early_cal["temperature"]
            test.loc[early_test.index, "early_phase_calibration_unknown_logit_shift"] = early_cal["unknown_logit_shift"]
            results[str(int(target))]["early_34_day_comparable"] = {
                "train_seasons": sorted(int(s) for s in early_train.season.unique()),
                "test_days_from_season_start": [0, 33],
                "raw": metrics(early_test, p_early_raw),
                "temperature_plus_unknown_shift": {
                    "calibrator": early_cal, "metrics": metrics(early_test, p_early_cal)
                },
                "frequency_baseline": metrics(early_test, early_test.frequency_probability.to_numpy(float)),
            }
        candidate_outputs.append(test)

    summary = {
        "experiment_id": "HOCKEY-LAB-GOALIE-STARTER-CALIBRATION-017",
        "status": "RESEARCH",
        "input_model": "HOCKEY-LAB-GOALIE-STARTER-PROBABILITY-016",
        "calibration_methods": ["scalar temperature", "scalar temperature plus unknown-alternative logit shift"],
        "validation": "strictly chronological outer folds; calibrator trained only on older seasons' OOF candidate logits; also early 34-day windows compared by same seasonal phase",
        "leakage_control": "candidate actual labels in each target season are excluded from its calibration fit; starter identity is label only; features used to produce logits were strictly prior-date",
        "results_by_target_season": results,
    }
    Path(out_path).write_text(json.dumps(summary, indent=2) + "\n")
    Path(out_path).with_name("calibrated_candidate_probabilities.parquet").parent.mkdir(parents=True, exist_ok=True)
    pd.concat(candidate_outputs, ignore_index=True).to_parquet(
        Path(out_path).with_name("calibrated_candidate_probabilities.parquet"), index=False)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--games", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.candidates, a.games, a.out)
