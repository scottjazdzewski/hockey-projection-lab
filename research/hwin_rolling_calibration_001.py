"""Expanding-window test of symmetric logit-slope recalibration."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression


def calibration_intercept_slope(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=2000).fit(x, np.asarray(y, dtype=int))
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def favorite_bin_adjustment(train, test, prior_games=200):
    edges = np.array([0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.000001])
    train_p = train.p_home.to_numpy()
    train_y = train.home_win.to_numpy()
    train_home_fav = train_p >= 0.5
    train_fav_p = np.where(train_home_fav, train_p, 1 - train_p)
    train_fav_y = np.where(train_home_fav, train_y, 1 - train_y)
    train_bin = np.clip(np.digitize(train_fav_p, edges) - 1, 0, len(edges) - 2)

    delta = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for b in range(len(delta)):
        mask = train_bin == b
        counts[b] = int(mask.sum())
        if counts[b]:
            raw_gap = float(np.mean(train_fav_y[mask] - train_fav_p[mask]))
            delta[b] = raw_gap * counts[b] / (counts[b] + prior_games)

    test_p = test.p_home.to_numpy()
    test_home_fav = test_p >= 0.5
    test_fav_p = np.where(test_home_fav, test_p, 1 - test_p)
    test_bin = np.clip(np.digitize(test_fav_p, edges) - 1, 0, len(edges) - 2)
    adjusted_fav = np.clip(test_fav_p + delta[test_bin], 0.5, 1 - 1e-6)
    adjusted_home = np.where(test_home_fav, adjusted_fav, 1 - adjusted_fav)
    return adjusted_home, delta


def main(input_path, output_path):
    d = pd.read_parquet(input_path)
    d = d[d.model.eq("baseline_b2b")].copy()
    d["p_home"] = pd.to_numeric(d.p_home, errors="raise").clip(1e-6, 1 - 1e-6)
    d["home_win"] = pd.to_numeric(d.home_win, errors="raise").astype(int)
    seasons = sorted(d.source_season.unique(), key=lambda value: int(value))
    fold_rows = []
    oof = []

    for i, season in enumerate(seasons[1:], start=1):
        train = d[d.source_season.isin(seasons[:i])]
        test = d[d.source_season.eq(season)].copy()

        # Fit one slope on paired home/away forecasts. No intercept preserves
        # 50% and guarantees calibrated probabilities still sum to 100%.
        p_train = np.r_[train.p_home.to_numpy(), 1 - train.p_home.to_numpy()]
        y_train = np.r_[train.home_win.to_numpy(), 1 - train.home_win.to_numpy()]
        x_train = np.log(p_train / (1 - p_train)).reshape(-1, 1)
        calibrator = LogisticRegression(
            C=1e6, fit_intercept=False, max_iter=2000
        ).fit(x_train, y_train)
        beta = float(calibrator.coef_[0, 0])

        p_raw = test.p_home.to_numpy()
        y = test.home_win.to_numpy()
        p_cal = 1 / (1 + np.exp(-beta * np.log(p_raw / (1 - p_raw))))
        isotonic = IsotonicRegression(out_of_bounds="clip").fit(p_train, y_train)
        p_iso = 0.5 * (isotonic.predict(p_raw) + 1 - isotonic.predict(1 - p_raw))
        p_iso = np.clip(p_iso, 1e-6, 1 - 1e-6)
        p_bin_all, delta_all = favorite_bin_adjustment(train, test)
        recent_train = d[d.source_season.isin(seasons[max(0, i - 2):i])]
        p_bin_recent, delta_recent = favorite_bin_adjustment(recent_train, test)
        raw_intercept, raw_slope = calibration_intercept_slope(y, p_raw)
        cal_intercept, cal_slope = calibration_intercept_slope(y, p_cal)
        iso_intercept, iso_slope = calibration_intercept_slope(y, p_iso)
        fold_rows.append({
            "test_season": season,
            "train_seasons": ",".join(map(str, seasons[:i])),
            "train_games": int(len(train)),
            "test_games": int(len(test)),
            "fitted_logit_slope": beta,
            "raw_brier": float(brier_score_loss(y, p_raw)),
            "calibrated_brier": float(brier_score_loss(y, p_cal)),
            "delta_brier_cal_minus_raw": float(brier_score_loss(y, p_cal) - brier_score_loss(y, p_raw)),
            "raw_log_loss": float(log_loss(y, p_raw, labels=[0, 1])),
            "calibrated_log_loss": float(log_loss(y, p_cal, labels=[0, 1])),
            "delta_log_loss_cal_minus_raw": float(log_loss(y, p_cal, labels=[0, 1]) - log_loss(y, p_raw, labels=[0, 1])),
            "isotonic_brier": float(brier_score_loss(y, p_iso)),
            "delta_brier_isotonic_minus_raw": float(brier_score_loss(y, p_iso) - brier_score_loss(y, p_raw)),
            "isotonic_log_loss": float(log_loss(y, p_iso, labels=[0, 1])),
            "delta_log_loss_isotonic_minus_raw": float(log_loss(y, p_iso, labels=[0, 1]) - log_loss(y, p_raw, labels=[0, 1])),
            "all_prior_favorite_bins_brier": float(brier_score_loss(y, p_bin_all)),
            "delta_brier_all_prior_favorite_bins": float(brier_score_loss(y, p_bin_all) - brier_score_loss(y, p_raw)),
            "all_prior_favorite_bins_log_loss": float(log_loss(y, p_bin_all, labels=[0, 1])),
            "delta_log_loss_all_prior_favorite_bins": float(log_loss(y, p_bin_all, labels=[0, 1]) - log_loss(y, p_raw, labels=[0, 1])),
            "last2_favorite_bins_brier": float(brier_score_loss(y, p_bin_recent)),
            "delta_brier_last2_favorite_bins": float(brier_score_loss(y, p_bin_recent) - brier_score_loss(y, p_raw)),
            "last2_favorite_bins_log_loss": float(log_loss(y, p_bin_recent, labels=[0, 1])),
            "delta_log_loss_last2_favorite_bins": float(log_loss(y, p_bin_recent, labels=[0, 1]) - log_loss(y, p_raw, labels=[0, 1])),
            "trained_60_65_gap_all_prior_pp": float(delta_all[2] * 100),
            "trained_60_65_gap_last2_pp": float(delta_recent[2] * 100),
            "raw_test_cal_intercept": raw_intercept,
            "raw_test_cal_slope": raw_slope,
            "calibrated_test_cal_intercept": cal_intercept,
            "calibrated_test_cal_slope": cal_slope,
            "isotonic_test_cal_intercept": iso_intercept,
            "isotonic_test_cal_slope": iso_slope,
        })
        test["p_raw"] = p_raw
        test["p_cal"] = p_cal
        test["p_iso"] = p_iso
        test["p_bin_all"] = p_bin_all
        test["p_bin_recent"] = p_bin_recent
        oof.append(test)

    folds = pd.DataFrame(fold_rows)
    combined = pd.concat(oof, ignore_index=True)
    y = combined.home_win.to_numpy()
    p_raw = combined.p_raw.to_numpy()
    p_cal = combined.p_cal.to_numpy()
    p_iso = combined.p_iso.to_numpy()
    p_bin_all = combined.p_bin_all.to_numpy()
    p_bin_recent = combined.p_bin_recent.to_numpy()
    aggregate = {
        "folds": fold_rows,
        "aggregate_oof_games": int(len(combined)),
        "aggregate_raw_brier": float(brier_score_loss(y, p_raw)),
        "aggregate_calibrated_brier": float(brier_score_loss(y, p_cal)),
        "aggregate_delta_brier_cal_minus_raw": float(brier_score_loss(y, p_cal) - brier_score_loss(y, p_raw)),
        "aggregate_raw_log_loss": float(log_loss(y, p_raw, labels=[0, 1])),
        "aggregate_calibrated_log_loss": float(log_loss(y, p_cal, labels=[0, 1])),
        "aggregate_delta_log_loss_cal_minus_raw": float(log_loss(y, p_cal, labels=[0, 1]) - log_loss(y, p_raw, labels=[0, 1])),
        "aggregate_isotonic_brier": float(brier_score_loss(y, p_iso)),
        "aggregate_delta_brier_isotonic_minus_raw": float(brier_score_loss(y, p_iso) - brier_score_loss(y, p_raw)),
        "aggregate_isotonic_log_loss": float(log_loss(y, p_iso, labels=[0, 1])),
        "aggregate_delta_log_loss_isotonic_minus_raw": float(log_loss(y, p_iso, labels=[0, 1]) - log_loss(y, p_raw, labels=[0, 1])),
        "aggregate_all_prior_favorite_bins_brier": float(brier_score_loss(y, p_bin_all)),
        "aggregate_delta_brier_all_prior_favorite_bins": float(brier_score_loss(y, p_bin_all) - brier_score_loss(y, p_raw)),
        "aggregate_all_prior_favorite_bins_log_loss": float(log_loss(y, p_bin_all, labels=[0, 1])),
        "aggregate_delta_log_loss_all_prior_favorite_bins": float(log_loss(y, p_bin_all, labels=[0, 1]) - log_loss(y, p_raw, labels=[0, 1])),
        "aggregate_last2_favorite_bins_brier": float(brier_score_loss(y, p_bin_recent)),
        "aggregate_delta_brier_last2_favorite_bins": float(brier_score_loss(y, p_bin_recent) - brier_score_loss(y, p_raw)),
        "aggregate_last2_favorite_bins_log_loss": float(log_loss(y, p_bin_recent, labels=[0, 1])),
        "aggregate_delta_log_loss_last2_favorite_bins": float(log_loss(y, p_bin_recent, labels=[0, 1]) - log_loss(y, p_raw, labels=[0, 1])),
        "promotion_status": "not_promoted",
        "reason": "None of the tested forward-fold maps improved average proper scores.",
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out, index=False)
    print(pd.Series({k: v for k, v in aggregate.items() if k != "folds"}).to_string())
    print("\nFOLDS")
    print(folds.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)
