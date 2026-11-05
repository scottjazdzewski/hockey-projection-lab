"""Forward holdout comparison of frozen Schedule Context 007 candidates.

All candidates are compared on the same games against baseline_b2b. Confidence
intervals use paired, stratified date-block bootstrap resampling, so every
forecast variant for a sampled game remains paired. Market data is not used.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = [
    "baseline_b2b",
    "plus_3in4",
    "plus_3in4_4in6",
    "plus_4in6",
    "plus_5in8",
    "plus_all_schedule_context",
    "plus_road_trip_finale_proxy",
]


def score_parts(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    return (p - y) ** 2, -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main(input_path, outdir, replicates=2000, seed=5102026):
    d = pd.read_parquet(input_path)
    d = d[d.model.isin(MODELS)].copy()
    d["home_win"] = pd.to_numeric(d.home_win, errors="raise").astype(int)
    d["p_home"] = pd.to_numeric(d.p_home, errors="raise").clip(1e-9, 1 - 1e-9)
    seasons = sorted(d.source_season.unique())
    rng = np.random.default_rng(seed)
    rows = []

    # Paired scoring rows by held-out season.
    for season in seasons:
        z = d[d.source_season.eq(season)].pivot(
            index=["game_date", "game_id"], columns="model", values=["home_win", "p_home"]
        )
        y = z[("home_win", "baseline_b2b")].to_numpy()
        p0 = z[("p_home", "baseline_b2b")].to_numpy()
        b0, l0 = score_parts(y, p0)
        for model in MODELS:
            p = z[("p_home", model)].to_numpy()
            b, ll = score_parts(y, p)
            rows.append({
                "scope": "season", "season": int(season), "model": model,
                "games": int(len(y)), "brier": float(b.mean()), "log_loss": float(ll.mean()),
                "delta_brier_vs_baseline": float((b - b0).mean()),
                "delta_log_loss_vs_baseline": float((ll - l0).mean()),
                "brier_ci_low": np.nan, "brier_ci_high": np.nan,
                "log_loss_ci_low": np.nan, "log_loss_ci_high": np.nan,
            })

    # Pool years by weighted game score. Resample dates within each season,
    # paired across model outputs and stratified to retain each season's size.
    for model in MODELS:
        observed_db, observed_dl, total_n = 0.0, 0.0, 0
        boot_db = np.zeros(replicates)
        boot_dl = np.zeros(replicates)
        boot_n = np.zeros(replicates)
        for season in seasons:
            z = d[d.source_season.eq(season)].pivot(
                index=["game_date", "game_id"], columns="model", values=["home_win", "p_home"]
            )
            y = z[("home_win", "baseline_b2b")].to_numpy()
            p0 = z[("p_home", "baseline_b2b")].to_numpy()
            p1 = z[("p_home", model)].to_numpy()
            b0, l0 = score_parts(y, p0)
            b1, l1 = score_parts(y, p1)
            delta_b, delta_l = b1 - b0, l1 - l0
            date_values = z.index.get_level_values("game_date")
            by_date = pd.DataFrame({"date": date_values, "db": delta_b, "dl": delta_l, "n": 1}).groupby("date").sum()
            observed_db += float(delta_b.sum())
            observed_dl += float(delta_l.sum())
            total_n += len(y)
            sample = rng.integers(0, len(by_date), size=(replicates, len(by_date)))
            boot_db += by_date.db.to_numpy()[sample].sum(axis=1)
            boot_dl += by_date.dl.to_numpy()[sample].sum(axis=1)
            boot_n += by_date.n.to_numpy()[sample].sum(axis=1)
        rows.append({
            "scope": "pooled", "season": "all", "model": model,
            "games": total_n,
            "brier": float(d[d.model.eq(model)].home_win.sub(d[d.model.eq(model)].p_home).pow(2).mean()),
            "log_loss": float(score_parts(
                d[d.model.eq(model)].home_win.to_numpy(),
                d[d.model.eq(model)].p_home.to_numpy()
            )[1].mean()),
            "delta_brier_vs_baseline": observed_db / total_n,
            "delta_log_loss_vs_baseline": observed_dl / total_n,
            "brier_ci_low": float(np.quantile(boot_db / boot_n, .025)),
            "brier_ci_high": float(np.quantile(boot_db / boot_n, .975)),
            "log_loss_ci_low": float(np.quantile(boot_dl / boot_n, .025)),
            "log_loss_ci_high": float(np.quantile(boot_dl / boot_n, .975)),
        })

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "HWIN_SCHEDULE_VARIANT_FORWARD_AUDIT_001.csv", index=False)
    print(pd.DataFrame(rows).query("scope == 'pooled'").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=5102026)
    args = parser.parse_args()
    main(args.input, args.out, args.replicates, args.seed)
