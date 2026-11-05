import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "research")
import hwin_team_strength_004 as ts
import hwin_goalie_layer_001 as gl

C_GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
TARGET_SEASONS = [2022, 2023, 2024, 2025, 2026]


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return {"n": 0, "brier": None, "log_loss": None, "accuracy": None}
    return {
        "n": int(len(y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
    }


def make_pipe(c):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", LogisticRegression(C=c, max_iter=2000)),
    ])


def fit_tuned(train, val, test, features):
    best = None
    for c in C_GRID:
        pipe = make_pipe(c)
        pipe.fit(train[features], train.home_win)
        p = pipe.predict_proba(val[features])[:, 1]
        ll = log_loss(val.home_win, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
        if best is None or ll < best[0]:
            best = (ll, c)
    c = best[1]
    fit = pd.concat([train, val], ignore_index=True)
    pipe = make_pipe(c)
    pipe.fit(fit[features], fit.home_win)
    p = pipe.predict_proba(test[features])[:, 1]
    return c, p


def merge_goalies(x, goalies):
    gp = gl.build_goalie_priors(goalies)
    base = ["game_id", "home_away", "sv_shrunk_prior"]
    h = gp[gp.home_away.astype("string").str.lower().eq("home")][base].drop_duplicates("game_id")
    a = gp[gp.home_away.astype("string").str.lower().eq("away")][base].drop_duplicates("game_id")
    h = h.rename(columns={"sv_shrunk_prior": "h_sv_shrunk_prior"})
    a = a.rename(columns={"sv_shrunk_prior": "a_sv_shrunk_prior"})
    x = x.merge(h.drop(columns=["home_away"]), on="game_id", how="left")
    x = x.merge(a.drop(columns=["home_away"]), on="game_id", how="left")
    x["d_g_sv_shrunk_prior"] = (
        pd.to_numeric(x.h_sv_shrunk_prior, errors="coerce")
        - pd.to_numeric(x.a_sv_shrunk_prior, errors="coerce")
    )
    return x


def schedule_features(spine):
    s = spine[["game_id", "game_date", "source_season",
               "home_team_abbr", "away_team_abbr"]].copy()
    s["game_date"] = pd.to_datetime(s.game_date, errors="coerce")
    h = s.rename(columns={"home_team_abbr": "team", "away_team_abbr": "opp"}).assign(home_away="home")
    a = s.rename(columns={"away_team_abbr": "team", "home_team_abbr": "opp"}).assign(home_away="away")
    long = pd.concat([h, a], ignore_index=True)
    long = long.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)

    rows = []
    for team, group in long.groupby("team", sort=False):
        g = group.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        hist_dates = []
        away_streak = 0
        for i, r in enumerate(g.itertuples(index=False)):
            d = r.game_date
            prev = hist_dates[-1] if hist_dates else pd.NaT
            rest = (d - prev).days if hist_dates and pd.notna(d) and pd.notna(prev) else np.nan
            prior3 = sum(1 for x in hist_dates if pd.notna(d) and 1 <= (d - x).days <= 3)
            prior5 = sum(1 for x in hist_dates if pd.notna(d) and 1 <= (d - x).days <= 5)
            prior7 = sum(1 for x in hist_dates if pd.notna(d) and 1 <= (d - x).days <= 7)

            is_away = r.home_away == "away"
            trip_no = away_streak + 1 if is_away else 0
            next_is_home = (
                i + 1 < len(g)
                and g.iloc[i + 1].home_away == "home"
            )
            # A schedule-only proxy: second or later consecutive road game,
            # with the next scheduled game at home. It does not claim to know
            # actual flight timing, distance, or local departure time.
            road_trip_finale = bool(is_away and trip_no >= 2 and next_is_home)

            rows.append({
                "game_id": r.game_id,
                "home_away": r.home_away,
                "sched_rest_days": rest,
                "sched_b2b": float(pd.notna(rest) and rest <= 1),
                "sched_3in4": float(prior3 >= 2),
                "sched_4in6": float(prior5 >= 3),
                "sched_5in8": float(prior7 >= 4),
                "sched_away_trip_finale": float(road_trip_finale),
            })
            hist_dates.append(d)
            away_streak = away_streak + 1 if is_away else 0

    z = pd.DataFrame(rows)
    cols = [c for c in z.columns if c.startswith("sched_")]
    hh = z[z.home_away.eq("home")][["game_id"] + cols].rename(
        columns={c: f"h_{c}" for c in cols}
    )
    aa = z[z.home_away.eq("away")][["game_id"] + cols].rename(
        columns={c: f"a_{c}" for c in cols}
    )
    out = hh.merge(aa, on="game_id", how="inner")
    for c in cols:
        out[f"d_{c}"] = (
            pd.to_numeric(out[f"h_{c}"], errors="coerce")
            - pd.to_numeric(out[f"a_{c}"], errors="coerce")
        )
    return out


def choose_base(train, val, x):
    best = None
    for alpha, ret in ts.OA_GRID:
        tag = f"a{alpha:.2f}_r{ret:.2f}".replace(".", "p")
        feats = [
            "elo_logit", "d_sog_share_r20", "d_xg_share_r20",
            "d_goals_r20", "d_rest_days",
            f"oa_sog_diff_{tag}", f"oa_xg_diff_{tag}",
            "d_g_sv_shrunk_prior",
        ]
        for c in C_GRID:
            pipe = make_pipe(c)
            pipe.fit(train[feats], train.home_win)
            p = pipe.predict_proba(val[feats])[:, 1]
            ll = log_loss(val.home_win, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
            if best is None or ll < best[0]:
                best = (ll, tag, feats)
    return best[1], best[2]


def paired_bootstrap(preds, candidate, condition=None, draws=2000, seed=20261105):
    models = ["baseline_b2b", candidate]
    z = preds[preds.model.isin(models)].copy()
    if condition:
        z = z[z[condition].astype(bool)]
    wide = z.pivot_table(
        index=["source_season", "game_id", "home_win"],
        columns="model",
        values="p_home",
        aggfunc="first",
    ).dropna(subset=models)
    if wide.empty:
        return [{"candidate": candidate, "condition": condition or "all_test_games",
                 "n": 0, "metric": None, "delta_candidate_minus_baseline": None,
                 "ci95_low": None, "ci95_high": None,
                 "bootstrap_share_improved": None, "draws": draws}]

    y = wide.index.get_level_values("home_win").to_numpy(dtype=int)
    p_base = np.clip(wide["baseline_b2b"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    p_candidate = np.clip(wide[candidate].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    season_values = wide.index.get_level_values("source_season").to_numpy()
    losses = {
        "brier": (y - p_candidate) ** 2 - (y - p_base) ** 2,
        "log_loss": (
            -(y * np.log(p_candidate) + (1 - y) * np.log(1 - p_candidate))
            + y * np.log(p_base) + (1 - y) * np.log(1 - p_base)
        ),
    }
    total = len(wide)
    rng = np.random.default_rng(seed)
    rows = []
    for metric, delta in losses.items():
        boot = np.zeros(draws, dtype=float)
        for season in np.sort(np.unique(season_values)):
            values = delta[season_values == season]
            n = len(values)
            if n == 0:
                continue
            ix = rng.integers(0, n, size=(draws, n))
            boot += values[ix].mean(axis=1) * (n / total)
        rows.append({
            "candidate": candidate,
            "condition": condition or "all_test_games",
            "metric": metric,
            "n": total,
            "delta_candidate_minus_baseline": float(delta.mean()),
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
            "bootstrap_share_improved": float(np.mean(boot < 0)),
            "draws": draws,
            "method": "paired game bootstrap, resampled within each target season",
        })
    return rows


def main(pbp_path, spine_path, goalie_path, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    pbp = pd.read_parquet(pbp_path)
    spine = pd.read_parquet(spine_path)
    goalie = pd.read_parquet(goalie_path)
    spine = spine[
        spine.result_available.fillna(False)
        & spine.game_type.astype("string").isin(["R", "P"])
    ].copy()
    spine["game_date"] = pd.to_datetime(spine.game_date, errors="coerce")
    spine["source_season"] = pd.to_numeric(spine.source_season, errors="coerce")

    g = ts.add_flat_and_ewm_features(ts.game_event_table(pbp))
    g = merge_goalies(g, goalie)
    g = g.merge(schedule_features(spine), on="game_id", how="left")

    elo = {}
    for k, ha, ret in ts.ELO_GRID:
        key = f"k{k}_h{ha}_r{ret}"
        elo[key] = ts.elo_all(spine, k, ha, ret)
    for alpha, ret in ts.OA_GRID:
        z, tag = ts.opponent_adjusted_features(g, alpha, ret)
        g = g.merge(z, on="game_id", how="left")

    fold_rows = []
    group_rows = []
    prediction_rows = []

    for target in TARGET_SEASONS:
        val_season = target - 1
        ec = None
        for key, pred in elo.items():
            vv = g[g.source_season.eq(val_season)][["game_id", "home_win"]].merge(
                pred[pred.source_season.eq(val_season)][["game_id", "elo_p"]],
                on="game_id",
            )
            if len(vv) < 100:
                continue
            ll = log_loss(vv.home_win, np.clip(vv.elo_p, 1e-6, 1 - 1e-6), labels=[0, 1])
            if ec is None or ll < ec[0]:
                ec = (ll, key)
        if ec is None:
            raise RuntimeError(f"No validation Elo candidate for target season {target}")

        x = g.merge(elo[ec[1]][["game_id", "elo_p"]], on="game_id", how="left")
        x["elo_logit"] = np.log(
            np.clip(x.elo_p, 1e-6, 1 - 1e-6)
            / np.clip(1 - x.elo_p, 1e-6, 1 - 1e-6)
        )
        train = x[x.source_season.le(target - 2)].copy()
        val = x[x.source_season.eq(val_season)].copy()
        test = x[x.source_season.eq(target)].copy()
        if min(len(train), len(val), len(test)) == 0:
            raise RuntimeError(f"Missing chronological fold rows for target season {target}")

        oa_tag, base = choose_base(train, val, x)
        # This is the tested B2B model and is held fixed as the comparison
        # baseline for each additional rest/travel feature.
        b2b = ["h_sched_b2b", "a_sched_b2b", "d_sched_b2b"]
        baseline = base + b2b
        specs = {
            "baseline_b2b": baseline,
            "plus_3in4": baseline + ["h_sched_3in4", "a_sched_3in4"],
            "plus_4in6": baseline + ["h_sched_4in6", "a_sched_4in6"],
            "plus_3in4_4in6": baseline + [
                "h_sched_3in4", "a_sched_3in4",
                "h_sched_4in6", "a_sched_4in6",
            ],
            "plus_5in8": baseline + ["h_sched_5in8", "a_sched_5in8"],
            "plus_road_trip_finale_proxy": baseline + ["a_sched_away_trip_finale"],
            "plus_all_schedule_context": baseline + [
                "h_sched_3in4", "a_sched_3in4",
                "h_sched_4in6", "a_sched_4in6",
                "h_sched_5in8", "a_sched_5in8",
                "a_sched_away_trip_finale",
            ],
        }

        conditions = {
            "either_team_3in4": test.h_sched_3in4.eq(1) | test.a_sched_3in4.eq(1),
            "either_team_4in6": test.h_sched_4in6.eq(1) | test.a_sched_4in6.eq(1),
            "either_team_5in8": test.h_sched_5in8.eq(1) | test.a_sched_5in8.eq(1),
            "away_road_trip_finale_proxy": test.a_sched_away_trip_finale.eq(1),
        }

        for name, features in specs.items():
            c, p = fit_tuned(train, val, test, features)
            m = metrics(test.home_win, p)
            fold_rows.append({
                "target_season": target, "model": name,
                "games": m["n"], "brier": m["brier"],
                "log_loss": m["log_loss"], "accuracy": m["accuracy"],
                "C": c, "elo_choice": ec[1], "oa_choice": oa_tag,
            })
            pred_frame = test[["game_id", "source_season", "home_win"]].copy()
            pred_frame["model"] = name
            pred_frame["p_home"] = p
            for condition, mask in conditions.items():
                pred_frame[condition] = mask.to_numpy()
                gm = metrics(test.loc[mask, "home_win"], p[mask.to_numpy()])
                group_rows.append({
                    "target_season": target, "model": name,
                    "condition": condition, **gm,
                })
            prediction_rows.append(pred_frame)

    folds = pd.DataFrame(fold_rows)
    preds = pd.concat(prediction_rows, ignore_index=True)
    groups = pd.DataFrame(group_rows)
    bootstrap_rows = []
    for candidate in [
        "plus_3in4", "plus_4in6", "plus_3in4_4in6",
        "plus_5in8", "plus_road_trip_finale_proxy",
        "plus_all_schedule_context",
    ]:
        bootstrap_rows.extend(paired_bootstrap(preds, candidate))
    bootstrap_rows.extend(paired_bootstrap(
        preds, "plus_road_trip_finale_proxy",
        condition="away_road_trip_finale_proxy",
    ))
    bootstrap = pd.DataFrame(bootstrap_rows)
    summary = {}
    for model, z in folds.groupby("model"):
        summary[model] = {
            "games": int(z.games.sum()),
            "weighted_brier": float(np.average(z.brier, weights=z.games)),
            "weighted_log_loss": float(np.average(z.log_loss, weights=z.games)),
            "weighted_accuracy": float(np.average(z.accuracy, weights=z.games)),
            "brier_by_season": {
                str(int(r.target_season)): float(r.brier) for r in z.itertuples()
            },
            "log_loss_by_season": {
                str(int(r.target_season)): float(r.log_loss) for r in z.itertuples()
            },
        }

    base_brier = summary["baseline_b2b"]["weighted_brier"]
    base_logloss = summary["baseline_b2b"]["weighted_log_loss"]
    for model, values in summary.items():
        values["delta_brier_vs_baseline_b2b"] = values["weighted_brier"] - base_brier
        values["delta_log_loss_vs_baseline_b2b"] = values["weighted_log_loss"] - base_logloss
        values["seasons_better_brier"] = int(sum(
            values["brier_by_season"][s] < summary["baseline_b2b"]["brier_by_season"][s]
            for s in summary["baseline_b2b"]["brier_by_season"]
        ))

    folds.to_csv(out / "rolling_schedule_007.csv", index=False)
    groups.to_csv(out / "schedule_007_condition_metrics.csv", index=False)
    bootstrap.to_csv(out / "paired_schedule_007_bootstrap.csv", index=False)
    preds.to_parquet(out / "schedule_007_test_predictions.parquet", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(folds.to_string(index=False))
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nCONDITION METRICS")
    print(groups.to_string(index=False))
    print("\nPAIRED BOOTSTRAP")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbp", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--goalie", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.pbp, args.spine, args.goalie, args.out)
