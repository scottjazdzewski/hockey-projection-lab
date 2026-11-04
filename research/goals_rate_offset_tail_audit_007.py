"""Paired diagnostics for the nested 2022-2026 goal-rate offset OOF forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE = "hockey_process_plus_league_scoring_validation_selected_distribution"
OFFSET = "hockey_process_plus_league_scoring_rate_offset_validation_selected_distribution"
BOOTSTRAPS = 2000


def clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-12, 1-1e-12)


def binary_log_loss(y, p):
    y = np.asarray(y, dtype=float)
    p = clip(p)
    return -(y*np.log(p) + (1-y)*np.log(1-p))


def bootstrap_ci(values, seed, n_boot=BOOTSTRAPS):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    chunk = 128
    for start in range(0, n_boot, chunk):
        stop = min(start+chunk, n_boot)
        idx = rng.integers(0, len(values), size=(stop-start, len(values)))
        draws[start:stop] = values[idx].mean(axis=1)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def team_season_bootstrap(df, value_col, seed, n_boot=BOOTSTRAPS):
    """Paired pigeonhole bootstrap of team identities within each target season."""
    rng = np.random.default_rng(seed)
    folds, weights_by_fold = [], []
    for _, group in df.groupby("source_season", sort=True):
        teams = np.unique(np.concatenate([
            group.home_team.astype(str).to_numpy(),
            group.away_team.astype(str).to_numpy(),
        ]))
        idx = {name:i for i,name in enumerate(teams)}
        hi = group.home_team.astype(str).map(idx).to_numpy(int)
        ai = group.away_team.astype(str).map(idx).to_numpy(int)
        values = group[value_col].to_numpy(float)
        draws = np.empty(n_boot, dtype=float)
        chunk = 128
        for start in range(0, n_boot, chunk):
            stop = min(start+chunk, n_boot)
            size = stop-start
            sampled = rng.integers(0, len(teams), size=(size, len(teams)))
            counts = np.zeros((size, len(teams)), dtype=np.int16)
            row = np.repeat(np.arange(size), len(teams))
            np.add.at(counts, (row, sampled.reshape(-1)), 1)
            game_weights = counts[:,hi].astype(float)*counts[:,ai].astype(float)
            denom = game_weights.sum(axis=1)
            valid = denom > 0
            block = np.full(size, np.nan, dtype=float)
            block[valid] = (game_weights[valid] @ values)/denom[valid]
            draws[start:stop] = block
        folds.append(draws)
        weights_by_fold.append(len(group))
    pooled = np.average(np.vstack(folds), axis=0, weights=np.asarray(weights_by_fold, dtype=float))
    pooled = pooled[np.isfinite(pooled)]
    return [float(v) for v in np.quantile(pooled, [0.025, 0.975])]


def metric_summary(y, p):
    y = np.asarray(y, dtype=float)
    p = clip(p)
    return {
        "brier": float(np.mean((p-y)**2)),
        "log_loss": float(np.mean(binary_log_loss(y,p))),
        "accuracy": float(np.mean((p >= 0.5) == (y >= 0.5))),
        "predicted_rate": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


def paired_model_rows(oof):
    required = {
        "game_id", "source_season", "home_team", "away_team", "model_variant",
        "reg_home_goals", "reg_away_goals", "official_final_home_goals",
        "official_final_away_goals", "went_ot", "exp_final_home_goals",
        "exp_final_away_goals", "p_reg_home_win", "p_reg_away_win", "p_reg_tie",
        "p_overtime", "p_full_home_win", "logp_final_score", "logp_final_total",
        "p_final_over_5_5", "p_final_over_6_5",
    }
    missing = sorted(required-set(oof.columns))
    if missing:
        raise ValueError(f"OOF file missing columns: {missing}")
    base = oof.loc[oof.model_variant.eq(BASE)].copy()
    offset = oof.loc[oof.model_variant.eq(OFFSET)].copy()
    key = ["source_season", "game_id"]
    if base.duplicated(key).any() or offset.duplicated(key).any():
        raise ValueError("Duplicate game-season keys in selected model variants")
    if set(map(tuple,base[key].to_numpy())) != set(map(tuple,offset[key].to_numpy())):
        raise ValueError("Baseline and offset forecasts do not cover the same games")
    pairs = base.merge(offset, on=key, suffixes=("_base","_offset"), validate="one_to_one")
    for col in [
        "reg_home_goals", "reg_away_goals", "official_final_home_goals",
        "official_final_away_goals", "went_ot",
    ]:
        if not np.array_equal(pairs[f"{col}_base"].to_numpy(),pairs[f"{col}_offset"].to_numpy()):
            raise ValueError(f"Outcome mismatch between variants for {col}")
    pairs["home_team"] = pairs["home_team_base"]
    pairs["away_team"] = pairs["away_team_base"]
    pairs["actual_total"] = (
        pairs.official_final_home_goals_base.astype(float)
        + pairs.official_final_away_goals_base.astype(float)
    )
    pairs["y_over_5_5"] = (pairs.actual_total > 5.5).astype(int)
    pairs["y_over_6_5"] = (pairs.actual_total > 6.5).astype(int)
    pairs["y_home_win"] = (
        pairs.official_final_home_goals_base > pairs.official_final_away_goals_base
    ).astype(int)
    pairs["y_ot"] = pairs.went_ot_base.astype(int)
    pairs["y_reg_home"] = (pairs.reg_home_goals_base > pairs.reg_away_goals_base).astype(int)
    pairs["y_reg_away"] = (pairs.reg_home_goals_base < pairs.reg_away_goals_base).astype(int)
    pairs["y_reg_tie"] = (pairs.reg_home_goals_base == pairs.reg_away_goals_base).astype(int)
    return pairs


def add_paired_losses(p):
    for model in ("base","offset"):
        total_hat = (
            p[f"exp_final_home_goals_{model}"].astype(float)
            + p[f"exp_final_away_goals_{model}"].astype(float)
        )
        p[f"total_abs_error_{model}"] = np.abs(total_hat-p.actual_total)
        p[f"total_bias_{model}"] = total_hat-p.actual_total
        p[f"final_score_loss_{model}"] = -p[f"logp_final_score_{model}"].astype(float)
        p[f"final_total_loss_{model}"] = -p[f"logp_final_total_{model}"].astype(float)
        for threshold in ("5_5","6_5"):
            y=p[f"y_over_{threshold[0]}.{threshold[-1]}"].to_numpy(float)
            prob=clip(p[f"p_final_over_{threshold}_{model}"])
            p[f"over_{threshold}_brier_{model}"]=(prob-y)**2
            p[f"over_{threshold}_logloss_{model}"]=binary_log_loss(y,prob)
        ph=clip(p[f"p_full_home_win_{model}"])
        yh=p.y_home_win.to_numpy(float)
        p[f"home_win_brier_{model}"]=(ph-yh)**2
        p[f"home_win_logloss_{model}"]=binary_log_loss(yh,ph)
        po=clip(p[f"p_overtime_{model}"])
        yo=p.y_ot.to_numpy(float)
        p[f"ot_brier_{model}"]=(po-yo)**2
        p[f"ot_logloss_{model}"]=binary_log_loss(yo,po)
        p3=p[[f"p_reg_away_win_{model}",f"p_reg_tie_{model}",f"p_reg_home_win_{model}"]].to_numpy(float)
        y3=np.column_stack([p.y_reg_away,p.y_reg_tie,p.y_reg_home]).astype(float)
        p[f"reg_3way_brier_{model}"]=np.sum((p3-y3)**2,axis=1)
        p[f"reg_3way_logloss_{model}"]=-np.log(np.clip(np.sum(p3*y3,axis=1),1e-12,1))
    for metric in [
        "total_abs_error","total_bias","final_score_loss","final_total_loss",
        "over_5_5_brier","over_5_5_logloss","over_6_5_brier","over_6_5_logloss",
        "home_win_brier","home_win_logloss","ot_brier","ot_logloss",
        "reg_3way_brier","reg_3way_logloss",
    ]:
        p[f"delta_{metric}"]=p[f"{metric}_offset"]-p[f"{metric}_base"]
    return p


def reliability_deciles(pairs):
    rows=[]
    for name in ("base","offset"):
        for threshold in ("5_5","6_5"):
            prob_col=f"p_final_over_{threshold}_{name}"
            y_col=f"y_over_{threshold[0]}.{threshold[-1]}"
            z=pairs[[prob_col,y_col,"source_season"]].copy()
            z["bin"]=pd.qcut(z[prob_col],q=10,labels=False,duplicates="drop")
            for b,g in z.groupby("bin",dropna=True):
                rows.append({
                    "model":name,"threshold":threshold,"decile":int(b),"games":len(g),
                    "predicted":float(g[prob_col].mean()),"observed":float(g[y_col].mean()),
                    "mean_probability":float(g[prob_col].mean()),
                    "mean_source_season":float(g.source_season.mean()),
                })
    return pd.DataFrame(rows)


def main(oof_path,outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    oof=pd.read_parquet(oof_path)
    pairs=add_paired_losses(paired_model_rows(oof))
    pairs.to_parquet(out/"paired_game_deltas.parquet",index=False)
    deltas=[
        "total_abs_error","total_bias","final_score_loss","final_total_loss",
        "over_5_5_brier","over_5_5_logloss","over_6_5_brier","over_6_5_logloss",
        "home_win_brier","home_win_logloss","ot_brier","ot_logloss",
        "reg_3way_brier","reg_3way_logloss",
    ]
    overall=[]
    for i,metric in enumerate(deltas):
        v=pairs[f"delta_{metric}"].to_numpy(float)
        row={"metric":metric,"games":len(v),"delta_offset_minus_baseline":float(v.mean()),
             "paired_game_bootstrap_95pct":bootstrap_ci(v,7000+i)}
        if metric in {"total_abs_error","final_total_loss","over_5_5_brier","over_6_5_brier",
                      "home_win_brier","reg_3way_brier","ot_brier"}:
            row["team_season_bootstrap_95pct"]=team_season_bootstrap(pairs,f"delta_{metric}",8000+i)
        overall.append(row)
    overall_df=pd.DataFrame(overall)
    overall_df.to_csv(out/"paired_metric_deltas.csv",index=False)
    season_rows=[]
    for season,g in pairs.groupby("source_season",sort=True):
        for metric in deltas:
            season_rows.append({
                "source_season":int(season),"metric":metric,"games":len(g),
                "delta_offset_minus_baseline":float(g[f"delta_{metric}"].mean())
            })
    pd.DataFrame(season_rows).to_csv(out/"season_deltas.csv",index=False)
    by_model=[]
    for name in ("base","offset"):
        y55=pairs.y_over_5_5.to_numpy(float)
        y65=pairs.y_over_6_5.to_numpy(float)
        summary={
            "model":name,"games":len(pairs),
            "final_total_bias":float(pairs[f"total_bias_{name}"].mean()),
            "final_total_mae":float(pairs[f"total_abs_error_{name}"].mean()),
            "final_total_log_loss":float(pairs[f"final_total_loss_{name}"].mean()),
            "final_score_log_loss":float(pairs[f"final_score_loss_{name}"].mean()),
            "full_game_home_win_brier":float(pairs[f"home_win_brier_{name}"].mean()),
            "full_game_home_win_log_loss":float(pairs[f"home_win_logloss_{name}"].mean()),
            "reg_3way_brier":float(pairs[f"reg_3way_brier_{name}"].mean()),
            "overtime_brier":float(pairs[f"ot_brier_{name}"].mean()),
            "over_5_5":metric_summary(y55,pairs[f"p_final_over_5_5_{name}"]),
            "over_6_5":metric_summary(y65,pairs[f"p_final_over_6_5_{name}"]),
        }
        by_model.append(summary)
    by_season=[]
    for season,g in pairs.groupby("source_season",sort=True):
        for name in ("base","offset"):
            by_season.append({
                "source_season":int(season),"model":name,"games":len(g),
                "total_bias":float(g[f"total_bias_{name}"].mean()),
                "total_mae":float(g[f"total_abs_error_{name}"].mean()),
                "total_log_loss":float(g[f"final_total_loss_{name}"].mean()),
                "score_log_loss":float(g[f"final_score_loss_{name}"].mean()),
                "home_win_brier":float(g[f"home_win_brier_{name}"].mean()),
                "reg_3way_brier":float(g[f"reg_3way_brier_{name}"].mean()),
                "over_5_5_brier":float(g[f"over_5_5_brier_{name}"].mean()),
                "over_6_5_brier":float(g[f"over_6_5_brier_{name}"].mean()),
            })
    pd.DataFrame(by_season).to_csv(out/"season_metrics.csv",index=False)
    rel=reliability_deciles(pairs)
    rel.to_csv(out/"tail_reliability_deciles.csv",index=False)
    expected_total_base=(
        pairs.exp_final_home_goals_base.astype(float)+pairs.exp_final_away_goals_base.astype(float)
    )
    pairs["base_expected_total_decile"]=pd.qcut(expected_total_base,10,labels=False,duplicates="drop")
    context=[]
    for dec,g in pairs.groupby("base_expected_total_decile",dropna=True):
        for name in ("base","offset"):
            context.append({
                "baseline_expected_total_decile":int(dec),"model":name,"games":len(g),
                "total_bias":float(g[f"total_bias_{name}"].mean()),
                "over_5_5_predicted":float(g[f"p_final_over_5_5_{name}"].mean()),
                "over_5_5_observed":float(g.y_over_5_5.mean()),
                "over_6_5_predicted":float(g[f"p_final_over_6_5_{name}"].mean()),
                "over_6_5_observed":float(g.y_over_6_5.mean()),
                "full_game_home_win_brier":float(g[f"home_win_brier_{name}"].mean()),
            })
    pd.DataFrame(context).to_csv(out/"scoring_environment_slices.csv",index=False)
    manifest={
        "schema_version":"1.0","source_oof_path":str(oof_path),
        "oof_sha256":hashlib.sha256(Path(oof_path).read_bytes()).hexdigest(),
        "source_workflow_runs":{"pbp":os.getenv("PBP_RUN_ID"),"spine":os.getenv("SPINE_RUN_ID")},
        "baseline_variant":BASE,"offset_variant":OFFSET,"target_seasons":sorted(pairs.source_season.unique().tolist()),
        "games":int(pairs.game_id.nunique()),"bootstrap_resamples":BOOTSTRAPS,
        "starter_goalie_status":"excluded from this historical score-distribution experiment",
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    summary={
        "experiment":"Goal-rate Offset 006 tail and win-probability diagnostic 007",
        "baseline_variant":BASE,"candidate_variant":OFFSET,
        "overall_models":by_model,"paired_deltas":overall,
        "primary_read":"Offset candidate is scored separately on total-goal calibration and game-outcome probabilities.",
        "interpretation_guard":"A total-goal calibration gain is not a win-probability gain if home-win or regulation-outcome Brier worsens.",
        "manifest":manifest,
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--oof",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    main(args.oof,args.out)
