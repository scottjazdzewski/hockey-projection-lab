"""Rolling-origin goalie metric audit with game-level OOF scores and uncertainty.

The actual starter is selected from the postgame goalie record, so this is a
retrospective upper bound. Goalie skill inputs remain strictly prior to each game.
This research audit does not alter the promoted model.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "research")
import hwin_goalie_age_shrinkage_004 as goalie_base
import hwin_team_strength_004 as ts
sys.path.insert(0, "src/production")
import hockey_lab_game_v1 as prod

TARGET_SEASONS = [2022, 2023, 2024, 2025, 2026]
SHRINKAGE_GRID = [(400, 15, 1800)]  # promoted model calibration; isolate metric differences first
RIDGE_GRID = [0.0, 0.01, 0.1, 1.0]

def age_rate_curve(goalies, through_season, numerator, denominator, factor, weight_col):
    g = goalies.copy()
    for col in ["source_season", "age_years", numerator, denominator, weight_col]:
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g = g[(g["source_season"] <= through_season) & g["starter_bool"].fillna(False)
          & g["age_years"].between(18, 50) & (g[denominator] > 0)
          & g[numerator].notna()].copy()
    if len(g) < 100:
        raise RuntimeError(f"Insufficient prior rows to fit age curve for {numerator}: {len(g)}")
    age = g["age_years"].to_numpy(float)
    weight = g[weight_col].to_numpy(float)
    rate = g[numerator].to_numpy(float) / g[denominator].to_numpy(float) * factor
    good = np.isfinite(age) & np.isfinite(weight) & (weight > 0) & np.isfinite(rate)
    age, weight, rate = age[good], weight[good], rate[good]
    center = float(np.average(age, weights=weight))
    scale = float(np.sqrt(np.average((age-center)**2, weights=weight)))
    z = (age-center)/scale
    design = np.column_stack([np.ones(len(z)), z, z*z])
    coef = np.linalg.lstsq(design*np.sqrt(weight)[:,None], rate*np.sqrt(weight), rcond=None)[0]
    return {"center": center, "scale": scale, "age_min": float(age.min()),
            "age_max": float(age.max()), "coefficients": [float(v) for v in coef]}


def rate_curve_value(age, curve):
    if not np.isfinite(age):
        return np.nan
    age = float(np.clip(age, curve["age_min"], curve["age_max"]))
    z = (age-curve["center"])/curve["scale"]
    b0,b1,b2 = curve["coefficients"]
    return float(b0+b1*z+b2*z*z)


def metric_ratings(goalies, curve, numerator, denominator, factor, better_high,
                   shots_k, starts_k, minutes_k, rating_name):
    g = goalies[goalies["starter_bool"].fillna(False)].copy()
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")
    g["goalie_id_key"] = g["goalie_id_key"].astype("string")
    for col in ["age_years", "shots_against_pbp", numerator, denominator, "toi_seconds_box"]:
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g = g.sort_values(["goalie_id_key", "game_date", "game_id"]).reset_index(drop=True)
    g["_rate"] = [rate_curve_value(a,curve) for a in g.age_years.to_numpy(float)]
    g["_exposure"] = g[denominator].fillna(0).clip(lower=0)
    g["_observed"] = g[numerator].fillna(0)
    g["_expected"] = g["_rate"] * g["_exposure"] / factor
    rows = []
    for _, z in g.groupby("goalie_id_key", sort=False):
        shots = z["shots_against_pbp"].fillna(0).clip(lower=0).to_numpy(float)
        starts = np.arange(len(z), dtype=float)
        minutes = np.maximum(z["toi_seconds_box"].fillna(0).to_numpy(float)/60.0, 0)
        exposure = z["_exposure"].to_numpy(float)
        observed = z["_observed"].to_numpy(float)
        expected = z["_expected"].to_numpy(float)
        prior_exposure = np.concatenate([[0.0], np.cumsum(exposure)[:-1]])
        prior_observed = np.concatenate([[0.0], np.cumsum(observed)[:-1]])
        prior_expected = np.concatenate([[0.0], np.cumsum(expected)[:-1]])
        prior_shots = np.concatenate([[0.0], np.cumsum(shots)[:-1]])
        prior_minutes = np.concatenate([[0.0], np.cumsum(minutes)[:-1]])
        prior_starts = starts
        residual_sign = 1.0 if better_high else -1.0
        residual = np.divide(
            residual_sign*(prior_observed-prior_expected), prior_exposure,
            out=np.zeros(len(z), dtype=float), where=prior_exposure>0
        ) * factor
        shot_rel = np.divide(prior_shots, prior_shots+shots_k,
                             out=np.zeros(len(z), dtype=float), where=prior_shots+shots_k>0)
        start_rel = prior_starts/(prior_starts+starts_k) if starts_k>0 else np.where(prior_starts>0,1.0,0.0)
        minute_rel = prior_minutes/(prior_minutes+minutes_k) if minutes_k>0 else np.where(prior_minutes>0,1.0,0.0)
        reliability = (shot_rel+start_rel+minute_rel)/3.0
        # Include the learned age baseline plus a shrunk, prior-only personal residual.
        age_component = np.array([rate_curve_value(a,curve) for a in z.age_years.to_numpy(float)])
        rating = (age_component if better_high else -age_component) + reliability*residual
        block = z[["game_id","home_away","goalie_id_key"]].copy()
        block[rating_name] = rating
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def rating_differences(ratings, rating_name, diff_name):
    h = ratings[ratings.home_away.astype("string").str.lower().eq("home")][["game_id",rating_name]].drop_duplicates("game_id")
    a = ratings[ratings.home_away.astype("string").str.lower().eq("away")][["game_id",rating_name]].drop_duplicates("game_id")
    z = h.merge(a,on="game_id",how="inner",suffixes=("_home","_away"))
    z[diff_name] = z[f"{rating_name}_home"]-z[f"{rating_name}_away"]
    return z[["game_id",diff_name]]


def fit_coefficients(p, y, x, ridge):
    p = np.clip(np.asarray(p,float),1e-6,1-1e-6)
    y = np.asarray(y,float)
    x = np.asarray(x,float)
    if x.ndim == 1:
        x = x[:,None]
    scale = np.sqrt(np.mean(x*x,axis=0))
    scale = np.where(np.isfinite(scale)&(scale>1e-12),scale,1.0)
    z = x/scale
    base = np.log(p/(1-p))
    b = np.zeros(z.shape[1],float)
    for _ in range(100):
        eta = np.clip(base+z@b,-30,30)
        q = 1/(1+np.exp(-eta))
        grad = z.T@(q-y) + ridge*b
        w = q*(1-q)
        hess = z.T@(z*w[:,None]) + (ridge+1e-10)*np.eye(z.shape[1])
        step = np.linalg.solve(hess,grad)
        b -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b/scale


def apply_coefficients(p, x, beta):
    p = np.clip(np.asarray(p,float),1e-6,1-1e-6)
    x = np.asarray(x,float)
    if x.ndim == 1:
        x = x[:,None]
    logit = np.log(p/(1-p)) + x@np.asarray(beta,float)
    return 1/(1+np.exp(-np.clip(logit,-30,30)))


def select_features(train, val, feature_sets):
    best=None
    for feats in feature_sets:
        for c in [0.05,0.1,0.25,0.5,1.0,2.0]:
            pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                           ("m",LogisticRegression(C=c,max_iter=2500))])
            pipe.fit(train[feats],train.home_win.astype(int))
            pv=np.clip(pipe.predict_proba(val[feats])[:,1],1e-6,1-1e-6)
            ll=log_loss(val.home_win.astype(int),pv,labels=[0,1])
            if best is None or ll<best["ll"]:
                best={"ll":ll,"features":feats,"C":c}
    return best


def score_pair(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    return {
        "games": int(len(y)),
        "brier": float(np.mean((p-y)**2)),
        "log_loss": float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))),
        "accuracy": float(np.mean((p >= 0.5) == (y >= 0.5)))
    }


def paired_game_bootstrap(values, seed, n_boot=2000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    chunk = 128
    for start in range(0, n_boot, chunk):
        stop = min(start+chunk, n_boot)
        idx = rng.integers(0, len(values), size=(stop-start, len(values)))
        draws[start:stop] = values[idx].mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def team_season_bootstrap(df, value_col, seed, n_boot=2000):
    """Pigeonhole bootstrap teams within each held-out season; game deltas stay paired."""
    rng = np.random.default_rng(seed)
    season_arrays = []
    season_sizes = []
    for _, group in df.groupby("target_season", sort=True):
        teams = np.unique(np.concatenate([
            group.home_team_abbr.astype(str).to_numpy(),
            group.away_team_abbr.astype(str).to_numpy()
        ]))
        team_idx = {team:i for i,team in enumerate(teams)}
        hi = group.home_team_abbr.astype(str).map(team_idx).to_numpy(int)
        ai = group.away_team_abbr.astype(str).map(team_idx).to_numpy(int)
        values = group[value_col].to_numpy(float)
        draws = np.empty(n_boot, dtype=float)
        chunk = 128
        for start in range(0, n_boot, chunk):
            stop = min(start+chunk, n_boot)
            size = stop-start
            sampled = rng.integers(0, len(teams), size=(size, len(teams)))
            counts = np.zeros((size, len(teams)), dtype=np.int16)
            row_idx = np.repeat(np.arange(size), len(teams))
            np.add.at(counts, (row_idx, sampled.reshape(-1)), 1)
            weights = counts[:,hi].astype(float) * counts[:,ai].astype(float)
            denom = weights.sum(axis=1)
            good = denom > 0
            block = np.full(size, np.nan, dtype=float)
            block[good] = (weights[good] @ values) / denom[good]
            draws[start:stop] = block
        season_arrays.append(draws)
        season_sizes.append(len(group))
    matrix = np.vstack(season_arrays)
    pooled = np.average(matrix, axis=0, weights=np.asarray(season_sizes, dtype=float))
    pooled = pooled[np.isfinite(pooled)]
    return [float(x) for x in np.quantile(pooled, [0.025, 0.975])]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(pbp_path,spine_path,goalie_path,outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    pbp=pd.read_parquet(pbp_path)
    spine=pd.read_parquet(spine_path)
    goalies=pd.read_parquet(goalie_path)
    spine=spine[spine.game_type.astype("string").isin(["R","P"])].copy()
    spine["game_date"]=pd.to_datetime(spine.game_date,errors="coerce")
    spine["source_season"]=pd.to_numeric(spine.source_season,errors="coerce")
    spine["home_win"]=pd.to_numeric(spine.home_win,errors="coerce")
    games=ts.add_flat_and_ewm_features(ts.game_event_table(pbp))
    x=games.merge(prod.build_schedule_features(spine),on="game_id",how="left")
    for alpha,retain in ts.OA_GRID:
        oa,tag=ts.opponent_adjusted_features(games,alpha,retain)
        x=x.merge(oa,on="game_id",how="left")
    elo={}
    for k,ha,retain in ts.ELO_GRID:
        elo[f"k{k}_h{ha}_r{retain}"]=prod.elo_with_future(spine,k,ha,retain)

    all_rows=[]
    oof_rows=[]
    for target in TARGET_SEASONS:
        vs=target-1
        ec=None
        for key,pred in elo.items():
            v=spine[spine.source_season.eq(vs)][["game_id","home_win"]].merge(
                pred[pred.source_season.eq(vs)][["game_id","elo_p"]],on="game_id",how="inner")
            if len(v)<100: continue
            ll=log_loss(v.home_win.astype(int),np.clip(v.elo_p,1e-6,1-1e-6),labels=[0,1])
            if ec is None or ll<ec[0]: ec=(ll,key)
        if ec is None: continue
        z=x.merge(elo[ec[1]][["game_id","elo_p"]],on="game_id",how="left")
        z["elo_logit"]=np.log(np.clip(z.elo_p,1e-6,1-1e-6)/np.clip(1-z.elo_p,1e-6,1-1e-6))
        train=z[z.source_season.le(target-2)&z.home_win.notna()].copy()
        val=z[z.source_season.eq(vs)&z.home_win.notna()].copy()
        test=z[z.source_season.eq(target)&z.home_win.notna()].copy()
        production_feature_sets=[]
        for alpha,retain in ts.OA_GRID:
            tag=f"a{alpha:.2f}_r{retain:.2f}".replace(".","p")
            production_feature_sets.append([
                "elo_logit","d_sog_share_r20","d_xg_share_r20","d_goals_r20","d_rest_days",
                f"oa_sog_diff_{tag}",f"oa_xg_diff_{tag}",
                "h_sched_b2b","a_sched_b2b","d_sched_b2b"])
        best=select_features(train,val,production_feature_sets)
        feats=best["features"]; C=best["C"]
        bpv,bpt=goalie_base.fit_base(train,val,test,feats,C)
        val_base=val[["game_id","home_win"]].copy(); val_base["base_p"]=bpv
        test_base=test[["game_id","home_win"]].copy(); test_base["base_p"]=bpt

        sv_curve=goalie_base.fit_age_curve(goalies,through_season=target-2)
        g=goalies.copy()
        g["_minutes"]=pd.to_numeric(g.toi_seconds_box,errors="coerce")/60.0
        gaa_curve=age_rate_curve(g,target-2,"goals_against_pbp","_minutes",60.0,"_minutes")
        gsax_curve=age_rate_curve(g,target-2,"sog_gsax","shots_against_pbp",30.0,"shots_against_pbp")
        variants={
            "sv%":["d_sv"],
            "gaa":["d_gaa"],
            "sog_gsax":["d_sog_gsax"],
            "sv%+gaa":["d_sv","d_gaa"],
            "sv%+sog_gsax":["d_sv","d_sog_gsax"],
        }
        best_variant={}
        for sk,nk,mk in SHRINKAGE_GRID:
            sv=goalie_base.goalie_ratings(g,sv_curve,sk,nk,mk)
            sv_diff=goalie_base.add_goalie_diff(z,sv)[["game_id","d_goalie_rating"]].rename(columns={"d_goalie_rating":"d_sv"})
            gaa=metric_ratings(g,gaa_curve,"goals_against_pbp","_minutes",60.0,False,sk,nk,mk,"gaa_rating")
            gaa_diff=rating_differences(gaa,"gaa_rating","d_gaa")
            gsax=metric_ratings(g,gsax_curve,"sog_gsax","shots_against_pbp",30.0,True,sk,nk,mk,"gsax_rating")
            gsax_diff=rating_differences(gsax,"gsax_rating","d_sog_gsax")
            tables={"d_sv":sv_diff,"d_gaa":gaa_diff,"d_sog_gsax":gsax_diff}
            for variant, columns in variants.items():
                v=val_base.copy(); t=test_base.copy()
                for col in columns:
                    v=v.merge(tables[col],on="game_id",how="inner")
                    t=t.merge(tables[col],on="game_id",how="inner")
                if len(v)<100 or len(t)<100: continue
                Xv=v[columns].to_numpy(float); yv=v.home_win.to_numpy(int); pv=v.base_p.to_numpy(float)
                for ridge in RIDGE_GRID:
                    beta=fit_coefficients(pv,yv,Xv,ridge)
                    adj=apply_coefficients(pv,Xv,beta)
                    ll=log_loss(yv,adj,labels=[0,1])
                    current=best_variant.get(variant)
                    if current is None or ll<current["validation_log_loss"]:
                        best_variant[variant]={"validation_log_loss":float(ll),"shots_k":sk,
                            "starts_k":nk,"minutes_k":mk,"ridge":ridge,"beta":beta.tolist()}
        for variant, selection in best_variant.items():
            sk,nk,mk=selection["shots_k"],selection["starts_k"],selection["minutes_k"]
            sv=goalie_base.goalie_ratings(g,sv_curve,sk,nk,mk)
            sv_diff=goalie_base.add_goalie_diff(z,sv)[["game_id","d_goalie_rating"]].rename(columns={"d_goalie_rating":"d_sv"})
            gaa=metric_ratings(g,gaa_curve,"goals_against_pbp","_minutes",60.0,False,sk,nk,mk,"gaa_rating")
            gaa_diff=rating_differences(gaa,"gaa_rating","d_gaa")
            gsax=metric_ratings(g,gsax_curve,"sog_gsax","shots_against_pbp",30.0,True,sk,nk,mk,"gsax_rating")
            gsax_diff=rating_differences(gsax,"gsax_rating","d_sog_gsax")
            tables={"d_sv":sv_diff,"d_gaa":gaa_diff,"d_sog_gsax":gsax_diff}
            v=test_base.copy()
            columns=variants[variant]
            for col in columns:
                v=v.merge(tables[col],on="game_id",how="inner")
            y=v.home_win.to_numpy(int); p=v.base_p.to_numpy(float); X=v[columns].to_numpy(float)
            beta=np.asarray(selection["beta"],float)
            adjusted=apply_coefficients(p,X,beta)
            baseline=goalie_base.metrics(y,p)
            model=goalie_base.metrics(y,adjusted)
            row={"target_season":target,"variant":variant,"games":len(v),
                 "shots_k":sk,"starts_k":nk,"minutes_k":mk,"ridge":selection["ridge"],
                 "beta":selection["beta"],"validation_log_loss":selection["validation_log_loss"],
                 "baseline_brier":baseline["brier"],"model_brier":model["brier"],
                 "delta_brier":model["brier"]-baseline["brier"],
                 "baseline_log_loss":baseline["log_loss"],"model_log_loss":model["log_loss"],
                 "delta_log_loss":model["log_loss"]-baseline["log_loss"]}
            all_rows.append(row)
            exposure=goalie_base.add_goalie_diff(test_base,sv)[[
                "game_id","h_prior_starts","a_prior_starts",
                "h_prior_shots","a_prior_shots","h_prior_minutes","a_prior_minutes"
            ]]
            game=v[["game_id","home_win","base_p"]].copy()
            game["target_season"]=int(target)
            game["variant"]=variant
            game["adjusted_p"]=adjusted
            for col in columns:
                game["feature_"+col]=v[col].to_numpy(float)
            game=game.merge(exposure,on="game_id",how="left")
            teams=spine.loc[spine.source_season.eq(target),[
                "game_id","home_team_abbr","away_team_abbr"
            ]].drop_duplicates("game_id")
            game=game.merge(teams,on="game_id",how="left")
            oof_rows.append(game)
        print(json.dumps({"target_season":target,
                          "folds":[r for r in all_rows if r["target_season"]==target]}), flush=True)
    result=pd.DataFrame(all_rows)
    result.to_csv(out/"goalie_metric_comparison.csv",index=False)
    summary=[]
    for variant, group in result.groupby("variant"):
        n=group.games.to_numpy(float)
        baseline_brier=float(np.average(group.baseline_brier,weights=n))
        model_brier=float(np.average(group.model_brier,weights=n))
        baseline_ll=float(np.average(group.baseline_log_loss,weights=n))
        model_ll=float(np.average(group.model_log_loss,weights=n))
        summary.append({"variant":variant,"games":int(group.games.sum()),
                        "weighted_baseline_brier":baseline_brier,"weighted_model_brier":model_brier,
                        "delta_brier":model_brier-baseline_brier,
                        "weighted_baseline_log_loss":baseline_ll,"weighted_model_log_loss":model_ll,
                        "delta_log_loss":model_ll-baseline_ll,
                        "improved_folds_brier":int((group.delta_brier<0).sum()),
                        "improved_folds_log_loss":int((group.delta_log_loss<0).sum())})
    oof=pd.concat(oof_rows,ignore_index=True)
    variants=sorted(oof.variant.unique().tolist())
    common_by_season={}
    for season in TARGET_SEASONS:
        sets=[set(oof.loc[oof.target_season.eq(season)&oof.variant.eq(v),"game_id"]) for v in variants]
        if not sets or any(len(x)==0 for x in sets):
            raise RuntimeError(f"Missing a goalie variant in target season {season}")
        common_by_season[int(season)]=set.intersection(*sets)
    allowed={(season,gid) for season,gameset in common_by_season.items() for gid in gameset}
    oof=oof[[(int(s),gid) in allowed for s,gid in zip(oof.target_season,oof.game_id)]].copy()
    if oof.empty:
        raise RuntimeError("No common game rows across goalie metric variants")
    oof["home_win"]=pd.to_numeric(oof.home_win,errors="raise").astype(int)
    oof["min_prior_starts"]=np.minimum(
        pd.to_numeric(oof.h_prior_starts,errors="coerce"),
        pd.to_numeric(oof.a_prior_starts,errors="coerce")
    )
    oof["min_prior_shots"]=np.minimum(
        pd.to_numeric(oof.h_prior_shots,errors="coerce"),
        pd.to_numeric(oof.a_prior_shots,errors="coerce")
    )
    oof["exposure_tier"]=pd.cut(
        oof.min_prior_starts,[-1,5,15,40,np.inf],
        labels=["0-4","5-14","15-39","40+"],right=False
    ).astype("string")
    y=oof.home_win.to_numpy(float)
    p0=np.clip(oof.base_p.to_numpy(float),1e-6,1-1e-6)
    p1=np.clip(oof.adjusted_p.to_numpy(float),1e-6,1-1e-6)
    oof["delta_brier"]=(p1-y)**2-(p0-y)**2
    oof["delta_log_loss"]=-(y*np.log(p1)+(1-y)*np.log(1-p1)) + (y*np.log(p0)+(1-y)*np.log(1-p0))
    oof.to_parquet(out/"matched_oof_predictions.parquet",index=False)
    fold_rows=[]
    for (season,variant),group in oof.groupby(["target_season","variant"],sort=True):
        base=score_pair(group.home_win,group.base_p)
        adj=score_pair(group.home_win,group.adjusted_p)
        fold_rows.append({
            "target_season":int(season),"variant":variant,"games":len(group),
            "baseline_brier":base["brier"],"adjusted_brier":adj["brier"],
            "delta_brier":adj["brier"]-base["brier"],
            "baseline_log_loss":base["log_loss"],"adjusted_log_loss":adj["log_loss"],
            "delta_log_loss":adj["log_loss"]-base["log_loss"],
            "baseline_accuracy":base["accuracy"],"adjusted_accuracy":adj["accuracy"],
            "delta_accuracy":adj["accuracy"]-base["accuracy"]
        })
    fold_frame=pd.DataFrame(fold_rows)
    fold_frame.to_csv(out/"matched_fold_metrics.csv",index=False)
    uncertainty=[]
    for i,variant in enumerate(variants):
        group=oof[oof.variant.eq(variant)].copy()
        b=group.delta_brier.to_numpy(float)
        ll=group.delta_log_loss.to_numpy(float)
        fold_group=fold_frame[fold_frame.variant.eq(variant)]
        uncertainty.append({
            "variant":variant,"games":len(group),
            "delta_brier":float(b.mean()),
            "paired_game_bootstrap_brier_95pct":paired_game_bootstrap(b,4100+i),
            "team_season_bootstrap_brier_95pct":team_season_bootstrap(group,"delta_brier",5100+i),
            "delta_log_loss":float(ll.mean()),
            "paired_game_bootstrap_log_loss_95pct":paired_game_bootstrap(ll,6100+i),
            "team_season_bootstrap_log_loss_95pct":team_season_bootstrap(group,"delta_log_loss",7100+i),
            "improved_folds_brier":int((fold_group.delta_brier<0).sum()),
            "improved_folds_log_loss":int((fold_group.delta_log_loss<0).sum())
        })
    pd.DataFrame(uncertainty).to_csv(out/"paired_uncertainty.csv",index=False)
    exposure_rows=[]
    for (variant,tier),group in oof.groupby(["variant","exposure_tier"],dropna=False,sort=True):
        if len(group)==0: continue
        base=score_pair(group.home_win,group.base_p)
        adj=score_pair(group.home_win,group.adjusted_p)
        exposure_rows.append({
            "variant":variant,"worst_goalie_prior_start_tier":str(tier),
            "games":len(group),"baseline_brier":base["brier"],"adjusted_brier":adj["brier"],
            "delta_brier":adj["brier"]-base["brier"],
            "baseline_log_loss":base["log_loss"],"adjusted_log_loss":adj["log_loss"],
            "delta_log_loss":adj["log_loss"]-base["log_loss"],
            "mean_min_prior_shots":float(group.min_prior_shots.mean())
        })
    pd.DataFrame(exposure_rows).to_csv(out/"exposure_strata.csv",index=False)
    manifest={
        "schema_version":"1.0",
        "analysis":"retrospective actual-starter identity; prior-only goalie skill",
        "source_workflow_runs":{
            "pbp":os.getenv("PBP_RUN_ID"),
            "spine":os.getenv("SPINE_RUN_ID"),
            "goalie":os.getenv("GOALIE_RUN_ID")
        },
        "inputs":{
            "pbp":{"path":str(pbp_path),"sha256":sha256_file(pbp_path)},
            "spine":{"path":str(spine_path),"sha256":sha256_file(spine_path)},
            "goalie":{"path":str(goalie_path),"sha256":sha256_file(goalie_path)}
        },
        "matched_oof_games_per_variant":int(oof[oof.variant.eq(variants[0])].game_id.nunique()),
        "target_seasons":TARGET_SEASONS,
        "bootstrap_resamples":2000,
        "cluster_method":"pigeonhole bootstrap of team identities within each target season; paired game deltas"
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\\n")
    output={
        "schema_version":"1.0",
        "baseline_feature_version":prod.FEATURE_VERSION,
        "target_seasons":TARGET_SEASONS,
        "games_per_variant":int(result[result.variant.eq("sv%")].games.sum()),
        "exposure_shrinkage_fixed_for_comparison":{
            "shots_k":400,"starts_k":15,"minutes_k":1800
        },
        "weight_fit":"validation season within each rolling-origin fold",
        "variants":summary
    }
    (out/"summary.json").write_text(json.dumps(output,indent=2)+"\n")
    matched_metrics={}
    for variant,group in fold_frame.groupby("variant",sort=True):
        matched_metrics[variant]={
            "games":int(group.games.sum()),
            "baseline_brier":float(np.average(group.baseline_brier,weights=group.games)),
            "adjusted_brier":float(np.average(group.adjusted_brier,weights=group.games)),
            "delta_brier":float(np.average(group.delta_brier,weights=group.games)),
            "baseline_log_loss":float(np.average(group.baseline_log_loss,weights=group.games)),
            "adjusted_log_loss":float(np.average(group.adjusted_log_loss,weights=group.games)),
            "delta_log_loss":float(np.average(group.delta_log_loss,weights=group.games))
        }
    output["matched_oof_metrics"]=matched_metrics
    output["uncertainty"]=uncertainty
    output["exposure_strata_file"]="exposure_strata.csv"
    output["starter_identity_caveat"]=(
        "Current game starter identity comes from the postgame box. These results "
        "are a retrospective upper bound, not a pregame starter backtest."
    )
    (out/"summary.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--pbp",required=True); p.add_argument("--spine",required=True)
    p.add_argument("--goalie",required=True); p.add_argument("--out",required=True)
    a=p.parse_args(); main(a.pbp,a.spine,a.goalie,a.out)
