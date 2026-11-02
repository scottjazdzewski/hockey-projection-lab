import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

C_GRID=[0.05,0.1,0.25,0.5,1.0,2.0]
ELO_GRID=[(5,20,.7),(10,20,.7),(10,40,.7),(5,40,.7),(10,20,.55),(10,40,.55),(10,20,.85)]
HALFLIVES=[5,10,20,35]
OA_GRID=[(.03,.55),(.05,.55),(.08,.55),(.05,.70),(.08,.70),(.12,.70),(.05,.85),(.08,.85),(.12,.85)]

def met(y,p):
    y=np.asarray(y,dtype=int)
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return {
        "n":int(len(y)),
        "brier":float(brier_score_loss(y,p)),
        "log_loss":float(log_loss(y,p,labels=[0,1])),
        "accuracy":float(accuracy_score(y,p>=.5))
    }

def fit_tuned(train,val,test,features):
    best=None
    for C in C_GRID:
        pipe=Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("sc",StandardScaler()),
            ("m",LogisticRegression(C=C,max_iter=1800))
        ])
        pipe.fit(train[features],train.home_win)
        pv=pipe.predict_proba(val[features])[:,1]
        ll=log_loss(val.home_win,np.clip(pv,1e-6,1-1e-6),labels=[0,1])
        if best is None or ll<best[0]:
            best=(ll,C)
    C=best[1]
    fit=pd.concat([train,val],ignore_index=True)
    pipe=Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("sc",StandardScaler()),
        ("m",LogisticRegression(C=C,max_iter=1800))
    ])
    pipe.fit(fit[features],fit.home_win)
    pt=pipe.predict_proba(test[features])[:,1]
    return C,met(test.home_win,pt)

def elo_prob(rh,ra,ha):
    return 1/(1+10**(-(rh+ha-ra)/400.0))

def elo_all(spine,k,ha,retain):
    ratings={}; prev_season=None; rows=[]
    for r in spine.sort_values(["game_date","game_id"]).itertuples(index=False):
        season=int(r.source_season)
        if prev_season is None:
            prev_season=season
        elif season!=prev_season:
            ratings={t:1500+retain*(x-1500) for t,x in ratings.items()}
            prev_season=season
        h=str(r.home_team_abbr); a=str(r.away_team_abbr)
        rh=ratings.get(h,1500.); ra=ratings.get(a,1500.)
        p=elo_prob(rh,ra,ha); y=int(bool(r.home_win))
        rows.append((r.game_id,season,p))
        d=k*(y-p); ratings[h]=rh+d; ratings[a]=ra-d
    return pd.DataFrame(rows,columns=["game_id","source_season","elo_p"])

def game_event_table(pbp):
    d=pbp.copy()
    d["game_date"]=pd.to_datetime(d.game_date,errors="coerce")
    d["source_season"]=pd.to_numeric(d.source_season,errors="coerce")
    d=d[d.game_type.astype("string").isin(["R","P"])].copy()
    keep=[
        "game_id","source_season","game_date","event_team_abbr","home_away","win",
        "fivev5_sog","fivev5_sog_against","fivev5_xg","fivev5_xg_against",
        "fivev5_sog_share","fivev5_xg_share","goals_non_en"
    ]
    d=d[[c for c in keep if c in d.columns]].copy()
    h=d[d.home_away.eq("home")].copy()
    a=d[d.home_away.eq("away")].copy()
    h=h.rename(columns={
        "event_team_abbr":"home_team",
        "fivev5_sog":"h_sog","fivev5_sog_against":"h_sog_against",
        "fivev5_xg":"h_xg","fivev5_xg_against":"h_xg_against",
        "fivev5_sog_share":"h_sog_share","fivev5_xg_share":"h_xg_share",
        "goals_non_en":"h_goals","win":"home_win"
    })
    a=a.rename(columns={
        "event_team_abbr":"away_team",
        "fivev5_sog":"a_sog","fivev5_sog_against":"a_sog_against",
        "fivev5_xg":"a_xg","fivev5_xg_against":"a_xg_against",
        "fivev5_sog_share":"a_sog_share","fivev5_xg_share":"a_xg_share",
        "goals_non_en":"a_goals"
    })
    cols_h=["game_id","source_season","game_date","home_team","home_win","h_sog","h_xg","h_sog_share","h_xg_share","h_goals"]
    cols_a=["game_id","away_team","a_sog","a_xg","a_sog_share","a_xg_share","a_goals"]
    x=h[cols_h].merge(a[cols_a],on="game_id",how="inner")
    for c in ["h_sog","a_sog","h_xg","a_xg","h_sog_share","a_sog_share","h_xg_share","a_xg_share","h_goals","a_goals"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    return x.sort_values(["game_date","game_id"]).reset_index(drop=True)

def add_flat_and_ewm_features(games):
    # Build long team-game table, then compute lagged rolling/ewm pregame form.
    h=games[["game_id","source_season","game_date","home_team","h_sog_share","h_xg_share","h_goals"]].copy()
    h.columns=["game_id","source_season","game_date","team","sog_share","xg_share","goals"]
    a=games[["game_id","source_season","game_date","away_team","a_sog_share","a_xg_share","a_goals"]].copy()
    a.columns=h.columns
    long=pd.concat([h,a],ignore_index=True).sort_values(["team","game_date","game_id"])
    prev=long.groupby("team")["game_date"].shift(1)
    gap=(long.game_date-prev).dt.total_seconds()/86400
    seg=gap.gt(240).fillna(True).groupby(long.team).cumsum()
    keys=[long.team,seg]
    for c in ["sog_share","xg_share","goals"]:
        v=pd.to_numeric(long[c],errors="coerce")
        long[f"{c}_r20"]=v.groupby(keys,sort=False).transform(lambda s:s.shift(1).rolling(20,min_periods=8).mean())
        for hl in HALFLIVES:
            # EWM alpha from half-life; reset at long gap segments.
            alpha=1-math.exp(math.log(.5)/hl)
            long[f"{c}_ewm{hl}"]=v.groupby(keys,sort=False).transform(
                lambda s:s.shift(1).ewm(alpha=alpha,adjust=False,min_periods=8).mean()
            )
    long["rest_days"]=gap.clip(0,14)
    feat_cols=[c for c in long.columns if c.endswith("_r20") or "_ewm" in c]+["rest_days"]
    home=long.rename(columns={"team":"home_team",**{c:f"h_{c}" for c in feat_cols}})
    away=long.rename(columns={"team":"away_team",**{c:f"a_{c}" for c in feat_cols}})
    out=games.merge(home[["game_id","home_team"]+[f"h_{c}" for c in feat_cols]],on=["game_id","home_team"],how="left")
    out=out.merge(away[["game_id","away_team"]+[f"a_{c}" for c in feat_cols]],on=["game_id","away_team"],how="left")
    for c in feat_cols:
        out[f"d_{c}"]=pd.to_numeric(out[f"h_{c}"],errors="coerce")-pd.to_numeric(out[f"a_{c}"],errors="coerce")
    return out

def opponent_adjusted_features(games,alpha,retain):
    # Online two-way offense/defense ratings. Each game is predicted BEFORE its observation updates ratings.
    off_sog={}; def_sog={}; off_xg={}; def_xg={}
    league_sog=None; league_xg=None
    prev_season=None
    rows=[]
    for r in games.sort_values(["game_date","game_id"]).itertuples(index=False):
        season=int(r.source_season)
        if prev_season is None:
            prev_season=season
        elif season!=prev_season:
            off_sog={t:v*retain for t,v in off_sog.items()}
            def_sog={t:v*retain for t,v in def_sog.items()}
            off_xg={t:v*retain for t,v in off_xg.items()}
            def_xg={t:v*retain for t,v in def_xg.items()}
            prev_season=season

        h=str(r.home_team); a=str(r.away_team)
        if league_sog is None:
            league_sog=np.nanmean([r.h_sog,r.a_sog])
        if league_xg is None:
            league_xg=np.nanmean([r.h_xg,r.a_xg])

        hos=off_sog.get(h,0.0); aos=off_sog.get(a,0.0)
        hds=def_sog.get(h,0.0); ads=def_sog.get(a,0.0)
        hox=off_xg.get(h,0.0); aox=off_xg.get(a,0.0)
        hdx=def_xg.get(h,0.0); adx=def_xg.get(a,0.0)

        pred_h_sog=max(0.1,league_sog+hos+ads)
        pred_a_sog=max(0.1,league_sog+aos+hds)
        pred_h_xg=max(.01,league_xg+hox+adx)
        pred_a_xg=max(.01,league_xg+aox+hdx)

        rows.append((
            r.game_id,
            pred_h_sog-pred_a_sog,
            pred_h_xg-pred_a_xg,
            hos-aos,
            ads-hds,
            hox-aox,
            adx-hdx
        ))

        # Update only after storing pregame features.
        if np.isfinite(r.h_sog) and np.isfinite(r.a_sog):
            res_h=r.h_sog-pred_h_sog
            res_a=r.a_sog-pred_a_sog
            step=alpha/2.0
            off_sog[h]=hos+step*res_h; def_sog[a]=ads+step*res_h
            off_sog[a]=aos+step*res_a; def_sog[h]=hds+step*res_a
            league_sog=(1-alpha*.15)*league_sog+(alpha*.15)*((r.h_sog+r.a_sog)/2)
        if np.isfinite(r.h_xg) and np.isfinite(r.a_xg):
            res_h=r.h_xg-pred_h_xg
            res_a=r.a_xg-pred_a_xg
            step=alpha/2.0
            off_xg[h]=hox+step*res_h; def_xg[a]=adx+step*res_h
            off_xg[a]=aox+step*res_a; def_xg[h]=hdx+step*res_a
            league_xg=(1-alpha*.15)*league_xg+(alpha*.15)*((r.h_xg+r.a_xg)/2)

    tag=f"a{alpha:.2f}_r{retain:.2f}".replace(".","p")
    cols=[
        "game_id",f"oa_sog_diff_{tag}",f"oa_xg_diff_{tag}",
        f"oa_sog_off_diff_{tag}",f"oa_sog_def_adv_{tag}",
        f"oa_xg_off_diff_{tag}",f"oa_xg_def_adv_{tag}"
    ]
    return pd.DataFrame(rows,columns=cols),tag

def main(pbp_path,spine_path,outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    pbp=pd.read_parquet(pbp_path)
    spine=pd.read_parquet(spine_path)
    spine=spine[spine.result_available.fillna(False)&spine.game_type.astype("string").isin(["R","P"])].copy()
    spine["game_date"]=pd.to_datetime(spine.game_date,errors="coerce")
    spine["source_season"]=pd.to_numeric(spine.source_season,errors="coerce")

    g=add_flat_and_ewm_features(game_event_table(pbp))

    # Elo candidates are precomputed with no future leakage.
    elo={}
    for k,ha,ret in ELO_GRID:
        key=f"k{k}_h{ha}_r{ret}"
        elo[key]=elo_all(spine,k,ha,ret)

    # Opponent-adjusted candidates.
    oa={}
    for alpha,ret in OA_GRID:
        z,tag=opponent_adjusted_features(g,alpha,ret)
        oa[tag]=z
        g=g.merge(z,on="game_id",how="left")

    rows=[]
    for target in [2022,2023,2024,2025,2026]:
        val_season=target-1

        # Tune Elo using ONLY prior validation season.
        elo_choice=None
        for key,pred in elo.items():
            vv=g[g.source_season.eq(val_season)][["game_id","home_win"]].merge(
                pred[pred.source_season.eq(val_season)][["game_id","elo_p"]],on="game_id")
            if len(vv)<100: continue
            ll=log_loss(vv.home_win,np.clip(vv.elo_p,1e-6,1-1e-6),labels=[0,1])
            if elo_choice is None or ll<elo_choice[0]:
                elo_choice=(ll,key)
        ekey=elo_choice[1]
        z=g.merge(elo[ekey][["game_id","elo_p"]],on="game_id",how="left")
        z["elo_logit"]=np.log(np.clip(z.elo_p,1e-6,1-1e-6)/np.clip(1-z.elo_p,1e-6,1-1e-6))

        train=z[z.source_season.le(target-2)]
        val=z[z.source_season.eq(val_season)]
        test=z[z.source_season.eq(target)]

        # Existing flat-20 backbone.
        base=["elo_logit","d_sog_share_r20","d_xg_share_r20","d_goals_r20","d_rest_days"]
        C,mm=fit_tuned(train,val,test,base)
        rows.append({"target_season":target,"model":"flat20_backbone","choice":f"{ekey};C={C}",**mm})

        # EWM family: choose half-life by validation performance, then evaluate target.
        ewm_choice=None
        for hl in HALFLIVES:
            feats=["elo_logit",f"d_sog_share_ewm{hl}",f"d_xg_share_ewm{hl}",f"d_goals_ewm{hl}","d_rest_days"]
            for C0 in C_GRID:
                pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                               ("m",LogisticRegression(C=C0,max_iter=1800))])
                pipe.fit(train[feats],train.home_win)
                pv=pipe.predict_proba(val[feats])[:,1]
                ll=log_loss(val.home_win,np.clip(pv,1e-6,1-1e-6),labels=[0,1])
                if ewm_choice is None or ll<ewm_choice[0]:
                    ewm_choice=(ll,hl,C0)
        _,hl,C0=ewm_choice
        feats=["elo_logit",f"d_sog_share_ewm{hl}",f"d_xg_share_ewm{hl}",f"d_goals_ewm{hl}","d_rest_days"]
        fit=pd.concat([train,val],ignore_index=True)
        pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                       ("m",LogisticRegression(C=C0,max_iter=1800))])
        pipe.fit(fit[feats],fit.home_win)
        mm=met(test.home_win,pipe.predict_proba(test[feats])[:,1])
        rows.append({"target_season":target,"model":"ewm_backbone","choice":f"{ekey};hl={hl};C={C0}",**mm})

        # Opponent-adjusted rating family: tune alpha/retention on validation.
        oa_choice=None
        for alpha,ret in OA_GRID:
            tag=f"a{alpha:.2f}_r{ret:.2f}".replace(".","p")
            feats=["elo_logit",f"oa_sog_diff_{tag}",f"oa_xg_diff_{tag}","d_goals_r20","d_rest_days"]
            for C0 in C_GRID:
                pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                               ("m",LogisticRegression(C=C0,max_iter=1800))])
                pipe.fit(train[feats],train.home_win)
                pv=pipe.predict_proba(val[feats])[:,1]
                ll=log_loss(val.home_win,np.clip(pv,1e-6,1-1e-6),labels=[0,1])
                if oa_choice is None or ll<oa_choice[0]:
                    oa_choice=(ll,tag,C0)
        _,tag,C0=oa_choice
        oa_feats=["elo_logit",f"oa_sog_diff_{tag}",f"oa_xg_diff_{tag}","d_goals_r20","d_rest_days"]
        fit=pd.concat([train,val],ignore_index=True)
        pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                       ("m",LogisticRegression(C=C0,max_iter=1800))])
        pipe.fit(fit[oa_feats],fit.home_win)
        mm=met(test.home_win,pipe.predict_proba(test[oa_feats])[:,1])
        rows.append({"target_season":target,"model":"opponent_adjusted","choice":f"{ekey};{tag};C={C0}",**mm})

        # Hybrid: existing flat20 + selected opponent-adjusted signals; tests incremental value.
        hybrid=base+[f"oa_sog_diff_{tag}",f"oa_xg_diff_{tag}"]
        C,mm=fit_tuned(train,val,test,hybrid)
        rows.append({"target_season":target,"model":"flat20_plus_oppadj","choice":f"{ekey};{tag};C={C}",**mm})

        # Richer hybrid separates offensive and defensive opponent-adjusted components.
        rich=base+[
            f"oa_sog_off_diff_{tag}",f"oa_sog_def_adv_{tag}",
            f"oa_xg_off_diff_{tag}",f"oa_xg_def_adv_{tag}"
        ]
        C,mm=fit_tuned(train,val,test,rich)
        rows.append({"target_season":target,"model":"flat20_plus_oppadj_components","choice":f"{ekey};{tag};C={C}",**mm})

    res=pd.DataFrame(rows)
    res.to_csv(out/"rolling_team_strength_004.csv",index=False)
    summary={}
    for model,z in res.groupby("model"):
        summary[model]={
            "folds":int(len(z)),
            "games":int(z.n.sum()),
            "weighted_brier":float(np.average(z.brier,weights=z.n)),
            "weighted_log_loss":float(np.average(z.log_loss,weights=z.n)),
            "weighted_accuracy":float(np.average(z.accuracy,weights=z.n)),
            "brier_by_season":{str(int(r.target_season)):float(r.brier) for r in z.itertuples()},
            "choices":{str(int(r.target_season)):str(r.choice) for r in z.itertuples()}
        }
    base=summary["flat20_backbone"]["weighted_brier"]
    for model in summary:
        summary[model]["delta_brier_vs_flat20"]=summary[model]["weighted_brier"]-base
    with open(out/"summary.json","w") as f: json.dump(summary,f,indent=2)
    print(res.to_string(index=False))
    print("\nSUMMARY")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--pbp",required=True); p.add_argument("--spine",required=True); p.add_argument("--out",required=True)
    a=p.parse_args(); main(a.pbp,a.spine,a.out)
