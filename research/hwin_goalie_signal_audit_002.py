import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

C_GRID=[0.05,0.1,0.25,0.5,1.0,2.0]
TEAM_METRICS=["fivev5_sog_share","fivev5_xg_share","goals_non_en"]

def met(y,p):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return {"n":int(len(y)),"brier":float(brier_score_loss(y,p)),
            "log_loss":float(log_loss(y,p,labels=[0,1])),
            "accuracy":float(accuracy_score(y,p>=.5)),
            "mean_pred":float(p.mean()),"actual_rate":float(y.mean())}

def elo_prob(rh,ra,ha=20):
    return 1/(1+10**(-(rh+ha-ra)/400))

def elo_history(g,k=10,ha=20,retain=.7):
    ratings={}; prev=None; rows=[]
    for r in g.sort_values(["game_date","game_id"]).itertuples(index=False):
        s=int(r.source_season)
        if prev is None: prev=s
        elif s!=prev:
            ratings={t:1500+retain*(v-1500) for t,v in ratings.items()}; prev=s
        h=str(r.home_team_abbr); a=str(r.away_team_abbr)
        rh=ratings.get(h,1500.); ra=ratings.get(a,1500.)
        p=elo_prob(rh,ra,ha); y=int(bool(r.home_win))
        rows.append((r.game_id,p))
        d=k*(y-p); ratings[h]=rh+d; ratings[a]=ra-d
    return pd.DataFrame(rows,columns=["game_id","elo_p"])

def build_team_features(pbp):
    d=pbp.copy()
    d["game_date"]=pd.to_datetime(d.game_date,errors="coerce")
    d["source_season"]=pd.to_numeric(d.source_season,errors="coerce")
    d=d[d.game_type.astype("string").isin(["R","P"])].copy()
    d=d.sort_values(["event_team_abbr","game_date","game_id"])
    prev=d.groupby("event_team_abbr")["game_date"].shift(1)
    gap=(d.game_date-prev).dt.total_seconds()/86400
    seg=gap.gt(240).fillna(True).groupby(d.event_team_abbr).cumsum()
    keys=[d.event_team_abbr,seg]
    for c in TEAM_METRICS:
        v=pd.to_numeric(d[c],errors="coerce")
        d[f"{c}_r20"]=v.groupby(keys,sort=False).transform(lambda s:s.shift(1).rolling(20,min_periods=8).mean())
    d["rest_days"]=gap.clip(0,14)
    cols=[f"{c}_r20" for c in TEAM_METRICS]+["rest_days"]
    h=d[d.home_away.eq("home")][["game_id","source_season","game_date","win"]+cols].copy()
    a=d[d.home_away.eq("away")][["game_id"]+cols].copy()
    h=h.rename(columns={"win":"home_win",**{c:f"h_{c}" for c in cols}})
    a=a.rename(columns={c:f"a_{c}" for c in cols})
    x=h.merge(a,on="game_id",how="inner")
    for c in cols:
        x[f"d_{c}"]=pd.to_numeric(x[f"h_{c}"],errors="coerce")-pd.to_numeric(x[f"a_{c}"],errors="coerce")
    return x

def build_goalie_priors(goalies):
    g=goalies.copy()
    g=g[g.starter_bool.fillna(False)].copy()
    g["game_date"]=pd.to_datetime(g.game_date,errors="coerce")
    g["goalie_id_key"]=g.goalie_id_key.astype("string")
    for c in ["shots_against_pbp","saves_pbp","sog_gsax","goals_against_pbp","xg_against","sog_xg_against"]:
        g[c]=pd.to_numeric(g[c],errors="coerce")
    g=g.sort_values(["goalie_id_key","game_date","game_id"]).reset_index(drop=True)

    out=[]
    for gid,z in g.groupby("goalie_id_key",sort=False):
        z=z.sort_values(["game_date","game_id"]).copy()
        shots=z.shots_against_pbp.fillna(0)
        saves=z.saves_pbp.fillna(0)
        sg=z.sog_gsax.fillna(0)
        dates=z.game_date
        prior_starts=np.arange(len(z))
        prev_date=dates.shift(1)
        rest=(dates-prev_date).dt.total_seconds()/86400

        for w in [5,10,20]:
            rs=shots.shift(1).rolling(w,min_periods=max(3,w//2)).sum()
            rv=saves.shift(1).rolling(w,min_periods=max(3,w//2)).sum()
            rg=sg.shift(1).rolling(w,min_periods=max(3,w//2)).sum()
            z[f"sv_r{w}"]=np.where(rs>0,rv/rs,np.nan)
            z[f"sog_gsax30_r{w}"]=np.where(rs>0,rg/rs*30,np.nan)

        # Expanding prior with shrinkage by shot volume toward .905 sv% and 0 GSAx.
        prior_shots=shots.shift(1).cumsum()
        prior_saves=saves.shift(1).cumsum()
        prior_gsax=sg.shift(1).cumsum()
        z["sv_shrunk_prior"]=(prior_saves + 200*.905)/(prior_shots+200)
        z["sog_gsax30_shrunk_prior"]=prior_gsax/(prior_shots+300)*30
        z["goalie_rest_days"]=rest.clip(0,21)
        z["prior_starts"]=prior_starts

        # Workload in prior 7 calendar days.
        shots7=[]; starts7=[]
        for i,dt in enumerate(dates):
            lo=dt-pd.Timedelta(days=7)
            prior=z.iloc[:i]
            m=(prior.game_date>=lo)&(prior.game_date<dt)
            shots7.append(float(prior.loc[m,"shots_against_pbp"].fillna(0).sum()))
            starts7.append(int(m.sum()))
        z["shots_prior_7d"]=shots7
        z["starts_prior_7d"]=starts7
        out.append(z)
    return pd.concat(out,ignore_index=True) if out else g

def fit_model(train,val,test,features):
    best=None
    for C in C_GRID:
        pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                       ("m",LogisticRegression(C=C,max_iter=1800))])
        pipe.fit(train[features],train.home_win)
        pv=pipe.predict_proba(val[features])[:,1]
        ll=log_loss(val.home_win,np.clip(pv,1e-6,1-1e-6),labels=[0,1])
        if best is None or ll<best[0]: best=(ll,C)
    C=best[1]
    fit=pd.concat([train,val],ignore_index=True)
    pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                   ("m",LogisticRegression(C=C,max_iter=1800))])
    pipe.fit(fit[features],fit.home_win)
    p=pipe.predict_proba(test[features])[:,1]
    co=pd.DataFrame({"feature":features,"coef":pipe.named_steps["m"].coef_[0]})
    return C,met(test.home_win,p),co

def main(pbp_path,spine_path,goalie_path,outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    pbp=pd.read_parquet(pbp_path)
    spine=pd.read_parquet(spine_path)
    goalies=pd.read_parquet(goalie_path)
    spine=spine[spine.result_available.fillna(False)&spine.game_type.astype("string").isin(["R","P"])].copy()
    spine["game_date"]=pd.to_datetime(spine.game_date,errors="coerce")
    spine["source_season"]=pd.to_numeric(spine.source_season,errors="coerce")

    x=build_team_features(pbp)
    elo=elo_history(spine,k=10,ha=20,retain=.7)
    x=x.merge(elo,on="game_id",how="left")
    x["elo_logit"]=np.log(np.clip(x.elo_p,1e-6,1-1e-6)/np.clip(1-x.elo_p,1e-6,1-1e-6))

    gp=build_goalie_priors(goalies)
    goalie_cols=[
        "sv_r5","sv_r10","sv_r20","sog_gsax30_r5","sog_gsax30_r10","sog_gsax30_r20",
        "sv_shrunk_prior","sog_gsax30_shrunk_prior","goalie_rest_days","prior_starts",
        "shots_prior_7d","starts_prior_7d"
    ]
    base_id=["game_id","home_away","goalie_id_key","player_name"]+goalie_cols
    gp=gp[base_id].copy()
    h=gp[gp.home_away.astype("string").str.lower().eq("home")].drop_duplicates("game_id")
    a=gp[gp.home_away.astype("string").str.lower().eq("away")].drop_duplicates("game_id")
    h=h.rename(columns={c:f"h_g_{c}" for c in goalie_cols})
    a=a.rename(columns={c:f"a_g_{c}" for c in goalie_cols})
    h=h.rename(columns={"goalie_id_key":"home_goalie_id","player_name":"home_goalie_name"})
    a=a.rename(columns={"goalie_id_key":"away_goalie_id","player_name":"away_goalie_name"})
    keep_h=["game_id","home_goalie_id","home_goalie_name"]+[f"h_g_{c}" for c in goalie_cols]
    keep_a=["game_id","away_goalie_id","away_goalie_name"]+[f"a_g_{c}" for c in goalie_cols]
    x=x.merge(h[keep_h],on="game_id",how="left").merge(a[keep_a],on="game_id",how="left")
    diff_goalie=[]
    for c in goalie_cols:
        dc=f"d_g_{c}"
        x[dc]=pd.to_numeric(x[f"h_g_{c}"],errors="coerce")-pd.to_numeric(x[f"a_g_{c}"],errors="coerce")
        diff_goalie.append(dc)

    team=[
        "elo_logit","d_fivev5_sog_share_r20","d_fivev5_xg_share_r20",
        "d_goals_non_en_r20","d_rest_days"
    ]
    # We test on development test 2026, tuning C on 2025; 2027 remains prospective shadow.
    train=x[x.source_season.le(2024)].copy()
    val=x[x.source_season.eq(2025)].copy()
    test=x[x.source_season.eq(2026)].copy()
    shadow=x[x.source_season.eq(2027)].copy()
    specs={"backbone":team,"goalie_only":diff_goalie,"backbone_plus_goalie":team+diff_goalie}
    results={}
    for name,features in specs.items():
        C,mm,co=fit_model(train,val,test,features)
        results[name]={"C":C,"test_2026":mm}
        # Fit same selected model on train+validation and score shadow without tuning on shadow.
        fit=pd.concat([train,val],ignore_index=True)
        pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                       ("m",LogisticRegression(C=C,max_iter=1800))])
        pipe.fit(fit[features],fit.home_win)
        if len(shadow):
            results[name]["shadow_2027"]=met(shadow.home_win,pipe.predict_proba(shadow[features])[:,1])
        co.to_csv(out/f"{name}_coefficients.csv",index=False)

    # Incremental goalie audit: add exactly one goalie feature to the backbone.
    univariate=[]
    backbone_test_brier=results["backbone"]["test_2026"]["brier"]
    for gf in diff_goalie:
        C,mm,co=fit_model(train,val,test,team+[gf])
        row={
            "goalie_feature":gf,
            "C":C,
            "test_brier":mm["brier"],
            "test_log_loss":mm["log_loss"],
            "test_accuracy":mm["accuracy"],
            "delta_brier_vs_backbone":mm["brier"]-backbone_test_brier
        }
        # Prospective shadow is reported but never used for feature choice.
        fit=pd.concat([train,val],ignore_index=True)
        pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                       ("m",LogisticRegression(C=C,max_iter=1800))])
        pipe.fit(fit[team+[gf]],fit.home_win)
        if len(shadow):
            sm=met(shadow.home_win,pipe.predict_proba(shadow[team+[gf]])[:,1])
            row["shadow_2027_brier"]=sm["brier"]
            row["shadow_2027_accuracy"]=sm["accuracy"]
        univariate.append(row)
    uni=pd.DataFrame(univariate).sort_values("test_brier")
    uni.to_csv(out/"incremental_goalie_univariate.csv",index=False)
    results["incremental_goalie_univariate_top"]=uni.head(12).to_dict(orient="records")

    both_experienced=x[(pd.to_numeric(x["h_g_prior_starts"],errors="coerce")>=5)&(pd.to_numeric(x["a_g_prior_starts"],errors="coerce")>=5)]
    results["coverage"]={
        "all_games":int(len(x)),
        "games_with_both_starters":int(x.home_goalie_id.notna().mul(x.away_goalie_id.notna()).sum()),
        "games_both_goalies_5plus_prior_starts":int(len(both_experienced)),
        "test_2026_games":int(len(test)),
        "test_2026_both_5plus":int(((pd.to_numeric(test["h_g_prior_starts"],errors="coerce")>=5)&(pd.to_numeric(test["a_g_prior_starts"],errors="coerce")>=5)).sum())
    }
    with open(out/"metrics.json","w") as f: json.dump(results,f,indent=2)
    x[["game_id","source_season","home_win","home_goalie_name","away_goalie_name","h_g_prior_starts","a_g_prior_starts"]].to_parquet(out/"goalie_join_audit.parquet",index=False)
    print(json.dumps(results,indent=2))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--pbp",required=True); p.add_argument("--spine",required=True); p.add_argument("--goalie",required=True); p.add_argument("--out",required=True)
    a=p.parse_args(); main(a.pbp,a.spine,a.goalie,a.out)
