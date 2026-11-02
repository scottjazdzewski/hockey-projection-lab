import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EVENTS=[
    "fivev5_xg_share","early_fivev5_xg_share","fivev5_corsi_share",
    "fivev5_sog_share","pp_xg","goals_non_en","high_xg_20","rebound_xg"
]

def m(y,p):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return {"n":int(len(y)),"brier":float(brier_score_loss(y,p)),
            "log_loss":float(log_loss(y,p,labels=[0,1])),
            "accuracy":float(accuracy_score(y,p>=.5)),
            "mean_pred":float(p.mean()),"actual_rate":float(y.mean())}

def elo_prob(rh,ra,ha=20.0):
    return 1/(1+10**(-(rh+ha-ra)/400.0))

def elo_history(g,k=10,ha=20.0,retain=.7):
    ratings={}; prev=None; rows=[]
    for r in g.sort_values(["game_date","game_id"]).itertuples(index=False):
        s=int(r.source_season)
        if prev is None: prev=s
        elif s!=prev:
            ratings={t:1500+retain*(x-1500) for t,x in ratings.items()}
            prev=s
        h=str(r.home_team_abbr); a=str(r.away_team_abbr)
        rh=ratings.get(h,1500.); ra=ratings.get(a,1500.)
        p=elo_prob(rh,ra,ha)
        y=int(bool(r.home_win))
        rows.append((r.game_id,p,rh-ra))
        d=k*(y-p); ratings[h]=rh+d; ratings[a]=ra-d
    return pd.DataFrame(rows,columns=["game_id","elo_p_home","elo_rating_diff"])

def main(pbp_path,spine_path,outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    pbp=pd.read_parquet(pbp_path)
    pbp["game_date"]=pd.to_datetime(pbp["game_date"],errors="coerce")
    pbp["source_season"]=pd.to_numeric(pbp["source_season"],errors="coerce")
    pbp=pbp[pbp["game_type"].astype("string").isin(["R","P"])].copy()

    # Pregame event histories, segmented only across extreme long gaps.
    pbp=pbp.sort_values(["event_team_abbr","game_date","game_id"]).copy()
    pbp["prev_date"]=pbp.groupby("event_team_abbr",sort=False)["game_date"].shift(1)
    gap=(pbp["game_date"]-pbp["prev_date"]).dt.total_seconds()/86400
    seg=gap.gt(240).fillna(True).groupby(pbp["event_team_abbr"],sort=False).cumsum().astype("int32")
    keys=[pbp["event_team_abbr"],seg]
    for c in EVENTS:
        if c not in pbp.columns: continue
        vals=pd.to_numeric(pbp[c],errors="coerce")
        pbp[f"{c}_r20"]=vals.groupby(keys,sort=False).transform(lambda s:s.shift(1).rolling(20,min_periods=8).mean())
    pbp["rest_days"]=gap.clip(lower=0,upper=14)

    rc=[f"{c}_r20" for c in EVENTS if f"{c}_r20" in pbp.columns]+["rest_days"]
    h=pbp[pbp.home_away.eq("home")][["game_id","source_season","game_date","win"]+rc].copy()
    a=pbp[pbp.home_away.eq("away")][["game_id"]+rc].copy()
    for c in rc:
        h=h.rename(columns={c:f"h_{c}"})
        a=a.rename(columns={c:f"a_{c}"})
    x=h.merge(a,on="game_id",how="inner").rename(columns={"win":"home_win"})
    diffs=[]
    for c in rc:
        dc=f"d_{c}"
        x[dc]=pd.to_numeric(x[f"h_{c}"],errors="coerce")-pd.to_numeric(x[f"a_{c}"],errors="coerce")
        diffs.append(dc)

    spine=pd.read_parquet(spine_path)
    spine=spine[spine["result_available"].fillna(False) & spine["game_type"].astype("string").isin(["R","P"])].copy()
    spine["game_date"]=pd.to_datetime(spine["game_date"],errors="coerce")
    spine["source_season"]=pd.to_numeric(spine["source_season"],errors="coerce")
    elo=elo_history(spine)
    x=x.merge(elo,on="game_id",how="left")
    x["elo_logit"]=np.log(np.clip(x.elo_p_home,1e-6,1-1e-6)/np.clip(1-x.elo_p_home,1e-6,1-1e-6))

    event_features=[
        "d_fivev5_xg_share_r20","d_early_fivev5_xg_share_r20",
        "d_fivev5_corsi_share_r20","d_fivev5_sog_share_r20",
        "d_pp_xg_r20","d_goals_non_en_r20","d_high_xg_20_r20",
        "d_rebound_xg_r20","d_rest_days"
    ]
    event_features=[c for c in event_features if c in x.columns]
    specs={
        "event_only":event_features,
        "elo_only":["elo_logit"],
        "fusion":["elo_logit"]+event_features,
    }
    results={}
    preds=x[["game_id","source_season","home_win","elo_p_home"]].copy()
    for name,features in specs.items():
        tr=x[x.source_season.le(2024)]
        pipe=Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("model",LogisticRegression(C=.5,max_iter=2000))
        ])
        pipe.fit(tr[features],tr.home_win)
        results[name]={"features":features}
        for split,sel in [
            ("train",x.source_season.le(2024)),
            ("validation_2025",x.source_season.eq(2025)),
            ("test_2026",x.source_season.eq(2026)),
            ("shadow_2027",x.source_season.eq(2027)),
        ]:
            d=x[sel]
            if len(d):
                p=pipe.predict_proba(d[features])[:,1]
                results[name][split]=m(d.home_win,p)
        if name=="fusion":
            preds["fusion_p_home"]=pipe.predict_proba(x[features])[:,1]
            coefs=pd.DataFrame({"feature":features,"coef_standardized":pipe.named_steps["model"].coef_[0]})
            coefs=coefs.sort_values("coef_standardized",key=lambda s:s.abs(),ascending=False)
            coefs.to_csv(out/"fusion_coefficients.csv",index=False)

    with open(out/"fusion_metrics.json","w") as f: json.dump(results,f,indent=2)
    preds.to_parquet(out/"fusion_predictions.parquet",index=False)
    print(json.dumps(results,indent=2))
    print("\nFUSION COEFFICIENTS")
    print(pd.read_csv(out/"fusion_coefficients.csv").to_string(index=False))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--pbp",required=True); p.add_argument("--spine",required=True); p.add_argument("--out",required=True)
    a=p.parse_args(); main(a.pbp,a.spine,a.out)
