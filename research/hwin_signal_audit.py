import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CANDIDATES = [
    "fivev5_xg_share",
    "early_fivev5_xg_share",
    "tied_fivev5_xg_share",
    "close_fivev5_xg_share",
    "fivev5_corsi_share",
    "fivev5_sog_share",
    "xg_share",
    "goals_non_en",
    "fivev5_goals",
    "pp_xg",
    "pp_goals",
    "sh_xg",
    "high_xg_20",
    "rebound_xg",
    "mean_xg_per_unblocked",
    "mean_shot_distance",
    "hits",
    "takeaways",
    "giveaways",
    "penalties",
    "pim",
    "faceoff_wins",
]


def metrics(y, p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    y=np.asarray(y,dtype=int)
    return {
        "n":int(len(y)),
        "brier":float(brier_score_loss(y,p)),
        "log_loss":float(log_loss(y,p,labels=[0,1])),
        "accuracy":float(accuracy_score(y,p>=.5)),
        "mean_pred":float(np.mean(p)),
        "actual_rate":float(np.mean(y)),
    }


def main(inp, outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(inp)
    df["game_date"]=pd.to_datetime(df["game_date"],errors="coerce")
    df["source_season"]=pd.to_numeric(df["source_season"],errors="coerce")
    df["team_final_score"]=pd.to_numeric(df["team_final_score"],errors="coerce")
    df["opp_final_score"]=pd.to_numeric(df["opp_final_score"],errors="coerce")
    df=df[
        df["game_type"].astype("string").isin(["R","P"])
        & df["team_final_score"].notna()
        & df["opp_final_score"].notna()
    ].copy()

    # Hard data gate.
    keydup=int(df.duplicated(["game_id","home_away"]).sum())
    sides=df.groupby("game_id")["home_away"].nunique()
    if keydup or int((sides!=2).sum()) or len(df)!=2*df["game_id"].nunique():
        raise RuntimeError("Input is not a clean two-side game ledger.")

    available=[c for c in CANDIDATES if c in df.columns]
    coverage=df.groupby(["source_season","game_type"]).agg(
        games=("game_id","nunique"), team_rows=("game_id","size")
    ).reset_index()
    coverage.to_csv(out/"coverage_by_season.csv",index=False)

    # ---------- DESCRIPTIVE: same-game traits, never used as pregame inputs directly ----------
    desc=[]
    for c in available:
        x=pd.to_numeric(df[c],errors="coerce")
        t=df[["game_id","win"]].copy()
        t["metric"]=x
        pivot=t.pivot(index="game_id",columns="win",values="metric").dropna()
        if 0 not in pivot.columns or 1 not in pivot.columns or len(pivot)<100:
            continue
        dif=(pivot[1]-pivot[0])
        # Edge winner rate: among games where the metric differs, did the higher-metric side win?
        edge=(dif!=0)
        higher_metric_winner=float((dif[edge]>0).mean()) if edge.any() else np.nan
        desc.append({
            "feature":c,
            "games":int(len(pivot)),
            "winner_mean":float(pivot[1].mean()),
            "loser_mean":float(pivot[0].mean()),
            "winner_minus_loser":float(dif.mean()),
            "higher_metric_side_won_rate":higher_metric_winner,
        })
    pd.DataFrame(desc).sort_values("higher_metric_side_won_rate",ascending=False).to_csv(
        out/"descriptive_win_drivers.csv",index=False
    )

    # ---------- PREDICTIVE: lag everything one game ----------
    # Abbreviation changes intentionally reset a team history rather than falsely bridge a franchise move.
    # Long gaps (>240 days) create a new form segment. This prevents the known missing 2014-15 PBP
    # season (and other extreme interruptions) from being treated as continuous recent form.
    df=df.sort_values(["event_team_abbr","game_date","game_id"]).copy()
    df["prev_game_date"]=df.groupby("event_team_abbr",sort=False)["game_date"].shift(1)
    raw_gap=(df["game_date"]-df["prev_game_date"]).dt.total_seconds()/86400
    new_segment=raw_gap.gt(240).fillna(True)
    df["form_segment"]=new_segment.groupby(df["event_team_abbr"],sort=False).cumsum().astype("int32")
    segment_keys=[df["event_team_abbr"],df["form_segment"]]

    for c in available:
        vals=pd.to_numeric(df[c],errors="coerce")
        df[f"{c}_r10"]=(
            vals.groupby(segment_keys,sort=False)
            .transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
        )
        df[f"{c}_r20"]=(
            vals.groupby(segment_keys,sort=False)
            .transform(lambda s:s.shift(1).rolling(20,min_periods=8).mean())
        )

    df["rest_days"]=raw_gap.clip(lower=0,upper=14)

    rollcols=[f"{c}_r10" for c in available]+[f"{c}_r20" for c in available]+["rest_days"]
    home=df[df.home_away.eq("home")][["game_id","source_season","game_type","win"]+rollcols].copy()
    away=df[df.home_away.eq("away")][["game_id"]+rollcols].copy()
    home=home.rename(columns={c:f"h_{c}" for c in rollcols})
    away=away.rename(columns={c:f"a_{c}" for c in rollcols})
    games=home.merge(away,on="game_id",how="inner")
    games=games.rename(columns={"win":"home_win"})

    diffcols=[]
    for c in rollcols:
        dc=f"d_{c}"
        games[dc]=pd.to_numeric(games[f"h_{c}"],errors="coerce")-pd.to_numeric(games[f"a_{c}"],errors="coerce")
        diffcols.append(dc)

    # Train/validation/test are chronological and untouched by later seasons.
    train=games[games.source_season.le(2024)].copy()
    valid=games[games.source_season.eq(2025)].copy()
    test=games[games.source_season.eq(2026)].copy()
    shadow=games[games.source_season.eq(2027)].copy()

    train_home_rate=float(train.home_win.mean())
    baseline={}
    for name,d in [("train",train),("validation_2025",valid),("test_2026",test),("shadow_2027",shadow)]:
        if len(d):
            baseline[name]=metrics(d.home_win,np.repeat(train_home_rate,len(d)))

    # Univariate audit: one lagged signal at a time against the same chronological test.
    univ=[]
    for c in diffcols:
        tr=train[["home_win",c]].dropna()
        va=valid[["home_win",c]].dropna()
        te=test[["home_win",c]].dropna()
        if len(tr)<1000 or len(te)<100:
            continue
        pipe=Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("model",LogisticRegression(C=1.0,max_iter=1000)),
        ])
        pipe.fit(tr[[c]],tr.home_win)
        coef=float(pipe.named_steps["model"].coef_[0][0])
        row={"feature":c,"train_n":len(tr),"coef_standardized":coef}
        if len(va):
            m=metrics(va.home_win,pipe.predict_proba(va[[c]])[:,1])
            row.update({f"val_{k}":v for k,v in m.items()})
        if len(te):
            m=metrics(te.home_win,pipe.predict_proba(te[[c]])[:,1])
            row.update({f"test_{k}":v for k,v in m.items()})
        univ.append(row)
    uni=pd.DataFrame(univ)
    if len(uni):
        uni=uni.sort_values(["test_brier","test_log_loss"],ascending=True)
    uni.to_csv(out/"predictive_univariate_audit.csv",index=False)

    # Conservative first multivariate benchmark, not a promoted model.
    preferred=[
        "d_fivev5_xg_share_r20","d_early_fivev5_xg_share_r20",
        "d_fivev5_corsi_share_r20","d_fivev5_sog_share_r20",
        "d_pp_xg_r20","d_goals_non_en_r20","d_high_xg_20_r20",
        "d_rebound_xg_r20","d_rest_days",
    ]
    use=[c for c in preferred if c in games.columns]
    model=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("model",LogisticRegression(C=.5,max_iter=2000)),
    ])
    model.fit(train[use],train.home_win)
    benchmark={"features":use,"baseline_constant_home_rate":train_home_rate,"baseline":baseline}
    for name,d in [("train",train),("validation_2025",valid),("test_2026",test),("shadow_2027",shadow)]:
        if len(d):
            benchmark[name]=metrics(d.home_win,model.predict_proba(d[use])[:,1])
    coefs=pd.DataFrame({
        "feature":use,
        "coef_standardized":model.named_steps["model"].coef_[0],
    }).sort_values("coef_standardized",key=lambda s:s.abs(),ascending=False)
    coefs.to_csv(out/"benchmark_coefficients.csv",index=False)
    with open(out/"benchmark_metrics.json","w") as f:
        json.dump(benchmark,f,indent=2)

    quality={
        "team_rows":int(len(df)),
        "games":int(df.game_id.nunique()),
        "duplicate_game_side_keys":keydup,
        "game_types":df.game_type.astype("string").value_counts().to_dict(),
        "seasons":sorted(int(x) for x in df.source_season.dropna().unique()),
        "missing_full_pbp_schedule_season":[2015],
        "analysis_note":"Descriptive files use same-game observations. Predictive files use lagged pregame rolling values only.",
    }
    with open(out/"quality.json","w") as f:
        json.dump(quality,f,indent=2)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    main(a.input,a.out)
