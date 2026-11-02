import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

def prob(home_rating, away_rating, home_adv):
    x=(home_rating + home_adv - away_rating)/400.0
    return 1.0/(1.0+10.0**(-x))

def score(y,p):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return {
        "n":int(len(y)),
        "brier":float(brier_score_loss(y,p)),
        "log_loss":float(log_loss(y,p,labels=[0,1])),
        "accuracy":float(accuracy_score(y,p>=.5)),
        "mean_pred":float(np.mean(p)),
        "actual_rate":float(np.mean(y)),
    }

def simulate(games,k,home_adv,retain):
    ratings={}
    prev_season=None
    rows=[]
    for row in games.itertuples(index=False):
        season=int(row.source_season)
        if prev_season is None:
            prev_season=season
        elif season!=prev_season:
            for t,r in list(ratings.items()):
                ratings[t]=1500.0 + retain*(r-1500.0)
            prev_season=season
        h=str(row.home_team_abbr); a=str(row.away_team_abbr)
        rh=ratings.get(h,1500.0); ra=ratings.get(a,1500.0)
        p=prob(rh,ra,home_adv)
        y=int(bool(row.home_win))
        rows.append((row.game_id,season,row.game_date,h,a,y,p,rh,ra))
        delta=k*(y-p)
        ratings[h]=rh+delta
        ratings[a]=ra-delta
    return pd.DataFrame(rows,columns=[
        "game_id","source_season","game_date","home_team","away_team","home_win","p_home","home_rating_pre","away_rating_pre"
    ])

def main(inp,outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    g=pd.read_parquet(inp)
    g=g[g["result_available"].fillna(False)].copy()
    g=g[g["game_type"].astype("string").isin(["R","P"])].copy()
    g["source_season"]=pd.to_numeric(g["source_season"],errors="coerce")
    g["game_date"]=pd.to_datetime(g["game_date"],errors="coerce")
    g=g.sort_values(["game_date","game_id"]).reset_index(drop=True)

    train=g[g.source_season.le(2024)]
    val=g[g.source_season.eq(2025)]
    test=g[g.source_season.eq(2026)]
    shadow=g[g.source_season.eq(2027)]

    grid=[]
    for k in [5,10,15,20,25,30,40,50]:
        for ha in [0,20,40,60,80,100]:
            for retain in [.4,.55,.7,.85,1.0]:
                sim=simulate(g[g.source_season.le(2024)],k,ha,retain)
                m=score(sim.home_win,sim.p_home)
                grid.append({"k":k,"home_adv":ha,"retain":retain,**m})
    gd=pd.DataFrame(grid).sort_values(["log_loss","brier"]).reset_index(drop=True)
    gd.to_csv(out/"elo_grid_train.csv",index=False)
    best=gd.iloc[0].to_dict()

    sim=simulate(g,int(best["k"]),float(best["home_adv"]),float(best["retain"]))
    sim.to_parquet(out/"elo_predictions.parquet",index=False)
    metrics={"best_params":{"k":int(best["k"]),"home_adv":float(best["home_adv"]),"retain":float(best["retain"])}}
    for name,season_filter in [
        ("train",lambda x:x<=2024),
        ("validation_2025",lambda x:x==2025),
        ("test_2026",lambda x:x==2026),
        ("shadow_2027",lambda x:x==2027),
    ]:
        d=sim[season_filter(sim.source_season)]
        if len(d): metrics[name]=score(d.home_win,d.p_home)

    # Constant home-rate baseline estimated from train only.
    train_rate=float(train.home_win.astype(float).mean())
    metrics["train_home_rate"]=train_rate
    for name,d in [("validation_2025",val),("test_2026",test),("shadow_2027",shadow)]:
        if len(d):
            metrics[f"constant_{name}"]=score(d.home_win,np.repeat(train_rate,len(d)))

    with open(out/"elo_metrics.json","w") as f: json.dump(metrics,f,indent=2)
    print(json.dumps(metrics,indent=2))
    print("\nTOP_GRID")
    print(gd.head(20).to_string(index=False))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    main(a.input,a.out)
