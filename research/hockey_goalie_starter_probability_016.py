from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

FEATURES=['share5','share10','share20','recent_start','log_days_since','unknown','recent_x_b2b','share5_x_b2b']

def build_choices(games_path, goalie_path):
    games=pd.read_parquet(games_path)
    games=games.loc[games.game_type.eq('R')].copy()
    games['game_id']=games.game_id.astype(str)
    games['game_date']=pd.to_datetime(games.game_date,errors='coerce').dt.normalize()
    box=pd.read_parquet(goalie_path)
    starters=box.loc[box.starter_bool.fillna(False).astype(bool),['game_id','team_abbrev','goalie_id_key']].copy()
    starters['game_id']=starters.game_id.astype(str)
    starters['goalie_id_key']=starters.goalie_id_key.astype(str)
    if starters.duplicated(['game_id','team_abbrev']).any(): raise ValueError('duplicate starter per team/game')
    startmap={(x.game_id,x.team_abbrev):x.goalie_id_key for x in starters.itertuples(index=False)}
    sides=[]
    for r in games.itertuples(index=False):
        for team in [r.home_team_abbr,r.away_team_abbr]:
            gid=str(r.game_id); key=(gid,team)
            if key in startmap:
                sides.append((gid,int(r.season),pd.Timestamp(r.game_date),team,startmap[key]))
    S=pd.DataFrame(sides,columns=['game_id','season','game_date','team','actual']).sort_values(['team','game_date','game_id'])
    rows=[]; groups=[]
    for team,td in S.groupby('team',sort=False):
        history=[]
        for date,day in td.groupby('game_date',sort=True):
            hh=history[-20:]
            if hh:
                b2b=int((date-hh[-1][0]).days==1)
                n5=min(len(hh),5); n10=min(len(hh),10); n20=min(len(hh),20)
                c5=Counter(x[1] for x in hh[-5:]); c10=Counter(x[1] for x in hh[-10:]); c20=Counter(x[1] for x in hh[-20:])
                candidates=list(dict.fromkeys(x[1] for x in reversed(hh)))
                latest={g:max(i for i,x in enumerate(hh) if x[1]==g) for g in candidates}
                for r in day.itertuples(index=False):
                    gid_actual=r.actual
                    target=gid_actual if gid_actual in candidates else '__UNKNOWN__'
                    choice_id=f'{r.game_id}:{team}'
                    for cand in candidates+['__UNKNOWN__']:
                        if cand=='__UNKNOWN__':
                            feat=dict(share5=0.,share10=0.,share20=0.,recent_start=0.,log_days_since=0.,unknown=1.,recent_x_b2b=0.,share5_x_b2b=0.)
                        else:
                            last=max(d for d,g in hh if g==cand)
                            dgap=max(0,min(60,(date-last).days))
                            sh5=c5[cand]/n5; sh10=c10[cand]/n10; sh20=c20[cand]/n20
                            recent=float(cand==hh[-1][1])
                            feat=dict(share5=sh5,share10=sh10,share20=sh20,recent_start=recent,log_days_since=np.log1p(dgap),unknown=0.,recent_x_b2b=recent*b2b,share5_x_b2b=sh5*b2b)
                        feat.update(choice_id=choice_id,candidate=cand,label=int(cand==target),season=r.season,game_date=date,team=team,game_id=r.game_id,b2b=b2b,actual_known=int(gid_actual in candidates),n_candidates=len(candidates))
                        rows.append(feat)
            # append after processing all same-date games to exclude same-day outcomes
            history.extend((date,r.actual) for r in day.itertuples(index=False))
    return pd.DataFrame(rows)

def fit_conditional_logit(train, alpha=1.0):
    group_arrays=[]; target_arrays=[]; group_sizes=[]
    for _,g in train.groupby('choice_id',sort=False):
        y=g.label.to_numpy(int)
        if y.sum()!=1: continue
        group_arrays.append(g[FEATURES].to_numpy(float))
        target_arrays.append(int(np.flatnonzero(y)[0]))
        group_sizes.append(len(g))
    if not group_arrays: raise ValueError('no training groups')
    Xall=np.vstack(group_arrays)
    mu=Xall.mean(axis=0); sd=Xall.std(axis=0); sd[sd<1e-8]=1.0
    X=(Xall-mu)/sd
    sizes=np.asarray(group_sizes,dtype=int)
    starts=np.r_[0,np.cumsum(sizes)[:-1]]
    target_rows=starts+np.asarray(target_arrays,dtype=int)
    def fg(beta):
        z=X@beta
        maxima=np.maximum.reduceat(z,starts)
        expz=np.exp(z-np.repeat(maxima,sizes))
        sums=np.add.reduceat(expz,starts)
        loss=float(np.sum(maxima+np.log(sums)-z[target_rows]) + .5*alpha*np.dot(beta,beta))
        p=expz/np.repeat(sums,sizes)
        grad=X.T@p-X[target_rows].sum(axis=0)+alpha*beta
        return loss,grad
    res=minimize(fg,np.zeros(len(FEATURES)),method='L-BFGS-B',jac=True,options={'maxiter':250,'ftol':1e-9})
    if not res.success and res.status not in (1,): print('optimizer warning',res.message)
    return res.x,mu,sd

def eval_season(choices, target_season):
    train=choices.loc[choices.season<target_season]
    test=choices.loc[choices.season.eq(target_season)]
    if train.empty or test.empty: return None
    beta,mu,sd=fit_conditional_logit(train)
    train_unknown_rate=float(train.loc[(train.candidate.eq('__UNKNOWN__')) & train.label.eq(1),'choice_id'].nunique()/train.choice_id.nunique())
    train_unknown_rate=float(np.clip(train_unknown_rate,0.005,0.20))
    out=[]; candidate_rows=[]
    for cid,g in test.groupby('choice_id',sort=False):
        X=g[FEATURES].to_numpy(float); z=((X-mu)/sd)@beta; p=np.exp(z-logsumexp(z))
        truth=int(np.flatnonzero(g.label.to_numpy(int))[0]); cand=g.candidate.tolist()
        pred=int(np.argmax(p)); correct=pred==truth
        b2b=int(g.b2b.iloc[0]); known=int(g.actual_known.iloc[0]); n=int(g.n_candidates.iloc[0])
        # Uniform over candidates plus explicit unseen/returning goalie class.
        uni=np.full(len(g),1/len(g))
        un_i=cand.index('__UNKNOWN__')
        w=g.share5.to_numpy(float).copy(); w[un_i]=0.0; w[~np.isfinite(w)]=0.0; w=w+np.where(np.arange(len(w))==un_i,0.0,0.05)
        fp=np.zeros(len(g)); fp[un_i]=train_unknown_rate; fp[np.arange(len(g))!=un_i]=(1-train_unknown_rate)*w[np.arange(len(g))!=un_i]/max(w.sum(),1e-12)
        out.append(dict(choice_id=cid,season=target_season,team=g.team.iloc[0],game_id=g.game_id.iloc[0],b2b=b2b,known=known,n_candidates=n,model_p=p[truth],model_logloss=-np.log(max(p[truth],1e-15)),model_brier=float(np.sum((p-np.eye(1,len(p),truth)[0])**2)),model_correct=correct,confidence=float(p[pred]),model_unknown_p=float(p[un_i]),frequency_logloss=-np.log(max(fp[truth],1e-15)),frequency_brier=float(np.sum((fp-np.eye(1,len(fp),truth)[0])**2)),frequency_correct=int(np.argmax(fp)==truth),uniform_logloss=float(np.log(len(g))),uniform_brier=float(np.sum((uni-np.eye(1,len(g),truth)[0])**2)),uniform_correct=int(pred==truth),actual_candidate=cand[truth],predicted_candidate=cand[pred]))
        for j,c in enumerate(cand): candidate_rows.append(dict(choice_id=cid,season=target_season,team=g.team.iloc[0],game_id=g.game_id.iloc[0],b2b=b2b,known=known,candidate=c,label=int(j==truth),logit=float(z[j]),model_probability=float(p[j]),frequency_probability=float(fp[j])))
    return pd.DataFrame(out),dict(train_groups=train.choice_id.nunique(),test_groups=test.choice_id.nunique(),optimizer_beta=dict(zip(FEATURES,beta.tolist())),converged=True),pd.DataFrame(candidate_rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--games',required=True); ap.add_argument('--goalies',required=True); ap.add_argument('--outdir',required=True)
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    choices=build_choices(a.games,a.goalies)
    seasons=sorted(int(x) for x in choices.season.dropna().unique() if int(x)>=2022)
    preds=[]; candidate_preds=[]; folds={}
    for s in seasons:
        result=eval_season(choices,s)
        if result is None: continue
        p,m,cp=result; preds.append(p); candidate_preds.append(cp); folds[str(s)]=m
    P=pd.concat(preds,ignore_index=True)
    P.to_parquet(out/'oof_starter_probabilities.parquet',index=False)
    pd.concat(candidate_preds,ignore_index=True).to_parquet(out/'candidate_probabilities.parquet',index=False)
    metrics={}
    for s,g in P.groupby('season'):
        metrics[str(int(s))]={
            'n_team_sides':int(len(g)),
            'candidate_coverage':float(g.known.mean()),
            'top1_accuracy':float(g.model_correct.mean()),
            'multiclass_logloss':float(g.model_logloss.mean()),
            'multiclass_brier':float(g.model_brier.mean()),
            'frequency_logloss':float(g.frequency_logloss.mean()),
            'frequency_brier':float(g.frequency_brier.mean()),
            'frequency_top1_accuracy':float(g.frequency_correct.mean()),
            'mean_predicted_unknown_probability':float(g.model_unknown_p.mean()),
            'uniform_logloss':float(g.uniform_logloss.mean()),
            'uniform_brier':float(g.uniform_brier.mean()),
            'b2b':{'n':int(g.b2b.sum()),'top1_accuracy':float(g.loc[g.b2b.eq(1),'model_correct'].mean()) if g.b2b.sum() else None,'logloss':float(g.loc[g.b2b.eq(1),'model_logloss'].mean()) if g.b2b.sum() else None},
            'non_b2b':{'n':int(g.b2b.eq(0).sum()),'top1_accuracy':float(g.loc[g.b2b.eq(0),'model_correct'].mean()) if g.b2b.eq(0).sum() else None,'logloss':float(g.loc[g.b2b.eq(0),'model_logloss'].mean()) if g.b2b.eq(0).sum() else None},
            'known_candidate_top1_accuracy':float(g.loc[g.known.eq(1),'model_correct'].mean()) if g.known.sum() else None,
            'unknown_actual_rate':float(1-g.known.mean())}
    P['confidence_bin']=pd.cut(P.confidence,[0,.3,.4,.5,.6,.7,.8,.9,1.000001],right=False)
    calibration=P.groupby('confidence_bin',observed=False).agg(n=('model_correct','size'),mean_confidence=('confidence','mean'),accuracy=('model_correct','mean')).reset_index().astype({'confidence_bin':'string'}).to_dict(orient='records')
    summary={'experiment_id':'HOCKEY-LAB-GOALIE-STARTER-PROBABILITY-016','status':'RESEARCH','target_seasons':seasons,'n_choices':int(len(P)),'design':'expanding chronological conditional-logit choice model; training seasons strictly earlier than target season; features from team starts on strictly earlier dates; unseen/returning goalie explicit alternative; L2 alpha=1.0 fixed; no tuning','features':FEATURES,'folds':folds,'metrics_by_season':metrics,'calibration_by_top_choice_confidence':calibration}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
