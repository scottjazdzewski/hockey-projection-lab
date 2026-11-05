"""Strictly lagged team goal-residual persistence screen on rolling-origin OOF rates."""
from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PRED=ROOT/'run_009_new_snapshot/goalie_team_form/oof_predictions.parquet'
OUT=ROOT/'research/exports/GOALS_XG_BATTERY_20261105'
OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_parquet(PRED)
df=df[df.model_variant.eq('full_reference_60_independent_poisson')].copy()
rows=[]
for side in ['home','away']:
    part=df[['game_id','source_season',f'{side}_team',f'exp_reg_{side}_goals',f'reg_{side}_goals']].copy()
    part.columns=['game_id','season','team','pred','actual']
    part['side']=side
    rows.append(part)
t=pd.concat(rows,ignore_index=True)
t['resid']=t.actual-t.pred
# Fit each team residual mean on the immediately prior target season only.
team_means=t.groupby(['season','team'],as_index=False).resid.mean().rename(columns={'season':'train_season','resid':'prior_resid'})
scales=[0.0,0.10,0.20,0.35,0.50,0.75,1.0]
eval_seasons=[]
for season in sorted(t.season.unique()):
    if season-1 not in set(t.season.unique()):
        continue
    target=t[t.season.eq(season)].merge(team_means[team_means.train_season.eq(season-1)][['team','prior_resid']],on='team',how='left')
    target['prior_resid']=target.prior_resid.fillna(0.0)
    eval_seasons.append(target.assign(eval_season=season))
e=pd.concat(eval_seasons,ignore_index=True)
# Season-to-season team residual correlation (diagnostic only).
wide=t.groupby(['season','team']).resid.mean().unstack('season')
corr=[]
for a,b in zip(sorted(wide.columns)[:-1],sorted(wide.columns)[1:]):
    q=wide[[a,b]].dropna()
    corr.append({'from_season':int(a),'to_season':int(b),'teams':len(q),'pearson_corr':q[a].corr(q[b]),'spearman_corr':q[a].corr(q[b],method='spearman')})
pd.DataFrame(corr).to_csv(OUT/'team_residual_adjacent_season_correlations.csv',index=False,float_format='%.8f')
# For each correction scale calculate expected-goal MAE and Poisson count NLL.
# This is an exploratory lagged team effect, not a proposed production offset.
results=[]
for scale in scales:
    x=e.copy()
    x['adjusted_pred']=np.clip(x.pred+scale*x.prior_resid,0.1,7.0)
    x['abs_error']=(x.actual-x.adjusted_pred).abs()
    x['poisson_nll']=x.adjusted_pred-x.actual*np.log(x.adjusted_pred)+np.vectorize(lambda v: math.lgamma(v+1))(x.actual)
    for season,g in x.groupby('eval_season'):
        results.append({'scale':scale,'season':int(season),'team_games':len(g),'mae':g.abs_error.mean(),'poisson_nll':g.poisson_nll.mean(),'mean_pred':g.adjusted_pred.mean(),'mean_residual_actual_minus_pred':(g.actual-g.adjusted_pred).mean()})
by=pd.DataFrame(results)
by.to_csv(OUT/'team_residual_lagged_adjustment_by_season.csv',index=False,float_format='%.8f')
# paired bootstrap on game IDs, resampled within target season; each game contributes both sides.
wide_pred=e.pivot(index=['eval_season','game_id'],columns='side',values=['actual','pred','prior_resid']).reset_index()
rows=[]
for scale in scales:
    frame=[]
    for side in ['home','away']:
        g=e[e.side.eq(side)].copy()
        g['adj']=np.clip(g.pred+scale*g.prior_resid,0.1,7.0)
        g['base_abs']=(g.actual-g.pred).abs(); g['adj_abs']=(g.actual-g.adj).abs()
        frame.append(g[['eval_season','game_id','base_abs','adj_abs']])
    z=frame[0].merge(frame[1],on=['eval_season','game_id'],suffixes=('_h','_a'))
    z['base_game_mae']=(z.base_abs_h+z.base_abs_a)/2
    z['adj_game_mae']=(z.adj_abs_h+z.adj_abs_a)/2
    delta=z.adj_game_mae-z.base_game_mae
    rng=np.random.default_rng(8675309+int(scale*1000))
    boots=[]
    for season,g in z.groupby('eval_season'):
        vals=(g.adj_game_mae-g.base_game_mae).to_numpy()
        idx=rng.integers(0,len(vals),size=(2500,len(vals)))
        boots.append(vals[idx].mean(axis=1))
    boot=np.mean(boots,axis=0)
    rows.append({'scale':scale,'games':len(z),'team_mae_delta':delta.mean(),'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975),'seasons_better':int((by[(by.scale.eq(scale))].groupby('season').mae.mean().values < by[(by.scale.eq(0))].groupby('season').mae.mean().values).sum())})
pd.DataFrame(rows).to_csv(OUT/'team_residual_lagged_adjustment_paired.csv',index=False,float_format='%.8f')
print('Adjacent-season team residual correlations:')
print(pd.DataFrame(corr).to_string(index=False))
print('\nPaired MAE:')
print(pd.DataFrame(rows).to_string(index=False))
print('\nBy season:')
print(by.to_string(index=False))
