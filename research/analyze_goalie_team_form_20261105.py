from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import poisson
ROOT=Path('/workspace/scratch/2369a7a4286e/run_009_new_snapshot/goalie_team_form')
d=pd.read_parquet(ROOT/'oof_predictions.parquet');d=d[d.model_variant.str.endswith('_independent_poisson')].copy();d['target_season']=d.source_season.astype(int);d['final_total']=d.official_final_home_goals+d.official_final_away_goals

def m(g):
 h,a=g.reg_home_goals,g.reg_away_goals;mh,ma=g.rate_mu_home,g.rate_mu_away
 return pd.Series({'games':len(g),'home_goal_bias':(mh-h).mean(),'home_goal_mae':np.abs(mh-h).mean(),'away_goal_bias':(ma-a).mean(),'away_goal_mae':np.abs(ma-a).mean(),'team_goal_mae':(np.abs(mh-h).sum()+np.abs(ma-a).sum())/(2*len(g)),'team_count_logloss':(-poisson.logpmf(h,mh).sum()-poisson.logpmf(a,ma).sum())/(2*len(g)),'final_total_mae':np.abs(g.exp_final_home_goals+g.exp_final_away_goals-g.final_total).mean(),'final_score_logloss':-g.logp_final_score.mean()})
by=d.groupby(['target_season','model_variant']).apply(m,include_groups=False).reset_index();pool=d.groupby('model_variant').apply(m,include_groups=False).reset_index()
by.to_csv(ROOT/'goalie_metrics_by_season.csv',index=False);pool.to_csv(ROOT/'goalie_metrics_pooled.csv',index=False)
base='full_reference_60_independent_poisson';wide={n:g.set_index('game_id') for n,g in d.groupby('model_variant')};b=wide[base]
metrics=['home_goal_mae','away_goal_mae','team_goal_mae','team_count_logloss','final_total_mae','final_score_logloss','home_goal_bias','away_goal_bias'];rows=[];rng=np.random.default_rng(20261105);seasons=sorted(b.target_season.unique());ix_by={s:np.flatnonzero(b.target_season.to_numpy()==s) for s in seasons};B=10000
for name,a in wide.items():
 if name==base:continue
 z=pd.DataFrame(index=b.index);z['target_season']=b.target_season.astype(int)
 vals={'home_goal_mae':np.abs(a.rate_mu_home-a.reg_home_goals)-np.abs(b.rate_mu_home-b.reg_home_goals),'away_goal_mae':np.abs(a.rate_mu_away-a.reg_away_goals)-np.abs(b.rate_mu_away-b.reg_away_goals),'team_goal_mae':(np.abs(a.rate_mu_home-a.reg_home_goals)+np.abs(a.rate_mu_away-a.reg_away_goals)-np.abs(b.rate_mu_home-b.reg_home_goals)-np.abs(b.rate_mu_away-b.reg_away_goals))/2,'team_count_logloss':(-poisson.logpmf(a.reg_home_goals,a.rate_mu_home)-poisson.logpmf(a.reg_away_goals,a.rate_mu_away)+poisson.logpmf(b.reg_home_goals,b.rate_mu_home)+poisson.logpmf(b.reg_away_goals,b.rate_mu_away))/2,'final_total_mae':np.abs(a.exp_final_home_goals+a.exp_final_away_goals-a.final_total)-np.abs(b.exp_final_home_goals+b.exp_final_away_goals-b.final_total),'final_score_logloss':-a.logp_final_score+b.logp_final_score,'home_goal_bias':(a.rate_mu_home-a.reg_home_goals)-(b.rate_mu_home-b.reg_home_goals),'away_goal_bias':(a.rate_mu_away-a.reg_away_goals)-(b.rate_mu_away-b.reg_away_goals)}
 for c,v in vals.items():z[c]=np.asarray(v)
 for c in metrics:
  delta=z[c].to_numpy();boot=np.zeros(B)
  for s,ix in ix_by.items():boot+=delta[rng.choice(ix,size=(B,len(ix)),replace=True)].mean(axis=1)/len(seasons)
  rows.append({'variant':name,'metric':c,'candidate_minus_reference':delta.mean(),'paired_season_stratified_95_low':np.quantile(boot,.025),'paired_season_stratified_95_high':np.quantile(boot,.975),'seasons_ablation_better':int(sum(z.loc[z.target_season.eq(s),c].mean()<0 for s in seasons)),'games':len(z)})
pd.DataFrame(rows).to_csv(ROOT/'goalie_paired_deltas.csv',index=False)
print('POOLED')
print(pool.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
print('\nABLATION MINUS FULL (negative is improvement)')
print(pd.DataFrame(rows).to_string(index=False,float_format=lambda x:f'{x:.5f}'))
