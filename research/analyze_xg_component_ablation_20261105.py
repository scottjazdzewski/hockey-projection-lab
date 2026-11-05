from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import poisson
ROOT=Path('/workspace/scratch/2369a7a4286e/run_009_new_snapshot')
D=ROOT/'xg_component_ablation'
OUT=D
x=pd.read_parquet(D/'oof_predictions.parquet')
x=x[x.model_variant.str.endswith('_independent_poisson')].copy()
x['target_season']=x.source_season.astype(int)
x['home_win_actual']=(x.official_final_home_goals>x.official_final_away_goals).astype(int)
x['final_total']=x.official_final_home_goals+x.official_final_away_goals

def met(g):
 yh,ya=g.reg_home_goals.to_numpy(),g.reg_away_goals.to_numpy(); mh,ma=g.rate_mu_home.to_numpy(),g.rate_mu_away.to_numpy()
 return pd.Series({'games':len(g),'team_count_logloss':(-poisson.logpmf(yh,mh).sum()-poisson.logpmf(ya,ma).sum())/(2*len(g)),'team_goal_mae':(np.abs(yh-mh).sum()+np.abs(ya-ma).sum())/(2*len(g)),'final_total_mae':np.abs(g.exp_final_home_goals+g.exp_final_away_goals-g.final_total).mean(),'final_score_logloss':-g.logp_final_score.mean(),'full_home_win_brier':((g.p_full_home_win-g.home_win_actual)**2).mean()})
by=x.groupby(['target_season','model_variant']).apply(met,include_groups=False).reset_index()
pooled=x.groupby('model_variant').apply(met,include_groups=False).reset_index()
by.to_csv(OUT/'component_metrics_by_season.csv',index=False);pooled.to_csv(OUT/'component_metrics_pooled.csv',index=False)
base='full_process_reference_independent_poisson'; wide={n:g.set_index('game_id') for n,g in x.groupby('model_variant')}; base_df=wide[base]
cols=['team_count_logloss','team_goal_mae','final_total_mae','final_score_logloss','full_home_win_brier']; out=[]; rng=np.random.default_rng(67051); seasons=sorted(x.target_season.unique()); ix_by={s:np.flatnonzero(base_df.target_season.to_numpy()==s) for s in seasons}; B=10000
for name,ca in wide.items():
 if name==base: continue
 z=pd.DataFrame(index=base_df.index); z['target_season']=base_df.target_season.astype(int)
 for c in cols:
  if c=='team_count_logloss':
   z[c]=(-poisson.logpmf(ca.reg_home_goals,ca.rate_mu_home)-poisson.logpmf(ca.reg_away_goals,ca.rate_mu_away)+poisson.logpmf(base_df.reg_home_goals,base_df.rate_mu_home)+poisson.logpmf(base_df.reg_away_goals,base_df.rate_mu_away))/2
  elif c=='team_goal_mae':
   z[c]=(np.abs(ca.reg_home_goals-ca.rate_mu_home)+np.abs(ca.reg_away_goals-ca.rate_mu_away)-np.abs(base_df.reg_home_goals-base_df.rate_mu_home)-np.abs(base_df.reg_away_goals-base_df.rate_mu_away))/2
  elif c=='final_total_mae':
   ytot=base_df.official_final_home_goals+base_df.official_final_away_goals
   z[c]=np.abs(ca.exp_final_home_goals+ca.exp_final_away_goals-ytot)-np.abs(base_df.exp_final_home_goals+base_df.exp_final_away_goals-ytot)
  elif c=='final_score_logloss':z[c]=-ca.logp_final_score+base_df.logp_final_score
  else:
   y=base_df.home_win_actual.astype(int); z[c]=(ca.p_full_home_win-y)**2-(base_df.p_full_home_win-y)**2
 boot=np.zeros((B,len(cols)))
 for s,ix in ix_by.items():
  sample=rng.choice(ix,size=(B,len(ix)),replace=True)
  for j,c in enumerate(cols):boot[:,j]+=z[c].to_numpy()[sample].mean(axis=1)/len(seasons)
 for j,c in enumerate(cols):out.append({'ablation':name,'metric':c,'ablation_minus_full_reference':z[c].mean(),'paired_stratified_bootstrap_low_95':np.quantile(boot[:,j],.025),'paired_stratified_bootstrap_high_95':np.quantile(boot[:,j],.975),'seasons_ablation_better':sum(z.loc[z.target_season.eq(s),c].mean()<0 for s in seasons),'target_seasons':len(seasons),'games':len(z)})
pd.DataFrame(out).to_csv(OUT/'component_paired_deltas.csv',index=False)
print('POOLED')
print(pooled.to_string(index=False,float_format=lambda v:f'{v:.6f}'))
print('\nABLATION MINUS FULL, NEGATIVE MEANS REMOVAL IS BETTER')
print(pd.DataFrame(out).to_string(index=False,float_format=lambda v:f'{v:.6f}'))
print('\nBY SEASON TEAM COUNT LL')
print(by.pivot(index='target_season',columns='model_variant',values='team_count_logloss').to_string(float_format=lambda v:f'{v:.6f}'))
