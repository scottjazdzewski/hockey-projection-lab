from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import poisson
ROOT=Path('/workspace/scratch/2369a7a4286e/run_009_new_snapshot/form_window_sensitivity')
variant='full_process_reference_validation_selected_distribution'

def load(w):
 d=pd.read_parquet(ROOT/f'window_{w}'/'oof_predictions.parquet')
 d=d[d.model_variant.eq(variant)].copy(); d['target_season']=d.source_season.astype(int)
 d['home_win_actual']=(d.official_final_home_goals>d.official_final_away_goals).astype(int)
 d['total_actual']=d.official_final_home_goals+d.official_final_away_goals
 d['team_count_logloss']=(-poisson.logpmf(d.reg_home_goals,d.rate_mu_home)-poisson.logpmf(d.reg_away_goals,d.rate_mu_away))/2
 d['team_goal_mae']=(np.abs(d.reg_home_goals-d.rate_mu_home)+np.abs(d.reg_away_goals-d.rate_mu_away))/2
 d['final_total_mae']=np.abs(d.exp_final_home_goals+d.exp_final_away_goals-d.total_actual)
 d['final_score_logloss']=-d.logp_final_score
 d['full_home_win_brier']=(d.p_full_home_win-d.home_win_actual)**2
 return d.set_index('game_id')
windows=[20,40,60,100]; data={w:load(w) for w in windows}
cols=['team_count_logloss','team_goal_mae','final_total_mae','final_score_logloss','full_home_win_brier']
seasons=sorted(data[60].target_season.unique())
rows=[]
for w,d in data.items():
 for s,g in d.groupby('target_season'):
  for c in cols:rows.append({'window':w,'target_season':int(s),'metric':c,'value':g[c].mean(),'games':len(g)})
by=pd.DataFrame(rows);by.to_csv(ROOT/'window_metrics_by_season.csv',index=False)
pooled=pd.DataFrame([{'window':w,'metric':c,'value':d[c].mean(),'games':len(d)} for w,d in data.items() for c in cols]);pooled.to_csv(ROOT/'window_metrics_pooled.csv',index=False)
out=[];rng=np.random.default_rng(11052026);ix_by={s:np.flatnonzero(data[60].target_season.to_numpy()==s) for s in seasons};B=10000
for w in [20,40,100]:
 a,b=data[w],data[60]
 for c in cols:
  delta=(a[c]-b[c]).to_numpy(); boot=np.zeros(B)
  for s,ix in ix_by.items():boot+=delta[rng.choice(ix,size=(B,len(ix)),replace=True)].mean(axis=1)/len(seasons)
  out.append({'window_vs_60':w,'metric':c,'window_minus_60':delta.mean(),'paired_stratified_bootstrap_low_95':np.quantile(boot,.025),'paired_stratified_bootstrap_high_95':np.quantile(boot,.975),'seasons_window_better':sum((a.loc[a.target_season.eq(s),c].mean()<b.loc[b.target_season.eq(s),c].mean()) for s in seasons),'target_seasons':len(seasons),'games':len(a)})
pd.DataFrame(out).to_csv(ROOT/'window_paired_deltas.csv',index=False)
folds=[]
for w in windows:
 f=pd.read_csv(ROOT/f'window_{w}'/'fold_metrics.csv')
 f=f[f.model_variant.eq(variant)].copy();f.insert(0,'window',w);folds.append(f)
pd.concat(folds,ignore_index=True).to_csv(ROOT/'window_selection_by_fold.csv',index=False)
print('POOLED:')
print(pooled.pivot(index='metric',columns='window',values='value').to_string(float_format=lambda x:f'{x:.6f}'))
print('\nWINDOW - 60; negative is better:')
print(pd.DataFrame(out).to_string(index=False,float_format=lambda x:f'{x:.6f}'))
print('\nSELECTED FOLD PARAMETERS:')
print(pd.concat(folds,ignore_index=True)[['window','target_season','selected_poisson_alpha','distribution_family','distribution_parameter','validation_reg_score_logloss']].to_string(index=False))
