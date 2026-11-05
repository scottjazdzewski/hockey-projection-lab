"""Paired held-out comparison of existing leakage-safe goal-feature variants."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson

BASE = Path('/workspace/scratch/2369a7a4286e/run_009_new_snapshot')
OUT = BASE / 'xg_battery'
OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_parquet(BASE / 'oof_predictions.parquet')
d = d[d.model_variant.str.endswith('_independent_poisson')].copy()
d['target_season'] = d.source_season.astype(int)
d['home_win_actual'] = (d.official_final_home_goals > d.official_final_away_goals).astype(int)
d['final_total_actual'] = d.official_final_home_goals + d.official_final_away_goals

def score(g):
    y_h, y_a = g.reg_home_goals.to_numpy(), g.reg_away_goals.to_numpy()
    mu_h, mu_a = g.rate_mu_home.to_numpy(), g.rate_mu_away.to_numpy()
    return pd.Series({
        'games': len(g),
        'team_count_logloss': (-poisson.logpmf(y_h, mu_h).sum() - poisson.logpmf(y_a, mu_a).sum())/(2*len(g)),
        'team_goal_mae': (np.abs(y_h-mu_h).sum()+np.abs(y_a-mu_a).sum())/(2*len(g)),
        'home_goal_mae': np.abs(y_h-mu_h).mean(),
        'away_goal_mae': np.abs(y_a-mu_a).mean(),
        'final_total_mae': np.abs(g.exp_final_home_goals+g.exp_final_away_goals-g.final_total_actual).mean(),
        'reg_score_logloss': -g.logp_reg_score.mean(),
        'final_score_logloss': -g.logp_final_score.mean(),
        'full_home_win_brier': ((g.p_full_home_win-g.home_win_actual)**2).mean(),
    })

metrics = d.groupby(['target_season','model_variant'], sort=True).apply(score, include_groups=False).reset_index()
pooled = d.groupby('model_variant', sort=True).apply(score, include_groups=False).reset_index()
metrics.to_csv(OUT/'feature_set_metrics_by_season.csv', index=False)
pooled.to_csv(OUT/'feature_set_metrics_pooled.csv', index=False)
# Matched per-game differences (candidate minus comparator; negative is better).
pairs = [
 ('goal_form_plus_xg_independent_poisson','team_goal_form_independent_poisson','recent_xg_increment_over_goal_form'),
 ('fivev5_xg_pace_independent_poisson','team_goal_form_independent_poisson','fivev5_xg_pace_over_goal_form'),
 ('hockey_process_independent_poisson','team_goal_form_independent_poisson','full_process_bundle_over_goal_form'),
 ('hockey_process_plus_league_scoring_independent_poisson','hockey_process_independent_poisson','league_scoring_increment_over_process'),
]
metric_cols=['team_count_logloss','team_goal_mae','final_total_mae','reg_score_logloss','final_score_logloss','full_home_win_brier']
wide={v:g.set_index('game_id') for v,g in d.groupby('model_variant')}
rows=[]
for cand,base,label in pairs:
    a,b=wide[cand],wide[base]
    z=pd.DataFrame(index=a.index)
    z['target_season']=a.target_season.astype(int)
    y_h,y_a=a.reg_home_goals.to_numpy(),a.reg_away_goals.to_numpy()
    y_hb,y_ab=b.reg_home_goals.to_numpy(),b.reg_away_goals.to_numpy()
    z['team_count_logloss']=(-poisson.logpmf(y_h,a.rate_mu_home.to_numpy())-poisson.logpmf(y_a,a.rate_mu_away.to_numpy()))/2-(-poisson.logpmf(y_hb,b.rate_mu_home.to_numpy())-poisson.logpmf(y_ab,b.rate_mu_away.to_numpy()))/2
    z['team_goal_mae']=(np.abs(y_h-a.rate_mu_home.to_numpy())+np.abs(y_a-a.rate_mu_away.to_numpy()))/2-(np.abs(y_hb-b.rate_mu_home.to_numpy())+np.abs(y_ab-b.rate_mu_away.to_numpy()))/2
    z['final_total_mae']=np.abs(a.exp_final_home_goals+a.exp_final_away_goals-a.official_final_home_goals-a.official_final_away_goals)-np.abs(b.exp_final_home_goals+b.exp_final_away_goals-b.official_final_home_goals-b.official_final_away_goals)
    z['reg_score_logloss']=-a.logp_reg_score+b.logp_reg_score
    z['final_score_logloss']=-a.logp_final_score+b.logp_final_score
    z['full_home_win_brier']=(a.p_full_home_win-a.home_win_actual.astype(int))**2-(b.p_full_home_win-b.home_win_actual.astype(int))**2
    rng=np.random.default_rng(6032026)
    seasons=sorted(z.target_season.unique())
    season_ix={s:np.flatnonzero(z.target_season.to_numpy()==s) for s in seasons}
    nboot=10000
    # Stratified paired bootstrap preserves each target season's sample size.
    boot=np.zeros((nboot,len(metric_cols)))
    for s,ix in season_ix.items():
        picks=rng.choice(ix,size=(nboot,len(ix)),replace=True)
        for j,col in enumerate(metric_cols): boot[:,j]+=z[col].to_numpy()[picks].mean(axis=1)/len(seasons)
    for j,col in enumerate(metric_cols):
        rows.append({'comparison':label,'candidate':cand,'baseline':base,'metric':col,'candidate_minus_baseline':z[col].mean(),'season_stratified_bootstrap_low_95':np.quantile(boot[:,j],.025),'season_stratified_bootstrap_high_95':np.quantile(boot[:,j],.975),'seasons_candidate_better':int(sum(z.loc[z.target_season.eq(s),col].mean()<0 for s in seasons)),'target_seasons':len(seasons),'games':len(z)})
pd.DataFrame(rows).to_csv(OUT/'paired_feature_deltas.csv',index=False)
# Baselines for overall win probability skill.
y=d[d.model_variant.eq('hockey_process_plus_league_scoring_independent_poisson')].copy()
p=y.p_full_home_win.to_numpy(); actual=y.home_win_actual.to_numpy(); observed=float(actual.mean())
summary=pd.DataFrame([
 {'measure':'full_game_home_win_brier','model':float(np.mean((p-actual)**2)),'baseline':float(np.mean((observed-actual)**2)),'interpretation':'OOF brier; constant held-out prevalence baseline is descriptive'},
 {'measure':'full_game_home_win_brier','model':float(np.mean((p-actual)**2)),'baseline':float(np.mean((0.5-actual)**2)),'interpretation':'OOF brier versus 50/50 baseline'},
 {'measure':'home_win_rate','model':float(p.mean()),'baseline':observed,'interpretation':'mean predicted probability versus observed frequency'},
])
summary.to_csv(OUT/'win_probability_reference.csv',index=False)
print('Pooled variants:')
print(pooled.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
print('\nPaired deltas (negative is improvement):')
print(pd.DataFrame(rows).to_string(index=False,float_format=lambda x:f'{x:.6f}'))
print('\nWin probability reference:')
print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
