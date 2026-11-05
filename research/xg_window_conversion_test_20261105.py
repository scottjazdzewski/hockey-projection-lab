"""Nested held-out test of xG-only rolling windows and finishing residuals."""
from pathlib import Path
import sys,json
import pandas as pd
import numpy as np
ROOT=Path('/workspace/scratch/2369a7a4286e')
sys.path.insert(0,str(ROOT/'research'))
import hockey_goals_team_form_60_009 as model
ref_features=list(model.FEATURE_SETS[model.CORE_FEATURE_SET])
base_builder=model.build_side_data
windows=[20,60,100]
metrics=['reg_gf','reg_ga','xg','xg_against','fivev5_xg','fivev5_xg_against','pp_xg','pp_xg_against']

def build_with_extra_xg(pbp,spine,labels):
    side=base_builder(pbp,spine,labels)
    daily=(side.groupby(['team','game_date'],as_index=False,sort=False)[metrics].mean()
           .sort_values(['team','game_date'],kind='stable').reset_index(drop=True))
    for w in windows:
        for metric in metrics:
            col=f'{metric}_w{w}'
            daily[col]=daily.groupby('team',sort=False)[metric].transform(
                lambda s:s.shift(1).rolling(w,min_periods=model.ROLLING_MIN_PERIODS).mean())
        team_cols=[f'{m}_w{w}' for m in metrics]
        own=daily[['team','game_date']+team_cols].rename(columns={c:f'own_{c}' for c in team_cols})
        opp=daily[['team','game_date']+team_cols].rename(columns={'team':'opponent',**{c:f'opp_{c}' for c in team_cols}})
        side=side.merge(own,on=['team','game_date'],how='left',validate='many_to_one')
        side=side.merge(opp,on=['opponent','game_date'],how='left',validate='many_to_one')
        for c in ['fivev5_xg','fivev5_xg_against','pp_xg','pp_xg_against']:
            side[f'xg{w}_{c}']=side[f'own_{c}_w{w}']
        for c in ['fivev5_xg','fivev5_xg_against','pp_xg','pp_xg_against']:
            side[f'xg{w}_opp_{c}']=side[f'opp_{c}_w{w}']
        side[f'xg{w}_team_finishing']=side[f'own_reg_gf_w{w}']-side[f'own_xg_w{w}']
        side[f'xg{w}_opp_goals_allowed_minus_xga']=side[f'opp_reg_ga_w{w}']-side[f'opp_xg_against_w{w}']
    return side

# Replace only rolling xG metrics; leave all non-xG features, opponent-adjusted xG,
# schedule, recent goals, and shot form fixed at their 60-game reference values.
def replace_xg_window(w):
    out=[]
    for f in ref_features:
        if f.startswith('team_r_fivev5_xg'):
            out.append(f'xg{w}_'+f.removeprefix('team_r_'))
        elif f.startswith('opp_r_fivev5_xg'):
            out.append(f'xg{w}_opp_'+f.removeprefix('opp_r_'))
        elif f=='team_r_pp_xg':out.append(f'xg{w}_pp_xg')
        elif f=='opp_r_pp_xg_against':out.append(f'xg{w}_opp_pp_xg_against')
        else:out.append(f)
    return out

sets={'reference_current_60':ref_features}
for w in [20,100]:sets[f'rolling_xg_{w}']=replace_xg_window(w)
for w in windows:sets[f'conversion_residual_{w}']=ref_features+[f'xg{w}_team_finishing',f'xg{w}_opp_goals_allowed_minus_xga']
for w in [20,100]:sets[f'rolling_xg_{w}_plus_conversion_{w}']=replace_xg_window(w)+[f'xg{w}_team_finishing',f'xg{w}_opp_goals_allowed_minus_xga']
model.build_side_data=build_with_extra_xg
model.FEATURE_SETS=sets
model.CORE_FEATURE_SET='reference_current_60'
out=ROOT/'run_009_new_snapshot'/'xg_window_conversion'
out.mkdir(parents=True,exist_ok=True)
manifest={'run_id':'GOALS-XG-WINDOW-CONVERSION-ROLLING-20261105','target_seasons':model.TARGET_SEASONS,'games_per_season':1312,'reference_features':ref_features,'candidate_sets':sets,'design':'All variants retrained in each rolling fold; alpha and score distribution selected on S-1 only; score S. Non-xG rolling inputs, schedule, opponent-adjusted xG ratings, and shot metrics held at current 60-game definition.','xg_window_policy':'Replace only rolling five-on-five and power-play xG features with strictly prior-date 20- or 100-game versions; reference remains current 60-game version.','conversion_policy':'Add strictly prior-date rolling goals-for minus all-situation xG and opponent goals-allowed minus xG-against, using 20/60/100-game windows.','no_inputs':['markets','HWIN probabilities','same-game goalie identity','same-game lineups'],'source_snapshot':'2026-11-04 rebuilt data; frozen EXP-0027/0028 source hashes differ','no_production_change':True}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
model.main(str(ROOT/'pbp_artifact'/'pbp_feature_output'/'game_team_event_features.parquet'),str(ROOT/'spine_artifact'/'spine_v3'/'games.parquet'),str(out))
print('Finished variants:',list(sets))
