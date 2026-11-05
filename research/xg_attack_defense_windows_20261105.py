"""Nested rolling test of separate 5v5 attack-xG and xGA windows."""
from pathlib import Path
import sys,json
import pandas as pd
ROOT=Path('/workspace/scratch/2369a7a4286e');sys.path.insert(0,str(ROOT/'research'))
import hockey_goals_team_form_60_009 as model
full=list(model.FEATURE_SETS[model.CORE_FEATURE_SET]);base_builder=model.build_side_data
metrics=['fivev5_xg','fivev5_xg_against'];windows=[20,100]
def build_extra(pbp,spine,labels):
 side=base_builder(pbp,spine,labels)
 daily=(side.groupby(['team','game_date'],as_index=False,sort=False)[metrics].mean().sort_values(['team','game_date'],kind='stable').reset_index(drop=True))
 for w in windows:
  for m in metrics:
   c=f'{m}_w{w}';daily[c]=daily.groupby('team',sort=False)[m].transform(lambda s:s.shift(1).rolling(w,min_periods=model.ROLLING_MIN_PERIODS).mean())
  own=daily[['team','game_date']+[f'{m}_w{w}' for m in metrics]].rename(columns={f'{m}_w{w}':f'own_{m}_w{w}' for m in metrics})
  side=side.merge(own,on=['team','game_date'],how='left',validate='many_to_one')
  for m in metrics: side[f'xg{w}_{m}']=side[f'own_{m}_w{w}']
 return side
model.build_side_data=build_extra

def replace(feature,window):
 return [f'xg{window}_{feature.removeprefix("team_r_")}' if x==feature else x for x in full]
sets={'reference_all_xg_60':full}
for w in windows:
 sets[f'team_attack_xg_{w}']=replace('team_r_fivev5_xg',w)
 sets[f'team_defense_xga_{w}']=replace('team_r_fivev5_xg_against',w)
 sets[f'attack_and_defense_xg_{w}']=replace('team_r_fivev5_xg',w)
 sets[f'attack_and_defense_xg_{w}']=[f'xg{w}_fivev5_xg_against' if x=='team_r_fivev5_xg_against' else x for x in sets[f'attack_and_defense_xg_{w}']]
model.FEATURE_SETS=sets;model.CORE_FEATURE_SET='reference_all_xg_60'
out=ROOT/'run_009_new_snapshot'/'xg_attack_defense_windows';out.mkdir(parents=True,exist_ok=True)
manifest={'run_id':'GOALS-XG-ATTACK-DEFENSE-WINDOWS-ROLLING-20261105','target_seasons':model.TARGET_SEASONS,'games_per_season':1312,'windows':[20,60,100],'features':sets,'design':'Replace 5v5 own attack xG and own xG-against inputs separately with 20/100-day rolling versions; all other features and rolling windows remain current 60. Also test both changed together. Strictly prior-date features; tune alpha and score family on S-1; score S.','no_leakage':'All same-date games excluded from historical rolling inputs.','source_snapshot':'2026-11-04 rebuilt data; frozen source hashes differ.','no_production_change':True}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
model.main(str(ROOT/'pbp_artifact'/'pbp_feature_output'/'game_team_event_features.parquet'),str(ROOT/'spine_artifact'/'spine_v3'/'games.parquet'),str(out))
print('Completed',list(sets))
