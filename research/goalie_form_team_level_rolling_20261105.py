"""Test leakage-safe opponent goalie form from prior team goalie games."""
from pathlib import Path
import sys,json
import pandas as pd,numpy as np
ROOT=Path('/workspace/scratch/2369a7a4286e');sys.path.insert(0,str(ROOT/'research'))
import hockey_goals_team_form_60_009 as model
base_builder=model.build_side_data
features=list(model.FEATURE_SETS[model.CORE_FEATURE_SET])
windows=[20,60];priors=[300,800]

def with_goalie_form(pbp,spine,labels):
 side=base_builder(pbp,spine,labels)
 goalie=pd.read_parquet(ROOT/'spine_artifact'/'spine_v3'/'goalie_box.parquet')
 goalie=goalie.loc[goalie.source_season.between(model.MIN_SOURCE_SEASON,2026)].copy()
 goalie['game_date']=pd.to_datetime(goalie.game_date,errors='coerce').dt.normalize()
 goalie['saves']=pd.to_numeric(goalie.saves,errors='coerce')
 goalie['shots_against']=pd.to_numeric(goalie.shots_against,errors='coerce')
 goalie=goalie.dropna(subset=['team_abbrev','game_date','saves','shots_against'])
 # Sum every goalie appearance by team/game, including relief appearances; no identity or same-game result is joined to a prediction.
 game=(goalie.groupby(['team_abbrev','game_id','game_date'],as_index=False,sort=False)[['saves','shots_against']].sum()
       .rename(columns={'team_abbrev':'team'}).sort_values(['team','game_date','game_id'],kind='stable'))
 daily=(game.groupby(['team','game_date'],as_index=False,sort=False)[['saves','shots_against']].sum()
        .sort_values(['team','game_date'],kind='stable').reset_index(drop=True))
 # League save baseline also uses only completed dates before the forecast date.
 league=(daily.groupby('game_date',as_index=False)[['saves','shots_against']].sum().sort_values('game_date'))
 league['league_prior_saves']=league.saves.shift(1).rolling(365,min_periods=30).sum()
 league['league_prior_shots']=league.shots_against.shift(1).rolling(365,min_periods=30).sum()
 league['league_prior_save_pct']=league.league_prior_saves/league.league_prior_shots
 daily=daily.merge(league[['game_date','league_prior_save_pct']],on='game_date',how='left',validate='many_to_one')
 for w in windows:
  for c in ['saves','shots_against']:
   daily[f'prior_{c}_{w}']=daily.groupby('team',sort=False)[c].transform(lambda s:s.shift(1).rolling(w,min_periods=8).sum())
  for k in priors:
   f=f'goalie_form_w{w}_k{k}'
   raw=(daily[f'prior_saves_{w}']+k*daily.league_prior_save_pct)/(daily[f'prior_shots_against_{w}']+k)
   daily[f'{f}_team']=raw-daily.league_prior_save_pct
   side=side.merge(daily[['team','game_date',f'{f}_team']],on=['team','game_date'],how='left',validate='many_to_one')
   opp=daily[['team','game_date',f'{f}_team']].rename(columns={'team':'opponent',f'{f}_team':f'opp_{f}'})
   side=side.merge(opp,on=['opponent','game_date'],how='left',validate='many_to_one')
 return side

sets={'full_reference_60':features}
for w in windows:
 for k in priors:sets[f'goalie_form_w{w}_k{k}']=features+[f'opp_goalie_form_w{w}_k{k}']
model.build_side_data=with_goalie_form;model.FEATURE_SETS=sets;model.CORE_FEATURE_SET='full_reference_60'
out=ROOT/'run_009_new_snapshot'/'goalie_team_form';out.mkdir(parents=True,exist_ok=True)
manifest={'run_id':'GOALS-OPPONENT-GOALIE-FORM-TEAM-LEVEL-20261105','target_seasons':model.TARGET_SEASONS,'games_per_season':1312,'features':sets,'design':'Nested rolling-origin; add opponent team-level goalie save form computed from prior completed team goalie games only. Save rates are shrunk to a strictly prior 365-day league save baseline with 300 or 800 shot-equivalent prior. Compare 20/60 prior team games. Same-game starter identity/results are never inputs.','limits':'Team-level proxy, not projected or confirmed starting goalie identity. It may mix goalie and team-defense effects.','source_snapshot':'2026-11-04 rebuilt data; frozen EXP-0027/0028 source hashes differ.','no_market_or_production_changes':True}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
model.main(str(ROOT/'pbp_artifact'/'pbp_feature_output'/'game_team_event_features.parquet'),str(ROOT/'spine_artifact'/'spine_v3'/'games.parquet'),str(out))
print('Completed',list(sets))
