"""Nested rolling-origin test of 5v5 xG per shot quality features."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
import hockey_goals_team_form_60_009 as model
base_builder=model.build_side_data
base=list(model.FEATURE_SETS[model.CORE_FEATURE_SET])
quality_attack='team_fivev5_xg_per_sog'
quality_allowed='opp_fivev5_xg_allowed_per_sog_against'

def build_with_quality(pbp,spine,labels):
    side=base_builder(pbp,spine,labels)
    # Rolling ratios are derived only from the base model's strictly prior-date 60-game aggregates.
    side[quality_attack]=side.team_r_fivev5_xg/side.team_r_fivev5_sog.replace(0,np.nan)
    side[quality_allowed]=side.opp_r_fivev5_xg_against/side.opp_r_fivev5_sog_against.replace(0,np.nan)
    return side

sets={
 'full_reference_60':base,
 'plus_attack_shot_quality':base+[quality_attack],
 'plus_defense_shot_quality_allowed':base+[quality_allowed],
 'plus_both_shot_quality':base+[quality_attack,quality_allowed],
}
model.build_side_data=build_with_quality
model.FEATURE_SETS=sets
model.CORE_FEATURE_SET='full_reference_60'
out=ROOT/'run_009_new_snapshot'/'xg_shot_quality';out.mkdir(parents=True,exist_ok=True)
manifest={
 'run_id':'GOALS-XG-SHOT-QUALITY-NESTED-20261105',
 'target_seasons':model.TARGET_SEASONS,
 'features':sets,
 'design':'Nested rolling-origin: add strictly prior-date 60-game 5v5 xG-per-shot features for team attack and opponent defensive xG allowed per shot against. Compare attack, defense, and both additions with the same retraining and prior-season tuning. No same-game features, market inputs, or production changes.',
 'source_snapshot':'2026-11-04 rebuilt data; frozen EXP-0027/0028 source hashes differ.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
model.main(str(ROOT/'pbp_artifact'/'pbp_feature_output'/'game_team_event_features.parquet'),str(ROOT/'spine_artifact'/'spine_v3'/'games.parquet'),str(out))
print('Completed nested xG per shot quality variants')
