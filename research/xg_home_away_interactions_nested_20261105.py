"""Nested test of home/away-specific learned 5v5 xG weights."""
from pathlib import Path
import sys,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
import hockey_goals_team_form_60_009 as model
base_builder=model.build_side_data
base=list(model.FEATURE_SETS[model.CORE_FEATURE_SET])
attack_home='team_fivev5_xg_x_home'
defense_home='opp_fivev5_xg_against_x_home'
def build_interactions(pbp,spine,labels):
    side=base_builder(pbp,spine,labels)
    side[attack_home]=side.team_r_fivev5_xg*side.is_home
    side[defense_home]=side.opp_r_fivev5_xg_against*side.is_home
    return side
sets={
 'full_reference_60':base,
 'home_attack_xg_interaction':base+[attack_home],
 'home_defense_xga_interaction':base+[defense_home],
 'home_attack_defense_xg_interactions':base+[attack_home,defense_home],
}
model.build_side_data=build_interactions
model.FEATURE_SETS=sets;model.CORE_FEATURE_SET='full_reference_60'
out=ROOT/'run_009_new_snapshot'/'xg_home_away_interactions';out.mkdir(parents=True,exist_ok=True)
(out/'manifest.json').write_text(json.dumps({'run_id':'GOALS-XG-HOME-AWAY-INTERACTIONS-20261105','target_seasons':model.TARGET_SEASONS,'features':sets,'design':'Nested rolling-origin retraining. Add interaction terms between home indicator and strictly prior 60-game 5v5 team attack xG / opponent 5v5 xG allowed. Compare attack, defense and both with the same five held-out folds. No market inputs or production change.','snapshot':'2026-11-04 rebuilt data; frozen EXP-0027/0028 hashes differ.'},indent=2)+'\n')
model.main(str(ROOT/'pbp_artifact'/'pbp_feature_output'/'game_team_event_features.parquet'),str(ROOT/'spine_artifact'/'spine_v3'/'games.parquet'),str(out))
print('Completed home/away xG interaction variants')
