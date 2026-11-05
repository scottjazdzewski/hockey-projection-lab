"""Nested rolling-origin ablations of individual xG input groups."""
from pathlib import Path
import sys,json
ROOT=Path('/workspace/scratch/2369a7a4286e');sys.path.insert(0,str(ROOT/'research'))
import hockey_goals_team_form_60_009 as model
full=list(model.FEATURE_SETS[model.CORE_FEATURE_SET])
def omit(pred):return [x for x in full if not pred(x)]
sets={
 'full_reference_60':full,
 'without_team_5v5_attack_xg':omit(lambda x:x=='team_r_fivev5_xg'),
 'without_opponent_5v5_xga':omit(lambda x:x=='opp_r_fivev5_xg_against'),
 'without_team_5v5_xga_context':omit(lambda x:x=='team_r_fivev5_xg_against'),
 'without_opponent_5v5_attack_xg':omit(lambda x:x=='opp_r_fivev5_xg'),
 'without_power_play_xg':omit(lambda x:x in {'team_r_pp_xg','opp_r_pp_xg_against'}),
 'without_opponent_adjusted_xg':omit(lambda x:x in {'oa_xg_attack_rel','oa_xg_defense_rel'}),
 'without_all_rolling_xg':omit(lambda x:x in {'team_r_fivev5_xg','opp_r_fivev5_xg_against','team_r_fivev5_xg_against','opp_r_fivev5_xg','team_r_pp_xg','opp_r_pp_xg_against'}),
}
model.FEATURE_SETS=sets;model.CORE_FEATURE_SET='full_reference_60'
out=ROOT/'run_009_new_snapshot'/'xg_component_isolation';out.mkdir(parents=True,exist_ok=True)
(out/'manifest.json').write_text(json.dumps({'run_id':'GOALS-XG-COMPONENT-ISOLATION-ROLLING-20261105','target_seasons':model.TARGET_SEASONS,'games_per_season':1312,'features':sets,'design':'Retrain each one-component ablation under fixed 60-day non-xG rolling features; alpha and score law selected on S-1; score S. Shot, schedule, goal-form, and remaining xG inputs held fixed.','source_snapshot':'2026-11-04 rebuilt data; frozen EXP-0027/0028 hashes differ','no_market_or_production_changes':True},indent=2)+'\n')
model.main(str(ROOT/'pbp_artifact'/'pbp_feature_output'/'game_team_event_features.parquet'),str(ROOT/'spine_artifact'/'spine_v3'/'games.parquet'),str(out))
print('Completed',list(sets))
