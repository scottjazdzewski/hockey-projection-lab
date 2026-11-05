"""Retrain leakage-safe rolling-origin xG/process component ablations."""
from pathlib import Path
import sys
import json

ROOT=Path('/workspace/scratch/2369a7a4286e')
sys.path.insert(0,str(ROOT/'research'))
import hockey_goals_team_form_60_009 as model

full=list(model.FEATURE_SETS[model.CORE_FEATURE_SET])

def minus(pred): return [x for x in full if not pred(x)]
sets={
 'full_process_reference':full,
 'minus_all_xg':minus(lambda x: '_xg' in x),
 'minus_5v5_xg':minus(lambda x: 'fivev5_xg' in x),
 'minus_power_play_process':minus(lambda x: ('pp_xg' in x or 'pp_goals' in x or 'penalt' in x)),
 'minus_shot_volume':minus(lambda x: 'sog' in x),
 'minus_recent_goal_form':minus(lambda x: x in {'team_r_reg_gf','opp_r_reg_ga','team_r_reg_ga','opp_r_reg_gf'}),
 'minus_schedule':minus(lambda x: x in {'rest_days','b2b','games_prior_3_days','games_prior_5_days'}),
 'minus_opponent_adjustments':minus(lambda x: x.startswith('oa_')),
}
model.FEATURE_SETS=sets
model.CORE_FEATURE_SET='full_process_reference'
out=ROOT/'run_009_new_snapshot'/'xg_component_ablation'
out.mkdir(parents=True,exist_ok=True)
manifest={
 'run_id':'GOALS-XG-COMPONENT-ABLATION-ROLLING-20261105',
 'scope':'retrain each feature ablation using existing rolling-origin protocol; no production change',
 'target_seasons':model.TARGET_SEASONS,
 'reference_features':full,
 'feature_sets':sets,
 'ablation_policy':'remove one coherent feature family from the fixed full process reference; alpha tuned on S-1 only; fit through S-1; score S',
 'distribution_policy':'score laws selected on S-1 for full process reference; each variant includes its independent-Poisson score; no market/HWIN/lineup/same-game goalie inputs',
 'source_snapshot':'2026-11-04 rebuilt data; frozen source hashes differ',
}
(ROOT/'run_009_new_snapshot'/'xg_component_ablation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
result=model.main(str(ROOT/'pbp_artifact'/'pbp_feature_output'/'game_team_event_features.parquet'),str(ROOT/'spine_artifact'/'spine_v3'/'games.parquet'),str(out))
print(json.dumps({'outdir':str(out),'folds':len(result.get('fold_metrics',[])) if isinstance(result,dict) else None,'files':[p.name for p in out.iterdir()]},default=str))
