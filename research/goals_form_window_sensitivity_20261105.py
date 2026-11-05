"""Nested rolling-origin form-window sensitivity for team goal/xG features."""
from pathlib import Path
import sys, json
ROOT=Path('/workspace/scratch/2369a7a4286e')
sys.path.insert(0,str(ROOT/'research'))
import hockey_goals_team_form_60_009 as model
features=list(model.FEATURE_SETS[model.CORE_FEATURE_SET])
model.FEATURE_SETS={'full_process_reference':features}
model.CORE_FEATURE_SET='full_process_reference'
pbp=ROOT/'pbp_artifact'/'pbp_feature_output'/'game_team_event_features.parquet'
spine=ROOT/'spine_artifact'/'spine_v3'/'games.parquet'
root=ROOT/'run_009_new_snapshot'/'form_window_sensitivity'
root.mkdir(parents=True,exist_ok=True)
manifest={'run_id':'GOALS-FORM-WINDOW-SENSITIVITY-ROLLING-20261105','target_seasons':model.TARGET_SEASONS,'windows':[20,40,60,100],'reference_features':features,'design':'for each window, build strictly prior-date rolling features; train through S-2, select alpha/distribution on S-1, refit through S-1, score S','control':'same feature family, target periods, alpha grid, score-family grid, no market/HWIN/lineup/same-game goalie inputs','source_snapshot':'2026-11-04 rebuilt data; frozen source hashes differ','note':'This changes the rolling window for all rolling team metrics together; it is not a pure xG-only window isolation.'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for w in manifest['windows']:
    model.ROLLING_WINDOW=w
    out=root/f'window_{w}'
    out.mkdir(parents=True,exist_ok=True)
    (root/'active_window.txt').write_text(str(w))
    model.main(str(pbp),str(spine),str(out))
    (root/'completed_windows.txt').write_text('\n'.join(str(x) for x in manifest['windows'] if (root/f'window_{x}'/'oof_predictions.parquet').exists())+'\n')
print('Completed windows:',(root/'completed_windows.txt').read_text())
