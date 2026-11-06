"""Nested rolling-origin test of adaptive exponential decay for xG inputs.

The 60-game model remains the reference. Only the xG feature families are
replaced by strictly prior-date EWMA values; goals, shots, schedule and all
other inputs retain the fixed-60 pipeline. No production weights or markets.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
import hockey_goals_team_form_60_009 as model

SPANS = (20, 40, 60, 100)
XG_METRICS = (
    "fivev5_xg", "fivev5_xg_against", "pp_xg", "pp_xg_against",
)
FEATURE_TO_SOURCE = {
    "team_r_fivev5_xg": ("team", "fivev5_xg"),
    "opp_r_fivev5_xg_against": ("opponent", "fivev5_xg_against"),
    "team_r_fivev5_xg_against": ("team", "fivev5_xg_against"),
    "opp_r_fivev5_xg": ("opponent", "fivev5_xg"),
    "team_r_pp_xg": ("team", "pp_xg"),
    "opp_r_pp_xg_against": ("opponent", "pp_xg_against"),
}

def file_sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def build_ewma_columns(side: pd.DataFrame) -> pd.DataFrame:
    """Attach same-day-collapsed, prior-date EWMA feature columns."""
    out=side.copy()
    daily=(side.groupby(["team","game_date"],as_index=False,sort=False)[list(XG_METRICS)]
           .mean().sort_values(["team","game_date"],kind="stable").reset_index(drop=True))
    for span in SPANS:
        for metric in XG_METRICS:
            name=f"ewma{span}_{metric}"
            daily[name]=daily.groupby("team",sort=False)[metric].transform(
                lambda s: s.shift(1).ewm(span=span,min_periods=8,adjust=False).mean()
            )
        team_cols=["team","game_date"]+[f"ewma{span}_{m}" for m in XG_METRICS]
        team=daily[team_cols].rename(columns={f"ewma{span}_{m}":f"team_{f'ewma{span}_{m}'}" for m in XG_METRICS})
        out=out.merge(team,on=["team","game_date"],how="left",validate="many_to_one")
        opp_cols=daily[["team","game_date"]+[f"ewma{span}_{m}" for m in XG_METRICS]].rename(
            columns={"team":"opponent",**{f"ewma{span}_{m}":f"opp_ewma{span}_{m}" for m in XG_METRICS}}
        )
        out=out.merge(opp_cols,on=["opponent","game_date"],how="left",validate="many_to_one")
    return out

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--pbp",type=Path,required=True)
    parser.add_argument("--spine",type=Path,required=True)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--expected-pbp-sha256")
    parser.add_argument("--expected-spine-sha256")
    args=parser.parse_args()
    if not args.pbp.is_file() or not args.spine.is_file():
        raise FileNotFoundError("Both canonical PBP and spine artifacts must exist")
    pbp_sha,spine_sha=file_sha256(args.pbp),file_sha256(args.spine)
    if args.expected_pbp_sha256 and pbp_sha != args.expected_pbp_sha256:
        raise ValueError(f"PBP source hash mismatch: {pbp_sha}")
    if args.expected_spine_sha256 and spine_sha != args.expected_spine_sha256:
        raise ValueError(f"spine source hash mismatch: {spine_sha}")
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.outdir}")
    args.outdir.mkdir(parents=True,exist_ok=True)
    base_features=list(model.FEATURE_SETS[model.CORE_FEATURE_SET])
    sets={"full_reference_60":base_features}
    for span in SPANS:
        replace={feature:f"team_ewma{span}_{source}" if role=="team" else f"opp_ewma{span}_{source}"
                 for feature,(role,source) in FEATURE_TO_SOURCE.items()}
        sets[f"xg_ewma_span{span}"]=[replace.get(f,f) for f in base_features]
    base_builder=model.build_side_data
    def build_side_with_ewma(pbp: pd.DataFrame,spine: pd.DataFrame,labels: pd.DataFrame)->pd.DataFrame:
        return build_ewma_columns(base_builder(pbp,spine,labels))
    model.build_side_data=build_side_with_ewma
    model.FEATURE_SETS=sets
    model.CORE_FEATURE_SET="full_reference_60"
    input_manifest={
        "run_id":"GOALS-XG-ADAPTIVE-DECAY-NESTED-20261106",
        "design":"Rolling-origin; train through T-2, tune on T-1, evaluate on T. Only xG features use prior-date EWMAs; all other features retain fixed-60 definitions.",
        "target_seasons":model.TARGET_SEASONS,
        "spans_games":list(SPANS),
        "min_prior_team_dates":8,
        "ewm_adjust":False,
        "inputs":{"pbp":{"path":str(args.pbp),"sha256":pbp_sha},"spine":{"path":str(args.spine),"sha256":spine_sha}},
        "no_market_inputs":True,
        "production_change":False,
        "promotion_gate":"Require paired season-stratified improvement in team-goal MAE or count log loss, no material regression in either home/away MAE or final-score log loss, and directionally stable season results."
    }
    (args.outdir/"input_manifest.json").write_text(json.dumps(input_manifest,indent=2)+"\n")
    model.main(str(args.pbp),str(args.spine),str(args.outdir))

if __name__=="__main__":
    main()
