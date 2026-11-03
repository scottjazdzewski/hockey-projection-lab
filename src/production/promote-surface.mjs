import fs from 'node:fs';
import path from 'node:path';

const root=process.cwd();
const dataDir=path.join(root,'public','data');
const surface=process.argv[2];
const apply=process.argv.includes('--apply');

const map={
  game_projections:'game-projections.json',
  model_performance:'model-performance.json',
  team_ratings:'team-ratings.json',
  player_ratings:'player-ratings.json',
  fantasy_rankings:'fantasy-rankings.json'
};

if(!surface || !map[surface]){
  console.error('Usage: node src/production/promote-surface.mjs <surface> [--apply]');
  console.error('Surfaces:',Object.keys(map).join(', '));
  process.exit(2);
}

const manifestPath=path.join(dataDir,'production-manifest.json');
const exportPath=path.join(dataDir,map[surface]);
const manifest=JSON.parse(fs.readFileSync(manifestPath,'utf8'));
const data=JSON.parse(fs.readFileSync(exportPath,'utf8'));

function fail(msg){ console.error('FAIL:',msg); process.exit(1); }

let modelId=null, modelVersion=null;

if(surface==='model_performance'){
  if(!(data?.sample?.games_scored>0)) fail('model performance has no scored games');
  modelId=data?.model?.model_id||null;
  modelVersion=data?.model?.version||null;
  if(!modelId || !modelVersion) fail('model performance metadata incomplete');
}else{
  if(data?.model?.status!=='PROMOTED') fail(surface+' export model is not PROMOTED');
  modelId=data?.model?.model_id||null;
  modelVersion=data?.model?.version||null;
  if(!modelId || !modelVersion) fail(surface+' model metadata incomplete');

  if(surface==='game_projections' && !(Array.isArray(data.projections)&&data.projections.length>0)) fail('game projection export is empty');
  if(surface==='team_ratings' && !(Array.isArray(data.teams)&&data.teams.length===32)) fail('team ratings must contain 32 teams');
  if(surface==='player_ratings' && !(Array.isArray(data.players)&&data.players.length>0)) fail('player ratings export is empty');
  if(surface==='fantasy_rankings' && !(Array.isArray(data.rankings)&&data.rankings.length>0)) fail('fantasy rankings export is empty');
}

const target=manifest?.surfaces?.[surface];
if(!target) fail('surface missing from production manifest');

const next=structuredClone(manifest);
next.surfaces[surface].state='READY';
next.surfaces[surface].model_id=modelId;
next.surfaces[surface].model_version=modelVersion;
next.updated_at=new Date().toISOString();

console.log('Promotion candidate:',surface,modelId,modelVersion);
console.log('Manifest state: '+target.state+' -> READY');

if(!apply){
  console.log('DRY RUN ONLY. Re-run with --apply, then run:');
  console.log('node src/production/validate-public-data.mjs');
  process.exit(0);
}

fs.writeFileSync(manifestPath,JSON.stringify(next,null,2)+'\n');
console.log('Updated '+manifestPath);
console.log('NEXT REQUIRED STEP: node src/production/validate-public-data.mjs');
