import fs from 'node:fs';
import path from 'node:path';

const root=process.cwd();
const dataDir=path.join(root,'public','data');

function readJson(name){
  const p=path.join(dataDir,name);
  if(!fs.existsSync(p)) return null;
  return JSON.parse(fs.readFileSync(p,'utf8'));
}
function fail(msg){ console.error('FAIL:',msg); process.exitCode=1; }
function ok(msg){ console.log('PASS:',msg); }
function finiteOrNull(v){ return v===null || (typeof v==='number' && Number.isFinite(v)); }

const schedule=readJson('today-games.json');
const projections=readJson('game-projections.json');
const perf=readJson('model-performance.json');
const teams=readJson('team-ratings.json');
const players=readJson('player-ratings.json');
const fantasy=readJson('fantasy-rankings.json');

if(!schedule || !Array.isArray(schedule.games)) fail('today-games.json missing or malformed');
else ok('schedule readable');

if(projections){
  if(projections.schema_version!=='1.0') fail('game projections schema_version must be 1.0');
  if(!projections.model?.model_id || !projections.model?.version || !projections.model?.feature_version) fail('game projections model metadata incomplete');
  if(!Array.isArray(projections.projections)) fail('game projections array missing');
  else {
    const scheduleIds=new Set((schedule?.games||[]).map(g=>String(g.game_id)));
    const seen=new Set();
    for(const g of projections.projections){
      const id=String(g.game_id);
      if(seen.has(id)) fail('duplicate projection game_id '+id); else seen.add(id);
      if(schedule && !scheduleIds.has(id)) fail('projection game_id not in today schedule: '+id);
      for(const side of ['away','home']){
        const p=g?.[side]?.win_probability;
        if(!finiteOrNull(p)) fail(id+' '+side+' win_probability must be finite or null');
        if(p!==null && (p<0 || p>1)) fail(id+' '+side+' win_probability outside [0,1]');
        for(const field of ['expected_goals','regulation_win_probability']){
          const v=g?.[side]?.[field];
          if(!finiteOrNull(v)) fail(id+' '+side+' '+field+' must be finite or null');
        }
      }
      const a=g?.away?.win_probability, h=g?.home?.win_probability;
      if(g.publish_state==='PUBLISHED'){
        if(projections.model.status!=='PROMOTED') fail(id+' published from non-PROMOTED model');
        if(a===null || h===null) fail(id+' published with missing win probability');
        if(typeof a==='number' && typeof h==='number' && Math.abs((a+h)-1)>0.001) fail(id+' away+home win_probability must equal 1');
      }
    }
    ok('game projections integrity checks complete');
  }
}

if(perf){
  const m=perf.metrics||{};
  if(perf.schema_version!=='1.0') fail('model-performance schema_version must be 1.0');
  if(!Number.isInteger(perf.sample?.games_scored) || perf.sample.games_scored<0) fail('games_scored invalid');
  for(const k of ['accuracy','brier','log_loss','calibration']){
    if(!finiteOrNull(m[k])) fail('model-performance '+k+' must be finite or null');
  }
  ok('model performance readable');
}

if(teams){
  if(teams.schema_version!=='1.1') fail('team-ratings schema_version must be 1.1');
  if(!teams.model?.model_id || !teams.model?.version || !teams.model?.status) fail('team-ratings model metadata incomplete');
  if(!Array.isArray(teams.teams)) fail('team-ratings teams must be array');
  else {
    const canonical=new Set(['ANA','BOS','BUF','CAR','CBJ','CGY','CHI','COL','DAL','DET','EDM','FLA','LAK','MIN','MTL','NJD','NSH','NYI','NYR','OTT','PHI','PIT','SEA','SJS','STL','TBL','TOR','UTA','VAN','VGK','WPG','WSH']);
    const ids=new Set();
    for(const t of teams.teams){
      if(!t.abbr || !t.name) fail('team rating missing identity');
      if(!canonical.has(t.abbr)) fail('unknown current team abbreviation '+t.abbr);
      if(ids.has(t.abbr)) fail('duplicate team '+t.abbr); else ids.add(t.abbr);
      for(const k of ['overall','offense','defense','five_on_five','power_play','penalty_kill','goaltending','form']){
        if(!finiteOrNull(t[k])) fail(t.abbr+' '+k+' must be finite or null');
      }
    }
    if(teams.model.status==='PROMOTED' && ids.size!==32) fail('PROMOTED team-ratings export must contain all 32 teams');
    ok('team ratings readable');
  }
}

if(players){
  if(!Array.isArray(players.players)) fail('player-ratings players must be array');
  else ok('player ratings readable');
}

if(fantasy){
  if(!Array.isArray(fantasy.rankings)) fail('fantasy rankings must be array');
  else ok('fantasy rankings readable');
}

if(process.exitCode) process.exit(process.exitCode);
console.log('All production data checks passed.');
