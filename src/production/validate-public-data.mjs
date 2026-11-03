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
const manifest=readJson('production-manifest.json');
const identities=readJson('team-identities.json');
const liveContext=readJson('live-context.json');
const forwardLedger=readJson('forward-ledger-summary.json');

if(!schedule || !Array.isArray(schedule.games)) fail('today-games.json missing or malformed');
else ok('schedule readable');

if(projections){
  if(projections.schema_version!=='1.0') fail('game projections schema_version must be 1.0');
  if(!projections.model?.model_id || !projections.model?.version || !projections.model?.feature_version) fail('game projections model metadata incomplete');
  if(!Array.isArray(projections.projections)) fail('game projections array missing');
  else {
    const scheduleIds=new Set((schedule?.games||[]).map(g=>String(g.game_id)));
    const scheduleById=new Map((schedule?.games||[]).map(g=>[String(g.game_id),g]));
    const seen=new Set();
    for(const g of projections.projections){
      const id=String(g.game_id);
      if(seen.has(id)) fail('duplicate projection game_id '+id); else seen.add(id);
      if(schedule && !scheduleIds.has(id)) fail('projection game_id not in today schedule: '+id);
      const scheduled=scheduleById.get(id);
      if(scheduled){
        if(g.away?.abbr!==scheduled.away_abbr) fail(id+' away abbreviation does not match schedule');
        if(g.home?.abbr!==scheduled.home_abbr) fail(id+' home abbreviation does not match schedule');
        if(g.game_date && schedule?.date && g.game_date!==schedule.date) fail(id+' game_date does not match schedule date');
      }
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

        if(!scheduled?.start_time_et) fail(id+' published without exact scheduled start_time_et');
        const startMs=scheduled?.start_time_et ? Date.parse(scheduled.start_time_et) : NaN;
        const generatedMs=Date.parse(projections.generated_at);
        const cutoffMs=Date.parse(projections.data_cutoff);
        if(!Number.isFinite(generatedMs) || !Number.isFinite(cutoffMs)) fail(id+' invalid projection timestamps');
        if(Number.isFinite(startMs)){
          if(generatedMs>startMs) fail(id+' generated after puck drop');
          if(cutoffMs>startMs) fail(id+' data cutoff after puck drop');
          if(cutoffMs>generatedMs) fail(id+' data cutoff later than generated_at');
        }
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
  if(players.schema_version!=='1.1') fail('player-ratings schema_version must be 1.1');
  if(!players.model?.model_id || !players.model?.version || !players.model?.status) fail('player-ratings model metadata incomplete');
  if(!Array.isArray(players.players)) fail('player-ratings players must be array');
  else {
    const canonical=new Set(['ANA','BOS','BUF','CAR','CBJ','CGY','CHI','COL','DAL','DET','EDM','FLA','LAK','MIN','MTL','NJD','NSH','NYI','NYR','OTT','PHI','PIT','SEA','SJS','STL','TBL','TOR','UTA','VAN','VGK','WPG','WSH']);
    const ids=new Set();
    for(const p of players.players){
      const id=String(p.player_id);
      if(!p.name || !p.team_abbr || !p.position) fail('player rating missing identity '+id);
      if(!canonical.has(p.team_abbr)) fail('player '+id+' unknown team '+p.team_abbr);
      if(ids.has(id)) fail('duplicate player_id '+id); else ids.add(id);
      for(const k of ['overall','offense','defense','shot_generation','finishing','playmaking','usage']){
        if(!finiteOrNull(p[k])) fail('player '+id+' '+k+' must be finite or null');
      }
    }
    if(players.model.status==='PROMOTED' && players.players.length===0) fail('PROMOTED player-ratings export cannot be empty');
    ok('player ratings readable');
  }
}

if(fantasy){
  if(fantasy.schema_version!=='1.1') fail('fantasy-rankings schema_version must be 1.1');
  if(!fantasy.model?.model_id || !fantasy.model?.version || !fantasy.model?.status) fail('fantasy model metadata incomplete');
  if(!Array.isArray(fantasy.rankings)) fail('fantasy rankings must be array');
  else {
    const ids=new Set(), ranks=new Set();
    for(const p of fantasy.rankings){
      const id=String(p.player_id);
      if(ids.has(id)) fail('duplicate fantasy player_id '+id); else ids.add(id);
      if(ranks.has(p.rank)) fail('duplicate fantasy rank '+p.rank); else ranks.add(p.rank);
      for(const k of ['ros_projection','trend','schedule_score']){
        if(!finiteOrNull(p[k])) fail('fantasy player '+id+' '+k+' must be finite or null');
      }
    }
    if(fantasy.model.status==='PROMOTED' && fantasy.rankings.length===0) fail('PROMOTED fantasy export cannot be empty');
    ok('fantasy rankings readable');
  }
}

if(identities){
  if(identities.schema_version!=='1.0') fail('team-identities schema_version must be 1.0');
  if(!Array.isArray(identities.teams)) fail('team-identities teams must be array');
  else {
    const ids=new Set();
    for(const t of identities.teams){
      if(!t.abbr || !t.name || !t.logo_url) fail('team identity incomplete');
      if(ids.has(t.abbr)) fail('duplicate team identity '+t.abbr); else ids.add(t.abbr);
    }
    if(ids.size!==32) fail('team-identities must contain exactly 32 teams');
    ok('team identities readable');
  }
}else fail('team-identities.json missing');

if(liveContext){
  if(liveContext.schema_version!=='1.0') fail('live-context schema_version must be 1.0');
  if(liveContext.date!==schedule?.date) fail('live-context date does not match schedule');
  if(!Array.isArray(liveContext.games)) fail('live-context games must be array');
  else {
    const scheduleById=new Map((schedule?.games||[]).map(g=>[String(g.game_id),g]));
    const seen=new Set();
    for(const g of liveContext.games){
      const id=String(g.game_id);
      if(seen.has(id)) fail('duplicate live-context game '+id); else seen.add(id);
      const s=scheduleById.get(id);
      if(!s) fail('live-context game not on schedule '+id);
      for(const side of ['away','home']){
        const t=g?.[side];
        if(!t?.abbr) fail('live-context '+id+' missing '+side+' team');
        if(s && t?.abbr!==s[side+'_abbr']) fail('live-context '+id+' '+side+' team mismatch');
        const goalie=t?.goalie||{};
        if(!['UNKNOWN','PROJECTED','CONFIRMED'].includes(goalie.status)) fail('live-context '+id+' invalid goalie status');
        if(['PROJECTED','CONFIRMED'].includes(goalie.status) && !goalie.name) fail('live-context '+id+' goalie status without name');
        if(goalie.status==='CONFIRMED' && (!goalie.source_timestamp || !goalie.source_url)) fail('live-context '+id+' confirmed goalie missing source evidence');
        const lineup=t?.lineup||{};
        if(!['UNKNOWN','PROJECTED'].includes(lineup.status)) fail('live-context '+id+' invalid lineup status');
        if(lineup.status==='PROJECTED'){
          for(const group of ['f1','f2','f3','f4','d1','d2','d3']){
            if(!Array.isArray(lineup.groups?.[group]) || lineup.groups[group].length===0) fail('live-context '+id+' '+side+' missing '+group);
          }
          if(!lineup.updated_at) fail('live-context '+id+' '+side+' projected lineup missing updated_at');
        }
      }
    }
    if(seen.size!==(schedule?.games||[]).length) fail('live-context must cover entire current slate');
    ok('live pregame context readable');
  }
}else fail('live-context.json missing');



const forwardGamesDir=path.join(root,'artifacts','forward','games');
if(fs.existsSync(forwardGamesDir)){
  const files=fs.readdirSync(forwardGamesDir).filter(x=>x.endsWith('.json'));
  for(const file of files){
    const rec=JSON.parse(fs.readFileSync(path.join(forwardGamesDir,file),'utf8'));
    const id=String(rec.game_id);
    if(rec.schema_version!=='1.0') fail('forward game '+id+' schema_version must be 1.0');
    const startMs=Date.parse(rec.start_time_et);
    const frozenMs=Date.parse(rec.frozen_at);
    const generatedMs=Date.parse(rec.forecast?.generated_at);
    const cutoffMs=Date.parse(rec.forecast?.data_cutoff);
    if(!Number.isFinite(startMs) || !Number.isFinite(frozenMs)) fail('forward game '+id+' invalid start/freeze timestamp');
    if(Number.isFinite(startMs) && Number.isFinite(frozenMs) && frozenMs>=startMs) fail('forward game '+id+' frozen at or after puck drop');
    if(!Number.isFinite(generatedMs) || generatedMs>startMs) fail('forward game '+id+' projection generated after puck drop');
    if(!Number.isFinite(cutoffMs) || cutoffMs>startMs) fail('forward game '+id+' data cutoff after puck drop');
    if(Number.isFinite(generatedMs) && Number.isFinite(cutoffMs) && cutoffMs>generatedMs) fail('forward game '+id+' data cutoff later than projection generation');
    const a=rec.forecast?.away?.win_probability;
    const h=rec.forecast?.home?.win_probability;
    if(typeof a!=='number' || typeof h!=='number' || Math.abs((a+h)-1)>0.001) fail('forward game '+id+' invalid frozen win probabilities');
    const status=rec.result?.status;
    if(!['OPEN','FINAL'].includes(status)) fail('forward game '+id+' invalid result status');
    if(status==='FINAL'){
      if(!rec.result?.winner_abbr) fail('forward game '+id+' FINAL without winner');
      if(!rec.evaluation || typeof rec.evaluation.winner_correct!=='boolean') fail('forward game '+id+' FINAL without evaluation');
    }else if(rec.evaluation!==null){
      fail('forward game '+id+' OPEN record must not have evaluation');
    }
  }
  ok('forward per-game immutability/timestamp checks complete');
}

if(forwardLedger){
  if(forwardLedger.schema_version!=='1.0') fail('forward-ledger summary schema_version must be 1.0');
  const s=forwardLedger.sample||{};
  for(const k of ['games_frozen','games_settled','games_open','winner_games_scored','xg_games_scored']){
    if(!Number.isInteger(s[k]) || s[k]<0) fail('forward-ledger sample '+k+' invalid');
  }
  if(Number.isInteger(s.games_frozen) && Number.isInteger(s.games_settled) && Number.isInteger(s.games_open) &&
     s.games_settled+s.games_open!==s.games_frozen) fail('forward-ledger frozen count must equal settled + open');
  const m=forwardLedger.metrics||{};
  for(const k of ['accuracy','brier','log_loss','avg_probability_assigned_to_winner','xg_mae_per_team','total_goals_mae']){
    if(!finiteOrNull(m[k])) fail('forward-ledger metric '+k+' must be finite or null');
  }
  if(!Array.isArray(forwardLedger.games)) fail('forward-ledger games must be array');
  else {
    const ids=new Set();
    for(const g of forwardLedger.games){
      const id=String(g.game_id);
      if(ids.has(id)) fail('duplicate forward-ledger game_id '+id); else ids.add(id);
      if(!['OPEN','FINAL'].includes(g.status)) fail('forward-ledger '+id+' invalid status');
      if(g.status==='FINAL' && !g.winner_abbr) fail('forward-ledger '+id+' FINAL without winner');
      for(const k of ['brier','log_loss','probability_assigned_to_winner','goal_mae_per_team','total_goals_abs_error']){
        if(!finiteOrNull(g[k])) fail('forward-ledger '+id+' '+k+' must be finite or null');
      }
    }
    if(ids.size!==s.games_frozen) fail('forward-ledger summary row count does not match games_frozen');
  }
  ok('forward prediction ledger summary readable');
}else fail('forward-ledger-summary.json missing');

if(manifest){
  if(manifest.schema_version!=='1.0') fail('production-manifest schema_version must be 1.0');
  if(!manifest.surfaces || typeof manifest.surfaces!=='object') fail('production-manifest surfaces missing');
  else {
    const expected={
      schedule:'today-games.json',
      game_projections:'game-projections.json',
      model_performance:'model-performance.json',
      team_ratings:'team-ratings.json',
      player_ratings:'player-ratings.json',
      fantasy_rankings:'fantasy-rankings.json',
      team_identities:'team-identities.json',
      live_context:'live-context.json',
      forward_ledger:'forward-ledger-summary.json'
    };
    for(const [key,file] of Object.entries(expected)){
      const s=manifest.surfaces[key];
      if(!s) { fail('production-manifest missing surface '+key); continue; }
      if(!['EMPTY','READY','WITHHELD'].includes(s.state)) fail('invalid manifest state '+key);
      if(s.max_age_minutes!==null && (!Number.isInteger(s.max_age_minutes) || s.max_age_minutes<1)) fail('invalid max_age_minutes '+key);
      const rel=s.file?.replace(/^public\/data\//,'');
      if(rel!==file) fail('manifest file mismatch for '+key);
      if(!fs.existsSync(path.join(dataDir,file))) fail('manifest points to missing file '+file);
    }
    if(manifest.surfaces.game_projections?.state==='READY'){
      if(projections?.model?.status!=='PROMOTED') fail('manifest game_projections READY without PROMOTED model');
      const scheduledIds=new Set((schedule?.games||[]).map(g=>String(g.game_id)));
      const projectedIds=new Set((projections?.projections||[]).map(g=>String(g.game_id)));
      for(const id of scheduledIds) if(!projectedIds.has(id)) fail('READY game projections missing scheduled game '+id);
      if(projectedIds.size!==scheduledIds.size) fail('READY game projections coverage does not exactly match slate');
      const ms=manifest.surfaces.game_projections;
      if(ms.model_id!==projections?.model?.model_id || ms.model_version!==projections?.model?.version) fail('game projection manifest model metadata mismatch');
    }
    if(manifest.surfaces.model_performance?.state==='READY'){
      if(!(perf?.sample?.games_scored>0)) fail('manifest model_performance READY without scored games');
      const ms=manifest.surfaces.model_performance;
      if(ms.model_id!==perf?.model?.model_id || ms.model_version!==perf?.model?.version) fail('model performance manifest metadata mismatch');
    }
    if(manifest.surfaces.team_ratings?.state==='READY'){
      if(teams?.model?.status!=='PROMOTED') fail('manifest team_ratings READY without PROMOTED model');
      const ms=manifest.surfaces.team_ratings;
      if(ms.model_id!==teams?.model?.model_id || ms.model_version!==teams?.model?.version) fail('team ratings manifest metadata mismatch');
    }
    if(manifest.surfaces.player_ratings?.state==='READY'){
      if(players?.model?.status!=='PROMOTED') fail('manifest player_ratings READY without PROMOTED model');
      const ms=manifest.surfaces.player_ratings;
      if(ms.model_id!==players?.model?.model_id || ms.model_version!==players?.model?.version) fail('player ratings manifest metadata mismatch');
    }
    if(manifest.surfaces.fantasy_rankings?.state==='READY'){
      if(fantasy?.model?.status!=='PROMOTED') fail('manifest fantasy_rankings READY without PROMOTED model');
      const ms=manifest.surfaces.fantasy_rankings;
      if(ms.model_id!==fantasy?.model?.model_id || ms.model_version!==fantasy?.model?.version) fail('fantasy rankings manifest metadata mismatch');
    }
    ok('production manifest readable');
  }
}else fail('production-manifest.json missing');

if(process.exitCode) process.exit(process.exitCode);
console.log('All production data checks passed.');
