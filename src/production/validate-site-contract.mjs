import fs from 'node:fs';
import path from 'node:path';

const root=process.cwd();
const required=[
  'public/data/today-games.json',
  'public/data/game-projections.json',
  'public/data/model-performance.json',
  'public/data/team-ratings.json',
  'public/data/player-ratings.json',
  'public/data/fantasy-rankings.json',
  'public/data/production-manifest.json',
  'public/data/team-identities.json',
  'public/data/live-context.json',
  'public/assets/production-data.js',
  'public/assets/game-cards.js',
  'public/assets/team-ratings.js',
  'public/assets/team-profile.js',
  'public/assets/performance.js',
  'public/assets/today-games.js',
  'public/index.html',
  'public/game-cards/index.html',
  'public/teams/index.html',
  'public/teams/profile/index.html',
  'public/performance/index.html'
];

let failed=false;
function fail(msg){ console.error('FAIL:',msg); failed=true; }
function pass(msg){ console.log('PASS:',msg); }
function read(p){ return fs.readFileSync(path.join(root,p),'utf8'); }

for(const p of required){
  if(!fs.existsSync(path.join(root,p))) fail('missing '+p);
}
if(fs.existsSync(path.join(root,'public/assets/site.css'))) fail('legacy public/assets/site.css must not exist');

const wiring=[
  ['public/index.html',['/assets/production-data.js','/assets/today-games.js']],
  ['public/game-cards/index.html',['/assets/production-data.js','/assets/game-cards.js']],
  ['public/teams/index.html',['/assets/production-data.js','/assets/team-ratings.js']],
  ['public/teams/profile/index.html',['/assets/production-data.js','/assets/team-profile.js']],
  ['public/performance/index.html',['/assets/production-data.js','/assets/performance.js']]
];
for(const [p,needles] of wiring){
  if(!fs.existsSync(path.join(root,p))) continue;
  const s=read(p);
  for(const n of needles) if(!s.includes(n)) fail(p+' missing '+n);
}

const teams=read('public/teams/index.html');
const links=[...teams.matchAll(/href="\/teams\/profile\/\?team=([A-Z]+)"/g)].map(m=>m[1]);
if(new Set(links).size!==32) fail('Team Profiles must link exactly 32 unique team profile routes');
else pass('32 team profile routes wired');

const manifest=JSON.parse(read('public/data/production-manifest.json'));
for(const [name,s] of Object.entries(manifest.surfaces||{})){
  const file=s.file?.replace(/^public\//,'public/');
  if(file && !fs.existsSync(path.join(root,file))) fail('manifest '+name+' points to missing '+file);
}
const gameCardsJs=read('public/assets/game-cards.js');
if(!gameCardsJs.includes("const schedule=await json('/data/today-games.json')")) fail('game-cards schedule independence missing');
else pass('game-cards schedule independence checked');

pass('required production site files checked');

if(failed) process.exit(1);
console.log('Site contract checks passed.');
