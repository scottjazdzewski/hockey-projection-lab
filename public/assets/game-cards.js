(function(){
  const logo=a=>'https://assets.nhle.com/logos/nhl/svg/'+a+'_light.svg';
  const esc=s=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
  const cityPrefix=/^(Anaheim|Boston|Buffalo|Calgary|Carolina|Chicago|Colorado|Columbus|Dallas|Detroit|Edmonton|Florida|Los Angeles|Minnesota|Montreal|Nashville|New Jersey|New York|Ottawa|Philadelphia|Pittsburgh|San Jose|Seattle|St\. Louis|Tampa Bay|Toronto|Utah|Vancouver|Vegas|Washington|Winnipeg)\s+/;

  function nick(name,abbr){ return String(name||abbr||'').replace(cityPrefix,'') || abbr || ''; }
  function pct(v){ return Number.isFinite(v) ? Math.round(v*100)+'%' : '—'; }
  function goals(a,h){ return Number.isFinite(a)&&Number.isFinite(h) ? a.toFixed(2)+' · '+h.toFixed(2) : '—'; }
  function pairPct(a,h){ return Number.isFinite(a)&&Number.isFinite(h) ? Math.round(a*100)+' · '+Math.round(h*100)+'%' : '—'; }
  function price(v){ return Number.isInteger(v) ? (v>0?'+':'')+v : '—'; }
  function context(label,state){
    if(!state || !state.status || state.status==='UNKNOWN') return label+' —';
    return label+' '+String(state.status).toLowerCase();
  }

  function usableProjection(bundle,p,manifestReady){
    if(!manifestReady || !bundle || bundle.model?.status!=='PROMOTED' || !p || p.publish_state!=='PUBLISHED') return false;
    const a=p.away?.win_probability, h=p.home?.win_probability;
    return Number.isFinite(a)&&Number.isFinite(h)&&Math.abs((a+h)-1)<=0.001;
  }

  function card(g,pBundle,p,projectionSurface){
    const away=nick(g.away_name,g.away_abbr);
    const home=nick(g.home_name,g.home_abbr);
    const awayClass=away.length>12?' gc-long':'';
    const homeClass=home.length>12?' gc-long':'';
    const live=usableProjection(pBundle,p,manifestReady);
    const state=String(g.state||'PRE').toUpperCase();
    const gameLabel=state==='LIVE'?'Live':(state==='FINAL'||state==='OFF')?'Final':'Tonight';
    const modelLabel=live ? 'Model '+esc(pBundle.model.version) : 'Model pending';
    const homeProb=live ? pct(p.home.win_probability) : '—';
    const xg=live ? goals(p.away.expected_goals,p.home.expected_goals) : '—';
    const reg=live ? pairPct(p.away.regulation_win_probability,p.home.regulation_win_probability) : '—';
    const fair=live ? price(p.home.fair_moneyline) : '—';
    const ot=live ? pct(p.overtime_probability) : '—';
    const goalies=live ? context('Goalies',p.goalie_state) : 'Goalies —';
    const rest=live ? context('Rest',p.rest_state) : 'Rest —';
    const lineup=live ? context('Lineup',p.lineup_state) : 'Lineup —';
    const modelPill=live ? 'Model live' : (surfaceState==='WITHHELD'?'Model withheld':'Model —');
    const footer=live
      ? 'Pregame projection · '+esc(pBundle.model.model_id)+' '+esc(pBundle.model.version)+' · cutoff '+esc(pBundle.data_cutoff||'')
      : (surfaceState==='WITHHELD'
          ? 'Real matchup · projection withheld by the production validation gate.'
          : 'Real matchup · model values publish only after a promoted pregame scorer clears validation.');

    return '<article class="gc-card">'+
      '<div class="gc-card-head"><span>'+gameLabel+' · '+modelLabel+'</span><span>'+esc(g.time_et||'TBD')+'</span></div>'+
      '<div class="gc-matchup">'+
        '<div class="gc-team"><img src="'+logo(g.away_abbr)+'" alt="'+esc(g.away_name||g.away_abbr)+'"><div class="gc-team-name'+awayClass+'">'+esc(away)+'</div><small>'+esc(g.away_abbr)+' · Away</small></div>'+
        '<div class="gc-prob"><strong>'+homeProb+'</strong><span>Home win</span><b>@</b></div>'+
        '<div class="gc-team"><img src="'+logo(g.home_abbr)+'" alt="'+esc(g.home_name||g.home_abbr)+'"><div class="gc-team-name'+homeClass+'">'+esc(home)+'</div><small>'+esc(g.home_abbr)+' · Home</small></div>'+
      '</div>'+
      '<div class="gc-metrics">'+
        '<div class="gc-metric"><strong>'+xg+'</strong><span>Expected goals A · H</span></div>'+
        '<div class="gc-metric"><strong>'+reg+'</strong><span>Reg win A · H</span></div>'+
        '<div class="gc-metric"><strong>'+fair+'</strong><span>Home fair price</span></div>'+
        '<div class="gc-metric"><strong>'+ot+'</strong><span>OT probability</span></div>'+
      '</div>'+
      '<div class="gc-pills"><span class="gc-pill">'+esc(goalies)+'</span><span class="gc-pill">'+esc(rest)+'</span><span class="gc-pill">'+esc(lineup)+'</span><span class="gc-pill">'+esc(modelPill)+'</span></div>'+
      '<div class="gc-card-foot">'+footer+'</div>'+
    '</article>';
  }

  async function json(url){
    const r=await fetch(url,{cache:'no-store'});
    if(!r.ok) throw new Error(url);
    return r.json();
  }

  async function load(){
    const dateEl=document.getElementById('gc-date');
    const grid=document.getElementById('gc-grid');
    try{
      const [schedule,projectionSurface]=await Promise.all([
        window.HockeyData.json('/data/today-games.json'),
        window.HockeyData.loadSurface('game_projections','/data/game-projections.json')
      ]);
      const projectionResult=projectionSurface.data;
      const games=Array.isArray(schedule.games)?schedule.games:[];
      const projections=projectionResult && Array.isArray(projectionResult.projections) ? projectionResult.projections : [];
      const byId=new Map(projections.map(p=>[String(p.game_id),p]));
      if(dateEl) dateEl.textContent=(schedule.display_date||schedule.date||'Today')+' · '+games.length+' game'+(games.length===1?'':'s');
      if(grid) grid.innerHTML=games.length
        ? games.map(g=>card(g,projectionResult,byId.get(String(g.game_id)),projectionSurface)).join('')
        : '<div class="gc-loading">No NHL games scheduled today.</div>';
    }catch(e){
      if(dateEl) dateEl.textContent='Today’s NHL slate';
      if(grid) grid.innerHTML='<div class="gc-loading">Schedule temporarily unavailable.</div>';
    }
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',load); else load();
})();