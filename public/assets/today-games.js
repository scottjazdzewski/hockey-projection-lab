(function(){
  const logo=a=>'https://assets.nhle.com/logos/nhl/svg/'+a+'_light.svg';
  const esc=s=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
  function miniGame(g){
    return '<div class="today-mini-game">'+
      '<span class="today-mini-time">'+esc(g.time_et||'TBD')+'</span>'+
      '<span class="today-mini-team"><img src="'+logo(g.away_abbr)+'" alt=""><b>'+esc(g.away_abbr)+'</b></span>'+
      '<span class="today-mini-at">@</span>'+
      '<span class="today-mini-team"><img src="'+logo(g.home_abbr)+'" alt=""><b>'+esc(g.home_abbr)+'</b></span>'+
    '</div>';
  }
  function fullGame(g){
    return '<article class="hwin-card live-slate-card">'+
      '<div class="hwin-top"><span>TONIGHT · HWIN PENDING</span><span>'+esc(g.time_et||'TBD')+'</span></div>'+
      '<div class="hwin-matchup">'+
        '<div class="hwin-team"><img src="'+logo(g.away_abbr)+'" alt="'+esc(g.away_name)+'"><strong>'+esc(g.away_name.replace(/^(Anaheim|Boston|Buffalo|Calgary|Carolina|Chicago|Colorado|Columbus|Dallas|Detroit|Edmonton|Florida|Los Angeles|Minnesota|Montreal|Nashville|New Jersey|New York|Ottawa|Philadelphia|Pittsburgh|San Jose|Seattle|St\. Louis|Tampa Bay|Toronto|Utah|Vancouver|Vegas|Washington|Winnipeg)\s+/,'')||g.away_abbr)+'</strong><small>'+esc(g.away_abbr)+' · Away</small></div>'+
        '<div class="hwin-center"><strong>—</strong><span>HWIN</span><span style="margin-top:8px">@</span></div>'+
        '<div class="hwin-team"><img src="'+logo(g.home_abbr)+'" alt="'+esc(g.home_name)+'"><strong>'+esc(g.home_name.replace(/^(Anaheim|Boston|Buffalo|Calgary|Carolina|Chicago|Colorado|Columbus|Dallas|Detroit|Edmonton|Florida|Los Angeles|Minnesota|Montreal|Nashville|New Jersey|New York|Ottawa|Philadelphia|Pittsburgh|San Jose|Seattle|St\. Louis|Tampa Bay|Toronto|Utah|Vancouver|Vegas|Washington|Winnipeg)\s+/,'')||g.home_abbr)+'</strong><small>'+esc(g.home_abbr)+' · Home</small></div>'+
      '</div>'+
      '<div class="hwin-metrics"><div><strong>—</strong><span>Expected goals</span></div><div><strong>—</strong><span>Regulation win</span></div><div><strong>—</strong><span>Fair price</span></div><div><strong>—</strong><span>OT probability</span></div></div>'+
      '<div class="hwin-context"><span class="context-chip">Goalies —</span><span class="context-chip">Rest —</span><span class="context-chip">Lineup —</span><span class="context-chip">Model —</span></div>'+
      '<div class="hwin-footer"><small>Real NHL matchup. Probability withheld until the live scorer clears validation.</small></div>'+
    '</article>';
  }
  async function load(){
    try{
      const r=await fetch('/data/today-games.json',{cache:'no-store'});
      if(!r.ok) throw new Error('schedule');
      const d=await r.json();
      const games=Array.isArray(d.games)?d.games:[];
      const count=document.getElementById('home-today-count');
      if(count) count.textContent=games.length ? games.length+' game'+(games.length===1?'':'s')+' today' : 'No games today';
      const grid=document.getElementById('todays-game-grid');
      const slate=document.getElementById('today-slate-label');
      if(slate) slate.textContent=(d.display_date||d.date||'Today')+' · '+games.length+' game'+(games.length===1?'':'s');
      if(grid) grid.innerHTML=games.length?games.map(fullGame).join(''):'<article class="profile-placeholder"><div class="eyebrow">Off day</div><h2>No NHL games today.</h2><p>The next slate will populate automatically.</p></article>';
    }catch(e){
      const count=document.getElementById('home-today-count');
      if(count) count.textContent='Today’s NHL slate';
    }
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',load); else load();
})();