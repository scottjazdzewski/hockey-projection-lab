(function(){
  const logo=a=>'https://assets.nhle.com/logos/nhl/svg/'+a+'_light.svg';
  const esc=s=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
  const cityPrefix=/^(Anaheim|Boston|Buffalo|Calgary|Carolina|Chicago|Colorado|Columbus|Dallas|Detroit|Edmonton|Florida|Los Angeles|Minnesota|Montreal|Nashville|New Jersey|New York|Ottawa|Philadelphia|Pittsburgh|San Jose|Seattle|St\. Louis|Tampa Bay|Toronto|Utah|Vancouver|Vegas|Washington|Winnipeg)\s+/;

  function nick(name,abbr){
    return String(name||abbr||'').replace(cityPrefix,'') || abbr || '';
  }

  function card(g){
    const away=nick(g.away_name,g.away_abbr);
    const home=nick(g.home_name,g.home_abbr);
    const awayClass=away.length>12?' gc-long':'';
    const homeClass=home.length>12?' gc-long':'';

    return '<article class="gc-card">'+
      '<div class="gc-card-head"><span>Tonight · Model pending</span><span>'+esc(g.time_et||'TBD')+'</span></div>'+
      '<div class="gc-matchup">'+
        '<div class="gc-team">'+
          '<img src="'+logo(g.away_abbr)+'" alt="'+esc(g.away_name||g.away_abbr)+'">'+
          '<div class="gc-team-name'+awayClass+'">'+esc(away)+'</div>'+
          '<small>'+esc(g.away_abbr)+' · Away</small>'+
        '</div>'+
        '<div class="gc-prob"><strong>—</strong><span>Win prob.</span><b>@</b></div>'+
        '<div class="gc-team">'+
          '<img src="'+logo(g.home_abbr)+'" alt="'+esc(g.home_name||g.home_abbr)+'">'+
          '<div class="gc-team-name'+homeClass+'">'+esc(home)+'</div>'+
          '<small>'+esc(g.home_abbr)+' · Home</small>'+
        '</div>'+
      '</div>'+
      '<div class="gc-metrics">'+
        '<div class="gc-metric"><strong>—</strong><span>Expected goals</span></div>'+
        '<div class="gc-metric"><strong>—</strong><span>Regulation win</span></div>'+
        '<div class="gc-metric"><strong>—</strong><span>Fair price</span></div>'+
        '<div class="gc-metric"><strong>—</strong><span>OT probability</span></div>'+
      '</div>'+
      '<div class="gc-pills">'+
        '<span class="gc-pill">Goalies —</span>'+
        '<span class="gc-pill">Rest —</span>'+
        '<span class="gc-pill">Lineup —</span>'+
        '<span class="gc-pill">Model —</span>'+
      '</div>'+
      '<div class="gc-card-foot">Real matchup · model values publish after the pregame scorer clears validation.</div>'+
    '</article>';
  }

  async function load(){
    const dateEl=document.getElementById('gc-date');
    const grid=document.getElementById('gc-grid');
    try{
      const r=await fetch('/data/today-games.json',{cache:'no-store'});
      if(!r.ok) throw new Error('schedule');
      const d=await r.json();
      const games=Array.isArray(d.games)?d.games:[];
      if(dateEl) dateEl.textContent=(d.display_date||d.date||'Today')+' · '+games.length+' game'+(games.length===1?'':'s');
      if(grid) grid.innerHTML=games.length?games.map(card).join(''):'<div class="gc-loading">No NHL games scheduled today.</div>';
    }catch(e){
      if(dateEl) dateEl.textContent='Today’s NHL slate';
      if(grid) grid.innerHTML='<div class="gc-loading">Schedule temporarily unavailable.</div>';
    }
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',load); else load();
})();