(function(){
  const labels={
    schedule:'Schedule',
    team_identities:'Team identities',
    game_projections:'Game projections',
    model_performance:'Model performance',
    team_ratings:'Team ratings',
    player_ratings:'Player ratings',
    fantasy_rankings:'Fantasy rankings'
  };

  async function load(){
    const grid=document.getElementById('status-grid');
    try{
      const m=await window.HockeyData.manifest();
      const order=['schedule','team_identities','game_projections','model_performance','team_ratings','player_ratings','fantasy_rankings'];
      const cards=[];
      for(const key of order){
        const s=m?.surfaces?.[key];
        if(!s) continue;
        const rel=(s.file||'').replace(/^public\/data\//,'');
        let d=null;
        try{ d=await window.HockeyData.json('/data/'+rel); }catch(e){}
        const generated=d?.generated_at||d?.updated_at||'—';
        const stale=window.HockeyData.isStale(generated,s.max_age_minutes);
        const model=d?.model ? [d.model.model_id,d.model.version].filter(Boolean).join(' ') : (s.model_id||'—');
        const count=Array.isArray(d?.games)?d.games.length:
          Array.isArray(d?.projections)?d.projections.length:
          Array.isArray(d?.teams)?d.teams.length:
          Array.isArray(d?.players)?d.players.length:
          Array.isArray(d?.rankings)?d.rankings.length:
          d?.sample?.games_scored??'—';
        const pillClass=s.state==='READY'?'status-ready':s.state==='WITHHELD'?'status-withheld':'status-empty';
        cards.push('<article class="status-card">'+
          '<div class="status-head"><h2>'+labels[key]+'</h2><span class="status-pill '+pillClass+'">'+s.state+(stale?' · STALE':'')+'</span></div>'+
          '<dl class="status-meta">'+
            '<dt>Owner</dt><dd>'+String(s.owner||'—')+'</dd>'+
            '<dt>Model</dt><dd>'+String(model||'—')+'</dd>'+
            '<dt>Generated</dt><dd class="'+(stale?'status-stale':'')+'">'+String(generated)+'</dd>'+
            '<dt>Records</dt><dd>'+String(count)+'</dd>'+
            '<dt>File</dt><dd>'+String(s.file||'—')+'</dd>'+
          '</dl></article>');
      }
      grid.innerHTML=cards.join('');
    }catch(e){
      grid.innerHTML='<div class="status-loading">Production status is temporarily unavailable.</div>';
    }
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',load); else load();
})();