(function(){
  function set(key,value){
    const el=document.querySelector('[data-performance="'+key+'"]');
    if(el) el.textContent=value;
  }
  function setForward(key,value){
    const el=document.querySelector('[data-forward="'+key+'"]');
    if(el) el.textContent=value;
  }
  function num(v,d=3){ return Number.isFinite(v) ? v.toFixed(d) : '—'; }
  function pct(v){ return Number.isFinite(v) ? (v*100).toFixed(1)+'%' : '—'; }

  async function loadBacktest(){
    try{
      const result=await window.HockeyData.loadSurface('model_performance','/data/model-performance.json');
      if(!result.ready) return;
      const d=result.data;
      const n=d?.sample?.games_scored;
      const m=d?.metrics||{};
      set('games_scored',Number.isInteger(n)&&n>0?String(n):'—');
      set('brier',num(m.brier));
      set('log_loss',num(m.log_loss));
      set('calibration',num(m.calibration));
    }catch(e){}
  }

  async function loadForward(){
    try{
      const result=await window.HockeyData.loadSurface('forward_ledger','/data/forward-ledger-summary.json');
      if(!result.ready) return;
      const d=result.data||{};
      const s=d.sample||{};
      const m=d.metrics||{};
      const sides=d.by_pick_side||{};
      const away=sides.away||{};
      const home=sides.home||{};

      setForward('games_settled',Number.isInteger(s.games_settled)?String(s.games_settled):'0');
      setForward('accuracy',pct(m.accuracy));
      setForward('brier',num(m.brier));
      setForward('log_loss',num(m.log_loss));
      setForward('away_accuracy',pct(away.accuracy));
      setForward('home_accuracy',pct(home.accuracy));
      setForward('away_picks',(Number.isInteger(away.games)?away.games:0)+' picks');
      setForward('home_picks',(Number.isInteger(home.games)?home.games:0)+' picks');
    }catch(e){}
  }

  async function load(){
    await Promise.all([loadBacktest(),loadForward()]);
  }

  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',load); else load();
})();
