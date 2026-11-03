(function(){
  function set(key,value){
    const el=document.querySelector('[data-performance="'+key+'"]');
    if(el) el.textContent=value;
  }
  function num(v,d=3){ return Number.isFinite(v) ? v.toFixed(d) : '—'; }
  async function load(){
    try{
      const r=await fetch('/data/model-performance.json',{cache:'no-store'});
      if(!r.ok) return;
      const d=await r.json();
      const n=d?.sample?.games_scored;
      const m=d?.metrics||{};
      set('games_scored',Number.isInteger(n)&&n>0?String(n):'—');
      set('brier',num(m.brier));
      set('log_loss',num(m.log_loss));
      set('calibration',num(m.calibration));
    }catch(e){}
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',load); else load();
})();