(function(){
  function fmt(v){
    if(!Number.isFinite(v)) return null;
    return (Math.abs(v)>=10 ? v.toFixed(0) : v.toFixed(1));
  }
  async function load(){
    try{
      const r=await fetch('/data/team-ratings.json',{cache:'no-store'});
      if(!r.ok) return;
      const d=await r.json();
      if(d?.schema_version!=='1.1' || d?.model?.status!=='PROMOTED' || !Array.isArray(d.teams)) return;
      const byAbbr=new Map(d.teams.map(t=>[t.abbr,t]));
      document.querySelectorAll('.ip-team-card[data-team]').forEach(card=>{
        const t=byAbbr.get(card.dataset.team);
        if(!t) return;
        const overall=fmt(t.overall);
        if(overall===null) return;
        let badge=card.querySelector('.ip-team-rating');
        if(!badge){
          badge=document.createElement('span');
          badge.className='ip-team-rating';
          card.appendChild(badge);
        }
        badge.textContent=overall;
        badge.setAttribute('aria-label','Overall rating '+overall);
        card.dataset.ratingReady='true';
      });
    }catch(e){}
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',load); else load();
})();