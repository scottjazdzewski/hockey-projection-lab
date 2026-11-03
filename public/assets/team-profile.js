(function(){
  const metrics=['overall','offense','defense','five_on_five','power_play','penalty_kill','goaltending','form'];
  function qs(id){ return document.getElementById(id); }
  function fmt(v){
    if(!Number.isFinite(v)) return '—';
    return Math.abs(v)>=10 ? v.toFixed(0) : v.toFixed(1);
  }
  function teamParam(){
    return (new URLSearchParams(location.search).get('team')||'').trim().toUpperCase();
  }
  async function load(){
    const abbr=teamParam();
    try{
      const identities=await window.HockeyData.json('/data/team-identities.json');
      const identity=(identities.teams||[]).find(t=>t.abbr===abbr);
      if(!identity){
        qs('tp-title').textContent='Team not found';
        qs('tp-state').textContent='Choose a team from Team Profiles.';
        return;
      }
      document.title=identity.name+' | Hockey Lab';
      qs('tp-title').textContent=identity.name;
      qs('tp-name').textContent=identity.name;
      qs('tp-abbr').textContent=identity.abbr;
      qs('tp-logo').src=identity.logo_url;
      qs('tp-logo').alt=identity.name+' logo';

      const result=await window.HockeyData.loadSurface('team_ratings','/data/team-ratings.json');
      if(!result.ready || result.data?.model?.status!=='PROMOTED'){
        qs('tp-state').textContent=result.stale?'Ratings temporarily stale.':'Ratings pending.';
        return;
      }
      const team=(result.data.teams||[]).find(t=>t.abbr===abbr);
      if(!team){
        qs('tp-state').textContent='Ratings withheld for this team.';
        return;
      }
      for(const k of metrics){
        const el=document.querySelector('[data-team-metric="'+k+'"]');
        if(el) el.textContent=fmt(team[k]);
      }
      qs('tp-state').textContent='Promoted team-strength output.';
      qs('tp-model').textContent='Model: '+result.data.model.model_id+' '+result.data.model.version;
      qs('tp-updated').textContent='Updated: '+(team.as_of||result.data.generated_at||'—');
    }catch(e){
      qs('tp-state').textContent='Team data temporarily unavailable.';
    }
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',load); else load();
})();