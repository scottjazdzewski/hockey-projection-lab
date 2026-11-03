(function(){
  async function load(){
    const count=document.getElementById('home-today-count');
    if(!count) return;
    try{
      const result=await window.HockeyData.loadSurface('schedule','/data/today-games.json');
      const d=result.data;
      const games=Array.isArray(d?.games)?d.games:[];
      if(result.stale){
        count.textContent='Schedule updating';
        return;
      }
      count.textContent=games.length
        ? games.length+' game'+(games.length===1?'':'s')+' today'
        : 'No games today';
    }catch(e){
      count.textContent='Today’s NHL slate';
    }
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',load); else load();
})();