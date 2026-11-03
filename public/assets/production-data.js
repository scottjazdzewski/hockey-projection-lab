(function(){
  const cache=new Map();

  async function json(url){
    const r=await fetch(url,{cache:'no-store'});
    if(!r.ok) throw new Error(url+' '+r.status);
    return r.json();
  }

  async function manifest(){
    if(!cache.has('manifest')){
      cache.set('manifest',json('/data/production-manifest.json').catch(()=>null));
    }
    return cache.get('manifest');
  }

  async function surface(name){
    const m=await manifest();
    return m?.surfaces?.[name]||null;
  }

  async function isReady(name){
    const s=await surface(name);
    return s?.state==='READY';
  }

  function isStale(generatedAt,maxAgeMinutes){
    if(!generatedAt || !Number.isFinite(maxAgeMinutes) || maxAgeMinutes<=0) return false;
    const t=Date.parse(generatedAt);
    if(!Number.isFinite(t)) return true;
    return (Date.now()-t) > maxAgeMinutes*60*1000;
  }

  async function loadSurface(name,url){
    const [m,d]=await Promise.all([manifest(),json(url).catch(()=>null)]);
    const s=m?.surfaces?.[name]||null;
    if(!s || s.state!=='READY' || !d) return {ready:false,surface:s,data:d,stale:false};
    const stamp=d.generated_at||d.updated_at||null;
    const stale=isStale(stamp,s.max_age_minutes);
    return {ready:!stale,surface:s,data:d,stale};
  }

  window.HockeyData={json,manifest,surface,isReady,isStale,loadSurface};
})();