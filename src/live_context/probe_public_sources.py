import json, re, sys
import requests
from bs4 import BeautifulSoup

URLS=[
  "https://www.dailyfaceoff.com/starting-goalies",
  "https://www.dailyfaceoff.com/teams",
  "https://www.dailyfaceoff.com/teams/anaheim-ducks/line-combinations",
]
headers={"User-Agent":"HockeyProjectionLab/0.1 (+private research; contact via repository owner)"}
for url in URLS:
    print("URL",url,flush=True)
    r=requests.get(url,headers=headers,timeout=45)
    print("STATUS",r.status_code,"BYTES",len(r.content),"FINAL",r.url,flush=True)
    r.raise_for_status()
    soup=BeautifulSoup(r.text,"html.parser")
    scripts=soup.find_all("script")
    print("SCRIPT_IDS",[s.get("id") for s in scripts if s.get("id")][:20],flush=True)
    nxt=soup.find("script",id="__NEXT_DATA__")
    print("NEXT_DATA",bool(nxt),flush=True)
    if nxt and nxt.string:
        data=json.loads(nxt.string)
        print("NEXT_TOP_KEYS",list(data.keys()),flush=True)
        pp=data.get("props",{}).get("pageProps",{})
        print("PAGEPROP_KEYS",list(pp.keys())[:100],flush=True)
        # recursively report list-valued paths and sample object keys, not values
        def walk(x,path="pageProps",depth=0):
            if depth>4:return
            if isinstance(x,dict):
                for k,v in list(x.items())[:100]:
                    walk(v,path+"."+str(k),depth+1)
            elif isinstance(x,list):
                if x:
                    sample=x[0]
                    sk=list(sample.keys())[:40] if isinstance(sample,dict) else type(sample).__name__
                    print("LIST",path,"N",len(x),"SAMPLE_KEYS",sk,flush=True)
                    if isinstance(sample,(dict,list)): walk(sample,path+"[0]",depth+1)
        walk(pp)
        if url.endswith("/starting-goalies") and pp.get("data"):
            wanted=["homeTeamName","homeTeamAbbreviation","homeGoalieName","homeNewsStrengthName","homeNewsCreatedAt","homeNewsSourceName","homeNewsSourceUrl","awayTeamName","awayTeamAbbreviation","awayGoalieName","awayNewsStrengthName","awayNewsCreatedAt","awayNewsSourceName","awayNewsSourceUrl","date"]
            for rec in pp["data"][:2]:
                print("GOALIE_SAMPLE",{k:rec.get(k) for k in wanted if k in rec},flush=True)
            for rec in pp["data"]:
                if rec.get("homeNewsStrengthName") or rec.get("awayNewsStrengthName"):
                    print("GOALIE_CONFIRMED_SAMPLE",{k:rec.get(k) for k in wanted if k in rec},flush=True)
                    break
        if "/line-combinations" in url and isinstance(pp.get("combinations"),dict):
            comb=pp["combinations"]
            print("COMB_KEYS",list(comb.keys()),flush=True)
            print("LINES_SAMPLE",comb.get("lines",[])[:13],flush=True)
            players=comb.get("players",[])
            print("PLAYER_SAMPLE",[{k:x.get(k) for k in ["name","positionIdentifier","groupIdentifier","groupName","categoryIdentifier","categoryName","injuryStatus","gameTimeDecision"]} for x in players[:12]],flush=True)
    text=soup.get_text(" ",strip=True)
    print("HAS_CONFIRMED","Confirmed" in text,"HAS_UNCONFIRMED","Unconfirmed" in text,flush=True)
    print("TEXT_PREFIX",text[:500].replace("\n"," "),flush=True)
