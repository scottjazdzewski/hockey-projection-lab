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
    text=soup.get_text(" ",strip=True)
    print("HAS_CONFIRMED","Confirmed" in text,"HAS_UNCONFIRMED","Unconfirmed" in text,flush=True)
    print("TEXT_PREFIX",text[:500].replace("\n"," "),flush=True)
