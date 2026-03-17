from dotenv import load_dotenv
import os,time,requests,numpy as np,joblib
API_URL=os.getenv("API_URL","https://algotrading.sanyamchhabra.in/")
API_KEY=os.getenv("TEAM_API_KEY","ak_3754ebcebb49579826c2d14fbddf27b4")
HEADERS={"X-API-Key":API_KEY}
CONF_THR=0.25
PARAMS={
    "pos_pct": 0.58,
    "stop": 0.03,
    "profit": 0.05,
    "trail": 0.02,
    "max_hold": 150,
    "cooldown": 180,
    "fee": 0.001
}
model=joblib.load("model.pkl")
def get_price():
    r=requests.get(f"{API_URL}/api/price",headers=HEADERS,timeout=5); r.raise_for_status(); return r.json()
def get_portfolio():
    r=requests.get(f"{API_URL}/api/portfolio",headers=HEADERS,timeout=5); r.raise_for_status(); return r.json()
def get_history():
    r=requests.get(f"{API_URL}/api/history",headers=HEADERS,timeout=5); r.raise_for_status(); return r.json()
def buy(q):
    r=requests.post(f"{API_URL}/api/buy",json={"quantity":int(q)},headers=HEADERS,timeout=5); r.raise_for_status(); return r.json()
def sell(q):
    r=requests.post(f"{API_URL}/api/sell",json={"quantity":int(q)},headers=HEADERS,timeout=5); r.raise_for_status(); return r.json()
def features(hc,hv,ho,hh,hl):
    if len(hc)<35: return None
    n=min(len(hc),60)
    c=np.array(hc[-n:],dtype=float); v=np.array(hv[-n:],dtype=float)
    o=np.array(ho[-n:],dtype=float); h=np.array(hh[-n:],dtype=float); l=np.array(hl[-n:],dtype=float)
    def rs(a,w): return np.std(a[-w:]) if len(a)>=w else np.nan
    def rm(a,w): return np.mean(a[-w:]) if len(a)>=w else np.nan
    def pc(a,w): return (a[-1]/a[-w-1]-1) if len(a)>w else np.nan
    row=[]
    for w in [3,5,10,20]:
        rv=rs(c,w); rp=rs(c[:-1],w); row.append(rv); row.append(rv/rp if rp and rp>0 else np.nan)
    for w in [5,10,20]:
        vm=rm(v,w); row.append(v[-1]/vm if vm and vm>0 else np.nan)
    for w in [1,2,3,5,10,20]: row.append(pc(c,w))
    s5=rm(c,5); s20=rm(c,20); s10=rm(c,10); s30=rm(c,30)
    row.append((s5/s20-1) if s20 and s20>0 else np.nan)
    row.append((s10/s30-1) if s30 and s30>0 else np.nan)
    for w in [10,20,30]:
        mu=rm(c,w); sd=rs(c,w); row.append((c[-1]-mu)/sd if sd and sd>0 else np.nan)
    row.append((c[-1]-o[-1])/o[-1] if o[-1]>0 else np.nan)
    row.append((h[-1]-max(c[-1],o[-1]))/o[-1] if o[-1]>0 else np.nan)
    row.append((min(c[-1],o[-1])-l[-1])/o[-1] if o[-1]>0 else np.nan)
    row.append((h[-1]-l[-1])/o[-1] if o[-1]>0 else np.nan)
    if len(c)>=15:
        d=np.diff(c[-15:]); g=np.mean(np.clip(d,0,None)); ls_=np.mean(np.clip(-d,0,None))
        row.append(100-(100/(1+g/ls_)) if ls_>0 else 100.0)
    else: row.append(np.nan)
    a=np.array(row,dtype=float); a[np.isnan(a)]=0.0; return a.reshape(1,-1)
if __name__=="__main__":
    hc,hv,ho,hh,hl=[],[],[],[],[]; ep=pp=None; hb=cb=0; p=PARAMS
    try:
        for t in get_history():
            hc.append(float(t["close"])); hv.append(float(t.get("volume",0)))
            ho.append(float(t.get("open",t["close"]))); hh.append(float(t.get("high",t["close"])))
            hl.append(float(t.get("low",t["close"])))
        print(f"Warmed up: {len(hc)} ticks")
    except Exception as e: print(f"Warmup: {e}")
    print("Agent running. Ctrl+C to stop.")
    while True:
        try:
            tick=get_price(); port=get_portfolio(); price=float(tick["close"])
            hc.append(price); hv.append(float(tick.get("volume",0)))
            ho.append(float(tick.get("open",price))); hh.append(float(tick.get("high",price)))
            hl.append(float(tick.get("low",price)))
            if tick.get("phase")=="closed":
                print(f"Closed. nw=${port['net_worth']:,.0f} pnl={port['pnl_pct']:+.2f}%"); break
            if hb>0: hb+=1; pp=max(pp or price,price)
            if cb>0: cb-=1
            if port["shares"]>0 and ep:
                ret=(price-ep)/ep; trail=(price-pp)/pp if pp else 0
                if ret<-p["stop"] or trail<-p["trail"] or ret>p["profit"] or hb>=p["max_hold"]:
                    sell(port["shares"]); print(f"SELL @ {price:.4f} ret={ret*100:+.2f}%")
                    ep=None; pp=None; hb=0; cb=p["cooldown"]
            elif port["shares"]==0 and cb==0:
                f=features(hc,hv,ho,hh,hl)
                if f is not None:
                    conf=float(model.predict_proba(f)[0,1])
                    if conf>=CONF_THR:
                        qty=min(int(port["cash"]*p["pos_pct"]/price),int(port["net_worth"]*0.58/price))
                        if qty>0: buy(qty); ep=price; pp=price; hb=1; print(f"BUY {qty} @ {price:.4f} conf={conf:.3f}")
                    else: print(f"HOLD | {price:.4f} | conf={conf:.3f} | pnl={port['pnl_pct']:+.2f}%")
            else: print(f"HOLD | {price:.4f} | pnl={port['pnl_pct']:+.2f}%")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code==429: time.sleep(10)
            else: print(f"HTTP: {e}")
        except KeyboardInterrupt: print("Stopped."); break
        except Exception as e: print(f"Error: {e}")
        time.sleep(10)