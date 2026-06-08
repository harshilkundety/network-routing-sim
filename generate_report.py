# 6G Intent-Aware Routing — Report Generator
# Output: 6G_Simulation_Report.pdf

import sys, math, random, json, time, os
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

#edit for whatever you want
CFG = {
    "n":         10,       # nodes
    "conn":       3,       # connectivity
    "ticks":    200,       # ticks per run
    "rate":       3,       # packets injected per tick
    "emerg_frac":20,       # % emergency traffic
    "proc_cap":   2,       # node processing capacity
    "scenario": "uniform", # uniform | burst | disaster | hotspot | ramp | voice
    "num_runs":  30,       # statistical runs
    "seed":      42,       # reproducibility seed
}
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "6G_Simulation_Report.pdf")

#ui stuff
BG     = "#0d1117"
BG2    = "#161b22"
BG3    = "#1c2330"
BORDER = "#30363d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"
RED    = "#f85149"
GREEN  = "#3fb950"
AMBER  = "#d29922"
BLUE   = "#58a6ff"
PURPLE = "#a371f7"
CYAN   = "#00d2ff"

def avg(lst):  return sum(lst) / len(lst) if lst else 0.0
def std(lst):
    if len(lst) < 2: return 0.0
    m = avg(lst)
    return math.sqrt(sum((x - m)**2 for x in lst) / len(lst))
def pct(a, b): return f"{(a-b)/b*100:.1f}" if b else "—"
def clamp(v, lo, hi): return max(lo, min(hi, v))
def rand(a, b):    return a + random.random() * (b - a)
def rand_int(a, b): return random.randint(a, b)

KW = {
    "mayday":9,"sos":9,"tornado":9,"cyclone":9,"tsunami":9,"hurricane":9,
    "fire":8,"emergency":8,"explosion":8,"disaster":8,
    "wildfire":8,"flood":8,"avalanche":8,"earthquake":8,"storm":8,
    "bomb":7,"attack":7,"collapse":7,"blizzard":7,"eruption":7,
    "nuclear":7,"chemical":7,"evacuation":7,"trapped":7,"casualty":7,
    "drowning":7,"outage":7,
    "critical":6,"ambulance":6,"rescue":6,"injured":6,"hostage":6,
    "crash":5,"accident":5,"interference":5,"signal_loss":5,
    "urgent":4,"warning":4,"alert":4,"danger":4,"hazard":4,
    "help":3,"degraded":3,
    "problem":2,"issue":2,"priority":1,"high":1,
    "hello":0,"hi":0,"test":0,"ok":0,"status":0,
    "update":0,"ping":0,"good":0,"normal":0,
}
SYNONYMS = {
    "burning":"fire","blaze":"fire","ablaze":"fire","inferno":"fire","conflagration":"fire",
    "medical":"ambulance","injury":"injured","injuries":"injured","wounded":"injured","hurt":"injured",
    "stranded":"trapped","missing":"rescue",
    "blackout":"outage","collision":"crash","wreck":"crash",
    "toxic":"chemical","radiation":"nuclear",
    "sinking":"drowning","submerging":"drowning",
    "imminent":"warning","severe":"critical",
}
PHRASES = [
    ("building is burning","fire"),("building on fire","fire"),
    ("mass casualty","casualty"),("structural collapse","collapse"),
    ("gas leak","chemical"),("active shooter","attack"),
    ("search and rescue","rescue"),("evacuation order","evacuation"),
    ("missing person","rescue"),("water rising","flood"),
    ("power outage","outage"),("wind damage","storm"),
]
CLASSIFIER_DATASET = [
    ("fire emergency sector 7","HIGH"),("mayday engine failure","HIGH"),
    ("flood warning critical zone","HIGH"),("sos rescue needed","HIGH"),
    ("earthquake collapse warning","HIGH"),("tornado alert evacuate now","HIGH"),
    ("hurricane force winds incoming","HIGH"),("tsunami warning coastal region","HIGH"),
    ("wildfire advancing evacuation","HIGH"),("storm surge disaster","HIGH"),
    ("avalanche emergency rescue","HIGH"),("blizzard evacuation critical","HIGH"),
    ("emergency call dispatch unit","HIGH"),("chemical outage hazard zone","HIGH"),
    ("nuclear plant explosion","HIGH"),("bomb threat near facility","HIGH"),
    ("casualty report incoming attack","HIGH"),("wildfire rescue trapped","HIGH"),
    ("hostage situation explosion","HIGH"),("drowning victim river flood","HIGH"),
    ("eruption volcanic emergency","HIGH"),("cyclone approaching landfall","HIGH"),
    ("nuclear explosion detected","HIGH"),("disaster relief rescue operation","HIGH"),
    ("attack on infrastructure bomb","HIGH"),("tsunami coastal collapse","HIGH"),
    ("chemical plant fire explosion","HIGH"),("wildfire evacuation critical status","HIGH"),
    ("mayday distress signal","HIGH"),("sos flood emergency","HIGH"),
    ("earthquake rescue casualty","HIGH"),("hurricane evacuation critical","HIGH"),
    ("storm emergency rescue","HIGH"),("fire collapse rescue team","HIGH"),
    ("bomb explosion attack site","HIGH"),("trapped miners rescue emergency","HIGH"),
    ("flood casualty rescue","HIGH"),("blizzard critical rescue","HIGH"),
    ("nuclear fallout evacuation","HIGH"),("drowning casualty flood","HIGH"),
    ("collapse building rescue","HIGH"),("chemical attack zone","HIGH"),
    ("eruption lava emergency","HIGH"),("avalanche rescue critical","HIGH"),
    ("disaster earthquake collapse","HIGH"),
    ("building is burning","HIGH"),("warehouse blaze spreading fast","HIGH"),
    ("structure ablaze with heavy smoke","HIGH"),("critical injury multiple victims","HIGH"),
    ("severe injuries at crash site","HIGH"),("multiple wounded soldiers need rescue","HIGH"),
    ("vessel sinking crew needs help","HIGH"),("radiation leak causing casualties","HIGH"),
    ("toxic spill near residential casualties","HIGH"),("blackout causing widespread casualties","HIGH"),
    ("people hurt in collision","HIGH"),("mass casualty event from structural collapse","HIGH"),
    ("evacuation order issued stranded residents","HIGH"),("inferno consuming the district","HIGH"),
    ("severe injury needs immediate care","HIGH"),("collision involving multiple vehicles casualties","HIGH"),
    ("sinking ship stranded passengers","HIGH"),("gas leak near evacuation zone","HIGH"),
    ("missing crew members on sinking vessel","HIGH"),("building ablaze rescue requested","HIGH"),
    ("workers hurt in collision evacuation","HIGH"),("toxic spill stranded workers","HIGH"),
    ("radiation leak evacuation order","HIGH"),("inferno spreading trapped residents","HIGH"),
    ("blaze at warehouse injured workers","HIGH"),
    ("urgent sensor offline maintenance","MEDIUM"),("ambulance requested downtown","MEDIUM"),
    ("hazard detected area b","MEDIUM"),("help needed at checkpoint","MEDIUM"),
    ("interference detected rerouting","MEDIUM"),("signal degraded alert","MEDIUM"),
    ("network outage partial restore","MEDIUM"),("warning system activated","MEDIUM"),
    ("danger zone proximity","MEDIUM"),("crash report road a12","MEDIUM"),
    ("accident at junction 7","MEDIUM"),("rescue team on standby","MEDIUM"),
    ("ambulance en route to scene","MEDIUM"),("urgent repair needed infrastructure","MEDIUM"),
    ("alert level elevated sector 5","MEDIUM"),("hazard posted on road","MEDIUM"),
    ("signal interference reported","MEDIUM"),("system degraded performance","MEDIUM"),
    ("outage affecting sector 3","MEDIUM"),("help required at station 4","MEDIUM"),
    ("rescue personnel deployed","MEDIUM"),("crash barriers activated","MEDIUM"),
    ("injured worker at site","MEDIUM"),("critical infrastructure check","MEDIUM"),
    ("accident involving two vehicles","MEDIUM"),("interference on channel 7","MEDIUM"),
    ("urgent maintenance required","MEDIUM"),("rescue drone deployed","MEDIUM"),
    ("alert issued for area","MEDIUM"),("sensor interference reported","MEDIUM"),
    ("warning signal activated sector","MEDIUM"),("danger of overheating detected","MEDIUM"),
    ("ambulance on standby","MEDIUM"),("help desk offline temporarily","MEDIUM"),
    ("crash detection system online","MEDIUM"),("injured persons reported","MEDIUM"),
    ("outage warning zone five","MEDIUM"),("critical system check pending","MEDIUM"),
    ("accident reconstruction team","MEDIUM"),("signal degraded rerouting","MEDIUM"),
    ("help coordinate response","MEDIUM"),("alert raised by sensor","MEDIUM"),
    ("hazard assessment ongoing","MEDIUM"),("interference mitigation applied","MEDIUM"),
    ("rescue helicopter on patrol","MEDIUM"),("urgent network maintenance","MEDIUM"),
    ("warning issued for weather","MEDIUM"),("crash site investigation","MEDIUM"),
    ("dangerous goods vehicle alert","MEDIUM"),("injured worker evacuated","MEDIUM"),
    ("critical check required","MEDIUM"),("accident investigation underway","MEDIUM"),
    ("danger alert at main gate","MEDIUM"),("hazard warning flashing","MEDIUM"),
    ("urgent warning from sensor","MEDIUM"),("critical alert raised","MEDIUM"),
    ("critical interference detected","MEDIUM"),
    ("blackout reported in district","MEDIUM"),("collision on motorway","MEDIUM"),
    ("vehicle wreck reported","MEDIUM"),("severe weather advisory","MEDIUM"),
    ("imminent weather concern","MEDIUM"),("workers missing from site","MEDIUM"),
    ("stranded vehicle on highway","MEDIUM"),("medical team required","MEDIUM"),
    ("persons hurt in fall","MEDIUM"),("power blackout district 4","MEDIUM"),
    ("two vehicles collision reported","MEDIUM"),("imminent system failure concern","MEDIUM"),
    ("severe outage risk zone","MEDIUM"),
    ("status update nominal","NORMAL"),("hello network ping","NORMAL"),
    ("ok test signal","NORMAL"),("diagnostic routine complete","NORMAL"),
    ("voice call setup complete","NORMAL"),("bandwidth allocation request","NORMAL"),
    ("system status normal","NORMAL"),("network check passed","NORMAL"),
    ("ping response received","NORMAL"),("update installed successfully","NORMAL"),
    ("connection established","NORMAL"),("data transfer complete","NORMAL"),
    ("node health check ok","NORMAL"),("routine maintenance scheduled","NORMAL"),
    ("configuration update applied","NORMAL"),("test signal strong","NORMAL"),
    ("firmware update complete","NORMAL"),("system reboot initiated","NORMAL"),
    ("network latency low","NORMAL"),("hello world test","NORMAL"),
    ("bandwidth utilization 42 percent","NORMAL"),("good signal quality detected","NORMAL"),
    ("node registration complete","NORMAL"),("uptime report generated","NORMAL"),
    ("voice quality excellent","NORMAL"),("packet delivery confirmed","NORMAL"),
    ("normal operations resumed","NORMAL"),("system log archived","NORMAL"),
    ("frequency scan complete","NORMAL"),("protocol handshake ok","NORMAL"),
    ("backup connection established","NORMAL"),("node synchronization done","NORMAL"),
    ("software version updated","NORMAL"),("latency within normal range","NORMAL"),
    ("calibration complete sensor","NORMAL"),("satellite link nominal","NORMAL"),
    ("test transmission received","NORMAL"),("status ok all systems","NORMAL"),
    ("routine data collection","NORMAL"),("communication channel clear","NORMAL"),
    ("system performance good","NORMAL"),("ping latency low","NORMAL"),
    ("hello from node five","NORMAL"),("update server complete","NORMAL"),
    ("test mode disabled","NORMAL"),("user authentication successful","NORMAL"),
    ("backup process finished","NORMAL"),("log file rotated","NORMAL"),
    ("memory usage normal","NORMAL"),("cpu load acceptable","NORMAL"),
    ("dns resolution ok","NORMAL"),("load balancer active","NORMAL"),
    ("ssl certificate valid","NORMAL"),("backup sync complete","NORMAL"),
    ("monitoring dashboard updated","NORMAL"),("disk space adequate","NORMAL"),
    ("network throughput normal","NORMAL"),("system uptime satisfactory","NORMAL"),
    ("help me find this file","NORMAL"),("please help with network setup","NORMAL"),
    ("urgent schedule meeting tomorrow","NORMAL"),("please help configure system","NORMAL"),
    ("help resolve the issue","NORMAL"),("warning light is normal behavior","NORMAL"),
    ("crash course in networking","NORMAL"),("alert me when done","NORMAL"),
    ("help desk ticket 1234","NORMAL"),("rescue helicopter sound file","NORMAL"),
    ("accident report template download","NORMAL"),("urgent team standup now","NORMAL"),
]

def _score(words):
    import re
    score = 0.0; matched = []
    for i, w in enumerate(words):
        v = KW.get(w)
        if v is not None and v > 0:
            score += v * max(0.5, 1 - i * 0.04)
            matched.append(f"{w}:{v}")
    score = round(score * 10) / 10
    if score >= 8:   priority, level = "HIGH", 3
    elif score >= 3: priority, level = "MEDIUM", 2
    else:            priority, level = "NORMAL", 1
    return {"priority": priority, "level": level, "score": score, "matched": matched}

def classify(text):
    import re
    return _score(re.findall(r'\w+', (text or "").lower()))

def classify_v2(text):
    import re
    t = (text or "").lower()
    extra = [kw for phrase, kw in PHRASES if phrase in t]
    words = extra + [SYNONYMS.get(w, w) for w in re.findall(r'\w+', t)]
    return _score(words)

def eval_classifier():
    labels = ["NORMAL", "MEDIUM", "HIGH"]; li = {l: i for i, l in enumerate(labels)}
    results = {}
    for tag, fn in [("v1", classify), ("v2", classify_v2)]:
        conf = [[0]*3 for _ in range(3)]
        for text, true_lbl in CLASSIFIER_DATASET:
            conf[li[true_lbl]][li[fn(text)["priority"]]] += 1
        total = len(CLASSIFIER_DATASET)
        correct = sum(conf[i][i] for i in range(3))
        acc = correct / total * 100
        pr_list, re_list, f1_list = [], [], []
        for i in range(3):
            tp = conf[i][i]
            fp = sum(conf[r][i] for r in range(3)) - tp
            fn_ = sum(conf[i][c] for c in range(3)) - tp
            p = tp/(tp+fp) if (tp+fp) else 0.0
            r = tp/(tp+fn_) if (tp+fn_) else 0.0
            f = 2*p*r/(p+r) if (p+r) else 0.0
            pr_list.append(p); re_list.append(r); f1_list.append(f)
        results[tag] = {
            "accuracy":  round(acc, 1),
            "precision": round(avg(pr_list)*100, 1),
            "recall":    round(avg(re_list)*100, 1),
            "f1":        round(avg(f1_list)*100, 1),
            "confusion": conf, "class_labels": labels,
        }
    return results

def build_topo(n, conn):
    W, H = 300, 200; cx, cy = W/2, H/2
    nodes = []
    for i in range(n):
        a = (2*math.pi*i/n) + rand(-0.3, 0.3)
        r = rand(0.22, 0.42) * min(W, H)
        nodes.append({"id":i, "x":clamp(cx+r*math.cos(a),20,W-20),
                       "y":clamp(cy+r*math.sin(a),20,H-20), "neighbors":[]})
    edges = []; adj = [[False]*n for _ in range(n)]
    for i in range(n):
        dists = sorted([{"j":j,"d":math.hypot(nodes[j]["x"]-nodes[i]["x"],
                         nodes[j]["y"]-nodes[i]["y"])} for j in range(n) if j!=i],
                       key=lambda x: x["d"])
        for k in range(min(conn, len(dists))):
            j = dists[k]["j"]
            if not adj[i][j]:
                adj[i][j]=adj[j][i]=True; edges.append({"a":i,"b":j})
                nodes[i]["neighbors"].append(j); nodes[j]["neighbors"].append(i)
    for nd in nodes:
        if not nd["neighbors"]:
            j = (nd["id"]+1)%n
            if not adj[nd["id"]][j]:
                adj[nd["id"]][j]=adj[j][nd["id"]]=True; edges.append({"a":nd["id"],"b":j})
                nd["neighbors"].append(j); nodes[j]["neighbors"].append(nd["id"])
    return {"nodes":nodes,"edges":edges}

# ─────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────
def generate_traffic_sequence(ticks, rate, emerg_frac, scenario, n):
    seq = []
    for t in range(ticks):
        seq.append([]); inj = rate; eff = emerg_frac
        if scenario=="ramp":    inj = max(1, math.ceil(rate*(t/ticks*2)))
        elif scenario=="burst": inj = rate*3 if ticks*0.3<t<ticks*0.6 else rate
        elif scenario=="disaster": inj=math.ceil(rate*2); eff=65
        elif scenario=="voice": inj=math.ceil(rate*1.4); eff=15
        for _ in range(inj):
            is_e = random.random()<eff/100
            if scenario=="burst" and ticks*0.3<t<ticks*0.6: lv=3
            elif scenario=="disaster": lv=3 if is_e else (2 if random.random()<0.25 else 1)
            else: lv=3 if is_e else (2 if random.random()<0.3 else 1)
            src=rand_int(0,n-1); dst=rand_int(0,n-1); tries=0
            while dst==src and tries<20: dst=rand_int(0,n-1); tries+=1
            if scenario=="hotspot": dst=0
            seq[t].append({"src":src,"dst":dst,"level":lv})
    return seq

W_ADAPTIVE = {"cong":1.2,"queue":0.9,"delay":0.6,"loss":0.7,"prog":0.8,"emerg_mult":2.5}
W_BASELINE = {"cong":0.0,"queue":0.0,"delay":0.0,"loss":0.0,"prog":1.0,"emerg_mult":1.0}

def sim_run(topo, traffic_seq, mode, proc_cap):
    N=len(topo["nodes"]); ticks=len(traffic_seq)
    W=W_ADAPTIVE if mode=="adaptive" else W_BASELINE
    ns=[{"id":nd["id"],"neighbors":list(nd["neighbors"]),"queue":[],
         "max_queue":10,"congestion":0.0,"delay":0.0,"loss_rate":0.0}
        for nd in topo["nodes"]]
    stats={"delivered":0,"dropped":0,"total":0,"emerg_lat":[],"normal_lat":[],
           "med_lat":[],"per_tick_cong":[],"per_tick_in_flight":[],"control_msgs":0}
    pkt_id=0; live={}

    def cost(to_id, pkt):
        node=ns[to_id]; dp=topo["nodes"][pkt["dst"]]; tp=topo["nodes"][to_id]
        dist=math.hypot(tp["x"]-dp["x"],tp["y"]-dp["y"])
        prog=1.0/(1.0+dist); em=W["emerg_mult"] if pkt["level"]==3 else 1.0
        return (W["cong"]*node["congestion"]*em+W["queue"]*len(node["queue"])/node["max_queue"]
                +W["delay"]*node["delay"]+W["loss"]*node["loss_rate"]-W["prog"]*prog*200)

    def next_hop(pkt):
        cur=ns[pkt["cur"]]
        if not cur["neighbors"]: return None
        if pkt["dst"] in cur["neighbors"]: return pkt["dst"]
        best,best_c=None,float("inf"); prev=pkt["path"][-2] if len(pkt["path"])>=2 else -1
        for nid in cur["neighbors"]:
            c=cost(nid,pkt)
            if nid==prev: c+=50
            if c<best_c: best_c=c; best=nid
        return best

    def deliver(pkt,t):
        lat=(t-pkt["born"])*10+pkt["hops"]*2
        if pkt["level"]==3: stats["emerg_lat"].append(lat)
        elif pkt["level"]==2: stats["med_lat"].append(lat)
        else: stats["normal_lat"].append(lat)
        stats["delivered"]+=1; live.pop(pkt["id"],None)

    def drop(pkt): stats["dropped"]+=1; live.pop(pkt["id"],None)

    for t in range(ticks):
        for pk in traffic_seq[t]:
            pid=pkt_id; pkt_id+=1
            pkt={"id":pid,"src":pk["src"],"dst":pk["dst"],"level":pk["level"],
                 "cur":pk["src"],"path":[pk["src"]],"born":t,"hops":0,"max_hops":N*3}
            stats["total"]+=1
            if pk["src"]==pk["dst"]: deliver(pkt,t); continue
            sn=ns[pk["src"]]
            if len(sn["queue"])<sn["max_queue"]: sn["queue"].append(pkt); live[pkt["id"]]=pkt
            else: drop(pkt)
        if t%5==0:
            alpha=0.5 if mode=="adaptive" else 0.15
            for node in ns:
                for nid in node["neighbors"]:
                    nb=ns[nid]
                    nb["congestion"]=nb["congestion"]*(1-alpha)+(len(nb["queue"])/nb["max_queue"])*alpha
                    nb["delay"]=nb["delay"]*0.8+(len(nb["queue"])*0.12)*0.2
                    stats["control_msgs"]+=1
        for node in ns:
            if mode=="adaptive": node["queue"].sort(key=lambda p: -p["level"])
            to_fwd=node["queue"][:proc_cap]; node["queue"]=node["queue"][proc_cap:]
            for pkt in to_fwd:
                if pkt["id"] not in live: continue
                if pkt["hops"]>=pkt["max_hops"]: drop(pkt); continue
                nh=next_hop(pkt)
                if nh is None: drop(pkt); continue
                pkt["path"].append(nh); pkt["hops"]+=1; pkt["cur"]=nh
                if nh==pkt["dst"]: deliver(pkt,t); continue
                nn=ns[nh]
                if len(nn["queue"])<nn["max_queue"]: nn["queue"].append(pkt)
                else:
                    if mode=="adaptive" and pkt["level"]==3:
                        vi=min(range(len(nn["queue"])),key=lambda i:nn["queue"][i]["level"])
                        if nn["queue"][vi]["level"]<pkt["level"]:
                            drop(nn["queue"][vi]); nn["queue"].pop(vi); nn["queue"].append(pkt)
                        else: drop(pkt)
                    else: drop(pkt)
            node["congestion"]=len(node["queue"])/node["max_queue"]
            node["loss_rate"]=clamp(node["loss_rate"]*0.93+(0.07 if node["congestion"]>0.75 else 0),0,1)
        stats["per_tick_cong"].append(avg([n["congestion"] for n in ns]))
        stats["per_tick_in_flight"].append(len(live))
    for _ in live: stats["dropped"]+=1
    stats["total"]=stats["delivered"]+stats["dropped"]
    el=stats["emerg_lat"]; nl=stats["normal_lat"]
    stats["p95_emerg"]  = float(np.percentile(el,95)) if len(el)>=5  else 0.0
    stats["p99_emerg"]  = float(np.percentile(el,99)) if len(el)>=5  else 0.0
    stats["p95_normal"] = float(np.percentile(nl,95)) if len(nl)>=5  else 0.0
    stats["p99_normal"] = float(np.percentile(nl,99)) if len(nl)>=5  else 0.0
    stats["throughput"] = stats["delivered"]/(ticks*0.01) if ticks else 0.0
    stats["loss_rate"]  = stats["dropped"]/stats["total"]*100 if stats["total"] else 0.0
    stats["pdr"]        = stats["delivered"]/stats["total"]*100 if stats["total"] else 0.0
    return stats

def aggregate_stats(runs):
    def _ms(k): return [avg(r[k]) for r in runs]
    def _sc(k): return [r[k] for r in runs]
    def _agg(v): m=avg(v); s=std(v); return round(m,2),round(s,2)
    em=_ms("emerg_lat"); md=_ms("med_lat"); nm=_ms("normal_lat")
    max_t=max(len(r["per_tick_cong"]) for r in runs)
    ptc=[avg([r["per_tick_cong"][ti] for r in runs if ti<len(r["per_tick_cong"])]) for ti in range(max_t)]
    em_m,em_s   =_agg(em);   md_m,md_s=_agg(md);   nm_m,nm_s=_agg(nm)
    p95e_m,p95e_s=_agg(_sc("p95_emerg")); p99e_m,p99e_s=_agg(_sc("p99_emerg"))
    p95n_m,p95n_s=_agg(_sc("p95_normal"))
    thr_m,thr_s  =_agg(_sc("throughput")); lr_m,lr_s=_agg(_sc("loss_rate"))
    pdr_m,pdr_s  =_agg(_sc("pdr")); ctrl_m,_=_agg(_sc("control_msgs"))
    dl_m,_=_agg(_sc("delivered")); dr_m,_=_agg(_sc("dropped")); tot_m,_=_agg(_sc("total"))
    return {
        "emerg_lat_mean":em_m,"emerg_lat_std":em_s,
        "med_lat_mean":md_m,"med_lat_std":md_s,
        "normal_lat_mean":nm_m,"normal_lat_std":nm_s,
        "p95_emerg_mean":p95e_m,"p95_emerg_std":p95e_s,
        "p99_emerg_mean":p99e_m,"p99_emerg_std":p99e_s,
        "p95_normal_mean":p95n_m,"p95_normal_std":p95n_s,
        "throughput_mean":thr_m,"throughput_std":thr_s,
        "loss_rate_mean":lr_m,"loss_rate_std":lr_s,
        "pdr_mean":pdr_m,"pdr_std":pdr_s,
        "control_msgs_mean":ctrl_m,
        "delivered_mean":dl_m,"dropped_mean":dr_m,"total_mean":tot_m,
        "per_tick_cong_mean":ptc,
    }

# ─────────────────────────────────────────────────────────────
# FIGURE STYLE HELPERS
# ─────────────────────────────────────────────────────────────
def style_ax(ax):
    ax.set_facecolor(BG2)
    ax.figure.patch.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
    ax.yaxis.label.set_color(MUTED); ax.xaxis.label.set_color(MUTED)
    ax.grid(color="#21262d", linestyle="-", linewidth=0.4, alpha=0.8)
    ax.set_title(ax.get_title(), color=TEXT, fontsize=10, pad=6)

def dark_fig(w=8, h=5):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    return fig

# ─────────────────────────────────────────────────────────────
# PAGE BUILDERS
# ─────────────────────────────────────────────────────────────
def page_title(pdf, A, B, cls, params):
    fig = dark_fig(11, 8.5)
    ax = fig.add_subplot(111); ax.axis("off"); ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    ax.text(0.5, 0.88, "6G Intent-Aware Routing", transform=ax.transAxes,
            ha="center", va="center", fontsize=28, color=BLUE, fontweight="bold")
    ax.text(0.5, 0.80, "Semantic Edge Network Simulator — Research Report",
            transform=ax.transAxes, ha="center", fontsize=16, color=TEXT)
    ax.text(0.5, 0.73, f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
            transform=ax.transAxes, ha="center", fontsize=11, color=MUTED)
    ax.axhline(y=0.68, xmin=0.1, xmax=0.9, color=BORDER, linewidth=1)
    ax.text(0.5, 0.62, "Experiment Configuration", transform=ax.transAxes,
            ha="center", fontsize=12, color=MUTED, fontweight="bold")
    cfg_lines = [
        f"Scenario: {params['scenario'].upper()}    Nodes: {params['n']}    Connectivity: {params['conn']}",
        f"Ticks/run: {params['ticks']}    Injection rate: {params['rate']} pkts/tick    Emergency fraction: {params['emerg_frac']}%",
        f"Processing capacity: {params['proc_cap']} pkts/tick    Runs: {params['num_runs']}    Seed: {params['seed']}",
        f"Total dataset messages evaluated: {len(CLASSIFIER_DATASET)}",
    ]
    for i, line in enumerate(cfg_lines):
        ax.text(0.5, 0.54 - i*0.06, line, transform=ax.transAxes,
                ha="center", fontsize=10, color=TEXT)
    ax.axhline(y=0.28, xmin=0.1, xmax=0.9, color=BORDER, linewidth=1)
    highlights = [
        ("PDR Adaptive", f"{A['pdr_mean']:.1f} ± {A['pdr_std']:.1f}%", GREEN),
        ("PDR Baseline",  f"{B['pdr_mean']:.1f} ± {B['pdr_std']:.1f}%", AMBER),
        ("Emerg Lat Adaptive", f"{A['emerg_lat_mean']:.1f} ms", GREEN),
        ("Emerg Lat Baseline",  f"{B['emerg_lat_mean']:.1f} ms", AMBER),
        ("Classifier V2 Acc", f"{cls.get('v2',{}).get('accuracy',0):.1f}%", CYAN),
    ]
    for i, (label, val, col) in enumerate(highlights):
        x = 0.12 + i*0.175
        ax.text(x, 0.20, val, transform=ax.transAxes, ha="center",
                fontsize=13, color=col, fontweight="bold")
        ax.text(x, 0.13, label, transform=ax.transAxes, ha="center",
                fontsize=8, color=MUTED)
    ax.text(0.5, 0.04,
            "Adaptive mode: intent-aware, congestion-sensitive, semantic QoS weighting  |  "
            "Baseline mode: greedy shortest-path, no QoS",
            transform=ax.transAxes, ha="center", fontsize=8, color=MUTED)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def page_metrics_table(pdf, A, B, cls, params):
    fig = dark_fig(11, 8.5); fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111); ax.axis("off"); ax.set_facecolor(BG)
    ax.text(0.5, 0.96, "Quantitative Results — Multi-Run Statistics", transform=ax.transAxes,
            ha="center", fontsize=14, color=TEXT, fontweight="bold")
    ax.text(0.5, 0.91, f"Mean ± Std across {params['num_runs']} independent runs (seed=42)",
            transform=ax.transAxes, ha="center", fontsize=10, color=MUTED)

    def _d(a, b, lower=True):
        d = pct(a, b)
        if d == "—": return "—"
        v = float(d)
        better = v < 0 if lower else v > 0
        arrow = "▼" if lower else "▲"
        return f"{arrow} {abs(v):.1f}%" if better else f"{'▲' if lower else '▼'} {abs(v):.1f}%"

    rows = [
        ["Metric", "Adaptive (Intent-Aware)", "Baseline (Greedy)", "Δ Change", "Better?"],
        ["PDR (%)",
         f"{A['pdr_mean']:.1f} ± {A['pdr_std']:.1f}",
         f"{B['pdr_mean']:.1f} ± {B['pdr_std']:.1f}",
         _d(A['pdr_mean'], B['pdr_mean'], lower=False),
         "✓ Adaptive" if A['pdr_mean'] > B['pdr_mean'] else "—"],
        ["Emergency Latency (ms)",
         f"{A['emerg_lat_mean']:.1f} ± {A['emerg_lat_std']:.1f}",
         f"{B['emerg_lat_mean']:.1f} ± {B['emerg_lat_std']:.1f}",
         _d(A['emerg_lat_mean'], B['emerg_lat_mean']),
         "✓ Adaptive" if A['emerg_lat_mean'] < B['emerg_lat_mean'] else "—"],
        ["Medium Latency (ms)",
         f"{A['med_lat_mean']:.1f} ± {A['med_lat_std']:.1f}",
         f"{B['med_lat_mean']:.1f} ± {B['med_lat_std']:.1f}",
         _d(A['med_lat_mean'], B['med_lat_mean']), "—"],
        ["Normal Latency (ms)",
         f"{A['normal_lat_mean']:.1f} ± {A['normal_lat_std']:.1f}",
         f"{B['normal_lat_mean']:.1f} ± {B['normal_lat_std']:.1f}",
         _d(A['normal_lat_mean'], B['normal_lat_mean']), "—"],
        ["P95 Emerg Latency (ms)",
         f"{A['p95_emerg_mean']:.1f} ± {A['p95_emerg_std']:.1f}",
         f"{B['p95_emerg_mean']:.1f} ± {B['p95_emerg_std']:.1f}",
         _d(A['p95_emerg_mean'], B['p95_emerg_mean']),
         "✓ Adaptive" if A['p95_emerg_mean'] < B['p95_emerg_mean'] else "—"],
        ["P99 Emerg Latency (ms)",
         f"{A['p99_emerg_mean']:.1f} ± {A['p99_emerg_std']:.1f}",
         f"{B['p99_emerg_mean']:.1f} ± {B['p99_emerg_std']:.1f}",
         _d(A['p99_emerg_mean'], B['p99_emerg_mean']), "—"],
        ["Throughput (pkts/s)",
         f"{A['throughput_mean']:.1f} ± {A['throughput_std']:.1f}",
         f"{B['throughput_mean']:.1f} ± {B['throughput_std']:.1f}",
         _d(A['throughput_mean'], B['throughput_mean'], lower=False),
         "✓ Adaptive" if A['throughput_mean'] > B['throughput_mean'] else "—"],
        ["Packet Loss Rate (%)",
         f"{A['loss_rate_mean']:.2f} ± {A['loss_rate_std']:.2f}",
         f"{B['loss_rate_mean']:.2f} ± {B['loss_rate_std']:.2f}",
         _d(A['loss_rate_mean'], B['loss_rate_mean']),
         "✓ Adaptive" if A['loss_rate_mean'] < B['loss_rate_mean'] else "—"],
        ["Delivered (mean pkts)",
         f"{A['delivered_mean']:.0f}", f"{B['delivered_mean']:.0f}", "—", "—"],
        ["Control Msgs (mean)",
         f"{A['control_msgs_mean']:.0f}", f"{B['control_msgs_mean']:.0f}", "—", "N/A (Adaptive only)"],
        ["Classifier Accuracy (v1)",
         f"{cls.get('v1',{}).get('accuracy',0):.1f}%", "—", "—", "—"],
        ["Classifier Accuracy (v2)",
         f"{cls.get('v2',{}).get('accuracy',0):.1f}%", "—",
         f"+{cls.get('v2',{}).get('accuracy',0)-cls.get('v1',{}).get('accuracy',0):.1f} pts", "✓ Enhanced"],
    ]

    col_widths = [0.26, 0.20, 0.20, 0.16, 0.18]
    col_x = [0.02, 0.28, 0.48, 0.68, 0.84]
    row_h = 0.055; top_y = 0.85

    for ri, row in enumerate(rows):
        y = top_y - ri * row_h
        bg = BG2 if ri == 0 else (BG if ri % 2 == 0 else BG2)
        rect = FancyBboxPatch((0.01, y - row_h*0.85), 0.98, row_h*0.9,
                               boxstyle="round,pad=0.002", facecolor=bg,
                               edgecolor=BORDER, linewidth=0.5,
                               transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        for ci, cell in enumerate(row):
            col = TEXT
            if ri > 0:
                if ci == 1:  col = GREEN
                elif ci == 2: col = AMBER
                elif ci == 3:
                    if "▼" in str(cell): col = GREEN
                    elif "▲" in str(cell): col = RED
                elif ci == 4 and "✓" in str(cell): col = GREEN
            fw = "bold" if ri == 0 else "normal"
            ax.text(col_x[ci] + col_widths[ci]*0.5, y - row_h*0.3,
                    str(cell), transform=ax.transAxes,
                    fontsize=8.5 if ri > 0 else 9, color=col,
                    ha="center", va="center", fontweight=fw)

    ax.text(0.5, 0.02,
            "▼ = improvement (lower is better)   ▲ = degradation   ✓ = Adaptive outperforms Baseline",
            transform=ax.transAxes, ha="center", fontsize=8, color=MUTED)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def page_routing_charts(pdf, A, B, params):
    fig = dark_fig(11, 8.5); fig.patch.set_facecolor(BG)
    fig.suptitle("Routing Performance — Comparative Analysis", color=TEXT, fontsize=13, y=0.97)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.95, top=0.92, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0]); style_ax(ax1)
    x = np.arange(3)
    ax1.bar(x-0.2, [A["emerg_lat_mean"], A["med_lat_mean"], A["normal_lat_mean"]],
            0.38, label="Adaptive", color=GREEN, alpha=0.9,
            yerr=[A["emerg_lat_std"], A["med_lat_std"], A["normal_lat_std"]],
            capsize=4, error_kw={"ecolor": TEXT, "elinewidth": 1})
    ax1.bar(x+0.2, [B["emerg_lat_mean"], B["med_lat_mean"], B["normal_lat_mean"]],
            0.38, label="Baseline", color=AMBER, alpha=0.9,
            yerr=[B["emerg_lat_std"], B["med_lat_std"], B["normal_lat_std"]],
            capsize=4, error_kw={"ecolor": TEXT, "elinewidth": 1})
    ax1.set_xticks(x); ax1.set_xticklabels(["Emergency","Medium","Normal"], fontsize=8)
    ax1.set_ylabel("Latency (ms)", color=MUTED, fontsize=8)
    ax1.set_title("Latency by Priority Class (mean ± std)", color=TEXT)
    ax1.legend(fontsize=8, facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT)

    ax2 = fig.add_subplot(gs[0, 1]); style_ax(ax2)
    ptc = A["per_tick_cong_mean"]; ptcb = B["per_tick_cong_mean"]
    step = max(1, len(ptc)//50)
    xs = list(range(0, len(ptc), step))
    ax2.plot(xs, ptc[::step], color=GREEN, lw=1.5, label="Adaptive")
    ax2.plot(xs, ptcb[::step], color=AMBER, lw=1.5, ls="--", label="Baseline")
    ax2.fill_between(xs, ptc[::step], alpha=0.12, color=GREEN)
    ax2.set_ylim(0, 1); ax2.set_xlabel("Tick", color=MUTED, fontsize=8)
    ax2.set_ylabel("Avg Congestion", color=MUTED, fontsize=8)
    ax2.set_title("Network Congestion over Time", color=TEXT)
    ax2.legend(fontsize=8, facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT)

    ax3 = fig.add_subplot(gs[1, 0]); style_ax(ax3)
    cats = ["P95 Emerg", "P99 Emerg"]; x3 = np.arange(2)
    ax3.bar(x3-0.2, [A["p95_emerg_mean"], A["p99_emerg_mean"]], 0.38, color=GREEN, alpha=0.9,
            yerr=[A["p95_emerg_std"], A["p99_emerg_std"]], capsize=4,
            error_kw={"ecolor":TEXT,"elinewidth":1}, label="Adaptive")
    ax3.bar(x3+0.2, [B["p95_emerg_mean"], B["p99_emerg_mean"]], 0.38, color=AMBER, alpha=0.9,
            yerr=[B["p95_emerg_std"], B["p99_emerg_std"]], capsize=4,
            error_kw={"ecolor":TEXT,"elinewidth":1}, label="Baseline")
    ax3.set_xticks(x3); ax3.set_xticklabels(cats, fontsize=8)
    ax3.set_ylabel("Latency (ms)", color=MUTED, fontsize=8)
    ax3.set_title("Tail Latency (P95 / P99)", color=TEXT)
    ax3.legend(fontsize=8, facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT)

    ax4 = fig.add_subplot(gs[1, 1]); style_ax(ax4)
    metrics = ["PDR (%)", "Loss Rate (%)", "Throughput\n(/100 pkts/s)"]
    a_vals = [A["pdr_mean"], A["loss_rate_mean"], A["throughput_mean"]/100]
    b_vals = [B["pdr_mean"], B["loss_rate_mean"], B["throughput_mean"]/100]
    a_errs = [A["pdr_std"],  A["loss_rate_std"],  A["throughput_std"]/100]
    b_errs = [B["pdr_std"],  B["loss_rate_std"],  B["throughput_std"]/100]
    x4 = np.arange(3)
    ax4.bar(x4-0.2, a_vals, 0.38, color=GREEN, alpha=0.9, yerr=a_errs, capsize=4,
            error_kw={"ecolor":TEXT,"elinewidth":1}, label="Adaptive")
    ax4.bar(x4+0.2, b_vals, 0.38, color=AMBER, alpha=0.9, yerr=b_errs, capsize=4,
            error_kw={"ecolor":TEXT,"elinewidth":1}, label="Baseline")
    ax4.set_xticks(x4); ax4.set_xticklabels(metrics, fontsize=7.5)
    ax4.set_title("PDR / Loss Rate / Throughput", color=TEXT)
    ax4.legend(fontsize=8, facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def page_classifier(pdf, cls):
    fig = dark_fig(11, 8.5); fig.patch.set_facecolor(BG)
    fig.suptitle("Semantic Intent Classifier — Evaluation Results", color=TEXT, fontsize=13, y=0.97)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4,
                           left=0.08, right=0.95, top=0.91, bottom=0.07)
    v1 = cls.get("v1", {}); v2 = cls.get("v2", {})

    ax1 = fig.add_subplot(gs[0, 0]); style_ax(ax1)
    bars = ax1.bar(["V1\nOriginal", "V2\nEnhanced"],
                   [v1.get("accuracy",0), v2.get("accuracy",0)],
                   color=[AMBER, GREEN], alpha=0.9, width=0.5)
    ax1.set_ylim(0, 105); ax1.set_ylabel("Accuracy (%)", color=MUTED, fontsize=8)
    ax1.set_title("Classifier Accuracy Comparison", color=TEXT)
    for bar, val in zip(bars, [v1.get("accuracy",0), v2.get("accuracy",0)]):
        ax1.text(bar.get_x()+bar.get_width()/2, val+1.5, f"{val:.1f}%",
                 ha="center", color=TEXT, fontsize=11, fontweight="bold")
    diff = v2.get("accuracy",0) - v1.get("accuracy",0)
    ax1.text(0.5, 0.12, f"Improvement: +{diff:.1f} percentage points",
             transform=ax1.transAxes, ha="center", fontsize=9, color=CYAN, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1]); style_ax(ax2)
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    v1_vals = [v1.get(k.lower(),0) for k in metrics]
    v2_vals = [v2.get(k.lower(),0) for k in metrics]
    x = np.arange(len(metrics))
    ax2.bar(x-0.2, v1_vals, 0.38, color=AMBER, alpha=0.9, label="V1 Original")
    ax2.bar(x+0.2, v2_vals, 0.38, color=GREEN,  alpha=0.9, label="V2 Enhanced")
    ax2.set_xticks(x); ax2.set_xticklabels(metrics, fontsize=8)
    ax2.set_ylim(0, 110); ax2.set_ylabel("Score (%)", color=MUTED, fontsize=8)
    ax2.set_title("All Classifier Metrics", color=TEXT)
    ax2.legend(fontsize=8, facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT)

    ax3 = fig.add_subplot(gs[1, 0]); style_ax(ax3)
    ax3.grid(False)
    conf1 = v1.get("confusion", [[0]*3]*3)
    ax3.imshow(conf1, cmap="Blues", aspect="auto", vmin=0)
    lbls = v1.get("class_labels", ["N","M","H"])
    ax3.set_xticks([0,1,2]); ax3.set_xticklabels(lbls, fontsize=9)
    ax3.set_yticks([0,1,2]); ax3.set_yticklabels(lbls, fontsize=9)
    ax3.set_xlabel("Predicted", color=MUTED, fontsize=8)
    ax3.set_ylabel("True Label", color=MUTED, fontsize=8)
    ax3.set_title("V1 Confusion Matrix", color=TEXT)
    for r in range(3):
        for c in range(3):
            ax3.text(c, r, str(conf1[r][c]), ha="center", va="center",
                     color="white", fontsize=11, fontweight="bold")

    ax4 = fig.add_subplot(gs[1, 1]); style_ax(ax4)
    ax4.grid(False)
    conf2 = v2.get("confusion", [[0]*3]*3)
    ax4.imshow(conf2, cmap="YlOrRd", aspect="auto", vmin=0)
    ax4.set_xticks([0,1,2]); ax4.set_xticklabels(lbls, fontsize=9)
    ax4.set_yticks([0,1,2]); ax4.set_yticklabels(lbls, fontsize=9)
    ax4.set_xlabel("Predicted", color=MUTED, fontsize=8)
    ax4.set_ylabel("True Label", color=MUTED, fontsize=8)
    ax4.set_title("V2 (Enhanced) Confusion Matrix", color=TEXT)
    for r in range(3):
        for c in range(3):
            ax4.text(c, r, str(conf2[r][c]), ha="center", va="center",
                     color="white", fontsize=11, fontweight="bold")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def page_summary(pdf, A, B, cls, params):
    fig = dark_fig(11, 8.5); fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111); ax.axis("off"); ax.set_facecolor(BG)
    ax.text(0.5, 0.96, "Final Summary & Research Notes", transform=ax.transAxes,
            ha="center", fontsize=14, color=TEXT, fontweight="bold")

    v1_acc = cls.get("v1",{}).get("accuracy",0)
    v2_acc = cls.get("v2",{}).get("accuracy",0)
    summary = [
        ["Metric", "Baseline", "Adaptive", "Improvement"],
        ["Packet Delivery Ratio (%)",
         f"{B['pdr_mean']:.1f} ± {B['pdr_std']:.1f}",
         f"{A['pdr_mean']:.1f} ± {A['pdr_std']:.1f}",
         f"+{pct(A['pdr_mean'],B['pdr_mean'])}%" if pct(A['pdr_mean'],B['pdr_mean'])!='—' else '—'],
        ["Emergency Latency (ms)",
         f"{B['emerg_lat_mean']:.1f} ± {B['emerg_lat_std']:.1f}",
         f"{A['emerg_lat_mean']:.1f} ± {A['emerg_lat_std']:.1f}",
         f"{pct(A['emerg_lat_mean'],B['emerg_lat_mean'])}%"],
        ["Packet Loss Rate (%)",
         f"{B['loss_rate_mean']:.2f} ± {B['loss_rate_std']:.2f}",
         f"{A['loss_rate_mean']:.2f} ± {A['loss_rate_std']:.2f}",
         f"{pct(A['loss_rate_mean'],B['loss_rate_mean'])}%"],
        ["P95 Emerg Latency (ms)",
         f"{B['p95_emerg_mean']:.1f} ± {B['p95_emerg_std']:.1f}",
         f"{A['p95_emerg_mean']:.1f} ± {A['p95_emerg_std']:.1f}",
         f"{pct(A['p95_emerg_mean'],B['p95_emerg_mean'])}%"],
        ["Throughput (pkts/s)",
         f"{B['throughput_mean']:.1f} ± {B['throughput_std']:.1f}",
         f"{A['throughput_mean']:.1f} ± {A['throughput_std']:.1f}",
         f"+{pct(A['throughput_mean'],B['throughput_mean'])}%" if pct(A['throughput_mean'],B['throughput_mean'])!='—' else '—'],
        ["Semantic Accuracy",
         f"{v1_acc:.1f}% (V1 Original)",
         f"{v2_acc:.1f}% (V2 Enhanced)",
         f"+{v2_acc-v1_acc:.1f} pts"],
    ]

    col_x = [0.03, 0.35, 0.57, 0.79]; row_h = 0.063; top_y = 0.88
    col_labels = ["Metric", "Baseline", "Adaptive (Intent-Aware)", "Improvement"]
    for ri, row in enumerate(summary):
        y = top_y - ri * row_h
        bg = BG2 if ri == 0 else (BG if ri % 2 == 0 else BG2)
        rect = FancyBboxPatch((0.01, y-row_h*0.88), 0.98, row_h*0.9,
                               boxstyle="round,pad=0.003", facecolor=bg,
                               edgecolor=BORDER, linewidth=0.5,
                               transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        for ci, cell in enumerate(row):
            col = TEXT
            if ri > 0:
                if ci == 1: col = AMBER
                elif ci == 2: col = GREEN
                elif ci == 3:
                    try:
                        v = float(str(cell).replace('%','').replace('+',''))
                        positive_better = (ri in [1, 5, 6])
                        col = GREEN if (v > 0) == positive_better else RED
                    except: col = CYAN
            fw = "bold" if ri == 0 else "normal"
            ax.text(col_x[ci]+0.12, y - row_h*0.38, str(cell),
                    transform=ax.transAxes, fontsize=9, color=col,
                    ha="center", va="center", fontweight=fw)

    ax.axhline(y=0.41, xmin=0.02, xmax=0.98, color=BORDER, linewidth=0.8)
    ax.text(0.02, 0.38, "RESEARCH NOTES", transform=ax.transAxes,
            fontsize=9, color=MUTED, fontweight="bold")
    notes = [
        f"[SETUP]    Scenario: {params['scenario']} · {params['n']} nodes · {params['conn']}-connectivity · "
        f"{params['ticks']} ticks/run · {params['num_runs']} runs · seed={params['seed']}",
        f"[ROUTING]  Adaptive PDR {A['pdr_mean']:.1f}% vs Baseline {B['pdr_mean']:.1f}% "
        f"(Δ {pct(A['pdr_mean'],B['pdr_mean'])}%). "
        f"Emerg latency: {A['emerg_lat_mean']:.1f}ms vs {B['emerg_lat_mean']:.1f}ms.",
        f"[TAIL LAT] P95 emerg: Adaptive {A['p95_emerg_mean']:.1f}ms vs Baseline {B['p95_emerg_mean']:.1f}ms. "
        f"P99: {A['p99_emerg_mean']:.1f}ms vs {B['p99_emerg_mean']:.1f}ms.",
        f"[LOSS]     Packet loss rate: Adaptive {A['loss_rate_mean']:.2f}% vs Baseline {B['loss_rate_mean']:.2f}% "
        f"(Δ {pct(A['loss_rate_mean'],B['loss_rate_mean'])}%).",
        f"[SEMANTIC] V1 classifier accuracy: {v1_acc:.1f}% → V2 enhanced: {v2_acc:.1f}% "
        f"(+{v2_acc-v1_acc:.1f} pts via synonym/phrase expansion).",
        "[MODEL]    Cost(n) = 1.2·C(n)·P_emerg + 0.9·Q(n) + 0.6·D(n) + 0.7·L(n) − 0.8·Progress(n); "
        "emergency multiplier = 2.5×.",
        "[6G REF]   URLLC target: <1ms physical layer; semantic-layer target <50ms for AI agent coordination.",
        "[BASELINE] Greedy shortest-path: progress-only, no congestion awareness, no semantic QoS weighting.",
    ]
    for i, note in enumerate(notes):
        ax.text(0.02, 0.34 - i*0.042, note, transform=ax.transAxes,
                fontsize=7.8, color=MUTED if i >= 5 else TEXT)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run_simulation(cfg):
    adaptive_runs, baseline_runs = [], []
    print(f"  Running {cfg['num_runs']} simulation runs (seed={cfg['seed']})...")
    for i in range(cfg["num_runs"]):
        random.seed(cfg["seed"] + i)
        np.random.seed(cfg["seed"] + i)
        topo = build_topo(cfg["n"], cfg["conn"])
        traffic = generate_traffic_sequence(
            cfg["ticks"], cfg["rate"], cfg["emerg_frac"], cfg["scenario"], cfg["n"])
        adaptive_runs.append(sim_run(topo, traffic, "adaptive", cfg["proc_cap"]))
        baseline_runs.append(sim_run(topo, traffic, "baseline", cfg["proc_cap"]))
        if (i+1) % 5 == 0:
            print(f"  ... {i+1}/{cfg['num_runs']} done")
    print("  Evaluating semantic classifier...")
    cls = eval_classifier()
    return aggregate_stats(adaptive_runs), aggregate_stats(baseline_runs), cls

def main():
    print("=" * 60)
    print("  6G Intent-Aware Routing — Report Generator")
    print("=" * 60)
    t0 = time.time()

    A, B, cls = run_simulation(CFG)

    print(f"  Generating PDF report -> {OUT_FILE}")
    with PdfPages(OUT_FILE) as pdf:
        info = pdf.infodict()
        info["Title"]   = "6G Intent-Aware Routing — Simulation Report"
        info["Author"]  = "Semantic Edge Simulator v3"
        info["Subject"] = "6G Network Simulation Results"
        info["Keywords"]= "6G, routing, semantic, QoS, URLLC"

        print("    Page 1/5: Title & overview")
        page_title(pdf, A, B, cls, CFG)
        print("    Page 2/5: Metrics table")
        page_metrics_table(pdf, A, B, cls, CFG)
        print("    Page 3/5: Routing charts")
        page_routing_charts(pdf, A, B, CFG)
        print("    Page 4/5: Classifier evaluation")
        page_classifier(pdf, cls)
        print("    Page 5/5: Summary & research notes")
        page_summary(pdf, A, B, cls, CFG)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Report saved: {OUT_FILE}")
    print()
    print("  KEY RESULTS")
    print(f"  PDR      — Adaptive: {A['pdr_mean']:.1f} ± {A['pdr_std']:.1f}%   "
          f"Baseline: {B['pdr_mean']:.1f} ± {B['pdr_std']:.1f}%")
    print(f"  Emerg lat— Adaptive: {A['emerg_lat_mean']:.1f} ± {A['emerg_lat_std']:.1f} ms   "
          f"Baseline: {B['emerg_lat_mean']:.1f} ± {B['emerg_lat_std']:.1f} ms")
    print(f"  Loss rate— Adaptive: {A['loss_rate_mean']:.2f}%   Baseline: {B['loss_rate_mean']:.2f}%")
    print(f"  Sem.Acc  — V1: {cls.get('v1',{}).get('accuracy',0):.1f}%   "
          f"V2: {cls.get('v2',{}).get('accuracy',0):.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
