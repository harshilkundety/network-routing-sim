
import sys
import math
import random
import json
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QTextEdit, QScrollArea, QFrame,
    QGridLayout, QSizePolicy, QDialog, QGroupBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QButtonGroup, QAbstractButton
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

# ─────────────────────────────────────────────
# COLOUR PALETTE (mirrors CSS variables)
# ─────────────────────────────────────────────
BG    = "#0d1117"
BG2   = "#161b22"
BG3   = "#1c2330"
BORDER= "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"
RED   = "#f85149"
GREEN = "#3fb950"
AMBER = "#d29922"
BLUE  = "#58a6ff"
PURPLE= "#a371f7"
CYAN  = "#00d2ff"

def qss_base():
    return f"""
    QWidget {{ background:{BG}; color:{TEXT}; font-family:'Segoe UI',sans-serif; font-size:12px; }}
    QFrame  {{ border:none; }}
    QScrollArea {{ border:none; }}
    QPushButton {{
        background:{BG3}; border:1px solid {BORDER}; color:{TEXT};
        padding:6px 14px; border-radius:6px; font-size:12px;
    }}
    QPushButton:hover {{ border-color:{BLUE}; color:{BLUE}; }}
    QPushButton:disabled {{ opacity:0.4; color:{MUTED}; }}
    QSlider::groove:horizontal {{ height:4px; background:{BG3}; border-radius:2px; }}
    QSlider::handle:horizontal {{
        background:{BLUE}; width:14px; height:14px;
        border-radius:7px; margin:-5px 0;
    }}
    QSlider::sub-page:horizontal {{ background:{BLUE}; border-radius:2px; }}
    QTextEdit {{
        background:{BG3}; border:1px solid {BORDER}; color:{TEXT};
        border-radius:5px; padding:4px; font-size:12px;
    }}
    QTextEdit:focus {{ border-color:{BLUE}; }}
    QScrollBar:vertical {{ width:6px; background:{BG}; }}
    QScrollBar::handle:vertical {{ background:{BORDER}; border-radius:3px; }}
    QLabel {{ background:transparent; }}
    QTableWidget {{
        background:{BG2}; border:1px solid {BORDER}; color:{TEXT};
        gridline-color:{BORDER}; font-size:12px;
    }}
    QHeaderView::section {{
        background:{BG3}; color:{MUTED}; border:none;
        border-bottom:1px solid {BORDER}; padding:6px;
        font-size:11px; font-weight:bold;
    }}
    QProgressBar {{
        background:{BG3}; border:none; border-radius:3px; height:6px;
    }}
    QProgressBar::chunk {{ background:{BLUE}; border-radius:3px; }}
    """

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0

def std(lst):
    if len(lst) < 2: return 0.0
    m = avg(lst)
    return math.sqrt(sum((x - m)**2 for x in lst) / len(lst))

def pct(a, b):
    if b == 0: return "—"
    return f"{(a - b) / b * 100:.1f}"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def rand(a, b):
    return a + random.random() * (b - a)

def rand_int(a, b):
    return random.randint(a, b)

# ─────────────────────────────────────────────
# SEMANTIC CLASSIFIER
# ─────────────────────────────────────────────
KW = {
    "mayday": 9, "sos": 9, "tornado": 9, "cyclone": 9, "tsunami": 9, "hurricane": 9,
    "fire": 8, "emergency": 8, "explosion": 8, "disaster": 8,
    "wildfire": 8, "flood": 8, "avalanche": 8, "earthquake": 8, "storm": 8,
    "bomb": 7, "attack": 7, "collapse": 7, "blizzard": 7, "eruption": 7,
    "nuclear": 7, "chemical": 7, "evacuation": 7, "trapped": 7, "casualty": 7,
    "drowning": 7, "outage": 7,
    "critical": 6, "ambulance": 6, "rescue": 6, "injured": 6, "hostage": 6,
    "crash": 5, "accident": 5, "interference": 5, "signal_loss": 5,
    "urgent": 4, "warning": 4, "alert": 4, "danger": 4, "hazard": 4,
    "help": 3, "degraded": 3,
    "problem": 2, "issue": 2, "priority": 1, "high": 1,
    "hello": 0, "hi": 0, "test": 0, "ok": 0, "status": 0,
    "update": 0, "ping": 0, "good": 0, "normal": 0,
}

def classify(text):
    import re
    words = re.findall(r'\w+', (text or "").lower())
    score = 0.0
    matched = []
    for i, w in enumerate(words):
        v = KW.get(w)
        if v is not None and v > 0:
            score += v * max(0.5, 1 - i * 0.04)
            matched.append(f"{w}:{v}")
    score = round(score * 10) / 10
    if score >= 8:
        priority, level = "HIGH", 3
    elif score >= 3:
        priority, level = "MEDIUM", 2
    else:
        priority, level = "NORMAL", 1
    return {"priority": priority, "level": level, "score": score, "matched": matched}

# ─────────────────────────────────────────────
# ENHANCED CLASSIFIER: SYNONYMS + PHRASES
# ─────────────────────────────────────────────
SYNONYMS = {
    "burning": "fire", "blaze": "fire", "ablaze": "fire", "inferno": "fire", "conflagration": "fire",
    "medical": "ambulance", "injury": "injured", "injuries": "injured", "wounded": "injured", "hurt": "injured",
    "stranded": "trapped", "missing": "rescue",
    "blackout": "outage",
    "collision": "crash", "wreck": "crash",
    "toxic": "chemical", "radiation": "nuclear",
    "sinking": "drowning", "submerging": "drowning",
    "imminent": "warning",
    "severe": "critical",
}

PHRASES = [
    ("building is burning",    "fire"),
    ("building on fire",       "fire"),
    ("mass casualty",          "casualty"),
    ("structural collapse",    "collapse"),
    ("gas leak",               "chemical"),
    ("active shooter",         "attack"),
    ("search and rescue",      "rescue"),
    ("evacuation order",       "evacuation"),
    ("missing person",         "rescue"),
    ("water rising",           "flood"),
    ("power outage",           "outage"),
    ("wind damage",            "storm"),
]

CLASSIFIER_DATASET = [
    ("fire emergency sector 7", "HIGH"),
    ("mayday engine failure", "HIGH"),
    ("flood warning critical zone", "HIGH"),
    ("sos rescue needed", "HIGH"),
    ("earthquake collapse warning", "HIGH"),
    ("tornado alert evacuate now", "HIGH"),
    ("hurricane force winds incoming", "HIGH"),
    ("tsunami warning coastal region", "HIGH"),
    ("wildfire advancing evacuation", "HIGH"),
    ("storm surge disaster", "HIGH"),
    ("avalanche emergency rescue", "HIGH"),
    ("blizzard evacuation critical", "HIGH"),
    ("emergency call dispatch unit", "HIGH"),
    ("chemical outage hazard zone", "HIGH"),
    ("nuclear plant explosion", "HIGH"),
    ("bomb threat near facility", "HIGH"),
    ("casualty report incoming attack", "HIGH"),
    ("wildfire rescue trapped", "HIGH"),
    ("hostage situation explosion", "HIGH"),
    ("drowning victim river flood", "HIGH"),
    ("eruption volcanic emergency", "HIGH"),
    ("cyclone approaching landfall", "HIGH"),
    ("nuclear explosion detected", "HIGH"),
    ("disaster relief rescue operation", "HIGH"),
    ("attack on infrastructure bomb", "HIGH"),
    ("tsunami coastal collapse", "HIGH"),
    ("chemical plant fire explosion", "HIGH"),
    ("wildfire evacuation critical status", "HIGH"),
    ("mayday distress signal", "HIGH"),
    ("sos flood emergency", "HIGH"),
    ("earthquake rescue casualty", "HIGH"),
    ("hurricane evacuation critical", "HIGH"),
    ("storm emergency rescue", "HIGH"),
    ("fire collapse rescue team", "HIGH"),
    ("bomb explosion attack site", "HIGH"),
    ("trapped miners rescue emergency", "HIGH"),
    ("flood casualty rescue", "HIGH"),
    ("blizzard critical rescue", "HIGH"),
    ("nuclear fallout evacuation", "HIGH"),
    ("drowning casualty flood", "HIGH"),
    ("collapse building rescue", "HIGH"),
    ("chemical attack zone", "HIGH"),
    ("eruption lava emergency", "HIGH"),
    ("avalanche rescue critical", "HIGH"),
    ("disaster earthquake collapse", "HIGH"),
    ("building is burning", "HIGH"),
    ("warehouse blaze spreading fast", "HIGH"),
    ("structure ablaze with heavy smoke", "HIGH"),
    ("critical injury multiple victims", "HIGH"),
    ("severe injuries at crash site", "HIGH"),
    ("multiple wounded soldiers need rescue", "HIGH"),
    ("vessel sinking crew needs help", "HIGH"),
    ("radiation leak causing casualties", "HIGH"),
    ("toxic spill near residential casualties", "HIGH"),
    ("blackout causing widespread casualties", "HIGH"),
    ("people hurt in collision", "HIGH"),
    ("mass casualty event from structural collapse", "HIGH"),
    ("evacuation order issued stranded residents", "HIGH"),
    ("inferno consuming the district", "HIGH"),
    ("severe injury needs immediate care", "HIGH"),
    ("collision involving multiple vehicles casualties", "HIGH"),
    ("sinking ship stranded passengers", "HIGH"),
    ("gas leak near evacuation zone", "HIGH"),
    ("missing crew members on sinking vessel", "HIGH"),
    ("building ablaze rescue requested", "HIGH"),
    ("workers hurt in collision evacuation", "HIGH"),
    ("toxic spill stranded workers", "HIGH"),
    ("radiation leak evacuation order", "HIGH"),
    ("inferno spreading trapped residents", "HIGH"),
    ("blaze at warehouse injured workers", "HIGH"),
    ("urgent sensor offline maintenance", "MEDIUM"),
    ("ambulance requested downtown", "MEDIUM"),
    ("hazard detected area b", "MEDIUM"),
    ("help needed at checkpoint", "MEDIUM"),
    ("interference detected rerouting", "MEDIUM"),
    ("signal degraded alert", "MEDIUM"),
    ("network outage partial restore", "MEDIUM"),
    ("warning system activated", "MEDIUM"),
    ("danger zone proximity", "MEDIUM"),
    ("crash report road a12", "MEDIUM"),
    ("accident at junction 7", "MEDIUM"),
    ("rescue team on standby", "MEDIUM"),
    ("ambulance en route to scene", "MEDIUM"),
    ("urgent repair needed infrastructure", "MEDIUM"),
    ("alert level elevated sector 5", "MEDIUM"),
    ("hazard posted on road", "MEDIUM"),
    ("signal interference reported", "MEDIUM"),
    ("system degraded performance", "MEDIUM"),
    ("outage affecting sector 3", "MEDIUM"),
    ("help required at station 4", "MEDIUM"),
    ("rescue personnel deployed", "MEDIUM"),
    ("crash barriers activated", "MEDIUM"),
    ("injured worker at site", "MEDIUM"),
    ("critical infrastructure check", "MEDIUM"),
    ("accident involving two vehicles", "MEDIUM"),
    ("interference on channel 7", "MEDIUM"),
    ("urgent maintenance required", "MEDIUM"),
    ("rescue drone deployed", "MEDIUM"),
    ("alert issued for area", "MEDIUM"),
    ("sensor interference reported", "MEDIUM"),
    ("warning signal activated sector", "MEDIUM"),
    ("danger of overheating detected", "MEDIUM"),
    ("ambulance on standby", "MEDIUM"),
    ("help desk offline temporarily", "MEDIUM"),
    ("crash detection system online", "MEDIUM"),
    ("injured persons reported", "MEDIUM"),
    ("outage warning zone five", "MEDIUM"),
    ("critical system check pending", "MEDIUM"),
    ("accident reconstruction team", "MEDIUM"),
    ("signal degraded rerouting", "MEDIUM"),
    ("help coordinate response", "MEDIUM"),
    ("alert raised by sensor", "MEDIUM"),
    ("hazard assessment ongoing", "MEDIUM"),
    ("interference mitigation applied", "MEDIUM"),
    ("rescue helicopter on patrol", "MEDIUM"),
    ("urgent network maintenance", "MEDIUM"),
    ("warning issued for weather", "MEDIUM"),
    ("crash site investigation", "MEDIUM"),
    ("dangerous goods vehicle alert", "MEDIUM"),
    ("injured worker evacuated", "MEDIUM"),
    ("critical check required", "MEDIUM"),
    ("accident investigation underway", "MEDIUM"),
    ("danger alert at main gate", "MEDIUM"),
    ("hazard warning flashing", "MEDIUM"),
    ("urgent warning from sensor", "MEDIUM"),
    ("critical alert raised", "MEDIUM"),
    ("critical interference detected", "MEDIUM"),
    ("blackout reported in district", "MEDIUM"),
    ("collision on motorway", "MEDIUM"),
    ("vehicle wreck reported", "MEDIUM"),
    ("severe weather advisory", "MEDIUM"),
    ("imminent weather concern", "MEDIUM"),
    ("workers missing from site", "MEDIUM"),
    ("stranded vehicle on highway", "MEDIUM"),
    ("medical team required", "MEDIUM"),
    ("persons hurt in fall", "MEDIUM"),
    ("power blackout district 4", "MEDIUM"),
    ("two vehicles collision reported", "MEDIUM"),
    ("imminent system failure concern", "MEDIUM"),
    ("severe outage risk zone", "MEDIUM"),
    ("status update nominal", "NORMAL"),
    ("hello network ping", "NORMAL"),
    ("ok test signal", "NORMAL"),
    ("diagnostic routine complete", "NORMAL"),
    ("voice call setup complete", "NORMAL"),
    ("bandwidth allocation request", "NORMAL"),
    ("system status normal", "NORMAL"),
    ("network check passed", "NORMAL"),
    ("ping response received", "NORMAL"),
    ("update installed successfully", "NORMAL"),
    ("connection established", "NORMAL"),
    ("data transfer complete", "NORMAL"),
    ("node health check ok", "NORMAL"),
    ("routine maintenance scheduled", "NORMAL"),
    ("configuration update applied", "NORMAL"),
    ("test signal strong", "NORMAL"),
    ("firmware update complete", "NORMAL"),
    ("system reboot initiated", "NORMAL"),
    ("network latency low", "NORMAL"),
    ("hello world test", "NORMAL"),
    ("bandwidth utilization 42 percent", "NORMAL"),
    ("good signal quality detected", "NORMAL"),
    ("node registration complete", "NORMAL"),
    ("uptime report generated", "NORMAL"),
    ("voice quality excellent", "NORMAL"),
    ("packet delivery confirmed", "NORMAL"),
    ("normal operations resumed", "NORMAL"),
    ("system log archived", "NORMAL"),
    ("frequency scan complete", "NORMAL"),
    ("protocol handshake ok", "NORMAL"),
    ("backup connection established", "NORMAL"),
    ("node synchronization done", "NORMAL"),
    ("software version updated", "NORMAL"),
    ("latency within normal range", "NORMAL"),
    ("calibration complete sensor", "NORMAL"),
    ("satellite link nominal", "NORMAL"),
    ("test transmission received", "NORMAL"),
    ("status ok all systems", "NORMAL"),
    ("routine data collection", "NORMAL"),
    ("communication channel clear", "NORMAL"),
    ("system performance good", "NORMAL"),
    ("ping latency low", "NORMAL"),
    ("hello from node five", "NORMAL"),
    ("update server complete", "NORMAL"),
    ("test mode disabled", "NORMAL"),
    ("user authentication successful", "NORMAL"),
    ("backup process finished", "NORMAL"),
    ("log file rotated", "NORMAL"),
    ("memory usage normal", "NORMAL"),
    ("cpu load acceptable", "NORMAL"),
    ("dns resolution ok", "NORMAL"),
    ("load balancer active", "NORMAL"),
    ("ssl certificate valid", "NORMAL"),
    ("backup sync complete", "NORMAL"),
    ("monitoring dashboard updated", "NORMAL"),
    ("disk space adequate", "NORMAL"),
    ("network throughput normal", "NORMAL"),
    ("system uptime satisfactory", "NORMAL"),
    ("help me find this file", "NORMAL"),
    ("please help with network setup", "NORMAL"),
    ("urgent schedule meeting tomorrow", "NORMAL"),
    ("please help configure system", "NORMAL"),
    ("help resolve the issue", "NORMAL"),
    ("warning light is normal behavior", "NORMAL"),
    ("crash course in networking", "NORMAL"),
    ("alert me when done", "NORMAL"),
    ("help desk ticket 1234", "NORMAL"),
    ("rescue helicopter sound file", "NORMAL"),
    ("accident report template download", "NORMAL"),
    ("urgent team standup now", "NORMAL"),
]

def classify_v2(text):
    import re
    t = (text or "").lower()
    extra = []
    for phrase, kw in PHRASES:
        if phrase in t:
            extra.append(kw)
    words = re.findall(r'\w+', t)
    words = extra + [SYNONYMS.get(w, w) for w in words]
    score = 0.0
    matched = []
    for i, w in enumerate(words):
        v = KW.get(w)
        if v is not None and v > 0:
            score += v * max(0.5, 1 - i * 0.04)
            matched.append(f"{w}:{v}")
    score = round(score * 10) / 10
    if score >= 8:
        priority, level = "HIGH", 3
    elif score >= 3:
        priority, level = "MEDIUM", 2
    else:
        priority, level = "NORMAL", 1
    return {"priority": priority, "level": level, "score": score, "matched": matched}

def eval_classifier():
    labels = ["NORMAL", "MEDIUM", "HIGH"]
    lbl_idx = {l: i for i, l in enumerate(labels)}
    results = {}
    for tag, fn in [("v1", classify), ("v2", classify_v2)]:
        conf = [[0]*3 for _ in range(3)]
        for text, true_lbl in CLASSIFIER_DATASET:
            pred = fn(text)["priority"]
            conf[lbl_idx[true_lbl]][lbl_idx[pred]] += 1
        total = len(CLASSIFIER_DATASET)
        correct = sum(conf[i][i] for i in range(3))
        accuracy = correct / total * 100
        precision_list, recall_list, f1_list = [], [], []
        for i in range(3):
            tp = conf[i][i]
            fp = sum(conf[r][i] for r in range(3)) - tp
            fn_ = sum(conf[i][c] for c in range(3)) - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn_) if (tp + fn_) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precision_list.append(p); recall_list.append(r); f1_list.append(f)
        results[tag] = {
            "accuracy":  round(accuracy, 1),
            "precision": round(sum(precision_list) / 3 * 100, 1),
            "recall":    round(sum(recall_list)    / 3 * 100, 1),
            "f1":        round(sum(f1_list)        / 3 * 100, 1),
            "confusion": conf,
            "class_labels": labels,
        }
    return results

# ─────────────────────────────────────────────
# TOPOLOGY
# ─────────────────────────────────────────────
def build_topo(n, conn):
    W, H = 300, 200
    cx, cy = W / 2, H / 2
    nodes = []
    for i in range(n):
        a = (2 * math.pi * i / n) + rand(-0.3, 0.3)
        r = rand(0.22, 0.42) * min(W, H)
        x = clamp(cx + r * math.cos(a), 20, W - 20)
        y = clamp(cy + r * math.sin(a), 20, H - 20)
        nodes.append({"id": i, "x": x, "y": y, "neighbors": []})

    edges = []
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        dists = sorted(
            [{"j": j, "d": math.hypot(nodes[j]["x"] - nodes[i]["x"],
                                       nodes[j]["y"] - nodes[i]["y"])}
             for j in range(n) if j != i],
            key=lambda x: x["d"]
        )
        for k in range(min(conn, len(dists))):
            j = dists[k]["j"]
            if not adj[i][j]:
                adj[i][j] = adj[j][i] = True
                edges.append({"a": i, "b": j})
                nodes[i]["neighbors"].append(j)
                nodes[j]["neighbors"].append(i)

    for nd in nodes:
        if not nd["neighbors"]:
            j = (nd["id"] + 1) % n
            if not adj[nd["id"]][j]:
                adj[nd["id"]][j] = adj[j][nd["id"]] = True
                edges.append({"a": nd["id"], "b": j})
                nd["neighbors"].append(j)
                nodes[j]["neighbors"].append(nd["id"])

    return {"nodes": nodes, "edges": edges}

def bfs_path(topo, src, dst):
    n = len(topo["nodes"])
    if n == 0 or src == dst: return None
    vis = [False] * n
    par = [-1] * n
    q = deque([src])
    vis[src] = True
    while q:
        cur = q.popleft()
        if cur == dst: break
        for nb in topo["nodes"][cur]["neighbors"]:
            if not vis[nb]:
                vis[nb] = True
                par[nb] = cur
                q.append(nb)
    if not vis[dst]: return None
    path = []
    c = dst
    while c != -1:
        path.insert(0, c)
        c = par[c]
    return path

# ─────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────
TRAFFIC_MSGS = [
    {"text": "fire emergency sector 7", "level": 3},
    {"text": "mayday engine failure", "level": 3},
    {"text": "flood warning critical zone", "level": 3},
    {"text": "sos rescue needed", "level": 3},
    {"text": "earthquake collapse warning", "level": 3},
    {"text": "tornado alert evacuate now", "level": 3},
    {"text": "hurricane force winds incoming", "level": 3},
    {"text": "tsunami warning coastal region", "level": 3},
    {"text": "wildfire advancing evacuate", "level": 3},
    {"text": "storm surge imminent disaster", "level": 3},
    {"text": "avalanche emergency rescue", "level": 3},
    {"text": "blizzard evacuation critical", "level": 3},
    {"text": "emergency call dispatch unit", "level": 3},
    {"text": "chemical outage hazard zone", "level": 3},
    {"text": "urgent warning sensor offline", "level": 2},
    {"text": "ambulance requested downtown", "level": 2},
    {"text": "hazard detected area b", "level": 2},
    {"text": "help needed at checkpoint", "level": 2},
    {"text": "interference detected rerouting", "level": 2},
    {"text": "signal degraded alert", "level": 2},
    {"text": "network outage partial restore", "level": 2},
    {"text": "status update nominal", "level": 1},
    {"text": "hello network ping", "level": 1},
    {"text": "ok test signal", "level": 1},
    {"text": "diagnostic routine complete", "level": 1},
    {"text": "voice call setup complete", "level": 1},
    {"text": "bandwidth allocation request", "level": 1},
]

def generate_traffic_sequence(ticks, rate, emerg_frac, scenario, n):
    seq = []
    for t in range(ticks):
        seq.append([])
        inject_rate = rate
        eff_emerg = emerg_frac
        if scenario == "ramp":
            inject_rate = max(1, math.ceil(rate * (t / ticks * 2)))
        elif scenario == "burst":
            inject_rate = rate * 3 if (ticks * 0.3 < t < ticks * 0.6) else rate
        elif scenario == "disaster":
            inject_rate = math.ceil(rate * 2)
            eff_emerg = 65
        elif scenario == "voice":
            inject_rate = math.ceil(rate * 1.4)
            eff_emerg = 15

        for _ in range(inject_rate):
            is_emerg = random.random() < eff_emerg / 100
            if scenario == "burst" and ticks * 0.3 < t < ticks * 0.6:
                level = 3
            elif scenario == "disaster":
                level = 3 if is_emerg else (2 if random.random() < 0.25 else 1)
            else:
                level = 3 if is_emerg else (2 if random.random() < 0.3 else 1)

            src = rand_int(0, n - 1)
            dst = rand_int(0, n - 1)
            tries = 0
            while dst == src and tries < 20:
                dst = rand_int(0, n - 1)
                tries += 1
            if scenario == "hotspot":
                dst = 0
            seq[t].append({"src": src, "dst": dst, "level": level})
    return seq

W_ADAPTIVE = {"cong": 1.2, "queue": 0.9, "delay": 0.6, "loss": 0.7, "prog": 0.8, "emerg_mult": 2.5}
W_BASELINE = {"cong": 0.0, "queue": 0.0, "delay": 0.0, "loss": 0.0, "prog": 1.0, "emerg_mult": 1.0}

def sim_run(topo, traffic_seq, mode, proc_cap):
    N = len(topo["nodes"])
    ticks = len(traffic_seq)
    W = W_ADAPTIVE if mode == "adaptive" else W_BASELINE

    ns = [{
        "id": nd["id"],
        "neighbors": list(nd["neighbors"]),
        "queue": [],
        "max_queue": 10,
        "congestion": 0.0,
        "delay": 0.0,
        "loss_rate": 0.0,
    } for nd in topo["nodes"]]

    stats = {
        "delivered": 0, "dropped": 0, "total": 0,
        "emerg_lat": [], "normal_lat": [], "med_lat": [],
        "per_tick_cong": [], "per_tick_in_flight": [],
        "control_msgs": 0,
    }

    pkt_id = 0
    live = {}

    def cost(to_id, pkt):
        node = ns[to_id]
        dst_pos = topo["nodes"][pkt["dst"]]
        to_pos  = topo["nodes"][to_id]
        dist = math.hypot(to_pos["x"] - dst_pos["x"], to_pos["y"] - dst_pos["y"])
        progress = 1.0 / (1.0 + dist)
        em = W["emerg_mult"] if pkt["level"] == 3 else 1.0
        return (W["cong"]  * node["congestion"] * em
              + W["queue"] * len(node["queue"]) / node["max_queue"]
              + W["delay"] * node["delay"]
              + W["loss"]  * node["loss_rate"]
              - W["prog"]  * progress * 200)

    def next_hop(pkt):
        cur = ns[pkt["cur"]]
        if not cur["neighbors"]: return None
        if pkt["dst"] in cur["neighbors"]: return pkt["dst"]
        best, best_c = None, float("inf")
        prev = pkt["path"][-2] if len(pkt["path"]) >= 2 else -1
        for nid in cur["neighbors"]:
            c = cost(nid, pkt)
            if nid == prev: c += 50
            if c < best_c:
                best_c = c
                best = nid
        return best

    def deliver(pkt, t):
        lat = (t - pkt["born"]) * 10 + pkt["hops"] * 2
        if pkt["level"] == 3: stats["emerg_lat"].append(lat)
        elif pkt["level"] == 2: stats["med_lat"].append(lat)
        else: stats["normal_lat"].append(lat)
        stats["delivered"] += 1
        live.pop(pkt["id"], None)

    def drop(pkt):
        stats["dropped"] += 1
        live.pop(pkt["id"], None)

    for t in range(ticks):
        for pk in traffic_seq[t]:
            nonlocal_id = pkt_id
            pkt_id += 1
            pkt = {
                "id": nonlocal_id, "src": pk["src"], "dst": pk["dst"],
                "level": pk["level"], "cur": pk["src"], "path": [pk["src"]],
                "born": t, "hops": 0, "max_hops": N * 3
            }
            stats["total"] += 1
            if pk["src"] == pk["dst"]:
                deliver(pkt, t)
                continue
            src_ns = ns[pk["src"]]
            if len(src_ns["queue"]) < src_ns["max_queue"]:
                src_ns["queue"].append(pkt)
                live[pkt["id"]] = pkt
            else:
                drop(pkt)

        if t % 5 == 0:
            alpha = 0.5 if mode == "adaptive" else 0.15
            for node in ns:
                for nid in node["neighbors"]:
                    nb = ns[nid]
                    nb["congestion"] = nb["congestion"] * (1 - alpha) + (len(nb["queue"]) / nb["max_queue"]) * alpha
                    nb["delay"] = nb["delay"] * 0.8 + (len(nb["queue"]) * 0.12) * 0.2
                    stats["control_msgs"] += 1

        for node in ns:
            if mode == "adaptive":
                node["queue"].sort(key=lambda p: -p["level"])
            to_forward = node["queue"][:proc_cap]
            node["queue"] = node["queue"][proc_cap:]
            for pkt in to_forward:
                if pkt["id"] not in live: continue
                if pkt["hops"] >= pkt["max_hops"]: drop(pkt); continue
                nh = next_hop(pkt)
                if nh is None: drop(pkt); continue
                pkt["path"].append(nh)
                pkt["hops"] += 1
                pkt["cur"] = nh
                if nh == pkt["dst"]:
                    deliver(pkt, t)
                    continue
                nh_ns = ns[nh]
                if len(nh_ns["queue"]) < nh_ns["max_queue"]:
                    nh_ns["queue"].append(pkt)
                else:
                    if mode == "adaptive" and pkt["level"] == 3:
                        victim_idx = min(range(len(nh_ns["queue"])),
                                         key=lambda i: nh_ns["queue"][i]["level"])
                        if nh_ns["queue"][victim_idx]["level"] < pkt["level"]:
                            drop(nh_ns["queue"][victim_idx])
                            nh_ns["queue"].pop(victim_idx)
                            nh_ns["queue"].append(pkt)
                        else:
                            drop(pkt)
                    else:
                        drop(pkt)
            node["congestion"] = len(node["queue"]) / node["max_queue"]
            node["loss_rate"] = clamp(node["loss_rate"] * 0.93 + (0.07 if node["congestion"] > 0.75 else 0), 0, 1)

        stats["per_tick_cong"].append(avg([n["congestion"] for n in ns]))
        stats["per_tick_in_flight"].append(len(live))

    for _ in live:
        stats["dropped"] += 1
    stats["total"] = stats["delivered"] + stats["dropped"]
    el = stats["emerg_lat"];  nl = stats["normal_lat"]
    stats["p95_emerg"]  = float(np.percentile(el, 95)) if len(el)  >= 5 else 0.0
    stats["p99_emerg"]  = float(np.percentile(el, 99)) if len(el)  >= 5 else 0.0
    stats["p95_normal"] = float(np.percentile(nl, 95)) if len(nl)  >= 5 else 0.0
    stats["p99_normal"] = float(np.percentile(nl, 99)) if len(nl)  >= 5 else 0.0
    stats["throughput"] = stats["delivered"] / (ticks * 0.01) if ticks else 0.0
    stats["loss_rate"]  = stats["dropped"] / stats["total"] * 100 if stats["total"] else 0.0
    stats["pdr"]        = stats["delivered"] / stats["total"] * 100 if stats["total"] else 0.0
    return stats


def aggregate_stats(runs):
    def _ms(key):  return [avg(r[key]) for r in runs]
    def _sc(key):  return [r[key]      for r in runs]
    def _agg(vals):
        m = avg(vals); s = std(vals); return round(m, 2), round(s, 2)
    em = _ms("emerg_lat");  md = _ms("med_lat");   nm = _ms("normal_lat")
    p95e = _sc("p95_emerg"); p99e = _sc("p99_emerg")
    p95n = _sc("p95_normal"); p99n = _sc("p99_normal")
    thr  = _sc("throughput"); lr   = _sc("loss_rate")
    pdr  = _sc("pdr"); ctrl = _sc("control_msgs")
    dl   = _sc("delivered"); dr   = _sc("dropped"); tot = _sc("total")
    max_t = max(len(r["per_tick_cong"]) for r in runs)
    ptc_mean = []
    for ti in range(max_t):
        vals = [r["per_tick_cong"][ti] for r in runs if ti < len(r["per_tick_cong"])]
        ptc_mean.append(avg(vals))
    em_m,  em_s  = _agg(em);   md_m,  md_s  = _agg(md);   nm_m,  nm_s  = _agg(nm)
    p95e_m,p95e_s= _agg(p95e); p99e_m,p99e_s= _agg(p99e)
    p95n_m,p95n_s= _agg(p95n); p99n_m,p99n_s= _agg(p99n)
    thr_m, thr_s = _agg(thr);  lr_m,  lr_s  = _agg(lr)
    pdr_m, pdr_s = _agg(pdr);  ctrl_m,_     = _agg(ctrl)
    dl_m,  _     = _agg(dl);   dr_m,  _     = _agg(dr);   tot_m, _ = _agg(tot)
    return {
        "emerg_lat_mean":   em_m,  "emerg_lat_std":   em_s,
        "med_lat_mean":     md_m,  "med_lat_std":     md_s,
        "normal_lat_mean":  nm_m,  "normal_lat_std":  nm_s,
        "p95_emerg_mean":   p95e_m,"p95_emerg_std":   p95e_s,
        "p99_emerg_mean":   p99e_m,"p99_emerg_std":   p99e_s,
        "p95_normal_mean":  p95n_m,"p95_normal_std":  p95n_s,
        "p99_normal_mean":  p99n_m,"p99_normal_std":  p99n_s,
        "throughput_mean":  thr_m, "throughput_std":  thr_s,
        "loss_rate_mean":   lr_m,  "loss_rate_std":   lr_s,
        "pdr_mean":         pdr_m, "pdr_std":         pdr_s,
        "control_msgs_mean":ctrl_m,
        "delivered_mean":   dl_m,  "dropped_mean":    dr_m,  "total_mean": tot_m,
        "per_tick_cong_mean": ptc_mean,
    }

# ─────────────────────────────────────────────
# MATPLOTLIB CANVAS WIDGET
# ─────────────────────────────────────────────
class MplCanvas(FigureCanvas):
    def __init__(self, fig=None, width=4, height=3, dpi=90):
        if fig is None:
            fig = Figure(figsize=(width, height), dpi=dpi, facecolor=BG2)
        super().__init__(fig)
        self.setStyleSheet(f"background:{BG2};")


# ─────────────────────────────────────────────
# NETWORK TOPOLOGY PREVIEW (sidebar)
# ─────────────────────────────────────────────
class TopoCanvas(MplCanvas):
    def __init__(self):
        fig = Figure(figsize=(3, 2.4), dpi=90, facecolor=BG2)
        super().__init__(fig)
        self.ax = fig.add_subplot(111, facecolor=BG2)
        self.setMinimumHeight(200)

    def draw_topo(self, topo, congestions=None, packets=None):
        self.ax.clear()
        self.ax.set_facecolor(BG2)
        self.figure.patch.set_facecolor(BG2)
        self.ax.set_xlim(0, 300); self.ax.set_ylim(0, 200)
        self.ax.axis("off")
        nodes = topo["nodes"]
        for e in topo["edges"]:
            a, b = nodes[e["a"]], nodes[e["b"]]
            self.ax.plot([a["x"], b["x"]], [a["y"], b["y"]],
                         color="#30363d", lw=1, zorder=1)
        for i, nd in enumerate(nodes):
            cong = (congestions or [])[i] if congestions else 0
            c = RED if cong > 0.7 else AMBER if cong > 0.4 else GREEN
            self.ax.add_patch(plt.Circle((nd["x"], nd["y"]), 10,
                                          fc=BG3, ec=c, lw=1.5, zorder=2))
            self.ax.text(nd["x"], nd["y"], str(i), ha="center", va="center",
                         color=MUTED, fontsize=7, zorder=3)
        self.ax.invert_yaxis()
        self.draw()


# ─────────────────────────────────────────────
# LIVE SIMULATION CANVAS
# ─────────────────────────────────────────────
class LiveCanvas(MplCanvas):
    def __init__(self):
        self.fig = Figure(facecolor="#0a0c10")
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, facecolor="#0a0c10")
        self.ax.axis("off")
        self.topo = None
        self.packets = []

    def set_topo(self, topo):
        self.topo = topo

    def update_packets(self, packets):
        self.packets = packets
        self._redraw()

    def _redraw(self):
        if self.topo is None: return
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#0a0c10")
        ax.axis("off")
        nodes = self.topo["nodes"]
        ax.set_xlim(0, 300); ax.set_ylim(0, 200)

        edge_lvl = {}
        src_set, dst_set = set(), set()
        for p in self.packets:
            si = p.get("step_index", 0)
            if si >= len(p["path"]) - 1: continue
            a, b = p["path"][si], p["path"][si + 1]
            key = (a, b)
            edge_lvl[key] = max(edge_lvl.get(key, 0), p["level"])
            src_set.add(p["path"][0])
            dst_set.add(p["path"][-1])

        for e in self.topo["edges"]:
            a, b = nodes[e["a"]], nodes[e["b"]]
            lvl_ab = edge_lvl.get((e["a"], e["b"]), 0)
            lvl_ba = edge_lvl.get((e["b"], e["a"]), 0)
            max_lvl = max(lvl_ab, lvl_ba)
            if max_lvl > 0:
                c = ("rgba(248,81,73,0.5)" if max_lvl == 3
                     else "rgba(210,153,34,0.5)" if max_lvl == 2
                     else "rgba(88,166,255,0.4)")
                color = (RED if max_lvl == 3 else AMBER if max_lvl == 2 else BLUE)
                ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color=color, lw=2, alpha=0.5, zorder=1)
                for (la, lb, src_node, dst_node) in [(lvl_ab, 0, e["a"], e["b"]),
                                                      (lvl_ba, 0, e["b"], e["a"])]:
                    if la > 0:
                        mx = a["x"] + (b["x"] - a["x"]) * 0.55
                        my = a["y"] + (b["y"] - a["y"]) * 0.55
                        dx = b["x"] - a["x"]; dy = b["y"] - a["y"]
                        dist = math.hypot(dx, dy) or 1
                        ax.annotate("", xy=(mx + dx/dist*3, my + dy/dist*3),
                                    xytext=(mx - dx/dist*3, my - dy/dist*3),
                                    arrowprops=dict(arrowstyle="->", color=RED if la == 3 else AMBER if la == 2 else BLUE, lw=1.2))
            else:
                ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color=BORDER, lw=0.8, alpha=0.7, zorder=1)

        for i, nd in enumerate(nodes):
            is_src = i in src_set
            is_dst = i in dst_set
            n_pkts = sum(1 for p in self.packets if not p.get("delivered") and
                         p["path"][p.get("step_index", 0)] == i if p.get("step_index", 0) < len(p["path"]))
            ring = (CYAN if is_src else PURPLE if is_dst else
                    RED if n_pkts > 4 else AMBER if n_pkts > 2 else GREEN)
            lw = 2.8 if (is_src or is_dst) else 1.8
            ax.add_patch(plt.Circle((nd["x"], nd["y"]), 15, fc=BG3, ec=ring, lw=lw, zorder=2))
            ax.text(nd["x"], nd["y"], str(i), ha="center", va="center",
                    color=TEXT, fontsize=9, fontweight="bold", zorder=3)
            if is_src:
                ax.text(nd["x"], nd["y"] - 20, "TX", ha="center", color=CYAN, fontsize=8, fontweight="bold", zorder=3)
            if is_dst:
                ax.text(nd["x"], nd["y"] + 22, "RX", ha="center", color=PURPLE, fontsize=8, fontweight="bold", zorder=3)

        for p in self.packets:
            si = p.get("step_index", 0)
            if si >= len(p["path"]) - 1: continue
            na = nodes[p["path"][si]]
            nb_ = nodes[p["path"][si + 1]]
            prog = p.get("progress", 0)
            px = na["x"] + (nb_["x"] - na["x"]) * prog
            py = na["y"] + (nb_["y"] - na["y"]) * prog
            c = RED if p["level"] == 3 else AMBER if p["level"] == 2 else BLUE
            ax.add_patch(plt.Circle((px, py), 7, fc=c, ec=c, zorder=4))
            lbl = "H" if p["level"] == 3 else "M" if p["level"] == 2 else "N"
            ax.text(px, py, lbl, ha="center", va="center", color="white",
                    fontsize=7, fontweight="bold", zorder=5)

        ax.invert_yaxis()
        self.draw()


# ─────────────────────────────────────────────
# SIMULATION WORKER THREAD
# ─────────────────────────────────────────────
class SimWorker(QThread):
    progress_sig = pyqtSignal(str, int)
    done_sig = pyqtSignal(dict)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        p = self.params
        num_runs = p.get("num_runs", 30)
        seed     = p.get("seed", 42)
        adaptive_runs, baseline_runs = [], []
        topo = None; traffic_seq = None
        for run_i in range(num_runs):
            random.seed(seed + run_i)
            np.random.seed(seed + run_i)
            topo = build_topo(p["n"], p["conn"])
            traffic_seq = generate_traffic_sequence(
                p["ticks"], p["rate"], p["emerg_frac"], p["scenario"], p["n"])
            adaptive_runs.append(sim_run(topo, traffic_seq, "adaptive",  p["proc_cap"]))
            baseline_runs.append(sim_run(topo, traffic_seq, "baseline",  p["proc_cap"]))
            pct_done = int(10 + 80 * (run_i + 1) / num_runs)
            self.progress_sig.emit(f"Run {run_i+1}/{num_runs} complete…", pct_done)
        self.progress_sig.emit("Evaluating semantic classifier…", 93)
        classifier_eval = eval_classifier()
        self.progress_sig.emit("Aggregating statistics…", 97)
        total_pkts = sum(len(t) for t in traffic_seq) if traffic_seq else 0
        self.done_sig.emit({
            "adaptive":   aggregate_stats(adaptive_runs),
            "baseline":   aggregate_stats(baseline_runs),
            "topo":       topo,
            "classifier": classifier_eval,
            "params":     {**p, "total_pkts": total_pkts},
        })


# ─────────────────────────────────────────────
# LIVE SIMULATION DIALOG
# ─────────────────────────────────────────────
class LiveSimDialog(QDialog):
    def __init__(self, topo, parent=None):
        super().__init__(parent)
        self.topo = topo
        self.setWindowTitle("⚡ Live 6G Network Simulation")
        self.setMinimumSize(900, 620)
        self.setStyleSheet(qss_base())
        self._build_ui()

        self.pkt_id = 0
        self.packets = []
        self.high_count = 0
        self.stats = {"injected": 0, "delivered": 0, "dropped": 0}
        self.speed = 0.9
        self.last_time = time.time()

        self.inject_timer = QTimer(self)
        self.inject_timer.timeout.connect(self._inject)
        self.inject_timer.start(750)

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate_step)
        self.anim_timer.start(33)  # ~30fps

        for _ in range(4):
            self._inject_one()
        self._update_stats()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hdr = QHBoxLayout()
        title_blk = QVBoxLayout()
        title = QLabel("⚡ Live 6G Network Simulation")
        title.setStyleSheet(f"color:{TEXT}; font-size:15px; font-weight:bold;")
        sub = QLabel("Intent-Aware Packet Routing · Real-time Visualization · BFS-Guided Paths")
        sub.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        title_blk.addWidget(title); title_blk.addWidget(sub)
        hdr.addLayout(title_blk)
        hdr.addStretch()

        speed_lbl = QLabel("Animation Speed")
        speed_lbl.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(2, 35)
        self.speed_slider.setValue(9)
        self.speed_slider.setFixedWidth(110)
        self.speed_slider.valueChanged.connect(self._on_speed)
        self.speed_val_lbl = QLabel("0.9×")
        self.speed_val_lbl.setStyleSheet(f"color:{TEXT}; font-size:11px;")
        close_btn = QPushButton("✕ Close")
        close_btn.setStyleSheet(f"color:{MUTED}; font-size:12px;")
        close_btn.clicked.connect(self.accept)
        hdr.addWidget(speed_lbl)
        hdr.addWidget(self.speed_slider)
        hdr.addWidget(self.speed_val_lbl)
        hdr.addWidget(close_btn)
        layout.addLayout(hdr)

        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"background:{BG2}; border:1px solid {BORDER}; border-radius:8px;")
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(0)
        self.lbl_inj = self._stat_widget("Injected", TEXT)
        self.lbl_del = self._stat_widget("Delivered", GREEN)
        self.lbl_fly = self._stat_widget("In-Flight", AMBER)
        self.lbl_high = self._stat_widget("HIGH pkts", RED)
        routing_w = self._stat_widget_fixed("Routing", CYAN, "Adaptive BFS")
        for w in [self.lbl_inj, self.lbl_del, self.lbl_fly, self.lbl_high, routing_w]:
            stats_layout.addWidget(w)
        layout.addWidget(stats_frame)

        self.live_canvas = LiveCanvas()
        self.live_canvas.set_topo(self.topo)
        self.live_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.live_canvas)

        legend_frame = QHBoxLayout()
        for color, label in [(RED, "HIGH / Emergency"), (AMBER, "MEDIUM / Priority"),
                              (BLUE, "NORMAL / Routine")]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{color}; font-size:14px;")
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{MUTED}; font-size:10px;")
            legend_frame.addWidget(dot); legend_frame.addWidget(lbl)
        tx = QLabel("TX = Source   RX = Destination   Ring: 🟢 low → 🟡 moderate → 🔴 high congestion")
        tx.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        legend_frame.addWidget(tx)
        legend_frame.addStretch()
        layout.addLayout(legend_frame)

    def _stat_widget(self, label_text, color):
        w = QWidget()
        w.setStyleSheet(f"border-right:1px solid {BORDER};")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(12, 8, 12, 8)
        lbl = QLabel(label_text.upper())
        lbl.setStyleSheet(f"color:{MUTED}; font-size:9px; letter-spacing:1px; border:none;")
        val = QLabel("0")
        val.setStyleSheet(f"color:{color}; font-size:18px; font-weight:bold; border:none;")
        val.setObjectName("val")
        vl.addWidget(lbl); vl.addWidget(val)
        return w

    def _stat_widget_fixed(self, label_text, color, fixed_val):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(12, 8, 12, 8)
        lbl = QLabel(label_text.upper())
        lbl.setStyleSheet(f"color:{MUTED}; font-size:9px; letter-spacing:1px; border:none;")
        val = QLabel(fixed_val)
        val.setStyleSheet(f"color:{color}; font-size:11px; font-weight:bold; border:none;")
        vl.addWidget(lbl); vl.addWidget(val)
        return w

    def _set_stat(self, widget, value):
        widget.findChild(QLabel, "val").setText(str(value))

    def _on_speed(self, v):
        self.speed = v / 10.0
        self.speed_val_lbl.setText(f"{self.speed:.1f}×")

    def _inject_one(self):
        n = len(self.topo["nodes"])
        if n < 2: return
        src = rand_int(0, n - 1)
        dst = src
        tries = 0
        while dst == src and tries < 10:
            dst = rand_int(0, n - 1); tries += 1
        if dst == src: return
        path = bfs_path(self.topo, src, dst)
        if not path or len(path) < 2: return
        r = random.random()
        level = 3 if r < 0.22 else 2 if r < 0.45 else 1
        if level == 3: self.high_count += 1
        self.packets.append({
            "id": self.pkt_id,
            "path": path, "level": level,
            "step_index": 0, "progress": 0.0,
            "delivered": False
        })
        self.pkt_id += 1
        self.stats["injected"] += 1

    def _inject(self):
        count = rand_int(1, 3)
        for _ in range(count):
            self._inject_one()
        self._update_stats()

    def _animate_step(self):
        now = time.time()
        dt = min(now - self.last_time, 0.06)
        self.last_time = now
        rate = self.speed * 0.55

        alive = []
        for p in self.packets:
            if p["delivered"]: continue
            p["progress"] += rate * dt
            while p["progress"] >= 1:
                p["progress"] -= 1
                p["step_index"] += 1
                if p["step_index"] >= len(p["path"]) - 1:
                    p["delivered"] = True
                    self.stats["delivered"] += 1
                    break
            if not p["delivered"]:
                alive.append(p)

        if len(alive) > 45:
            self.stats["dropped"] += len(alive) - 45
            alive = alive[-45:]
        self.packets = alive

        self._update_stats()
        self.live_canvas.update_packets(self.packets)

    def _update_stats(self):
        self._set_stat(self.lbl_inj,  self.stats["injected"])
        self._set_stat(self.lbl_del,  self.stats["delivered"])
        self._set_stat(self.lbl_fly,  len(self.packets))
        self._set_stat(self.lbl_high, self.high_count)

    def closeEvent(self, e):
        self.inject_timer.stop()
        self.anim_timer.stop()
        super().closeEvent(e)


# ─────────────────────────────────────────────
# RESULTS PANEL
# ─────────────────────────────────────────────
class ResultsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background:{BG};")
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(16, 16, 16, 16)
        self.layout_.setSpacing(14)
        self._show_empty()

    def _show_empty(self):
        self._clear()
        empty = QWidget()
        vl = QVBoxLayout(empty)
        vl.setAlignment(Qt.AlignCenter)
        icon = QLabel("📡")
        icon.setStyleSheet("font-size:36px; opacity:0.4;")
        icon.setAlignment(Qt.AlignCenter)
        h2 = QLabel("No experiment run yet")
        h2.setStyleSheet(f"color:{TEXT}; font-size:16px;")
        h2.setAlignment(Qt.AlignCenter)
        p = QLabel("Configure parameters on the left, then click Run Both Modes to generate comparative results.\n"
                   "The simulation runs Adaptive (intent-aware, congestion-sensitive) and Baseline\n"
                   "(greedy shortest-path) routing under identical 6G traffic conditions.")
        p.setStyleSheet(f"color:{MUTED}; font-size:12px;")
        p.setAlignment(Qt.AlignCenter)
        p.setWordWrap(True)
        vl.addWidget(icon); vl.addWidget(h2); vl.addWidget(p)
        self.layout_.addWidget(empty)

    def _clear(self):
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def render_results(self, R):
        self._clear()
        A, B = R["adaptive"], R["baseline"]
        params  = R["params"]
        cls_eval = R.get("classifier", {})

        # ── Header ──────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("📊 6G Routing Experiment Results")
        title.setStyleSheet(f"color:{TEXT}; font-size:16px; font-weight:bold;")
        nr = params.get("num_runs", 1)
        badge = QLabel(f"{params['scenario']} · {params['n']} nodes · {params['ticks']} ticks · {nr} runs · seed=42")
        badge.setStyleSheet(f"background:{BG2}; border:1px solid {BORDER}; border-radius:5px; padding:4px 12px; color:{MUTED}; font-size:11px;")
        hdr.addWidget(title); hdr.addStretch(); hdr.addWidget(badge)
        hdr_w = QWidget(); hdr_w.setLayout(hdr)
        self.layout_.addWidget(hdr_w)

        # ── Key findings cards ───────────────────────────────
        em_imp  = pct(A["emerg_lat_mean"], B["emerg_lat_mean"])
        pdr_imp = pct(A["pdr_mean"],       B["pdr_mean"])
        overhead = (f"{A['control_msgs_mean'] / (params['ticks'] * params['n']) * 100:.1f}"
                    if params["n"] and params["ticks"] else "—")
        findings_row = QHBoxLayout()
        for (lbl, val, sub, col) in [
            ("Emergency Latency Reduction",
             (f"{abs(float(em_imp)):.1f}%" if em_imp != "—" and float(em_imp) < 0 else f"{em_imp}%"),
             f"Adaptive {A['emerg_lat_mean']:.1f} ± {A['emerg_lat_std']:.1f} ms vs Baseline", GREEN),
            ("PDR Improvement",
             (f"+{pdr_imp}%" if pdr_imp != "—" and float(pdr_imp) > 0 else f"{pdr_imp}%"),
             f"Adaptive {A['pdr_mean']:.1f} ± {A['pdr_std']:.1f}% vs Baseline {B['pdr_mean']:.1f}%", GREEN),
            ("Control Overhead", f"{overhead}%",
             "Agent coordination msgs per tick·node", AMBER),
        ]:
            card = QFrame()
            card.setStyleSheet(f"background:{BG2}; border:1px solid {GREEN if col==GREEN else AMBER}; border-radius:8px;")
            cl = QVBoxLayout(card)
            l1 = QLabel(lbl.upper()); l1.setStyleSheet(f"color:{MUTED}; font-size:10px; font-weight:bold; border:none;")
            l2 = QLabel(val);         l2.setStyleSheet(f"color:{col}; font-size:22px; font-weight:bold; letter-spacing:-1px; border:none;")
            l3 = QLabel(sub);         l3.setStyleSheet(f"color:{MUTED}; font-size:11px; border:none;")
            cl.addWidget(l1); cl.addWidget(l2); cl.addWidget(l3)
            findings_row.addWidget(card)
        fw = QWidget(); fw.setLayout(findings_row)
        self.layout_.addWidget(fw)

        # ── Comparison table ─────────────────────────────────
        def _build_table(rows_data):
            t = QTableWidget()
            t.setStyleSheet(f"background:{BG2}; color:{TEXT};")
            t.setColumnCount(5)
            t.setHorizontalHeaderLabels(["Metric", "Adaptive (AI Agent)", "Baseline (Greedy)", "Δ Change", "Better?"])
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            t.setEditTriggers(QTableWidget.NoEditTriggers)
            t.setAlternatingRowColors(False); t.verticalHeader().setVisible(False)
            t.setRowCount(len(rows_data))
            for ri, (metric, a_v, b_v, delta, sig, better) in enumerate(rows_data):
                for ci, cell in enumerate([metric, a_v, b_v, delta, sig]):
                    item = QTableWidgetItem(str(cell))
                    if ci == 3 and better is not None:
                        item.setForeground(QColor(GREEN if better else RED))
                    elif ci == 4 and cell == "✓ yes":
                        item.setForeground(QColor(GREEN))
                    else:
                        item.setForeground(QColor(TEXT))
                    t.setItem(ri, ci, item)
            return t

        def _row(label, a_m, a_s, b_m, b_s, lower=True):
            if a_m == 0 and b_m == 0:
                return (label, "—", "—", "—", "—", None)
            d = pct(a_m, b_m)
            better = (a_m < b_m) if lower else (a_m > b_m)
            sig = abs(a_m - b_m) > (a_s + b_s) * 0.5
            arrow = "↓ " if better else "↑ "
            delta = (arrow + str(d) + "%") if d != "—" else "—"
            return (label, f"{a_m:.1f} ± {a_s:.1f}", f"{b_m:.1f} ± {b_s:.1f}", delta,
                    "✓ yes" if sig else "~ marginal", better)

        tbl_rows = [
            _row("Emergency latency (ms)",   A["emerg_lat_mean"],  A["emerg_lat_std"],  B["emerg_lat_mean"],  B["emerg_lat_std"]),
            _row("Medium latency (ms)",      A["med_lat_mean"],    A["med_lat_std"],    B["med_lat_mean"],    B["med_lat_std"]),
            _row("Normal latency (ms)",      A["normal_lat_mean"], A["normal_lat_std"], B["normal_lat_mean"], B["normal_lat_std"]),
            _row("P95 emerg latency (ms)",   A["p95_emerg_mean"],  A["p95_emerg_std"],  B["p95_emerg_mean"],  B["p95_emerg_std"]),
            _row("P99 emerg latency (ms)",   A["p99_emerg_mean"],  A["p99_emerg_std"],  B["p99_emerg_mean"],  B["p99_emerg_std"]),
            _row("Throughput (pkts/s)",      A["throughput_mean"], A["throughput_std"], B["throughput_mean"], B["throughput_std"], lower=False),
            _row("Packet loss rate (%)",     A["loss_rate_mean"],  A["loss_rate_std"],  B["loss_rate_mean"],  B["loss_rate_std"]),
            _row("PDR (%)",                  A["pdr_mean"],        A["pdr_std"],        B["pdr_mean"],        B["pdr_std"],       lower=False),
            ("Delivered (mean)",  f"{A['delivered_mean']:.0f}", f"{B['delivered_mean']:.0f}", "—", "—", None),
            ("Control msgs (mean)",f"{A['control_msgs_mean']:.0f}",f"{B['control_msgs_mean']:.0f}","—","—",None),
        ]
        table = _build_table(tbl_rows)
        table.setMaximumHeight(280)
        table_frame = QFrame()
        table_frame.setStyleSheet(f"background:{BG2}; border:1px solid {BORDER}; border-radius:8px;")
        tfl = QVBoxLayout(table_frame); tfl.setContentsMargins(0, 0, 0, 0); tfl.addWidget(table)
        self.layout_.addWidget(table_frame)

        # ── Charts 2×2 ────────────────────────────────────────
        charts_w = QWidget(); charts_grid = QGridLayout(charts_w); charts_grid.setSpacing(10)

        fig1, ax1 = plt.subplots(facecolor=BG2); self._style_ax(ax1)
        x = np.arange(3)
        ax1.bar(x-0.2, [A["emerg_lat_mean"], A["med_lat_mean"], A["normal_lat_mean"]], 0.4, label="Adaptive", color=GREEN, alpha=0.9)
        ax1.bar(x+0.2, [B["emerg_lat_mean"], B["med_lat_mean"], B["normal_lat_mean"]], 0.4, label="Baseline", color=AMBER, alpha=0.9)
        ax1.set_xticks(x); ax1.set_xticklabels(["Emergency","Medium","Normal"])
        ax1.set_title("Latency by Priority Class (ms)", color=MUTED, fontsize=10); ax1.legend(fontsize=9)
        c1 = FigureCanvas(fig1); c1.setFixedHeight(200)

        fig2, ax2 = plt.subplots(facecolor=BG2); self._style_ax(ax2)
        ptc = A["per_tick_cong_mean"]; ptcb = B["per_tick_cong_mean"]
        step = max(1, len(ptc) // 40)
        xs = list(range(0, len(ptc), step))
        ax2.plot(xs, ptc[::step], color=GREEN, lw=1.5, label="Adaptive")
        ax2.plot(xs, ptcb[::step], color=AMBER, lw=1.5, linestyle="--", label="Baseline")
        ax2.set_ylim(0, 1); ax2.set_title("Network Congestion over Time", color=MUTED, fontsize=10); ax2.legend(fontsize=9)
        c2 = FigureCanvas(fig2); c2.setFixedHeight(200)

        fig3, ax3 = plt.subplots(facecolor=BG2); self._style_ax(ax3)
        x3 = np.arange(2)
        ax3.bar(x3-0.2, [A["loss_rate_mean"], B["loss_rate_mean"]], 0.4, label="Loss Rate %", color=RED, alpha=0.9)
        ax3.bar(x3+0.2, [A["p95_emerg_mean"], B["p95_emerg_mean"]], 0.4, label="P95 Emerg (ms)", color=BLUE, alpha=0.9)
        ax3.set_xticks(x3); ax3.set_xticklabels(["Adaptive","Baseline"])
        ax3.set_title("Packet Loss & P95 Latency", color=MUTED, fontsize=10); ax3.legend(fontsize=9)
        c3 = FigureCanvas(fig3); c3.setFixedHeight(200)

        fig4, ax4 = plt.subplots(facecolor=BG2); self._style_ax(ax4)
        x4 = np.arange(2)
        ax4.bar(x4-0.2, [A["emerg_lat_mean"] or 0, B["emerg_lat_mean"] or 0], 0.4, label="Emergency", color=RED,  alpha=0.9)
        ax4.bar(x4+0.2, [A["normal_lat_mean"] or 0, B["normal_lat_mean"] or 0], 0.4, label="Normal",    color=BLUE, alpha=0.9)
        ax4.set_xticks(x4); ax4.set_xticklabels(["Adaptive","Baseline"])
        ax4.set_title("Emergency vs Normal Latency Split", color=MUTED, fontsize=10); ax4.legend(fontsize=9)
        c4 = FigureCanvas(fig4); c4.setFixedHeight(200)

        for c in [c1,c2,c3,c4]: c.setStyleSheet(f"background:{BG2};")
        charts_grid.addWidget(c1,0,0); charts_grid.addWidget(c2,0,1)
        charts_grid.addWidget(c3,1,0); charts_grid.addWidget(c4,1,1)
        self.layout_.addWidget(charts_w)

        # ── Semantic Classifier Evaluation ───────────────────
        if cls_eval:
            v1 = cls_eval.get("v1", {}); v2 = cls_eval.get("v2", {})
            sec_lbl = QLabel("SEMANTIC CLASSIFIER EVALUATION")
            sec_lbl.setStyleSheet(f"color:{MUTED}; font-size:10px; font-weight:bold; letter-spacing:1px;")
            self.layout_.addWidget(sec_lbl)

            cls_row = QHBoxLayout()

            # Metrics table
            cls_tbl = QTableWidget(); cls_tbl.setStyleSheet(f"background:{BG2}; color:{TEXT};")
            cls_tbl.setColumnCount(3); cls_tbl.setRowCount(4)
            cls_tbl.setHorizontalHeaderLabels(["Metric", "V1 (Original)", "V2 (Enhanced)"])
            cls_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            cls_tbl.setEditTriggers(QTableWidget.NoEditTriggers)
            cls_tbl.verticalHeader().setVisible(False)
            for ri, (metric, k) in enumerate([("Accuracy (%)", "accuracy"), ("Precision (%)", "precision"),
                                               ("Recall (%)", "recall"),   ("F1 Score (%)", "f1")]):
                cls_tbl.setItem(ri, 0, QTableWidgetItem(metric))
                for ci, d in enumerate([v1, v2], 1):
                    val = d.get(k, 0)
                    item = QTableWidgetItem(f"{val:.1f}")
                    item.setForeground(QColor(GREEN if ci==2 else AMBER))
                    cls_tbl.setItem(ri, ci, item)
            cls_tbl.setFixedHeight(140)
            cls_row.addWidget(cls_tbl)

            conf = v2.get("confusion", [[0]*3]*3)
            fig_cm, ax_cm = plt.subplots(figsize=(3, 2.2), facecolor=BG2)
            self._style_ax(ax_cm)
            ax_cm.imshow(conf, cmap="YlOrRd", aspect="auto", vmin=0)
            labels_ = v2.get("class_labels", ["N","M","H"])
            ax_cm.set_xticks([0,1,2]); ax_cm.set_xticklabels(labels_, fontsize=8)
            ax_cm.set_yticks([0,1,2]); ax_cm.set_yticklabels(labels_, fontsize=8)
            ax_cm.set_xlabel("Predicted", color=MUTED, fontsize=8)
            ax_cm.set_ylabel("True", color=MUTED, fontsize=8)
            ax_cm.set_title("Confusion Matrix (V2)", color=MUTED, fontsize=9)
            for r in range(3):
                for c_ in range(3):
                    ax_cm.text(c_, r, str(conf[r][c_]), ha="center", va="center",
                               color="white", fontsize=10, fontweight="bold")
            c_cm = FigureCanvas(fig_cm); c_cm.setFixedHeight(200); c_cm.setStyleSheet(f"background:{BG2};")
            cls_row.addWidget(c_cm)

            fig_acc, ax_acc = plt.subplots(figsize=(2.8, 2.2), facecolor=BG2)
            self._style_ax(ax_acc)
            ax_acc.bar(["Original", "Enhanced"], [v1.get("accuracy",0), v2.get("accuracy",0)],
                       color=[AMBER, GREEN], alpha=0.9, width=0.5)
            ax_acc.set_ylim(0, 100)
            ax_acc.set_title("Classifier Accuracy (%)", color=MUTED, fontsize=9)
            for xi, (_, val) in enumerate([("V1", v1.get("accuracy",0)), ("V2", v2.get("accuracy",0))]):
                ax_acc.text(xi, val + 1.5, f"{val:.1f}%", ha="center", color=TEXT, fontsize=9, fontweight="bold")
            c_acc = FigureCanvas(fig_acc); c_acc.setFixedHeight(200); c_acc.setStyleSheet(f"background:{BG2};")
            cls_row.addWidget(c_acc)

            cls_w = QWidget(); cls_w.setLayout(cls_row)
            cls_frame = QFrame()
            cls_frame.setStyleSheet(f"background:{BG2}; border:1px solid {BORDER}; border-radius:8px;")
            cfl = QVBoxLayout(cls_frame); cfl.setContentsMargins(8,8,8,8); cfl.addWidget(cls_w)
            self.layout_.addWidget(cls_frame)

        # ── Final Summary Table ───────────────────────────────
        sum_lbl = QLabel("FINAL SUMMARY TABLE")
        sum_lbl.setStyleSheet(f"color:{MUTED}; font-size:10px; font-weight:bold; letter-spacing:1px;")
        self.layout_.addWidget(sum_lbl)

        v1_acc = cls_eval.get("v1", {}).get("accuracy", 0) if cls_eval else 0
        v2_acc = cls_eval.get("v2", {}).get("accuracy", 0) if cls_eval else 0
        acc_imp = f"+{v2_acc - v1_acc:.1f} pts" if v2_acc and v1_acc else "—"

        def _simp(a_m, a_s, b_m, b_s, lower=True, pct_fmt=False):
            a_str = f"{a_m:.1f} ± {a_s:.1f}" + ("%" if pct_fmt else "")
            b_str = f"{b_m:.1f} ± {b_s:.1f}" + ("%" if pct_fmt else "")
            d = pct(a_m, b_m)
            if d == "—": return a_str, b_str, "—", None
            better = (a_m < b_m) if lower else (a_m > b_m)
            sign = "↓ " if better else "↑ "
            return a_str, b_str, sign + d + "%", better

        sum_rows = []
        a_s, b_s, d_s, bt = _simp(A["pdr_mean"], A["pdr_std"], B["pdr_mean"], B["pdr_std"], lower=False, pct_fmt=True)
        sum_rows.append(("Packet Delivery Ratio", b_s, a_s, d_s, bt))
        a_s, b_s, d_s, bt = _simp(A["emerg_lat_mean"], A["emerg_lat_std"], B["emerg_lat_mean"], B["emerg_lat_std"])
        sum_rows.append(("Emergency Latency (ms)", b_s, a_s, d_s, bt))
        a_s, b_s, d_s, bt = _simp(A["loss_rate_mean"], A["loss_rate_std"], B["loss_rate_mean"], B["loss_rate_std"])
        sum_rows.append(("Packet Loss Rate (%)", b_s, a_s, d_s, bt))
        a_s, b_s, d_s, bt = _simp(A["p95_emerg_mean"], A["p95_emerg_std"], B["p95_emerg_mean"], B["p95_emerg_std"])
        sum_rows.append(("P95 Latency (ms)", b_s, a_s, d_s, bt))
        a_s, b_s, d_s, bt = _simp(A["throughput_mean"], A["throughput_std"], B["throughput_mean"], B["throughput_std"], lower=False)
        sum_rows.append(("Throughput (pkts/s)", b_s, a_s, d_s, bt))
        sum_rows.append(("Semantic Accuracy",
                         f"{v1_acc:.1f}% (V1)" if v1_acc else "—",
                         f"{v2_acc:.1f}% (V2)" if v2_acc else "—",
                         acc_imp, True if v2_acc > v1_acc else None))

        sum_tbl = QTableWidget(); sum_tbl.setStyleSheet(f"background:{BG2}; color:{TEXT};")
        sum_tbl.setColumnCount(4); sum_tbl.setRowCount(len(sum_rows))
        sum_tbl.setHorizontalHeaderLabels(["Metric", "Baseline", "Adaptive", "Improvement"])
        sum_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        sum_tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        sum_tbl.verticalHeader().setVisible(False); sum_tbl.setMaximumHeight(200)
        for ri, (metric, b_v, a_v, imp, better) in enumerate(sum_rows):
            for ci, cell in enumerate([metric, b_v, a_v, imp]):
                item = QTableWidgetItem(str(cell))
                if ci == 3 and better is not None:
                    item.setForeground(QColor(GREEN if better else RED))
                elif ci == 2:
                    item.setForeground(QColor(GREEN))
                else:
                    item.setForeground(QColor(TEXT))
                sum_tbl.setItem(ri, ci, item)
        sum_frame = QFrame()
        sum_frame.setStyleSheet(f"background:{BG2}; border:1px solid {BORDER}; border-radius:8px;")
        sfl = QVBoxLayout(sum_frame); sfl.setContentsMargins(0,0,0,0); sfl.addWidget(sum_tbl)
        self.layout_.addWidget(sum_frame)
        self.layout_.addStretch()

    def _style_ax(self, ax):
        ax.set_facecolor(BG2)
        ax.figure.patch.set_facecolor(BG2)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.yaxis.label.set_color(MUTED)
        ax.xaxis.label.set_color(MUTED)
        ax.grid(color="#21262d", linestyle="-", linewidth=0.5, alpha=0.7)


# ─────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("6G Intent-Aware Routing — Semantic Edge Simulator v3")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(qss_base())
        self.topo = None
        self.active_scenario = "uniform"
        self.last_results = None
        self._build_ui()
        self._rebuild_net()
        self._test_classify()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_h = QHBoxLayout(central)
        main_h.setContentsMargins(0, 0, 0, 0)
        main_h.setSpacing(0)

        # ── LEFT SIDEBAR ──
        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        sidebar.setStyleSheet(f"background:{BG2}; border-right:1px solid {BORDER};")
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidget(sidebar)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(320)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setStyleSheet(f"background:{BG2}; border:none;")

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        brand = QWidget()
        brand.setStyleSheet(f"background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0d1a2e,stop:1 #1c2330); border-bottom:2px solid #1c3a5e;")
        bl = QVBoxLayout(brand)
        bl.setContentsMargins(14, 14, 14, 12)
        top_row = QHBoxLayout()
        icon_lbl = QLabel("📡"); icon_lbl.setStyleSheet("font-size:18px;")
        t_lbl = QLabel("6G INTENT-AWARE ROUTING")
        t_lbl.setStyleSheet(f"color:{BLUE}; font-size:12px; font-weight:bold; letter-spacing:1px;")
        top_row.addWidget(icon_lbl); top_row.addWidget(t_lbl); top_row.addStretch()
        sub_lbl = QLabel("Autonomous AI Agents · Voice Communication\nSemantic Edge Network Simulator v3")
        sub_lbl.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        tags_row = QHBoxLayout()
        for tag in ["⚡ URLLC", "🧠 Semantic QoS", "🔄 Adaptive"]:
            t = QLabel(tag)
            t.setStyleSheet(f"background:#0d1f35; border:1px solid #1c3a5e; border-radius:3px; padding:1px 6px; color:{BLUE}; font-size:9px; font-weight:bold;")
            tags_row.addWidget(t)
        tags_row.addStretch()
        bl.addLayout(top_row); bl.addWidget(sub_lbl); bl.addLayout(tags_row)
        sidebar_layout.addWidget(brand)

        sidebar_layout.addWidget(self._section("Experiment Parameters", self._params_widget()))
        sidebar_layout.addWidget(self._section("Traffic Scenario", self._scenario_widget()))
        sidebar_layout.addWidget(self._section("Voice Intent Classifier — 6G Semantic", self._classifier_widget()))
        sidebar_layout.addWidget(self._section("Run Experiment", self._run_widget()))
        sidebar_layout.addWidget(self._section("Network Topology Preview", self._net_preview_widget()))
        sidebar_layout.addStretch()

        main_h.addWidget(sidebar_scroll)

        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setStyleSheet(f"background:{BG}; border:none;")
        self.results_panel = ResultsPanel()
        results_scroll.setWidget(self.results_panel)
        main_h.addWidget(results_scroll)

    def _section(self, title, content_widget):
        frame = QFrame()
        frame.setStyleSheet(f"background:{BG2}; border-bottom:1px solid {BORDER};")
        vl = QVBoxLayout(frame)
        vl.setContentsMargins(14, 12, 14, 12)
        vl.setSpacing(4)
        t = QLabel(title.upper())
        t.setStyleSheet(f"color:{MUTED}; font-size:10px; font-weight:bold; letter-spacing:1px;")
        vl.addWidget(t)
        vl.addWidget(content_widget)
        return frame

    def _params_widget(self):
        w = QWidget()
        vl = QVBoxLayout(w); vl.setContentsMargins(0, 4, 0, 0); vl.setSpacing(6)

        def slider_row(label, id_, min_, max_, val, fmt=None):
            lbl = QLabel(label); lbl.setStyleSheet(f"color:{MUTED}; font-size:11px;")
            row = QHBoxLayout()
            sl = QSlider(Qt.Horizontal); sl.setRange(min_, max_); sl.setValue(val)
            val_lbl = QLabel(str(val) if fmt is None else fmt(val))
            val_lbl.setStyleSheet(f"color:{TEXT}; font-size:12px; min-width:36px;")
            sl.valueChanged.connect(lambda v, vl=val_lbl, f=fmt: vl.setText(str(v) if f is None else f(v)))
            row.addWidget(sl); row.addWidget(val_lbl)
            vl.addWidget(lbl); vl.addLayout(row)
            return sl

        self.sl_ticks = slider_row("Simulation ticks per run", "ticks", 50, 500, 200)
        self.sl_nodes = slider_row("Nodes in network", "nodes", 6, 18, 10)
        self.sl_nodes.valueChanged.connect(lambda _: self._rebuild_net())
        self.sl_conn  = slider_row("Connectivity (avg. neighbors)", "conn", 2, 5, 3)
        self.sl_conn.valueChanged.connect(lambda _: self._rebuild_net())
        self.sl_rate  = slider_row("Packet injection rate (per tick)", "rate", 1, 6, 3)
        self.sl_emerg = slider_row("Emergency traffic fraction", "emerg", 5, 60, 20, fmt=lambda v: f"{v}%")
        self.sl_proc  = slider_row("Node processing capacity (per tick)", "proc", 1, 5, 2)
        self.sl_runs  = slider_row("Experiment runs (statistics)", "runs", 5, 50, 30)
        seed_row = QHBoxLayout()
        seed_lbl = QLabel("Random seed:"); seed_lbl.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        seed_val = QLabel("42"); seed_val.setStyleSheet(f"color:{GREEN}; font-size:12px; font-weight:bold;")
        seed_row.addWidget(seed_lbl); seed_row.addWidget(seed_val); seed_row.addStretch()
        vl.addLayout(seed_row)
        return w

    def _scenario_widget(self):
        w = QWidget()
        gl = QGridLayout(w); gl.setSpacing(5)
        scenarios = [
            ("uniform",  "Uniform",      False, "Balanced load across all nodes"),
            ("burst",    "Emerg. Burst", False, "Sudden surge of emergency traffic midway"),
            ("hotspot",  "Hotspot",      False, "All traffic targets one congested node"),
            ("ramp",     "Ramp Load",    False, "Traffic load gradually increases over time"),
            ("disaster", "🌪 Disaster",  True,  "Mass casualty / natural disaster — 65% emergency, 2× packet rate"),
            ("voice",    "📞 Voice 6G",  False, "Voice call establishment — mixed priority streams"),
        ]
        self.sc_buttons = {}
        for idx, (key, label, is_disaster, tip) in enumerate(scenarios):
            btn = QPushButton(label)
            btn.setToolTip(tip)
            color = RED if is_disaster else BLUE
            btn.setStyleSheet(f"""
                QPushButton {{
                    background:{BG3}; border:1px solid {BORDER}; color:{MUTED};
                    padding:7px 6px; border-radius:5px; font-size:10px;
                }}
                QPushButton:hover {{ border-color:{color}; color:{TEXT}; }}
                QPushButton[active=true] {{ background:#1c2f45; border-color:{color}; color:{color}; }}
            """)
            btn.clicked.connect(lambda checked, k=key: self._set_scenario(k))
            gl.addWidget(btn, idx // 3, idx % 3)
            self.sc_buttons[key] = btn
        self._set_scenario("uniform")
        return w

    def _set_scenario(self, name):
        self.active_scenario = name
        for k, btn in self.sc_buttons.items():
            btn.setProperty("active", k == name)
            btn.style().unpolish(btn); btn.style().polish(btn)

    def _classifier_widget(self):
        w = QWidget()
        vl = QVBoxLayout(w); vl.setContentsMargins(0, 4, 0, 0); vl.setSpacing(6)
        desc = QLabel("Type any message to see QoS priority level assigned by the semantic engine. "
                       "Disaster, emergency, and weather-hazard terms all trigger HIGH.")
        desc.setStyleSheet(f"color:{MUTED}; font-size:10px;"); desc.setWordWrap(True)
        vl.addWidget(desc)
        self.cls_input = QTextEdit()
        self.cls_input.setPlainText("fire emergency at sector 7")
        self.cls_input.setFixedHeight(54)
        self.cls_input.textChanged.connect(self._test_classify)
        vl.addWidget(self.cls_input)
        result_row = QHBoxLayout()
        self.cls_badge = QLabel("HIGH")
        self.cls_badge.setStyleSheet(f"background:#3d1a1a; color:{RED}; border-radius:4px; padding:2px 8px; font-size:10px; font-weight:bold;")
        self.cls_score = QLabel("score: 8")
        self.cls_score.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        self.cls_match = QLabel("")
        self.cls_match.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        result_row.addWidget(self.cls_badge); result_row.addWidget(self.cls_score)
        result_row.addStretch(); result_row.addWidget(self.cls_match)
        vl.addLayout(result_row)
        ex_row = QHBoxLayout()
        ex_lbl = QLabel("Try:"); ex_lbl.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        ex_row.addWidget(ex_lbl)
        for txt, color in [("tornado", RED), ("hurricane", RED), ("storm", AMBER), ("flood", RED)]:
            btn = QPushButton(txt)
            btn.setStyleSheet(f"background:transparent; border:none; color:{color}; font-size:10px; padding:0; text-decoration:underline; cursor:pointer;")
            btn.clicked.connect(lambda checked, t=txt: self._set_example(t))
            ex_row.addWidget(btn)
        ex_row.addStretch()
        vl.addLayout(ex_row)
        return w

    def _set_example(self, word):
        examples = {
            "tornado": "tornado warning evacuate",
            "hurricane": "hurricane force winds incoming",
            "storm": "storm alert coastal region",
            "flood": "flood emergency rescue needed",
        }
        self.cls_input.setPlainText(examples.get(word, word))

    def _test_classify(self):
        txt = self.cls_input.toPlainText()
        r = classify(txt)
        self.cls_badge.setText(r["priority"])
        colors = {"HIGH": (RED, "#3d1a1a"), "MEDIUM": (AMBER, "#3d2a00"), "NORMAL": (GREEN, "#1a3a1a")}
        fg, bg = colors.get(r["priority"], (GREEN, "#1a3a1a"))
        self.cls_badge.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:4px; padding:2px 8px; font-size:10px; font-weight:bold;")
        self.cls_score.setText(f"score: {r['score']}")
        self.cls_match.setText("↑ " + ", ".join(r["matched"]) if r["matched"] else "no keywords matched")

    def _run_widget(self):
        w = QWidget()
        vl = QVBoxLayout(w); vl.setContentsMargins(0, 4, 0, 0); vl.setSpacing(6)
        self.btn_run = QPushButton("▶ Run Both Modes (Adaptive + Baseline)")
        self.btn_run.setStyleSheet(f"background:#1a3a1a; border:1px solid {GREEN}; color:{GREEN}; padding:6px 14px; border-radius:6px;")
        self.btn_run.clicked.connect(self._run_experiment)
        self.btn_live = QPushButton("⚡ Live Simulation View")
        self.btn_live.setStyleSheet(f"background:#1c3a5e; border:1px solid {BLUE}; color:{BLUE}; padding:6px 14px; border-radius:6px;")
        self.btn_live.clicked.connect(self._open_live)
        self.btn_export = QPushButton("↓ Export Results (JSON)")
        self.btn_export.clicked.connect(self._export_results)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.status_lbl = QLabel("Configure parameters and run experiment")
        self.status_lbl.setStyleSheet(f"color:{MUTED}; font-size:11px;"); self.status_lbl.setAlignment(Qt.AlignCenter)
        vl.addWidget(self.btn_run); vl.addWidget(self.btn_live)
        vl.addWidget(self.btn_export); vl.addWidget(self.progress_bar); vl.addWidget(self.status_lbl)
        return w

    def _net_preview_widget(self):
        w = QWidget()
        vl = QVBoxLayout(w); vl.setContentsMargins(0, 4, 0, 0)
        self.net_label = QLabel("10 nodes · 3-connectivity")
        self.net_label.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        vl.addWidget(self.net_label)
        net_frame = QFrame()
        net_frame.setStyleSheet(f"background:{BG2}; border:1px solid {BORDER}; border-radius:8px;")
        nfl = QVBoxLayout(net_frame); nfl.setContentsMargins(0, 0, 0, 0)
        self.topo_canvas = TopoCanvas()
        nfl.addWidget(self.topo_canvas)
        vl.addWidget(net_frame)
        return w

    def _rebuild_net(self):
        n = self.sl_nodes.value() if hasattr(self, "sl_nodes") else 10
        conn = self.sl_conn.value() if hasattr(self, "sl_conn") else 3
        self.topo = build_topo(n, conn)
        if hasattr(self, "net_label"):
            self.net_label.setText(f"{n} nodes · {conn}-connectivity")
        if hasattr(self, "topo_canvas"):
            self.topo_canvas.draw_topo(self.topo, [], [])

    def _run_experiment(self):
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ Running…")
        params = {
            "ticks":    self.sl_ticks.value(),
            "n":        self.sl_nodes.value(),
            "conn":     self.sl_conn.value(),
            "rate":     self.sl_rate.value(),
            "emerg_frac": self.sl_emerg.value(),
            "proc_cap": self.sl_proc.value(),
            "scenario": self.active_scenario,
            "num_runs": self.sl_runs.value(),
            "seed":     42,
        }
        self.worker = SimWorker(params)
        self.worker.progress_sig.connect(self._on_progress)
        self.worker.done_sig.connect(self._on_done)
        self.worker.start()

    def _on_progress(self, msg, p):
        self.status_lbl.setText(msg)
        self.progress_bar.setValue(p)

    def _on_done(self, R):
        self.last_results = R
        self.topo = R["topo"]
        self.topo_canvas.draw_topo(self.topo, [], [])
        self.results_panel.render_results(R)
        p = R["params"]
        self.status_lbl.setText(f"Done — {p['total_pkts']} packets · {p['ticks']} ticks · {p['scenario']} scenario")
        self.progress_bar.setValue(100)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶ Run Both Modes (Adaptive + Baseline)")

    def _open_live(self):
        if self.topo is None:
            self._rebuild_net()
        dlg = LiveSimDialog(self.topo, self)
        dlg.exec_()

    def _export_results(self):
        if not self.last_results:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "No results", "Run an experiment first.")
            return
        R = self.last_results
        A, B = R["adaptive"], R["baseline"]
        p = R["params"]
        cls = R.get("classifier", {})
        out = {
            "metadata": {k: v for k, v in p.items() if k != "topo"},
            "adaptive": {
                "pdr":              f"{A['pdr_mean']:.2f} ± {A['pdr_std']:.2f}",
                "delivered_mean":   A["delivered_mean"], "dropped_mean": A["dropped_mean"],
                "emerg_latency_ms": f"{A['emerg_lat_mean']:.1f} ± {A['emerg_lat_std']:.1f}",
                "p95_emerg_ms":     f"{A['p95_emerg_mean']:.1f} ± {A['p95_emerg_std']:.1f}",
                "throughput_pps":   f"{A['throughput_mean']:.1f} ± {A['throughput_std']:.1f}",
                "loss_rate_pct":    f"{A['loss_rate_mean']:.2f} ± {A['loss_rate_std']:.2f}",
                "control_msgs_mean": A["control_msgs_mean"],
            },
            "baseline": {
                "pdr":              f"{B['pdr_mean']:.2f} ± {B['pdr_std']:.2f}",
                "delivered_mean":   B["delivered_mean"], "dropped_mean": B["dropped_mean"],
                "emerg_latency_ms": f"{B['emerg_lat_mean']:.1f} ± {B['emerg_lat_std']:.1f}",
                "p95_emerg_ms":     f"{B['p95_emerg_mean']:.1f} ± {B['p95_emerg_std']:.1f}",
                "throughput_pps":   f"{B['throughput_mean']:.1f} ± {B['throughput_std']:.1f}",
                "loss_rate_pct":    f"{B['loss_rate_mean']:.2f} ± {B['loss_rate_std']:.2f}",
                "control_msgs_mean": B["control_msgs_mean"],
            },
            "delta": {
                "emerg_latency_pct": pct(A["emerg_lat_mean"], B["emerg_lat_mean"]) + "%",
                "pdr_pct":           pct(A["pdr_mean"],       B["pdr_mean"])       + "%",
                "loss_rate_pct":     pct(A["loss_rate_mean"], B["loss_rate_mean"]) + "%",
            },
            "classifier": {
                "v1_accuracy": cls.get("v1", {}).get("accuracy", 0),
                "v2_accuracy": cls.get("v2", {}).get("accuracy", 0),
                "v1_f1":       cls.get("v1", {}).get("f1", 0),
                "v2_f1":       cls.get("v2", {}).get("f1", 0),
            },
        }
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Export Results", f"6g_routing_results_{int(time.time())}.json", "JSON Files (*.json)")
        if path:
            with open(path, "w") as f:
                json.dump(out, f, indent=2)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor(BG2))
    palette.setColor(QPalette.AlternateBase, QColor(BG3))
    palette.setColor(QPalette.ToolTipBase, QColor(BG2))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(BG3))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor(BLUE))
    palette.setColor(QPalette.HighlightedText, QColor(BG))
    app.setPalette(palette)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
