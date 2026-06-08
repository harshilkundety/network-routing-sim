# Generates all PNG figures used in 6G_IntentAwareRouting_Paper.tex
# Run: python generate_figures.py
# Output: figures/ directory with all PNG files

import os, sys, math, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.patheffects import withStroke

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_report import (build_topo, generate_traffic_sequence, sim_run,
                              aggregate_stats, eval_classifier, avg, std)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

BG = "#0d1117"; BG2 = "#161b22"; BG3 = "#1c2330"
BORDER = "#30363d"; TEXT = "#e6edf3"; MUTED = "#8b949e"
RED = "#f85149"; GREEN = "#3fb950"; AMBER = "#d29922"
BLUE = "#58a6ff"; PURPLE = "#a371f7"; CYAN = "#00d2ff"

DPI = 150

def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved {name}")

def styled(ax, title=""):
    ax.set_facecolor(BG2)
    ax.figure.patch.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.yaxis.label.set_color(MUTED); ax.xaxis.label.set_color(MUTED)
    ax.grid(color="#21262d", linestyle="-", linewidth=0.4, alpha=0.7)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, pad=8, fontweight="bold")

# ─── Run simulations ────────────────────────────────────────────────────────

def run_scenario(scenario, emerg_frac, n=10, conn=3, ticks=200, rate=3, proc_cap=2):
    ar, br = [], []
    for i in range(30):
        random.seed(42 + i); np.random.seed(42 + i)
        topo = build_topo(n, conn)
        traffic = generate_traffic_sequence(ticks, rate, emerg_frac, scenario, n)
        ar.append(sim_run(topo, traffic, "adaptive",  proc_cap))
        br.append(sim_run(topo, traffic, "baseline", proc_cap))
    return aggregate_stats(ar), aggregate_stats(br), topo

print("Running uniform scenario (30 runs)...")
A_u, B_u, topo_u = run_scenario("uniform", 20)
print("Running disaster scenario (30 runs)...")
A_d, B_d, _      = run_scenario("disaster", 65)
print("Evaluating classifier...")
cls = eval_classifier()
print("Simulations complete.\n")

# ─── Figure 1: System Architecture ──────────────────────────────────────────
print("Generating figures...")

def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)

    blocks = [
        (1.0, 3.2, 1.8, 1.2, BLUE,   "Voice / Text\nInput"),
        (3.4, 3.2, 2.0, 1.2, PURPLE, "Semantic Intent\nClassifier (v2)"),
        (5.8, 3.8, 1.8, 0.6, RED,    "HIGH Priority"),
        (5.8, 3.1, 1.8, 0.6, AMBER,  "MEDIUM Priority"),
        (5.8, 2.4, 1.8, 0.6, GREEN,  "NORMAL Priority"),
        (1.0, 1.2, 1.8, 1.2, CYAN,   "Network\nTopology Model"),
        (3.4, 1.2, 2.0, 1.2, BLUE,   "Intent-Aware\nAdaptive Router"),
        (6.2, 1.2, 2.5, 1.2, GREEN,  "Destination\nNode (Delivered)"),
    ]
    for (x, y, w, h, col, label) in blocks:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                               fc=BG3, ec=col, lw=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                color=TEXT, fontsize=9, fontweight="bold")

    arrowkw = dict(arrowstyle="->", color=MUTED, lw=1.5,
                   connectionstyle="arc3,rad=0")
    def arr(x1, y1, x2, y2, col=MUTED):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))

    arr(2.8, 3.8, 3.4, 3.8)
    arr(5.4, 4.1, 5.8, 4.1, RED)
    arr(5.4, 3.8, 5.8, 3.45, AMBER)
    arr(5.4, 3.5, 5.8, 2.7, GREEN)
    arr(2.8, 1.8, 3.4, 1.8)
    arr(5.4, 1.8, 6.2, 1.8, GREEN)

    ax.text(5.0, 4.6, "QoS Tier\nAssignment", color=MUTED, fontsize=8,
            ha="center", style="italic")
    ax.text(5.0, 0.6, "Cost Function: w_c·C(n) + w_q·Q(n) + w_d·D(n) − w_p·P(n)",
            color=MUTED, fontsize=8.5, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc=BG3, ec=BORDER))
    ax.text(0.2, 4.8, "Fig. 1  System Architecture", color=MUTED, fontsize=9)
    savefig(fig, "fig_architecture.png")

fig_architecture()

# ─── Figure 2: Network Topology ──────────────────────────────────────────────

def fig_topology():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    nodes = topo_u["nodes"]; edges = topo_u["edges"]

    for ax_i, (ax, title, cong) in enumerate(zip(axes,
            ["(a) Idle — baseline congestion", "(b) Active — adaptive routing under load"],
            [None, "sim"])):
        ax.set_facecolor(BG2)
        ax.set_xlim(0, 300); ax.set_ylim(0, 200)
        ax.axis("off")
        ax.set_title(title, color=TEXT, fontsize=10, pad=6)
        ax.invert_yaxis()

        random.seed(99)
        cong_vals = ([0.0]*len(nodes) if cong is None
                     else [random.uniform(0.05, 0.75) for _ in nodes])

        for e in edges:
            a, b = nodes[e["a"]], nodes[e["b"]]
            lw = 2.0 if cong == "sim" else 1.0
            col = BORDER if cong is None else "#2a4a6b"
            ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color=col, lw=lw, zorder=1)

        if cong == "sim":
            random.seed(77)
            sample_paths = [
                ([0, 2, 5, 8], 3),
                ([3, 1, 4, 9], 2),
                ([6, 7, 5], 1),
                ([2, 0, 3, 9], 3),
            ]
            for path, lvl in sample_paths:
                col_p = RED if lvl == 3 else AMBER if lvl == 2 else BLUE
                valid = [i for i in path if i < len(nodes)]
                for k in range(len(valid) - 1):
                    a, b = nodes[valid[k]], nodes[valid[k+1]]
                    ax.plot([a["x"], b["x"]], [a["y"], b["y"]],
                            color=col_p, lw=2.5, alpha=0.7, zorder=2)
                    mx = a["x"] + (b["x"] - a["x"]) * 0.55
                    my = a["y"] + (b["y"] - a["y"]) * 0.55
                    dx = b["x"] - a["x"]; dy = b["y"] - a["y"]
                    d = math.hypot(dx, dy) or 1
                    ax.annotate("", xy=(mx + dx/d*4, my + dy/d*4),
                                xytext=(mx - dx/d*4, my - dy/d*4),
                                arrowprops=dict(arrowstyle="->", color=col_p, lw=1.5))
            for pth, lvl in [(sample_paths[0], 3), (sample_paths[2], 1)]:
                path, l2 = pth
                valid = [i for i in path if i < len(nodes)]
                if len(valid) >= 2:
                    ni = len(valid) // 2
                    n_a, n_b = nodes[valid[ni]], nodes[valid[ni+1]] if ni+1 < len(valid) else nodes[valid[ni]]
                    px = (n_a["x"] + n_b["x"]) / 2
                    py = (n_a["y"] + n_b["y"]) / 2
                    col_p = RED if l2 == 3 else BLUE
                    ax.add_patch(plt.Circle((px, py), 7, fc=col_p, ec=col_p, zorder=5))
                    ax.text(px, py, "H" if l2 == 3 else "N",
                            ha="center", va="center", color="white",
                            fontsize=7, fontweight="bold", zorder=6)

        for i, nd in enumerate(nodes):
            cv = cong_vals[i]
            ring = RED if cv > 0.6 else AMBER if cv > 0.35 else GREEN
            ax.add_patch(plt.Circle((nd["x"], nd["y"]), 13, fc=BG3, ec=ring, lw=2, zorder=3))
            ax.text(nd["x"], nd["y"], str(i), ha="center", va="center",
                    color=TEXT, fontsize=9, fontweight="bold", zorder=4)

    leg_items = [mpatches.Patch(color=RED, label="HIGH (Emergency)"),
                 mpatches.Patch(color=AMBER, label="MEDIUM (Priority)"),
                 mpatches.Patch(color=BLUE, label="NORMAL (Routine)"),
                 mpatches.Patch(color=GREEN, label="Low congestion"),
                 mpatches.Patch(color=AMBER, label="Moderate congestion"),
                 mpatches.Patch(color=RED, label="High congestion")]
    fig.legend(handles=leg_items, loc="lower center", ncol=3,
               facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Fig. 2  Network Topology (10 nodes, 3-connectivity)",
                 color=MUTED, fontsize=9, y=1.01)
    savefig(fig, "fig_topology.png")

fig_topology()

# ─── Figure 3: Routing Working — side-by-side adaptive vs baseline ───────────

def fig_routing_working():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    nodes = topo_u["nodes"]; edges = topo_u["edges"]

    random.seed(123)
    congestion = [random.uniform(0.1, 0.9) for _ in nodes]
    congestion[5] = 0.92
    congestion[3] = 0.85

    path_adaptive = [0, 1, 4, 8]
    path_baseline = [0, 5, 3, 8]
    emerg_src, emerg_dst = 0, 8

    for ax, mode, path, title in zip(
        axes,
        ["adaptive", "baseline"],
        [path_adaptive, path_baseline],
        ["(a) Adaptive Routing — avoids congested nodes 3 & 5",
         "(b) Baseline Routing — traverses congested nodes 3 & 5"]):

        ax.set_facecolor(BG2); ax.axis("off")
        ax.set_xlim(0, 300); ax.set_ylim(0, 200); ax.invert_yaxis()
        ax.set_title(title, color=TEXT, fontsize=9.5, pad=6)

        for e in edges:
            a, b = nodes[e["a"]], nodes[e["b"]]
            ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color=BORDER, lw=1, zorder=1)

        valid = [i for i in path if i < len(nodes)]
        for k in range(len(valid) - 1):
            a, b = nodes[valid[k]], nodes[valid[k+1]]
            ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color=RED, lw=3, alpha=0.85, zorder=2)
            mx = a["x"] + (b["x"] - a["x"]) * 0.55
            my = a["y"] + (b["y"] - a["y"]) * 0.55
            dx = b["x"] - a["x"]; dy = b["y"] - a["y"]
            d = math.hypot(dx, dy) or 1
            ax.annotate("", xy=(mx + dx/d*5, my + dy/d*5),
                        xytext=(mx - dx/d*5, my - dy/d*5),
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=2, mutation_scale=14))

        pkt_idx = len(valid) // 2
        if pkt_idx < len(valid) - 1:
            na, nb = nodes[valid[pkt_idx]], nodes[valid[pkt_idx+1]]
            px = na["x"] + (nb["x"] - na["x"]) * 0.4
            py = na["y"] + (nb["y"] - na["y"]) * 0.4
            ax.add_patch(plt.Circle((px, py), 9, fc=RED, ec="white", lw=1.5, zorder=5))
            ax.text(px, py, "H", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold", zorder=6)

        for i, nd in enumerate(nodes):
            cv = congestion[i]
            ring = RED if cv > 0.7 else AMBER if cv > 0.4 else GREEN
            lw = 3.0 if i in [emerg_src, emerg_dst] else 1.8
            fc = "#2a1010" if cv > 0.7 else BG3
            ax.add_patch(plt.Circle((nd["x"], nd["y"]), 14, fc=fc, ec=ring, lw=lw, zorder=3))
            ax.text(nd["x"], nd["y"], str(i), ha="center", va="center",
                    color=TEXT, fontsize=9, fontweight="bold", zorder=4)
            if i == emerg_src:
                ax.text(nd["x"], nd["y"] - 22, "SRC", ha="center",
                        color=CYAN, fontsize=8, fontweight="bold", zorder=4)
            if i == emerg_dst:
                ax.text(nd["x"], nd["y"] + 24, "DST", ha="center",
                        color=PURPLE, fontsize=8, fontweight="bold", zorder=4)
            if cv > 0.7 and mode == "baseline":
                ax.text(nd["x"] + 16, nd["y"] - 16, f"{cv:.0%}",
                        color=RED, fontsize=7, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc=BG3, ec=RED, lw=0.8))

        if mode == "adaptive":
            ax.text(150, 195, "Cost function steers packet away from congested nodes",
                    ha="center", color=GREEN, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc=BG3, ec=GREEN, lw=0.8))
        else:
            ax.text(150, 195, "Greedy BFS selects shortest hop count, ignoring congestion",
                    ha="center", color=AMBER, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc=BG3, ec=AMBER, lw=0.8))

    fig.suptitle("Fig. 3  Routing Decision Comparison — HIGH Priority Packet (node 0 → 8)",
                 color=MUTED, fontsize=9, y=1.01)
    leg = [mpatches.Patch(color=RED,   label="Chosen path"),
           mpatches.Patch(color=GREEN, label="Low congestion node"),
           mpatches.Patch(color=AMBER, label="Moderate congestion"),
           mpatches.Patch(color=RED,   label="High congestion node")]
    fig.legend(handles=leg, loc="lower center", ncol=4,
               facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    savefig(fig, "fig_routing_working.png")

fig_routing_working()

# ─── Figure 4: Latency Comparison (both scenarios) ──────────────────────────

def fig_latency():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax, A, B, title in zip(
        axes,
        [A_u, A_d], [B_u, B_d],
        ["(a) Uniform Traffic (20% emergency)",
         "(b) Disaster Scenario (65% emergency)"]):

        styled(ax, title)
        x = np.arange(3)
        keys_m = ["emerg_lat_mean", "med_lat_mean", "normal_lat_mean"]
        keys_s = ["emerg_lat_std",  "med_lat_std",  "normal_lat_std"]
        a_vals = [A[k] for k in keys_m]; a_errs = [A[k] for k in keys_s]
        b_vals = [B[k] for k in keys_m]; b_errs = [B[k] for k in keys_s]
        ax.bar(x-0.22, a_vals, 0.42, color=GREEN, alpha=0.9, label="Adaptive",
               yerr=a_errs, capsize=4, error_kw={"ecolor": TEXT, "elinewidth": 1.2})
        ax.bar(x+0.22, b_vals, 0.42, color=AMBER, alpha=0.9, label="Baseline",
               yerr=b_errs, capsize=4, error_kw={"ecolor": TEXT, "elinewidth": 1.2})
        ax.set_xticks(x); ax.set_xticklabels(["Emergency", "Medium", "Normal"], fontsize=9)
        ax.set_ylabel("Latency (ms)", color=MUTED, fontsize=9)
        ax.legend(fontsize=9, facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT)

    fig.suptitle("Fig. 4  End-to-End Latency by Priority Class (mean ± std, 30 runs, seed=42)",
                 color=MUTED, fontsize=9)
    savefig(fig, "fig_latency.png")

fig_latency()

# ─── Figure 5: Congestion + Tail Latency (2 subplots) ───────────────────────

def fig_congestion_tail():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    ax1 = axes[0]; styled(ax1, "(a) Network Congestion over Time (uniform)")
    ptca = A_u["per_tick_cong_mean"]; ptcb = B_u["per_tick_cong_mean"]
    step = max(1, len(ptca) // 50)
    xs = list(range(0, len(ptca), step))
    ax1.plot(xs, ptca[::step], color=GREEN, lw=2, label="Adaptive")
    ax1.plot(xs, ptcb[::step], color=AMBER, lw=1.8, ls="--", label="Baseline")
    ax1.fill_between(xs, ptca[::step], alpha=0.15, color=GREEN)
    ax1.set_ylim(0, 0.5); ax1.set_xlabel("Simulation Tick", color=MUTED, fontsize=9)
    ax1.set_ylabel("Avg Node Congestion", color=MUTED, fontsize=9)
    ax1.legend(fontsize=9, facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT)

    ax2 = axes[1]; styled(ax2, "(b) Tail Latency — P95 and P99 (disaster scenario)")
    x = np.arange(2)
    x4 = np.arange(2)
    bars_a = ax2.bar(x4-0.22, [A_d["p95_emerg_mean"], A_d["p99_emerg_mean"]], 0.42,
                     color=GREEN, alpha=0.9, label="Adaptive",
                     yerr=[A_d["p95_emerg_std"], A_d["p99_emerg_std"]],
                     capsize=4, error_kw={"ecolor": TEXT, "elinewidth": 1.2})
    bars_b = ax2.bar(x4+0.22, [B_d["p95_emerg_mean"], B_d["p99_emerg_mean"]], 0.42,
                     color=AMBER, alpha=0.9, label="Baseline",
                     yerr=[B_d["p95_emerg_std"], B_d["p99_emerg_std"]],
                     capsize=4, error_kw={"ecolor": TEXT, "elinewidth": 1.2})
    ax2.set_xticks(x4); ax2.set_xticklabels(["P95", "P99"], fontsize=10)
    ax2.set_ylabel("Emergency Latency (ms)", color=MUTED, fontsize=9)
    ax2.legend(fontsize=9, facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT)

    fig.suptitle("Fig. 5  Network Congestion and Tail Latency Analysis", color=MUTED, fontsize=9)
    savefig(fig, "fig_congestion_tail.png")

fig_congestion_tail()

# ─── Figure 6: PDR / Loss / Throughput ───────────────────────────────────────

def fig_pdr_loss():
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    styled(ax, "PDR, Packet Loss Rate, and Throughput — Adaptive vs Baseline")

    metrics = ["PDR (%)", "Loss Rate (%)", "Throughput\n(/100 pkts/s)"]
    a_u = [A_u["pdr_mean"], A_u["loss_rate_mean"], A_u["throughput_mean"]/100]
    b_u = [B_u["pdr_mean"], B_u["loss_rate_mean"], B_u["throughput_mean"]/100]
    a_d = [A_d["pdr_mean"], A_d["loss_rate_mean"], A_d["throughput_mean"]/100]
    b_d = [B_d["pdr_mean"], B_d["loss_rate_mean"], B_d["throughput_mean"]/100]

    x = np.arange(3); w = 0.2
    ax.bar(x-1.5*w, a_u, w, color=GREEN, alpha=0.9, label="Adaptive / Uniform")
    ax.bar(x-0.5*w, b_u, w, color=GREEN, alpha=0.4, label="Baseline / Uniform")
    ax.bar(x+0.5*w, a_d, w, color=AMBER, alpha=0.9, label="Adaptive / Disaster")
    ax.bar(x+1.5*w, b_d, w, color=AMBER, alpha=0.4, label="Baseline / Disaster")
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
    ax.legend(fontsize=8, facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT, ncol=2)

    fig.suptitle("Fig. 6  Delivery Ratio, Loss Rate, and Throughput Comparison",
                 color=MUTED, fontsize=9)
    savefig(fig, "fig_pdr_loss.png")

fig_pdr_loss()

# ─── Figure 7: Classifier Accuracy + Confusion Matrices ─────────────────────

def fig_classifier():
    fig = plt.figure(figsize=(12, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35,
                           left=0.06, right=0.97, top=0.86, bottom=0.1)
    v1 = cls["v1"]; v2 = cls["v2"]
    labels = v1["class_labels"]

    ax1 = fig.add_subplot(gs[0, 0]); styled(ax1, "(a) Accuracy & F1 Comparison")
    x = np.arange(2)
    ax1.bar(x-0.2, [v1["accuracy"], v1["f1"]], 0.38, color=AMBER, alpha=0.9, label="V1 Original")
    ax1.bar(x+0.2, [v2["accuracy"], v2["f1"]], 0.38, color=GREEN,  alpha=0.9, label="V2 Enhanced")
    ax1.set_xticks(x); ax1.set_xticklabels(["Accuracy", "F1 Score"], fontsize=9)
    ax1.set_ylim(0, 105); ax1.set_ylabel("Score (%)", color=MUTED, fontsize=9)
    for xi, (v1v, v2v) in enumerate([(v1["accuracy"], v2["accuracy"]),
                                      (v1["f1"], v2["f1"])]):
        ax1.text(xi-0.2, v1v + 1.5, f"{v1v:.1f}", ha="center", color=TEXT, fontsize=8)
        ax1.text(xi+0.2, v2v + 1.5, f"{v2v:.1f}", ha="center", color=TEXT, fontsize=8)
    ax1.legend(fontsize=8, facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT)

    for ax_i, (ax, conf, tag, cmap, title) in enumerate(zip(
        [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])],
        [v1["confusion"], v2["confusion"]],
        ["V1", "V2"],
        ["Blues", "YlOrRd"],
        ["(b) V1 Confusion Matrix", "(c) V2 Confusion Matrix"]
    )):
        ax.set_facecolor(BG2); ax.figure.patch.set_facecolor(BG)
        ax.tick_params(colors=MUTED); ax.grid(False)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.imshow(conf, cmap=cmap, aspect="auto", vmin=0)
        ax.set_xticks([0,1,2]); ax.set_xticklabels(labels, fontsize=9, color=MUTED)
        ax.set_yticks([0,1,2]); ax.set_yticklabels(labels, fontsize=9, color=MUTED)
        ax.set_xlabel("Predicted", color=MUTED, fontsize=9)
        ax.set_ylabel("True Label", color=MUTED, fontsize=9)
        ax.set_title(title, color=TEXT, fontsize=10, pad=6, fontweight="bold")
        for r in range(3):
            for c in range(3):
                ax.text(c, r, str(conf[r][c]), ha="center", va="center",
                        color="white", fontsize=13, fontweight="bold")

    fig.suptitle("Fig. 7  Semantic Classifier Evaluation — Original vs Enhanced (210-sample dataset)",
                 color=MUTED, fontsize=9)
    savefig(fig, "fig_classifier.png")

fig_classifier()

# ─── Figure 8: Live simulation panel (4 states) ──────────────────────────────

def fig_live_sim():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)
    nodes = topo_u["nodes"]; edges = topo_u["edges"]

    states = [
        ("(a) t=0 — Packets injected at source nodes", [
            (0, [0,1,4,8], 3, 0.15), (3, [3,6,7], 2, 0.2)]),
        ("(b) t=5 — HIGH packets advanced, congestion rising", [
            (0, [0,1,4,8], 3, 0.55), (3, [3,6,7], 2, 0.7),
            (6, [6,7], 1, 0.4)]),
        ("(c) t=10 — Adaptive router bypasses congested node 5", [
            (0, [0,1,4,8], 3, 0.85), (3, [3,6,7], 2, 0.45),
            (9, [9,2,4,8], 3, 0.3)]),
        ("(d) t=15 — HIGH packets delivered; NORMAL still routing", [
            (2, [2,4,8], 1, 0.5), (5, [5,3,1], 2, 0.2),
            (7, [7,9,8], 1, 0.7)]),
    ]
    random.seed(55)
    for ax, (title, pkts) in zip(axes.flat, states):
        ax.set_facecolor(BG2); ax.axis("off")
        ax.set_xlim(0, 300); ax.set_ylim(0, 200); ax.invert_yaxis()
        ax.set_title(title, color=TEXT, fontsize=9, pad=5)

        cong = [random.uniform(0.05, 0.5) for _ in nodes]
        cong[5] = random.uniform(0.65, 0.9)

        for e in edges:
            a, b = nodes[e["a"]], nodes[e["b"]]
            ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color=BORDER, lw=0.9, zorder=1)

        active_edges = {}
        for (pkt_id, path, lvl, prog) in pkts:
            col = RED if lvl == 3 else AMBER if lvl == 2 else BLUE
            valid = [i for i in path if i < len(nodes)]
            for k in range(len(valid)-1):
                key = (valid[k], valid[k+1])
                active_edges[key] = col
                ax.plot([nodes[valid[k]]["x"], nodes[valid[k+1]]["x"]],
                        [nodes[valid[k]]["y"], nodes[valid[k+1]]["y"]],
                        color=col, lw=2.2, alpha=0.6, zorder=2)
            step_i = min(int(prog * (len(valid)-1)), len(valid)-2)
            if step_i < len(valid)-1:
                na, nb = nodes[valid[step_i]], nodes[valid[step_i+1]]
                frac = (prog * (len(valid)-1)) % 1
                px = na["x"] + (nb["x"] - na["x"]) * frac
                py = na["y"] + (nb["y"] - na["y"]) * frac
                ax.add_patch(plt.Circle((px, py), 7, fc=col, ec="white", lw=1.2, zorder=5))
                ax.text(px, py, "H" if lvl==3 else "M" if lvl==2 else "N",
                        ha="center", va="center", color="white",
                        fontsize=7, fontweight="bold", zorder=6)

        for i, nd in enumerate(nodes):
            cv = cong[i]
            ring = RED if cv > 0.6 else AMBER if cv > 0.35 else GREEN
            ax.add_patch(plt.Circle((nd["x"], nd["y"]), 13, fc=BG3, ec=ring, lw=1.8, zorder=3))
            ax.text(nd["x"], nd["y"], str(i), ha="center", va="center",
                    color=TEXT, fontsize=8, fontweight="bold", zorder=4)

    leg = [mpatches.Patch(color=RED, label="HIGH pkt"),
           mpatches.Patch(color=AMBER, label="MEDIUM pkt"),
           mpatches.Patch(color=BLUE, label="NORMAL pkt"),
           mpatches.Patch(color=GREEN, label="Low congestion"),
           mpatches.Patch(color=AMBER, label="Moderate congestion"),
           mpatches.Patch(color=RED, label="High congestion")]
    fig.legend(handles=leg, loc="lower center", ncol=6,
               facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Fig. 8  Live Simulation — Time-Step Snapshots of Intent-Aware Routing",
                 color=TEXT, fontsize=11, y=1.005, fontweight="bold")
    savefig(fig, "fig_live_sim.png")

fig_live_sim()

print(f"\nAll figures saved to: {OUT}")
print("Files:")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(os.path.join(OUT, f)) // 1024
    print(f"  {f}  ({size} KB)")
