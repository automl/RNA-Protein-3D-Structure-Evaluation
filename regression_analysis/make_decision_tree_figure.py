#!/usr/bin/env python3
"""Render Figure 3, the accuracy-calibrated decision tree. Reads
decision_branch_stats.csv (produced by decision_branch_stats.py); every branch
number comes from that file. Writes tex/figures/Fig3_decision_tree.{png,pdf,svg,eps}."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pandas as pd
from figsave import save_all_formats   # figsave.py sits next to this script

df = pd.read_csv("decision_branch_stats.csv")
def val(level, branch, metric="assigned_rna_tm"):
    r = df[(df.level == level) & (df.branch == branch) & (df.metric == metric)].iloc[0]
    return float(r["mean"]), int(r["n"])

sys_classes = [("Single-chain RNA", "Single-chain\nRNA"),
               ("RNA-protein 1v1", "RNA-protein\n1v1"),
               ("RNA-protein multichain", "RNA-protein\nmultichain")]
comb = {k: val("L1_system", k) for k, _ in sys_classes}
post = {k: val("CUT_post", f"{k} | post-cutoff") for k, _ in sys_classes}
len_mc = [("$\\leq$40", *val("L3_len", "RNA-protein multichain | ≤40 nt")),
          ("41-100",   *val("L3_len", "RNA-protein multichain | 41-100 nt")),
          ("101-200",  *val("L3_len", "RNA-protein multichain | 101-200 nt")),
          (">200",     *val("L3_len", "RNA-protein multichain | >200 nt"))]
len_ro = [val("L3_len", "Single-chain RNA | ≤40 nt")[0], val("L3_len", "Single-chain RNA | >200 nt")[0]]
cx_1v1_tm = val("CX_class", "1v1", "Complex_TM")[0]
cx_mc_tm  = val("CX_class", "multichain", "Complex_TM")[0]

norm = Normalize(0.30, 1.00); cmap = plt.get_cmap("RdYlGn")
def col(v): return cmap(norm(v))
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.2, 5.2)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def box(x, y, w, h, text, face="#eef2f7", fs=10, weight="normal", ec="#3a4a5a"):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.012,rounding_size=0.02",
                 linewidth=1.1, edgecolor=ec, facecolor=face))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight=weight, zorder=5)
def line(x1, y1, x2, y2): ax.plot([x1, x2], [y1, y2], color="#7a8794", lw=1.1, zorder=0)

box(0.5, 0.93, 0.46, 0.08, "AF3 RNA / RNA-protein prediction\n(no experimental reference)",
    face="#dde8f2", fs=10, weight="bold")
for (key, label), x in zip(sys_classes, [0.17, 0.5, 0.83]):
    m, n = comb[key]; pm, _ = post[key]
    line(0.5, 0.89, x, 0.715)
    box(x, 0.66, 0.27, 0.135, f"{label}\nRNA TM {m:.2f}  (n={n})\nnovel: {pm:.2f}",
        face=col(m), fs=9.5, weight="bold")
ax.text(0.5, 0.50, "RNA chain length effect (within class, conditional mean RNA TM)",
        ha="center", va="center", fontsize=9.5, fontstyle="italic", color="#26323c")
for (lab, m, n), x in zip(len_mc, [0.20, 0.40, 0.60, 0.80]):
    box(x, 0.37, 0.165, 0.10, f"{lab} nt\nTM {m:.2f}\n(n={n})", face=col(m), fs=9)
    line(0.5, 0.585, x, 0.425)
ax.text(0.5, 0.255, f"Multichain shown; single-chain RNA spans {len_ro[0]:.2f} ($\\leq$40 nt) to "
        f"{len_ro[1]:.2f} (>200 nt).\nThe gradient tracks alignment depth and reverses without an RNA MSA.",
        ha="center", va="center", fontsize=8.4, color="#3a4a5a")
ax.text(0.5, 0.135, f"Complex level: Complex TM {cx_1v1_tm:.2f} (1v1) vs {cx_mc_tm:.2f} (multichain); "
        "interface DockQ $\\approx$0.59 in both.\nA high complex TM does not imply a reliable interface.",
        ha="center", va="center", fontsize=8.6, color="#3a4a5a")
ax.text(0.5, 0.035, "On top of these expectations, a per-structure good/mid/bad tier is read "
        "from AF3 confidence (Appendix: operational tiers).",
        ha="center", va="center", fontsize=8.6, fontstyle="italic", color="#26323c")
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cax = fig.add_axes([0.91, 0.18, 0.016, 0.30]); cb = fig.colorbar(sm, cax=cax)
cb.set_label("expected RNA TM", fontsize=8.5); cb.ax.tick_params(labelsize=8)

os.makedirs("tex/figures", exist_ok=True)
written = save_all_formats(fig, "tex/figures/Fig3_decision_tree.pdf", dpi=600, close=True)
print("wrote:", [os.path.basename(str(p)) for p in written])
