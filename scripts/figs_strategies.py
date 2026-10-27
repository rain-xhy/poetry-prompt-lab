"""Figures 1 to 3 from the stage 2 analysis files: strategy means, trap type heatmap, model dumbbell."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from figs_common import *  # noqa: E402,F401,F403

A = os.path.join(RES, "stage2_main", "analysis")


def fig1():
    rows = sorted(read_csv(os.path.join(A, "descriptive_accuracy.csv")), key=lambda r: float(r["mean"]))
    base = next(float(r["mean"]) for r in rows if r["strategy"] == "S0")
    n = len(rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for i, r in enumerate(rows):
        ax.plot([float(r["ci_lo"]), float(r["ci_hi"])], [i, i], color=CAT[0], lw=1.5,
                solid_capstyle="round", zorder=2)
        ax.plot([float(r["mean"])], [i], "o", ms=7, mfc=CAT[0], mec=SURFACE, mew=1.5, zorder=3)
    ax.axvline(base, color=MUTED, lw=0.8, zorder=1)
    ax.text(base, n - 0.3, "基线均值 %.3f" % base, ha="center", va="bottom", fontsize=8, color=INK2,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5), zorder=4)
    top = rows[-1]
    ax.text(float(top["ci_hi"]) + 0.008, n - 1, "%.3f" % float(top["mean"]), va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([NAME[r["strategy"]] for r in rows])
    ax.set_xlim(0.5, 0.92)
    ax.set_ylim(-0.6, n + 0.3)
    ax.set_xlabel("语义保留率，诗层面均值与百分之九十五自助区间")
    clean(ax, "x")
    save(fig, "fig1_strategy_means", ["strategy", "mean", "ci_lo", "ci_hi"],
         [[r["strategy"], r["mean"], r["ci_lo"], r["ci_hi"]] for r in reversed(rows)])


def fig2():
    rows = read_csv(os.path.join(A, "by_trap_accuracy.csv"))
    first = list(rows[0].keys())[0]
    strategies = [k for k in rows[0] if k != first]
    mat = {r[first]: {s: float(r[s]) for s in strategies} for r in rows}
    traps = [t for t in TRAP if t in mat]
    order = sorted(strategies, key=lambda s: -np.mean([mat[t][s] for t in traps]))
    data = np.array([[mat[t][s] for t in traps] for s in order])
    cmap = LinearSegmentedColormap.from_list("blue", SEQ)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    im = ax.pcolormesh(data, cmap=cmap, vmin=0.5, vmax=0.9, edgecolors=SURFACE, linewidth=2)
    ax.set_xticks(np.arange(len(traps)) + 0.5)
    ax.set_xticklabels(traps)
    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_yticklabels([NAME[s] for s in order])
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    for i in range(len(order)):
        for j in range(len(traps)):
            ax.text(j + 0.5, i + 0.5, "%.2f" % data[i, j], ha="center", va="center", fontsize=8.5,
                    color=text_on(cmap((data[i, j] - 0.5) / 0.4)))
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, labelsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8)
    cb.set_label("语义保留率", color=INK2, fontsize=8.5)
    save(fig, "fig2_trap_heatmap", ["strategy"] + traps,
         [[s] + ["%.4f" % mat[t][s] for t in traps] for s in order])


def fig3():
    rows = read_csv(os.path.join(A, "by_model_accuracy.csv"))
    first = list(rows[0].keys())[0]
    data = {r[first]: (float(r["Qwen25"]), float(r["Qwen3"])) for r in rows}
    order = sorted(data, key=lambda s: sum(data[s]))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, s in enumerate(order):
        a, b = data[s]
        ax.plot([a, b], [i, i], color=AXIS, lw=1.5, zorder=1)
        ax.plot([a], [i], MARK[0], ms=7, mfc=CAT[0], mec=SURFACE, mew=1.5, zorder=3)
        ax.plot([b], [i], MARK[1], ms=7, mfc=CAT[1], mec=SURFACE, mew=1.5, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAME[s] for s in order])
    ax.set_xlim(0.55, 0.8)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("语义保留率，各模型内的均值")
    clean(ax, "x")
    ax.legend(handles=[Line2D([], [], marker=MARK[k], ls="none", mfc=CAT[k], mec=SURFACE, ms=7, label=MODEL[m])
                       for k, m in enumerate(("Qwen25", "Qwen3"))],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8.5, handletextpad=0.3)
    save(fig, "fig3_model_dumbbell", ["strategy", "Qwen2.5 3B", "Qwen3 1.7B"],
         [[s, "%.4f" % data[s][0], "%.4f" % data[s][1]] for s in reversed(order)])


if __name__ == "__main__":
    style()
    fig1()
    fig2()
    fig3()
