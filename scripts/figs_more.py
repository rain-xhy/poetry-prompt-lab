"""Figures 4 to 7: adversarial set, stage 4 attribution, scorer versus human labels, format compliance."""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from figs_common import *  # noqa: E402,F401,F403
from stage1_analyze import load_rows  # noqa: E402

S3 = ["S0", "S1_T1", "S5_T1"]


def fig4():
    """Adversarial set: unit pass rate per text and strategy."""
    df = load_rows("stage3_adversarial")
    m = df.groupby(["poem", "strategy"]).accuracy.mean().unstack()[S3]
    order = sorted(m.index, key=lambda p: m.loc[p].mean())
    data = m.loc[order].values
    cmap = LinearSegmentedColormap.from_list("blue", SEQ)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    im = ax.pcolormesh(data, cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    ax.set_xticks(np.arange(3) + 0.5)
    ax.set_xticklabels([NAME[s] for s in S3])
    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    for i in range(len(order)):
        for j in range(3):
            ax.text(j + 0.5, i + 0.5, "%.2f" % data[i, j], ha="center", va="center", fontsize=8.5,
                    color=text_on(cmap(data[i, j])))
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, labelsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8)
    cb.set_label("语义单元通过率", color=INK2, fontsize=8.5)
    save(fig, "fig4_adversarial", ["text"] + S3, [[p] + ["%.4f" % v for v in m.loc[p]] for p in order])


def fig5():
    """Stage 4: share of unit outcomes by trap type, parsed outputs only."""
    df = pd.read_csv(os.path.join(RES, "stage4_path", "analysis", "error_attribution.csv"))
    df = df[df.parsed == True]  # noqa: E712
    cls = ["success", "understanding_error", "expression_error"]
    lab = {"success": "成功", "understanding_error": "理解错误", "expression_error": "表达错误"}
    groups = [("全部", df)] + [(t, df[df.trap_type == t]) for t in TRAP]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    rows = []
    for i, (g, d) in enumerate(groups):
        n = len(d)
        left = 0.0
        shares = [(d.attribution == c).mean() for c in cls]
        for k, (c, s) in enumerate(zip(cls, shares)):
            ax.barh(i, s, left=left, height=0.55, color=CAT[k], edgecolor=SURFACE, linewidth=1.5)
            if s >= 0.09:
                ax.text(left + s / 2, i, "%.0f%%" % (100 * s), ha="center", va="center", fontsize=8,
                        color=text_on(CAT[k]))
            left += s
        rows.append([g, n] + ["%.4f" % s for s in shares])
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(["%s（%d）" % (g, len(d)) for g, d in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("语义单元占比，括号内为单元数")
    clean(ax, None)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CAT[k]) for k in range(3)],
              labels=[lab[c] for c in cls], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, fontsize=8.5, handlelength=1, handleheight=1)
    save(fig, "fig5_stage4_attribution", ["group", "n_units"] + cls, rows)


def fig6():
    """Scorer verdict against the researcher's labels on the 179 audited units."""
    A = os.path.join(RES, "audit")
    with open(os.path.join(A, "audit_key.json"), encoding="utf-8") as f:
        key = json.load(f)
    raw = open(os.path.join(A, "audit_sheet_expert.csv"), "rb").read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("gbk")
    import io
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    hcol = next(c for c in rows[0] if c.startswith("human_pass"))
    icol = next(c for c in rows[0] if c.startswith("item"))
    cnt = np.zeros((2, 2), dtype=int)  # rows: human pass, fail; cols: scorer pass, fail
    for r in rows:
        h = int(r[hcol].strip())
        s = bool(key[r[icol].strip()]["scorer_pass"])
        cnt[0 if h else 1, 0 if s else 1] += 1
    n = cnt.sum()
    cmap = LinearSegmentedColormap.from_list("blue", SEQ)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.pcolormesh(cnt, cmap=cmap, vmin=0, vmax=cnt.max() * 1.05, edgecolors=SURFACE, linewidth=3)
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5, "%d\n%.0f%%" % (cnt[i, j], 100 * cnt[i, j] / n), ha="center",
                    va="center", fontsize=10, color=text_on(cmap(cnt[i, j] / (cnt.max() * 1.05))))
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["评审器判通过", "评审器判不通过"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["人工判通过", "人工判不通过"])
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, labelsize=8.5)
    save(fig, "fig6_scorer_vs_human", ["human", "scorer_pass", "scorer_fail"],
         [["pass", cnt[0, 0], cnt[0, 1]], ["fail", cnt[1, 0], cnt[1, 1]]])


def fig7():
    """Format compliance by strategy, stage 2: Chinese left in the body, single line, extra paragraphs."""
    with open(os.path.join(RES, "stage2_main", "analysis", "format_check.json"), encoding="utf-8") as f:
        summ = json.load(f)["summary"]
    keys = [("has_chinese", "正文夹有汉字"), ("single_line", "没有分行"), ("extra_paragraphs", "另起段落评论")]
    order = sorted(summ, key=lambda s: -summ[s]["has_chinese"] / summ[s]["n"])
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.3), sharey=True)
    rows = []
    for ax, (k, title) in zip(axes, keys):
        vals = [summ[s][k] / summ[s]["n"] for s in order]
        ax.barh(range(len(order)), vals, height=0.5, color=CAT[0], edgecolor=SURFACE)
        for i, v in enumerate(vals):
            ax.text(v + 0.012, i, "%.0f%%" % (100 * v), va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, 0.8)
        ax.set_xticks([0, 0.2, 0.4, 0.6])
        ax.set_xticklabels(["0", "20%", "40%", "60%"])
        ax.set_title(title, fontsize=9, color=INK2)
        clean(ax, "x")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([NAME[s] for s in order])
    axes[0].invert_yaxis()
    for s in order:
        rows.append([s, summ[s]["n"]] + [summ[s][k] for k, _ in keys])
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig7_format_compliance", ["strategy", "n"] + [k for k, _ in keys], rows)


if __name__ == "__main__":
    style()
    fig4()
    fig5()
    fig6()
    fig7()
