"""Shared style and helpers for the paper figures.

Dataviz method: categorical hues in fixed order (slots 1 to 3 passed the all-pairs validator; aqua is
below 3:1 on the light surface, so every chart that uses it carries visible labels and a CSV table view),
one-hue sequential and ordinal ramps, thin marks, a surface gap between touching fills, a hairline solid
grid on the value axis only, and text in ink tokens, never in a series colour.
Each figure also writes its numbers to a CSV beside the PNG.
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figures")

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
MARK = ["o", "s", "D"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
TRAP = ["情感反差", "拟人意象", "否定逻辑", "时态叙事"]
MODEL = {"Qwen25": "Qwen2.5 3B", "Qwen3": "Qwen3 1.7B"}
NAME = {
    "S0": "基线", "S1_T1": "通用指导·译介", "S1_T2": "通用指导·朗读", "S1_T3": "通用指导·教学",
    "S2_T1": "中文解读·译介", "S3_T1": "边界示例·译介", "S4_T1": "错误警示·译介",
    "S5_T1": "层次化组合·译介", "S5_T2": "层次化组合·朗读", "S5_T3": "层次化组合·教学",
    "S3_only": "单独边界示例",
}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    })


def clean(ax, grid="x"):
    """Recessive axes: no top or right spine, a hairline solid grid on the value axis only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.6, linestyle="-")
    ax.tick_params(length=0, labelsize=8.5)


def text_on(color):
    """White or near-black text, whichever contrasts more with the given fill."""
    rgb = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(color)]
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "#ffffff" if 1.05 / (lum + 0.05) >= (lum + 0.05) / 0.053 else INK


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(OUT, name + ".csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("saved %s (%d table rows)" % (name, len(rows)))
