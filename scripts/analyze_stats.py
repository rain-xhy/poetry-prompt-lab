"""Paired statistics per 06 guide. Usage: python scripts/analyze_stats.py <stage> [metric]

metric: accuracy (layer 1, default) or quality (layer 2 overall).
Unit of analysis = poem (repeats and models are averaged within poem first), so the
translations are never treated as independent poems.

Outputs (results/<stage>/analysis/):
  descriptive_<metric>.csv   per-strategy mean + bootstrap 95% CI (poem resampling)
  paired_<metric>.csv        all strategy pairs: mean diff, bootstrap CI, paired t, Cohen's d,
                             Holm- and BH-corrected p
  by_trap_<metric>.csv       strategy x trap type means
  by_model_<metric>.csv      strategy x model means (cross-model stability)
  mixed_<metric>.txt         mixed model: score ~ strategy * trap_type + model, random poem intercept
"""
import itertools
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage1_analyze import load_rows  # noqa: E402

N_BOOT = 2000
SEED = 20261026


def poem_matrix(df, metric):
    """poems x strategies matrix of poem-level means."""
    pm = df.groupby(["poem", "strategy"])[metric].mean().unstack()
    return pm.dropna(axis=0, how="any")


def boot_ci(x, rng):
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main(stage, metric="accuracy"):
    df = load_rows(stage)
    df = df.dropna(subset=[metric])
    out = os.path.join(cfg.results_dir, stage, "analysis")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    pm = poem_matrix(df, metric)
    print("%s / %s: %d poems x %d strategies (poems with all strategies)" % (stage, metric, *pm.shape))

    desc = []
    for s in pm.columns:
        x = pm[s].values
        lo, hi = boot_ci(x, rng)
        desc.append({"strategy": s, "mean": x.mean(), "sd_across_poems": x.std(ddof=1),
                     "ci_lo": lo, "ci_hi": hi, "n_poems": len(x)})
    desc = pd.DataFrame(desc).sort_values("mean", ascending=False)
    desc.to_csv(os.path.join(out, "descriptive_%s.csv" % metric), index=False, encoding="utf-8-sig")
    print(desc.round(3).to_string(index=False))

    rows = []
    for a, b in itertools.combinations(pm.columns, 2):
        d = (pm[a] - pm[b]).values
        lo, hi = boot_ci(d, rng)
        if np.allclose(d, d[0]):
            t, p, cd = float("nan"), float("nan"), float("nan")
        else:
            t, p = stats.ttest_rel(pm[a], pm[b])
            cd = d.mean() / d.std(ddof=1)
        rows.append({"a": a, "b": b, "mean_diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                     "t": t, "p": p, "cohen_d": cd, "n_poems": len(d)})
    pr = pd.DataFrame(rows)
    ok = pr.p.notna()
    for name, method in (("p_holm", "holm"), ("p_bh", "fdr_bh")):
        pr[name] = np.nan
        pr.loc[ok, name] = multipletests(pr.loc[ok, "p"], method=method)[1]
    pr.to_csv(os.path.join(out, "paired_%s.csv" % metric), index=False, encoding="utf-8-sig")
    print("\npairs with Holm p < .05:")
    print(pr[pr.p_holm < 0.05].round(4).to_string(index=False) or "  none")
    print("(%d pairs tested; exploratory, small n of poems)" % ok.sum())

    pt = df.groupby(["poem", "trap_type", "strategy"])[metric].mean().reset_index() \
        if "trap_type" in df.columns else None
    if pt is not None:
        pt.groupby(["trap_type", "strategy"])[metric].mean().unstack().to_csv(
            os.path.join(out, "by_trap_%s.csv" % metric), encoding="utf-8-sig")
    df.groupby(["strategy", "model"])[metric].mean().unstack().to_csv(
        os.path.join(out, "by_model_%s.csv" % metric), encoding="utf-8-sig")

    if pt is not None:
        try:
            import statsmodels.formula.api as smf
            m = df.merge(pt[["poem", "trap_type"]].drop_duplicates(), on="poem", suffixes=("", "_y"))
            res = smf.mixedlm("%s ~ C(strategy) + C(trap_type) + C(model)" % metric, m, groups=m["poem"]).fit()
            with open(os.path.join(out, "mixed_%s.txt" % metric), "w", encoding="utf-8") as f:
                f.write(str(res.summary()))
            print("mixed model written")
        except Exception as e:  # small designs can be singular; report, do not hide
            print("mixed model failed:", e)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "accuracy")
