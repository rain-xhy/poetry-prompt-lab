"""Extra analyses required by the 06 guide, CPU only.

Usage: python scripts/analyze_extra.py <stage> [metric=accuracy]

Outputs (results/<stage>/analysis/):
  anova_<metric>.csv          two-way ANOVA on poem-level means: strategy x trap_type
  interaction_<metric>.txt    mixed model with the strategy x trap_type interaction (poem random intercept)
  sens_threshold_<metric>.csv share of translations above 0.25 / 0.5 / 0.8 accuracy, per strategy
  sens_outliers_<metric>.csv  strategy means after dropping the 2 easiest / 2 hardest poems
  composite.csv               four-layer weighted score per strategy (50/30/10/10)

Layer 4 for the adversarial stage is the trap-detection rate (= layer 1 accuracy on adversarial
units); for other stages layer 4 does not apply and its weight is renormalised away.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402
from src.evaluator import composite  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage1_analyze import load_rows  # noqa: E402


def main(stage, metric="accuracy"):
    df = load_rows(stage).dropna(subset=[metric])
    out = os.path.join(cfg.results_dir, stage, "analysis")
    os.makedirs(out, exist_ok=True)

    pm = df.groupby(["poem", "trap_type", "strategy"])[metric].mean().reset_index()

    # two-way ANOVA with interaction (poem-level means, so n is poems, not translations)
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        fit = smf.ols("%s ~ C(strategy) * C(trap_type)" % metric, pm).fit()
        table = sm.stats.anova_lm(fit, typ=2)
        table.to_csv(os.path.join(out, "anova_%s.csv" % metric), encoding="utf-8-sig")
        print(table.round(4).to_string())
    except Exception as e:
        print("ANOVA failed:", e)

    # mixed model including the interaction, as the 06 guide specifies
    try:
        import statsmodels.formula.api as smf
        m = smf.mixedlm("%s ~ C(strategy) * C(trap_type) + C(model)" % metric, df, groups=df["poem"]).fit()
        with open(os.path.join(out, "interaction_%s.txt" % metric), "w", encoding="utf-8") as f:
            f.write(str(m.summary()))
        print("interaction mixed model written")
    except Exception as e:
        print("interaction mixed model failed (design may be singular):", e)

    # sensitivity 1: threshold on per-translation accuracy
    thr = {}
    for name, t in (("lenient_0.25", 0.25), ("moderate_0.5", 0.5), ("strict_0.8", 0.8)):
        thr[name] = (df[metric] >= t).groupby(df["strategy"]).mean()
    pd.DataFrame(thr).to_csv(os.path.join(out, "sens_threshold_%s.csv" % metric), encoding="utf-8-sig")

    # sensitivity 2: drop extreme poems
    diff = pm.groupby("poem")[metric].mean().sort_values()
    easy, hard = list(diff.index[-2:]), list(diff.index[:2])
    cols = {}
    for name, drop in (("full", []), ("no_easy", easy), ("no_hard", hard), ("no_extremes", easy + hard)):
        cols[name] = pm[~pm.poem.isin(drop)].groupby("strategy")[metric].mean()
    pd.DataFrame(cols).to_csv(os.path.join(out, "sens_outliers_%s.csv" % metric), encoding="utf-8-sig")
    print("dropped easy=%s hard=%s" % (easy, hard))

    # four-layer composite per strategy
    per = df.groupby("strategy").agg(l1=("accuracy", "mean"), l2=("quality", "mean"),
                                     l3a=("academic", "mean"), l3p=("poetic", "mean"))
    rows = []
    for s, r in per.iterrows():
        l3 = np.nanmean([r.l3a, r.l3p]) if not (np.isnan(r.l3a) and np.isnan(r.l3p)) else None
        l4 = r.l1 if "adversarial" in stage else None
        rows.append({"strategy": s, "layer1": r.l1, "layer2": r.l2, "layer3": l3, "layer4": l4,
                     "composite": composite(r.l1, r.l2, l3, l4)})
    comp = pd.DataFrame(rows).sort_values("composite", ascending=False)
    comp.to_csv(os.path.join(out, "composite.csv"), index=False, encoding="utf-8-sig")
    print(comp.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "accuracy")
