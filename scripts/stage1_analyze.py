"""Stage 1 analysis + candidate lock (rule per 04 guide, step 5).

Rule (fixed here before looking at results):
  - S0 is always kept as baseline.
  - Remaining strategies are ranked by mean semantic accuracy (layer 1), ties broken by
    mean quality overall (layer 2), then by shorter mean input tokens.
  - The top 4 are kept. Result is written to results/candidate_lock.json with a SHA-256.
"""
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402

BASELINE = "S0"
N_KEEP = 4


def load_rows(stage):
    base = os.path.join(cfg.results_dir, stage)
    meta = {}
    for p in glob.glob(os.path.join(base, "translations", "*_r*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        meta[r["id"]] = r
    rows = []
    for p in glob.glob(os.path.join(base, "scores", "*.json")):
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        t = meta[s["translation_id"]]
        l1, l2, l3 = s["layer1"], s["layer2"], s.get("layer3") or {}
        rows.append({
            "id": s["translation_id"], "poem": s["poem_id"], "strategy": s["strategy"],
            "model": s["model"], "repeat": s["repeat"], "trap_type": t.get("trap_type", ""),
            "accuracy": l1["accuracy"], "valid_units": l1["valid_units"], "total_units": l1["total_units"],
            "quality": l2.get("overall") if l2.get("valid") else None,
            "academic": l3.get("academic_translation") if l3.get("valid") else None,
            "poetic": l3.get("poetic_reading") if l3.get("valid") else None,
            "input_tokens": t["metadata"].get("input_tokens", 0),
            "n_chars": len(t["translation"]),
        })
    return pd.DataFrame(rows)


def main():
    df = load_rows("stage1_dev")
    out_dir = os.path.join(cfg.results_dir, "stage1_dev", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "dev_scores.csv"), index=False, encoding="utf-8-sig")

    print("scored: %d  (invalid-unit rows: %d, invalid-quality rows: %d)" % (
        len(df), int((df.valid_units < df.total_units).sum()), int(df.quality.isna().sum())))

    # poem-level mean first (avoid treating repeats as independent poems), then strategy mean
    per_poem = df.groupby(["strategy", "poem"]).agg(
        accuracy=("accuracy", "mean"), quality=("quality", "mean"), tokens=("input_tokens", "mean")).reset_index()
    summ = per_poem.groupby("strategy").agg(
        accuracy=("accuracy", "mean"), quality=("quality", "mean"), tokens=("tokens", "mean")).reset_index()
    by_model = df.groupby(["strategy", "model"]).accuracy.mean().unstack()
    summ = summ.merge(by_model.reset_index(), on="strategy")
    summ = summ.sort_values(["accuracy", "quality", "tokens"], ascending=[False, False, True])
    print(summ.round(3).to_string(index=False))
    summ.to_csv(os.path.join(out_dir, "dev_strategy_summary.csv"), index=False, encoding="utf-8-sig")

    others = summ[summ.strategy != BASELINE]
    selected = [BASELINE] + others.strategy.head(N_KEEP).tolist()

    decision = {
        "timestamp": datetime.now().isoformat(),
        "dev_results": {r.strategy: round(float(r.accuracy), 4) for r in summ.itertuples()},
        "dev_quality": {r.strategy: round(float(r.quality), 4) for r in summ.itertuples()},
        "selected_strategies": selected,
        "selection_rule": "S0 baseline always kept; top %d others by mean semantic accuracy "
                          "(ties: quality overall, then fewer input tokens)" % N_KEEP,
        "n_translations_scored": int(len(df)),
        "scorer": "qwen3:4b (temperature=0, think off; calibrated acc 0.925 on rubric good/bad examples)",
    }
    decision["sha256"] = hashlib.sha256(json.dumps(decision, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    path = os.path.join(cfg.results_dir, "candidate_lock.json")
    if os.path.exists(path):
        print("candidate_lock.json already exists; NOT overwritten (lock is immutable).")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)
    print("locked:", selected, "sha256", decision["sha256"])


if __name__ == "__main__":
    main()
