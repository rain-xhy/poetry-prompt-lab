"""Multi-check of the scorer (04 guide, stage 2 step 4; 05 guide, 3.3).

Usage: python scripts/scorer_checks.py <stage> <set> [n=30] [second_model=gemma2:2b]

1. Repeat scoring: n random translations are scored again with a fresh seed. We report the
   mean absolute difference of the per-translation semantic accuracy and the share of units
   whose verdict is identical.
2. Second-model check: 20 random translations are scored by a different model family on
   layer 1 and the unit-level agreement with the main scorer is reported.
Writes results/<stage>/analysis/scorer_checks.json. Run only when the GPU is free.
"""
import glob
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm  # noqa: E402

from src.config import DEFAULT_CONFIG as cfg, ModelConfig  # noqa: E402
from src.evaluator import Evaluator, strip_notes  # noqa: E402
from src.generator import load_corpus  # noqa: E402


def load_pair(stage, path):
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    with open(os.path.join(cfg.results_dir, stage, "translations", s["translation_id"] + ".json"),
              encoding="utf-8") as f:
        rec = json.load(f)
    return s, rec


def unit_agreement(a_details, b_details):
    a = {d["unit_id"]: d["pass"] for d in a_details if d.get("valid")}
    b = {d["unit_id"]: d["pass"] for d in b_details if d.get("valid")}
    ids = sorted(set(a) & set(b))
    return [(a[i], b[i]) for i in ids]


def summarise(pairs, accs):
    n = len(pairs)
    same = sum(1 for x, y in pairs if x == y)
    return {"n_units": n, "unit_agreement": same / n if n else None,
            "mean_abs_diff_accuracy": sum(accs) / len(accs) if accs else None}


def main(stage, which, n=30, second="gemma2:2b"):
    poems = {p["id"]: p for p in load_corpus(cfg.data_dir, which)}
    paths = sorted(glob.glob(os.path.join(cfg.results_dir, stage, "scores", "*.json")))
    rng = random.Random(cfg.random_seed)
    out = {}

    main_ev = Evaluator()
    sample = rng.sample(paths, min(n, len(paths)))
    pairs, accs = [], []
    for p in tqdm(sample, desc="repeat scoring"):
        s, rec = load_pair(stage, p)
        again = main_ev.evaluate(rec, poems[s["poem_id"]], seed=7919)
        pairs += unit_agreement(s["layer1"]["details"], again["layer1"]["details"])
        if s["layer1"]["accuracy"] is not None and again["layer1"]["accuracy"] is not None:
            accs.append(abs(s["layer1"]["accuracy"] - again["layer1"]["accuracy"]))
    out["repeat_scoring"] = dict(summarise(pairs, accs), n_translations=len(sample), seed=7919)

    cfg2 = ModelConfig("Second", second, think=False, temperature=0.0, top_p=1.0, max_tokens=400)
    ev2 = Evaluator(cfg2)
    sample2 = rng.sample(paths, min(20, len(paths)))
    pairs, accs = [], []
    for p in tqdm(sample2, desc="second model " + second):
        s, rec = load_pair(stage, p)
        other = ev2.evaluate(rec, poems[s["poem_id"]], seed=0)
        pairs += unit_agreement(s["layer1"]["details"], other["layer1"]["details"])
        if s["layer1"]["accuracy"] is not None and other["layer1"]["accuracy"] is not None:
            accs.append(abs(s["layer1"]["accuracy"] - other["layer1"]["accuracy"]))
    tp = sum(1 for a, b in pairs if a and b)
    out["second_model"] = dict(summarise(pairs, accs), model=second, n_translations=len(sample2),
                               both_pass=tp, main_pass_only=sum(1 for a, b in pairs if a and not b),
                               second_pass_only=sum(1 for a, b in pairs if b and not a))

    d = os.path.join(cfg.results_dir, stage, "analysis")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "scorer_checks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 30, a[4] if len(a) > 4 else "gemma2:2b")
