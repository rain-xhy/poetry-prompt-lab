"""Scorer calibration with known answers.

Each non-adversarial semantic unit ships good_examples (should PASS) and bad_examples
(should FAIL). The scorer judges each example as if it were a translation; we report
accuracy, false-pass and false-fail rates. Usage: python scripts/calibrate_scorer.py [max_units]
Output: results/scorer_calibration.json
"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm  # noqa: E402

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402
from src.evaluator import Evaluator  # noqa: E402
from src.generator import load_corpus  # noqa: E402


def clean(ex: str) -> str:
    return re.sub(r"\s*\([^)]*[一-鿿][^)]*\)", "", ex).strip()  # drop Chinese annotations


def main(max_units: int):
    ev = Evaluator()
    items = []
    for which in ("dev", "main"):
        for p in load_corpus(cfg.data_dir, which):
            for u in p["semantic_units"]:
                if "good_examples" not in u:
                    continue
                items.append((p["id"], u, "good", u["good_examples"][0]))
                if u.get("bad_examples"):
                    items.append((p["id"], u, "bad", u["bad_examples"][0]))
    items = items[:max_units * 2]
    rows = []
    for pid, u, kind, ex in tqdm(items, desc="calibrate"):
        r = ev.check_unit(u, clean(ex))
        rows.append({"poem": pid, "unit": u["id"], "kind": kind, "example": clean(ex),
                     "scorer_pass": r["pass"], "valid": r["valid"], "reasoning": r.get("reasoning", "")})
    valid = [r for r in rows if r["valid"]]
    good = [r for r in valid if r["kind"] == "good"]
    bad = [r for r in valid if r["kind"] == "bad"]
    tp = sum(1 for r in good if r["scorer_pass"])
    tn = sum(1 for r in bad if not r["scorer_pass"])
    summary = {
        "scorer": ev.scorer_name, "n_valid": len(valid), "n_invalid": len(rows) - len(valid),
        "good_pass_rate": tp / len(good) if good else None,
        "bad_reject_rate": tn / len(bad) if bad else None,
        "accuracy": (tp + tn) / len(valid) if valid else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    os.makedirs(cfg.results_dir, exist_ok=True)
    with open(os.path.join(cfg.results_dir, "scorer_calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
