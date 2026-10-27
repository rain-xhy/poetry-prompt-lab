"""Build a blank human-labelling sheet for scorer agreement (04/05 guide: human gold standard).

Samples scored semantic units from the 8 gold-standard poems (H01-H08) in stage 2/3 results.
Output: results/audit/audit_sheet.csv with an EMPTY `human_pass` column (fill 1 / 0 by hand)
and results/audit/audit_key.json (scorer verdicts, kept apart so labelling stays blind).

Usage: python scripts/make_audit_sheet.py [n_per_poem=6]
Run only after stage 2 / 3 scoring has finished.
"""
import csv
import glob
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402
from src.evaluator import strip_notes  # noqa: E402
from src.generator import load_corpus  # noqa: E402

GOLD = {"M01", "M07", "M14", "M16", "M11", "M04", "A01", "A02"}  # H01-H08 per 06_人工金标准.md


def main(n_per_poem: int):
    poems = {}
    for which in ("main", "adversarial"):
        for p in load_corpus(cfg.data_dir, which):
            poems[p["id"]] = p

    rng = random.Random(cfg.random_seed)
    rows, key = [], {}
    for stage in ("stage2_main", "stage3_adversarial"):
        for path in sorted(glob.glob(os.path.join(cfg.results_dir, stage, "scores", "*.json"))):
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
            if s["poem_id"] in GOLD:
                rows.append((stage, s))
    if not rows:
        sys.exit("no scored translations for gold poems yet; finish scoring first")

    items = []
    by_poem = {}
    for stage, s in rows:
        by_poem.setdefault(s["poem_id"], []).append((stage, s))
    for pid, lst in by_poem.items():
        rng.shuffle(lst)
        for stage, s in lst[:n_per_poem]:
            tpath = os.path.join(cfg.results_dir, stage, "translations", s["translation_id"] + ".json")
            with open(tpath, encoding="utf-8") as f:
                trans = json.load(f)["translation"]
            for d in s["layer1"]["details"]:
                if not d.get("valid"):
                    continue
                unit = next(u for u in poems[pid]["semantic_units"] if u["id"] == d["unit_id"])
                items.append((pid, s["translation_id"], unit, strip_notes(trans), d["pass"]))
    rng.shuffle(items)

    os.makedirs(os.path.join(cfg.results_dir, "audit"), exist_ok=True)
    sheet = os.path.join(cfg.results_dir, "audit", "audit_sheet.csv")
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "original_phrase", "must_preserve", "translation", "human_pass(1/0)"])
        for i, (pid, tid, unit, trans, verdict) in enumerate(items, 1):
            code = "A%04d" % i
            focus = unit.get("chinese") or unit.get("trap_point", "")
            need = unit.get("must_preserve") or unit.get("must_detect", "")
            w.writerow([code, focus, need, trans, ""])
            key[code] = {"translation_id": tid, "poem_id": pid, "unit_id": unit["id"], "scorer_pass": bool(verdict)}
    with open(os.path.join(cfg.results_dir, "audit", "audit_key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=2)
    print("wrote %d items -> %s" % (len(items), sheet))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
