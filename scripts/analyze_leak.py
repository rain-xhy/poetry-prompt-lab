"""Post-hoc sensitivity analysis: source text left in the translation body.

Usage: python scripts/analyze_leak.py <stage> <set>      e.g. stage2_main main

Treatments of layer 1, applied to the same scored units:
  as_scored       the pre-specified rule, the primary result
  source_as_fail  a unit also counts as failed when two or more characters of its own source phrase
                  are left untranslated in the translation body
  no_chinese      only translations whose body holds no Chinese character
The detector is crude: any two-character sequence of the source phrase found in the body counts.
These analyses were added after the results had been seen and are exploratory.
Output: results/<stage>/analysis/leak_sensitivity.csv
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402
from src.evaluator import strip_notes  # noqa: E402
from src.generator import load_corpus  # noqa: E402

CJK = re.compile(r"[一-鿿]")


def leaked(phrase, body):
    pairs = {phrase[i:i + 2] for i in range(len(phrase) - 1) if re.fullmatch(r"[一-鿿]{2}", phrase[i:i + 2])}
    return any(p in body for p in pairs)


def by_strategy(rows, field):
    """Mean per (poem, strategy) first, then over poems, as in the main analysis."""
    cell = defaultdict(list)
    for r in rows:
        cell[(r["poem"], r["strategy"])].append(r[field])
    per = defaultdict(list)
    for (_, st), v in cell.items():
        per[st].append(sum(v) / len(v))
    return {st: (sum(v) / len(v), len(v)) for st, v in per.items()}


def main(stage, which):
    poems = {p["id"]: p for p in load_corpus(cfg.data_dir, which)}
    base = os.path.join(cfg.results_dir, stage)
    trans = []
    for path in glob.glob(os.path.join(base, "scores", "*.json")):
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        with open(os.path.join(base, "translations", s["translation_id"] + ".json"), encoding="utf-8") as f:
            body = strip_notes(json.load(f)["translation"])
        units = {u["id"]: u for u in poems[s["poem_id"]]["semantic_units"]}
        valid = [d for d in s["layer1"]["details"] if d.get("valid")]
        if not valid:
            continue
        scored = sum(bool(d["pass"]) for d in valid) / len(valid)
        strict = sum(bool(d["pass"]) and not leaked(units[d["unit_id"]].get("chinese", ""), body)
                     for d in valid) / len(valid)
        trans.append({"poem": s["poem_id"], "strategy": s["strategy"], "has_cn": bool(CJK.search(body)),
                      "scored": scored, "strict": strict})

    a = by_strategy(trans, "scored")
    b = by_strategy(trans, "strict")
    c = by_strategy([r for r in trans if not r["has_cn"]], "scored")
    share = defaultdict(lambda: [0, 0])
    for r in trans:
        share[r["strategy"]][1] += 1
        share[r["strategy"]][0] += int(r["has_cn"])

    out = os.path.join(base, "analysis", "leak_sensitivity.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    order = sorted(a, key=lambda x: -a[x][0])
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "share_with_chinese", "as_scored", "source_as_fail", "no_chinese", "n_poems_no_chinese"])
        for st in order:
            w.writerow([st, round(share[st][0] / share[st][1], 3), round(a[st][0], 3), round(b[st][0], 3),
                        round(c[st][0], 3) if st in c else "", c[st][1] if st in c else 0])
    print("strategy  CN_share  as_scored  source_as_fail  no_chinese(n_poems)")
    for st in order:
        print("  %-6s  %.2f      %.3f      %.3f           %.3f (%d)" % (
            st, share[st][0] / share[st][1], a[st][0], b[st][0], c[st][0], c[st][1]))
    for name, d in (("as_scored", a), ("source_as_fail", b), ("no_chinese", c)):
        vals = [v[0] for v in d.values()]
        print("%-15s spread %.3f  rank %s" % (name, max(vals) - min(vals), sorted(d, key=lambda x: -d[x][0])))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
