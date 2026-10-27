"""Expert-facing audit sheet that shows the SAME criteria the AI judge received.

The earlier sheets showed `must_preserve`. The judge was actually given `semantic_content`
(`must_detect` for adversarial units) plus the full pass conditions. This sheet shows those, so a
disagreement reflects the judge's reliability and not a difference in criteria.

Inputs: results/audit/audit_sheet.csv (item ids and translations with notes already stripped) and
results/audit/audit_key.json (only poem id and unit id are read, scorer verdicts are never copied).
Output: results/audit/audit_sheet_expert.csv
"""
import csv
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402
from src.evaluator import _conds  # noqa: E402
from src.generator import load_corpus  # noqa: E402
from src.prompts import format_poem  # noqa: E402

HEADERS = [
    "item(题号)",
    "original_poem(原诗全文)",
    "original_phrase(本行检查的原句)",
    "must_preserve(需要保住的意思)",
    "pass_conditions(判定条件，须全部满足)",
    "translation(模型译文)",
    "human_pass(1/0)(请填写)",
    "note(选填)",
]


def one_line(text):
    return " / ".join(l.strip() for l in text.splitlines() if l.strip())


def conds_text(unit):
    cs = [c for c in _conds(unit) if c]
    return "  ".join("%d) %s" % (i, c) for i, c in enumerate(cs, 1))


def main():
    base = os.path.join(cfg.results_dir, "audit")
    with open(os.path.join(base, "audit_key.json"), encoding="utf-8") as f:
        key = json.load(f)
    poems = {}
    for which in ("main", "adversarial"):
        for p in load_corpus(cfg.data_dir, which):
            poems[p["id"]] = p
    with open(os.path.join(base, "audit_sheet.csv"), encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))[1:]

    out = os.path.join(base, "audit_sheet_expert.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADERS)
        for item, _phrase, _need, trans, _ in rows:
            k = key[item]
            poem = poems[k["poem_id"]]
            unit = next(u for u in poem["semantic_units"] if u["id"] == k["unit_id"])
            phrase = unit.get("chinese") or unit.get("trap_point", "")
            content = unit.get("semantic_content") or unit.get("must_detect", "")
            w.writerow([item, one_line(format_poem(poem["text"])), phrase, content,
                        conds_text(unit), one_line(trans), "", ""])
    print("wrote %d rows -> %s" % (len(rows), out))


if __name__ == "__main__":
    main()
