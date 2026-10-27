"""Compare scorer verdicts with the human labels in the returned audit sheet.

Usage: python scripts/audit_agreement.py [sheet_file=audit_sheet_expert.csv]

The sheet is looked up in results/audit/. The file may be UTF-8 or GBK (Excel "CSV" default).
Reports agreement, Cohen's kappa, the confusion matrix (lenient = scorer pass / human fail,
strict = scorer fail / human pass) and agreement per poem. Rows with an empty label are
skipped, rows with a label other than 0 or 1 are counted and reported.
"""
import csv
import io
import json
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402


def read_sheet(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return list(csv.DictReader(io.StringIO(raw.decode(enc))))
        except UnicodeDecodeError:
            continue
    raise SystemExit("cannot decode %s" % path)


def column(row, prefix):
    for k in row:
        if k and k.strip().startswith(prefix):
            return k
    raise SystemExit("column starting with %r not found; headers: %s" % (prefix, list(row)))


def main(sheet_name):
    base = os.path.join(cfg.results_dir, "audit")
    with open(os.path.join(base, "audit_key.json"), encoding="utf-8") as f:
        key = json.load(f)
    rows = read_sheet(os.path.join(base, sheet_name))
    item_col, label_col = column(rows[0], "item"), column(rows[0], "human_pass")

    human, scorer, poem, bad = [], [], [], 0
    for r in rows:
        v = (r.get(label_col) or "").strip()
        if v == "":
            continue
        if v not in ("0", "1"):
            bad += 1
            continue
        k = key[r[item_col].strip()]
        human.append(int(v))
        scorer.append(int(k["scorer_pass"]))
        poem.append(k["poem_id"])
    n = len(human)
    if n == 0:
        sys.exit("no labelled rows yet")

    tp = sum(1 for h, s in zip(human, scorer) if h and s)
    tn = sum(1 for h, s in zip(human, scorer) if not h and not s)
    fp = sum(1 for h, s in zip(human, scorer) if not h and s)
    fn = sum(1 for h, s in zip(human, scorer) if h and not s)
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")

    per = defaultdict(lambda: [0, 0])
    for h, s, p in zip(human, scorer, poem):
        per[p][0] += int(h == s)
        per[p][1] += 1
    out = {
        "sheet": sheet_name, "n_labelled": n, "n_rows": len(rows), "n_invalid_label": bad,
        "agreement": round(po, 3), "cohen_kappa": round(kappa, 3),
        "human_pass_rate": round(sum(human) / n, 3), "scorer_pass_rate": round(sum(scorer) / n, 3),
        "tp": tp, "tn": tn, "fp_lenient": fp, "fn_strict": fn,
        "by_poem": {p: {"n": c[1], "agreement": round(c[0] / c[1], 3)} for p, c in sorted(per.items())},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(os.path.join(base, "agreement.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "audit_sheet_expert.csv")
