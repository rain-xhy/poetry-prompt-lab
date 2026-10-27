"""Format check (04 guide, stage 2 step 4). Usage: python scripts/format_check.py stage1_dev

Rules are mechanical and independent of any scorer:
  - non-empty output
  - no Chinese characters in the translation body
  - at least 2 non-empty lines (poems must be line-broken)
  - at most 1 bracketed note / 'Note:' line for T1, 2 for T3, 0 for T2 / baseline strategies are only reported
  - no extra commentary paragraph after a blank line
Violations are recorded, never used to regenerate.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402

CJK = re.compile(r"[一-鿿]")


def check(text: str) -> dict:
    body = re.sub(r"\[[^\]]*\]", "", text)
    body = re.sub(r"(?im)^\s*note:.*$", "", body).strip()
    lines = [l for l in body.splitlines() if l.strip()]
    notes = len(re.findall(r"\[[^\]]*\]", text)) + len(re.findall(r"(?im)^\s*note:", text))
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    return {
        "empty": not text.strip(),
        "has_chinese": bool(CJK.search(body)),
        "n_lines": len(lines),
        "single_line": len(lines) < 2,
        "n_notes": notes,
        "extra_paragraphs": len(paragraphs) > 1,
    }


def main(stage: str):
    tdir = os.path.join(cfg.results_dir, stage, "translations")
    rows, summary = [], defaultdict(lambda: defaultdict(int))
    for p in sorted(glob.glob(os.path.join(tdir, "*_r*.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        c = check(r["translation"])
        c.update(id=r["id"], strategy=r["strategy"], model=r["model"])
        rows.append(c)
        s = summary[r["strategy"]]
        s["n"] += 1
        for k in ("empty", "has_chinese", "single_line", "extra_paragraphs"):
            s[k] += int(c[k])
        s["with_notes"] += int(c["n_notes"] > 0)
    out_dir = os.path.join(cfg.results_dir, stage, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "format_check.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print("%d translations checked" % len(rows))
    for st, s in sorted(summary.items()):
        print("%-8s n=%d empty=%d chinese=%d single_line=%d extra_para=%d with_notes=%d"
              % (st, s["n"], s["empty"], s["has_chinese"], s["single_line"], s["extra_paragraphs"], s["with_notes"]))


if __name__ == "__main__":
    main(sys.argv[1])
