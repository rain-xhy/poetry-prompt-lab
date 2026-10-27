"""Rebuild the audit sheet in a form that is easier to fill by hand.

Reads results/audit/audit_sheet.csv (unchanged) and audit_key.json (only the poem id is used,
scorer verdicts are never copied), and writes results/audit/audit_sheet_easy.csv with columns
  item, original_poem, original_phrase, must_preserve, translation, human_pass(1/0)
where original_poem is the full Chinese poem being translated and translation is flattened to one
line with " / " between lines. Item ids are identical to the original sheet, so the key still applies.
"""
import csv
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402
from src.generator import load_corpus  # noqa: E402
from src.prompts import format_poem  # noqa: E402


def one_line(text):
    return " / ".join(l.strip() for l in text.splitlines() if l.strip())


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

    out = os.path.join(base, "audit_sheet_easy.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "original_poem", "original_phrase", "must_preserve", "translation", "human_pass(1/0)"])
        for item, phrase, need, trans, _ in rows:
            poem = poems[key[item]["poem_id"]]
            w.writerow([item, one_line(format_poem(poem["text"])), phrase, need, one_line(trans), ""])
    print("wrote %d rows -> %s" % (len(rows), out))


if __name__ == "__main__":
    main()
