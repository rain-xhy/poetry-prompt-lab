"""Merge the Week-1 semantic-unit JSON files into one corpus file.

Output: data/poems/corpus.json  (list of poem records)
Also prints an overlap check between dev / main / adversarial sets.
"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.abspath(os.path.join(ROOT, "..", "..", "材料准备"))
OUT = os.path.join(ROOT, "data", "poems", "corpus.json")

SKIP_KEYS = {"metadata", "comment", "note", "summary", "main_test"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_records(d):
    for k, v in d.items():
        if k == "main_test":
            yield from v.items()
        elif k not in SKIP_KEYS and isinstance(v, dict):
            yield k, v


def main():
    records = []
    for path in sorted(glob.glob(os.path.join(SRC, "05_*.json"))):
        if "v1_废弃" in path or (not "v2" in path and "part4" in path and "开发集.json" in path):
            print(f"skip old dev: {os.path.basename(path)}")
            continue
        for pid, v in iter_records(load(path)):
            prefix = pid[0]
            rec = {
                "id": pid.split("_")[0],
                "key": pid,
                "set": {"M": "main", "D": "dev", "A": "adversarial"}[prefix],
                "poem_name": v.get("poem_name", ""),
                "author": v.get("author", ""),
                "trap_type": v.get("trap_type", ""),
                "text": v.get("text") or v.get("version_A", ""),
                "text_b": v.get("version_B"),
                "correct_understanding": v.get("correct_understanding"),
                "wrong_understanding": v.get("wrong_understanding"),
                "note": v.get("note"),
                "semantic_units": v.get("semantic_units", []),
            }
            records.append(rec)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    by_set = {}
    for r in records:
        by_set.setdefault(r["set"], []).append(r)
    for s, rs in by_set.items():
        units = sum(len(r["semantic_units"]) for r in rs)
        print(f"{s}: {len(rs)} poems, {units} units")

    # overlap check: dev vs main by poem_name
    main_names = {r["poem_name"] for r in by_set.get("main", [])}
    print("\nDev poems also present in main test set (by poem_name):")
    for r in by_set.get("dev", []):
        hit = r["poem_name"] in main_names
        print(f"  {r['id']} {r['poem_name']}: {'OVERLAP' if hit else 'distinct'}  note={r['note']}")
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()
