"""Blind scoring for one stage. Usage: python scripts/stage_score.py stage1_dev dev

Reads results/<stage>/translations/*.json, writes results/<stage>/scores/<id>.json.
Order is shuffled by anonymous code; the scorer never receives strategy/model/prompt.
Resumable: existing score files are skipped.
"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm  # noqa: E402

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402
from src.evaluator import Evaluator, anon_code  # noqa: E402
from src.generator import load_corpus  # noqa: E402


def main(stage: str, which: str):
    tdir = os.path.join(cfg.results_dir, stage, "translations")
    sdir = os.path.join(cfg.results_dir, stage, "scores")
    os.makedirs(sdir, exist_ok=True)
    poems = {p["id"]: p for p in load_corpus(cfg.data_dir, which)}

    recs = []
    for path in glob.glob(os.path.join(tdir, "*_r*.json")):
        with open(path, encoding="utf-8") as f:
            recs.append(json.load(f))
    recs.sort(key=lambda r: anon_code(r["id"]))  # blind, strategy-independent order
    print("%s: %d translations to score" % (stage, len(recs)))

    ev = Evaluator()
    skipped_err = 0
    for rec in tqdm(recs, desc="scoring " + stage):
        out = os.path.join(sdir, rec["id"] + ".json")
        if os.path.exists(out):
            continue
        if not rec["translation"].strip():
            skipped_err += 1
            continue
        res = ev.evaluate(rec, poems[rec["poem_id"]])
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    print("done; empty translations skipped: %d" % skipped_err)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
