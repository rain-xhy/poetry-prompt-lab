"""Add layer 3 (purpose suitability) and the anchored layer 2 to existing score files.

Usage: python scripts/rescore_layers.py <stage> <set>   e.g. stage1_dev dev
Layer 1 is never touched, so the stage-1 candidate lock stays valid. The earlier layer 2
result is kept under layer2_v1. Resumable: files that already hold layer3 are skipped.
Run only when the GPU is free.
"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm  # noqa: E402

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402
from src.evaluator import Evaluator, strip_notes  # noqa: E402
from src.generator import load_corpus  # noqa: E402
from src.prompts import build_semantic_analysis  # noqa: E402


def main(stage, which):
    poems = {p["id"]: p for p in load_corpus(cfg.data_dir, which)}
    ev = Evaluator()
    paths = sorted(glob.glob(os.path.join(cfg.results_dir, stage, "scores", "*.json")))
    for p in tqdm(paths, desc="rescore " + stage):
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        if "layer3" in s:
            continue
        with open(os.path.join(cfg.results_dir, stage, "translations", s["translation_id"] + ".json"),
                  encoding="utf-8") as f:
            rec = json.load(f)
        poem = poems[s["poem_id"]]
        text = strip_notes(rec["translation"])
        s["layer2_v1"] = s.get("layer2")
        s["layer2"] = ev.score_quality(poem["text"], text, 0, build_semantic_analysis(poem))
        s["layer3"] = ev.score_suitability(poem["text"], text, 0)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
