"""Smoke test: print the 9 prompts for one dev poem, then run 2 real calls per model."""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_CONFIG as CFG
from src.generator import generate_batch, load_corpus
from src.prompts import STRATEGIES, build_prompt

dev = load_corpus(CFG.data_dir, "dev")
adv = load_corpus(CFG.data_dir, "adversarial")
print("dev:", len(dev), "adversarial (A/B split):", len(adv))

for s in STRATEGIES:
    p = build_prompt(dev[0], s)
    print("=" * 30, s, "chars=%d" % len(p))
print()
print(build_prompt(dev[0], "S5_T1"))
print()

out = os.path.join(CFG.results_dir, "_smoke")
res = generate_batch(CFG, dev[:1], ["S0", "S5_T1"], [m.name for m in CFG.models], 1, out)
for r in res:
    print("-" * 20, r["id"], r["metadata"].get("finish_reason"), r["metadata"].get("latency_ms"))
    print(r["translation"])
