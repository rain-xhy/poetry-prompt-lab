"""Stage 1: dev-set generation. 6 poems x 9 strategies x 2 models x 2 repeats = 216 calls.

Resumable: existing result files are skipped.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402
from src.generator import generate_batch, load_corpus  # noqa: E402
from src.prompts import STRATEGIES  # noqa: E402


def main():
    poems = load_corpus(cfg.data_dir, "dev")
    strategies = list(STRATEGIES.keys())
    models = [m.name for m in cfg.models]
    out_dir = os.path.join(cfg.results_dir, "stage1_dev", "translations")
    expected = len(poems) * len(strategies) * len(models) * cfg.n_repeats_dev
    print("poems=%d strategies=%d models=%d repeats=%d -> %d calls"
          % (len(poems), len(strategies), len(models), cfg.n_repeats_dev, expected))
    res = generate_batch(cfg, poems, strategies, models, cfg.n_repeats_dev, out_dir)
    errs = [r for r in res if r["metadata"].get("finish_reason") == "error"]
    print("done: %d records, %d errors" % (len(res), len(errs)))


if __name__ == "__main__":
    main()
