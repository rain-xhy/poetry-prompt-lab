"""Stage 2: main test set. 20 poems x 5 locked strategies x 2 models x 3 repeats = 600 calls.

Strategies come from the verified candidate_lock.json. Resumable.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402
from src.generator import generate_batch, load_corpus  # noqa: E402
from src.lock import load_lock  # noqa: E402


def main():
    lock = load_lock()
    locked = lock["selected_strategies"]
    # Added per 03 guide (S5_T1 mandatory; S3_T1 needed for the example marginal contribution).
    # Not chosen by the dev-set ranking; reported as such. The lock file itself is unchanged.
    ADDED = ["S5_T1", "S3_T1"]
    strategies = locked + [s for s in ADDED if s not in locked]
    print("lock verified, sha256=%s\nlocked=%s\nadded per guide=%s" % (lock["sha256"], locked, ADDED))
    poems = load_corpus(cfg.data_dir, "main")
    models = [m.name for m in cfg.models]
    out_dir = os.path.join(cfg.results_dir, "stage2_main", "translations")
    expected = len(poems) * len(strategies) * len(models) * cfg.n_repeats_main
    print("poems=%d strategies=%d models=%d repeats=%d -> %d calls"
          % (len(poems), len(strategies), len(models), cfg.n_repeats_main, expected))
    res = generate_batch(cfg, poems, strategies, models, cfg.n_repeats_main, out_dir)
    errs = [r for r in res if r["metadata"].get("finish_reason") == "error"]
    print("done: %d records, %d errors" % (len(res), len(errs)))


if __name__ == "__main__":
    main()
