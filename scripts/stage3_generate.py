"""Stage 3: adversarial set. Fixed strategies S0 / S1_T1 / S5_T1 (04 guide, 5.2).

10 adversarial items x 3 strategies x 2 models x 3 repeats. A07 is an A/B pair and is
split into A07A / A07B, so 11 texts are generated: 11 x 3 x 2 x 3 = 198 calls
(the guide's 180 counts A07 once). Resumable.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402
from src.generator import generate_batch, load_corpus  # noqa: E402
from src.lock import load_lock  # noqa: E402

STRATEGIES = ["S0", "S1_T1", "S5_T1"]


def main():
    load_lock()  # adversarial stage must also run after a verified lock
    poems = load_corpus(cfg.data_dir, "adversarial")
    models = [m.name for m in cfg.models]
    out_dir = os.path.join(cfg.results_dir, "stage3_adversarial", "translations")
    expected = len(poems) * len(STRATEGIES) * len(models) * cfg.n_repeats_adversarial
    print("texts=%d strategies=%d models=%d repeats=%d -> %d calls"
          % (len(poems), len(STRATEGIES), len(models), cfg.n_repeats_adversarial, expected))
    res = generate_batch(cfg, poems, STRATEGIES, models, cfg.n_repeats_adversarial, out_dir)
    errs = [r for r in res if r["metadata"].get("finish_reason") == "error"]
    print("done: %d records, %d errors" % (len(res), len(errs)))


if __name__ == "__main__":
    main()
