"""Stage 2b: component-isolation strategies S2_T1 and S4_T1 on the main set.

20 poems x 2 strategies x 2 models x 3 repeats = 240 calls.
Added after the stage-1 review so that RQ1.2 (Chinese analysis) and RQ1.3 (pitfall warnings)
can be separated from the full S5 combination. These two were NOT screened on the dev set;
they are appended per the research questions and must be reported as such.
Writes into the same stage2_main/translations directory (ids do not collide). Resumable.
Run only after stage2_generate.py has finished, so the two jobs do not share the GPU.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402
from src.generator import generate_batch, load_corpus  # noqa: E402
from src.lock import load_lock  # noqa: E402

STRATEGIES = ["S2_T1", "S4_T1"]


def main():
    load_lock()
    poems = load_corpus(cfg.data_dir, "main")
    models = [m.name for m in cfg.models]
    out_dir = os.path.join(cfg.results_dir, "stage2_main", "translations")
    expected = len(poems) * len(STRATEGIES) * len(models) * cfg.n_repeats_main
    print("poems=%d strategies=%s models=%d repeats=%d -> %d calls"
          % (len(poems), STRATEGIES, len(models), cfg.n_repeats_main, expected))
    res = generate_batch(cfg, poems, STRATEGIES, models, cfg.n_repeats_main, out_dir)
    errs = [r for r in res if r["metadata"].get("finish_reason") == "error"]
    print("done: %d records, %d errors" % (len(res), len(errs)))


if __name__ == "__main__":
    main()
