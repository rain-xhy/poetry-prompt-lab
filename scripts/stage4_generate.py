"""Stage 4: translation-path generation (04 guide, section 6).

Deviation from the guide (documented): instead of 10 pre-chosen cases x 2 models = 20 calls,
all 20 main-set poems are run with S5_T1 + understanding-then-translation JSON prompt,
1 repeat, 2 models = 40 calls. The 10 report cases are picked afterwards from these,
so case selection is not made before seeing results of a single run.
Requires verified lock. Resumable.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402
from src.generator import generate_batch, load_corpus  # noqa: E402
from src.lock import load_lock  # noqa: E402


def main():
    load_lock()
    poems = load_corpus(cfg.data_dir, "main")
    models = [m.name for m in cfg.models]
    out_dir = os.path.join(cfg.results_dir, "stage4_path", "translations")
    print("poems=%d models=%d -> %d calls" % (len(poems), len(models), len(poems) * len(models)))
    res = generate_batch(cfg, poems, ["S5_T1"], models, 1, out_dir, path_mode=True)
    errs = [r for r in res if r["metadata"].get("finish_reason") == "error"]
    print("done: %d records, %d errors" % (len(res), len(errs)))


if __name__ == "__main__":
    main()
