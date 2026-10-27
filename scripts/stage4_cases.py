"""Re-select the stage 4 report cases from error_attribution.csv, no model call.

Usage: python scripts/stage4_cases.py

The first run of stage4_analyze.py used a stricter rule, a case counted as a success only if all
four units succeeded. That gave 3 successes, so only 8 cases were chosen. The rule is now the
majority attribution over the four units. The earlier list is kept as selected_cases.allunits_rule.json.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

from stage4_analyze import select_cases  # noqa: E402
from src.config import DEFAULT_CONFIG as cfg  # noqa: E402


def main():
    out = os.path.join(cfg.results_dir, "stage4_path", "analysis")
    df = pd.read_csv(os.path.join(out, "error_attribution.csv"))
    df["parsed"] = df.parsed.astype(str) == "True"
    cases = select_cases(df)
    target = os.path.join(out, "selected_cases.json")
    backup = os.path.join(out, "selected_cases.allunits_rule.json")
    if os.path.exists(target) and not os.path.exists(backup):
        shutil.copy(target, backup)
    with open(target, "w", encoding="utf-8") as f:
        json.dump([{"id": i, "class": c} for i, c in cases], f, ensure_ascii=False, indent=2)
    info = df[df.parsed].drop_duplicates("id").set_index("id")
    print(len(cases), "cases")
    for i, c in cases:
        print(" ", c, i, info.loc[i, "trap_type"])


if __name__ == "__main__":
    main()
