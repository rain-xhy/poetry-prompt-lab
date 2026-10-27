"""Stage 4 analysis: error attribution (understanding vs expression).

For each path output: parse JSON -> understanding text + translation lines.
Run the same unit check (qwen3:4b scorer) on both texts:
  understanding pass & translation pass -> success
  understanding fail                    -> understanding_error
  understanding pass & translation fail -> expression_error
Writes results/stage4_path/analysis/{path_analysis.jsonl,error_attribution.csv}.
"""
import glob
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import DEFAULT_CONFIG as cfg  # noqa: E402
from src.evaluator import Evaluator, parse_json  # noqa: E402
from src.generator import load_corpus  # noqa: E402


def split_path(raw):
    data = parse_json(raw)
    if not data:
        return None, None
    und = data.get("understanding")
    tr = data.get("translation")
    if isinstance(und, list):
        parts = []
        for u in und:
            if isinstance(u, dict):
                parts.append("%s: %s" % (u.get("literal", ""), u.get("deep_meaning", "")))
            else:
                parts.append(str(u))
        und = "\n".join(parts)
    if isinstance(tr, list):
        tr = "\n".join(str(t) for t in tr)
    if not und or not tr:
        return None, None
    return str(und), str(tr)


def check_consistency(ev, und, tr):
    """Does the translation keep what the model itself said it understood? (05 guide, check 2)

    The scorer sees the stated understanding and the translation only, never the poem's units.
    Returns True / False, or None if the scorer output could not be parsed.
    """
    unit = {
        "id": 0,
        "chinese": "the model's own stated understanding",
        "semantic_content": "the translation must express the same meaning, negation, "
                            "subjects and devices as the stated understanding",
        "pass_conditions": [
            "every meaning the understanding states is present in the translation",
            "no meaning appears in the translation that contradicts the understanding",
        ],
    }
    r = ev.check_unit(unit, "Stated understanding:\n%s\n\nTranslation:\n%s" % (und, tr))
    return bool(r["pass"]) if r["valid"] else None


def select_cases(df, k=10):
    """Pick report cases after the fact (04 guide 6.3): 5 successes, 3 understanding errors,
    2 expression errors. A case is classified by its most frequent attribution over its four units,
    ties going to the more severe class. One case per trap type is taken first, then the rest, always
    in id order, so the choice is reproducible. Cases illustrate the statistics, they are not used
    for inference."""
    ok = df[df.parsed == True]  # noqa: E712
    severity = {"understanding_error": 0, "expression_error": 1, "success": 2}

    def majority(s):
        counts = s.value_counts()
        top = counts[counts == counts.max()].index
        return min(top, key=lambda c: severity[c])

    per = ok.groupby("id").attribution.agg(majority)
    quota = {"success": 5, "understanding_error": 3, "expression_error": 2}
    chosen = []
    for cls, n in quota.items():
        ids = sorted(per[per == cls].index)
        seen, picked = set(), []
        for i in ids:  # one per trap type first
            t = ok[ok.id == i].trap_type.iloc[0]
            if t not in seen:
                seen.add(t)
                picked.append(i)
        picked += [i for i in ids if i not in picked]
        chosen += [(i, cls) for i in picked[:n]]
    return chosen[:k]


def attribute(u_ok, t_ok):
    if u_ok and t_ok:
        return "success"
    if not u_ok:
        return "understanding_error"
    return "expression_error"


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "stage4_path"
    base = os.path.join(cfg.results_dir, stage)
    out_dir = os.path.join(base, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    poems = {p["id"]: p for p in load_corpus(cfg.data_dir, "main")}
    ev = Evaluator()
    rows, lines = [], []
    for path in tqdm(sorted(glob.glob(os.path.join(base, "translations", "*_r*.json")))):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        und, tr = split_path(rec["translation"])
        poem = poems[rec["poem_id"]]
        if und is None:
            rows.append({"id": rec["id"], "poem": rec["poem_id"], "model": rec.get("model") or rec["id"].split("_")[-2],
                         "trap_type": rec.get("trap_type"), "parsed": False})
            continue
        for u in poem["semantic_units"]:
            ur = ev.check_unit(u, und)
            tr_r = ev.check_unit(u, tr)
            if not (ur["valid"] and tr_r["valid"]):
                continue
            rows.append({
                "id": rec["id"], "poem": rec["poem_id"], "model": rec.get("model") or rec["id"].split("_")[-2],
                "trap_type": rec.get("trap_type"), "unit": u["id"], "parsed": True,
                "understanding_pass": bool(ur["pass"]), "translation_pass": bool(tr_r["pass"]),
                "attribution": attribute(ur["pass"], tr_r["pass"]),
            })
        cons = check_consistency(ev, und, tr)
        lines.append({"id": rec["id"], "understanding": und, "translation": tr, "consistent": cons})
        for r in rows:
            if r["id"] == rec["id"]:
                r["consistent"] = cons
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "error_attribution.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "path_analysis.jsonl"), "w", encoding="utf-8") as f:
        for l in lines:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    if len(df) == 0:
        print("no parsed outputs"); return
    print("outputs:", df.groupby("model").parsed.agg(["count", "sum"]).to_string())
    ok = df[df.parsed == True]  # noqa: E712
    if len(ok):
        cases = select_cases(df)
        with open(os.path.join(out_dir, "selected_cases.json"), "w", encoding="utf-8") as f:
            json.dump([{"id": i, "class": c} for i, c in cases], f, ensure_ascii=False, indent=2)
        cs = ok.drop_duplicates("id").consistent.dropna()
        print("consistency (understanding -> translation): %.3f over %d outputs" % (cs.astype(float).mean(), len(cs)))
        print("selected cases:", cases)
        print(ok.groupby("model").attribution.value_counts(normalize=True).round(3).to_string())
        print(ok.groupby("trap_type").attribution.value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
