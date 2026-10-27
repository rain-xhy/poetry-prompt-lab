"""Batch translation generator: shuffled, resumable, one JSON file per call."""
import hashlib
import json
import os
import random
import zlib
from typing import Dict, List, Optional

from tqdm import tqdm

from .config import ExperimentConfig, ModelConfig
from .models import OllamaModel
from .prompts import build_path_prompt, build_prompt


def load_corpus(data_dir: str, which: str) -> List[dict]:
    """Return poems of one set ('main' / 'dev' / 'adversarial'); A/B pairs are split."""
    with open(os.path.join(data_dir, "poems", "corpus.json"), encoding="utf-8") as f:
        records = [r for r in json.load(f) if r["set"] == which]
    out = []
    for r in records:
        if r.get("text_b"):
            for suffix, txt in (("A", r["text"]), ("B", r["text_b"])):
                v = dict(r)
                v["id"] = "%s%s" % (r["id"], suffix)
                v["text"] = txt
                out.append(v)
        else:
            out.append(r)
    return out


def _seed(task_id: str) -> int:
    return zlib.crc32(task_id.encode("utf-8")) & 0x7FFFFFFF


def generate_batch(
    cfg: ExperimentConfig,
    poems: List[dict],
    strategies: List[str],
    model_names: List[str],
    n_repeats: int,
    out_dir: str,
    path_mode: bool = False,
    limit: Optional[int] = None,
) -> List[dict]:
    models: Dict[str, ModelConfig] = {m.name: m for m in cfg.models}
    clients = {n: OllamaModel(models[n]) for n in model_names}

    tasks = [
        (p, s, m, r)
        for p in poems for s in strategies for m in model_names for r in range(1, n_repeats + 1)
    ]
    random.Random(cfg.random_seed).shuffle(tasks)  # fixed-seed shuffle
    if limit:
        tasks = tasks[:limit]

    os.makedirs(out_dir, exist_ok=True)
    results = []
    for poem, strategy, mname, rep in tqdm(tasks, desc=os.path.basename(out_dir)):
        tid = "%s_%s_%s_r%d" % (poem["id"], strategy, mname, rep)
        path = os.path.join(out_dir, tid + ".json")
        if os.path.exists(path):  # resume
            with open(path, encoding="utf-8") as f:
                results.append(json.load(f))
            continue
        prompt = (build_path_prompt if path_mode else build_prompt)(poem, strategy)
        resp = clients[mname].generate(prompt, seed=_seed(tid))
        rec = {
            "id": tid,
            "poem_id": poem["id"],
            "trap_type": poem["trap_type"],
            "strategy": strategy,
            "model": mname,
            "model_path": models[mname].model_path,
            "repeat": rep,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "translation": resp["text"],
            "metadata": {k: v for k, v in resp.items() if k != "text"},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        results.append(rec)

    with open(os.path.join(out_dir, "all_translations.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return results
