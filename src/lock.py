"""Load and verify results/candidate_lock.json (04 guide, stage 2 step 1)."""
import hashlib
import json
import os

from .config import DEFAULT_CONFIG


def load_lock(results_dir: str = DEFAULT_CONFIG.results_dir) -> dict:
    path = os.path.join(results_dir, "candidate_lock.json")
    if not os.path.exists(path):
        raise FileNotFoundError("candidate_lock.json missing: run stage 1 analysis first")
    with open(path, encoding="utf-8") as f:
        lock = json.load(f)
    body = {k: v for k, v in lock.items() if k != "sha256"}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if lock.get("sha256") != expected:
        raise ValueError("candidate_lock.json was modified after locking (SHA-256 mismatch)")
    return lock
