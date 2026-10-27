"""One-shot pipeline: finish generation, then score, check and analyse, in order.

Usage: python scripts/run_pipeline.py        (resumable; safe to start again after a crash)

Status: results/pipeline_status.json   Log: results/pipeline.log
Each step is skipped if already recorded as ok. A failed step is logged and the pipeline
continues, so one failure does not stop the independent steps after it.
Generation and scoring steps use the interpreter that launched this script; analysis steps
use .venv (SciPy pinned to match NumPy 1.26).
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
STATUS = os.path.join(RES, "pipeline_status.json")
LOG = os.path.join(RES, "pipeline.log")
PY = sys.executable
VENV = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
VPY = VENV if os.path.exists(VENV) else PY

S = lambda name: os.path.join("scripts", name)  # noqa: E731

# (step name, interpreter, argv)
STEPS = [
    ("gen_stage3", PY, [S("stage3_generate.py")]),
    ("gen_stage4", PY, [S("stage4_generate.py")]),
    ("score_stage2", PY, [S("stage_score.py"), "stage2_main", "main"]),
    ("score_stage3", PY, [S("stage_score.py"), "stage3_adversarial", "adversarial"]),
    ("rescore_stage1_layers", PY, [S("rescore_layers.py"), "stage1_dev", "dev"]),
    ("stage4_analyze", PY, [S("stage4_analyze.py")]),
    ("format_stage2", PY, [S("format_check.py"), "stage2_main"]),
    ("format_stage3", PY, [S("format_check.py"), "stage3_adversarial"]),
    ("scorer_checks", PY, [S("scorer_checks.py"), "stage2_main", "main", "30", "gemma2:2b"]),
    ("stats_stage2_accuracy", VPY, [S("analyze_stats.py"), "stage2_main", "accuracy"]),
    ("stats_stage2_quality", VPY, [S("analyze_stats.py"), "stage2_main", "quality"]),
    ("extra_stage2", VPY, [S("analyze_extra.py"), "stage2_main", "accuracy"]),
    ("extra_stage3", VPY, [S("analyze_extra.py"), "stage3_adversarial", "accuracy"]),
    ("audit_sheet", PY, [S("make_audit_sheet.py"), "6"]),
]


def log(msg):
    line = "%s %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_status():
    if os.path.exists(STATUS):
        with open(STATUS, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_status(st):
    with open(STATUS, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def running_generators():
    """Python processes still running a generate script (not the shells that merely mention it)."""
    cmd = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and "
           "$_.CommandLine -match 'stage[0-9a-z]*_generate\\.py' } | "
           "Select-Object -ExpandProperty ProcessId")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                         capture_output=True, text=True).stdout.split()
    return [p for p in out if p.isdigit()]


def main():
    os.makedirs(RES, exist_ok=True)
    os.chdir(ROOT)
    st = load_status()
    log("pipeline start; generation/scoring python=%s analysis python=%s" % (PY, VPY))

    while True:
        pids = running_generators()
        if not pids:
            break
        log("waiting for running generation process(es): %s" % ",".join(pids))
        time.sleep(60)

    for name, py, argv in STEPS:
        if st.get(name, {}).get("status") == "ok":
            log("skip %s (already ok)" % name)
            continue
        log("start %s" % name)
        t0 = time.time()
        step_log = os.path.join(RES, "pipeline_%s.log" % name)
        with open(step_log, "w", encoding="utf-8") as f:
            rc = subprocess.run([py] + argv, stdout=f, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONIOENCODING="utf-8")).returncode
        st[name] = {"status": "ok" if rc == 0 else "failed", "returncode": rc,
                    "seconds": round(time.time() - t0), "log": step_log}
        save_status(st)
        log("%s %s in %ds (rc=%d)" % (name, st[name]["status"], st[name]["seconds"], rc))

    failed = [k for k, v in st.items() if v["status"] != "ok"]
    log("pipeline finished; failed steps: %s" % (failed or "none"))


if __name__ == "__main__":
    main()
