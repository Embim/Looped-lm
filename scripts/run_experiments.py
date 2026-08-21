"""Run a queue of presets sequentially, one process per run.

Each run is a fresh subprocess so a crash or an OOM cannot poison the queue, and
the queue is idempotent: a preset whose results/<name>.json already exists is
skipped unless --force.  Before every run the shared hardware board is asked for
the GPU and it is released afterwards, so other sessions can slot in between
runs rather than waiting for the whole queue.

    python scripts/run_experiments.py --group A
    python scripts/run_experiments.py --presets C_adaln,C_depth_rope,D_normalized
    python scripts/run_experiments.py --group A,B,C,D --tokens 15000000
    python scripts/run_experiments.py --group A --dry_run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loopedlm.config import TrainConfig                             # noqa: E402
from loopedlm.presets import GROUPS, P, estimate_minutes, resolve   # noqa: E402

BOARD = Path(os.environ.get("BOARD_CLI", r"C:\ml\gpu_board\gpu_board.py"))
OWNER = os.environ.get("BOARD_OWNER", "looped-lm (chat-1)")
RESULTS = ROOT / "results"


def board(*args, timeout: int = 60) -> int:
    if not BOARD.exists():
        return 0
    try:
        r = subprocess.run([sys.executable, str(BOARD), *args], capture_output=True,
                           text=True, timeout=timeout)
        out = (r.stdout or r.stderr).strip().splitlines()
        if out:
            print(f"    board: {out[-1]}", flush=True)
        return r.returncode
    except subprocess.TimeoutExpired:
        print("    board: timeout, proceeding", flush=True)
        return 0


def est(name: str, tokens: int, tok_s: float) -> float:
    return estimate_minutes(resolve(name)[0], tokens, tok_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None, help="comma-separated group letters, e.g. A,B,C")
    ap.add_argument("--presets", default=None, help="comma-separated preset names")
    ap.add_argument("--tokens", type=int, default=None, help="override train.total_tokens")
    ap.add_argument("--suffix", default="", help="appended to the run name")
    ap.add_argument("--depth_sweep", default=None)
    ap.add_argument("--tok_s", type=float, default=52700.0, help="measured throughput at T=16,n_core=2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_board", action="store_true")
    ap.add_argument("--mlflow", default=os.environ.get("MLFLOW_TRACKING_URI", ""))
    a = ap.parse_args()

    names = []
    if a.group:
        for g in a.group.split(","):
            names += sorted(GROUPS.get(g.strip(), []))
    if a.presets:
        names += [x.strip() for x in a.presets.split(",")]
    if not names:
        raise SystemExit("nothing to run: pass --group and/or --presets")
    seen, queue = set(), []
    for n in names:
        if n not in P:
            raise SystemExit(f"unknown preset {n!r}")
        if n not in seen:
            seen.add(n); queue.append(n)

    RESULTS.mkdir(exist_ok=True)
    total_est = sum(est(n, a.tokens or resolve(n)[1]["total_tokens"], a.tok_s)
                    for n in queue)
    print(f"{len(queue)} runs, estimated {total_est/60:.1f} h total\n", flush=True)
    for n in queue:
        print(f"  {n:24s} ~{est(n, a.tokens or resolve(n)[1]['total_tokens'], a.tok_s):5.1f} min",
              flush=True)
    if a.dry_run:
        return

    done, failed = [], []
    for i, name in enumerate(queue, 1):
        run_name = name + a.suffix
        res_file = RESULTS / f"{run_name}.json"
        if res_file.exists() and not a.force:
            print(f"\n[{i}/{len(queue)}] {run_name}: already done, skipping", flush=True)
            done.append(run_name)
            continue

        e = est(name, a.tokens or resolve(name)[1]["total_tokens"], a.tok_s)
        print(f"\n[{i}/{len(queue)}] {run_name}  (~{e:.0f} min)", flush=True)
        # No claim here on purpose: scripts/train.py holds the reservation itself
        # (loopedlm/hwlock.py), with a heartbeat and release-on-exit, so the card
        # can never be held by a run that the board does not show.
        cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--preset", name,
               "--name", run_name]
        if a.tokens:
            cmd += ["--set", f"train.total_tokens={a.tokens}"]
        if a.depth_sweep:
            cmd += ["--depth_sweep", a.depth_sweep]
        env = dict(os.environ, PYTHONUTF8="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        if a.mlflow:
            env["MLFLOW_TRACKING_URI"] = a.mlflow
        t0 = time.time()
        r = subprocess.run(cmd, env=env)
        dt = (time.time() - t0) / 60
        if a.no_board:
            board("release", "--owner", OWNER)

        out_dir = resolve(name)[1].get("out_dir") or TrainConfig.out_dir
        summary = Path(out_dir) / run_name / "summary.json"
        if r.returncode == 0 and summary.exists():
            shutil.copy(summary, res_file)
            s = json.loads(res_file.read_text())
            print(f"    OK {dt:.1f} min | loss {s['final']['val_loss']:.4f} "
                  f"ppl {s['final']['val_ppl']:.2f} bpb {s['final']['val_bpb']:.4f}", flush=True)
            done.append(run_name)
        else:
            print(f"    FAILED (rc={r.returncode}) after {dt:.1f} min", flush=True)
            failed.append(run_name)

    print(f"\ndone: {len(done)}   failed: {len(failed)}")
    if failed:
        print("failed:", ", ".join(failed))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_results.py")])


if __name__ == "__main__":
    main()
