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


def _current_machine() -> tuple[str, bool]:
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu = "unknown"
    return gpu, os.environ.get("LOOPEDLM_COMPILE", "0") == "1"


CUR_GPU, CUR_COMPILED = _current_machine()


def est(name: str, tokens: int, tok_s: float) -> float:
    return estimate_minutes(resolve(name)[0], tokens, tok_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None, help="comma-separated group letters, e.g. A,B,C")
    ap.add_argument("--presets", default=None, help="comma-separated preset names")
    ap.add_argument("--tokens", type=int, default=None, help="override train.total_tokens")
    ap.add_argument("--suffix", default="", help="appended to the run name")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds; each preset runs once per seed as "
                         "<preset>_s<seed>.  Differences below the seed spread are "
                         "not results, so any claimed effect needs this.")
    ap.add_argument("--set", action="append", dest="sets", default=[],
                    help="passed through to scripts/train.py, e.g. train.lr=2e-3")
    ap.add_argument("--depth_sweep", default=None)
    ap.add_argument("--tok_s", type=float, default=52700.0, help="measured throughput at T=16,n_core=2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_board", action="store_true")
    ap.add_argument("--shard", default=None,
                    help="i/N: take every Nth config starting at i.  The model is "
                         "dispatch-bound, so several concurrent workers raise "
                         "aggregate throughput 1.8x (measured) even though each one "
                         "gets slower; per-run wall_seconds is then not comparable.")
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
    # Expand seeds before printing the plan: a preview that under-reports the queue
    # is worse than no preview.
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [None]
    queue = [(n, s) for n in queue for s in seeds]

    if a.shard:
        i, n_sh = (int(x) for x in a.shard.split("/"))
        queue = queue[i::n_sh]
        print(f"shard {i}/{n_sh}: {len(queue)} of the configs", flush=True)

    total_est = sum(est(n, a.tokens or resolve(n)[1]["total_tokens"], a.tok_s)
                    for n, _ in queue)
    print(f"{len(queue)} runs, estimated {total_est/60:.1f} h total\n", flush=True)
    for n, s in queue:
        label = n + (f" (seed {s})" if s is not None else "")
        print(f"  {label:32s} ~{est(n, a.tokens or resolve(n)[1]['total_tokens'], a.tok_s):5.1f} min",
              flush=True)
    if a.dry_run:
        return

    done, failed = [], []
    for i, (name, seed) in enumerate(queue, 1):
        run_name = name + a.suffix + (f"_s{seed}" if seed is not None else "")
        res_file = RESULTS / f"{run_name}.json"
        if res_file.exists() and not a.force:
            # Skip only a result produced by *this* machine and this compile mode.
            # A result carried over from another GPU silently entered a table it
            # does not belong in: the numbers are not comparable across builds, and
            # the skip made a stale Windows run look like a fresh A100 one.
            prev = {}
            try:
                prev = json.loads(res_file.read_text()).get("machine") or {}
            except Exception:
                pass
            same = (prev.get("gpu") == CUR_GPU and bool(prev.get("compiled")) == CUR_COMPILED)
            if same:
                print(f"\n[{i}/{len(queue)}] {run_name}: already done here, skipping", flush=True)
                done.append(run_name)
                continue
            print(f"\n[{i}/{len(queue)}] {run_name}: existing result is from "
                  f"{prev.get('gpu', 'an unrecorded machine')} "
                  f"(compiled={prev.get('compiled')}), re-running on {CUR_GPU}", flush=True)
            res_file.rename(res_file.with_suffix(".json.other-machine"))

        e = est(name, a.tokens or resolve(name)[1]["total_tokens"], a.tok_s)
        print(f"\n[{i}/{len(queue)}] {run_name}  (~{e:.0f} min)", flush=True)
        # No claim here on purpose: scripts/train.py holds the reservation itself
        # (loopedlm/hwlock.py), with a heartbeat and release-on-exit, so the card
        # can never be held by a run that the board does not show.
        cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--preset", name,
               "--name", run_name]
        if a.tokens:
            cmd += ["--set", f"train.total_tokens={a.tokens}"]
        if seed is not None:
            cmd += ["--set", f"train.seed={seed}"]
        for s in a.sets:
            cmd += ["--set", s]
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
