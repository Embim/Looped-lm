"""Train one configuration.

    python scripts/train.py --preset A_depth16
    python scripts/train.py --preset C_adaln --name C_adaln_full --set train.total_tokens=100000000
    python scripts/train.py --set model.n_loops=32 --set model.update=normalized --name custom

`--set` takes `model.<field>=<value>` or `train.<field>=<value>`; values are
parsed as JSON when possible, otherwise kept as strings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loopedlm.config import ModelConfig, TrainConfig          # noqa: E402
from loopedlm.hwlock import hold_gpu                           # noqa: E402
from loopedlm.presets import DEPTH_SWEEP, P, estimate_minutes, resolve  # noqa: E402
from loopedlm.train import train                               # noqa: E402


def parse_set(items):
    mo, to = {}, {}
    for it in items or []:
        key, _, raw = it.partition("=")
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        scope, _, field = key.partition(".")
        if scope == "model":
            mo[field] = val
        elif scope == "train":
            to[field] = val
        else:
            raise SystemExit(f"--set key must start with 'model.' or 'train.': {key}")
    return mo, to


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, help=f"one of {len(P)} presets, see loopedlm/presets.py")
    ap.add_argument("--name", default=None, help="run name (defaults to the preset name)")
    ap.add_argument("--set", action="append", dest="sets", default=[])
    ap.add_argument("--depth_sweep", default=None,
                    help="comma-separated loop counts to evaluate at the end; 'none' to skip")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no_lock", action="store_true",
                    help="skip the shared-hardware reservation (only for CPU-only checks)")
    a = ap.parse_args()

    if a.list:
        for k in sorted(P):
            print(k)
        return
    if a.preset and a.preset not in P:
        raise SystemExit(f"unknown preset {a.preset!r}; --list to see them all")

    mo, to = parse_set(a.sets)
    m, t = resolve(a.preset or "", mo, to)
    t["run_name"] = a.name or a.preset or "custom"
    mc, tc = ModelConfig.from_dict(m), TrainConfig.from_dict(t)

    if a.depth_sweep is None:
        sweep = DEPTH_SWEEP
    elif a.depth_sweep.lower() == "none":
        sweep = []
    else:
        sweep = [int(x) for x in a.depth_sweep.split(",")]

    if a.no_lock:
        train(mc, tc, depth_sweep=sweep)
    else:
        mins = estimate_minutes(m, tc.total_tokens) + 3 * len(sweep)
        with hold_gpu(f"looped-lm {tc.run_name}", minutes=int(mins) + 5):
            train(mc, tc, depth_sweep=sweep)


if __name__ == "__main__":
    main()
