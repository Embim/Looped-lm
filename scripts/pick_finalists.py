"""Choose which configs deserve seeds and a full-budget run, from the screening table.

Prints a ranking and, with --emit, a comma-separated list for run_experiments.
Only results from the current machine and compile mode are considered, and the
out-of-budget reference runs are excluded from the ranking (they are an upper
bound, not candidates).

    python scripts/pick_finalists.py
    python scripts/pick_finalists.py --top 6 --emit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def current_machine():
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu = "unknown"
    return gpu, os.environ.get("LOOPEDLM_COMPILE", "0") == "1"


def load(same_machine=True):
    gpu, comp = current_machine()
    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "final" not in d:
            continue
        m = d.get("machine") or {}
        if same_machine and (m.get("gpu") != gpu or bool(m.get("compiled")) != comp):
            continue
        rows.append(d)
    return rows, gpu, comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--any_machine", action="store_true")
    a = ap.parse_args()

    rows, gpu, comp = load(not a.any_machine)
    if not rows:
        print(f"no results for {gpu} (compiled={comp})", file=sys.stderr)
        sys.exit(1)

    # the reference runs deliberately exceed the parameter budget
    cand = [d for d in rows if d["params"] <= 10_000_000 and "_s" not in d["run"]]
    base = min((d["final"]["val_loss"] for d in cand
                if d["run"].startswith("A_depth") and d["model"]["n_loops"] == 16),
               default=None)
    cand.sort(key=lambda d: d["final"]["val_loss"])

    if not a.emit:
        print(f"machine: {gpu}  compiled={comp}   {len(cand)} candidate runs"
              + (f"   plain-loop T=16 baseline: {base:.4f}" if base else ""))
        print(f"{'run':26s} {'T':>4s} {'loss':>8s} {'vs base':>8s} {'ppl':>8s} "
              f"{'bpb':>7s} {'GF/tok':>7s} {'min':>6s}")
        for d in cand:
            f = d["final"]
            delta = f"{f['val_loss'] - base:+.4f}" if base else "   -"
            print(f"{d['run']:26s} {d['model']['n_loops']:4d} {f['val_loss']:8.4f} "
                  f"{delta:>8s} {f['val_ppl']:8.2f} {f['val_bpb']:7.4f} "
                  f"{d['fwd_flops_per_token']/1e9:7.3f} {d['wall_seconds']/60:6.1f}")
        refs = [d for d in rows if d["params"] > 10_000_000]
        for d in refs:
            print(f"{d['run']:26s} {d['model']['n_loops']:4d} {d['final']['val_loss']:8.4f} "
                  f"{'  (ref)':>8s} {d['final']['val_ppl']:8.2f} {d['final']['val_bpb']:7.4f}"
                  f"   {d['params']/1e6:.1f}M params, out of budget")

    # Mechanism candidates only: groups A (depth) and B (learning rate) answer
    # different questions and should not crowd out the mechanisms.
    mech = [d for d in cand if not d["run"].startswith(("A_", "B_"))]
    picked = [d["run"] for d in mech[:a.top]]
    if a.emit:
        print(",".join(picked))
    else:
        print(f"\nfinalists for seeds ({a.top}): {','.join(picked)}")


if __name__ == "__main__":
    main()
