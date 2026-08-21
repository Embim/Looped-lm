"""Evaluate a trained checkpoint: depth sweep, difficulty strata, early exit.

    python scripts/eval.py --ckpt C:\\ml\\looped-lm\\runs\\A_depth16\\ckpt_best.pt
    python scripts/eval.py --ckpt ... --loops 1,2,4,8,16,32,64 --out results/A_depth16.eval.json
    python scripts/eval.py --run A_depth16                 # resolves the usual paths
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loopedlm.config import TrainConfig       # noqa: E402
from loopedlm.eval import full_report          # noqa: E402
from loopedlm.hwlock import hold_gpu           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--run", default=None, help="run name under the default out_dir")
    ap.add_argument("--which", default="best", choices=["best", "last"])
    ap.add_argument("--data_dir", default=TrainConfig.data_dir)
    ap.add_argument("--loops", default=None, help="comma-separated loop counts")
    ap.add_argument("--analysis_loops", type=int, default=None)
    ap.add_argument("--eval_tokens", type=int, default=4_000_000)
    ap.add_argument("--analysis_tokens", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_lock", action="store_true")
    a = ap.parse_args()

    if a.run:
        ckpt = Path(TrainConfig.out_dir) / a.run / f"ckpt_{a.which}.pt"
        out = a.out or str(ROOT / "results" / f"{a.run}.eval.json")
    elif a.ckpt:
        ckpt = Path(a.ckpt)
        out = a.out or str(ckpt.parent / "eval.json")
    else:
        raise SystemExit("pass --ckpt or --run")
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")
    loops = [int(x) for x in a.loops.split(",")] if a.loops else None

    def go():
        rep = full_report(ckpt, a.data_dir, out_json=out, loops=loops,
                          analysis_loops=a.analysis_loops, eval_tokens=a.eval_tokens,
                          analysis_tokens=a.analysis_tokens, batch=a.batch)
        d = rep["depth_curve"]["loss"]
        print(f"\ndepth curve (read-out after each loop): "
              f"{' '.join(f'{v:.3f}' for v in d)}")
        best = min(range(len(d)), key=lambda i: d[i])
        print(f"best read-out step: {best} (loss {d[best]:.4f})")
        for name, rows in rep["early_exit"].items():
            print(f"\nearly exit by {name}:")
            for r in rows:
                print(f"  thr {r['threshold']:<7g} avg_loops {r['avg_loops']:5.2f}  "
                      f"loss {r['loss']:.4f}  ppl {r['ppl']:7.2f}  bpb {r['bpb']:.4f}  "
                      f"at_max {r['frac_max_depth']*100:4.1f}%")
        print(f"\nwrote {out}")

    if a.no_lock:
        go()
    else:
        with hold_gpu(f"looped-lm eval {ckpt.parent.name}", minutes=25):
            go()


if __name__ == "__main__":
    main()
