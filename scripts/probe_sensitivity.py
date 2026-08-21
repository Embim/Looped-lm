"""Is the recurrence alive?  Decompose the block's output variance per loop step.

External review proposed the strongest competing explanation to "collinear drift":
the block may be nearly insensitive to the state -- a depth analogue of posterior
collapse, where the re-injected token embedding e is a bypass that lets the block
cut loss without ever reading h.  If true, latent-geometry fixes treat the wrong
disease and the cure is removing the bypass, not reshaping the trajectory.

Method, per loop step t: take the real states h_t from a forward pass, then
measure how much the block's output moves when h is replaced by another sample's
state (state sensitivity) versus when e is replaced (token sensitivity), holding
the other fixed.  Shuffling across the batch keeps the marginal distribution
right.  Reported as the share of total movement due to the state.

    python scripts/probe_sensitivity.py --run A_depth16 [--steps 0,1,3,7,15]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loopedlm.config import TrainConfig       # noqa: E402
from loopedlm.data import TokenStream          # noqa: E402
from loopedlm.eval import load_checkpoint      # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--which", default="best")
    ap.add_argument("--data_dir", default=TrainConfig.data_dir)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--steps", default=None, help="comma-separated loop indices")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ckpt = Path(TrainConfig.out_dir) / a.run / f"ckpt_{a.which}.pt"
    model, mc, ck = load_checkpoint(ckpt, "cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    T = mc.n_loops
    steps = ([int(x) for x in a.steps.split(",")] if a.steps
             else sorted({0, 1, 2, 3, T // 4, T // 2, 3 * T // 4, T - 1}))
    steps = [t for t in steps if 0 <= t < T]
    seq = ck["train_config"]["seq_len"]

    stream = TokenStream(Path(a.data_dir) / "val.bin", seq, a.batch,
                         device=str(device), shuffle=False,
                         max_windows=a.n_batches * a.batch)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
           if str(device).startswith("cuda") else contextlib.nullcontext())

    acc = {t: {"state": 0.0, "token": 0.0, "base": 0.0} for t in steps}
    n_acc = 0
    for x, _ in stream.batches():
        with amp:
            out = model(x, collect_states=True)
            e = model.embed_norm(model.embed_tokens(x))
            cos, sin = out["cos"], out["sin"]
            perm = torch.randperm(x.shape[0], device=device)
            for t in steps:
                h = out["traj"][t]                      # state entering loop t
                f = model._core_forward(h, e, t, T, cos, sin, None, None).float()
                f_hperm = model._core_forward(h[perm], e, t, T, cos, sin, None, None).float()
                f_eperm = model._core_forward(h, e[perm], t, T, cos, sin, None, None).float()
                acc[t]["state"] += float((f - f_hperm).pow(2).mean())
                acc[t]["token"] += float((f - f_eperm).pow(2).mean())
                acc[t]["base"] += float(f.pow(2).mean())
        n_acc += 1

    rep = {"run": a.run, "n_loops": T, "batches": n_acc, "rows": []}
    print(f"{a.run} (T={T}): share of block-output movement due to the state vs the token")
    print(f"{'t':>4s} {'E|f|^2':>9s} {'state-driven':>13s} {'token-driven':>13s} {'state share':>12s}")
    for t in steps:
        s, k, b = acc[t]["state"] / n_acc, acc[t]["token"] / n_acc, acc[t]["base"] / n_acc
        share = s / max(s + k, 1e-12)
        rep["rows"].append({"t": t, "f2": b, "state": s, "token": k, "state_share": share})
        print(f"{t:4d} {b:9.3f} {s:13.4f} {k:13.4f} {share:12.3f}")

    out_path = a.out or (ROOT / "results" / f"{a.run}.sensitivity.json")
    Path(out_path).write_text(json.dumps(rep, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
