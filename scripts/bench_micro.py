"""Where does the step time actually go, and what recovers it.

The looped model is dispatch-bound at this size, so the interesting axis is
kernels-per-layer, not FLOPs.  This measures the fused-norm change, GQA paths and
CUDA graphs (the only torch.compile backend available without Triton on Windows).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopedlm.config import ModelConfig, TrainConfig      # noqa: E402
from loopedlm.losses import compute_loss                  # noqa: E402
from loopedlm.model import LoopedQwen3                     # noqa: E402
from loopedlm.presets import BASE_MODEL                    # noqa: E402
from loopedlm.train import build_optimizer                 # noqa: E402


def step_time(T=16, B=32, seq=512, iters=7, backend=None, **over):
    cfg = dict(BASE_MODEL); cfg.update(n_loops=T, max_loops=max(64, T), max_seq_len=seq)
    cfg.update(over)
    mc = ModelConfig(**cfg)
    m = LoopedQwen3(mc).cuda()
    if backend:
        for blk in list(m.core) + list(m.prelude) + list(m.coda):
            blk.forward = torch.compile(blk.forward, backend=backend, dynamic=False)
    opt = build_optimizer(m, TrainConfig())
    x = torch.randint(0, mc.vocab_size, (B, seq), device="cuda")
    y = torch.randint(0, mc.vocab_size, (B, seq), device="cuda")
    g = torch.Generator(device="cuda"); g.manual_seed(0)
    ts = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(iters):
        torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x, n_loops=T)
            loss, _ = compute_loss(m, out, y, g)
        loss.backward()
        opt.step()
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    warm = 3 if backend else 2
    good = sorted(ts[warm:]); dt = good[len(good) // 2]
    peak = torch.cuda.max_memory_allocated() / 2**30
    tok = B * seq
    del m, opt, out, loss; torch.cuda.empty_cache()
    return dt, tok / dt, peak


def show(label, **kw):
    try:
        dt, tps, peak = step_time(**kw)
        print(f"{label:42s} {dt*1e3:7.0f} ms  {tps/1e3:6.1f}k tok/s  peak {peak:5.2f}G  "
              f"25M->{25e6/tps/60:5.1f}min  100M->{100e6/tps/60:5.1f}min", flush=True)
        return tps
    except Exception as e:
        print(f"{label:42s} FAILED {type(e).__name__}: {str(e)[:90]}", flush=True)
        torch.cuda.empty_cache()
        return 0.0


if __name__ == "__main__":
    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}\n")
    print("-- baseline with the fused-norm change (T=16, B=32, ckpt on) --")
    base = show("fused RMSNorm")
    print("\n-- kernel-count experiments --")
    show("cudagraphs backend", backend="cudagraphs")
    print("\n-- attention path --")
    show("T=16 no depth mechanisms", T=16)
    print("\n-- depth scaling --")
    for T in (8, 32, 64):
        show(f"T={T}", T=T)
    print("\n-- truncated bptt (cheaper deep training) --")
    show("T=32 bptt=8", T=32, bptt_last_k=8)
    show("T=64 bptt=8", T=64, bptt_last_k=8)
    show("T=64 bptt=4", T=64, bptt_last_k=4)
    print("\n-- larger micro-batch (same tokens/step via accumulation elsewhere) --")
    show("T=16 B=64", B=64)
