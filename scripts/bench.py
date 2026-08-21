"""Measure step time / peak memory / achieved TFLOPS.

The looped model runs T * n_core transformer layers per step, so at this size the
step is dominated by kernel-launch overhead rather than FLOPs.  This script is
what the batch size / checkpointing / torch.compile decisions are based on.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopedlm.config import ModelConfig, TrainConfig      # noqa: E402
from loopedlm.losses import compute_loss                  # noqa: E402
from loopedlm.model import LoopedQwen3                     # noqa: E402
from loopedlm.presets import BASE_MODEL                    # noqa: E402
from loopedlm.train import build_optimizer                 # noqa: E402


def bench(T, batch, ckpt, bptt=0, seq=512, iters=7, compile_blocks=False, accum=1, **over):
    cfg = dict(BASE_MODEL)
    cfg.update(n_loops=T, grad_ckpt=ckpt, bptt_last_k=bptt, max_loops=max(64, T), max_seq_len=seq)
    cfg.update(over)
    mc = ModelConfig(**cfg)
    m = LoopedQwen3(mc).cuda()
    if compile_blocks:
        # compile the shared block once; the same graph is reused for every loop step
        for blk in list(m.core) + list(m.prelude) + list(m.coda):
            blk.forward = torch.compile(blk.forward, dynamic=False)
    opt = build_optimizer(m, TrainConfig())
    x = torch.randint(0, mc.vocab_size, (batch, seq), device="cuda")
    y = torch.randint(0, mc.vocab_size, (batch, seq), device="cuda")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for i in range(iters):
        torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(x, n_loops=T)
                loss, _ = compute_loss(m, out, y, gen)
            (loss / accum).backward()
        opt.step()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    warm = 3 if compile_blocks else 2
    good = sorted(ts[warm:])
    dt = good[len(good) // 2]
    peak = torch.cuda.max_memory_allocated() / 2**30
    tok = batch * seq * accum
    fwd = m.flops_per_token(seq_len=seq)
    mult = 4 if (ckpt and bptt == 0) else 3
    tflops = mult * fwd * tok / dt / 1e12
    label = f"T={T:3d} B={batch:3d}x{accum} S={seq} ckpt={int(ckpt)} bptt={bptt} cmp={int(compile_blocks)}"
    if over:
        label += " " + " ".join(f"{k}={v}" for k, v in over.items())
    print(f"{label:74s} {dt*1e3:7.0f} ms  {tok/dt/1e3:6.1f}k tok/s  peak {peak:5.2f}G  "
          f"{tflops:5.1f} TF  25M->{25e6/(tok/dt)/60:5.1f}min  100M->{100e6/(tok/dt)/60:5.1f}min",
          flush=True)
    del m, opt, x, y, out, loss
    torch.cuda.empty_cache()
    return tok / dt


def try_bench(*a, **kw):
    try:
        return bench(*a, **kw)
    except Exception as e:
        print(f"  FAILED {a} {kw}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        traceback.clear_frames(sys.exc_info()[2])
        torch.cuda.empty_cache()
        return 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}\n", flush=True)

    if a.stage in ("all", "batch"):
        print("-- batch size / checkpointing (T=16, effective batch fixed at 16k tokens) --")
        try_bench(16, 32, True)
        try_bench(16, 32, False)
        try_bench(16, 16, False, seq=1024)
        print("-- larger effective batch --")
        try_bench(16, 64, True)
        try_bench(16, 128, True)

    if a.stage in ("all", "compile"):
        print("\n-- torch.compile on the shared block --")
        try_bench(16, 32, True, compile_blocks=True)
        try_bench(16, 32, False, compile_blocks=True)

    if a.stage in ("all", "depth"):
        print("\n-- depth scaling (ckpt on) --")
        for T in (4, 8, 32, 64):
            try_bench(T, 32, True)
        print("-- truncated bptt --")
        try_bench(32, 32, True, bptt=8)
        try_bench(64, 32, True, bptt=8)

    if a.stage in ("all", "mech"):
        print("\n-- per-mechanism overhead (T=16, B=32) --")
        for kw in [dict(readout="pool_gate"), dict(loop_memory="depth_attn"),
                   dict(depth_cond="adaln"), dict(depth_cond="depth_rope"),
                   dict(update="normalized"), dict(momentum=True, momentum_read=True),
                   dict(deep_supervision="random_k"),
                   dict(halting="ponder", deep_supervision="random_k")]:
            try_bench(16, 32, True, **kw)
