"""Clean A/B of the attention path and of truncated BPTT, on a reserved card."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopedlm.config import ModelConfig, TrainConfig
from loopedlm.hwlock import hold_gpu
from loopedlm.losses import compute_loss
from loopedlm.model import LoopedQwen3
from loopedlm.presets import BASE_MODEL
from loopedlm.train import build_optimizer


def step_time(T=16, B=32, seq=512, iters=7, **over):
    cfg = dict(BASE_MODEL); cfg.update(n_loops=T, max_loops=max(64, T), max_seq_len=seq)
    cfg.update(over)
    mc = ModelConfig(**cfg)
    m = LoopedQwen3(mc).cuda(); m.train()
    opt = build_optimizer(m, TrainConfig())
    x = torch.randint(0, mc.vocab_size, (B, seq), device="cuda")
    y = torch.randint(0, mc.vocab_size, (B, seq), device="cuda")
    g = torch.Generator(device="cuda"); g.manual_seed(0)
    ts = []; torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            out = m(x, n_loops=T)
            loss, _ = compute_loss(m, out, y, g)
        loss.backward(); opt.step()
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    good = sorted(ts[2:]); dt = good[len(good)//2]
    peak = torch.cuda.max_memory_allocated()/2**30
    del m, opt, out, loss; torch.cuda.empty_cache()
    return dt, B*seq/dt, peak


def show(label, **kw):
    try:
        dt, tps, peak = step_time(**kw)
        print(f"{label:38s} {dt*1e3:7.0f} ms  {tps/1e3:6.1f}k tok/s  peak {peak:5.2f}G  "
              f"25M->{25e6/tps/60:5.1f}min  100M->{100e6/tps/60:6.1f}min", flush=True)
    except Exception as e:
        print(f"{label:38s} FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    with hold_gpu("looped-lm attention/bptt benchmark", minutes=25):
        free = torch.cuda.mem_get_info()[0]/2**30
        print(f"{torch.cuda.get_device_name(0)}  free VRAM {free:.2f} GiB\n", flush=True)
        print("-- attention path, T=16 B=32 --")
        show("enable_gqa=True (math fallback)", attn_gqa_mode="enable_gqa")
        show("repeat KV explicitly", attn_gqa_mode="repeat")
        show("full MHA, n_kv=8, mlp 1024", n_kv_heads=8, intermediate_size=1024)
        print("\n-- depth, with the fast attention path --")
        for T in (8, 16, 32, 64):
            show(f"T={T}", T=T)
        print("\n-- truncated BPTT --")
        show("T=32 bptt=8", T=32, bptt_last_k=8)
        show("T=64 bptt=8", T=64, bptt_last_k=8)
        show("T=64 bptt=4", T=64, bptt_last_k=4)
        show("T=64 bptt=8 no ckpt", T=64, bptt_last_k=8, grad_ckpt=False)
