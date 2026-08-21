"""How to run GQA attention fast on a build with no FlashAttention kernels.

Measures forward and forward+backward for every way of getting from 2 KV heads to
8 query heads, because the naive `enable_gqa=True` silently selects the math
backend here.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

B, S, H, HKV, HD = 32, 512, 8, 2, 64
REP = H // HKV


def timed(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def make(hkv, grad):
    q = torch.randn(B, H, S, HD, device="cuda", dtype=torch.bfloat16, requires_grad=grad)
    k = torch.randn(B, hkv, S, HD, device="cuda", dtype=torch.bfloat16, requires_grad=grad)
    v = torch.randn(B, hkv, S, HD, device="cuda", dtype=torch.bfloat16, requires_grad=grad)
    return q, k, v


def make_transposed(hkv, grad):
    """Same layout the model produces: projection -> view -> transpose(1,2)."""
    q = torch.randn(B, S, H, HD, device="cuda", dtype=torch.bfloat16, requires_grad=grad)
    k = torch.randn(B, S, hkv, HD, device="cuda", dtype=torch.bfloat16, requires_grad=grad)
    v = torch.randn(B, S, hkv, HD, device="cuda", dtype=torch.bfloat16, requires_grad=grad)
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), (q, k, v)


def variants():
    return {
        "enable_gqa=True": lambda q, k, v: F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True),
        "repeat_interleave": lambda q, k, v: F.scaled_dot_product_attention(
            q, k.repeat_interleave(REP, 1), v.repeat_interleave(REP, 1), is_causal=True),
        "ri+contiguous": lambda q, k, v: F.scaled_dot_product_attention(
            q, k.contiguous().repeat_interleave(REP, 1), v.contiguous().repeat_interleave(REP, 1),
            is_causal=True),
        "expand+reshape": lambda q, k, v: F.scaled_dot_product_attention(
            q,
            k.unsqueeze(2).expand(B, HKV, REP, S, HD).reshape(B, H, S, HD),
            v.unsqueeze(2).expand(B, HKV, REP, S, HD).reshape(B, H, S, HD),
            is_causal=True),
        "expand only (no reshape)": lambda q, k, v: F.scaled_dot_product_attention(
            q.view(B, HKV, REP, S, HD),
            k.unsqueeze(2).expand(B, HKV, REP, S, HD),
            v.unsqueeze(2).expand(B, HKV, REP, S, HD),
            is_causal=True).reshape(B, H, S, HD),
    }


def run(label, tensors, fns, grad):
    print(f"\n{label}  (grad={grad})")
    for name, fn in fns.items():
        try:
            q, k, v = tensors
            if grad:
                def step():
                    o = fn(q, k, v)
                    o.sum().backward()
                    for t in (q, k, v):
                        t.grad = None
                dt = timed(step, iters=10, warm=3)
            else:
                with torch.no_grad():
                    dt = timed(lambda: fn(q, k, v), iters=20, warm=5)
            print(f"   {name:26s} {dt*1e6:8.0f} us   x{dt/1e-6/290:5.1f} vs 290us ref")
        except Exception as e:
            print(f"   {name:26s} FAILED {type(e).__name__}: {str(e)[:60]}")


if __name__ == "__main__":
    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"available SDPA backends: flash={torch.backends.cuda.flash_sdp_enabled()} "
          f"mem_eff={torch.backends.cuda.mem_efficient_sdp_enabled()} "
          f"math={torch.backends.cuda.math_sdp_enabled()}")
    fns = variants()
    run("contiguous [B,H,S,D] tensors", make(HKV, False), fns, False)
    run("contiguous [B,H,S,D] tensors", make(HKV, True), fns, True)
    q, k, v, leaves = make_transposed(HKV, True)
    run("transposed (as the model builds them)", (q, k, v), fns, True)

    print("\nfull MHA reference (n_kv = n_heads, no expansion needed)")
    q2, k2, v2 = make(H, True)
    def step2():
        o = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
        o.sum().backward()
        for t in (q2, k2, v2):
            t.grad = None
    print(f"   {'MHA':26s} {timed(step2, iters=10, warm=3)*1e6:8.0f} us")
