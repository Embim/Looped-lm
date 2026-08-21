"""Component-level timing: which op inside one looped layer costs the step time."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopedlm.config import ModelConfig      # noqa: E402
from loopedlm.model import Block, LoopedQwen3  # noqa: E402
from loopedlm.presets import BASE_MODEL      # noqa: E402

B, S, D, I, H, HKV, HD = 32, 512, 512, 1280, 8, 2, 64
TOK = B * S


def timed(fn, iters=25, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def mm(label, m, k, n):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    dt = timed(lambda: a @ w)
    print(f"  {label:34s} {dt*1e6:8.0f} us  {2*m*k*n/dt/1e12:6.1f} TFLOPS")
    return dt


def main():
    print(f"{torch.cuda.get_device_name(0)}\n")
    print(f"raw matmuls at our shapes (tokens={TOK}):")
    t_qkvo = mm("[T,512]x[512,512]  (q/o proj)", TOK, D, D)
    mm("[T,512]x[512,128]  (k/v proj)", TOK, D, HKV * HD)
    t_mlp = mm("[T,512]x[512,1280] (gate/up)", TOK, D, I)
    mm("[T,1280]x[1280,512] (down)", TOK, I, D)
    mm("[T,512]x[512,8192]  (lm head)", TOK, D, 8192)
    mm("large reference 4096^3", 4096, 4096, 4096)

    print("\nSDPA (causal, B=32 S=512 H=8 hd=64):")
    q = torch.randn(B, H, S, HD, device="cuda", dtype=torch.bfloat16)
    k2 = torch.randn(B, HKV, S, HD, device="cuda", dtype=torch.bfloat16)
    v2 = torch.randn(B, HKV, S, HD, device="cuda", dtype=torch.bfloat16)
    kf = k2.repeat_interleave(H // HKV, dim=1).contiguous()
    vf = v2.repeat_interleave(H // HKV, dim=1).contiguous()
    a_flops = 2 * 2 * B * H * S * S * HD / 2      # causal
    for label, fn in [
        ("enable_gqa=True", lambda: F.scaled_dot_product_attention(q, k2, v2, is_causal=True, enable_gqa=True)),
        ("repeat_kv then sdpa", lambda: F.scaled_dot_product_attention(q, kf, vf, is_causal=True)),
        ("MHA (all heads real)", lambda: F.scaled_dot_product_attention(q, kf, vf, is_causal=True)),
    ]:
        try:
            dt = timed(fn)
            print(f"  {label:34s} {dt*1e6:8.0f} us  {a_flops/dt/1e12:6.1f} TFLOPS")
        except Exception as e:
            print(f"  {label:34s} FAILED {str(e)[:70]}")
    for name, backend in [("FLASH", SDPBackend.FLASH_ATTENTION),
                          ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
                          ("MATH", SDPBackend.MATH)]:
        for lbl, qq, kk, vv, gqa in [("gqa", q, k2, v2, True), ("repeat", q, kf, vf, False)]:
            try:
                with sdpa_kernel(backend):
                    dt = timed(lambda: F.scaled_dot_product_attention(
                        qq, kk, vv, is_causal=True, enable_gqa=gqa))
                print(f"  backend {name:10s} {lbl:7s}          {dt*1e6:8.0f} us")
            except Exception as e:
                print(f"  backend {name:10s} {lbl:7s}          n/a ({str(e)[:45]})")

    print("\none Block forward / backward:")
    mc = ModelConfig(**dict(BASE_MODEL))
    blk = Block(mc).cuda()
    x = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cos = torch.randn(S, HD, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(S, HD, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dt_f = timed(lambda: blk(x, cos, sin))
    print(f"  block forward                      {dt_f*1e6:8.0f} us")
    ideal = 2 * t_qkvo + 3 * t_mlp
    print(f"  sum of its 5 big matmuls           {ideal*1e6:8.0f} us  "
          f"-> overhead x{dt_f/ideal:.1f}")

    print("\nfull model step breakdown (T=16):")
    m = LoopedQwen3(mc).cuda()
    idx = torch.randint(0, 8192, (B, S), device="cuda")
    y = torch.randint(0, 8192, (B, S), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        dt_fwd = timed(lambda: m(idx, n_loops=16), iters=8, warm=3)
    print(f"  forward only (no grad)             {dt_fwd*1e3:8.1f} ms")

    def full():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(idx, n_loops=16)
            lg = m.head(o["hidden"])
            loss = F.cross_entropy(lg.float().reshape(-1, 8192), y.reshape(-1))
        loss.backward()
        m.zero_grad(set_to_none=True)

    print(f"  fwd+bwd (ckpt on)                  {timed(full, iters=6, warm=2)*1e3:8.1f} ms")
    mc2 = ModelConfig(**dict(BASE_MODEL, grad_ckpt=False))
    m.cfg = mc2
    print(f"  fwd+bwd (ckpt off)                 {timed(full, iters=6, warm=2)*1e3:8.1f} ms")


if __name__ == "__main__":
    main()
