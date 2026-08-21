"""Correctness checks for the things that would silently invalidate every result.

Runs on CPU by default (so it needs no GPU reservation); pass --cuda to also
check the paths that only differ on GPU.

    python scripts/tests.py
    python scripts/tests.py --cuda
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import shutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loopedlm.config import ModelConfig, TrainConfig     # noqa: E402
from loopedlm.data import TokenStream, load_meta          # noqa: E402
from loopedlm.losses import compute_loss                  # noqa: E402
from loopedlm.model import LoopedQwen3                     # noqa: E402
from loopedlm.presets import P, resolve                    # noqa: E402

# This suite runs torch on CPU, which by default grabs every core.  On a box with
# 252 cores that starved the dispatch thread of a concurrent GPU run and slowed it
# 9x -- the looped model is launch-bound, so it is sensitive to CPU contention,
# not just to GPU contention.
torch.set_num_threads(min(8, torch.get_num_threads()))

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def small(**kw) -> ModelConfig:
    base = dict(vocab_size=256, d_model=64, n_heads=2, head_dim=32, n_kv_heads=1,
                intermediate_size=128, n_core=2, n_loops=4, max_loops=8, max_seq_len=32)
    base.update(kw)
    return ModelConfig(**base)


# ---------------------------------------------------------------------------
def test_causality(device):
    """A change at position j must not move the logits at any position i < j."""
    for kw in ({}, {"loop_memory": "depth_attn"}, {"readout": "pool_gate"},
               {"depth_cond": "depth_rope"}, {"attn_window_schedule": "local2global"},
               {"update": "normalized", "momentum": True, "momentum_read": True}):
        torch.manual_seed(0)
        m = LoopedQwen3(small(**kw)).to(device).eval()
        x = torch.randint(0, 256, (2, 16), device=device)
        with torch.no_grad():
            a = m.head(m(x)["hidden"])
            x2 = x.clone()
            x2[:, 11] = (x2[:, 11] + 7) % 256
            b = m.head(m(x2)["hidden"])
        past = (a[:, :11] - b[:, :11]).abs().max().item()
        future = (a[:, 11:] - b[:, 11:]).abs().max().item()
        check(f"causality {kw or 'base'}", past < 1e-5 and future > 1e-6,
              f"max|d| before={past:.2e} at/after={future:.2e}")


def test_window_mask(device):
    """The sliding-window mask must be causal and exactly w wide."""
    m = LoopedQwen3(small(attn_window_schedule="local2global")).to(device)
    mask = m._window_mask(16, 5, torch.device(device))[0, 0]
    i = torch.arange(16, device=device)
    want = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < 5)
    check("window mask", bool((mask == want).all()), "w=5, S=16")
    check("window mask rows non-empty", bool(mask.any(-1).all()))


def test_depth_rope_inverse(device):
    """The step-dependent rotation must be exactly invertible (it conjugates the block)."""
    m = LoopedQwen3(small(depth_cond="depth_rope", depth_rope_frac=0.5)).to(device)
    h = torch.randn(2, 8, 64, device=device)
    worst = 0.0
    for t in range(8):
        back = m._depth_rotate(m._depth_rotate(h, t), t, inverse=True)
        worst = max(worst, (back - h).abs().max().item())
    check("depth-RoPE inverse", worst < 1e-5, f"max|R^-1 R h - h|={worst:.2e}")
    d = [m._depth_rotate(h, t) for t in range(4)]
    diff = min((d[i] - d[j]).abs().max().item() for i in range(4) for j in range(i + 1, 4))
    check("depth-RoPE steps differ", diff > 1e-3, f"min pairwise diff={diff:.2e}")


def test_loop_count_effect(device):
    """More loops must actually change the output (the loop is not a no-op)."""
    torch.manual_seed(0)
    m = LoopedQwen3(small(n_loops=4)).to(device).eval()
    x = torch.randint(0, 256, (2, 16), device=device)
    with torch.no_grad():
        h = [m(x, n_loops=T)["hidden"] for T in (1, 2, 4, 8)]
    diffs = [(h[i + 1] - h[i]).abs().max().item() for i in range(3)]
    check("loop count changes the state", min(diffs) > 1e-4, f"diffs={[f'{d:.3f}' for d in diffs]}")


def test_bptt_equivalence(device):
    """bptt_last_k >= T must give exactly the gradient of full backprop."""
    def grads(k, ckpt):
        torch.manual_seed(0)
        m = LoopedQwen3(small(n_loops=4, bptt_last_k=k, grad_ckpt=ckpt)).to(device)
        m.train()
        torch.manual_seed(1)
        x = torch.randint(0, 256, (2, 16), device=device)
        y = torch.randint(0, 256, (2, 16), device=device)
        out = m(x)
        loss = torch.nn.functional.cross_entropy(
            m.head(out["hidden"]).reshape(-1, 256), y.reshape(-1))
        loss.backward()
        return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    g_full = grads(0, True)
    for k, ckpt, expect_equal in ((4, True, True), (0, False, True), (2, True, False)):
        g = grads(k, ckpt)
        worst = max((g[n] - g_full[n]).abs().max().item() for n in g_full)
        rel = worst / max(max(v.abs().max().item() for v in g_full.values()), 1e-12)
        ok = (rel < 1e-4) if expect_equal else (rel > 1e-6)
        check(f"bptt_last_k={k} ckpt={int(ckpt)} "
              f"{'== full' if expect_equal else '!= full (truncated)'}", ok, f"rel={rel:.2e}")


def test_param_budget():
    """Every preset must fit its declared budget, and the base config must fit 10M."""
    over = []
    for name in sorted(P):
        m, t = resolve(name)
        budget = t.get("param_budget", TrainConfig.param_budget)
        cfg = ModelConfig.from_dict(m)
        n = LoopedQwen3(cfg).n_params()
        if n > budget:
            over.append(f"{name}: {n/1e6:.3f}M > {budget/1e6:.1f}M")
    check(f"parameter budget for all {len(P)} presets", not over, "; ".join(over[:4]))
    base = LoopedQwen3(ModelConfig.from_dict(resolve("A_depth16")[0]))
    check("base config under 10M", base.n_params() <= 10_000_000,
          f"{base.n_params()/1e6:.3f}M total, {base.n_params(True)/1e6:.3f}M non-embedding")


def test_extra_params_are_scale_free():
    """Each mechanism must cost a negligible, O(T*d)-not-O(T*d^2) share of the budget."""
    base = LoopedQwen3(ModelConfig.from_dict(resolve("A_depth16")[0])).n_params()
    rows = []
    for name in ("C_adaln", "C_depth_rope", "C_learned_emb", "D_normalized", "D_momentum_read",
                 "E_pool_gate", "E_pool_learned", "F_depth_attn", "I_ponder"):
        cfg = ModelConfig.from_dict(resolve(name)[0])
        rows.append((name, LoopedQwen3(cfg).n_params() - base))
    worst = max(rows, key=lambda r: r[1])
    for n, d in rows:
        print(f"        {n:18s} +{d:8d} params  ({100*d/base:5.2f}%)")
    check("mechanism overhead under 2% of the budget", worst[1] / base < 0.02,
          f"worst={worst[0]} +{worst[1]} ({100*worst[1]/base:.2f}%)")


def test_data():
    """Token budget bookkeeping, and that train and val come from different shards."""
    try:
        meta = load_meta(TrainConfig.data_dir)
    except Exception as e:
        check("data meta present", False, str(e)[:70])
        return
    check("train and val are different FineWeb shards",
          Path(meta["train"]["shard"]).name != Path(meta["val"]["shard"]).name,
          f"{Path(meta['train']['shard']).name} vs {Path(meta['val']['shard']).name}")
    check("val bytes/token recorded", meta["val"]["bytes_per_token"] > 1,
          f"{meta['val']['bytes_per_token']:.4f} bytes/token")

    s = TokenStream(Path(TrainConfig.data_dir) / "train.bin", 512, 32, device="cpu",
                    seed=1337, max_windows=100 * 32)
    starts = set(s.starts.tolist())
    check("training windows do not repeat a token", len(starts) == len(s.starts),
          f"{len(s.starts)} windows, stride == seq_len")
    x, y = next(iter(s.batches()))
    check("targets are inputs shifted by one", bool((x[:, 1:] == y[:, :-1]).all()),
          f"x{tuple(x.shape)} y{tuple(y.shape)}")
    check("token ids inside the vocabulary", int(x.max()) < meta["vocab_size"],
          f"max id {int(x.max())} < {meta['vocab_size']}")


def test_eval_determinism(device):
    """Evaluation must be reproducible: no dropout or noise leaking into eval."""
    torch.manual_seed(0)
    m = LoopedQwen3(small(noise_std=0.2, loop_dropout=0.2)).to(device)
    x = torch.randint(0, 256, (2, 16), device=device)
    m.eval()
    with torch.no_grad():
        a, b = m(x)["hidden"], m(x)["hidden"]
    check("eval is deterministic with noise+dropout configured",
          torch.equal(a, b), "noise_std=0.2, loop_dropout=0.2")
    m.train()
    with torch.no_grad():
        c, d = m(x)["hidden"], m(x)["hidden"]
    check("train mode does inject noise", not torch.equal(c, d))


def test_losses(device):
    """Auxiliary losses must be finite, produce gradients, AND actually fire.

    The `expect` key is the point of this test: an earlier version only checked
    finiteness, and passed while deep supervision was a silent no-op because the
    model was not keeping the per-step states it needs.
    """
    cases = {
        "I_deepsup_k2": ["deep_sup"],
        "I_deepsup_all": ["deep_sup"],
        "I_deepsup_kl": ["deep_sup"],
        "I_ponder": ["deep_sup", "ponder", "exp_loops"],
        "J_decorr": ["cos_succ"],
    }
    for name, expect in cases.items():
        cfg = resolve(name)[0]
        mc = small(**{k: v for k, v in cfg.items()
                      if k in ("deep_supervision", "deep_sup_k", "deep_sup_weight",
                               "deep_sup_detach_teacher", "halting", "decorr_weight")})
        torch.manual_seed(0)
        m = LoopedQwen3(mc).to(device)
        m.train()
        x = torch.randint(0, 256, (2, 16), device=device)
        y = torch.randint(0, 256, (2, 16), device=device)
        gen = torch.Generator(device=device); gen.manual_seed(0)
        out = m(x)
        loss, logs = compute_loss(m, out, y, gen)
        loss.backward()
        gn = sum(float(p.grad.pow(2).sum()) for p in m.parameters() if p.grad is not None) ** 0.5
        missing = [k for k in expect if k not in logs]
        moved = abs(logs["loss"] - logs["ce"]) > 1e-6
        check(f"loss {name}",
              math.isfinite(float(loss)) and gn > 0 and not missing and moved,
              f"loss={float(loss):.4f} |g|={gn:.4f}"
              + (f" MISSING {missing}" if missing else "")
              + ("" if moved else " total == plain CE, auxiliary had no effect")
              + f" parts={ {k: round(v, 3) for k, v in logs.items()} }")


def test_deep_sup_changes_gradients(device):
    """Deep supervision must change the gradient, not merely appear in the logs."""
    def run(kw):
        torch.manual_seed(0)
        m = LoopedQwen3(small(**kw)).to(device)
        m.train()
        torch.manual_seed(1)
        x = torch.randint(0, 256, (2, 16), device=device)
        y = torch.randint(0, 256, (2, 16), device=device)
        gen = torch.Generator(device=device); gen.manual_seed(0)
        loss, _ = compute_loss(m, m(x), y, gen)
        loss.backward()
        return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    g0 = run({})
    g1 = run({"deep_supervision": "all", "deep_sup_weight": 0.5, "deep_sup_pos_frac": 1.0})
    rel = max((g1[n] - g0[n]).abs().max().item() for n in g0) / max(
        max(v.abs().max().item() for v in g0.values()), 1e-12)
    check("deep supervision changes the gradient", rel > 1e-3, f"rel diff={rel:.2e}")


def test_early_exit_synthetic():
    """Exit-step selection and the reported averages, on a case computed by hand."""
    from loopedlm.eval import early_exit
    T, bpt = 4, 4.0
    # nll[t, i]: rows are read-out steps 0..T, columns are three tokens
    nll = np.array([
        [9.0, 9.0, 9.0],      # t=0, never an exit candidate
        [1.0, 5.0, 5.0],      # t=1
        [2.0, 4.0, 4.0],      # t=2
        [3.0, 0.5, 3.0],      # t=3
        [4.0, 0.6, 2.0],      # t=4
    ])
    conf = np.array([
        [0.0, 0.0, 0.0],
        [0.9, 0.10, 0.1],     # token 0 passes 0.8 at t=1
        [0.9, 0.20, 0.2],
        [0.9, 0.95, 0.3],     # token 1 passes at t=3
        [0.9, 0.99, 0.4],     # token 2 never passes -> forced to t=T
    ])
    r = early_exit(nll, conf, "ge", [0.8], bpt)[0]
    want_steps = [1, 3, 4]
    want_loss = (1.0 + 0.5 + 2.0) / 3
    check("early exit picks the first passing step",
          abs(r["avg_loops"] - float(np.mean(want_steps))) < 1e-9
          and abs(r["loss"] - want_loss) < 1e-9,
          f"avg_loops={r['avg_loops']:.3f} (want {np.mean(want_steps):.3f}), "
          f"loss={r['loss']:.4f} (want {want_loss:.4f})")
    check("tokens that never pass are charged full depth",
          abs(r["frac_max_depth"] - 1 / 3) < 1e-9, f"frac_max_depth={r['frac_max_depth']:.3f}")

    # An unreachable threshold must reproduce the fixed-depth-T result exactly:
    # this is the anchor that ties the early-exit curve to the depth sweep.
    r2 = early_exit(nll, conf, "ge", [1.01], bpt)[0]
    check("unreachable threshold == fixed depth T",
          abs(r2["loss"] - nll[T].mean()) < 1e-12 and r2["avg_loops"] == T,
          f"loss={r2['loss']:.4f} vs nll[T].mean()={nll[T].mean():.4f}")

    # 'le' mode (KL stability): exit when the prediction stops moving
    kl = np.array([[0., 0., 0.], [0.5, 0.5, 0.5], [0.01, 0.5, 0.5],
                   [0.5, 0.005, 0.5], [0.5, 0.5, 0.5]])
    r3 = early_exit(nll, kl, "le", [0.02], bpt)[0]
    check("le-mode exit uses the first step below the threshold",
          abs(r3["avg_loops"] - np.mean([2, 3, 4])) < 1e-9,
          f"avg_loops={r3['avg_loops']:.3f} (want {np.mean([2,3,4]):.3f})")
    check("monotone thresholds give monotone compute",
          [x["avg_loops"] for x in early_exit(nll, conf, "ge", [0.05, 0.5, 0.95, 1.01], bpt)]
          == sorted(x["avg_loops"] for x in early_exit(nll, conf, "ge", [0.05, 0.5, 0.95, 1.01], bpt)))


def test_strata_selection_bias():
    """The stratification must not manufacture a depth effect out of pure noise.

    With nll drawn i.i.d. per step there is no real gain from depth.  Ranking
    tokens by their own final-step loss still produces an apparent improvement
    (the selected final point is inflated by construction); ranking by an
    independent array must not.  This is why the report never relies on a single
    stratification.
    """
    from loopedlm.eval import difficulty_strata
    rng = np.random.default_rng(0)
    T, N = 8, 40000
    nll = rng.gamma(2.0, 2.0, size=(T + 1, N))       # no depth structure at all
    indep = rng.gamma(2.0, 2.0, size=N)

    by_final = difficulty_strata(nll, rank_by=nll[-1])["top5pct_hardest"]["loss"]
    drop_final = by_final[1] - by_final[-1]
    by_step1 = difficulty_strata(nll)["top5pct_hardest"]["loss"]
    drop_step1 = by_step1[1] - by_step1[-1]
    by_indep = difficulty_strata(nll, rank_by=indep)["top5pct_hardest"]["loss"]
    drop_indep = abs(by_indep[1] - by_indep[-1])
    scale = nll.mean()

    check("ranking by the final step manufactures a spurious depth gain",
          drop_final < -0.5, f"apparent change t1->tT = {drop_final:+.3f} nats (should be strongly negative)")
    check("ranking by step 1 manufactures the opposite spurious gain",
          drop_step1 > 0.5, f"apparent change t1->tT = {drop_step1:+.3f} nats")
    check("ranking by an independent model shows no spurious gain",
          drop_indep < 0.1 * scale, f"|change| = {drop_indep:.3f} nats, mean nll = {scale:.2f}")


def test_analysis_pipeline(device):
    """full_report end to end on an untrained tiny model: shapes, keys, consistency."""
    from loopedlm.eval import full_report
    try:
        meta = load_meta(TrainConfig.data_dir)
    except Exception:
        check("analysis pipeline", False, "no prepared data; skipping")
        return
    mc = small(vocab_size=meta["vocab_size"], n_loops=3, max_loops=4, max_seq_len=64,
               halting="ponder")
    m = LoopedQwen3(mc)
    tmp = Path(TrainConfig.out_dir) / "_pipeline_test"
    tmp.mkdir(parents=True, exist_ok=True)
    ck = tmp / "ckpt_best.pt"
    torch.save({"model": m.state_dict(), "model_config": json.loads(mc.to_json()),
                "train_config": {"seq_len": 64}, "step": 0}, ck)
    rep = full_report(ck, TrainConfig.data_dir, loops=[1, 2, 3], analysis_loops=3,
                      eval_tokens=64 * 8, analysis_tokens=64 * 8, batch=4, device=device)
    dc = rep["depth_curve"]["loss"]
    ok = (len(dc) == 4 and all(math.isfinite(v) for v in dc)
          and set(rep["early_exit"]) >= {"confidence", "kl_stability", "halting_head"}
          and "difficulty_strata_by_final" in rep)
    check("analysis pipeline runs end to end", ok,
          f"depth curve len={len(dc)}, exit modes={sorted(rep['early_exit'])}")
    fixed = rep["depth_sweep"]["3"]["val_loss"]
    check("depth-sweep loss agrees with the per-step read-out at the same T",
          abs(fixed - dc[-1]) < 0.05 * max(abs(fixed), 1.0),
          f"sweep T=3: {fixed:.4f} vs depth_curve[-1]: {dc[-1]:.4f}")
    anchor = [r for r in rep["early_exit"]["confidence"] if r["threshold"] > 1.0]
    if anchor:
        check("early-exit anchor equals the fixed-depth result",
              abs(anchor[0]["loss"] - dc[-1]) < 1e-6,
              f"{anchor[0]['loss']:.6f} vs {dc[-1]:.6f}")
    shutil.rmtree(tmp, ignore_errors=True)


def test_bpb_consistency():
    from loopedlm.data import bpb_from_nll
    check("bits-per-byte formula", abs(bpb_from_nll(math.log(2), 1.0) - 1.0) < 1e-12,
          "1 nat/token at 1 byte/token == 1/ln2 bits/byte * ln2 = 1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    device = "cuda" if (a.cuda and torch.cuda.is_available()) else "cpu"
    print(f"running on {device}\n")
    print("model semantics")
    test_causality(device)
    test_window_mask(device)
    test_depth_rope_inverse(device)
    test_loop_count_effect(device)
    print("\ngradients")
    test_bptt_equivalence(device)
    print("\nbudgets")
    test_param_budget()
    test_extra_params_are_scale_free()
    print("\ndata")
    test_data()
    print("\nanalysis")
    test_early_exit_synthetic()
    test_strata_selection_bias()
    test_analysis_pipeline(device)
    print("\nlosses and evaluation")
    test_eval_determinism(device)
    test_losses(device)
    test_deep_sup_changes_gradients(device)
    test_bpb_consistency()
    print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
