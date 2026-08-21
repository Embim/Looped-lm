"""The experiment program.

Every entry is a pair of override dicts on top of BASE_MODEL / BASE_TRAIN, so a
preset name is a complete, reproducible description of a run.  Groups are what
`scripts/run_experiments.py` schedules.

Screening runs use a reduced token budget (SCREEN_TOKENS) because ~50 configs at
the full budget would not fit in the compute we have; the finalists are then
re-run at the full 100M-token budget.  The reduced budget is identical for every
screened config, so comparisons inside a group are fair.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

SCREEN_TOKENS = 25_000_000
FULL_TOKENS = 100_000_000

# ---------------------------------------------------------------------------
# Parameter allocation.  vocab 8192 x d 512 tied embeddings = 4.19M, leaving
# 5.25M for the two-layer looped core -> 9.44M total, inside the 10M budget.
BASE_MODEL: Dict = dict(
    vocab_size=8192, max_seq_len=512,
    d_model=512, n_heads=8, n_kv_heads=2, head_dim=64, intermediate_size=1280,
    n_prelude=0, n_core=2, n_coda=0, n_loops=16, max_loops=64,
    state_init="embed", inject_input="add", update="residual",
    depth_cond="none", readout="last", loop_memory="none",
    loop_sampling="fixed", bptt_last_k=0, grad_ckpt=True,
)

BASE_TRAIN: Dict = dict(
    total_tokens=SCREEN_TOKENS, seq_len=512, micro_batch=32, grad_accum=1,
    lr=1.5e-3, schedule="cosine", warmup_frac=0.03, weight_decay=0.1,
    eval_every=250, eval_tokens=2_000_000, final_eval_tokens=4_000_000,
    log_every=25,
)

DEPTH_SWEEP = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]

# ---------------------------------------------------------------------------
P: Dict[str, Tuple[Dict, Dict]] = {}


def add(name: str, model: Dict | None = None, train: Dict | None = None):
    P[name] = (model or {}, train or {})


# --- A. baselines: how much does looping buy at all, and where does it stop? --
add("A_depth1", {"n_loops": 1})
add("A_depth2", {"n_loops": 2})
add("A_depth4", {"n_loops": 4})
add("A_depth8", {"n_loops": 8})
add("A_depth16", {"n_loops": 16})
add("A_depth32", {"n_loops": 32})
add("A_depth48", {"n_loops": 48})
add("A_depth64", {"n_loops": 64})
# out-of-budget references: unshared depth at the same effective layer count
add("A_ref_unshared8", {"n_core": 8, "n_loops": 1}, {"param_budget": 40_000_000})
add("A_ref_unshared16", {"n_core": 16, "n_loops": 1}, {"param_budget": 60_000_000})

# --- B. is the "saturation" of deep loops just an untuned learning rate? -----
for _lr in (5e-4, 1e-3, 2e-3, 3e-3):
    add(f"B_lr{_lr:g}_T8", {"n_loops": 8}, {"lr": _lr})
    add(f"B_lr{_lr:g}_T32", {"n_loops": 32}, {"lr": _lr})
add("B_lrscale_T32", {"n_loops": 32}, {"lr_scale_with_loops": "inv_sqrt"})
add("B_initscale_T32", {"n_loops": 32, "init_residual_scale": "inv_sqrt_depth"})

# --- C. breaking stationarity: tell the block which step it is on ------------
add("C_sinusoid", {"depth_cond": "sinusoid"})
add("C_learned_emb", {"depth_cond": "learned_emb"})
add("C_adaln", {"depth_cond": "adaln"})
add("C_depth_rope", {"depth_cond": "depth_rope"})
add("C_depth_rope_half", {"depth_cond": "depth_rope", "depth_rope_frac": 0.5})
add("C_window_l2g", {"attn_window_schedule": "local2global"})
add("C_window_g2l", {"attn_window_schedule": "global2local"})

# --- D. the update rule: stop the trajectory from collapsing -----------------
add("D_gated", {"update": "gated"})
add("D_normalized", {"update": "normalized"})
add("D_sphere", {"update": "sphere"})
add("D_momentum", {"momentum": True})
add("D_momentum_read", {"momentum": True, "momentum_read": True})
add("D_norm_momentum", {"update": "normalized", "momentum": True, "momentum_read": True})

# --- E. read-out: use the whole trajectory, not just its end point -----------
add("E_pool_learned", {"readout": "pool_learned"})
add("E_pool_gate", {"readout": "pool_gate"})

# --- F. loop memory: give the recurrence more than d dimensions of state -----
add("F_depth_attn", {"loop_memory": "depth_attn"})
add("F_depth_attn_wide", {"loop_memory": "depth_attn", "depth_attn_dim": 128})

# --- G. input injection and state initialisation -----------------------------
add("G_inject_none", {"inject_input": "none"})
add("G_inject_adapter", {"inject_input": "adapter"})
add("G_init_zeros", {"state_init": "zeros"})
add("G_init_randn", {"state_init": "randn"})

# --- H. how to spend the parameter budget (all ~9.4M total) -----------------
add("H_d384_L4", {"d_model": 384, "n_heads": 6, "intermediate_size": 1024, "n_core": 4,
                  "n_loops": 8})
add("H_d640_L1", {"d_model": 640, "n_heads": 10, "intermediate_size": 1664, "n_core": 1,
                  "n_loops": 32})
add("H_prelude", {"n_prelude": 1, "n_core": 1})
add("H_coda", {"n_core": 1, "n_coda": 1})

# --- I. supervision of the intermediate steps -------------------------------
add("I_deepsup_k2", {"deep_supervision": "random_k", "deep_sup_k": 2})
add("I_deepsup_all", {"deep_supervision": "all", "deep_sup_weight": 0.2})
# Self-distillation across depth.  KL(step_t || final) is ~1e-4 at initialisation
# (an untrained head barely differs between steps) and grows as the steps
# differentiate, so unlike the CE variant its weight has to be set for the
# converged regime, not the initial one -- hence two weights.
add("I_deepsup_kl", {"deep_supervision": "random_k", "deep_sup_k": 2,
                     "deep_sup_detach_teacher": True, "deep_sup_weight": 1.0})
add("I_deepsup_kl_strong", {"deep_supervision": "random_k", "deep_sup_k": 2,
                            "deep_sup_detach_teacher": True, "deep_sup_weight": 10.0})
add("I_ponder", {"halting": "ponder", "deep_supervision": "random_k", "deep_sup_k": 2})

# --- J. exploration inside the loop ----------------------------------------
add("J_noise_const", {"noise_std": 0.05})
add("J_noise_decay", {"noise_std": 0.1, "noise_schedule": "linear_decay"})
add("J_loop_dropout", {"loop_dropout": 0.1})
add("J_decorr", {"decorr_weight": 0.02})

# --- K. training-time loop schedule and truncated backprop ------------------
add("K_uniform_T", {"loop_sampling": "uniform", "loop_min": 1, "n_loops": 32})
add("K_poisson_T", {"loop_sampling": "poisson", "loop_min": 4, "n_loops": 16})
add("K_bptt8_T32", {"n_loops": 32, "bptt_last_k": 8})
add("K_bptt8_T64", {"n_loops": 64, "bptt_last_k": 8})
add("K_bptt4_T64", {"n_loops": 64, "bptt_last_k": 4})

# When the card is contended, run this order rather than the full 59: the
# saturation curve first (it is the baseline everything else is measured against),
# then the cheapest decisive test of the LR confound, then one representative of
# each mechanism family, then the rest of each family.
CORE: List[str] = [
    "A_depth1", "A_depth4", "A_depth16", "A_depth32",
    "B_lr0.003_T32", "B_lr0.0005_T32", "B_initscale_T32",
    "C_adaln", "C_depth_rope", "D_normalized", "D_momentum_read",
    "E_pool_gate", "F_depth_attn", "K_uniform_T", "K_bptt8_T64",
]

GROUPS: Dict[str, List[str]] = {}
for _k in P:
    GROUPS.setdefault(_k.split("_")[0], []).append(_k)
GROUPS["CORE"] = CORE


def resolve(name: str, model_over: Dict | None = None, train_over: Dict | None = None):
    """Return (model_kwargs, train_kwargs) for a preset name plus ad-hoc overrides."""
    mo, to = P.get(name, ({}, {}))
    m = deepcopy(BASE_MODEL); m.update(mo); m.update(model_over or {})
    t = deepcopy(BASE_TRAIN); t.update(to); t.update(train_over or {})
    return m, t


# Measured throughput reference: tokens/s at n_loops=16, n_core=2, B=32, S=512 on
# an RTX 5080 with an idle card (311 ms/step).  Step time is linear in T * n_core
# (verified from T=8 to T=64), so everything scales off this one number.  Under
# truncated BPTT the no-grad steps cost about a quarter of a full step, which is
# what the bptt branch below models (measured: T=64 bptt=8 -> 455 ms vs 1332 ms).
TOK_S_REF = 52700.0
REF_LAYERS = 16 * 2


def estimate_minutes(model_cfg: Dict, tokens: int, tok_s_ref: float = TOK_S_REF) -> float:
    """Wall-clock estimate for a run, used for honest board reservations."""
    T = model_cfg.get("n_loops", 16)
    if model_cfg.get("loop_sampling") == "uniform":
        T = (model_cfg.get("loop_min", 1) + T) / 2
    layers = T * model_cfg.get("n_core", 2) + model_cfg.get("n_prelude", 0) + model_cfg.get("n_coda", 0)
    if model_cfg.get("bptt_last_k"):
        k = min(model_cfg["bptt_last_k"], T)
        layers = (k * model_cfg.get("n_core", 2) * 1.0 + (T - k) * model_cfg.get("n_core", 2) * 0.25)
    rate = tok_s_ref * REF_LAYERS / max(layers, 1)
    return tokens / rate / 60 * 1.25 + 5
