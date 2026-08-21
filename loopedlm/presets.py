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

# --- P. the depth-RoPE frontier ---------------------------------------------
# First mechanism to beat the naive optimum: at T=16, rotating a fraction of the
# channels by a step-dependent angle gives 4.1612 at frac=0.25 and 4.1405 at
# frac=0.5, against 4.2346 for the plain T=16 loop and 4.2028 for the best naive
# depth (T=8).  Zero learned parameters -- and the *learned* versions of the same
# idea (per-step embedding 4.2663, per-step adaLN 4.2634) both HURT, so what helps
# is breaking the symmetry between steps structurally, not adding capacity.
# More rotation was better than less, so the fraction and the angle scale are the
# axes to push, and the payoff should be largest exactly where naive looping fails.
for _f in (0.75, 1.0):
    add(f"P_drope_f{_f}_T16", {"depth_cond": "depth_rope", "depth_rope_frac": _f})
for _f in (0.25, 0.5, 1.0):
    add(f"P_drope_f{_f}_T32", {"n_loops": 32, "depth_cond": "depth_rope",
                               "depth_rope_frac": _f})
add("P_drope_f0.5_T64", {"n_loops": 64, "depth_cond": "depth_rope", "depth_rope_frac": 0.5})
add("P_drope_f0.5_T8", {"n_loops": 8, "depth_cond": "depth_rope", "depth_rope_frac": 0.5})
# the angle scale: theta controls how fast the frame turns per step
add("P_drope_theta100_T32", {"n_loops": 32, "depth_cond": "depth_rope",
                             "depth_rope_frac": 0.5, "depth_rope_theta": 100.0})
add("P_drope_theta10k_T32", {"n_loops": 32, "depth_cond": "depth_rope",
                             "depth_rope_frac": 0.5, "depth_rope_theta": 10000.0})
# does it compose with the mechanisms that fix the trajectory geometry?
add("P_drope_orthosphere", {"n_loops": 32, "depth_cond": "depth_rope",
                            "depth_rope_frac": 0.5, "update": "orthogonal_sphere"})
add("P_drope_geodesic", {"n_loops": 32, "depth_cond": "depth_rope",
                         "depth_rope_frac": 0.5, "update": "geodesic"})
add("P_drope_pool", {"n_loops": 32, "depth_cond": "depth_rope", "depth_rope_frac": 0.5,
                     "readout": "pool_gate"})
add("P_drope_chains", {"n_chains": 4, "n_loops": 8, "depth_cond": "depth_rope",
                       "depth_rope_frac": 0.5})
# and does it let the model extrapolate past its training depth?
add("P_drope_uniform", {"loop_sampling": "uniform", "loop_min": 1, "n_loops": 32,
                        "depth_cond": "depth_rope", "depth_rope_frac": 0.5})

# --- N. the state does not have to live in a flat space ---------------------
# The residual stream is a flat vector space and the update is addition, which is
# precisely what degenerates into a straight-line drift.  These change the geometry
# instead of the step size, and each is parameter-free apart from one learned angle
# or step length per loop.
#
#   geodesic   - exact rotation of the sphere towards the block's direction.  The
#                norm is preserved algebraically, and "distance travelled" becomes a
#                bounded angle rather than an unbounded norm.
#   phase      - the state is d/2 unit complex numbers and the loop rotates their
#                phases.  On a torus nothing can grow and no drift exists; the loop
#                can only redistribute phase, over an exponentially large reachable
#                set.
#   hyperbolic - the state lives in the Poincare ball.  Volume grows exponentially
#                with radius, so steps of fixed hyperbolic length keep reaching new
#                territory instead of retracing a direction -- and negatively curved
#                space embeds hierarchies with low distortion, which language has.
add("N_geodesic_T8", {"n_loops": 8, "update": "geodesic"})
add("N_geodesic_T32", {"n_loops": 32, "update": "geodesic"})
add("N_geodesic_T64", {"n_loops": 64, "update": "geodesic"})
add("N_phase_T8", {"n_loops": 8, "update": "phase"})
add("N_phase_T32", {"n_loops": 32, "update": "phase"})
add("N_hyper_T8", {"n_loops": 8, "update": "hyperbolic", "state_init": "zeros"})
add("N_hyper_T32", {"n_loops": 32, "update": "hyperbolic", "state_init": "zeros"})
add("N_hyper_T32_c025", {"n_loops": 32, "update": "hyperbolic", "state_init": "zeros",
                         "curvature": 0.25})
add("N_hyper_T64", {"n_loops": 64, "update": "hyperbolic", "state_init": "zeros"})
add("N_geodesic_uniform", {"loop_sampling": "uniform", "loop_min": 1, "n_loops": 32,
                           "update": "geodesic"})
add("N_geodesic_chains", {"n_chains": 4, "n_loops": 8, "update": "geodesic"})
add("N_geodesic_pool", {"n_loops": 32, "update": "geodesic", "readout": "pool_gate"})

# --- R. radical departures, all compute-matched to the naive T=32 run -------
# Naive looping peaks at T=8 (4.2028) and T=32 is worse (4.2880), while the
# trajectory diagnostics say the loop loses dimensionality rather than motion.
# These abandon a premise rather than tuning a knob, and each is matched in FLOPs
# to naive T=32 so the comparison is about *how* the compute is spent.
#
# 1. Forced turning: make collinearity geometrically impossible instead of
#    penalising it.  Zero parameters.
#    A 2x2 that separates the two pathologies, which the loss curve alone cannot:
#      growing norm + collinear steps = residual  (A_depth32)
#      fixed norm   + collinear steps = sphere    (R_sphere_T32)
#      growing norm + orthogonal steps= orthogonal
#      fixed norm   + orthogonal steps= orthogonal_sphere
#    Measured before running: a spherical update alone does NOT decorrelate the
#    steps (cos stays 0.99 -- a great circle is still "straight"), while explicit
#    projection drives the applied-step cosine to exactly 0.  So the pair isolates
#    whether depth is wasted on norm or on direction.
add("R_sphere_T32", {"n_loops": 32, "update": "sphere"})
add("R_ortho_T32", {"n_loops": 32, "update": "orthogonal"})
add("R_orthosphere_T32", {"n_loops": 32, "update": "orthogonal_sphere"})
add("R_orthosphere_T64", {"n_loops": 64, "update": "orthogonal_sphere"})
add("R_orthosphere_T8", {"n_loops": 8, "update": "orthogonal_sphere"})
#
# 2. Spend the FLOPs on breadth instead of depth: several shorter trajectories
#    from different learned starting points, combined at the read-out.  4x8 and
#    8x4 both cost exactly what one 32-loop chain costs.
add("R_chains4_T8", {"n_chains": 4, "n_loops": 8})
add("R_chains4_T8_gate", {"n_chains": 4, "n_loops": 8, "chain_combine": "gate"})
add("R_chains8_T4", {"n_chains": 8, "n_loops": 4})
add("R_chains2_T16", {"n_chains": 2, "n_loops": 16})
#
# 3. The loop as annealing rather than accumulation: start from noise and let the
#    block denoise it over the steps.  Diffusion models are the existence proof
#    that hundreds of sequential applications of one network keep paying; if that
#    mechanism transfers, depth should stop saturating.
add("R_anneal_T32", {"n_loops": 32, "state_init": "randn", "state_init_std": 1.0,
                     "noise_std": 0.5, "noise_schedule": "linear_decay"})
add("R_anneal_sqrt_T32", {"n_loops": 32, "state_init": "randn", "state_init_std": 1.0,
                          "noise_std": 0.5, "noise_schedule": "sqrt_decay"})
#
# 4. Combinations of whichever of the above survives.
add("R_orthosphere_chains", {"n_chains": 4, "n_loops": 8, "update": "orthogonal_sphere"})
add("R_orthosphere_uniform", {"loop_sampling": "uniform", "loop_min": 1, "n_loops": 32,
                              "update": "orthogonal_sphere"})

# --- M. a second model size, to make the scaling claim a measurement --------
# Every other result comes from one model size, so any statement about scaling is
# an argument about asymptotic cost rather than an observed trend.  This repeats
# the saturation curve at ~3.5M parameters (d_model 256, same recipe, same token
# budget).  The question is whether the useful depth moves with model size: if the
# optimum stays at T=8 in both, the ceiling is a property of the data and the
# recipe, not of capacity -- which is a result either way.
_SMALL = dict(d_model=256, n_heads=4, head_dim=64, n_kv_heads=1, intermediate_size=704)
for _t in (2, 4, 8, 16, 32):
    add(f"M_small_T{_t}", dict(_SMALL, n_loops=_t))

# --- L. combinations suggested by the measured failure mode -----------------
# Diagnosis on A_depth16: updates do NOT decay (RMS ~1.13, no fixed point) but
# become collinear -- cos(delta_t, delta_{t-1}) reaches 0.999 by t=16 -- while the
# state norm grows linearly. The loop loses dimensionality rather than motion, so
# the interesting combinations are the ones that force each step into a new
# direction and stop the norm from growing.
add("L_decorr0.05", {"decorr_weight": 0.05})
add("L_decorr0.1", {"decorr_weight": 0.1})
add("L_decorr_norm", {"decorr_weight": 0.05, "update": "normalized"})
add("L_decorr_sphere", {"decorr_weight": 0.05, "update": "sphere"})
add("L_decorr_drope", {"decorr_weight": 0.05, "depth_cond": "depth_rope"})
add("L_norm_drope", {"update": "normalized", "depth_cond": "depth_rope"})
add("L_sphere_adaln", {"update": "sphere", "depth_cond": "adaln"})
add("L_pool_norm", {"readout": "pool_gate", "update": "normalized"})
add("L_uniform_drope", {"loop_sampling": "uniform", "loop_min": 1, "n_loops": 32,
                        "depth_cond": "depth_rope"})
add("L_uniform_norm_decorr", {"loop_sampling": "uniform", "loop_min": 1, "n_loops": 32,
                              "update": "normalized", "decorr_weight": 0.05})

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
    nc = model_cfg.get("n_core", 2)
    # Parallel chains multiply the looped work; without this a 4-chain config was
    # estimated at a quarter of its real cost and the queue plan under-reported.
    chains = model_cfg.get("n_chains", 1)
    layers = T * nc * chains + model_cfg.get("n_prelude", 0) + model_cfg.get("n_coda", 0)
    if model_cfg.get("bptt_last_k"):
        k = min(model_cfg["bptt_last_k"], T)
        layers = (k * nc + (T - k) * nc * 0.25) * chains
    rate = tok_s_ref * REF_LAYERS / max(layers, 1)
    return tokens / rate / 60 * 1.25 + 5
