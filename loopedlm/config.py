"""Configuration objects for looped-transformer pretraining experiments."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field, fields
from typing import Optional


@dataclass
class ModelConfig:
    """Qwen3-style decoder whose *core* block is applied `n_loops` times.

    The flags below are the knobs of the ablation study; every mechanism can be
    switched off independently so that a run reduces to a plain (looped) Qwen3.
    """

    # ---------------- shape ----------------
    vocab_size: int = 8192
    max_seq_len: int = 512
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 1280
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attn_gqa_mode: str = "repeat"   # repeat | enable_gqa (see Attention.forward)
    tie_embeddings: bool = True
    embed_norm: bool = True

    # ---------------- depth structure ----------------
    n_prelude: int = 0          # non-looped layers applied once, before the loop
    n_core: int = 2             # layers inside the looped block
    n_coda: int = 0             # non-looped layers applied once, after the loop
    n_loops: int = 16           # T used at train time (mean if loop_sampling != fixed)
    max_loops: int = 64         # size of per-step parameter tables / eval range

    # ---------------- state dynamics ----------------
    state_init: str = "embed"        # embed | zeros | randn
    state_init_std: float = 0.4
    inject_input: str = "add"        # none | add | adapter
    # residual | gated | normalized | sphere
    #   orthogonal        - project the update off the previous update direction,
    #                       so successive steps cannot be collinear by construction
    #   orthogonal_sphere - also project off the radial direction (a step tangent to
    #                       the sphere) and renormalise, so the norm cannot grow
    #                       either: the trajectory is forced to turn every step
    update: str = "residual"
    update_alpha_init: float = -1.0  # <0 -> 1/sqrt(n_loops)
    momentum: bool = False           # heavy-ball on the residual stream
    momentum_beta: float = 0.9
    momentum_learn_beta: bool = True
    momentum_read: bool = False      # block also reads the velocity

    # ---------------- breaking stationarity ----------------
    depth_cond: str = "none"         # none | sinusoid | learned_emb | adaln | depth_rope
    depth_rope_frac: float = 0.25
    depth_rope_theta: float = 1000.0
    attn_window_schedule: str = "none"   # none | coarse2fine | fine2coarse

    # ---------------- loop memory ----------------
    loop_memory: str = "none"        # none | depth_attn
    depth_attn_dim: int = 64

    # ---------------- parallel chains instead of depth ----------------
    # Sequential depth stops paying after T=8 on this setup, so the same FLOPs can
    # be spent on several independent shorter trajectories from different random
    # starts, combined at the read-out.  Compute scales with n_chains, parameters
    # do not.  This asks whether the budget is better spent exploring than going
    # deeper, which is a different answer to "FLOPs per parameter" than looping.
    n_chains: int = 1
    chain_combine: str = "mean"      # mean | gate  (gate = learned per-token weights)
    chain_init_noise: float = 0.5    # spread of the per-chain starting perturbation

    # ---------------- read-out ----------------
    readout: str = "last"            # last | pool_learned | pool_gate

    # ---------------- exploration ----------------
    noise_std: float = 0.0
    noise_schedule: str = "const"    # const | linear_decay | sqrt_decay
    loop_dropout: float = 0.0

    # ---------------- train-time loop schedule ----------------
    loop_sampling: str = "fixed"     # fixed | uniform | poisson
    loop_min: int = 1
    bptt_last_k: int = 0             # 0 -> full BPTT (gradient checkpointed)
    grad_ckpt: bool = True

    # ---------------- supervision ----------------
    deep_supervision: str = "none"   # none | random_k | all
    deep_sup_k: int = 2
    deep_sup_weight: float = 0.3
    deep_sup_pos_frac: float = 0.25
    deep_sup_detach_teacher: bool = False   # KL to the final step instead of CE
    halting: str = "none"            # none | ponder
    halting_weight: float = 0.01
    halting_target_loops: float = 8.0

    # ---------------- regularisers on the trajectory ----------------
    decorr_weight: float = 0.0       # penalise cos(delta_t, delta_{t-1})
    init_residual_scale: str = "one"  # one | inv_sqrt_depth

    def __post_init__(self):
        assert self.n_heads * self.head_dim == self.d_model, (
            f"n_heads*head_dim ({self.n_heads}*{self.head_dim}) must equal d_model ({self.d_model})"
        )
        assert self.n_heads % self.n_kv_heads == 0
        assert self.max_loops >= self.n_loops

    # ------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown ModelConfig keys: {sorted(unknown)}")
        return cls(**d)


# Paths and block compilation come from the environment so the same code runs
# unchanged on the Windows box (no Triton, so no compile) and on a Linux GPU.
_OUT_DIR = os.environ.get("LOOPEDLM_OUT", r"C:\ml\looped-lm\runs")
_DATA_DIR = os.environ.get("LOOPEDLM_DATA", r"C:\ml\looped-lm\data\tok8192")
_COMPILE = os.environ.get("LOOPEDLM_COMPILE", "0") == "1"


@dataclass
class TrainConfig:
    run_name: str = "dev"
    out_dir: str = _OUT_DIR
    data_dir: str = _DATA_DIR

    total_tokens: int = 100_000_000   # hard budget of *training* tokens
    seq_len: int = 512
    micro_batch: int = 32             # sequences per forward
    grad_accum: int = 1

    lr: float = 1.5e-3
    min_lr_frac: float = 0.05
    warmup_frac: float = 0.02
    schedule: str = "cosine"          # cosine | wsd
    wsd_decay_frac: float = 0.2
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    lr_scale_with_loops: str = "none"  # none | inv_sqrt

    param_budget: int = 10_000_000
    seed: int = 1337
    dtype: str = "bfloat16"
    compile: bool = _COMPILE

    eval_every: int = 500
    eval_tokens: int = 2_000_000      # tokens used for periodic validation
    final_eval_tokens: int = 8_000_000
    log_every: int = 20
    save_best: bool = True
    save_last: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown TrainConfig keys: {sorted(unknown)}")
        return cls(**d)
