"""Looped Qwen3-style decoder.

The model follows the Qwen3 recipe (pre-RMSNorm, QK-norm, GQA, SwiGLU, RoPE,
no biases, tied embeddings) but the middle of the network is a *core* block of
`n_core` layers that is applied `n_loops` times with shared weights:

    e  = Embed(idx)
    h  = Prelude(e)                     # optional, applied once
    for t = 1..T:  h = Loop(h, e, t)    # shared weights
    y  = Coda(ReadOut(h_1..h_T))
    logits = Head(RMSNorm(y))

Everything that turns this into one of our experimental variants lives in
`ModelConfig`: how the state is initialised, how the input is re-injected, how
the step index is signalled to the block (depth conditioning), how the update is
applied (plain residual / gated / fixed-norm / hyperspherical / heavy-ball),
whether the block may read the history of its own states (loop memory), and how
the final state is read out (last state / pooled trajectory).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """RMSNorm through the fused kernel, with the gain cast to the input dtype.

    Two measured traps, both worth ~5x here because a looped model evaluates
    4 * n_core * T norms per step:

    * the hand-written form (cast, pow, mean, add, rsqrt, mul, cast, mul) is
      eight kernel launches and measured 419us on a [16384, 512] input;
    * F.rms_norm with a float32 gain on a bfloat16 input cannot dispatch to the
      fused implementation ("Mismatch dtype between input and weight") and falls
      back to the same composite path: 351us versus 67us once the gain matches.

    Casting the gain is exactly what autocast already does to every Linear
    weight, so this makes the norms consistent with the rest of the model rather
    than special; the master parameter stays float32 for the optimiser.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise: bool = True):
        super().__init__()
        self.eps = eps
        self.dim = (dim,)
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if w is not None and w.dtype != x.dtype:
            w = w.to(x.dtype)
        return F.rms_norm(x, self.dim, w, self.eps)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv)                       # [S, hd/2]
    emb = torch.cat((freqs, freqs), dim=-1)             # [S, hd]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, S, hd]; cos/sin: [S, hd]
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class Attention(nn.Module):
    """Qwen3 attention: per-head QK RMSNorm, GQA, RoPE, no biases."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.hd = cfg.head_dim
        self.rep = cfg.n_heads // cfg.n_kv_heads
        self.gqa_mode = cfg.attn_gqa_mode
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_mask: Optional[torch.Tensor] = None):
        B, S, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, S, self.n_heads, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, S, self.n_kv, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # Expand the KV groups explicitly instead of passing enable_gqa=True: no
        # fused SDPA backend implements GQA on this build, so enable_gqa silently
        # falls back to the math path, which materialises the [B,H,S,S] score
        # matrix and measured 18x slower than repeating K/V (5254us vs 290us per
        # layer).  GQA here is a parameter-budget choice, not a speed one.
        gqa = self.gqa_mode == "enable_gqa"
        if self.rep > 1 and not gqa:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        if attn_mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=gqa)
        o = o.transpose(1, 2).reshape(B, S, self.n_heads * self.hd)
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """One Qwen3 layer, optionally modulated by a per-step (scale, gate) pair."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, mod=None, attn_mask=None):
        h = self.input_layernorm(x)
        if mod is not None:
            h = h * mod[0]
        a = self.self_attn(h, cos, sin, attn_mask)
        x = x + (a * mod[1] if mod is not None else a)
        h = self.post_attention_layernorm(x)
        if mod is not None:
            h = h * mod[2]
        m = self.mlp(h)
        x = x + (m * mod[3] if mod is not None else m)
        return x


class DepthAttention(nn.Module):
    """Attention of the current state over the history of its own loop states.

    Queries come from h_t at every sequence position; keys and values come from
    the same position's earlier states h_1..h_{t-1}.  The memory bank is
    detached, so this adds O(T) compute per token and O(T * d_mem) activation
    memory without creating a gradient path through the whole history.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        dm = cfg.depth_attn_dim
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.q_proj = nn.Linear(cfg.d_model, dm, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, dm, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, dm, bias=False)
        self.o_proj = nn.Linear(dm, cfg.d_model, bias=False)
        self.scale = dm ** -0.5

    def key_value(self, h: torch.Tensor):
        hn = self.norm(h)
        return self.k_proj(hn), self.v_proj(hn)

    def forward(self, h: torch.Tensor, mem_k: torch.Tensor, mem_v: torch.Tensor):
        # h: [B,S,d]; mem_*: [B,S,t,dm]
        q = self.q_proj(self.norm(h)).unsqueeze(-2)                  # [B,S,1,dm]
        att = (q * mem_k).sum(-1) * self.scale                       # [B,S,t]
        w = att.float().softmax(-1).to(mem_v.dtype).unsqueeze(-1)    # [B,S,t,1]
        o = (w * mem_v).sum(-2)                                      # [B,S,dm]
        return self.o_proj(o)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class LoopedQwen3(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.embed_tokens = nn.Embedding(cfg.vocab_size, d)
        # Puts the residual stream on an O(1) scale so that input re-injection and
        # additive depth codes stay comparable to the loop updates.
        self.embed_norm = RMSNorm(d, cfg.rms_norm_eps) if cfg.embed_norm else nn.Identity()
        self.prelude = nn.ModuleList([Block(cfg) for _ in range(cfg.n_prelude)])
        self.core = nn.ModuleList([Block(cfg) for _ in range(cfg.n_core)])
        self.coda = nn.ModuleList([Block(cfg) for _ in range(cfg.n_coda)])
        self.norm = RMSNorm(d, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # ---- input re-injection ----
        if cfg.inject_input == "add":
            self.inject_alpha = nn.Parameter(torch.ones(1))
        if cfg.inject_input == "adapter":
            self.inject_norm = RMSNorm(d, cfg.rms_norm_eps)
            self.inject_proj = nn.Linear(2 * d, d, bias=False)

        # ---- depth conditioning ----
        T = cfg.max_loops
        self.d_rot = 0
        if cfg.depth_cond == "learned_emb":
            self.depth_emb = nn.Parameter(torch.zeros(T, d))
        if cfg.depth_cond == "adaln":
            # per-step, per-core-layer RMSNorm gains + residual-branch gates
            self.ada_scale = nn.Parameter(torch.ones(T, cfg.n_core, 2, d))
            self.ada_gate = nn.Parameter(torch.ones(T, cfg.n_core, 2))
        if cfg.depth_cond == "depth_rope":
            self._build_depth_rope()

        # ---- update rule ----
        a0 = cfg.update_alpha_init if cfg.update_alpha_init > 0 else 1.0 / math.sqrt(cfg.n_loops)
        if cfg.update in ("gated", "normalized", "sphere"):
            self.step_alpha = nn.Parameter(torch.full((T,), float(a0)))
        if cfg.momentum and cfg.momentum_learn_beta:
            b = min(max(cfg.momentum_beta, 1e-3), 1 - 1e-3)
            self.mom_logit = nn.Parameter(torch.tensor(math.log(b / (1 - b))))
        if cfg.momentum and cfg.momentum_read:
            self.mom_gamma = nn.Parameter(torch.zeros(1))

        # ---- loop memory ----
        self.depth_attn = DepthAttention(cfg) if cfg.loop_memory == "depth_attn" else None

        # ---- read-out ----
        if cfg.readout == "pool_learned":
            self.pool_logits = nn.Parameter(torch.zeros(T + 1))
        if cfg.readout == "pool_gate":
            self.pool_query = nn.Parameter(torch.zeros(d))
            self.pool_norm = RMSNorm(d, cfg.rms_norm_eps)

        # ---- halting ----
        if cfg.halting == "ponder":
            self.halt_head = nn.Linear(d, 1, bias=True)
            nn.init.zeros_(self.halt_head.weight)
            nn.init.constant_(self.halt_head.bias, -2.0)

        self._rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._mask_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self.apply(self._init_weights)
        self._rescale_residual_init()
        if cfg.halting == "ponder":
            nn.init.zeros_(self.halt_head.weight)
            nn.init.constant_(self.halt_head.bias, -2.0)

    # ------------------------------------------------------------------
    def _build_depth_rope(self):
        cfg = self.cfg
        d_rot = int(cfg.d_model * cfg.depth_rope_frac) // 2 * 2
        self.d_rot = d_rot
        if d_rot == 0:
            return
        half = d_rot // 2
        inv = 1.0 / (cfg.depth_rope_theta ** (torch.arange(half).float() / half))
        t = torch.arange(cfg.max_loops).float()
        ang = torch.outer(t, inv)                       # [T, half]
        self.register_buffer("drope_cos", ang.cos(), persistent=False)
        self.register_buffer("drope_sin", ang.sin(), persistent=False)

    def _depth_rotate(self, h: torch.Tensor, t: int, inverse: bool = False):
        if self.cfg.depth_cond != "depth_rope" or self.d_rot == 0:
            return h
        ti = min(t, self.cfg.max_loops - 1)
        c = self.drope_cos[ti].to(h.dtype)
        s = self.drope_sin[ti].to(h.dtype)
        if inverse:
            s = -s
        x, rest = h[..., : self.d_rot], h[..., self.d_rot:]
        x1, x2 = x.chunk(2, dim=-1)
        y = torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
        return torch.cat((y, rest), dim=-1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rescale_residual_init(self):
        cfg = self.cfg
        if cfg.init_residual_scale != "inv_sqrt_depth":
            return
        eff = 2 * cfg.n_core * cfg.n_loops + 2 * (cfg.n_prelude + cfg.n_coda)
        s = 1.0 / math.sqrt(max(eff, 1))
        with torch.no_grad():
            for blocks in (self.prelude, self.core, self.coda):
                for b in blocks:
                    b.self_attn.o_proj.weight.mul_(s)
                    b.mlp.down_proj.weight.mul_(s)

    # ------------------------------------------------------------------
    def rope(self, S, device, dtype):
        need = max(S, self.cfg.max_seq_len)
        if self._rope is None or self._rope[0].shape[0] < need or self._rope[0].device != device:
            self._rope = build_rope_cache(need, self.cfg.head_dim, self.cfg.rope_theta, device, torch.float32)
        cos, sin = self._rope
        return cos[:S].to(dtype), sin[:S].to(dtype)

    def _window_mask(self, S: int, w: int, device) -> Optional[torch.Tensor]:
        if w >= S:
            return None
        key = (S, w)
        m = self._mask_cache.get(key)
        if m is None or m.device != device:
            i = torch.arange(S, device=device)
            m = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < w)
            m = m[None, None]
            self._mask_cache[key] = m
        return m

    def _window_for_step(self, t: int, T: int, S: int) -> Optional[int]:
        sch = self.cfg.attn_window_schedule
        if sch == "none":
            return None
        frac = (t + 1) / T
        if sch == "local2global":
            w = int(max(32, math.ceil(S * frac)))
        elif sch == "global2local":
            w = int(max(32, math.ceil(S * (1.0 - frac) + 1e-6)))
        else:
            raise ValueError(sch)
        return min(w, S)

    # ------------------------------------------------------------------
    def _mod_for(self, t: int, layer: int):
        if self.cfg.depth_cond != "adaln":
            return None
        ti = min(t, self.cfg.max_loops - 1)
        sc = self.ada_scale[ti, layer]
        g = self.ada_gate[ti, layer]
        return (sc[0], g[0], sc[1], g[1])

    def _sinusoid(self, t: int, d: int, device, dtype):
        half = d // 2
        inv = torch.exp(-math.log(10000.0) * torch.arange(half, device=device).float() / half)
        ang = t * inv
        return torch.cat((ang.sin(), ang.cos()), dim=-1).to(dtype)

    def _core_forward(self, h, e, t, T, cos, sin, mem_k, mem_v):
        """One loop iteration.  Returns the update `delta` (not the new state)."""
        cfg = self.cfg
        x = h

        # --- input re-injection ---
        if cfg.inject_input == "add":
            x = x + self.inject_alpha.to(x.dtype) * e
        elif cfg.inject_input == "adapter":
            x = self.inject_proj(torch.cat((self.inject_norm(x), e), dim=-1))

        # --- depth conditioning (additive variants) ---
        if cfg.depth_cond == "sinusoid":
            x = x + self._sinusoid(t, x.shape[-1], x.device, x.dtype)
        elif cfg.depth_cond == "learned_emb":
            x = x + self.depth_emb[min(t, cfg.max_loops - 1)].to(x.dtype)

        # --- loop memory ---
        if self.depth_attn is not None and mem_k is not None:
            x = x + self.depth_attn(x, mem_k, mem_v)

        # --- depth-RoPE conjugation: compute in a step-dependent frame ---
        rotated = cfg.depth_cond == "depth_rope"
        if rotated:
            x = self._depth_rotate(x, t, inverse=False)

        w = self._window_for_step(t, T, x.shape[1])
        attn_mask = self._window_mask(x.shape[1], w, x.device) if w is not None else None

        y = x
        for li, blk in enumerate(self.core):
            y = blk(y, cos, sin, self._mod_for(t, li), attn_mask)
        delta = y - x
        if rotated:
            delta = self._depth_rotate(delta, t, inverse=True)
        return delta

    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        n_loops: Optional[int] = None,
        bptt_last_k: Optional[int] = None,
        collect_states: bool = False,
        collect_stats: bool = False,
    ):
        cfg = self.cfg
        B, S = idx.shape
        T = int(n_loops if n_loops is not None else cfg.n_loops)
        k = cfg.bptt_last_k if bptt_last_k is None else bptt_last_k
        k = T if k <= 0 else min(k, T)

        e = self.embed_norm(self.embed_tokens(idx))
        cos, sin = self.rope(S, idx.device, e.dtype)

        h = e
        for blk in self.prelude:
            h = blk(h, cos, sin)

        if cfg.state_init == "zeros":
            h = torch.zeros_like(h)
        elif cfg.state_init == "randn":
            h = torch.randn_like(h) * cfg.state_init_std

        vel = torch.zeros_like(h) if cfg.momentum else None
        keep_traj = collect_states or cfg.readout != "last" or cfg.halting != "none"
        states: List[torch.Tensor] = [h] if keep_traj else []
        stats = {"delta_rms": [], "state_rms": [], "cos_prev": []} if collect_stats else None
        prev_delta = None
        deltas_for_reg: List[torch.Tensor] = []
        mem_list_k: List[torch.Tensor] = []
        mem_list_v: List[torch.Tensor] = []

        for t in range(T):
            grad_on = torch.is_grad_enabled() and (t >= T - k)
            mem_k = mem_v = None
            if self.depth_attn is not None and mem_list_k:
                mem_k = torch.stack(mem_list_k, dim=-2)
                mem_v = torch.stack(mem_list_v, dim=-2)

            with torch.set_grad_enabled(grad_on):
                h_in = h
                if cfg.momentum and cfg.momentum_read and vel is not None:
                    h_in = h_in + self.mom_gamma * vel
                if cfg.noise_std > 0 and self.training:
                    h_in = h_in + self._noise(h_in, t, T)

                if cfg.grad_ckpt and grad_on and self.training:
                    delta = checkpoint(
                        self._core_forward, h_in, e, t, T, cos, sin, mem_k, mem_v,
                        use_reentrant=False,
                    )
                else:
                    delta = self._core_forward(h_in, e, t, T, cos, sin, mem_k, mem_v)

                if cfg.loop_dropout > 0 and self.training:
                    keep = (torch.rand(B, 1, 1, device=h.device) > cfg.loop_dropout).to(delta.dtype)
                    delta = delta * keep

                h, vel = self._apply_update(h, delta, vel, t)

            if cfg.decorr_weight > 0 and grad_on:
                deltas_for_reg.append(delta)
            if self.depth_attn is not None:
                with torch.no_grad():
                    mk, mv = self.depth_attn.key_value(h.detach())
                mem_list_k.append(mk)
                mem_list_v.append(mv)
            if keep_traj:
                states.append(h)
            if collect_stats:
                with torch.no_grad():
                    stats["delta_rms"].append(delta.float().pow(2).mean().sqrt().item())
                    stats["state_rms"].append(h.float().pow(2).mean().sqrt().item())
                    if prev_delta is not None:
                        a = delta.float().flatten(0, 1)
                        b = prev_delta.float().flatten(0, 1)
                        stats["cos_prev"].append(F.cosine_similarity(a, b, dim=-1).mean().item())
                    prev_delta = delta

        out = self._readout(states, h)
        for blk in self.coda:
            out = blk(out, cos, sin)

        res = {"hidden": out, "n_loops": T, "cos": cos, "sin": sin}
        if keep_traj:
            res["traj"] = states
        if collect_stats:
            res["stats"] = stats
        if cfg.decorr_weight > 0 and deltas_for_reg:
            res["decorr"] = self._decorr(deltas_for_reg)
        return res

    # ------------------------------------------------------------------
    def _noise(self, h, t, T):
        cfg = self.cfg
        if cfg.noise_schedule == "const":
            s = 1.0
        elif cfg.noise_schedule == "linear_decay":
            s = max(0.0, 1.0 - t / max(T - 1, 1))
        elif cfg.noise_schedule == "sqrt_decay":
            s = 1.0 / math.sqrt(t + 1)
        else:
            raise ValueError(cfg.noise_schedule)
        rms = h.float().pow(2).mean(-1, keepdim=True).sqrt().to(h.dtype)
        return torch.randn_like(h) * (cfg.noise_std * s) * rms

    def _apply_update(self, h, delta, vel, t):
        cfg = self.cfg
        a = None
        if cfg.update in ("gated", "normalized", "sphere"):
            a = self.step_alpha[min(t, cfg.max_loops - 1)]
        if cfg.update == "residual":
            step = delta
        elif cfg.update == "gated":
            step = a * delta
        elif cfg.update in ("normalized", "sphere"):
            rms = delta.float().pow(2).mean(-1, keepdim=True).add(1e-8).sqrt().to(delta.dtype)
            step = a * delta / rms
        else:
            raise ValueError(cfg.update)

        if cfg.momentum:
            beta = torch.sigmoid(self.mom_logit) if cfg.momentum_learn_beta else cfg.momentum_beta
            vel = beta * vel + step
            step = vel
        h = h + step
        if cfg.update == "sphere":
            rms = h.float().pow(2).mean(-1, keepdim=True).add(1e-8).sqrt().to(h.dtype)
            h = h / rms
        return h, vel

    def _readout(self, states, h):
        cfg = self.cfg
        if cfg.readout == "last":
            return h
        St = torch.stack(states, dim=0)                        # [T+1,B,S,d]
        if cfg.readout == "pool_learned":
            w = self.pool_logits[: St.shape[0]].float().softmax(0).to(St.dtype)
            return (St * w[:, None, None, None]).sum(0)
        if cfg.readout == "pool_gate":
            sc = (self.pool_norm(St) * self.pool_query.to(St.dtype)).sum(-1) * (St.shape[-1] ** -0.5)
            w = sc.float().softmax(0).to(St.dtype)             # [T+1,B,S]
            return (St * w.unsqueeze(-1)).sum(0)
        raise ValueError(cfg.readout)

    def _decorr(self, deltas):
        loss = deltas[0].new_zeros((), dtype=torch.float32)
        n = 0
        for a, b in zip(deltas[:-1], deltas[1:]):
            x = a.float().flatten(0, 1)
            y = b.float().flatten(0, 1)
            loss = loss + F.cosine_similarity(x, y, dim=-1).abs().mean()
            n += 1
        return loss / max(n, 1)

    # ------------------------------------------------------------------
    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden))

    def readout_head(self, h: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        """Logits from an arbitrary loop state (used for deep supervision / early exit)."""
        for blk in self.coda:
            h = blk(h, cos, sin)
        return self.lm_head(self.norm(h))

    def halt_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.halt_head(self.norm(hidden)).squeeze(-1)

    # ------------------------------------------------------------------
    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_tokens.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def flops_per_token(self, n_loops: Optional[int] = None, seq_len: Optional[int] = None) -> float:
        """Forward FLOPs per token (matmuls + attention scores, 2 FLOPs per MAC)."""
        cfg = self.cfg
        T = n_loops if n_loops is not None else cfg.n_loops
        S = seq_len if seq_len is not None else cfg.max_seq_len
        d, hd = cfg.d_model, cfg.head_dim
        proj = 2 * (d * cfg.n_heads * hd + 2 * d * cfg.n_kv_heads * hd + cfg.n_heads * hd * d)
        score = 2 * 2 * cfg.n_heads * hd * (S / 2.0)      # causal: half the keys on average
        mlp = 2 * 3 * d * cfg.intermediate_size
        per_layer = proj + score + mlp
        n_once = cfg.n_prelude + cfg.n_coda
        f = per_layer * (n_once + T * cfg.n_core)
        if cfg.loop_memory == "depth_attn":
            dm = cfg.depth_attn_dim
            f += T * (2 * (3 * d * dm + dm * d) + 2 * 2 * dm * (T / 2.0))
        if cfg.inject_input == "adapter":
            f += T * 2 * 2 * d * d
        f += 2 * d * cfg.vocab_size
        return f
