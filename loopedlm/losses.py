"""Loss assembly: next-token CE plus the optional per-loop auxiliaries."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import ModelConfig


def logits_at(model, state: torch.Tensor, cos, sin, pos: Optional[torch.Tensor]) -> torch.Tensor:
    """Logits from one loop state, optionally only at a subset of flat positions.

    The coda (if any) needs the full sequence, so it runs first; the vocabulary
    projection -- which is what actually costs memory -- runs only on `pos`.
    """
    h = state
    for blk in model.coda:
        h = blk(h, cos, sin)
    h = model.norm(h).reshape(-1, h.shape[-1])
    if pos is not None:
        h = h.index_select(0, pos)
    return model.lm_head(h)


def _pick_steps(n_states: int, cfg: ModelConfig, gen: torch.Generator) -> List[int]:
    # States are h_0..h_T.  h_0 carries no computation, so it is never supervised.
    # h_T is excluded when the read-out is the last state: it is already the main
    # CE target, and for the self-distillation variant it *is* the teacher, so
    # picking it would spend one of the k slots on a guaranteed zero.
    last = n_states - 1 if cfg.readout == "last" else n_states
    cand = list(range(1, last))
    if not cand:
        cand = list(range(1, n_states))
    if cfg.deep_supervision == "all":
        return cand
    k = min(cfg.deep_sup_k, len(cand))
    if k <= 0:
        return []
    idx = torch.randperm(len(cand), generator=gen).tolist()[:k]
    return sorted(cand[i] for i in idx)


def compute_loss(
    model,
    out: Dict,
    targets: torch.Tensor,
    gen: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    cfg: ModelConfig = model.cfg
    V = cfg.vocab_size
    logits = model.head(out["hidden"])
    ce = F.cross_entropy(logits.float().reshape(-1, V), targets.reshape(-1))
    total = ce
    logs = {"ce": float(ce.detach())}

    flat_t = targets.reshape(-1)
    n_flat = flat_t.numel()
    pos = None
    needs_steps = cfg.deep_supervision != "none" or cfg.halting == "ponder"
    if needs_steps:
        n_pos = max(int(n_flat * cfg.deep_sup_pos_frac), 256)
        if n_pos < n_flat:
            pos = torch.randint(0, n_flat, (n_pos,), device=targets.device, generator=gen)
        tgt_sub = flat_t.index_select(0, pos) if pos is not None else flat_t

    traj: List[torch.Tensor] = out.get("traj", [])
    cos, sin = out["cos"], out["sin"]

    # ---- deep supervision: every supervised step must already be decodable ----
    if cfg.deep_supervision != "none" and traj:
        steps = _pick_steps(len(traj), cfg, gen)
        if steps:
            aux = 0.0
            for t in steps:
                lg = logits_at(model, traj[t], cos, sin, pos).float()
                if cfg.deep_sup_detach_teacher:
                    with torch.no_grad():
                        ref = logits.reshape(-1, V)
                        ref = ref.index_select(0, pos) if pos is not None else ref
                        p = ref.float().softmax(-1)
                    aux = aux + F.kl_div(lg.log_softmax(-1), p, reduction="batchmean")
                else:
                    aux = aux + F.cross_entropy(lg, tgt_sub)
            aux = aux / len(steps)
            total = total + cfg.deep_sup_weight * aux
            logs["deep_sup"] = float(aux.detach())

    # ---- PonderNet-style halting head (auxiliary: trains the exit policy) ----
    if cfg.halting == "ponder" and traj:
        lam = []
        for t in range(1, len(traj)):
            l = model.halt_logits(traj[t]).reshape(-1)
            lam.append(torch.sigmoid(l.index_select(0, pos) if pos is not None else l))
        lam = torch.stack(lam, 0).clamp(1e-4, 1 - 1e-4)          # [T, P]
        not_halt = torch.cumprod(1.0 - lam, 0)
        p = torch.cat([lam[:1], lam[1:] * not_halt[:-1]], 0)
        p = p / p.sum(0, keepdim=True).clamp_min(1e-6)
        with torch.no_grad():
            ce_t = []
            for t in range(1, len(traj)):
                lg = logits_at(model, traj[t], cos, sin, pos).float()
                ce_t.append(F.cross_entropy(lg, tgt_sub, reduction="none"))
            ce_t = torch.stack(ce_t, 0)                           # [T, P]
        ponder = (p * ce_t).sum(0).mean()
        lam_p = 1.0 / max(cfg.halting_target_loops, 1.0)
        steps_ar = torch.arange(1, len(traj), device=p.device, dtype=p.dtype)
        prior = lam_p * (1 - lam_p) ** (steps_ar - 1)
        prior = (prior / prior.sum()).unsqueeze(1)
        kl = (p * (p.clamp_min(1e-8).log() - prior.log())).sum(0).mean()
        total = total + cfg.halting_weight * (ponder + 0.01 * kl)
        logs["ponder"] = float(ponder.detach())
        logs["exp_loops"] = float((p * steps_ar.unsqueeze(1)).sum(0).mean().detach())

    # ---- trajectory decorrelation ----
    if "decorr" in out:
        total = total + cfg.decorr_weight * out["decorr"]
        logs["cos_succ"] = float(out["decorr"].detach())

    logs["loss"] = float(total.detach())
    return total, logs
