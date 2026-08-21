"""Evaluation and analysis of a trained looped model.

Beyond val loss / perplexity / bits-per-byte this produces the three diagnostics
the whole study is built on:

1. depth curve      -- loss when reading out after each loop t = 0..T.  Where it
                       flattens is where extra compute stops paying.
2. difficulty strata-- the same curve computed separately for the easiest and
                       hardest tokens.  The mean curve saturating while the hard
                       tail keeps improving is the signature that a *uniform*
                       loop count is the wrong allocation, not that the loop has
                       run out of useful computation.
3. early exit       -- per-token exit by confidence / prediction stability /
                       learned halting head, giving loss as a function of the
                       *average* number of loops actually spent.
"""
from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig
from .data import TokenStream, bpb_from_nll, load_meta
from .losses import logits_at
from .model import LoopedQwen3


def load_checkpoint(path: str | Path, device: str = "cuda", n_loops: Optional[int] = None):
    ck = torch.load(path, map_location=device, weights_only=False)
    mc = ModelConfig.from_dict(ck["model_config"])
    if n_loops is not None:
        mc.n_loops = n_loops
    model = LoopedQwen3(mc).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, mc, ck


# ---------------------------------------------------------------------------
@torch.no_grad()
def per_step_tables(model, data_dir: str, seq_len: int, batch: int, n_loops: int,
                    eval_tokens: int, device: str = "cuda") -> Dict[str, np.ndarray]:
    """Collect per-token, per-loop-step quantities over the validation set.

    Returns arrays of shape [T+1, N] (N = number of target tokens):
      nll   -- negative log-likelihood of the true token when reading out at step t
      conf  -- max softmax probability at step t
      kl    -- KL(p_t || p_{t-1}), 0 at t = 0
      halt  -- halting-head probability at step t (only if the model has one)
    """
    stream = TokenStream(Path(data_dir) / "val.bin", seq_len, batch, device=device,
                         shuffle=False, max_windows=max(eval_tokens // seq_len, batch))
    amp = torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False) if         str(device).startswith("cuda") else contextlib.nullcontext()
    nll_c, conf_c, kl_c, halt_c = [], [], [], []
    has_halt = model.cfg.halting == "ponder"
    for x, y in stream.batches():
        with amp:
            out = model(x, n_loops=n_loops, collect_states=True)
        traj = out["traj"]
        tgt = y.reshape(-1)
        nll_b, conf_b, kl_b, halt_b = [], [], [], []
        prev_lp = None
        for t, st in enumerate(traj):
            with amp:
                lg = logits_at(model, st, out["cos"], out["sin"], None)
            lp = lg.float().log_softmax(-1)
            nll_b.append(-lp.gather(1, tgt[:, None]).squeeze(1))
            conf_b.append(lp.max(-1).values.exp())
            if prev_lp is None:
                kl_b.append(torch.zeros_like(nll_b[-1]))
            else:
                kl_b.append((lp.exp() * (lp - prev_lp)).sum(-1))
            prev_lp = lp
            if has_halt:
                halt_b.append(torch.sigmoid(model.halt_logits(st)).reshape(-1))
        nll_c.append(torch.stack(nll_b).cpu().numpy())
        conf_c.append(torch.stack(conf_b).cpu().numpy())
        kl_c.append(torch.stack(kl_b).cpu().numpy())
        if has_halt:
            halt_c.append(torch.stack(halt_b).cpu().numpy())
    res = {"nll": np.concatenate(nll_c, 1), "conf": np.concatenate(conf_c, 1),
           "kl": np.concatenate(kl_c, 1)}
    if has_halt:
        res["halt"] = np.concatenate(halt_c, 1)
    return res


def depth_curve(nll: np.ndarray, bpt: float) -> Dict:
    m = nll.mean(1)
    return {"loss": m.tolist(), "ppl": np.exp(m).tolist(),
            "bpb": [bpb_from_nll(v, bpt) for v in m]}


def difficulty_strata(nll: np.ndarray, rank_by: Optional[np.ndarray] = None,
                      quantiles=(0.5, 0.8, 0.95, 0.99)) -> Dict:
    """Depth curve restricted to the hardest tokens.

    The ranking criterion matters more than it looks.  Grading difficulty by the
    loss at the *final* step and then reading the depth curve of that stratum is
    selection on the outcome: tokens picked for a high final loss have an
    inflated final point by construction, so the curve's right-hand end is
    biased exactly where the conclusion is drawn.  Grading by the loss at step 1
    moves the bias to the left-hand end instead.

    Neither is trustworthy alone, so `full_report` computes both, plus -- when
    `rank_by` carries the per-token loss of an *independent* model -- a ranking
    that is not a function of this model's own noise at all.  The claim "the hard
    tail keeps improving after the mean has flattened" is only reported as
    supported if it survives all of them.
    """
    key = nll[1] if rank_by is None else rank_by
    order = np.argsort(key)
    out: Dict = {}
    n = len(key)
    for q in quantiles:
        idx = order[int(n * q):]
        out[f"top{int(round((1 - q) * 100))}pct_hardest"] = {
            "n": int(len(idx)), "loss": nll[:, idx].mean(1).tolist()}
    idx = order[: int(n * 0.5)]
    out["easiest50pct"] = {"n": int(len(idx)), "loss": nll[:, idx].mean(1).tolist()}
    return out


def early_exit(nll: np.ndarray, score: np.ndarray, mode: str,
               thresholds: List[float], bpt: float) -> List[Dict]:
    """Per-token exit at the first step whose score passes the threshold.

    mode 'ge': exit when score >= thr (confidence, halting probability)
    mode 'le': exit when score <= thr (KL between successive predictions)
    Step 0 (no loop applied) is never an exit candidate.
    """
    T = nll.shape[0] - 1
    res = []
    for thr in thresholds:
        ok = (score[1:] >= thr) if mode == "ge" else (score[1:] <= thr)
        first = np.argmax(ok, axis=0)
        never = ~ok.any(axis=0)
        first[never] = T - 1
        exit_step = first + 1
        loss = nll[exit_step, np.arange(nll.shape[1])].mean()
        res.append({"threshold": float(thr), "avg_loops": float(exit_step.mean()),
                    "median_loops": float(np.median(exit_step)),
                    "frac_max_depth": float((exit_step == T).mean()),
                    "loss": float(loss), "ppl": float(math.exp(loss)),
                    "bpb": bpb_from_nll(float(loss), bpt)})
    return res


@torch.no_grad()
def trajectory_stats(model, data_dir: str, seq_len: int, batch: int, n_loops: int,
                     device: str = "cuda", n_batches: int = 8) -> Dict:
    """||delta_t||, ||h_t||, cos(delta_t, delta_{t-1}) and the effective rank of
    the update sequence -- how much of the state space the loop actually uses."""
    stream = TokenStream(Path(data_dir) / "val.bin", seq_len, batch, device=device,
                         shuffle=False, max_windows=n_batches * batch)
    acc: Dict[str, List] = {}
    ranks = []
    for bi, (x, y) in enumerate(stream.batches()):
        with (torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
              if str(device).startswith("cuda") else contextlib.nullcontext()):
            out = model(x, n_loops=n_loops, collect_stats=True, collect_states=True)
        for k, v in out["stats"].items():
            acc.setdefault(k, []).append(v)
        if bi == 0:
            st = torch.stack(out["traj"], 0).float()          # [T+1,B,S,d]
            d = st[1:] - st[:-1]                              # updates
            D = d[:, 0, -1, :]                                # last position, first sequence
            D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            s = torch.linalg.svdvals(D.cpu())
            p = (s / s.sum()).clamp_min(1e-12)
            ranks.append(float(torch.exp(-(p * p.log()).sum())))
    res = {k: np.mean(np.array(v, dtype=np.float64), axis=0).tolist() for k, v in acc.items()}
    if ranks:
        res["update_effective_rank"] = float(np.mean(ranks))
    return res


# ---------------------------------------------------------------------------
def full_report(ckpt: str | Path, data_dir: str, out_json: Optional[str | Path] = None,
                loops: Optional[List[int]] = None, analysis_loops: Optional[int] = None,
                eval_tokens: int = 4_000_000, analysis_tokens: int = 1_000_000,
                batch: int = 32, device: str = "cuda",
                ref_ckpt: Optional[str | Path] = None) -> Dict:
    from .train import evaluate

    model, mc, ck = load_checkpoint(ckpt, device)
    meta = load_meta(data_dir)
    bpt = meta["val"]["bytes_per_token"]
    seq = ck["train_config"]["seq_len"]
    loops = loops or sorted({1, 2, 4, 8, 16, 32, min(64, mc.max_loops), mc.n_loops})
    Ta = analysis_loops or min(max(mc.n_loops, 16), mc.max_loops)

    rep: Dict = {"checkpoint": str(ckpt), "params": model.n_params(),
                 "params_non_emb": model.n_params(True), "trained_loops": mc.n_loops,
                 "bytes_per_token": bpt, "model_config": ck["model_config"],
                 "flops_per_token": {str(T): model.flops_per_token(T, seq) for T in loops}}

    rep["depth_sweep"] = {}
    for T in loops:
        if T > mc.max_loops:
            continue
        r = evaluate(model, data_dir, seq, batch, T, eval_tokens, device)
        rep["depth_sweep"][str(T)] = {k: r[k] for k in ("val_loss", "val_ppl", "val_bpb")}
        print(f"  T={T:3d}  loss {r['val_loss']:.4f}  ppl {r['val_ppl']:7.2f}  "
              f"bpb {r['val_bpb']:.4f}", flush=True)

    print(f"  per-token analysis at T={Ta} ...", flush=True)
    tab = per_step_tables(model, data_dir, seq, batch, Ta, analysis_tokens, device)
    rep["analysis_loops"] = Ta
    rep["depth_curve"] = depth_curve(tab["nll"], bpt)
    # Three rankings, because each has a different selection bias; see
    # difficulty_strata.  `ref_ckpt` supplies a ranking from an independent model.
    rep["difficulty_strata"] = difficulty_strata(tab["nll"])
    rep["difficulty_strata_by_final"] = difficulty_strata(tab["nll"], rank_by=tab["nll"][-1])
    if ref_ckpt:
        ref_model, ref_mc, _ = load_checkpoint(ref_ckpt, device)
        ref = per_step_tables(ref_model, data_dir, seq, batch, ref_mc.n_loops,
                              analysis_tokens, device)
        m = min(ref["nll"].shape[1], tab["nll"].shape[1])
        rep["difficulty_strata_by_reference"] = difficulty_strata(
            tab["nll"][:, :m], rank_by=ref["nll"][-1][:m])
        rep["reference_model"] = {"checkpoint": str(ref_ckpt), "loops": ref_mc.n_loops,
                                  "mean_loss": float(ref["nll"][-1].mean())}
        del ref_model
    rep["early_exit"] = {
        "confidence": early_exit(tab["nll"], tab["conf"], "ge",
                                 [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01], bpt),
        "kl_stability": early_exit(tab["nll"], tab["kl"], "le",
                                   [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3], bpt),
    }
    if "halt" in tab:
        rep["early_exit"]["halting_head"] = early_exit(
            tab["nll"], tab["halt"], "ge", [0.1, 0.2, 0.3, 0.5, 0.7, 0.9], bpt)
    rep["trajectory"] = trajectory_stats(model, data_dir, seq, batch, Ta, device)

    if out_json:
        Path(out_json).write_text(json.dumps(rep, indent=2))
    return rep
