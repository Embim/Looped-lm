"""Training loop with an exact token budget and a test-time depth sweep."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig
from .data import TokenStream, bpb_from_nll, load_meta
from .losses import compute_loss, logits_at
from .model import LoopedQwen3
from . import tracking

NO_DECAY = ("norm", "ada_", "step_alpha", "pool_", "mom_", "inject_alpha", "depth_emb", "halt_head")


def build_optimizer(model: torch.nn.Module, tc: TrainConfig):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or any(k in n for k in NO_DECAY):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": tc.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2), eps=1e-8, fused=True)


def lr_at(step: int, total: int, tc: TrainConfig) -> float:
    warm = max(int(total * tc.warmup_frac), 1)
    if step < warm:
        return tc.lr * (step + 1) / warm
    if tc.schedule == "cosine":
        t = (step - warm) / max(total - warm, 1)
        return tc.lr * (tc.min_lr_frac + (1 - tc.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))
    if tc.schedule == "wsd":                      # warmup - stable - decay
        d0 = total - int(total * tc.wsd_decay_frac)
        if step < d0:
            return tc.lr
        t = (step - d0) / max(total - d0, 1)
        return tc.lr * (tc.min_lr_frac + (1 - tc.min_lr_frac) * (1 - t))
    raise ValueError(tc.schedule)


def sample_loops(mc: ModelConfig, rng: np.random.Generator) -> int:
    if mc.loop_sampling == "fixed":
        return mc.n_loops
    if mc.loop_sampling == "uniform":
        return int(rng.integers(mc.loop_min, mc.n_loops + 1))
    if mc.loop_sampling == "poisson":
        # Huginn-style: mostly around n_loops, occasionally much deeper
        t = mc.loop_min + int(rng.poisson(max(mc.n_loops - mc.loop_min, 1)))
        return int(min(max(t, mc.loop_min), mc.max_loops))
    raise ValueError(mc.loop_sampling)


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data_dir: str, seq_len: int, batch: int, n_loops: int,
             eval_tokens: int, device: str = "cuda", collect_stats: bool = False,
             per_step: bool = False) -> Dict:
    """Validation NLL at a given loop count.

    per_step=True also returns the NLL of reading out after each individual loop,
    which is what the depth-scaling and early-exit analyses are built on.
    """
    meta = load_meta(data_dir)
    bpt = meta["val"]["bytes_per_token"]
    stream = TokenStream(Path(data_dir) / "val.bin", seq_len, batch, device=device,
                         shuffle=False, max_windows=max(eval_tokens // seq_len, batch))
    model.eval()
    tot, n = 0.0, 0
    step_tot: Optional[torch.Tensor] = None
    stats_acc: Dict[str, List[float]] = {}
    for x, y in stream.batches():
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            out = model(x, n_loops=n_loops, collect_states=per_step, collect_stats=collect_stats)
            logits = model.head(out["hidden"])
        ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                             reduction="sum")
        tot += float(ce)
        n += y.numel()
        if per_step:
            traj = out["traj"]
            if step_tot is None:
                step_tot = torch.zeros(len(traj), device=device, dtype=torch.float64)
            for t, st in enumerate(traj):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = logits_at(model, st, out["cos"], out["sin"], None)
                step_tot[t] += F.cross_entropy(lg.float(), y.reshape(-1), reduction="sum").double()
        if collect_stats:
            for k, v in out["stats"].items():
                stats_acc.setdefault(k, []).append(v)
    model.train()
    nll = tot / n
    res = {"n_loops": n_loops, "val_loss": nll, "val_ppl": math.exp(nll),
           "val_bpb": bpb_from_nll(nll, bpt), "eval_tokens": n}
    if per_step and step_tot is not None:
        per = (step_tot / n).tolist()
        res["per_step_loss"] = per
        res["per_step_ppl"] = [math.exp(v) for v in per]
    if collect_stats and stats_acc:
        res["stats"] = {k: np.mean(np.array(v, dtype=np.float64), axis=0).tolist()
                        for k, v in stats_acc.items() if len(v) and len(v[0])}
    return res


# ---------------------------------------------------------------------------
def train(mc: ModelConfig, tc: TrainConfig, depth_sweep: Optional[List[int]] = None) -> Dict:
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = load_meta(tc.data_dir)
    assert meta["vocab_size"] == mc.vocab_size, (meta["vocab_size"], mc.vocab_size)
    mc.max_seq_len = max(mc.max_seq_len, tc.seq_len)

    run_dir = Path(tc.out_dir) / tc.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_config.json").write_text(mc.to_json())
    (run_dir / "train_config.json").write_text(tc.to_json())
    logf = (run_dir / "log.jsonl").open("w", encoding="utf-8")

    model = LoopedQwen3(mc).to(device)
    opt = build_optimizer(model, tc)
    lr_mult = 1.0 / math.sqrt(mc.n_loops) if tc.lr_scale_with_loops == "inv_sqrt" else 1.0

    tokens_per_step = tc.micro_batch * tc.grad_accum * tc.seq_len
    steps = tc.total_tokens // tokens_per_step
    stream = TokenStream(Path(tc.data_dir) / "train.bin", tc.seq_len,
                         tc.micro_batch, device=device, seed=tc.seed,
                         max_windows=steps * tc.micro_batch * tc.grad_accum)
    it = stream.batches()
    rng = np.random.default_rng(tc.seed)
    gen = torch.Generator(device=device); gen.manual_seed(tc.seed)

    n_par, n_par_ne = model.n_params(), model.n_params(True)
    fwd_flops = model.flops_per_token(seq_len=tc.seq_len)
    header = {"event": "start", "run": tc.run_name, "params": n_par, "params_non_emb": n_par_ne,
              "steps": steps, "tokens_per_step": tokens_per_step,
              "total_tokens": steps * tokens_per_step,
              "fwd_flops_per_token": fwd_flops,
              "train_flops_est": 3 * fwd_flops * steps * tokens_per_step,
              "model": asdict(mc), "train": asdict(tc)}
    logf.write(json.dumps(header) + "\n"); logf.flush()
    print(f"[{tc.run_name}] params={n_par/1e6:.3f}M (non-emb {n_par_ne/1e6:.3f}M) "
          f"steps={steps} tokens={steps*tokens_per_step/1e6:.1f}M "
          f"fwd_flops/tok={fwd_flops/1e6:.1f}M", flush=True)
    assert n_par <= tc.param_budget, f"parameter budget exceeded: {n_par} > {tc.param_budget}"

    best = {"val_loss": float("inf")}
    t0 = time.time()
    tok_seen = 0
    # Rate over the last logging window, not tokens-since-start / time-since-start:
    # the cumulative form includes startup and warm-up, so it climbs through the
    # run (49k -> 152k tok/s in one 2M-token run) and is not a throughput anyone
    # should plan from.  Board reservations use the benchmarked median step time.
    win_t, win_tok = t0, 0
    for step in range(steps):
        lr = lr_at(step, steps, tc) * lr_mult
        for g in opt.param_groups:
            g["lr"] = lr
        T = sample_loops(mc, rng)

        opt.zero_grad(set_to_none=True)
        logs_acc: Dict[str, float] = {}
        for micro in range(tc.grad_accum):
            x, y = next(it)
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                out = model(x, n_loops=T)
                loss, logs = compute_loss(model, out, y, gen)
            (loss / tc.grad_accum).backward()
            for k, v in logs.items():
                logs_acc[k] = logs_acc.get(k, 0.0) + v / tc.grad_accum
            tok_seen += y.numel()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()

        if step % tc.log_every == 0 or step == steps - 1:
            rec = {"event": "train", "step": step, "tokens": tok_seen, "lr": lr, "T": T,
                   "gnorm": float(gnorm), "elapsed": round(time.time() - t0, 1),
                   "tok_per_s_window": round((tok_seen - win_tok) / max(time.time() - win_t, 1e-9), 1),
                   **logs_acc}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            if step % (tc.log_every * 10) == 0 or step == steps - 1:
                now = time.time()
                tps = (tok_seen - win_tok) / max(now - win_t, 1e-9)
                win_t, win_tok = now, tok_seen
                print(f"  step {step:5d}/{steps} loss {logs_acc.get('ce', 0):.4f} "
                      f"lr {lr:.2e} T={T} |g|={float(gnorm):.2f} {tps/1e3:.1f}k tok/s", flush=True)

        last = step == steps - 1
        if (step + 1) % tc.eval_every == 0 or last:
            ev = evaluate(model, tc.data_dir, tc.seq_len, tc.micro_batch, mc.n_loops,
                          tc.eval_tokens, device)
            ev.update({"event": "eval", "step": step, "tokens": tok_seen})
            logf.write(json.dumps(ev) + "\n"); logf.flush()
            print(f"  [eval @{step+1}] loss {ev['val_loss']:.4f} ppl {ev['val_ppl']:.2f} "
                  f"bpb {ev['val_bpb']:.4f}", flush=True)
            tracking.log({"val_loss": ev["val_loss"], "val_ppl": ev["val_ppl"],
                          "val_bpb": ev["val_bpb"]}, step=step)
            if ev["val_loss"] < best["val_loss"]:
                best = {k: ev[k] for k in ("val_loss", "val_ppl", "val_bpb")}
                best["step"] = step
                if tc.save_best:
                    torch.save({"model": model.state_dict(), "model_config": asdict(mc),
                                "train_config": asdict(tc), "step": step, "best": best},
                               run_dir / "ckpt_best.pt")
    if tc.save_last:
        torch.save({"model": model.state_dict(), "model_config": asdict(mc),
                    "train_config": asdict(tc), "step": steps, "best": best},
                   run_dir / "ckpt_last.pt")

    # ---------------- final evaluation ----------------
    final = evaluate(model, tc.data_dir, tc.seq_len, tc.micro_batch, mc.n_loops,
                     tc.final_eval_tokens, device, collect_stats=True, per_step=True)
    sweep = {}
    for T in (depth_sweep or []):
        if T > mc.max_loops:
            continue
        r = evaluate(model, tc.data_dir, tc.seq_len, tc.micro_batch, T, tc.eval_tokens, device)
        sweep[T] = {k: r[k] for k in ("val_loss", "val_ppl", "val_bpb")}
        print(f"  [T={T:3d}] loss {r['val_loss']:.4f} ppl {r['val_ppl']:.2f} bpb {r['val_bpb']:.4f}",
              flush=True)

    summary = {"event": "final", "run": tc.run_name, "params": n_par, "params_non_emb": n_par_ne,
               "train_tokens": steps * tokens_per_step, "wall_seconds": round(time.time() - t0, 1),
               "fwd_flops_per_token": fwd_flops,
               "train_flops_est": 3 * fwd_flops * steps * tokens_per_step,
               "best_periodic": best, "final": final, "depth_sweep": sweep,
               "model": asdict(mc), "train": asdict(tc)}
    logf.write(json.dumps(summary) + "\n")
    logf.close()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    tracking.log({"final_val_loss": final["val_loss"], "final_val_ppl": final["val_ppl"],
                  "final_val_bpb": final["val_bpb"],
                  **{f"sweep_T{T}_loss": v["val_loss"] for T, v in sweep.items()}})
    tracking.log_artifact(str(run_dir / "summary.json"))
    tracking.finish()
    print(f"[{tc.run_name}] DONE final loss {final['val_loss']:.4f} "
          f"ppl {final['val_ppl']:.2f} bpb {final['val_bpb']:.4f} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    return summary
