"""Package a trained run for the Hugging Face Hub.

Builds a self-contained folder: weights in safetensors, the exact model and train
configs, the tokenizer, the evaluation numbers, and a model card that states the
budgets and the tokenizer caveat.  Prints the upload command; it does not upload
by itself, so publishing stays an explicit act.

    python scripts/export_checkpoint.py --run C_depth_rope --repo user/looped-lm-10m
    python scripts/export_checkpoint.py --run C_depth_rope --out C:\\ml\\export
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loopedlm.config import ModelConfig, TrainConfig     # noqa: E402
from loopedlm.model import LoopedQwen3                    # noqa: E402

CARD = """---
license: apache-2.0
datasets:
  - HuggingFaceFW/fineweb
language:
  - en
library_name: pytorch
tags:
  - looped-transformer
  - depth-recurrence
  - universal-transformer
  - small-language-model
---

# {name}

A **looped** (depth-recurrent) Qwen3-style decoder: a core block of {n_core}
layers is applied **{n_loops} times with shared weights**, so the model spends
{eff_layers} effective layers of compute while holding only {n_core} layers of
parameters.

Trained from scratch on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
under two hard budgets: **{params_m:.2f}M parameters in total (including
embeddings)** and **{tokens_m:.0f}M training tokens**.

## Results

| metric | value |
|---|---|
| validation loss (nats/token) | {val_loss:.4f} |
| validation perplexity | {val_ppl:.2f} |
| **validation bits-per-byte** | **{val_bpb:.4f}** |
| parameters, total | {params:,} |
| parameters, non-embedding | {params_ne:,} |
| loops at evaluation | {n_loops} |
| forward FLOPs / token | {fwd_gflops:.3f} G |

> **Perplexity here is not comparable to other models.** It uses a byte-level BPE
> with a vocabulary of 8192 trained on this dataset, because the 10M-parameter
> budget counts embeddings: a 151936-entry vocabulary would consume the entire
> budget before the transformer got any. Compare the **bits-per-byte** column
> instead, which is tokenizer-independent. On this validation text the tokenizer
> produces 3.78 bytes/token (GPT-2: 4.48).

{depth_table}

## Architecture

Qwen3 recipe — pre-RMSNorm, per-head QK-norm, GQA, SwiGLU, RoPE, no biases, tied
embeddings — with the middle of the network looped:

```
e   = EmbedNorm(Embed(idx))
h_0 = e
h_t = Update(h_{{t-1}}, Core(h_{{t-1}}, e, t))     # t = 1..T, shared weights
logits = Head(RMSNorm(ReadOut(h_0..h_T)))
```

| | |
|---|---|
| d_model | {d_model} |
| layers in the looped core | {n_core} |
| loops (train / max) | {n_loops} / {max_loops} |
| heads (query / KV) | {n_heads} / {n_kv} |
| head dim | {head_dim} |
| MLP intermediate | {intermediate} |
| context | {seq_len} |
| vocabulary | {vocab} |
| step conditioning | `{depth_cond}` |
| update rule | `{update}` |
| loop memory | `{loop_memory}` |
| read-out | `{readout}` |

## Usage

The architecture is custom, so load it with the code from the repository rather
than with `AutoModel`:

```python
import json, torch
from safetensors.torch import load_file
from loopedlm.config import ModelConfig
from loopedlm.model import LoopedQwen3
from tokenizers import Tokenizer

cfg = ModelConfig.from_dict(json.load(open("model_config.json")))
model = LoopedQwen3(cfg).eval()
model.load_state_dict(load_file("model.safetensors"))
tok = Tokenizer.from_file("tokenizer.json")

ids = torch.tensor([tok.encode("The looped transformer").ids])
out = model(ids, n_loops={n_loops})          # n_loops is a runtime choice
logits = model.head(out["hidden"])
```

`n_loops` is a runtime argument: the same weights can be evaluated at any depth,
which is the point of the architecture. See the repository for the depth-scaling
and early-exit numbers.

## Training

- Data: FineWeb `sample/10BT`, shard `000` for training, shard `014` for
  validation (disjoint documents).
- {tokens_m:.0f}M tokens, non-overlapping windows in a seeded permutation, so every
  training token is seen exactly once.
- AdamW, lr {lr:g}, {schedule} schedule, weight decay {wd:g}, grad clip {clip:g},
  bf16 autocast, batch {batch} x {seq_len} tokens.

## Limitations

Under 10M parameters and 100M tokens this is a research artefact for studying
depth recurrence, not a usable assistant: it produces locally plausible English
and nothing more. It has no instruction tuning and no safety alignment.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--which", default="best", choices=["best", "last"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--repo", default=None, help="target repo id, for the printed command")
    ap.add_argument("--tokenizer", default=r"C:\ml\looped-lm\data\tokenizer_bpe8192.json")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    run_dir = Path(TrainConfig.out_dir) / a.run
    ckpt_path = run_dir / f"ckpt_{a.which}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}")
    out = Path(a.out or (Path(TrainConfig.out_dir) / "export" / a.run))
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["model_config"])
    tc = ck["train_config"]
    model = LoopedQwen3(mc)
    model.load_state_dict(ck["model"])

    # safetensors refuses shared storage, so untie the head before saving
    sd = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    if mc.tie_embeddings:
        sd.pop("lm_head.weight", None)
    save_file(sd, str(out / "model.safetensors"),
              metadata={"format": "pt", "tied_embeddings": str(mc.tie_embeddings)})

    (out / "model_config.json").write_text(mc.to_json())
    (out / "train_config.json").write_text(json.dumps(tc, indent=2))
    if Path(a.tokenizer).exists():
        shutil.copy(a.tokenizer, out / "tokenizer.json")
    for extra in ("summary.json", "log.jsonl"):
        if (run_dir / extra).exists():
            shutil.copy(run_dir / extra, out / extra)
    ev = ROOT / "results" / f"{a.run}.eval.json"
    if ev.exists():
        shutil.copy(ev, out / "evaluation.json")

    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    final = summary.get("final", {})
    sweep = summary.get("depth_sweep") or {}
    if sweep:
        rows = "\n".join(f"| {T} | {v['val_loss']:.4f} | {v['val_ppl']:.2f} | {v['val_bpb']:.4f} |"
                         for T, v in sorted(sweep.items(), key=lambda kv: int(kv[0])))
        depth_table = ("## Depth scaling at inference\n\nThe same weights evaluated at "
                       "different loop counts:\n\n| loops | val loss | ppl | bits/byte |\n"
                       "|---|---|---|---|\n" + rows + "\n")
    else:
        depth_table = ""

    name = a.name or f"looped-qwen3-{model.n_params()/1e6:.1f}M-T{mc.n_loops}"
    card = CARD.format(
        name=name, n_core=mc.n_core, n_loops=mc.n_loops, max_loops=mc.max_loops,
        eff_layers=mc.n_core * mc.n_loops, params=model.n_params(),
        params_ne=model.n_params(True), params_m=model.n_params() / 1e6,
        tokens_m=summary.get("train_tokens", tc.get("total_tokens", 0)) / 1e6,
        val_loss=final.get("val_loss", float("nan")), val_ppl=final.get("val_ppl", float("nan")),
        val_bpb=final.get("val_bpb", float("nan")),
        fwd_gflops=summary.get("fwd_flops_per_token", 0) / 1e9,
        d_model=mc.d_model, n_heads=mc.n_heads, n_kv=mc.n_kv_heads, head_dim=mc.head_dim,
        intermediate=mc.intermediate_size, seq_len=tc.get("seq_len", mc.max_seq_len),
        vocab=mc.vocab_size, depth_cond=mc.depth_cond, update=mc.update,
        loop_memory=mc.loop_memory, readout=mc.readout, depth_table=depth_table,
        lr=tc.get("lr", 0), schedule=tc.get("schedule", "cosine"),
        wd=tc.get("weight_decay", 0), clip=tc.get("grad_clip", 0),
        batch=tc.get("micro_batch", 0) * tc.get("grad_accum", 1),
    )
    (out / "README.md").write_text(card, encoding="utf-8")

    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"exported {a.run} -> {out}  ({total/1e6:.1f} MB)")
    for p in sorted(out.iterdir()):
        print(f"    {p.name:24s} {p.stat().st_size/1e6:7.2f} MB")
    repo = a.repo or "<your-user>/" + name
    print("\nto publish (run this yourself, it needs your HF token):")
    print(f'    huggingface-cli upload {repo} "{out}" . --repo-type model')


if __name__ == "__main__":
    main()
