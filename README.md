# looped-lm — getting many useful loops out of a small looped transformer

A looped (depth-recurrent) Qwen3-style decoder pretrained on FineWeb under two
hard budgets: **≤ 10M parameters in total, including embeddings** and **≤ 100M
training tokens**. The question the repository is built around is not "does
looping help" — it does, cheaply — but *why the gains stop* after a handful of
loops, and what makes a 32- or 64-loop model better than a 4-loop one.

The report (in Russian) is [`report/report.md`](report/report.md).
The final checkpoint is on the Hub: [Embim/looped-qwen3-9.4M-T32-fineweb](https://huggingface.co/Embim/looped-qwen3-9.4M-T32-fineweb) —
**val loss 3.7219 / ppl 41.3 / bits-per-byte 1.4275** at T=32, 9.44M parameters, 100M FineWeb tokens.

## The short version

Depth-recurrence saturates for three independent reasons, and each one has a
cheap, scale-free fix. The experiment program is organised around testing them
separately:

| # | Why the loop stops paying | What is tried |
|---|---|---|
| 1 | The loop is a **stationary contraction**: one shared function can only implement a stationary iterative rule, and a contraction reaches its fixed point in a few steps, after which further compute is provably wasted. | Step conditioning (`depth_cond`): additive sinusoid, learned per-step embedding, per-step adaLN gains/gates, and a parameter-free **depth-RoPE** that conjugates the state by a step-dependent rotation, giving T distinct effective weight matrices for zero parameters. Fixed-norm and hyperspherical updates, and heavy-ball momentum, which turns a contraction into an oscillator. |
| 2 | The recurrence has a **fixed-size state**: an L-layer transformer carries `L·d` of activations, a looped one only `d`, so the loop runs out of scratch space. | `loop_memory=depth_attn`: each step attends over the history of its own states at the same position (detached bank, O(T) extra compute, no gradient path through the history). Momentum as a second, slower state. |
| 3 | Compute is **allocated uniformly** while the need for it is not: most FineWeb tokens are settled after one or two steps, so the mean curve flattens while the hard tail is still improving. | Per-loop read-out and deep supervision, a PonderNet-style halting head, and evaluation that reports loss against the *average* number of loops actually spent, plus depth curves computed separately for the easiest half and the hardest few percent of tokens. |

Read-out is also treated as a design choice rather than a given: `readout=pool_gate`
lets every token attend over its own trajectory instead of only its end point, so
a converged trajectory becomes measurably wasteful rather than merely useless.

## Layout

```
loopedlm/
  config.py     ModelConfig / TrainConfig — every switch of the study
  model.py      LoopedQwen3: prelude / looped core / coda, all loop mechanisms
  losses.py     next-token CE + deep supervision + halting + trajectory regularisers
  data.py       uint16 memmap streams with an exact, auditable token budget
  train.py      training loop, periodic eval, end-of-run depth sweep
  eval.py       depth curves, difficulty strata, early-exit simulation, trajectory stats
  presets.py    the experiment program (58 configs in 11 groups)
  hwlock.py     reservation of the shared GPU (claim + heartbeat + release)
  tracking.py   optional MLflow mirror; local JSONL stays the source of truth
scripts/
  train_tokenizer.py  prepare_data.py  train.py  eval.py
  run_experiments.py  collect_results.py  plots.py  stop_all.py
  bench*.py           throughput / memory / attention-path measurements
```

## Reproducing

```bash
pip install -r requirements.txt
python scripts/train_tokenizer.py --vocab_size 8192
python scripts/prepare_data.py --train_tokens 101000000 --val_tokens 10000000
python scripts/train.py --preset A_depth16
python scripts/eval.py --run A_depth16
python scripts/run_experiments.py --group A,B,C,D,E,F,G,H,I,J,K
python scripts/collect_results.py && python scripts/plots.py
```

`scripts/train.py --list` prints every preset. Ad-hoc configurations need no new
preset: `--set model.n_loops=32 --set model.update=normalized --set train.lr=2e-3`.

## Budgets and how they are enforced

- **Parameters.** `vocab 8192 × d_model 512` tied embeddings cost 4.19M, leaving
  5.25M for the two-layer looped core: **9.44M total**. `train()` asserts against
  `TrainConfig.param_budget` (10M) and refuses to start otherwise, so no run can
  silently exceed it. Two deliberately out-of-budget runs (`A_ref_unshared8/16`)
  raise the limit explicitly; they exist only as an upper reference for what
  unshared depth of the same effective length would buy.
- **Tokens.** Training windows are non-overlapping and visited in a seeded
  permutation, so a run consumes exactly `total_tokens` distinct target tokens —
  no repeats and no epoch bookkeeping. The count is logged per run.
- **Tokenizer.** A byte-level BPE with vocab 8192, trained only on the training
  shard (FineWeb `sample/10BT/000`), validated on a different shard (`014`).
  Qwen3's own 151936-entry vocabulary would need more than the entire parameter
  budget for embeddings alone, which is why a small vocabulary is used; the
  measured cost is 3.78 bytes/token against 4.48 for GPT-2 on the same text.
- **Comparability.** Token-level perplexity is not comparable across tokenizers,
  so every result is also reported as **bits-per-byte**, and every run reports the
  forward FLOPs per token and the total training FLOPs, because more loops at
  fixed parameters means more compute — the comparison is only meaningful with
  that column visible.

## Notes on this machine

Measured on an RTX 5080 (sm_120) on Windows with torch 2.9.1+cu130. Three silent
slow paths cost a factor of 3.0 in step time (928 → 311 ms at T=16), and are
worth knowing about before trusting any throughput number here:

- PyTorch's Windows wheels ship **no FlashAttention kernels**; only mem-efficient
  and math are available.
- `scaled_dot_product_attention(..., enable_gqa=True)` therefore has no fused
  kernel and silently selects the math backend, materialising the `[B,H,S,S]`
  score matrix: **5254 µs against 290 µs** per layer versus expanding K/V
  explicitly. GQA is a parameter-budget choice here, not a speed one.
- `F.rms_norm` with a float32 gain on a bfloat16 input cannot dispatch to the
  fused implementation (**351 µs against 67 µs**); the gain is cast to the input
  dtype, exactly as autocast already does for every `Linear` weight.
- Autocast's weight cache breaks gradient checkpointing when part of the loop runs
  under `no_grad` (truncated BPTT), so training runs with `cache_enabled=False`.
  With that fixed, `bptt_last_k=8` makes T=64 nearly three times cheaper.
- `torch.compile` is unavailable (no Triton on Windows), and the `cudagraphs`
  backend cannot capture a re-entrant loop.
