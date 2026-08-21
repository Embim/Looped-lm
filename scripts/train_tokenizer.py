"""Train a byte-level BPE tokenizer on FineWeb.

Why a custom tokenizer: the parameter budget is 10M *including* embeddings.
Qwen3's 151936-entry vocabulary would need 151936*d parameters, which at any
usable d_model exceeds the whole budget, so essentially none of the budget would
be left for the recurrent block.  A byte-level BPE with vocab 8192 costs 4.2M
parameters at d_model=512 and is lossless (no <unk>), which keeps ~56% of the
budget in the part of the network that is actually looped.

Because token-level perplexity is not comparable across tokenizers, every
result in the report is additionally given as bits-per-byte.

Usage:
    python scripts/train_tokenizer.py --vocab_size 8192
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

EOS = "<|endoftext|>"


def iter_texts(path: Path, max_bytes: int, batch_size: int = 2000):
    """Yield documents from a FineWeb parquet shard until max_bytes is reached."""
    seen = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
        for t in batch.column("text").to_pylist():
            if not t:
                continue
            seen += len(t)
            yield t
            if seen >= max_bytes:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_shard", default=r"C:\ml\looped-lm\data\raw\sample\10BT\000_00000.parquet")
    ap.add_argument("--out_dir", default=r"C:\ml\looped-lm\data")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--train_bytes", type=int, default=400_000_000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tokenizer_bpe{args.vocab_size}.json"

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
        min_frequency=2,
    )

    t0 = time.time()
    # The tokenizer only ever sees the *training* shard, never the validation shard.
    tok.train_from_iterator(iter_texts(Path(args.train_shard), args.train_bytes), trainer=trainer)
    tok.save(str(out))
    print(f"saved {out}  vocab={tok.get_vocab_size()}  ({time.time()-t0:.0f}s)")

    probe = "The looped transformer applies the same block 32 times; \u0442\u0435\u0441\u0442 123.\n"
    ids = tok.encode(probe).ids
    assert tok.decode(ids) == probe, "byte-level BPE must round-trip exactly"
    print(f"round-trip OK | {len(probe)} bytes -> {len(ids)} tokens "
          f"({len(probe)/len(ids):.2f} bytes/token)")
    print("eos id:", tok.token_to_id(EOS))
    (out_dir / f"tokenizer_bpe{args.vocab_size}.meta.json").write_text(
        json.dumps({"vocab_size": tok.get_vocab_size(), "eos_id": tok.token_to_id(EOS),
                    "train_shard": args.train_shard, "train_bytes": args.train_bytes}, indent=2))


if __name__ == "__main__":
    main()
