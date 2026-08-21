"""Tokenise FineWeb shards into flat uint16 memmaps (nanoGPT style).

train.bin comes from shard 000, val.bin from shard 014 -- different files of the
`sample/10BT` split, so there is no document overlap between them.

The meta file records the number of UTF-8 bytes that produced the tokens, which
is what lets us report tokenizer-independent bits-per-byte alongside perplexity.

Usage:
    python scripts/prepare_data.py --train_tokens 101000000 --val_tokens 10000000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def encode_shard(shard: Path, tok: Tokenizer, eos_id: int, target_tokens: int,
                 out_path: Path, batch_docs: int = 1024):
    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(target_tokens,))
    n_tok = 0
    n_bytes = 0
    n_docs = 0
    t0 = time.time()
    pf = pq.ParquetFile(shard)
    done = False
    for batch in pf.iter_batches(batch_size=batch_docs, columns=["text"]):
        texts = [t for t in batch.column("text").to_pylist() if t]
        if not texts:
            continue
        encs = tok.encode_batch_fast(texts)
        for text, enc in zip(texts, encs):
            ids = enc.ids
            need = target_tokens - n_tok
            # +1 for the document separator
            if len(ids) + 1 <= need:
                arr[n_tok:n_tok + len(ids)] = ids
                arr[n_tok + len(ids)] = eos_id
                n_tok += len(ids) + 1
                n_bytes += len(text.encode("utf-8"))
                n_docs += 1
            else:
                # truncate the final document so the file has an exact length
                cut = max(need - 1, 0)
                if cut > 0:
                    arr[n_tok:n_tok + cut] = ids[:cut]
                    n_tok += cut
                    n_bytes += len(tok.decode(ids[:cut]).encode("utf-8"))
                    n_docs += 1
                if n_tok < target_tokens:
                    arr[n_tok] = eos_id
                    n_tok += 1
                done = True
                break
        if done:
            break
        if n_docs % (batch_docs * 50) < batch_docs:
            print(f"  {n_tok/1e6:7.1f}M / {target_tokens/1e6:.0f}M tokens "
                  f"({n_docs} docs, {time.time()-t0:.0f}s)", flush=True)
    arr.flush()
    del arr
    assert n_tok == target_tokens, f"got {n_tok} tokens, wanted {target_tokens}"
    return {"n_tokens": n_tok, "n_bytes": n_bytes, "n_docs": n_docs,
            "bytes_per_token": n_bytes / n_tok, "shard": str(shard),
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=r"C:\ml\looped-lm\data\raw\sample\10BT")
    ap.add_argument("--tokenizer", default=r"C:\ml\looped-lm\data\tokenizer_bpe8192.json")
    ap.add_argument("--out_dir", default=r"C:\ml\looped-lm\data\tok8192")
    ap.add_argument("--train_shard", default="000_00000.parquet")
    ap.add_argument("--val_shard", default="014_00000.parquet")
    ap.add_argument("--train_tokens", type=int, default=101_000_000)
    ap.add_argument("--val_tokens", type=int, default=10_000_000)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<|endoftext|>")
    assert eos_id is not None
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw_dir)

    meta = {"vocab_size": tok.get_vocab_size(), "eos_id": eos_id,
            "tokenizer": args.tokenizer}
    print("== val ==")
    meta["val"] = encode_shard(raw / args.val_shard, tok, eos_id, args.val_tokens, out / "val.bin")
    print(meta["val"])
    print("== train ==")
    meta["train"] = encode_shard(raw / args.train_shard, tok, eos_id, args.train_tokens, out / "train.bin")
    print(meta["train"])

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
