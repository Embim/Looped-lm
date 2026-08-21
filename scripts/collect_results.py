"""Collect results/*.json into a sortable table (CSV + printed markdown)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

COLS = ["run", "params", "params_non_emb", "train_tokens", "T_train", "T_eval",
        "val_loss", "val_ppl", "val_bpb", "fwd_gflops_per_token", "train_pflops",
        "minutes", "best_T", "best_loss"]


def row(p: Path) -> dict:
    s = json.loads(p.read_text())
    m, f = s["model"], s["final"]
    sweep = s.get("depth_sweep") or {}
    best_T, best_loss = None, None
    if sweep:
        best_T = min(sweep, key=lambda k: sweep[k]["val_loss"])
        best_loss = sweep[best_T]["val_loss"]
    return {
        "run": s["run"], "params": s["params"], "params_non_emb": s["params_non_emb"],
        "train_tokens": s["train_tokens"], "T_train": m["n_loops"], "T_eval": f["n_loops"],
        "val_loss": round(f["val_loss"], 4), "val_ppl": round(f["val_ppl"], 3),
        "val_bpb": round(f["val_bpb"], 4),
        "fwd_gflops_per_token": round(s["fwd_flops_per_token"] / 1e9, 3),
        "train_pflops": round(s["train_flops_est"] / 1e15, 2),
        "minutes": round(s["wall_seconds"] / 60, 1),
        "best_T": best_T, "best_loss": None if best_loss is None else round(best_loss, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", default="val_loss")
    ap.add_argument("--filter", default="")
    a = ap.parse_args()
    files = sorted(RESULTS.glob("*.json"))
    files = [f for f in files if a.filter in f.stem]
    if not files:
        print("no results yet")
        return
    rows = []
    for f in files:
        try:
            rows.append(row(f))
        except Exception as e:
            print(f"  skip {f.name}: {e}", file=sys.stderr)
    rows.sort(key=lambda r: (r.get(a.sort) is None, r.get(a.sort)))

    out = RESULTS / "index.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    show = ["run", "T_train", "val_loss", "val_ppl", "val_bpb", "best_T", "best_loss",
            "params", "minutes"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in show}
    print("| " + " | ".join(c.ljust(widths[c]) for c in show) + " |")
    print("|" + "|".join("-" * (widths[c] + 2) for c in show) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]).ljust(widths[c]) for c in show) + " |")
    print(f"\n{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
