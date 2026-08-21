"""Figures for the report, built from results/*.json and results/*.eval.json.

    python scripts/plots.py                 # everything it can find
    python scripts/plots.py --only depth_scaling
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "report" / "figures"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def load(pattern="*.json"):
    out = {}
    for p in sorted(RESULTS.glob(pattern)):
        if p.name == "index.csv" or p.suffixes[:1] == [".eval"]:
            continue
        try:
            d = json.loads(p.read_text())
            if "final" in d:
                out[d["run"]] = d
        except Exception:
            pass
    return out


def fig_depth_scaling(runs):
    """Val loss of models *trained* at T, against T -- the saturation curve."""
    pts = sorted((d["model"]["n_loops"], d["final"]["val_loss"], d["final"]["val_bpb"],
                  d["train_flops_est"])
                 for n, d in runs.items() if n.startswith("A_depth"))
    if len(pts) < 2:
        return None
    T = [p[0] for p in pts]
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.1))
    ax[0].plot(T, [p[1] for p in pts], "o-", color="#2563eb")
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(T); ax[0].set_xticklabels(T)
    ax[0].set_xlabel("loops T (trained and evaluated at T)")
    ax[0].set_ylabel("val loss (nats/token)")
    ax[0].set_title("Depth-recurrence scaling")
    ax[1].plot([p[3] / 1e15 for p in pts], [p[1] for p in pts], "o-", color="#b42318")
    for p in pts:
        ax[1].annotate(f"T={p[0]}", (p[3] / 1e15, p[1]), fontsize=7,
                       xytext=(3, 3), textcoords="offset points")
    ax[1].set_xscale("log"); ax[1].set_xlabel("training compute (PFLOPs)")
    ax[1].set_ylabel("val loss"); ax[1].set_title("Loss vs compute at fixed parameters")
    fig.tight_layout()
    return fig


def fig_readout_curve(runs, names=None):
    """Loss when reading out after each individual loop of one trained model."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    any_ = False
    for n, d in sorted(runs.items()):
        per = d["final"].get("per_step_loss")
        if not per or (names and n not in names):
            continue
        ax.plot(range(1, len(per)), per[1:], "-", lw=1.4, label=f"{n} (T={d['model']['n_loops']})")
        any_ = True
    if not any_:
        return None
    ax.set_xlabel("read-out after loop t"); ax.set_ylabel("val loss")
    ax.set_title("Where the trajectory stops helping"); ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_trajectory(runs):
    """||delta_t||, ||h_t|| and cos(delta_t, delta_{t-1}) -- collapse diagnostics."""
    cand = {n: d for n, d in runs.items() if d["final"].get("stats", {}).get("delta_rms")}
    if not cand:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for n, d in sorted(cand.items()):
        st = d["final"]["stats"]
        if len(st["delta_rms"]) < 3:
            continue
        x = range(1, len(st["delta_rms"]) + 1)
        ax[0].plot(x, st["delta_rms"], lw=1.3, label=n)
        ax[1].plot(x, st["state_rms"], lw=1.3, label=n)
        if st.get("cos_prev"):
            ax[2].plot(range(2, len(st["cos_prev"]) + 2), st["cos_prev"], lw=1.3, label=n)
    for a_, t, yl in zip(ax, ["update size", "state norm", "successive update alignment"],
                         ["RMS(delta_t)", "RMS(h_t)", "cos(delta_t, delta_{t-1})"]):
        a_.set_xlabel("loop t"); a_.set_ylabel(yl); a_.set_title(t)
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    return fig


def fig_early_exit():
    """Loss against the average number of loops actually spent per token."""
    files = sorted(RESULTS.glob("*.eval.json"))
    if not files:
        return None
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for p in files:
        d = json.loads(p.read_text())
        name = p.name.replace(".eval.json", "")
        dc = d.get("depth_curve", {}).get("loss")
        if dc:
            ax.plot(range(1, len(dc)), dc[1:], "--", lw=1.0, color="#9aa3af",
                    label="fixed depth" if p is files[0] else None)
        for mode, rows in d.get("early_exit", {}).items():
            ax.plot([r["avg_loops"] for r in rows], [r["loss"] for r in rows], "o-",
                    ms=3, lw=1.3, label=f"{name}: exit by {mode}")
    ax.set_xlabel("average loops per token"); ax.set_ylabel("val loss")
    ax.set_title("Adaptive depth at equal average compute"); ax.legend(fontsize=6)
    fig.tight_layout()
    return fig


def fig_strata():
    """Depth curve for the easiest half against the hardest few percent."""
    files = sorted(RESULTS.glob("*.eval.json"))
    if not files:
        return None
    d = json.loads(files[0].read_text())
    st = d.get("difficulty_strata")
    if not st:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for k, v in st.items():
        y = v["loss"]
        ax[0].plot(range(1, len(y)), y[1:], lw=1.4, label=f"{k} (n={v['n']})")
        y0 = y[1] if len(y) > 1 else 1.0
        ax[1].plot(range(1, len(y)), [yy / y0 for yy in y[1:]], lw=1.4, label=k)
    ax[0].set_ylabel("val loss"); ax[0].set_yscale("log")
    ax[1].set_ylabel("loss relative to t=1")
    for a_ in ax:
        a_.set_xlabel("read-out after loop t")
    ax[0].set_title("Absolute loss by token difficulty")
    ax[1].set_title("Relative gain from depth")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_group_bars(runs, prefix, title):
    sel = {n: d for n, d in runs.items() if n.startswith(prefix)}
    if len(sel) < 2:
        return None
    base = runs.get("A_depth16", {}).get("final", {}).get("val_loss")
    items = sorted(sel.items(), key=lambda kv: kv[1]["final"]["val_loss"])
    fig, ax = plt.subplots(figsize=(5.6, 0.34 * len(items) + 1.3))
    names = [n for n, _ in items]
    vals = [d["final"]["val_loss"] for _, d in items]
    ax.barh(names, vals, color="#2563eb", height=0.6)
    if base:
        ax.axvline(base, color="#b42318", lw=1.2, ls="--", label="plain loop, T=16")
        ax.legend(fontsize=7)
    lo, hi = min(vals + ([base] if base else [])), max(vals + ([base] if base else []))
    ax.set_xlim(lo - 0.05 * (hi - lo + 1e-6), hi + 0.05 * (hi - lo + 1e-6))
    ax.invert_yaxis(); ax.set_xlabel("val loss"); ax.set_title(title)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


FIGURES = {
    "depth_scaling": lambda r: fig_depth_scaling(r),
    "readout_curve": lambda r: fig_readout_curve(r),
    "trajectory": lambda r: fig_trajectory(r),
    "early_exit": lambda r: fig_early_exit(),
    "difficulty_strata": lambda r: fig_strata(),
    "group_C_depth_cond": lambda r: fig_group_bars(r, "C_", "Depth conditioning"),
    "group_D_update": lambda r: fig_group_bars(r, "D_", "Update rule"),
    "group_E_readout": lambda r: fig_group_bars(r, "E_", "Read-out"),
    "group_F_memory": lambda r: fig_group_bars(r, "F_", "Loop memory"),
    "group_I_supervision": lambda r: fig_group_bars(r, "I_", "Intermediate supervision"),
    "group_J_exploration": lambda r: fig_group_bars(r, "J_", "Exploration in the loop"),
    "group_K_schedule": lambda r: fig_group_bars(r, "K_", "Loop schedule / truncated BPTT"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    runs = load()
    if not runs:
        print("no results yet")
        return
    made = 0
    for name, fn in FIGURES.items():
        if a.only and a.only != name:
            continue
        try:
            fig = fn(runs)
        except Exception as e:
            print(f"  {name}: failed ({e})")
            continue
        if fig is None:
            continue
        p = FIGS / f"{name}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p.relative_to(ROOT)}")
        made += 1
    print(f"{made} figure(s) from {len(runs)} runs")


if __name__ == "__main__":
    main()
