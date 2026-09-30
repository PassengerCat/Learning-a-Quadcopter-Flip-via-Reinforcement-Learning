"""MAP-Elites analysis (M5, step 1): archive heatmap and convergence curves.

Reads a finished (or checkpointed) run from MAP_Elites/runs/<name>/ and writes
figures to MAP_Elites/runs/<name>/figures/. It never re-runs episodes.

    python MAP_Elites/analysis.py --name ME1
    python MAP_Elites/analysis.py --compare ME1=ME1,ME1_s1,ME1_s2 ME2=ME2,ME2_s1,ME2_s2
        -> MAP_Elites/runs/comparison/seed_comparison.png (one colour per configuration, one line per seed)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from archive import Archive            # noqa: E402

# one-hue sequential ramp (light = worse, dark = better); neutral fill for empty cells
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
EMPTY, SURFACE, INK, INK2, GRID = "#f0efec", "#fcfcfb", "#0b0b0b", "#52514e", "#b9b8b2"


def load_run(run_dir):
    run = Path(run_dir)
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    return (Archive.load(run / state["archive"]),
            json.loads((run / "history.json").read_text(encoding="utf-8")),
            json.loads((run / "config.json").read_text(encoding="utf-8")))


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_heatmap(archive: Archive, path, mark=None, title="MAP-Elites archive"):
    """Fitness per cell (darker = better); empty cells in neutral grey. `mark` = descriptors to star."""
    grid = archive.fitness_grid()                       # shape (n_duration, n_climb)
    (d0, d1), (c0, c1) = archive.bounds
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.add_patch(plt.Rectangle((d0, c0), d1 - d0, c1 - c0, color=EMPTY, zorder=0))
    filled = grid[np.isfinite(grid)]
    if filled.size:
        img = ax.imshow(np.ma.masked_invalid(grid.T), origin="lower", extent=(d0, d1, c0, c1),
                        aspect="auto", cmap=LinearSegmentedColormap.from_list("seq", SEQ),
                        vmin=filled.min(), vmax=filled.max(), interpolation="nearest", zorder=1)
        cb = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("fitness  −(effort + altitude loss)   (higher is better)", color=INK2, fontsize=8)
        cb.ax.tick_params(colors=INK2, labelsize=7)
        cb.outline.set_edgecolor(GRID)
    for x in np.linspace(d0, d1, archive.shape[0] + 1):
        ax.axvline(x, color=SURFACE, lw=1.2, zorder=2)          # 2px-style gap between cells
    for y in np.linspace(c0, c1, archive.shape[1] + 1):
        ax.axhline(y, color=SURFACE, lw=1.2, zorder=2)
    if mark is not None:
        ax.scatter([min(max(mark[0], d0), d1)], [min(max(mark[1], c0), c1)], marker="*", s=140,
                   color=INK, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.annotate("default controller", (mark[0], mark[1]), xytext=(8, 6), textcoords="offset points",
                    fontsize=8, color=INK, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.9))
    s = archive.stats()
    ax.set_xlim(d0, d1); ax.set_ylim(c0, c1)
    ax.set_xlabel("rotation duration  [s]", color=INK2, fontsize=9)
    ax.set_ylabel("max climb above start  [m]", color=INK2, fontsize=9)
    ax.set_title(f"{title}\n{s['filled']}/{int(np.prod(archive.shape))} cells filled · best fitness "
                 f"{s['best_fitness']:.3f} · QD-score {s['qd_score']:.1f}   (grey = no valid flip found)"
                 if s["filled"] else f"{title}\nempty archive", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_convergence(history, path, n_cells=100):
    """Coverage and QD-score against evaluations: two panels, one y-axis each."""
    ev = np.array([h["evals"] for h in history])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, label, scale in ((axes[0], "filled", "cells filled", 1.0),
                                  (axes[1], "qd_score", "QD-score", 1.0)):
        _style(ax)
        ax.plot(ev, np.array([h[key] for h in history], float) * scale, color="#2a78d6", lw=2)
        ax.set_xlabel("evaluations", color=INK2, fontsize=9)
        ax.set_title(label, color=INK, fontsize=9.5, loc="left")
        ax.grid(axis="y", color=EMPTY, lw=0.8)
    axes[0].set_ylim(0, n_cells)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


CATEGORICAL = ["#2a78d6", "#eb6834"]        # validated pair (light surface): blue, orange
MARKERS = ["o", "s"]                         # secondary encoding, so identity is not colour-only


def plot_seed_comparison(groups: dict, path):
    """groups = {label: [run_dir, ...]}. Three panels, one measure each (no shared y-axis):
    QD-score vs evaluations (one line per seed), final best fitness, final median cell fitness."""
    if not 1 <= len(groups) <= len(CATEGORICAL):
        raise ValueError(f"need 1..{len(CATEGORICAL)} groups")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), dpi=150, gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        _style(ax)
        ax.grid(axis="y", color=EMPTY, lw=0.8)
    for gi, (label, runs) in enumerate(groups.items()):
        col, mk = CATEGORICAL[gi], MARKERS[gi]
        best, med = [], []
        for run in runs:
            archive, history, _ = load_run(run)
            ev = [h["evals"] for h in history]
            axes[0].plot(ev, [h["qd_score"] for h in history], color=col, lw=1.5, alpha=0.9)
            f = np.array([e["fitness"] for e in archive.cells.values()])
            best.append(f.max())
            med.append(float(np.median(f)))
        axes[0].plot([], [], color=col, lw=2, marker=mk, label=f"{label} ({len(runs)} seeds)")
        for ax, vals in ((axes[1], best), (axes[2], med)):
            x = gi + np.linspace(-0.08, 0.08, len(vals))
            ax.scatter(x, vals, color=col, marker=mk, s=40, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.hlines(np.mean(vals), gi - 0.2, gi + 0.2, color=INK2, lw=1.2, zorder=2)
    axes[0].set_xlabel("evaluations", color=INK2, fontsize=9)
    axes[0].set_title("QD-score", color=INK, fontsize=9.5, loc="left")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    for ax, title in ((axes[1], "final best fitness"), (axes[2], "final median cell fitness")):
        ax.set_xticks(range(len(groups)), list(groups), fontsize=8)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_title(title, color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("dots: seeds · bar: mean", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def parse_groups(specs) -> dict:
    """['ME1=ME1,ME1_s1', 'ME2=ME2,ME2_s1'] -> {'ME1': [runs/ME1, runs/ME1_s1], ...}"""
    groups = {}
    for spec in specs:
        label, sep, names = spec.partition("=")
        runs = [n for n in names.split(",") if n]
        if not sep or not label or not runs or label in groups:
            raise ValueError(f"--compare {spec!r}: expected LABEL=run,run,... with a new label")
        groups[label] = [HERE / "runs" / n for n in runs]
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--compare", nargs="+", metavar="LABEL=run,run")
    a = ap.parse_args()
    if bool(a.name) == bool(a.compare):
        ap.error("give exactly one of --name or --compare")
    if a.compare:
        try:
            groups = parse_groups(a.compare)
        except ValueError as e:
            ap.error(str(e))
        out = HERE / "runs" / "comparison"
        out.mkdir(parents=True, exist_ok=True)
        plot_seed_comparison(groups, out / "seed_comparison.png")
        print("figure:", out / "seed_comparison.png")
        return
    run = HERE / "runs" / a.name
    archive, history, _ = load_run(run)
    import problem as P
    fig_dir = run / "figures"
    fig_dir.mkdir(exist_ok=True)
    default = P.evaluate_genome(P.FlipParams().as_array())
    plot_heatmap(archive, fig_dir / "archive_heatmap.png", mark=default.descriptors, title=f"MAP-Elites archive — {a.name}")
    plot_convergence(history, fig_dir / "convergence.png", int(np.prod(archive.shape)))
    print(json.dumps(archive.stats(), indent=1))
    print("figures:", fig_dir)


if __name__ == "__main__":
    main()
