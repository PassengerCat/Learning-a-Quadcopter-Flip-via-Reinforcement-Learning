"""Report figure and table: success rates of every controller under each condition.

Reads finished final-comparison runs (Evaluation/runs/<name>/<condition>/summary.json, as
written by final_comparison.py) and never re-runs an episode. Several runs can be combined
(e.g. one run per group of controllers); a controller found in more than one run is taken
from the last run listed. All runs must use the same detector and seeds.

Figure: one panel per condition (small multiples, one shared controller axis), a bar per
controller labelled with its successes out of 50 and, in the noisy conditions, a horizontal line
spanning the 95% Wilson interval of the harness summaries (--no-ci leaves it out). The nominal condition is deterministic, so its 50
episodes are identical: it is drawn as pass/fail without an interval.

    python Evaluation/plot_comparison.py --runs FINAL_PT_ALL --name success_rates
Output: Evaluation/runs/figures/<name>.png and .pdf, <name>_table.tex (LaTeX rows) and
        <name>_table.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

LABELS = {"scripted_flip": "Scripted flip", "three_phase_default": "Three-phase (default)",
          "map_elites_final": "MAP-Elites", "bc_only": "BC only", "bc_ppo": "BC-PPO",
          "ppo_mtr": "MTR-PPO (CTBR)", "ppo_mtr_motors": "MTR-PPO (motors)",
          "pid_hover": "PID hover", "random": "Random"}
ORDER = ["pid_hover", "random", "scripted_flip", "three_phase_default", "map_elites_final",
         "bc_only", "bc_ppo", "ppo_mtr", "ppo_mtr_motors"]
COND_LABELS = {"nominal": "Nominal (deterministic)", "ppo_track_nominal": "Noisy nominal",
               "ppo_track_stress": "Stress", "action_noise_005": "Action noise 0.05",
               "stress": "Stress (no sensor noise)"}
DETERMINISTIC = {"nominal"}
BAR = "#2a78d6"          # one hue: every panel shows the same quantity (reference palette, slot 1)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(runs, conditions):
    data, meta = {c: {} for c in conditions}, {}
    for run in runs:
        for c in conditions:
            path = HERE / "runs" / run / c / "summary.json"
            if not path.exists():
                continue
            s = json.loads(path.read_text(encoding="utf-8"))
            m = s["meta"]
            key = (m.get("detector"), tuple(m.get("seeds", [])))
            if c in meta and meta[c] != key:
                raise SystemExit(f"{run}/{c}: detector or seeds differ from the other runs ({key} vs {meta[c]})")
            meta[c] = key
            data[c].update(s["controllers"])
    return data, meta


def table_rows(data, conditions, controllers, show_ci: bool = False):
    rows = []
    for n in controllers:
        cells = []
        for c in conditions:
            d = data[c].get(n)
            if d is None:
                cells.append(None)
                continue
            k, e = d["successes"], d["episodes"]
            lo, hi = d["success_ci95"]
            cells.append((k, e, c in DETERMINISTIC, (lo, hi) if show_ci and c not in DETERMINISTIC else None))
        rows.append((n, cells))
    return rows


def write_tables(rows, conditions, out):
    txt = [f"{'controller':<24}" + "".join(f"{COND_LABELS.get(c, c):>26}" for c in conditions)]
    tex = ["% controller & " + " & ".join(COND_LABELS.get(c, c) for c in conditions) + r" \\"]
    for n, cells in rows:
        t, x = [], []
        for cell in cells:
            if cell is None:
                t.append(f"{'-':>26}"), x.append("--")
                continue
            k, e, det, ci = cell
            if ci is None:
                t.append(f"{f'{k}/{e}':>26}"), x.append(f"{k}/{e}")
            else:
                t.append(f"{f'{k}/{e} [{ci[0]:.2f}, {ci[1]:.2f}]':>26}")
                x.append(f"{k}/{e} [{ci[0]:.2f}, {ci[1]:.2f}]")
        txt.append(f"{LABELS.get(n, n):<24}" + "".join(t))
        tex.append(f"{LABELS.get(n, n)} & " + " & ".join(x) + r" \\")
    out.with_name(out.name + "_table.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    out.with_name(out.name + "_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(txt))


def figure(rows, conditions, out, show_ci: bool = False):
    names = [LABELS.get(n, n) for n, _ in rows]
    fig, axes = plt.subplots(1, len(conditions), figsize=(3.0 * len(conditions) + 1.6, 0.42 * len(rows) + 1.3),
                             sharey=True)
    axes = [axes] if len(conditions) == 1 else list(axes)
    y = list(range(len(rows)))[::-1]                     # first controller on top
    for j, (ax, c) in enumerate(zip(axes, conditions)):
        for yi, (n, cells) in zip(y, rows):
            cell = cells[j]
            if cell is None:
                ax.text(0.02, yi, "not run", va="center", fontsize=8, color=MUTED)
                continue
            k, e, det, ci = cell
            rate = k / e
            ax.barh(yi, rate, height=0.56, color=BAR, edgecolor="white", linewidth=0)
            if ci is not None:
                ax.plot([ci[0], ci[1]], [yi, yi], color=INK, lw=1.2)
                ax.plot([ci[0]] * 2, [yi - 0.12, yi + 0.12], color=INK, lw=1.2)
                ax.plot([ci[1]] * 2, [yi - 0.12, yi + 0.12], color=INK, lw=1.2)
            label = ("pass" if k == e else ("fail" if k == 0 else f"{k}/{e}")) if det else f"{k}/{e}"
            ax.text(max(rate, ci[1] if ci else rate) + 0.03, yi, label, va="center", ha="left",
                    fontsize=7.5, color=INK)
        ax.set_xlim(0, 1.25)
        ax.spines["bottom"].set_bounds(0, 1.0)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "50%", "100%"], fontsize=8, color=MUTED)
        ax.set_title(COND_LABELS.get(c, c), fontsize=9.5, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=8.5, color=INK)
    fig.supxlabel("success rate (primary detector), 50 episodes" + ("; horizontal lines: 95% Wilson interval" if show_ci else ""),
                  fontsize=8.5, color=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.parent / f"{out.name}.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="final-comparison runs under Evaluation/runs/")
    ap.add_argument("--conditions", nargs="+", default=["nominal", "ppo_track_nominal", "ppo_track_stress"])
    ap.add_argument("--controllers", nargs="+", default=None, help="order of the rows (default: all found)")
    ap.add_argument("--name", default="success_rates")
    ap.add_argument("--no-ci", action="store_true", help="leave out the 95%% Wilson interval of the noisy conditions")
    a = ap.parse_args()
    data, _ = load(a.runs, a.conditions)
    found = {n for c in a.conditions for n in data[c]}
    if not found:
        raise SystemExit("no summary.json found for these runs and conditions")
    controllers = a.controllers or [n for n in ORDER if n in found] + sorted(found - set(ORDER))
    out_dir = HERE / "runs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / a.name
    rows = table_rows(data, a.conditions, controllers, not a.no_ci)
    write_tables(rows, a.conditions, out)
    figure(rows, a.conditions, out, not a.no_ci)
    print("written to", out_dir)


if __name__ == "__main__":
    main()
