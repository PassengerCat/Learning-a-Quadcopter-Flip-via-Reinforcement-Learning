"""
robustness_sweep.py — success and drift as FUNCTIONS of disturbance strength (depth item 3).

Instead of one "stress" point, sweep:
  * noise:  actuator σ ∈ {0, .05, .10, .15, .20, .30}, with gyro σ = (4/3)·actuator σ
            (so 0.15 ↔ the 0.2 rad/s gyro noise of the stress test)
  * wind:   median speed ∈ {0, 2, 4, 6, 8} m/s (the env's RANDOMSINE model)
for the scripted flip, the PPO policy and the PPO + PID hybrid, on the same seeds,
judged by the §04 detector. Episodes run in parallel on all cores.

Outputs (in --out): robustness.csv, robustness_summary.md, robustness_curves.png

    python robustness_sweep.py --model models/ppo_flip.zip --episodes 10
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
WIND_LEVELS = [0.0, 2.0, 4.0, 6.0, 8.0]
CONTROLLERS = ["scripted_flip", "ppo_flip", "ppo_flip+pid"]


def _job(args):
    ctrl_name, kind, level, model_path, n, seed0 = args
    import warnings
    warnings.filterwarnings("ignore")
    from run_baselines import make_raw_env, rollout, quick_metrics, wilson
    from baselines import make_controller
    from flip_reward import SuccessDetector, SuccessConfig
    env = make_raw_env(action_repeat=4)
    if ctrl_name == "scripted_flip":
        ctrl = make_controller("scripted_flip", env)
    else:
        from flip_policy import LearnedFlipController
        ctrl = LearnedFlipController(model_path, pid_handoff=(ctrl_name == "ppo_flip+pid"))
    det = SuccessDetector(SuccessConfig())
    an = level if kind == "noise" else 0.0
    on = level * 4.0 / 3.0 if kind == "noise" else 0.0
    wind = level if kind == "wind" else 0.0
    rows = []
    for i in range(n):
        tr = rollout(env, ctrl, seed0 + i, action_noise=an, obs_noise=on, detector=det, wind=wind)
        rows.append(quick_metrics(tr))
    env.close()
    k = sum(r["success_detector"] for r in rows)
    lo, hi = wilson(k, n)
    return dict(controller=ctrl_name, kind=kind, level=level, n=n, success=k / n, ci_lo=lo, ci_hi=hi,
                crash=float(np.mean([r["terminated"] for r in rows])),
                drift=float(np.mean([r["final_xy_drift_m"] for r in rows])),
                final_rate=float(np.mean([r["final_rate"] for r in rows])),
                final_tilt=float(np.mean([r["final_tilt_deg"] for r in rows])))


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualize_flip import COLORS, LABELS, INK_2, MUTED, GRID
    colors = dict(COLORS, **{"ppo_flip+pid": "#1f7a6d"})
    labels = dict(LABELS, **{"ppo_flip+pid": "PPO + PID hold"})
    markers = {"scripted_flip": "o", "ppo_flip": "s", "ppo_flip+pid": "^"}
    styles = {"scripted_flip": "-", "ppo_flip": "-", "ppo_flip+pid": (0, (4, 2))}   # hybrid dashed: after the
    # flip it runs the same PID as the scripted flip, so under noise the two curves coincide
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.4))
    panels = [(axs[0, 0], "noise", "success", "§04 success rate"),
              (axs[0, 1], "wind", "success", "§04 success rate"),
              (axs[1, 0], "noise", "final_rate", "final ‖ω‖ [rad/s]"),
              (axs[1, 1], "wind", "drift", "horizontal drift [m]")]
    for ax, kind, key, ylabel in panels:
        for c in CONTROLLERS:
            rr = sorted([r for r in rows if r["controller"] == c and r["kind"] == kind], key=lambda r: r["level"])
            if not rr:
                continue
            x = [r["level"] for r in rr]
            y = [r[key] for r in rr]
            ax.plot(x, y, color=colors[c], lw=2, ls=styles[c], marker=markers[c], ms=6,
                    mfc="white" if c == "ppo_flip+pid" else colors[c], label=labels[c])
            if key == "success":
                ax.fill_between(x, [r["ci_lo"] for r in rr], [r["ci_hi"] for r in rr], color=colors[c],
                                alpha=0.10, lw=0)
        if key == "success":
            ax.set_ylim(-0.03, 1.03)
        if key == "final_rate":
            ax.axhline(0.8, color=MUTED, lw=0.8, ls=":")
            ax.text(0.0, 0.83, "§04 stillness threshold (0.8 rad/s)", fontsize=8, color=INK_2)
        ax.set_xlabel("actuator noise σ (gyro σ = 4/3 of it)" if kind == "noise" else "median wind [m/s]")
        ax.set_ylabel(ylabel)
        ax.grid(color=GRID, lw=0.6)
    axs[0, 0].legend(fontsize=8, loc="lower left")
    axs[0, 0].set_title("Noise robustness"); axs[0, 1].set_title("Wind robustness")
    fig.suptitle("Where each controller breaks (shaded: 95% Wilson interval)", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "robustness_curves.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/ppo_flip.zip")
    ap.add_argument("--episodes", type=int, default=10, help="per controller and level")
    ap.add_argument("--seed0", type=int, default=40_000, help="disjoint from training/selection seeds")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results/depth/robustness")
    ap.add_argument("--replot", action="store_true", help="only redraw the figure from robustness.csv")
    a = ap.parse_args()
    if a.replot:
        with open(os.path.join(a.out, "robustness.csv"), encoding="utf-8") as f:
            rows = [{k: (float(v) if k not in ("controller", "kind") else v) for k, v in r.items()}
                    for r in csv.DictReader(f)]
        plot(rows, a.out)
        print("->", a.out)
        return
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    os.makedirs(a.out, exist_ok=True)
    jobs = [(c, "noise", lv, a.model, a.episodes, a.seed0) for c in CONTROLLERS for lv in NOISE_LEVELS] + \
           [(c, "wind", lv, a.model, a.episodes, a.seed0) for c in CONTROLLERS for lv in WIND_LEVELS]
    print(f"{len(jobs)} jobs x {a.episodes} episodes on {a.workers} workers ...", flush=True)
    with Pool(a.workers) as p:
        rows = []
        for r in p.imap_unordered(_job, jobs):
            rows.append(r)
            print(f"  {r['controller']:14s} {r['kind']:5s} {r['level']:4g}: success {r['success'] * 100:5.1f}%  "
                  f"drift {r['drift']:.2f} m  final ω {r['final_rate']:.2f}", flush=True)
    rows.sort(key=lambda r: (r["kind"], r["controller"], r["level"]))
    with open(os.path.join(a.out, "robustness.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    lines = [f"# Robustness sweep ({a.episodes} episodes per point, seeds {a.seed0}+)", ""]
    for kind, levels in (("noise", NOISE_LEVELS), ("wind", WIND_LEVELS)):
        head = "actuator σ" if kind == "noise" else "wind [m/s]"
        lines += [f"## {kind}", "", f"| {head} | " + " | ".join(CONTROLLERS) + " |",
                  "|---|" + "---|" * len(CONTROLLERS)]
        for lv in levels:
            cells = []
            for c in CONTROLLERS:
                r = next((x for x in rows if x["controller"] == c and x["kind"] == kind and x["level"] == lv), None)
                cells.append("—" if r is None else
                             f"{r['success'] * 100:.0f}%" + (f" / {r['drift']:.2f} m" if kind == "wind" else
                                                             f" / ω {r['final_rate']:.2f}"))
            lines.append(f"| {lv:g} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("Cells: §04 success / (noise: final ‖ω‖ in rad/s · wind: horizontal drift in m).")
    with open(os.path.join(a.out, "robustness_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    plot(rows, a.out)
    print("->", a.out)


if __name__ == "__main__":
    main()
