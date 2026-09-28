"""
aggregate_runs.py — turn many training runs into the report's tables and curves
(depth items 1, 2 and 5).

For every run folder under --runs (one per `train_ppo.py` call; tag and seed are read
from its config.json) it:
  1. evaluates the run's FINAL model (not the "best" one: that one was selected on the
     periodic-evaluation seeds, which would bias the comparison) on a FRESH seed bank
     (default 30000+, never used for training or selection) in three conditions:
       nominal (actuator σ 0.05) · stress (actuator σ 0.15 + gyro σ 0.2) · wind 5 m/s
     all judged by the independent §04 detector;
  2. groups runs by tag and reports mean ± std over seeds (and every seed's value);
  3. draws learning curves (§04 success of the periodic evaluation vs environment
     steps), mean ± std across seeds, from each run's eval_log.csv.

Outputs (in --out): per_run.csv, summary.md, learning_curves.png, ablation_bars.png

    python aggregate_runs.py --runs runs --episodes 20
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

CONDITIONS = {
    "nominal": dict(action_noise=0.05, obs_noise=0.0, wind=0.0),
    "stress": dict(action_noise=0.15, obs_noise=0.2, wind=0.0),
    "wind5": dict(action_noise=0.0, obs_noise=0.0, wind=5.0),
}

# what each tag means in run_all.bat (unknown tags are shown as-is)
TAG_INFO = {
    "final": ("Final configuration", "BC warm start + PPO, CTBR, asymmetric critic, all reward terms"),
    "bc_only": ("− PPO (BC only)", "imitation of the expert, no RL fine-tuning"),
    "no_bc": ("− BC (pure RL)", "PPO from scratch, same reward and action space"),
    "symmetric": ("− asymmetric critic", "critic sees only what the actor sees"),
    "no_prog": ("− 'new record' term", "w_prog = 0"),
    "no_demo": ("− demonstration starts", "every training episode starts at hover"),
    "motors": ("− CTBR (motor commands)", "policy outputs the 4 motor commands; pure RL (the expert speaks CTBR)"),
    "wind_ctrl": ("Wind training (control)", "70% windy episodes, stronger position terms"),
    "wind_integral": ("Wind training + position integral", "same, plus ∫(p − p0) dt in the actor's input"),
}
ORDER = list(TAG_INFO)


def find_runs(root):
    runs = []
    for cfgp in glob.glob(os.path.join(root, "*", "config.json")):
        d = os.path.dirname(cfgp)
        model = next((os.path.join(d, f) for f in ("final_model.zip", "best_model.zip")
                      if os.path.exists(os.path.join(d, f))), None)
        if model is None:
            continue
        with open(cfgp, encoding="utf-8") as f:
            rc = json.load(f)
        args = rc.get("args", {})
        runs.append(dict(dir=d, model=model, tag=args.get("tag", os.path.basename(d)),
                         seed=int(args.get("seed", 0)), timesteps=int(args.get("timesteps", 0))))
    return sorted(runs, key=lambda r: (r["tag"], r["seed"]))


def _eval_job(job):
    run, cond, n, seed0 = job
    import warnings
    warnings.filterwarnings("ignore")
    from run_baselines import make_raw_env, rollout, quick_metrics
    from flip_policy import LearnedFlipController
    from flip_reward import SuccessDetector, SuccessConfig
    env = make_raw_env(action_repeat=4)
    ctrl = LearnedFlipController(run["model"])
    det = SuccessDetector(SuccessConfig())
    c = CONDITIONS[cond]
    rows = [quick_metrics(rollout(env, ctrl, seed0 + i, detector=det, **c)) for i in range(n)]
    env.close()
    rec = [r["time_to_recover_s"] for r in rows if np.isfinite(r["time_to_recover_s"])]
    return dict(tag=run["tag"], seed=run["seed"], condition=cond, episodes=n,
                success=float(np.mean([r["success_detector"] for r in rows])),
                crash=float(np.mean([r["terminated"] for r in rows])),
                drift=float(np.mean([r["final_xy_drift_m"] for r in rows])),
                final_rate=float(np.mean([r["final_rate"] for r in rows])),
                recovery=float(np.mean(rec)) if rec else float("nan"),
                effort=float(np.mean([r["control_effort"] for r in rows])),
                model=os.path.relpath(run["model"]))


def fmt(vals, pct=False, p=2):
    v = np.array([x for x in vals if np.isfinite(x)])
    if not v.size:
        return "—"
    if pct:
        return f"{100 * v.mean():.0f}% ± {100 * v.std():.0f}" if v.size > 1 else f"{100 * v.mean():.0f}%"
    return f"{v.mean():.{p}f} ± {v.std():.{p}f}" if v.size > 1 else f"{v.mean():.{p}f}"


def learning_curves(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualize_flip import INK_2, GRID
    by_tag = defaultdict(list)
    for r in runs:
        p = os.path.join(r["dir"], "eval_log.csv")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                by_tag[r["tag"]].append((np.array([float(x["timesteps"]) for x in rows]),
                                         np.array([float(x["success_detector"]) for x in rows])))
    if not by_tag:
        return None
    palette = ["#c2410c", "#574fae", "#1f7a6d", "#b45309", "#0369a1", "#9d174d", "#4d7c0f", "#6b7280", "#7c3aed"]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    tags = [t for t in ORDER if t in by_tag] + [t for t in by_tag if t not in ORDER]
    for i, tag in enumerate(tags):
        curves = by_tag[tag]
        grid = np.unique(np.concatenate([c[0] for c in curves]))
        Y = np.array([np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in curves])
        m, s = np.nanmean(Y, axis=0), np.nanstd(Y, axis=0)
        col = palette[i % len(palette)]
        name = TAG_INFO.get(tag, (tag,))[0]
        ax.plot(grid, m, color=col, lw=2, marker="o", ms=4, label=f"{name} (n={len(curves)})")
        if len(curves) > 1:
            ax.fill_between(grid, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=col, alpha=0.12, lw=0)
    ax.set_ylim(-0.03, 1.03); ax.set_xlabel("environment steps (decisions)")
    ax.set_ylabel("§04 success (periodic eval)"); ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("Learning curves (mean ± std over seeds; step 0 = after BC, if any)")
    fig.tight_layout()
    p = os.path.join(out, "learning_curves.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def ablation_bars(agg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualize_flip import GRID
    tags = [t for t in ORDER if t in agg] + [t for t in agg if t not in ORDER]
    conds = list(CONDITIONS)
    cols = {"nominal": "#574fae", "stress": "#c2410c", "wind5": "#1f7a6d"}
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(tags) + 2), 4.4))
    w = 0.26
    x = np.arange(len(tags))
    for j, c in enumerate(conds):
        means = [np.mean(agg[t][c]["success"]) if c in agg[t] else np.nan for t in tags]
        stds = [np.std(agg[t][c]["success"]) if c in agg[t] else 0 for t in tags]
        ax.bar(x + (j - 1) * w, means, w * 0.92, yerr=stds, capsize=3, color=cols[c], label=c)
    ax.set_xticks(x); ax.set_xticklabels([TAG_INFO.get(t, (t,))[0] for t in tags], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("§04 success"); ax.grid(color=GRID, lw=0.6, axis="y")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.set_title("Ablations: success per condition (mean ± std over seeds)")
    fig.tight_layout()
    p = os.path.join(out, "ablation_bars.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=30_000, help="fresh seed bank for the final comparison")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--out", default="results/depth/aggregate")
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    os.makedirs(a.out, exist_ok=True)
    runs = find_runs(a.runs)
    if not runs:
        raise SystemExit(f"no finished runs (config.json + final_model.zip) under {a.runs}")
    print(f"{len(runs)} runs: " + ", ".join(f"{r['tag']}/s{r['seed']}" for r in runs), flush=True)
    jobs = [(r, c, a.episodes, a.seed0) for r in runs for c in a.conditions]
    rows = []
    with Pool(a.workers) as p:
        for res in p.imap_unordered(_eval_job, jobs):
            rows.append(res)
            print(f"  {res['tag']:14s} seed {res['seed']}  {res['condition']:7s} success {100 * res['success']:5.1f}%  "
                  f"drift {res['drift']:.2f} m", flush=True)
    rows.sort(key=lambda r: (r["tag"], r["seed"], r["condition"]))
    with open(os.path.join(a.out, "per_run.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        for k in ("success", "crash", "drift", "final_rate", "recovery", "effort"):
            agg[r["tag"]][r["condition"]][k].append(r[k])
    tags = [t for t in ORDER if t in agg] + [t for t in agg if t not in ORDER]
    lines = [f"# Aggregated results ({a.episodes} episodes per run and condition, fresh seeds {a.seed0}+, "
             f"final models, §04 detector)", "",
             "Mean ± std over training seeds (n = number of seeds). Conditions: nominal = actuator σ 0.05; "
             "stress = actuator σ 0.15 + gyro σ 0.2 rad/s; wind5 = 5 m/s median wind.", "",
             "| Configuration | n | Success nominal | Success stress | Success wind 5 | Drift wind 5 [m] | "
             "Recovery nominal [s] | Effort nominal |", "|---|---|---|---|---|---|---|---|"]
    for t in tags:
        g = agg[t]
        n = len(g[next(iter(g))]["success"])
        lines.append(f"| {TAG_INFO.get(t, (t,))[0]} | {n} | {fmt(g['nominal']['success'], True)} | "
                     f"{fmt(g['stress']['success'], True)} | {fmt(g['wind5']['success'], True)} | "
                     f"{fmt(g['wind5']['drift'])} | {fmt(g['nominal']['recovery'])} | {fmt(g['nominal']['effort'], p=3)} |")
    lines += ["", "What each row changes (relative to the final configuration):", ""]
    for t in tags:
        if t in TAG_INFO:
            lines.append(f"- **{TAG_INFO[t][0]}** (`{t}`): {TAG_INFO[t][1]}")
    lines += ["", "Per-seed values: `per_run.csv`. Learning curves: `learning_curves.png`. "
              "Bars: `ablation_bars.png`.", "",
              "How to read it: a component matters if removing it changes the result by more than the seed-to-seed "
              "spread (the ± column). With 3 seeds, treat differences smaller than ~2 std as inconclusive, and say so."]
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("->", learning_curves(runs, a.out))
    print("->", ablation_bars(agg, a.out))
    print("->", os.path.join(a.out, "summary.md"))


if __name__ == "__main__":
    main()
