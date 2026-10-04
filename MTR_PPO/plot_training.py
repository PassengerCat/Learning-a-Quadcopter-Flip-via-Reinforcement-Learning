"""Report figure: training curves of MTR-PPO runs.

Three panels with a shared step axis: the mean episode return during training
(rollout/ep_rew_mean), the success rate of the evaluation during training (10 nominal harness
episodes every 0.2M steps, primary detector) and the mean final body rate of those episodes
(both from eval_log.csv; the last one shows a residual oscillation in the final hover). A run is given as LABEL=PATH, where PATH is either
  * a training run folder (MTR_PPO/runs/<run>/ with eval_log.csv; the TensorBoard events are read
    from the sibling folder <run>_1/), or
  * a log folder of the repository (Logs/MTR_PPO/runs/<run>/ with eval_log.csv and
    training_curves.txt, where the return is stored every 0.1M or 0.2M steps).
The return is not comparable between runs whose reward differs (e.g. with and without the
survival bonus); compare the evaluation panel across such runs.

    python MTR_PPO/plot_training.py --name mtr_final ^
        "CTBR (run 13)=Logs/MTR_PPO/runs/run13_ctbr_seed0_scratch_all_components" ^
        "motors (run 14)=Logs/MTR_PPO/runs/run14_motors_seed0_scratch_all_components" --mark 1.5e6
Output: MTR_PPO/runs/figures/<name>.png and .pdf
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "Evaluation"))

import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

from plot_comparison import GRID, INK, MUTED   # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def read_eval(folder: Path):
    with open(folder / "eval_log.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return ([float(r["timesteps"]) for r in rows], [float(r["success_ppo_track"]) for r in rows],
            [float(r["final_omega"]) for r in rows])


def read_return(folder: Path):
    events = glob.glob(str(folder.parent / (folder.name + "_1") / "events*"))
    if events:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(events[0], size_guidance={"scalars": 0})
        ea.Reload()
        sc = ea.Scalars("rollout/ep_rew_mean")
        return [s.step for s in sc], [s.value for s in sc]
    curves = folder / "training_curves.txt"
    if curves.exists():
        lines = curves.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "rollout/ep_rew_mean:":
                pts = re.findall(r"([\d.]+)M=([-\d.e+]+)", lines[i + 1])
                return [float(s) * 1e6 for s, _ in pts], [float(v) for _, v in pts]
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="LABEL=PATH")
    ap.add_argument("--name", default="mtr_training")
    ap.add_argument("--mark", type=float, nargs="*", default=[], help="vertical marks at these steps (e.g. a chosen checkpoint)")
    a = ap.parse_args()
    if len(a.runs) > len(SERIES):
        ap.error(f"at most {len(SERIES)} runs in one figure")
    fig, (ax_r, ax_s, ax_w) = plt.subplots(3, 1, figsize=(6.6, 6.6), sharex=True)
    for i, spec in enumerate(a.runs):
        label, sep, path = spec.partition("=")
        if not sep:
            ap.error(f"{spec!r}: expected LABEL=PATH")
        folder = Path(path) if Path(path).is_absolute() else (Path.cwd() / path)
        c = SERIES[i]
        steps, ret = read_return(folder)
        if steps:
            ax_r.plot([s / 1e6 for s in steps], ret, color=c, lw=1.6, label=label)
        es, succ, wfin = read_eval(folder)
        ax_s.plot([s / 1e6 for s in es], succ, color=c, lw=1.6, marker="o", ms=4, label=label)
        ax_w.plot([s / 1e6 for s in es], [max(w, 1e-3) for w in wfin], color=c, lw=1.6, marker="o", ms=4, label=label)
    for m in a.mark:
        for ax in (ax_r, ax_s, ax_w):
            ax.axvline(m / 1e6, color=MUTED, lw=0.9, ls="--")
    ax_r.set_ylabel("mean episode return", fontsize=8.5, color=INK)
    ax_s.set_ylabel("evaluation success\n(10 nominal episodes)", fontsize=8.5, color=INK)
    ax_s.set_ylim(-0.05, 1.05)
    ax_w.set_yscale("log")                 # values below 1e-3 rad/s are drawn at 1e-3
    ax_w.axhline(0.8, color=MUTED, lw=0.9, ls=":")        # the primary detector's stillness threshold
    ax_w.set_ylabel("final body rate\n(evaluation) [rad/s]", fontsize=8.5, color=INK)
    ax_w.set_xlabel("training steps [millions]", fontsize=8.5, color=INK)
    for ax in (ax_r, ax_s, ax_w):
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(labelsize=8, colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
    ax_r.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    out = HERE / "runs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{a.name}.{ext}", dpi=200)
    print("written to", out)


if __name__ == "__main__":
    main()
