"""Report figure and table: how each controller flies the flip (one nominal episode each).

Runs one recorded nominal episode per controller on the harness (same seed for all; the
nominal benchmark is deterministic, so any seed gives the same episode) and reports
  * time series: rotation (the §04 detector's revolution), altitude change, deviation of the
    pitch axis from its start direction, and body-rate magnitude;
  * the off-axis table: the body rates integrated over the flip (until the primary detector's
    flip time + 0.3 s), the largest pitch-axis deviation in that window and the heading at the
    end of the episode, together with the flip and recovery times of the primary detector.
A pure pitch flip has a zero roll and yaw integral, zero axis deviation and an unchanged heading.

    python Evaluation/flip_analysis.py --name flips_nominal
    python Evaluation/flip_analysis.py --name flips_learned --controllers bc_ppo ppo_mtr ppo_mtr_motors
Output: Evaluation/runs/figures/<name>.png and .pdf, <name>_offaxis.txt and <name>_offaxis.tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (HERE, REPO / "MTR_PPO"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

import evaluate as harness                              # noqa: E402
from final_comparison import CONDITIONS, CONTROLLERS, make_controller   # noqa: E402
from plot_comparison import INK, GRID, LABELS, MUTED    # noqa: E402

DEFAULT = ["scripted_flip", "three_phase_default", "map_elites_final", "bc_ppo", "ppo_mtr", "ppo_mtr_motors"]
# reference categorical palette (light), fixed order -- never cycled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
WINDOW_AFTER_FLIP = 0.3      # s, end of the off-axis window after the primary detector's flip time


def rot(q):
    """Rotation matrix body -> world of the simulator's quaternion [w, x, y, z]."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def heading_deg(q):
    w, x, y, z = q
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def analyse(name, seed, seconds):
    r = harness.evaluate(make_controller(name), [seed], episode_seconds=seconds,
                         condition=CONDITIONS["nominal"], record=True)[0]
    tr = r.trajectory
    t, q, w = tr["t"], tr["quat"], tr["omega"]
    y0 = rot(q[0])[:, 1]
    dev = np.degrees(np.arccos(np.clip([rot(qq)[:, 1] @ y0 for qq in q], -1.0, 1.0)))
    t_flip = r.ppo_track["t_flip_s"]
    m = t <= (t_flip if t_flip is not None else t[-1]) + WINDOW_AFTER_FLIP
    integ = [float(np.degrees(np.trapz(w[m, k], t[m]))) for k in range(3)]
    return dict(name=name, t=t, rev=np.degrees(tr["rev"]), dz=-(tr["pos"][:, 2] - tr["pos"][0, 2]),
                dev=dev, wnorm=np.linalg.norm(w, axis=1), success=bool(r.ppo_track["success"]),
                t_flip=t_flip, t_rec=r.ppo_track["t_recovered_s"], int_wx=integ[0], int_wy=integ[1],
                int_wz=integ[2], max_dev=float(dev[m].max()), heading_end=heading_deg(q[-1]) - heading_deg(q[0]),
                alt_loss=float(r.max_alt_loss_m))


def write_table(rows, out):
    head = (f"{'controller':<24}{'success':>8}{'t_flip':>8}{'t_rec':>8}{'int wx':>8}{'int wy':>8}"
            f"{'int wz':>8}{'max axis dev':>14}{'heading end':>13}")
    lines = [head, "-" * len(head)]
    tex = ["% controller & t_flip & t_rec & int wx & int wy & int wz & max axis dev & heading end \\\\"]
    f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
    for d in rows:
        lines.append(f"{LABELS.get(d['name'], d['name']):<24}{str(d['success']):>8}{f(d['t_flip']):>8}{f(d['t_rec']):>8}"
                     f"{d['int_wx']:8.0f}{d['int_wy']:8.0f}{d['int_wz']:8.0f}{d['max_dev']:14.1f}{d['heading_end']:13.1f}")
        tex.append(f"{LABELS.get(d['name'], d['name'])} & {f(d['t_flip'])} & {f(d['t_rec'])} & {d['int_wx']:.0f} & "
                   f"{d['int_wy']:.0f} & {d['int_wz']:.0f} & {d['max_dev']:.1f} & {d['heading_end']:.1f} \\\\")
    note = (f"(nominal episode; integrals in degrees over [0, t_flip + {WINDOW_AFTER_FLIP} s]; axis deviation = angle "
            "of the body y-axis from its start direction in that window; heading = yaw change at the end of the episode; "
            "t_flip, t_rec and success by the primary detector)")
    out.with_name(out.name + "_offaxis.txt").write_text("\n".join(lines + [note]) + "\n", encoding="utf-8")
    out.with_name(out.name + "_offaxis.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(lines + [note]))


def figure(rows, out, t_max):
    panels = [("rev", "rotation [deg]"), ("dz", "altitude change [m]"),
              ("dev", "pitch-axis deviation [deg]"), ("wnorm", r"body rate $\|\omega\|$ [rad/s]")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.0, 8.2), sharex=True)
    for i, d in enumerate(rows):
        c = SERIES[i]
        m = d["t"] <= t_max
        for ax, (k, _) in zip(axes, panels):
            ax.plot(d["t"][m], d[k][m], color=c, lw=1.6, label=LABELS.get(d["name"], d["name"]))
    for ax, (k, lab) in zip(axes, panels):
        ax.set_ylabel(lab, fontsize=8.5, color=INK)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(labelsize=8, colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
    axes[0].axhline(360, color=MUTED, lw=0.8, ls="--")
    axes[-1].set_xlabel("time [s]", fontsize=8.5, color=INK)
    axes[-1].set_xlim(0, t_max)
    axes[0].legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.42), frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.parent / f"{out.name}.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="flips_nominal")
    ap.add_argument("--controllers", nargs="+", default=DEFAULT, choices=CONTROLLERS)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--seconds", type=float, default=10.0, help="episode length (the benchmark's 10 s)")
    ap.add_argument("--t-max", type=float, default=3.0, help="time range of the figure [s]")
    a = ap.parse_args()
    if len(a.controllers) > len(SERIES):
        ap.error(f"at most {len(SERIES)} controllers in one figure")
    rows = []
    for n in a.controllers:
        print("running", n, flush=True)
        rows.append(analyse(n, a.seed, a.seconds))
    out_dir = HERE / "runs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / a.name
    write_table(rows, out)
    figure(rows, out, a.t_max)
    print("written to", out_dir)


if __name__ == "__main__":
    main()
