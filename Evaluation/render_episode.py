"""Render one harness episode as a side-view animation (GIF) for the report's demo videos.

    python Evaluation/render_episode.py --controller map_elites_final --condition nominal --seed 0
    python Evaluation/render_episode.py --controller map_elites_final --condition stress --seed 0 --seconds 6

Side view of the pitch plane: world x (forward) horizontally, height (= -z, NED) vertically.
The body is drawn as its x-axis arm (front rotor marked) with the thrust direction, plus the
trail of past positions. The title shows time, flip progress and, at the end, the §04 verdict.
The episode itself is the normal harness episode (same env, seed, condition and detector).
Output: Evaluation/runs/videos/<controller>_<condition>_s<seed>.gif (and a PNG still with --still).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import evaluate as harness                                  # noqa: E402
from final_comparison import CONDITIONS, CONTROLLERS, make_controller  # noqa: E402

SURFACE, INK, INK2, GRID, BODY, THRUST = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de", "#2a78d6", "#eb6834"
ARM = 0.35            # drawn arm half-length [m] (the real 0.16 m is enlarged for visibility)


def body_axes(quat):
    """World-frame body x (forward) and body z (down) axes from a scalar-first quaternion."""
    w, x, y, z = np.asarray(quat, float) / np.linalg.norm(quat)
    bx = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    bz = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return bx, bz


def side(v):
    """World (NED) vector -> side-view plane (x forward, height up)."""
    return np.array([v[0], -v[2]])


def frame_geometry(pos, quat):
    bx, bz = body_axes(quat)
    c = side(pos)
    front, back = c + ARM * side(bx), c - ARM * side(bx)
    thrust_tip = c + 0.45 * side(-bz)          # thrust acts along -body z
    return c, front, back, thrust_tip


def render(result, path, fps: int = 25, still: bool = False, title: str = ""):
    tr = result.trajectory
    t, pos, quat, rev = tr["t"], tr["pos"], tr["quat"], tr["rev"]
    stride = max(1, int(round(1.0 / (fps * float(t[1] - t[0])))))
    idx = np.arange(0, len(t), stride)
    xy = np.array([side(p) for p in pos])
    pad = 0.8
    ylim = (min(xy[:, 1].min(), 0.0) - pad, xy[:, 1].max() + pad)
    half = max((xy[:, 0].max() - xy[:, 0].min()) / 2 + pad, (ylim[1] - ylim[0]) / 2)
    cx = (xy[:, 0].max() + xy[:, 0].min()) / 2
    xlim = (cx - half, cx + half)                   # square view: equal aspect without empty margins
    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.8)
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel("x forward [m]", color=INK2, fontsize=9)
    ax.set_ylabel("height above start [m]", color=INK2, fontsize=9)
    ax.axhline(0.0, color=INK2, lw=0.8, ls=":")
    trail, = ax.plot([], [], color=INK2, lw=1.0, alpha=0.6)
    arm, = ax.plot([], [], color=BODY, lw=4, solid_capstyle="round")
    front_dot, = ax.plot([], [], "o", color=BODY, ms=8, mec=SURFACE, mew=1.5)
    thrust, = ax.plot([], [], color=THRUST, lw=2)
    head = ax.set_title("", color=INK, fontsize=10, loc="left")
    verdict = "SUCCESS" if result.success else "FAIL"

    def draw(k):
        i = idx[k]
        c, f, b, tt = frame_geometry(pos[i], quat[i])
        trail.set_data(xy[: i + 1, 0], xy[: i + 1, 1])
        arm.set_data([b[0], f[0]], [b[1], f[1]])
        front_dot.set_data([f[0]], [f[1]])
        thrust.set_data([c[0], tt[0]], [c[1], tt[1]])
        end = f"   §04: {verdict}" if k == len(idx) - 1 else ""
        head.set_text(f"{title}\nt = {t[i]:4.2f} s   flip {np.degrees(rev[i]):5.0f}°{end}")
        return trail, arm, front_dot, thrust, head

    if still:
        mid = int(np.argmin(np.abs(np.degrees(rev[idx]) - 180.0)))
        draw(mid)
        fig.savefig(Path(path).with_suffix(".png"), facecolor=SURFACE)
    anim = FuncAnimation(fig, draw, frames=len(idx), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return len(idx)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", default="map_elites_final", choices=CONTROLLERS)
    ap.add_argument("--condition", default="nominal", choices=list(CONDITIONS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=4.0, help="episode length to simulate and show")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--still", action="store_true", help="also save a PNG of the inverted moment")
    a = ap.parse_args(argv)
    if a.seconds <= 0 or a.fps < 1:
        ap.error("--seconds must be > 0 and --fps >= 1")
    result = harness.run_episode(make_controller(a.controller), a.seed, record=True,
                                 episode_seconds=a.seconds, condition=CONDITIONS[a.condition])
    out = HERE / "runs" / "videos"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.controller}_{a.condition}_s{a.seed}.gif"
    n = render(result, path, a.fps, a.still, title=f"{a.controller} · {a.condition} · seed {a.seed}")
    print(f"{n} frames, verdict: {result.reason} -> {path}")


if __name__ == "__main__":
    main()
