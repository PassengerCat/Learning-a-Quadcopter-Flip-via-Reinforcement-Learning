"""
visualize_flip.py — report figures and an animation of the flip (scripted §02 vs PPO §03).

Both controllers are run on the SAME seed through the same env stack as the
evaluation (make_flip_env(...).env, action-repeat 4), and the vehicle is drawn in
a side view (world x vs altitude): the flip is a rotation about body-y, so it
happens in exactly this plane.

Outputs (in --out):
  flip_filmstrip.png   "chronophotography": the quad drawn every --every seconds,
                       light -> dark = early -> late, one panel per controller.
  flip_phases.png      flip angle, altitude and the commands each controller sends,
                       with the scripted controller's phases shaded.
  flip_animation.gif   side-by-side animation (slow motion), with a live flip-angle trace.

Usage:
  python visualize_flip.py --model models/ppo_flip.zip
  python visualize_flip.py --model models/ppo_flip.zip --seed 10003 --slowmo 3 --out results/viz
  python visualize_flip.py                      # scripted flip only (no model)
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgb

from run_baselines import make_raw_env, check_simulator
from baselines import make_controller, quat_to_rotmat, QuadModel, control_dt_from_env
from flip_reward import gravity_pitch_increment

# ---- style (same identities as run_baselines.plot) -------------------------- #
COLORS = {"scripted_flip": "#574fae", "ppo_flip": "#c2410c"}
LABELS = {"scripted_flip": "Scripted flip (§02)", "ppo_flip": "PPO policy (§03)"}
INK, INK_2, MUTED, GRID = "#151f29", "#54636f", "#8593a0", "#e3e7ec"
PHASE_NAMES = ["settle", "climb", "pulse", "coast", "brake", "recover"]
PHASE_TINT = ["#eef0f3", "#e3eef7", "#f7e7df", "#f3eee0", "#efe3f1", "#e2f1ec"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


# --------------------------------------------------------------------------- #
# Rollout with everything we want to draw                                      #
# --------------------------------------------------------------------------- #
def run(env, ctrl, seed: int, seconds: float, action_noise: float = 0.0) -> Dict[str, np.ndarray]:
    dt = control_dt_from_env(env)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    ctrl.reset(seed=seed)
    base = env.unwrapped
    rec: Dict[str, List] = {k: [] for k in ("t", "pos", "R", "omega", "motors", "motor_action",
                                            "phase", "policy_action")}

    def log(t, a=None, pa=None, ph=-1):
        q = base.quad
        rec["t"].append(t); rec["pos"].append(np.array(q.pos)); rec["R"].append(quat_to_rotmat(q.quat))
        rec["omega"].append(np.array(q.omega)); rec["motors"].append(np.array(q.wMotor))
        rec["motor_action"].append(np.zeros(4) if a is None else a)
        rec["policy_action"].append(np.full(4, np.nan) if pa is None else pa)
        rec["phase"].append(ph)

    log(0.0)
    n = int(round(seconds / dt))
    for i in range(n):
        a = np.asarray(ctrl.act(obs), np.float32)
        if action_noise > 0:
            a = np.clip(a + rng.normal(0, action_noise, 4), -1, 1).astype(np.float32)
        d = ctrl.diagnostics
        pa = getattr(getattr(ctrl, "builder", None), "last_action", None)   # PPO: its own CTBR command
        obs, _, te, tr, _ = env.step(a)
        log((i + 1) * dt, a, None if pa is None else np.array(pa, float), d.get("phase_id", -1))
        if te or tr:
            break
    out = {k: np.asarray(v) for k, v in rec.items()}
    # geometric flip angle (same measure as the reward and the §04 detector)
    th = [0.0]
    for k in range(1, len(out["R"])):
        th.append(th[-1] + gravity_pitch_increment(_r2q(out["R"][k - 1]), _r2q(out["R"][k])))
    out["theta"] = np.degrees(np.array(th))
    out["tilt"] = np.degrees(np.arccos(np.clip(out["R"][:, 2, 2], -1, 1)))
    out["dt"] = dt
    return out


def _r2q(R):
    """Rotation matrix -> scalar-first quaternion (robust branch)."""
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1.0)
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.empty(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def rotation_times(tr):
    """Times at which the geometric flip angle first passes 10 deg and 350 deg."""
    th, t = tr["theta"], tr["t"]
    a = np.flatnonzero(th >= 10); b = np.flatnonzero(th >= 350)
    return (t[a[0]] if a.size else np.nan), (t[b[0]] if b.size else np.nan)


def settle_time(tr):
    """First time after the inversion at which the vehicle stays upright (<10 deg) and still (<1 rad/s)."""
    tilt, t, w = tr["tilt"], tr["t"], np.linalg.norm(tr["omega"], axis=1)
    inv = np.flatnonzero(tilt > 150)
    if not inv.size:
        return np.nan
    ok = (tilt < 10) & (w < 1.0)
    for k in range(inv[-1], len(t)):
        if ok[k:].all() or (k + 10 < len(t) and ok[k:k + 10].all()):
            return t[k]
    return np.nan


def flip_window(tr, pad: float = 0.15):
    """From the first noticeable rotation to settling upright after the inversion."""
    th, t, tilt = np.abs(tr["theta"]), tr["t"], tr["tilt"]
    start = np.flatnonzero(th > 8)
    if not start.size:
        return 0.0, min(t[-1], 2.0)
    i0 = start[0]
    inv = np.flatnonzero(tilt > 150)
    j = inv[-1] if inv.size else i0
    after = np.flatnonzero((tilt[j:] < 10) & (np.linalg.norm(tr["omega"][j:], axis=1) < 1.5))
    i1 = j + (after[0] if after.size else len(t) - 1 - j)
    return max(0.0, t[i0] - pad), min(t[-1], t[i1] + pad)


# --------------------------------------------------------------------------- #
# Drawing the vehicle (side view: x horizontal, altitude = -z vertical)        #
# --------------------------------------------------------------------------- #
def quad_segments(pos, R, size: float, thrust_ratio: float):
    """Return 2-D geometry for one quad pose. size = drawn half-arm length [m]."""
    c = np.array([pos[0], -pos[2]])
    bx = np.array([R[0, 0], -R[2, 0]])            # body x (nose) projected
    up = np.array([-R[0, 2], R[2, 2]])            # thrust direction (-body z) projected
    front, rear = c + size * bx, c - size * bx
    rot = 0.30 * size
    rotor_f = (front + 0.18 * size * up - rot * bx, front + 0.18 * size * up + rot * bx)
    rotor_r = (rear + 0.18 * size * up - rot * bx, rear + 0.18 * size * up + rot * bx)
    arrow = (c, c + up * size * 1.2 * thrust_ratio)
    return c, front, rear, rotor_f, rotor_r, arrow


def draw_quad(ax, pos, R, size, thrust_ratio, color, alpha=1.0, lw=2.0, arrow=True, z=3):
    c, f, r, rf, rr, (a0, a1) = quad_segments(pos, R, size, thrust_ratio)
    arts = []
    arts += ax.plot([r[0], f[0]], [r[1], f[1]], color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=z)
    arts += ax.plot([rr[0][0], rr[1][0]], [rr[0][1], rr[1][1]], color=color, lw=lw * 1.4, alpha=alpha,
                    solid_capstyle="round", zorder=z)
    arts += ax.plot([rf[0][0], rf[1][0]], [rf[0][1], rf[1][1]], color=color, lw=lw * 1.4, alpha=alpha,
                    solid_capstyle="round", zorder=z)
    arts += ax.plot([f[0]], [f[1]], marker="o", ms=4.5, color=INK, alpha=alpha, zorder=z + 1,
                    markeredgecolor="white", markeredgewidth=0.6)   # the NOSE
    arts += ax.plot([c[0]], [c[1]], "o", ms=3.5, color=color, alpha=alpha, zorder=z + 1)
    if arrow and thrust_ratio > 0.05:
        arts.append(ax.annotate("", xy=a1, xytext=a0, zorder=z,
                                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, alpha=0.8 * alpha,
                                                mutation_scale=9)))
    return arts


def thrust_ratio(tr, k, model: QuadModel):
    return float(np.sum(model.kTh * tr["motors"][k] ** 2) / (model.mass * model.g))


def fade(color, a):
    """Light -> full color (one hue, sequential in lightness)."""
    c = np.array(to_rgb(color))
    return tuple(1 - (1 - c) * a)


# --------------------------------------------------------------------------- #
# Figure 1: filmstrip                                                          #
# --------------------------------------------------------------------------- #
def filmstrip(trajs, model, out, every: float, size: float):
    """One row per controller, each around ITS OWN flip. Poses are laid out left ->
    right in time like film frames (x = spread * t) at their true altitude; equal
    aspect, so every drawn attitude is the true angle."""
    names = list(trajs)
    spread = 2.7 * size / every                  # horizontal metres per second: poses don't touch
    wins = {}
    for n in names:
        tr = trajs[n]
        r0, r1 = rotation_times(tr)
        ts = settle_time(tr)
        t0 = (r0 if np.isfinite(r0) else 0.0) - 3 * every
        t1 = (ts if np.isfinite(ts) else r1 + 0.3) + 2 * every
        wins[n] = (max(0.0, t0), min(tr["t"][-1], t1), r0, r1, ts)
    width = max(w[1] - w[0] for w in wins.values()) * spread
    alt = np.concatenate([-trajs[n]["pos"][:, 2] for n in names])
    y_lo, y_hi = alt.min() - 1.8 * size, alt.max() + 1.8 * size
    fig_w = 13.0
    row_h = fig_w * (y_hi - y_lo + 1.2) / (width + 4 * size) + 0.9
    fig, axs = plt.subplots(len(names), 1, figsize=(fig_w, row_h * len(names) + 0.8), squeeze=False)
    axs = axs[:, 0]
    for ax, n in zip(axs, names):
        tr, col = trajs[n], COLORS[n]
        t0, t1, r0, r1, ts = wins[n]
        m = (tr["t"] >= t0 - 1e-9) & (tr["t"] <= t1 + 1e-9)
        ax.plot(spread * tr["t"][m], -tr["pos"][m, 2], color=GRID, lw=5, solid_capstyle="round", zorder=1)
        step = max(1, int(round(every / tr["dt"])))
        for k in list(np.flatnonzero(m))[::step]:
            pp = tr["pos"][k].copy()
            pp[0] = spread * tr["t"][k]
            draw_quad(ax, pp, tr["R"][k], size, thrust_ratio(tr, k, model), col, lw=1.8, arrow=True, z=3)
        k_inv = int(np.argmax(np.where(m, tr["tilt"], -1)))
        events = [(r0, "rotation starts (10°)"), (tr["t"][k_inv], f"most inverted ({tr['tilt'][k_inv]:.0f}°)"),
                  (r1, "360° reached"), (ts, "upright & still")]
        for i, (tt, lab) in enumerate(events):
            if np.isfinite(tt):
                ax.axvline(spread * tt, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=0)
                ax.text(spread * tt, y_hi + (0.05 if i % 2 == 0 else 0.55), f"{lab}  {tt:.2f} s",
                        fontsize=8, color=INK_2, ha="center", va="bottom",
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        climb = np.max(-tr["pos"][:, 2]) - (-tr["pos"][0, 2])
        qmax = np.max(np.abs(tr["omega"][:, 1]))
        ax.set_title(f"{LABELS[n]} — 10°→350° in {r1 - r0:.2f} s · peak pitch rate {qmax:.0f} rad/s · "
                     f"climb {climb:.1f} m", loc="left", fontsize=10, color=col, pad=26)
        ax.set_xlim(spread * t0 - 1.5 * size, spread * t0 + width + 1.5 * size)
        ax.set_ylim(y_lo, y_hi + 1.0)
        ax.set_aspect("equal")
        ticks = np.arange(np.ceil(t0 * 10) / 10, t1 + 1e-9, 0.1)
        ax.set_xticks(spread * ticks); ax.set_xticklabels([f"{x:.1f}" for x in ticks])
        ax.set_ylabel("altitude [m]"); ax.set_xlabel("time [s]")
        ax.grid(color=GRID, lw=0.6, axis="y")
        ax.spines["left"].set_bounds(y_lo, y_hi)
        ax.set_yticks([v for v in ax.get_yticks() if y_lo <= v <= y_hi])
    fig.suptitle("The flip, frame by frame (side view, nose to the right at the start)",
                 fontsize=12, fontweight="semibold", color=INK)
    fig.text(0.5, 0.005, f"One pose every {every:.2f} s, drawn at its time (x) and true altitude (y). "
             f"Dark dot = nose; arrow = total thrust (length ∝ thrust/weight). "
             f"Vehicle drawn ×{size / 0.16:.1f} for visibility; attitudes are exact.",
             ha="center", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    p = os.path.join(out, "flip_filmstrip.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Figure 2: phases and commands                                                #
# --------------------------------------------------------------------------- #
def phases_figure(trajs, model, out, rate_max: Optional[float]):
    names = list(trajs)
    t_end = max(flip_window(trajs[n])[1] for n in names) + 0.4
    fig, axs = plt.subplots(4, 1, figsize=(8.6, 9.2), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.3, 1, 1, 1]))
    sc = trajs.get("scripted_flip")
    if sc is not None:      # phase bands of the scripted controller, on every panel
        ph, t = sc["phase"], sc["t"]
        for a in axs:
            s = 1
            for k in range(2, len(ph) + 1):
                if k == len(ph) or ph[k] != ph[s]:
                    if ph[s] >= 0 and t[s - 1] < t_end:
                        a.axvspan(t[s - 1], min(t[k - 1], t_end), color=PHASE_TINT[ph[s]], lw=0, zorder=0)
                    s = k
        j = 0
        for k in range(1, len(ph)):
            if ph[k] != ph[k - 1] or k == 1:
                if 0 <= ph[k] and t[k] < t_end - 0.1:
                    axs[0].text(t[k - 1] + 0.01, 468 - 30 * (j % 2), PHASE_NAMES[ph[k]], fontsize=7.5,
                                color=INK_2, va="top")
                    j += 1
    for n in names:
        tr, col = trajs[n], COLORS[n]
        axs[0].plot(tr["t"], tr["theta"], color=col, lw=2, label=LABELS[n])
        axs[1].plot(tr["t"], -tr["pos"][:, 2], color=col, lw=2)
        axs[2].plot(tr["t"], tr["omega"][:, 1], color=col, lw=2)
        T = np.sum(model.kTh * tr["motors"] ** 2, axis=1) / (model.mass * model.g)
        axs[3].plot(tr["t"], T, color=col, lw=2)
        pa = tr["policy_action"]
        if n == "ppo_flip" and rate_max and np.isfinite(pa).any():
            axs[2].plot(tr["t"], pa[:, 2] * rate_max, color=col, lw=1.2, ls=(0, (3, 2)))
            axs[2].text(tr["t"][np.nanargmax(pa[:, 2])], rate_max * 1.02, "PPO command", color=INK_2, fontsize=8)
    axs[0].axhline(180, color=MUTED, lw=0.8, ls=":"); axs[0].axhline(360, color=MUTED, lw=0.8, ls=":")
    axs[0].text(0.02, 186, "inverted (180°)", fontsize=8, color=INK_2)
    axs[0].text(0.02, 335, "full flip (360°)", fontsize=8, color=INK_2, va="top")
    axs[3].axhline(1, color=MUTED, lw=0.8, ls=":")
    axs[3].text(1.0, 1.07, "hover thrust", fontsize=8, color=INK_2)
    axs[0].set_ylabel("flip angle θ [°]"); axs[1].set_ylabel("altitude [m]")
    axs[2].set_ylabel("pitch rate q [rad/s]"); axs[3].set_ylabel("thrust / weight")
    axs[0].set_ylim(-40, 475)
    axs[0].set_yticks([0, 90, 180, 270, 360])
    axs[3].set_xlabel("time [s]"); axs[3].set_xlim(0, t_end)
    axs[1].legend(loc="lower right")
    axs[2].text(0.99, 0.95, "solid: measured · dashed: PPO's commanded rate", transform=axs[2].transAxes,
                ha="right", va="top", fontsize=8, color=INK_2)
    for a in axs:
        a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    axs[0].set_title("What each controller does, and when (shaded: scripted-flip phases)")
    fig.tight_layout()
    p = os.path.join(out, "flip_phases.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Animation                                                                    #
# --------------------------------------------------------------------------- #
def animation(trajs, model, out, size, seconds, slowmo, fps=25):
    names = list(trajs)
    dt = trajs[names[0]]["dt"]
    n_frames = int(min(seconds, min(tr["t"][-1] for tr in trajs.values())) / dt)
    step = max(1, int(round((1.0 / fps) / (dt * slowmo))))   # sim samples per frame
    frames = list(range(0, n_frames, step))
    fig = plt.figure(figsize=(9.5, 6.2))
    gs = fig.add_gridspec(2, len(names), height_ratios=[3, 1.1])
    axq = [fig.add_subplot(gs[0, i]) for i in range(len(names))]
    axt = fig.add_subplot(gs[1, :])
    xs = np.concatenate([tr["pos"][:n_frames, 0] for tr in trajs.values()])
    ys = np.concatenate([-tr["pos"][:n_frames, 2] for tr in trajs.values()])
    span = max(np.ptp(xs), np.ptp(ys)) + 5 * size
    xm, ym = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    for ax, n in zip(axq, names):
        ax.set_xlim(xm - span / 2, xm + span / 2); ax.set_ylim(ym - span / 2, ym + span / 2)
        ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.6)
        ax.set_title(LABELS[n], color=COLORS[n]); ax.set_xlabel("x [m]")
    axq[0].set_ylabel("altitude [m]")
    for n in names:
        tr = trajs[n]
        axt.plot(tr["t"][:n_frames], tr["theta"][:n_frames], color=COLORS[n], lw=2, label=LABELS[n])
    axt.axhline(360, color=MUTED, lw=0.8, ls=":"); axt.axhline(180, color=MUTED, lw=0.8, ls=":")
    axt.set_ylabel("flip angle [°]"); axt.set_xlabel("time [s]"); axt.grid(color=GRID, lw=0.6)
    axt.set_ylim(-40, 420); axt.legend(loc="lower right", fontsize=8)
    cursor = axt.axvline(0, color=INK, lw=1)
    txt = fig.text(0.5, 0.965, "", ha="center", fontsize=11, color=INK)
    dyn = []

    def update(fi):
        for a in dyn:
            a.remove()
        dyn.clear()
        k = frames[fi]
        for ax, n in zip(axq, names):
            tr = trajs[n]
            kk = min(k, len(tr["t"]) - 1)
            dyn.extend(ax.plot(tr["pos"][:kk + 1, 0], -tr["pos"][:kk + 1, 2], color=fade(COLORS[n], 0.35), lw=1.5))
            dyn.extend(draw_quad(ax, tr["pos"][kk], tr["R"][kk], size, thrust_ratio(tr, kk, model), COLORS[n], lw=2.4))
            dyn.append(ax.text(0.03, 0.95, f"θ = {tr['theta'][kk]:6.0f}°\ntilt = {tr['tilt'][kk]:5.0f}°",
                               transform=ax.transAxes, va="top", fontsize=9, color=INK_2, family="monospace"))
        t = trajs[names[0]]["t"][min(k, len(trajs[names[0]]["t"]) - 1)]
        cursor.set_xdata([t, t])
        txt.set_text(f"t = {t:4.2f} s   (slow motion ×{slowmo:g})")
        return dyn

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out, "flip_animation.gif")
    anim.save(p, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="trained policy (.zip); omit to show the scripted flip only")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--action-noise", type=float, default=0.0)
    ap.add_argument("--every", type=float, default=0.04, help="filmstrip pose spacing [s]")
    ap.add_argument("--size", type=float, default=0.25, help="drawn half-arm length [m] (real: 0.16)")
    ap.add_argument("--slowmo", type=float, default=2.0)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--out", default="results/viz")
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    os.makedirs(a.out, exist_ok=True)

    env = make_raw_env(action_repeat=4)
    check_simulator(env)
    model = QuadModel.from_env(env)
    trajs, rate_max = {}, None
    trajs["scripted_flip"] = run(env, make_controller("scripted_flip", env), a.seed, a.seconds, a.action_noise)
    if a.model:
        from flip_policy import LearnedFlipController
        ctrl = LearnedFlipController(a.model, a.config)
        rate_max = (ctrl.rc.get("ctbr") or {}).get("rate_max_rp")
        trajs["ppo_flip"] = run(env, ctrl, a.seed, a.seconds, a.action_noise)
    env.close()

    print("->", filmstrip(trajs, model, a.out, a.every, a.size))
    print("->", phases_figure(trajs, model, a.out, rate_max))
    if not a.no_gif:
        print("->", animation(trajs, model, a.out, 1.5 * a.size, a.seconds, a.slowmo))


if __name__ == "__main__":
    main()
