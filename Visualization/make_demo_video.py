"""
make_demo_video.py — the §06 demo video: representative SUCCESSES and FAILURES.

Every clip is a real rollout through the SAME harness as the evaluation
(run_baselines.rollout conventions: same seeds, same controller-side noise
injection, the env's own wind model), and the verdict on screen is the live
state of the independent §04 SuccessDetector, updated from the TRUE simulator
state, not from the reward.

Clips (in order):
  1. PPO policy, nominal                             -> success
  2. Scripted flip (§02), nominal                    -> success
  3. PPO policy under stress (actuator σ .15, gyro σ .2)  -> success
  4. Scripted flip under the same stress             -> FAILURE (never settles)
  5. Run-4 policy: reward hacking by coning          -> FAILURE (no real flip)
  6. PPO policy in 5 m/s wind                        -> success, but drifts
  7. PPO + PID hold in 5 m/s wind                    -> success, holds position
  (+ an optional motor-level policy clip with --motor-model)

For clips that must show a failure (or a success), the script scans seeds
starting at --seed0 until the detector's verdict matches, so the video always
shows what the caption says; the seed is printed on screen.

    python make_demo_video.py                         # -> results/video/flip_demo.mp4
    python make_demo_video.py --only 1 4 5 --out tmp  # a subset
    python make_demo_video.py --split                 # also one .mp4 per clip

Needs:  pip install imageio imageio-ffmpeg   (a bundled ffmpeg, works on Windows)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

from run_baselines import make_raw_env, check_simulator            # noqa: E402
from baselines import (QuadModel, make_controller, quat_to_rotmat,  # noqa: E402
                       control_dt_from_env)
from flip_reward import SuccessDetector, SuccessConfig, gravity_pitch_increment  # noqa: E402

W, H, DPI, FPS = 1280, 720, 100, 30
INK, INK_2, MUTED, GRID = "#151f29", "#54636f", "#8593a0", "#e3e7ec"
OK, BAD, PENDING = "#1f7a4d", "#b42318", "#8593a0"
BODY, NOSE = "#1f2a36", "#c2410c"
TRAIL = "#574fae"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


# --------------------------------------------------------------------------- #
# Clip definitions                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class Clip:
    title: str
    subtitle: str
    make_ctrl: Callable            # env -> controller
    expect: Optional[bool]         # True / False: scan seeds until the verdict matches; None: first seed
    action_noise: float = 0.0
    obs_noise: float = 0.0
    wind: float = 0.0
    seconds: float = 6.0
    show_q_int: bool = False       # also plot the integrated pitch rate (coning clip)
    takeaway: str = ""
    result: Dict = field(default_factory=dict)


def build_clips(a) -> List[Clip]:
    from flip_policy import LearnedFlipController
    ppo = lambda env: LearnedFlipController(a.model)
    hyb = lambda env: LearnedFlipController(a.model, pid_handoff=True)
    scr = lambda env: make_controller("scripted_flip", env)
    stress = dict(action_noise=0.15, obs_noise=0.2)
    clips = [
        Clip("PPO policy — nominal", "no wind, no noise · the conditions of the primary benchmark",
             ppo, True, seconds=4.0,
             takeaway="Full 360° through the inverted attitude, then upright and still."),
        Clip("Scripted flip (§02 baseline) — nominal", "hand-tuned open-loop pulse + PID recovery",
             scr, True, seconds=4.0,
             takeaway="The rule-based baseline also succeeds when nothing disturbs it."),
        Clip("PPO policy — stress", "actuator noise σ = 0.15 · gyro noise σ = 0.2 rad/s",
             ppo, True, seconds=5.0, **stress,
             takeaway="The learned policy still flips and settles (100% in evaluation)."),
        Clip("Scripted flip — same stress", "actuator noise σ = 0.15 · gyro noise σ = 0.2 rad/s",
             scr, False, seconds=6.0, **stress,
             takeaway="FAILURE: the flip happens, but the PID amplifies gyro noise and never "
                      "gets ‖ω‖ below 0.8 rad/s (20% success)."),
    ]
    run4 = os.path.join(os.path.dirname(os.path.abspath(a.model)), "ablation", "run4_coning.zip")
    if os.path.exists(run4):
        clips.append(Clip(
            "Reward hacking — run-4 policy (coning)",
            "reward on INTEGRATED pitch rate: the policy tilts and yaws instead of flipping",
            lambda env: LearnedFlipController(run4), False, seconds=4.0, show_q_int=True,
            takeaway="FAILURE: ∫q dt grows far past 360° while the geometric flip angle stays "
                     "small and the vehicle never inverts. Fixed by the geometric measure."))
    clips += [
        Clip("PPO policy — 5 m/s wind", "the env's RANDOMSINE wind · the policy never saw wind in training",
             ppo, True, seconds=8.0, wind=5.0,
             takeaway="Success, but a memoryless policy drifts downwind (≈ 1.9 m mean)."),
        Clip("PPO flip + PID position hold — 5 m/s wind", "hybrid: the policy flips, the §02 PID holds position",
             hyb, True, seconds=8.0, wind=5.0,
             takeaway="Success, and the PID's integral action cancels the drift (≈ 0.04 m)."),
    ]
    if a.motor_model:
        clips.insert(1, Clip("Motor-level PPO policy — nominal", "a_t = [m1, m2, m3, m4]: the policy "
                             "commands the four motors directly", lambda env: LearnedFlipController(a.motor_model),
                             True, seconds=4.0,
                             takeaway="Same manoeuvre with the policy acting directly on the motors."))
    return clips


# --------------------------------------------------------------------------- #
# Recording (same conventions as run_baselines.rollout)                        #
# --------------------------------------------------------------------------- #
def record(env, ctrl, seed: int, clip: Clip) -> Dict:
    dt = control_dt_from_env(env)
    rng = np.random.default_rng(np.random.SeedSequence([12345, int(seed)]))
    obs, _ = env.reset(seed=int(seed))
    if clip.wind > 0:
        random.seed(int(seed))
        env.unwrapped.enable_random_wind(True, magnitude=clip.wind, heading_deg=180.0, elevation_deg=10.0)
    ctrl.reset(seed=int(seed))
    base = env.unwrapped
    det = SuccessDetector(SuccessConfig())
    det.reset()
    det.update(None, env=base)
    rec: Dict[str, list] = {k: [] for k in ("t", "pos", "quat", "omega", "vel", "motors", "rev",
                                            "inv", "hold", "succ", "alt_ok", "qint")}
    qint = 0.0
    z0 = [float(base.quad.pos[2])]

    def log(t):
        q = base.quad
        rec["t"].append(t); rec["pos"].append(np.array(q.pos, float)); rec["quat"].append(np.array(q.quat, float))
        rec["omega"].append(np.array(q.omega, float)); rec["vel"].append(np.array(q.vel, float))
        rec["motors"].append(np.array(q.wMotor, float))
        rec["rev"].append(det.revolution); rec["inv"].append(det.passed_inverted)
        # hold_count / altitude_ok exist in the current flip_reward.py; with an older
        # copy fall back to the private counter and to the same 1 m rule computed here
        rec["hold"].append(getattr(det, "hold_count", getattr(det, "_hold", 0)))
        rec["succ"].append(det.succeeded)
        alt_ok = getattr(det, "altitude_ok", None)
        if alt_ok is None:
            z = float(q.pos[2])
            alt_ok = (not rec["alt_ok"] or rec["alt_ok"][-1]) and z - z0[0] <= 1.0
        rec["alt_ok"].append(bool(alt_ok)); rec["qint"].append(qint)

    log(0.0)
    terminated = False
    n = int(round(clip.seconds / dt))
    for i in range(n):
        o = np.asarray(obs, dtype=np.float64)
        if clip.obs_noise > 0:
            o = o.copy(); o[13:16] += rng.normal(0, clip.obs_noise, 3)
        act = np.asarray(ctrl.act(o), dtype=np.float32)
        if clip.action_noise > 0:
            act = np.clip(act + rng.normal(0, clip.action_noise, 4), -1, 1).astype(np.float32)
        w_prev = float(base.quad.omega[1])
        obs, _, terminated, truncated, _ = env.step(act)
        qint += 0.5 * (w_prev + float(base.quad.omega[1])) * dt
        det.update(None, env=base)
        log((i + 1) * dt)
        if terminated or truncated:
            break
    tr = {k: np.asarray(v) for k, v in rec.items()}
    tr["dt"] = dt
    tr["terminated"] = bool(terminated)
    tr["success"] = bool(det.succeeded and not terminated)
    tr["R"] = np.array([quat_to_rotmat(q) for q in tr["quat"]])
    tr["tilt"] = np.degrees(np.arccos(np.clip(tr["R"][:, 2, 2], -1, 1)))
    tr["seed"] = int(seed)
    return tr


def failure_reason(tr) -> str:
    """The first unmet §04 condition, in the order a flip has to meet them."""
    if tr["success"]:
        return ""
    if tr["terminated"]:
        return "crashed / left the region"
    if not tr["inv"][-1]:
        return "never inverted, no real flip"
    if tr["rev"][-1] < SuccessConfig().full_rev_rad:
        return "no complete 360° rotation"
    if not tr["alt_ok"][-1]:
        return "fell below the minimum altitude"
    return "never settled upright and still"


# --------------------------------------------------------------------------- #
# Time warp: slow motion through the flip, fast-forward once it has settled    #
# --------------------------------------------------------------------------- #
def frame_times(tr, slowmo: float, ff: float):
    t, rev, succ = tr["t"], tr["rev"], tr["succ"]
    moving = np.flatnonzero(np.abs(rev) > np.deg2rad(8))
    if moving.size:
        t0 = max(0.0, t[moving[0]] - 0.15)
        done = np.flatnonzero(np.abs(rev) >= np.deg2rad(350))
        t1 = t[done[0]] + 0.35 if done.size else t[moving[-1]] + 0.2
        if not done.size:                 # no real flip (e.g. coning): milder slow motion
            slowmo = min(slowmo, 2.0)
    else:
        t0 = t1 = np.inf
    ts_ = np.flatnonzero(succ)
    t_ff = (t[ts_[0]] + 0.8) if ts_.size else t1 + 1.5
    out, speeds, x = [], [], 0.0
    while x <= t[-1] + 1e-9:
        sp = (1.0 / slowmo) if t0 <= x <= t1 else (ff if x >= t_ff else 1.0)
        out.append(x); speeds.append(sp)
        x += sp / FPS
    out.append(t[-1]); speeds.append(speeds[-1])
    for _ in range(int(0.6 * FPS)):        # hold the last frame so the verdict can be read
        out.append(t[-1]); speeds.append(0.0)
    return np.array(out), np.array(speeds)


def interp_state(tr, x):
    t = tr["t"]
    k = int(np.clip(np.searchsorted(t, x) - 1, 0, len(t) - 2))
    u = float(np.clip((x - t[k]) / max(t[k + 1] - t[k], 1e-9), 0, 1))
    pos = (1 - u) * tr["pos"][k] + u * tr["pos"][k + 1]
    q0, q1 = tr["quat"][k], tr["quat"][k + 1]
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1 - u) * q0 + u * q1
    q /= np.linalg.norm(q)
    j = k + 1 if u > 0.5 else k
    return pos, quat_to_rotmat(q), j


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #
def ned_to_plot(p):
    """NED -> plotting frame (x forward, y left, z up)."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], -p[..., 1], -p[..., 2]], axis=-1)


def canvas_rgb(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def title_card(lines, small=None, y0=0.60):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    y = y0
    for i, (txt, size, color) in enumerate(lines):
        fig.text(0.5, y, txt, ha="center", va="center", fontsize=size, color=color,
                 fontweight="semibold" if i == 0 else "normal")
        y -= 0.09 if i == 0 else 0.065
    if small:
        fig.text(0.5, 0.12, small, ha="center", va="center", fontsize=10, color=MUTED)
    img = canvas_rgb(fig)
    plt.close(fig)
    return img


def render_clip(tr, clip: Clip, model: QuadModel, idx: int, n_clips: int, slowmo: float, ff: float):
    xs, speeds = frame_times(tr, slowmo, ff)
    P = ned_to_plot(tr["pos"])
    start = P[0]
    span = max(3.0, float(np.max(np.ptp(P, axis=0))) + 1.6)
    ctr = 0.5 * (P.max(0) + P.min(0))
    arm = span / 11.0                      # drawn half-diagonal (real: 0.23 m)
    ground = ctr[2] - span / 2

    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(4, 2, width_ratios=[1.45, 1], height_ratios=[1, 1, 1, 1.25],
                          left=0.02, right=0.975, top=0.86, bottom=0.07, wspace=0.12, hspace=0.55)
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ax.set_xlim(ctr[0] - span / 2, ctr[0] + span / 2)
    ax.set_ylim(ctr[1] - span / 2, ctr[1] + span / 2)
    ax.set_zlim(ctr[2] - span / 2, ctr[2] + span / 2)
    ax.set_box_aspect((1, 1, 1), zoom=1.15)
    ax.view_init(elev=14, azim=-72)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("altitude [m]")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.975, 0.98, 1.0))
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"]["color"] = GRID
    ax.tick_params(labelsize=8)
    # start marker: a pole from the ground to the start position (drift reference)
    ax.plot([start[0]] * 2, [start[1]] * 2, [ground, start[2]], color=MUTED, lw=1, ls=":")
    ax.scatter([start[0]], [start[1]], [ground], color=MUTED, s=18, marker="x")

    t_all = tr["t"]
    th_deg = np.degrees(tr["rev"])
    wn = np.linalg.norm(tr["omega"], axis=1)
    ax_th = fig.add_subplot(gs[0, 1]); ax_tl = fig.add_subplot(gs[1, 1]); ax_w = fig.add_subplot(gs[2, 1])
    ax_th.plot(t_all, th_deg, color=GRID, lw=1.2)
    for yv in (180, 360):
        ax_th.axhline(yv, color=MUTED, lw=0.7, ls=":")
    ax_th.set_ylabel("flip angle θ [°]", fontsize=9)
    lo = min(-30, th_deg.min() - 20)
    hi = max(420, th_deg.max() + 20)
    if clip.show_q_int:
        qd = np.degrees(tr["qint"])
        ax_th.plot(t_all, qd, color=GRID, lw=1.0, ls="--")
        lo, hi = min(lo, qd.min() - 20), max(hi, qd.max() + 20)
    ax_th.set_ylim(lo, hi)
    ax_tl.plot(t_all, tr["tilt"], color=GRID, lw=1.2)
    ax_tl.axhline(150, color=MUTED, lw=0.7, ls=":"); ax_tl.axhline(15, color=MUTED, lw=0.7, ls=":")
    ax_tl.set_ylim(-5, 185); ax_tl.set_ylabel("tilt [°]", fontsize=9)
    ax_w.plot(t_all, wn, color=GRID, lw=1.2)
    ax_w.axhline(0.8, color=MUTED, lw=0.7, ls=":")
    ax_w.set_ylim(0, max(4.0, min(25.0, wn.max() * 1.05))); ax_w.set_ylabel("‖ω‖ [rad/s]", fontsize=9)
    ax_w.set_xlabel("time [s]", fontsize=9)
    ax_w.text(t_all[-1], 0.85, "0.8 rad/s", fontsize=7, color=INK_2, ha="right", va="bottom")
    for a_ in (ax_th, ax_tl, ax_w):
        a_.set_xlim(0, t_all[-1]); a_.grid(color=GRID, lw=0.5); a_.tick_params(labelsize=8)
    l_th, = ax_th.plot([], [], color=TRAIL, lw=2)
    l_q = ax_th.plot([], [], color=NOSE, lw=1.6, ls="--")[0] if clip.show_q_int else None
    if clip.show_q_int:
        ax_th.legend([l_th, l_q], ["θ geometric (§04)", "∫q dt (run-4 reward)"], fontsize=7, loc="upper left")
    l_tl, = ax_tl.plot([], [], color=TRAIL, lw=2)
    l_w, = ax_w.plot([], [], color=TRAIL, lw=2)
    cur = [a_.axvline(0, color=INK, lw=0.8) for a_ in (ax_th, ax_tl, ax_w)]

    ax_c = fig.add_subplot(gs[3, 1]); ax_c.axis("off")
    ax_c.text(0.0, 1.02, "§04 success detector (live, from the true state)", fontsize=9.5,
              color=INK, fontweight="semibold", transform=ax_c.transAxes, va="bottom")
    rows = [ax_c.text(0.0, 0.84 - 0.18 * i, "", fontsize=9.5, transform=ax_c.transAxes, va="top",
                      family="DejaVu Sans") for i in range(5)]

    fig.text(0.02, 0.955, f"{idx}/{n_clips}  {clip.title}", fontsize=15, fontweight="semibold", color=INK)
    fig.text(0.02, 0.915, clip.subtitle + f"   ·   seed {tr['seed']}", fontsize=10, color=INK_2)
    verdict = fig.text(0.975, 0.945, "", fontsize=15, fontweight="bold", ha="right", va="center")
    speed_t = fig.text(0.02, 0.025, "", fontsize=10, color=INK_2)
    take = fig.text(0.34, 0.025, "", fontsize=10, color=INK, ha="left")

    trail, = ax.plot([], [], [], color=TRAIL, lw=1.4, alpha=0.8)
    shadow, = ax.plot([], [], [], color="#c9ced6", lw=1.0)
    arms = [ax.plot([], [], [], color=BODY, lw=3, solid_capstyle="round")[0] for _ in range(2)]
    rotors = [ax.plot([], [], [], color=(NOSE if i < 2 else BODY), lw=2)[0] for i in range(4)]
    thrust, = ax.plot([], [], [], color=OK, lw=2)
    # motor positions in the body frame (NED body: x fwd, y right): FL, FR, RR, RL
    mb = np.array([[1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]], float) / np.sqrt(2) * arm
    circ = np.linspace(0, 2 * np.pi, 20)
    ring = np.stack([np.cos(circ), np.sin(circ), np.zeros_like(circ)], 1) * 0.42 * arm
    frames = []
    cfg = SuccessConfig()
    for fi, (x, sp) in enumerate(zip(xs, speeds)):
        pos, R, k = interp_state(tr, x)
        c = ned_to_plot(pos)
        w_pts = ned_to_plot(pos + (R @ mb.T).T)
        arms[0].set_data_3d(*np.stack([w_pts[0], w_pts[2]]).T)
        arms[1].set_data_3d(*np.stack([w_pts[1], w_pts[3]]).T)
        for i in range(4):
            rp = ned_to_plot(pos + (R @ (mb[i] + ring).T).T)
            rotors[i].set_data_3d(rp[:, 0], rp[:, 1], rp[:, 2])
        T_ratio = float(np.sum(model.kTh * tr["motors"][k] ** 2) / (model.mass * model.g))
        up = ned_to_plot(pos + R @ np.array([0, 0, -1.0]) * arm * 1.6 * T_ratio)
        thrust.set_data_3d([c[0], up[0]], [c[1], up[1]], [c[2], up[2]])
        kk = k + 1
        trail.set_data_3d(P[:kk, 0], P[:kk, 1], P[:kk, 2])
        shadow.set_data_3d(P[:kk, 0], P[:kk, 1], np.full(kk, ground))
        l_th.set_data(t_all[:kk], th_deg[:kk])
        if l_q is not None:
            l_q.set_data(t_all[:kk], np.degrees(tr["qint"][:kk]))
        l_tl.set_data(t_all[:kk], tr["tilt"][:kk])
        l_w.set_data(t_all[:kk], wn[:kk])
        for cl in cur:
            cl.set_xdata([x, x])
        # live checklist
        rev_ok = tr["rev"][k] >= cfg.full_rev_rad
        inv_ok = bool(tr["inv"][k])
        alt_ok = bool(tr["alt_ok"][k])
        crashed = tr["terminated"] and k == len(t_all) - 1
        succ = bool(tr["succ"][k]) and not crashed
        final = fi >= len(xs) - int(0.6 * FPS) - 1
        drop = float(np.max(tr["pos"][:kk, 2]) - tr["pos"][0, 2])       # NED: + = below start
        # state per item: True = ✓, False = ✗, None = still pending
        items = [
            (rev_ok or (False if final else None), f"full 360° rotation   (θ = {np.degrees(tr['rev'][k]):5.0f}°)"),
            (inv_ok or (False if final else None), f"passed inverted   (max tilt {tr['tilt'][:kk].max():4.0f}° ≥ 150°)"),
            (alt_ok, f"never >1 m below start   (lowest {max(drop, 0.0):4.2f} m below)"),
            (succ or (False if final else None),
             f"upright <15°, ‖v‖<0.4, ‖ω‖<0.8 for 20 steps   ({min(int(tr['hold'][k]), 20)}/20)"),
            (not crashed, "no crash"),
        ]
        for row, (ok, txt) in zip(rows, items):
            sym, colr = ("○", PENDING) if ok is None else (("✓", OK) if ok else ("✗", BAD))
            row.set_text(f"{sym}  {txt}"); row.set_color(colr)
        if succ:
            verdict.set_text("§04: SUCCESS"); verdict.set_color(OK)
        elif final:
            verdict.set_text("§04: FAIL — " + failure_reason(tr)); verdict.set_color(BAD)
        else:
            verdict.set_text("")
        speed_t.set_text("paused" if sp == 0 else
                         (f"slow motion ×{1 / sp:g}" if sp < 1 else ("real time" if sp == 1 else f"fast ×{sp:g}"))
                         + f"   t = {x:4.2f} s")
        take.set_text(clip.takeaway if final else "")
        frames.append(canvas_rgb(fig))
    plt.close(fig)
    return frames


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/ppo_flip.zip")
    ap.add_argument("--motor-model", default=None, help="optional motor-level policy to add as a clip")
    ap.add_argument("--seed0", type=int, default=10_000, help="first seed tried for every clip")
    ap.add_argument("--max-tries", type=int, default=40)
    ap.add_argument("--slowmo", type=float, default=4.0)
    ap.add_argument("--ff", type=float, default=2.0, help="fast-forward once the detector has fired")
    ap.add_argument("--only", type=int, nargs="+", default=None, help="clip numbers (1-based)")
    ap.add_argument("--split", action="store_true", help="also write one .mp4 per clip")
    ap.add_argument("--out", default="results/video")
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("pip install imageio imageio-ffmpeg")
    os.makedirs(a.out, exist_ok=True)

    env = make_raw_env(action_repeat=4, episode_seconds=10.0)
    check_simulator(env)
    model = QuadModel.from_env(env)
    clips = build_clips(a)
    chosen = [(i + 1, c) for i, c in enumerate(clips) if a.only is None or (i + 1) in a.only]

    trajs = []
    for n, clip in chosen:
        tr = None
        for s in range(a.seed0, a.seed0 + (a.max_tries if clip.expect is not None else 1)):
            tr = record(env, clip.make_ctrl(env), s, clip)
            if clip.expect is None or tr["success"] == clip.expect:
                break
        else:
            print(f"  [warn] clip {n}: no seed in {a.max_tries} gave the expected verdict; showing the last one")
        clip.result = dict(seed=tr["seed"], success=tr["success"], reason=failure_reason(tr),
                           drift=float(np.linalg.norm(tr["pos"][-1, :2] - tr["pos"][0, :2])))
        print(f"clip {n}: {clip.title:45s} seed {tr['seed']}  -> "
              f"{'SUCCESS' if tr['success'] else 'FAIL (' + clip.result['reason'] + ')'}", flush=True)
        trajs.append((n, clip, tr))
    env.close()

    path = os.path.join(a.out, "flip_demo.mp4")
    writer = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    intro = title_card([("Learning a Quadcopter Flip", 30, INK),
                        ("demo: representative successes and failures", 16, INK_2),
                        ("every verdict is the live output of the independent §04 success detector", 12, INK_2),
                        ("(geometric: full 360° about the pitch axis, through inversion, then upright and still)",
                         11, MUTED)],
                       small="Intelligent Control · Project 1 · all clips are real rollouts on the evaluation harness")
    for _ in range(int(3.0 * FPS)):
        writer.append_data(intro)
    for k, (n, clip, tr) in enumerate(trajs):
        head = "SUCCESS" if clip.result["success"] else "FAILURE"
        card = title_card([(f"{k + 1}. {clip.title}", 24, INK), (clip.subtitle, 13, INK_2),
                           (f"outcome: {head.lower()}", 12, OK if clip.result["success"] else BAD)])
        frames = render_clip(tr, clip, model, k + 1, len(trajs), a.slowmo, a.ff)
        for _ in range(int(1.6 * FPS)):
            writer.append_data(card)
        for f in frames:
            writer.append_data(f)
        if a.split:
            w2 = imageio.get_writer(os.path.join(a.out, f"clip{n}.mp4"), fps=FPS, codec="libx264",
                                    quality=8, macro_block_size=8)
            for f in frames:
                w2.append_data(f)
            w2.close()
        print(f"  rendered clip {k + 1}/{len(trajs)} ({len(frames)} frames)", flush=True)
    summary = [("Summary", 26, INK)]
    for k, (n, clip, tr) in enumerate(trajs):
        r = clip.result
        v = "success" if r["success"] else f"FAIL — {r['reason']}"
        extra = f", drift {r['drift']:.2f} m" if clip.wind > 0 else ""
        summary.append((f"{k + 1}. {clip.title}:  {v}{extra}", 12, OK if r["success"] else BAD))
    end = title_card(summary, y0=0.80, small="Full statistics (50 episodes, Wilson intervals) are in results/final/*/summary.md")
    for _ in range(int(4.0 * FPS)):
        writer.append_data(end)
    writer.close()
    with open(os.path.join(a.out, "clips.md"), "w", encoding="utf-8") as f:
        f.write("| # | Clip | Seed | §04 verdict |\n|---|---|---|---|\n")
        for k, (n, clip, tr) in enumerate(trajs):
            r = clip.result
            f.write(f"| {k + 1} | {clip.title} | {r['seed']} | "
                    f"{'success' if r['success'] else 'FAIL: ' + r['reason']} |\n")
    print("->", path)


if __name__ == "__main__":
    main()
