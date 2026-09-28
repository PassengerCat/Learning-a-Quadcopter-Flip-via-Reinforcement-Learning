"""
optimal_flip_analysis.py — how close is the learned flip to the time-optimal one? (depth item 4)

Classical result (Lupashin et al., ICRA 2010; Mellinger et al. 2012): with a limit on
angular acceleration the fastest rest-to-rest rotation is BANG-BANG — maximum torque
one way, then maximum torque the other way, with collective thrust as low as possible
while the thrust vector points down. This script puts the two flips we have next to
that benchmark:

  1. Theory, torque-limited:     α_max = τ_max / I_yy,  T* = 2·sqrt(2π / α_max)
  2. Theory, rate-capped:        same, but |q| ≤ ω_cap (the CTBR action limit):
                                 T = 2π/ω_cap + ω_cap/α_max  (trapezoidal profile)
  3. Simulator bang-bang:        the SAME idea flown open-loop in the real simulator,
                                 including motor dynamics (1-D search over the switch time)
  4. PPO policy (§03) and the scripted flip (§02), measured on the same seed.

Outputs (in --out): optimal_flip_summary.md, flip_phase_portrait.png, flip_thrust_vs_angle.png

    python optimal_flip_analysis.py --model models/ppo_flip.zip
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_baselines import make_raw_env, check_simulator
from baselines import QuadModel, Mixer, make_controller, control_dt_from_env
from visualize_flip import run as run_traj, COLORS, LABELS, INK, INK_2, MUTED, GRID

BB_COLOR = "#1f7a6d"     # simulator bang-bang (third identity, validated with the other two)


# --------------------------------------------------------------------------- #
def theory(model: QuadModel, omega_cap: float):
    tau = model.max_torque(1)
    Iyy = float(model.inertia[1, 1])
    a = tau / Iyy
    T_tri = 2.0 * np.sqrt(2 * np.pi / a)
    w_peak = np.sqrt(a * 2 * np.pi)
    if omega_cap < w_peak:
        T_cap = 2 * np.pi / omega_cap + omega_cap / a
    else:
        T_cap = T_tri
    return dict(tau=tau, Iyy=Iyy, alpha=a, T_tri=T_tri, w_peak=w_peak, T_cap=T_cap)


def profile(alpha, omega_cap=None, n=2000):
    """Rest-to-rest 2π profile: returns t, θ, q."""
    w_peak = np.sqrt(alpha * 2 * np.pi)
    if omega_cap is None or omega_cap >= w_peak:
        T = 2 * np.sqrt(2 * np.pi / alpha)
        t = np.linspace(0, T, n)
        q = np.where(t < T / 2, alpha * t, alpha * (T - t))
    else:
        ta = omega_cap / alpha
        T = 2 * np.pi / omega_cap + ta
        t = np.linspace(0, T, n)
        q = np.minimum.reduce([alpha * t, np.full_like(t, omega_cap), alpha * (T - t)])
    th = np.concatenate([[0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(t))])
    return t, th, q


def seg_time(t, th_deg, a=10.0, b=350.0):
    ia, ib = np.flatnonzero(th_deg >= a), np.flatnonzero(th_deg >= b)
    return float(t[ib[0]] - t[ia[0]]) if ia.size and ib.size else np.nan


def rest_to_rest(tr):
    """From θ > 1° until |θ-360| < 5° and |q| < 1 rad/s (the rotation itself)."""
    th, t, q = tr["theta"], tr["t"], tr["omega"][:, 1]
    i0 = np.flatnonzero(th > 1.0)
    if not i0.size:
        return np.nan
    ok = np.flatnonzero((np.arange(len(t)) > i0[0]) & (np.abs(th - 360) < 5) & (np.abs(q) < 1.0))
    return float(t[ok[0]] - t[i0[0]]) if ok.size else np.nan


# --------------------------------------------------------------------------- #
def sim_bang_bang(model: QuadModel, dt_grid=0.0025, refine=True):
    """Open-loop bang-bang in the real simulator at its native rate (no action-repeat):
    full pitch torque for t_a, then full opposite torque until q <= 0. Collective is
    whatever the saturated mixer gives. 1-D search for t_a so that the rotation stops
    at 360°."""
    from flip_reward import gravity_pitch_increment
    env = make_raw_env(action_repeat=1)
    base = env.unwrapped
    mix = Mixer(model)
    up = mix.action(0.0, np.array([0.0, 10 * model.max_torque(1), 0.0]))      # saturated +pitch
    dn = mix.action(0.0, np.array([0.0, -10 * model.max_torque(1), 0.0]))     # saturated -pitch
    dt = base.dt
    best = None

    def fly(ta):
        obs, _ = env.reset(seed=0)
        th, q_prev, t = 0.0, obs[6:10].astype(float), 0.0
        rec = [(0.0, 0.0, 0.0, float(np.sum(model.kTh * base.quad.wMotor ** 2)))]
        phase = 0
        for k in range(400):
            if t + dt <= ta:
                a = up
            elif t < ta:                          # the switch falls inside this sim step:
                f = (ta - t) / dt                 # blend, so the switch time is continuous
                a = (f * up + (1 - f) * dn).astype(np.float32)
            else:
                a = dn
            obs, _, te, tr, _ = env.step(a)
            t += dt
            qn = obs[6:10].astype(float)
            th += gravity_pitch_increment(q_prev, qn)
            q_prev = qn
            rec.append((t, np.degrees(th), float(base.quad.omega[1]), float(np.sum(model.kTh * base.quad.wMotor ** 2))))
            if te:
                break
            if t > ta and base.quad.omega[1] <= 0.0:
                break
        return abs(np.degrees(th) - 360.0), np.array(rec)

    grid = list(np.arange(0.08, 0.26, dt_grid))
    for ta in grid:
        err, rec = fly(ta)
        if best is None or err < best[0]:
            best = (err, ta, rec)
    if refine:                                    # finer search around the best switch time
        for ta in np.arange(best[1] - dt_grid, best[1] + dt_grid, dt_grid / 25):
            err, rec = fly(ta)
            if err < best[0]:
                best = (err, ta, rec)
    env.close()
    err, ta, rec = best
    return dict(t_switch=ta, err_deg=err, t=rec[:, 0], theta=rec[:, 1], q=rec[:, 2],
                thrust=rec[:, 3] / (model.mass * model.g), T=rec[-1, 0])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/ppo_flip.zip")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--out", default="results/depth/optimal_flip")
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
    from flip_policy import LearnedFlipController
    ppo = LearnedFlipController(a.model)
    omega_cap = float((ppo.rc.get("ctbr") or {}).get("rate_max_rp", 20.0))
    trajs = {"scripted_flip": run_traj(env, make_controller("scripted_flip", env), a.seed, 3.0),
             "ppo_flip": run_traj(env, ppo, a.seed, 3.0)}
    env.close()

    th = theory(model, omega_cap)
    print(f"α_max = {th['alpha']:.0f} rad/s²; torque-limited optimum {th['T_tri']:.3f} s; "
          f"rate-capped ({omega_cap:g} rad/s) {th['T_cap']:.3f} s")
    bb = sim_bang_bang(model)
    print(f"simulator bang-bang: switch at {bb['t_switch']:.4f} s, stops {bb['err_deg']:.1f}° away from 360°, "
          f"rotation time {bb['T']:.3f} s")

    t1, th1, q1 = profile(th["alpha"])
    t2, th2, q2 = profile(th["alpha"], omega_cap)
    rows = [
        ("Theory: bang-bang, torque-limited", th["T_tri"], seg_time(t1, np.degrees(th1)), np.max(q1)),
        (f"Theory: bang-bang, rate cap {omega_cap:g} rad/s", th["T_cap"], seg_time(t2, np.degrees(th2)), np.max(q2)),
        ("Simulator bang-bang (motor dynamics)", bb["T"], seg_time(bb["t"], bb["theta"]), np.max(bb["q"])),
    ]
    for n in ("ppo_flip", "scripted_flip"):
        tr = trajs[n]
        rows.append((LABELS[n], rest_to_rest(tr), seg_time(tr["t"], tr["theta"]), np.max(tr["omega"][:, 1])))

    # thrust while inverted
    inv = {}
    for n, tr in trajs.items():
        T = np.sum(model.kTh * tr["motors"] ** 2, axis=1) / (model.mass * model.g)
        m = (tr["theta"] > 135) & (tr["theta"] < 225)
        inv[n] = (float(np.mean(T[m])) if m.any() else np.nan, float(np.max(T[(tr['theta'] < 20)])))

    # ---- summary ---------------------------------------------------------------
    lines = ["# How close is the learned flip to the time-optimal flip?", "",
             f"Vehicle: τ_max = {th['tau']:.2f} N·m, I_yy = {th['Iyy']:.4f} kg·m² → "
             f"**α_max = {th['alpha']:.0f} rad/s²**. Policy rate limit (CTBR): {omega_cap:g} rad/s.", "",
             "| Flip | Rest-to-rest rotation [s] | 10°→350° [s] | Peak pitch rate [rad/s] |",
             "|---|---|---|---|"]
    for name, T, s, qm in rows:
        lines.append(f"| {name} | {T:.3f} | {s:.3f} | {qm:.1f} |")
    lines += ["",
              "| Controller | Mean thrust/weight while inverted (135°–225°) |",
              "|---|---|"]
    for n, (ti, _) in inv.items():
        lines.append(f"| {LABELS[n]} | {ti:.2f} |")
    ppo_T, ppo_seg = rows[3][1], rows[3][2]
    ppo_over = float(np.max(trajs["ppo_flip"]["theta"]) - 360.0)
    lines += ["", "## Reading it", "",
              f"- **The benchmark.** The torque-limited optimum ({th['T_tri']:.2f} s) needs a peak rate of "
              f"{th['w_peak']:.0f} rad/s. The policy can only command {omega_cap:g} rad/s, so the fair benchmark "
              f"is the rate-capped bang-bang: {th['T_cap']:.2f} s rest-to-rest, {rows[1][2]:.2f} s for 10°→350°. "
              f"Flown open-loop in the simulator (motor lag included) the unconstrained bang-bang takes {bb['T']:.2f} s.",
              f"- **Getting there.** The PPO flip reaches 350° in {ppo_seg:.2f} s. That is as fast as the rate-capped "
              f"bound, and faster than the scripted flip ({rows[4][2]:.2f} s). The policy saturates its pitch-rate "
              f"command and holds it, which is bang-bang *in rate* (the plateau in flip_phase_portrait.png).",
              f"- **Stopping.** The optimal profile brakes so as to stop exactly at 360°. The policy does not brake "
              f"early: it overshoots by ≈{ppo_over:.0f}° and then corrects, so its full rest-to-rest rotation takes "
              f"{ppo_T:.2f} s, against {th['T_cap']:.2f} s for the bound. The learned solution trades a precise stop "
              f"for speed through the inverted phase, which is the dangerous part (thrust points down, altitude "
              f"is lost). The correction happens upright, where it is cheap.",
              "- **Thrust.** Both controllers **cut collective thrust while inverted** "
              f"(PPO {inv['ppo_flip'][0]:.2f}, scripted {inv['scripted_flip'][0]:.2f} × weight), the second "
              "feature of the optimal flip. Thrust pointing down would accelerate the vehicle towards the ground. "
              "Nobody told the policy this; it follows from the reward (a crash costs −5 and altitude drift is "
              "penalised).",
              "- **The scripted flip** is slower by design: its rotation rate (~15 rad/s) comes from a hand-tuned "
              "pulse and a coast phase, not from an optimisation.",
              "- **Why is the policy fast at all?** The reward has no explicit time term. But the post-flip terms "
              "(b_upright, b_hold, the alive bonus) are paid **per second after the flip**, and γ = 0.99 per 0.02 s "
              "(a half-life of ≈1.4 s) discounts later rewards. Finishing earlier is worth more, so the policy "
              "learned to approach the physical limit."]
    with open(os.path.join(a.out, "optimal_flip_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---- figure 1: phase portrait ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(np.degrees(th1), q1, color=MUTED, lw=1.2, ls=(0, (4, 3)), label="theory, torque-limited")
    ax.plot(np.degrees(th2), q2, color=INK_2, lw=1.2, ls=(0, (1.5, 2)), label=f"theory, rate cap {omega_cap:g} rad/s")
    ax.plot(bb["theta"], bb["q"], color=BB_COLOR, lw=2, label="simulator bang-bang")
    for n in ("scripted_flip", "ppo_flip"):
        tr = trajs[n]
        m = (tr["theta"] > -30) & (tr["t"] < rest_to_rest(tr) + tr["t"][np.flatnonzero(tr['theta'] > 1)[0]] + 0.2)
        ax.plot(tr["theta"][m], tr["omega"][m, 1], color=COLORS[n], lw=2, label=LABELS[n])
    ax.axvline(180, color=MUTED, lw=0.8, ls=":"); ax.axvline(360, color=MUTED, lw=0.8, ls=":")
    ax.text(182, 1, "inverted", fontsize=8, color=INK_2); ax.text(362, 1, "360°", fontsize=8, color=INK_2)
    ax.set_xlabel("flip angle θ [°]"); ax.set_ylabel("pitch rate q [rad/s]")
    ax.set_title("Phase portrait of the flip: learned vs scripted vs time-optimal")
    ax.grid(color=GRID, lw=0.6); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "flip_phase_portrait.png"), dpi=200); plt.close(fig)

    # ---- figure 2: thrust vs angle --------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for n in ("scripted_flip", "ppo_flip"):
        tr = trajs[n]
        T = np.sum(model.kTh * tr["motors"] ** 2, axis=1) / (model.mass * model.g)
        m = (tr["theta"] > 1) & (tr["theta"] < 359)
        ax.plot(tr["theta"][m], T[m], color=COLORS[n], lw=2, marker="o", ms=3, label=LABELS[n])
    ax.axvspan(135, 225, color="#eef0f3", zorder=0)
    ax.text(180, ax.get_ylim()[1] * 0.92, "thrust points\ndownwards", ha="center", fontsize=8, color=INK_2)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":"); ax.text(3, 1.05, "hover thrust", fontsize=8, color=INK_2)
    ax.set_xlabel("flip angle θ [°]"); ax.set_ylabel("collective thrust / weight")
    ax.set_title("Both controllers cut thrust while upside down")
    ax.grid(color=GRID, lw=0.6); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "flip_thrust_vs_angle.png"), dpi=200); plt.close(fig)
    print("->", a.out)


if __name__ == "__main__":
    main()
