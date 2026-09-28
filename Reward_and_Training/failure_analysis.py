"""
failure_analysis.py — three failure cases, each with a figure and a mechanism (depth item 6).

  A. Reward hacking by coning (run-4 policy, models/ablation/run4_coning.zip):
     integrated pitch rate says "660°", gravity-based flip angle says "~70°",
     tilt never passes 90°. Why the flip is measured geometrically.
  B. PID-based controllers under sensor/actuator noise (stress test):
     the scripted flip completes the rotation but never becomes "still" enough.
     A small control experiment isolates the cause: hover-only under the same
     noise with (i) the §02 cascaded PID, (ii) the bare CTBR rate loop with a zero
     command, and (iii) the PPO policy.
  C. Steady wind: the memoryless policy drifts downwind; the PID (integral action)
     does not. Top-down trajectories of scripted / PPO / PPO + PID.

Outputs (in --out): failure_summary.md, failure_A_coning.png, failure_B_noise.png, failure_C_wind.png

    python failure_analysis.py --model models/ppo_flip.zip --coning models/ablation/run4_coning.zip
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_baselines import make_raw_env, check_simulator, rollout, quick_metrics
from baselines import make_controller, Controller, QuadModel
from flip_reward import SuccessDetector, SuccessConfig
from visualize_flip import run as run_traj, COLORS, LABELS, INK, INK_2, MUTED, GRID

HYB = "#1f7a6d"
STRESS = dict(action_noise=0.15, obs_noise=0.2)


class ZeroCTBR(Controller):
    """The bare CTBR inner loop with a zero command (hold level, zero rates, hover thrust).
    Used only to isolate WHERE noise sensitivity comes from."""
    name = "ctbr_zero"

    def __init__(self, rc):
        from flip_policy import CTBRMapper, model_from_dict
        self.m = CTBRMapper(model_from_dict(rc["quad_model"]), **rc["ctbr"])

    def act(self, obs):
        return self.m.motor_action(np.zeros(4), np.asarray(obs)[13:16])


def stillness(tr, t_from):
    """Mean true |ω| and % of decisions below the 0.8 rad/s §04 threshold after t_from."""
    w = np.linalg.norm(tr["obs"][:, 13:16], axis=1)
    m = tr["t"] >= t_from
    return float(np.mean(w[m])), float(np.mean(w[m] < 0.8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/ppo_flip.zip")
    ap.add_argument("--coning", default="models/ablation/run4_coning.zip")
    ap.add_argument("--seed", type=int, default=None,
                    help="default: the first seed >= 10000 where the scripted flip fails the stress test")
    ap.add_argument("--episodes", type=int, default=10, help="for the noise control experiment")
    ap.add_argument("--out", default="results/depth/failures")
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    os.makedirs(a.out, exist_ok=True)
    from flip_policy import LearnedFlipController
    env = make_raw_env(action_repeat=4)
    check_simulator(env)
    lines = ["# Failure analysis", ""]

    # ---------------- A: coning --------------------------------------------------
    if os.path.exists(a.coning):
        tr = run_traj(env, LearnedFlipController(a.coning), 10_000, 4.0)
        q_int = np.degrees(np.concatenate([[0], np.cumsum(tr["omega"][1:, 1] * tr["dt"])]))
        fig, axs = plt.subplots(3, 1, figsize=(8.5, 7.2), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1, 0.8]))
        axs[0].plot(tr["t"], q_int, color="#c2410c", lw=2, label="integrated pitch rate ∫q dt  (what run-4's reward paid)")
        axs[0].plot(tr["t"], tr["theta"], color="#574fae", lw=2, label="geometric flip angle θ  (what we use now)")
        axs[0].axhline(360, color=MUTED, lw=0.8, ls=":"); axs[0].text(0.02, 368, "a real flip", fontsize=8, color=INK_2)
        axs[0].set_ylabel("angle [°]"); axs[0].legend(fontsize=8, loc="upper left"); axs[0].grid(color=GRID, lw=0.6)
        axs[1].plot(tr["t"], tr["tilt"], color=INK, lw=2, label="tilt from upright")
        axs[1].axhline(150, color=MUTED, lw=0.8, ls=":"); axs[1].text(0.02, 155, "§04 'inverted' (150°)", fontsize=8, color=INK_2)
        axs[1].set_ylabel("tilt [°]"); axs[1].set_ylim(0, 185)
        axs[1].grid(color=GRID, lw=0.6)
        axs[2].plot(tr["t"], tr["omega"][:, 2], color=INK_2, lw=1.6)
        axs[2].axhline(0, color=MUTED, lw=0.8)
        axs[2].set_ylabel("yaw rate r [rad/s]"); axs[2].set_xlabel("time [s]")
        axs[2].grid(color=GRID, lw=0.6)
        fig.suptitle("A · Reward hacking by coning (run-4 policy)", fontweight="semibold")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "failure_A_coning.png"), dpi=200); plt.close(fig)
        lines += ["## A. Reward hacking by coning (run-4)", "",
                  f"- Integrated pitch rate after 4 s: **{q_int[-1]:.0f}°**, i.e. \"almost two flips\" "
                  f"by the run-4 reward.",
                  f"- Gravity-based flip angle: **{tr['theta'][-1]:.0f}°**. Maximum tilt: **{tr['tilt'].max():.0f}°**. "
                  f"The vehicle never went upside down.",
                  f"- Mean yaw rate {np.mean(tr['omega'][:, 2]):+.2f} rad/s. The vehicle is tilted and spinning about "
                  "the vertical (a cone). In the body frame this makes q oscillate with a non-zero mean, so ∫q dt grows "
                  "without bound. The body-y quaternion increment behaves the same way.",
                  "- **Consequence:** the flip is now measured from the gravity direction, θ = atan2(−g_x, g_z), which only "
                  "reaches 360° if gravity sweeps once around the body x-z plane. The §04 detector also requires an "
                  "explicit inversion (tilt ≥ 150°). The coning policy scores 0 on both.", ""]
        print("A done")

    # ---------------- B: noise --------------------------------------------------------
    rc_ppo = LearnedFlipController(a.model).rc
    det = SuccessDetector(SuccessConfig())
    if a.seed is None:                              # pick a representative FAILURE of the scripted flip
        sc = make_controller("scripted_flip", env)
        a.seed = 10_000
        for sd in range(10_000, 10_020):
            if not bool(rollout(env, sc, sd, detector=det, **STRESS)["detector_success"]):
                a.seed = sd
                break
    trs = {}
    for name, ctrl in (("scripted_flip", make_controller("scripted_flip", env)),
                       ("ppo_flip", LearnedFlipController(a.model))):
        trs[name] = rollout(env, ctrl, a.seed, detector=det, **STRESS)
    fig = plt.figure(figsize=(8.5, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1], hspace=0.45)
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    for name, tr in trs.items():
        w = np.linalg.norm(tr["obs"][:, 13:16], axis=1)
        k = max(1, int(round(0.1 / float(tr["dt"]))))
        axs[0].plot(tr["t"], np.convolve(w, np.ones(k) / k, mode="same"), color=COLORS[name], lw=2,
                    label=f"{LABELS[name]} — §04 {'success' if tr['detector_success'] else 'FAIL'}")
    axs[0].axhline(0.8, color=INK, lw=1, ls=":", label="§04 stillness threshold (0.8 rad/s)")
    axs[0].set_ylim(0, 3); axs[0].set_ylabel("‖ω‖ [rad/s] (0.1 s mean)"); axs[0].set_xlabel("time [s]")
    axs[0].legend(fontsize=8, loc="upper right"); axs[0].grid(color=GRID, lw=0.6)
    # control experiment: hover only, same noise
    ctrls = {"PID hover (§02)": lambda: make_controller("pid_hover", env),
             "CTBR loop, zero command": lambda: ZeroCTBR(rc_ppo),
             "PPO policy (§03)": lambda: LearnedFlipController(a.model)}
    ctrl_col = {"PID hover (§02)": "#574fae", "CTBR loop, zero command": MUTED, "PPO policy (§03)": "#c2410c"}
    res = {}
    for label, mk in ctrls.items():
        c = mk()
        ws, frac = [], []
        for i in range(a.episodes):
            tr = rollout(env, c, 50_000 + i, **STRESS, max_seconds=6.0)
            m, f = stillness(tr, 3.0)            # for PPO: the flip is long over by t = 3 s
            ws.append(m); frac.append(f)
        res[label] = (np.mean(ws), np.std(ws), np.mean(frac))
    xs = np.arange(len(res))
    axs[1].bar(xs, [v[0] for v in res.values()], yerr=[v[1] for v in res.values()], width=0.55,
               color=[ctrl_col[k] for k in res], capsize=4)
    axs[1].axhline(0.8, color=INK, lw=1, ls=":")
    axs[1].set_xticks(xs); axs[1].set_xticklabels(list(res)); axs[1].set_ylabel("mean ‖ω‖, t ≥ 3 s")
    axs[1].set_xlabel("control experiment: same noise, same seeds, t ≥ 3 s")
    axs[1].grid(color=GRID, lw=0.6, axis="y")
    fig.suptitle("B · Under noise the PID-based controllers never become 'still'", fontweight="semibold")
    fig.savefig(os.path.join(a.out, "failure_B_noise.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
    sc_rot = float(np.degrees(np.sum(trs["scripted_flip"]["obs"][:, 14]) * float(trs["scripted_flip"]["dt"])))
    lines += ["## B. Stress test: why the scripted flip fails (and PPO does not)", "",
              f"Actuator σ = {STRESS['action_noise']}, gyro σ = {STRESS['obs_noise']} rad/s. Seed {a.seed}: scripted "
              f"{'success' if trs['scripted_flip']['detector_success'] else 'FAIL'}, PPO "
              f"{'success' if trs['ppo_flip']['detector_success'] else 'FAIL'}. "
              f"The scripted flip does complete the rotation (≈{sc_rot:.0f}°); what it misses is the stillness "
              "condition afterwards (upper plot).", "",
              "Control experiment (hover only, same noise, mean ‖ω‖ for t ≥ 3 s, and % of decisions below 0.8 rad/s):", "",
              "| Controller | mean ‖ω‖ [rad/s] | time below 0.8 rad/s |", "|---|---|---|"]
    for k, (m, s, f) in res.items():
        lines.append(f"| {k} | {m:.2f} ± {s:.2f} | {f * 100:.0f}% |")
    (pid_m, _, pid_f), (zero_m, _, zero_f), (ppo_m, _, ppo_f) = list(res.values())
    lines += ["",
              f"- **The flip is not the problem; the hover after it is.** Even without any flip, the §02 PID hover "
              f"sits at ‖ω‖ ≈ {pid_m:.2f} rad/s and is below the 0.8 rad/s threshold only {pid_f * 100:.0f}% of the "
              f"time. The detector needs 0.4 s in a row, which rarely happens.",
              f"- **Where the jitter comes from.** The bare rate loop with a zero command already reaches ≈ {zero_m:.2f} "
              f"rad/s. The noisy gyro enters the P rate loop directly, and actuator noise excites the airframe. "
              f"The PID's outer loops (attitude, velocity and position, with integrators) add to it "
              f"({zero_m:.2f} → {pid_m:.2f}).",
              f"- **The PPO policy goes below the bare loop** (≈ {ppo_m:.2f} rad/s, {ppo_f * 100:.0f}% of the time "
              f"under the threshold). It uses the same rate loop, so its rate *commands* must actively cancel part "
              f"of the disturbance, rather than simply being low-gain. It was trained under the env's sensor noise, "
              f"with a reward that pays for being still after the flip, so it learned exactly this. A PID tuned for "
              f"nominal conditions was never asked to.", ""]
    print("B done")

    # ---------------- C: wind --------------------------------------------------------
    wind = 5.0
    tw = {}
    for name, ctrl in (("scripted_flip", make_controller("scripted_flip", env)),
                       ("ppo_flip", LearnedFlipController(a.model)),
                       ("ppo_flip+pid", LearnedFlipController(a.model, pid_handoff=True))):
        tw[name] = rollout(env, ctrl, a.seed, detector=det, wind=wind)
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    col = dict(COLORS, **{"ppo_flip+pid": HYB})
    lab = dict(LABELS, **{"ppo_flip+pid": "PPO + PID hold"})
    for name, tr in tw.items():
        xy = tr["obs"][:, 3:5]
        ax.plot(xy[:, 1], xy[:, 0], color=col[name], lw=2,
                label=f"{lab[name]} — final drift {np.linalg.norm(xy[-1] - xy[0]):.2f} m")
        ax.plot(xy[-1, 1], xy[-1, 0], "o", color=col[name], ms=7)
    ax.plot(0, 0, marker="+", color=INK, ms=14, mew=2); ax.text(0.05, 0.05, "start", fontsize=8, color=INK_2)
    ax.set_xlabel("east y [m]"); ax.set_ylabel("north x [m]"); ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6); ax.legend(fontsize=8, loc="best")
    ax.set_title(f"C · Top view, {wind:g} m/s wind, 10 s episode", fontweight="semibold")
    fig.savefig(os.path.join(a.out, "failure_C_wind.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
    lines += [f"## C. Steady wind ({wind:g} m/s): drift of the memoryless policy", "",
              "| Controller | §04 | final drift [m] |", "|---|---|---|"]
    for name, tr in tw.items():
        m = quick_metrics(tr)
        lines.append(f"| {lab[name]} | {'success' if m['success_detector'] else 'FAIL'} | {m['final_xy_drift_m']:.2f} |")
    lines += ["",
              "- **Why PPO drifts.** A constant force needs a constant counter-lean. A controller that only sees the "
              "current state (position error, velocity) settles where the lean it commands balances the wind, which is "
              "a non-zero position error. This is the classical steady-state error of proportional control.",
              "- **Why the PID does not.** It removes that error with its integrator.",
              "- **Two ways out.** The hybrid (PID after the flip). Or giving the policy the integral as an input: "
              "depth experiment 5 (`--pos-integral`), whose result is in `results/depth/aggregate/`."]
    with open(os.path.join(a.out, "failure_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    env.close()
    print("->", a.out)


if __name__ == "__main__":
    main()
