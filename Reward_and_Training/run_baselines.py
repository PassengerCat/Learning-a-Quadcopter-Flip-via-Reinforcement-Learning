"""
run_baselines.py — run, tune and summarise the §02 baseline controllers.

Validates the whole evaluation path BEFORE any training: every controller is
rolled out through the same flip environment (make_flip_env, identical config
and action-repeat to §03), trajectories are recorded, and a set of quick-look
metrics is computed. The authoritative success verdict belongs to the §04
detector; the ``quick_check`` below is a provisional stand-in with its
tolerances printed next to every result.

Usage
-----
    python run_baselines.py                          # all baselines, 50 episodes each
    python run_baselines.py --episodes 100 --action-noise 0.05
    python run_baselines.py --tune                   # grid-search the scripted flip
    python run_baselines.py --controllers scripted_flip --plot

Seeds: evaluation uses seeds [seed0, seed0+episodes); tuning uses a DISJOINT
bank starting at --tune-seed0, so the scripted baseline is never tuned on the
episodes it is scored on.
"""
from __future__ import annotations

import argparse
import csv
import random
import itertools
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool
from typing import Dict, List, Optional

import numpy as np

from flip_env import make_flip_env
from baselines import (BASELINES, FlipScript, QuadModel, control_dt_from_env,
                       make_controller, quat_to_rotmat, AXIS_INDEX)

try:   # the independent §04 success detector (reward-agnostic), when available
    from flip_reward import SuccessDetector, SuccessConfig
except ImportError:  # pragma: no cover
    SuccessDetector = SuccessConfig = None

# Provisional success tolerances (the §04 detector freezes the real ones).
QUICK_TOL = {
    "min_rotation_deg": 330.0,   # integrated rotation on the flip axis
    "inverted_tilt_deg": 150.0,  # must pass through at least this tilt
    "final_tilt_deg": 15.0,      # upright at the end
    "final_rate": 1.0,           # |omega| at the end [rad/s]
    "max_alt_loss_m": 3.0,       # below the start altitude
    "settle_window_s": 0.5,      # "final" = mean over this last window
}


# --------------------------------------------------------------------------- #
# Environment used by the baselines: the SAME stack as training, minus the    #
# observation transform (controllers act on the raw observation).             #
# --------------------------------------------------------------------------- #
def make_raw_env(action_repeat: int = 4, episode_seconds: float = 10.0, **kw):
    flip = make_flip_env(action_repeat=action_repeat, episode_seconds=episode_seconds, **kw)
    return flip.env          # FlipObservation -> ActionRepeat(base): raw 20-dim obs


def check_simulator(env) -> None:
    """Step the BASE simulator once at hover. ActionRepeat swallows integrator
    exceptions (every episode would silently 'crash' at step 1), so a broken
    install — typically numpy 2.x with the bobzwik dynamics — is caught here."""
    import scipy
    base = env.unwrapped
    base.reset(seed=0)
    hover = 2.0 * (base.hover_w - base.min_w) / (base.max_w - base.min_w) - 1.0
    try:
        base.step(np.full(4, hover, dtype=np.float32))
    except Exception as e:
        raise RuntimeError(
            f"The simulator cannot take a single step: {e!r}\n"
            f"numpy {np.__version__}, scipy {scipy.__version__}. The bobzwik dynamics "
            f"(quadFiles/quad.py) need numpy<2 and scipy<1.13, e.g. a Python 3.11/3.12 venv:\n"
            f"    py -3.12 -m venv .venv  &&  .venv\\Scripts\\activate\n"
            f"    pip install \"numpy<2\" \"scipy<1.13\" gymnasium matplotlib pytest"
        ) from e
    base.reset(seed=0)


# --------------------------------------------------------------------------- #
# Rollout + recording                                                         #
# --------------------------------------------------------------------------- #
def rollout(env, ctrl, seed: int, *, action_noise: float = 0.0, obs_noise: float = 0.0,
            max_seconds: Optional[float] = None, detector=None,
            wind: float = 0.0) -> Dict[str, np.ndarray]:
    """One episode. Noise is injected on the controller side (between controller
    and env), so the environment itself stays untouched. ``detector`` (optional,
    e.g. flip_reward.SuccessDetector) is updated once per decision from the TRUE
    simulator state and its verdict is stored as ``detector_success``."""
    dt = control_dt_from_env(env)
    rng = np.random.default_rng(np.random.SeedSequence([12345, int(seed)]))
    obs, _ = env.reset(seed=int(seed))
    if wind > 0:   # the env's own wind model (RANDOMSINE), reproducible per seed
        random.seed(int(seed))
        env.unwrapped.enable_random_wind(True, magnitude=wind, heading_deg=180.0, elevation_deg=10.0)
    ctrl.reset(seed=int(seed))
    if detector is not None:
        detector.reset()
        detector.update(None, env=env.unwrapped)
    rec = {k: [] for k in ("t", "obs", "action", "phase", "reward")}
    rec["obs"].append(np.asarray(obs, dtype=np.float64)); rec["t"].append(0.0)
    terminated = truncated = False
    info: Dict = {}
    n = 0
    while True:
        o = np.asarray(obs, dtype=np.float64)
        if obs_noise > 0:
            o = o.copy(); o[13:16] += rng.normal(0, obs_noise, 3)   # gyro noise
        a = np.asarray(ctrl.act(o), dtype=np.float32)
        if action_noise > 0:
            a = np.clip(a + rng.normal(0, action_noise, 4), -1, 1).astype(np.float32)
        rec["phase"].append(ctrl.diagnostics.get("phase_id", -1))
        obs, r, terminated, truncated, info = env.step(a)
        n += 1
        if detector is not None:
            detector.update(None, env=env.unwrapped)
        rec["action"].append(a); rec["reward"].append(float(r))
        rec["obs"].append(np.asarray(obs, dtype=np.float64)); rec["t"].append(n * dt)
        if terminated or truncated:
            break
        if max_seconds is not None and n * dt >= max_seconds:
            break
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["dt"] = np.float64(dt)
    out["terminated"] = np.bool_(terminated)
    out["truncated"] = np.bool_(truncated)
    out["integrator_error"] = np.bool_("integrator_error" in info)
    if detector is not None:
        out["detector_success"] = np.bool_(detector.succeeded)
    return out


# --------------------------------------------------------------------------- #
# Quick-look metrics (provisional — §04 is the arbiter)                       #
# --------------------------------------------------------------------------- #
def quick_metrics(tr: Dict[str, np.ndarray], axis: str = "pitch",
                  hover_action: float = 0.0, tol: Dict = QUICK_TOL) -> Dict:
    obs, t, dt = tr["obs"], tr["t"], float(tr["dt"])
    ax = AXIS_INDEX[axis]
    rate = obs[:, 13:16]
    R22 = np.array([quat_to_rotmat(q)[2, 2] for q in obs[:, 6:10]])
    tilt = np.degrees(np.arccos(np.clip(R22, -1, 1)))
    w_ax = rate[:, ax]
    rot = np.concatenate([[0.0], np.cumsum(0.5 * (w_ax[1:] + w_ax[:-1]) * dt)])
    rot_deg = float(np.degrees(rot[-1]))
    z = obs[:, 5]                                   # NED: +z is down
    alt_loss = float(max(0.0, z.max() - z[0]))
    wn = np.linalg.norm(rate, axis=1)
    nwin = max(1, int(round(tol["settle_window_s"] / dt)))
    final_tilt = float(tilt[-nwin:].mean())
    final_rate = float(wn[-nwin:].mean())

    # time to recover: from the first inverted sample until the vehicle is upright
    # and quiet (rate smoothed over 0.1 s against sensor/actuator jitter) for a
    # continuous settle window
    inv_idx = np.flatnonzero(tilt >= tol["inverted_tilt_deg"])
    ks = max(1, min(int(round(0.1 / dt)), len(wn)))   # short (crashed) episodes too
    wn_s = np.convolve(wn, np.ones(ks) / ks, mode="same")
    ok = (tilt <= tol["final_tilt_deg"]) & (wn_s <= tol["final_rate"])
    t_rec = np.nan
    if inv_idx.size:
        run = 0
        for i in range(inv_idx[0], len(ok)):
            run = run + 1 if ok[i] else 0
            if run >= nwin:
                t_rec = float(t[i - nwin + 1] - t[inv_idx[0]])
                break
    a = tr["action"]
    effort = float(np.mean(np.sum((a - hover_action) ** 2, axis=1))) if len(a) else np.nan

    terminated = bool(tr["terminated"])
    success = (abs(rot_deg) >= tol["min_rotation_deg"] and inv_idx.size > 0 and not terminated
               and final_tilt <= tol["final_tilt_deg"] and final_rate <= tol["final_rate"]
               and alt_loss <= tol["max_alt_loss_m"])
    out = {
        "success_quick": bool(success),
        "rotation_deg": rot_deg,
        "max_tilt_deg": float(tilt.max()),
        "passed_inverted": bool(inv_idx.size > 0),
        "final_tilt_deg": final_tilt,
        "final_rate": final_rate,
        "altitude_loss_m": alt_loss,
        "time_to_recover_s": t_rec,
        "control_effort": effort,
        "terminated": terminated,
        "integrator_error": bool(tr["integrator_error"]),
        "episode_s": float(t[-1]),
        "return": float(np.sum(tr["reward"])),
    }
    out["final_xy_drift_m"] = float(np.linalg.norm(obs[-1, 3:5] - obs[0, 3:5]))
    if "detector_success" in tr:     # §04 verdict, and the vehicle must not crash afterwards
        out["success_detector"] = bool(tr["detector_success"]) and not terminated
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarise(rows: List[Dict]) -> Dict:
    n = len(rows)
    k = sum(r["success_quick"] for r in rows)
    lo, hi = wilson(k, n)
    out = {"episodes": n, "success_rate": k / n if n else np.nan, "success_ci95": [lo, hi]}
    if rows and "success_detector" in rows[0]:
        kd = sum(r["success_detector"] for r in rows)
        out["success_detector_rate"] = kd / n
        out["success_detector_ci95"] = list(wilson(kd, n))
    for key in ("rotation_deg", "final_tilt_deg", "final_rate", "altitude_loss_m",
                "time_to_recover_s", "control_effort", "episode_s", "final_xy_drift_m"):
        v = np.array([r[key] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        out[key] = {"mean": float(v.mean()) if v.size else None,
                    "std": float(v.std()) if v.size else None, "n": int(v.size)}
    out["termination_rate"] = float(np.mean([r["terminated"] for r in rows])) if n else np.nan
    return out


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def load_script(path: Optional[str]) -> FlipScript:
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return FlipScript.from_dict(json.load(f))
    return FlipScript()


def evaluate(names, episodes, seed0, action_repeat, action_noise, obs_noise, script, outdir,
             save_traj=True, factories=None, wind: float = 0.0):
    """``factories``: optional {name: callable(env) -> Controller} for controllers that
    are not baselines (e.g. the learned policy); they run through the identical loop."""
    factories = factories or {}
    os.makedirs(outdir, exist_ok=True)
    env = make_raw_env(action_repeat=action_repeat)
    model = QuadModel.from_env(env)
    results, all_rows = {}, []
    for name in names:
        ctrl = factories[name](env) if name in factories else make_controller(name, env, seed=0, script=script)
        det = SuccessDetector(SuccessConfig(flip_direction=float(script.direction))) \
            if SuccessDetector is not None and script.axis == "pitch" else None
        rows, trajs = [], {}
        t0 = time.time()
        for i in range(episodes):
            seed = seed0 + i
            tr = rollout(env, ctrl, seed, action_noise=action_noise, obs_noise=obs_noise, detector=det, wind=wind)
            m = quick_metrics(tr, axis=script.axis, hover_action=model.hover_action)
            m.update({"controller": name, "seed": seed})
            rows.append(m)
            if save_traj and i < 10:
                trajs[f"seed{seed}"] = tr
        results[name] = summarise(rows)
        results[name]["wall_s"] = time.time() - t0
        all_rows += rows
        if save_traj:
            np.savez_compressed(os.path.join(outdir, f"traj_{name}.npz"),
                                **{f"{s}__{k}": v for s, tr in trajs.items() for k, v in tr.items()})
        s = results[name]
        det_txt = (f"§04 {s['success_detector_rate']*100:5.1f}%  " if "success_detector_rate" in s else "")
        print(f"  {name:14s} {det_txt}success {s['success_rate']*100:5.1f}%  "
              f"[{s['success_ci95'][0]*100:.0f}-{s['success_ci95'][1]*100:.0f}]  "
              f"rot {s['rotation_deg']['mean']:+7.1f}°  "
              f"alt-loss {s['altitude_loss_m']['mean']:.2f} m  "
              f"term {s['termination_rate']*100:.0f}%  ({s['wall_s']:.1f}s)")
    env.close()

    with open(os.path.join(outdir, "per_episode.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    meta = {"episodes": episodes, "seed0": seed0, "action_repeat": action_repeat,
            "action_noise": action_noise, "obs_noise": obs_noise, "wind": wind,
            "quick_tolerances": QUICK_TOL, "scripted_params": script.to_dict(),
            "control_dt": control_dt_from_env(make_raw_env(action_repeat=action_repeat))}
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    write_markdown(results, meta, os.path.join(outdir, "summary.md"))
    return results


def _fmt(d, p=2):
    return "—" if d["mean"] is None else f"{d['mean']:.{p}f} ± {d['std']:.{p}f}"


def write_markdown(results, meta, path):
    lines = [f"# Baseline results ({meta['episodes']} episodes/controller, seeds {meta['seed0']}–"
             f"{meta['seed0'] + meta['episodes'] - 1}, action-repeat {meta['action_repeat']}, "
             f"action noise σ={meta['action_noise']}, gyro noise σ={meta['obs_noise']}, "
             f"wind {meta.get('wind', 0)} m/s)", "",
             "| Controller | Success §04 (95% CI) | Success quick (95% CI) | Rotation [°] | Final tilt [°] | "
             "Final ‖ω‖ [rad/s] | Altitude loss [m] | Horizontal drift [m] | Recovery time [s] | "
             "Control effort | Terminated |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in results.items():
        lo, hi = s["success_ci95"]
        if "success_detector_rate" in s:
            dl, dh = s["success_detector_ci95"]
            det = f"{s['success_detector_rate']*100:.0f}% ({dl*100:.0f}–{dh*100:.0f})"
        else:
            det = "—"
        lines.append(f"| {name} | {det} | {s['success_rate']*100:.0f}% ({lo*100:.0f}–{hi*100:.0f}) | "
                     f"{_fmt(s['rotation_deg'], 0)} | {_fmt(s['final_tilt_deg'], 1)} | "
                     f"{_fmt(s['final_rate'])} | {_fmt(s['altitude_loss_m'])} | {_fmt(s['final_xy_drift_m'])} | "
                     f"{_fmt(s['time_to_recover_s'])} | {_fmt(s['control_effort'], 3)} | "
                     f"{s['termination_rate']*100:.0f}% |")
    lines += ["", "Success here is the provisional quick-check "
              f"({', '.join(f'{k}={v}' for k, v in QUICK_TOL.items())}); "
              "the §04 detector is authoritative."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Tuning of the scripted flip (grid search on a disjoint seed bank)           #
# --------------------------------------------------------------------------- #
TUNE_GRID = {
    "pulse_time": [0.06, 0.08, 0.10, 0.12],
    "pulse_torque_frac": [0.6, 0.8, 1.0],
    "brake_angle_deg": [200.0, 240.0, 280.0],
    "climb_time": [0.2, 0.3, 0.4],
}


def _score_config(args):
    params, seeds, action_repeat, action_noise, obs_noise = args
    env = make_raw_env(action_repeat=action_repeat)
    model = QuadModel.from_env(env)
    script = FlipScript.from_dict(params)
    ctrl = make_controller("scripted_flip", env, script=script)
    horizon = script.settle_time + script.climb_time + script.max_flip_time + 2.5
    rows = []
    for s in seeds:
        tr = rollout(env, ctrl, s, action_noise=action_noise, obs_noise=obs_noise, max_seconds=horizon)
        m = quick_metrics(tr, axis=script.axis, hover_action=model.hover_action)
        rows.append(m)
    env.close()
    sr = np.mean([r["success_quick"] for r in rows])
    alt = np.mean([r["altitude_loss_m"] for r in rows])
    rec = np.nanmean([r["time_to_recover_s"] for r in rows] + [np.nan]) if sr > 0 else np.inf
    tilt = np.mean([r["final_tilt_deg"] for r in rows])
    # lexicographic: success first, then smaller altitude loss, faster recovery, then tilt
    return params, float(sr), float(alt), float(rec), float(tilt)


def tune(n_seeds, seed0, action_repeat, action_noise, obs_noise, out_path, workers):
    base = FlipScript().to_dict()
    keys = list(TUNE_GRID)
    configs = [{**base, **dict(zip(keys, vals))} for vals in itertools.product(*TUNE_GRID.values())]
    seeds = list(range(seed0, seed0 + n_seeds))
    jobs = [(c, seeds, action_repeat, action_noise, obs_noise) for c in configs]
    print(f"tuning {len(configs)} configs x {n_seeds} seeds on {workers} workers "
          f"(tuning seeds {seeds[0]}–{seeds[-1]}, noise σ={action_noise}) ...")
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as p:
            res = p.map(_score_config, jobs)
    else:
        res = [_score_config(j) for j in jobs]
    res.sort(key=lambda r: (-r[1], r[2], r[3], r[4]))
    print(f"done in {time.time() - t0:.0f}s. top 5:")
    for p, sr, alt, rec, tilt in res[:5]:
        print(f"  success {sr*100:5.1f}%  alt-loss {alt:.2f} m  recover {rec:.2f}s  tilt {tilt:.1f}°  "
              + "  ".join(f"{k}={p[k]}" for k in keys))
    best = res[0][0]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    with open(os.path.splitext(out_path)[0] + "_grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys + ["success", "alt_loss", "recover_s", "final_tilt"])
        for p, sr, alt, rec, tilt in res:
            w.writerow([p[k] for k in keys] + [sr, alt, rec, tilt])
    print(f"best parameters -> {out_path}")
    return FlipScript.from_dict(best)


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def plot(outdir, names, axis="pitch"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax_i = AXIS_INDEX[axis]
    colors = {"random": "#9aa5b1", "pid_hover": "#1f7a6d", "scripted_flip": "#574fae", "ppo_flip": "#c2410c"}
    fig, axs = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    for name in names:
        path = os.path.join(outdir, f"traj_{name}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        first = sorted({k.split("__")[0] for k in d.files})[0]
        obs, t, dt = d[f"{first}__obs"], d[f"{first}__t"], float(d[f"{first}__dt"])
        w = obs[:, 13 + ax_i]
        rot = np.degrees(np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * dt)]))
        tilt = np.degrees(np.arccos(np.clip([quat_to_rotmat(q)[2, 2] for q in obs[:, 6:10]], -1, 1)))
        c = colors.get(name, None)
        axs[0].plot(t, rot, label=name, color=c)
        axs[1].plot(t, tilt, color=c)
        axs[2].plot(t, -obs[:, 5], color=c)
        axs[3].plot(t, np.linalg.norm(obs[:, 13:16], axis=1), color=c)
    axs[0].axhline(360, ls=":", c="k", lw=0.8)
    axs[1].axhline(180, ls=":", c="k", lw=0.8)
    axs[0].set_ylabel(f"integrated {axis} [°]")
    axs[1].set_ylabel("tilt from upright [°]")
    axs[2].set_ylabel("altitude (−z) [m]")
    axs[3].set_ylabel("‖ω‖ [rad/s]")
    axs[3].set_xlabel("time [s]")
    axs[0].legend(frameon=False, ncol=3)
    for a in axs:
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    axs[3].set_xlim(0, 4)
    fig.suptitle("Controllers — representative episode (first evaluation seed)")
    fig.tight_layout()
    p = os.path.join(outdir, "baselines_timeseries.png")
    fig.savefig(p, dpi=150)
    print(f"plot -> {p}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controllers", nargs="+", default=list(BASELINES), choices=BASELINES)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=10_000, help="first evaluation seed")
    ap.add_argument("--action-repeat", type=int, default=4, help="must match §03 training")
    ap.add_argument("--action-noise", type=float, default=0.0, help="σ of actuator noise (action units)")
    ap.add_argument("--obs-noise", type=float, default=0.0, help="σ of gyro noise [rad/s]")
    ap.add_argument("--wind", type=float, default=0.0, help="median wind speed [m/s] (env's RANDOMSINE model)")
    ap.add_argument("--params", default="scripted_params.json", help="scripted-flip parameters (JSON)")
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--tune", action="store_true", help="grid-search the scripted flip first")
    ap.add_argument("--tune-seeds", type=int, default=8)
    ap.add_argument("--tune-seed0", type=int, default=500, help="disjoint from evaluation seeds")
    ap.add_argument("--tune-noise", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    # Windows consoles with a legacy code page (e.g. cp1253) must never crash on ° or —
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass

    _env = make_raw_env(action_repeat=a.action_repeat)
    check_simulator(_env)
    _env.close()

    if a.tune:
        script = tune(a.tune_seeds, a.tune_seed0, a.action_repeat, a.tune_noise, a.obs_noise,
                      a.params, a.workers)
    else:
        script = load_script(a.params)
    print(f"evaluating {a.controllers} — {a.episodes} episodes, seeds {a.seed0}..{a.seed0 + a.episodes - 1}")
    evaluate(a.controllers, a.episodes, a.seed0, a.action_repeat, a.action_noise, a.obs_noise,
             script, a.out, wind=a.wind)
    if a.plot:
        plot(a.out, a.controllers, script.axis)
    print(f"results -> {a.out}/summary.md, summary.json, per_episode.csv, traj_*.npz")


if __name__ == "__main__":
    main()
