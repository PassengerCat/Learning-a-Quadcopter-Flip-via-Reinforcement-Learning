"""
make_controller_videos.py — demo videos of the MAP-Elites and MTR-PPO controllers.

Every clip is an episode of the evaluation harness itself (Evaluation/evaluate.py), with the
controllers of Evaluation/final_comparison.py, the same conditions (nominal, ppo_track_stress)
and the same seeds as the reported evaluation (10000 and above), so each clip is exactly an
evaluated episode. The episode always runs for the benchmark's 10 s; the clip shows its first
part (until 1.5 s after the success, or --fail-seconds for a failure).

The rendering is the one of make_demo_video.py (3D view, live flip-angle / tilt / body-rate
plots, live detector checklist), reused unchanged. The live checklist is the PPO track's
success detector (Success_Detector/ppo_track_detector.py, the report's success criterion),
replayed on the recorded trajectory once per decision (every 20 ms), as during the
evaluation; its final verdict is checked against the harness verdict of the same episode.
Note: the renderer labels this detector "§04" (the brief's success criterion section).

Clips:
  1. MAP-Elites, nominal                 -> success
  2. MTR-PPO (CTBR), nominal             -> success
  3. MTR-PPO (motors), nominal           -> success
  4. MTR-PPO (motors), stress            -> success
  5. MTR-PPO (CTBR), stress              -> failure (no 0.4 s hold)
  6. MAP-Elites, stress                  -> failure
For a clip that must show a success (or a failure), the seeds are scanned from --seed0 until the
verdict matches; the seed is shown on screen.

Run from the repository root (needs: pip install imageio imageio-ffmpeg):
    python Visualization/make_controller_videos.py                    # all clips -> Visualization/videos/
    python Visualization/make_controller_videos.py --only 2 3 --split # a subset, plus one .mp4 per clip
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (HERE, REPO / "Evaluation", REPO / "Reward_and_Training", REPO / "Baselines", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import evaluate as harness                                                   # noqa: E402
from final_comparison import CONDITIONS, make_controller                     # noqa: E402
from Success_Detector.ppo_track_detector import SuccessDetector, SuccessConfig   # noqa: E402
import run_baselines                                                         # noqa: E402
if not hasattr(run_baselines, "check_simulator"):
    # make_demo_video.py imports check_simulator, which Baselines/run_baselines.py does not define;
    # only its own main() calls it, the renderer used here does not.
    run_baselines.check_simulator = lambda env: None
import make_demo_video as demo                                               # noqa: E402  (renderer)

DECISION_STEPS = harness.PPO_TRACK_UPDATE_STEPS        # 4 x 5 ms = one decision
EPISODE_SECONDS = harness.EPISODE_SECONDS              # the benchmark's 10 s

STRESS = "actuator noise σ = 0.15 · gyro noise σ = 0.2 rad/s · sensor noise on"
CLIPS = [
    # (controller, condition, expect, title, subtitle, takeaway)
    ("map_elites_final", "nominal", True, "MAP-Elites three-phase controller — nominal",
     "climb, rotate on a braking profile, recover · parameters from the MAP-Elites archives",
     "A slow, high flip: long climb, then the rotation and a calm recovery."),
    ("ppo_mtr", "nominal", True, "MTR-PPO, CTBR actions — nominal",
     "collective thrust + body rates, fixed rate loop · trained without demonstrations",
     "A fast pure-pitch flip with little altitude loss."),
    ("ppo_mtr_motors", "nominal", True, "MTR-PPO, motor commands — nominal",
     "a_t = [m1, m2, m3, m4]: the policy commands the four motors directly",
     "The same manoeuvre with the policy acting directly on the motors."),
    ("ppo_mtr_motors", "ppo_track_stress", True, "MTR-PPO, motor commands — stress", STRESS,
     "The motor policy still flips and settles under strong noise."),
    ("ppo_mtr", "ppo_track_stress", False, "MTR-PPO, CTBR actions — stress", STRESS,
     "FAILURE: the flip is completed, but the hover never stays still for 0.4 s."),
    ("map_elites_final", "ppo_track_stress", False, "MAP-Elites three-phase controller — stress", STRESS,
     "FAILURE: the hand-designed recovery does not settle under this noise."),
]


class _Quad:
    __slots__ = ("pos", "quat", "vel", "omega")


class _EnvView:
    """The detector reads the state from env.quad; this feeds it one recorded sample."""
    def __init__(self):
        self.quad = _Quad()

    def set(self, pos, quat, vel, omega):
        self.quad.pos, self.quad.quat, self.quad.vel, self.quad.omega = pos, quat, vel, omega
        return self


def to_demo_trajectory(r, seed: int, motor_range) -> dict:
    """Harness trajectory (5 ms) -> the decision-rate trajectory dict that the renderer expects,
    with the PPO track's detector replayed once per decision."""
    tr = r.trajectory
    idx = np.arange(0, len(tr["t"]), DECISION_STEPS)
    det, view = SuccessDetector(SuccessConfig()), _EnvView()
    det.reset()
    w_min, w_max = motor_range
    act = np.nan_to_num(tr["action"], nan=0.0)                        # row 0 holds no action yet
    out = {k: [] for k in ("t", "pos", "quat", "omega", "vel", "motors", "rev", "inv", "hold",
                           "succ", "alt_ok", "qint")}
    z0 = float(tr["pos"][0, 2])
    alt_ok = True
    for i in idx:
        det.update(None, env=view.set(tr["pos"][i], tr["quat"][i], tr["vel"][i], tr["omega"][i]))
        alt_ok = alt_ok and float(tr["pos"][i, 2]) - z0 <= 1.0
        out["t"].append(float(tr["t"][i])); out["pos"].append(tr["pos"][i]); out["quat"].append(tr["quat"][i])
        out["omega"].append(tr["omega"][i]); out["vel"].append(tr["vel"][i])
        # commanded motor speeds (the env's linear map of the normalised command); used for the thrust arrow
        out["motors"].append(w_min + 0.5 * (np.clip(act[i], -1, 1) + 1.0) * (w_max - w_min))
        out["rev"].append(det.revolution); out["inv"].append(det.passed_inverted)
        out["hold"].append(det._hold); out["succ"].append(det.succeeded)
        out["alt_ok"].append(alt_ok); out["qint"].append(0.0)
    d = {k: np.asarray(v) for k, v in out.items()}
    d["dt"] = DECISION_STEPS * harness.DT
    d["terminated"] = bool(r.terminated)
    d["success"] = bool(det.succeeded and not r.terminated)
    d["R"] = np.array([demo.quat_to_rotmat(q) for q in d["quat"]])
    d["tilt"] = np.degrees(np.arccos(np.clip(d["R"][:, 2, 2], -1, 1)))
    d["seed"] = int(seed)
    if bool(det.succeeded) != bool(r.ppo_track["success"]):
        raise RuntimeError(f"seed {seed}: replayed detector ({det.succeeded}) differs from the harness "
                           f"verdict ({r.ppo_track['success']})")
    return d


def trim(d: dict, t_end: float) -> dict:
    """Keep the samples up to t_end (the verdict of the full episode is kept)."""
    n = int(np.searchsorted(d["t"], t_end, side="right"))
    out = {k: (v[:n] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(d["t"]) else v)
           for k, v in d.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed0", type=int, default=10_000, help="first seed tried for every clip")
    ap.add_argument("--max-tries", type=int, default=50)
    ap.add_argument("--only", type=int, nargs="+", default=None, help="clip numbers (1-based)")
    ap.add_argument("--fail-seconds", type=float, default=6.0, help="length shown for a failure clip")
    ap.add_argument("--slowmo", type=float, default=4.0)
    ap.add_argument("--ff", type=float, default=2.0, help="fast-forward once the detector has fired")
    ap.add_argument("--split", action="store_true", help="also write one .mp4 per clip")
    ap.add_argument("--max-frames", type=int, default=None, help="render at most N frames per clip (quick check)")
    ap.add_argument("--out", default=str(HERE / "videos"))
    ap.add_argument("--name", default="controllers_demo")
    a = ap.parse_args()
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("pip install imageio imageio-ffmpeg")
    os.makedirs(a.out, exist_ok=True)

    env = harness.make_benchmark_env(EPISODE_SECONDS, CONDITIONS["nominal"])
    model = demo.QuadModel.from_env(env)
    motor_range = (float(env.min_w), float(env.max_w))
    env.close()

    chosen = [(i + 1, c) for i, c in enumerate(CLIPS) if a.only is None or (i + 1) in a.only]
    rendered = []
    for n, (ctrl, cond, expect, title, subtitle, takeaway) in chosen:
        d = None
        for seed in range(a.seed0, a.seed0 + a.max_tries):
            r = harness.run_episode(make_controller(ctrl), seed, record=True,
                                    episode_seconds=EPISODE_SECONDS, condition=CONDITIONS[cond])
            d = to_demo_trajectory(r, seed, motor_range)
            if d["success"] == expect:
                break
        else:
            print(f"  [warn] clip {n}: no seed in {a.max_tries} gave the expected verdict; showing the last one")
        ts = np.flatnonzero(d["succ"])
        d = trim(d, (d["t"][ts[0]] + 1.5) if (d["success"] and ts.size) else a.fail_seconds)
        clip = demo.Clip(title, subtitle, None, expect, seconds=float(d["t"][-1]), takeaway=takeaway)
        clip.result = dict(seed=d["seed"], success=d["success"], reason=demo.failure_reason(d), drift=0.0)
        print(f"clip {n}: {title:48s} seed {d['seed']}  -> "
              f"{'SUCCESS' if d['success'] else 'FAIL (' + clip.result['reason'] + ')'}", flush=True)
        rendered.append((n, clip, d))

    path = os.path.join(a.out, f"{a.name}.mp4")
    writer = imageio.get_writer(path, fps=demo.FPS, codec="libx264", quality=8, macro_block_size=8)
    intro = demo.title_card([("Learning a Quadcopter Flip", 30, demo.INK),
                             ("MAP-Elites and MTR-PPO controllers", 16, demo.INK_2),
                             ("every clip is an episode of the evaluation harness (same seeds and noise)", 12, demo.INK_2),
                             ("verdict: the success detector, live (full 360° through inversion, then upright and still)",
                              11, demo.MUTED)],
                            small="Intelligent Control · all clips are real rollouts on the evaluation harness")
    for _ in range(int(3.0 * demo.FPS)):
        writer.append_data(intro)
    for k, (n, clip, d) in enumerate(rendered):
        ok = clip.result["success"]
        card = demo.title_card([(f"{k + 1}. {clip.title}", 24, demo.INK), (clip.subtitle, 13, demo.INK_2),
                                (f"outcome: {'success' if ok else 'failure'}", 12, demo.OK if ok else demo.BAD)])
        frames = demo.render_clip(d, clip, model, k + 1, len(rendered), a.slowmo, a.ff)
        if a.max_frames:
            frames = frames[:a.max_frames]
        for _ in range(int(1.6 * demo.FPS)):
            writer.append_data(card)
        for f in frames:
            writer.append_data(f)
        if a.split:
            w2 = imageio.get_writer(os.path.join(a.out, f"{a.name}_clip{n}.mp4"), fps=demo.FPS,
                                    codec="libx264", quality=8, macro_block_size=8)
            for f in frames:
                w2.append_data(f)
            w2.close()
        print(f"  rendered clip {k + 1}/{len(rendered)} ({len(frames)} frames)", flush=True)
    summary = [("Summary", 26, demo.INK)]
    for k, (n, clip, d) in enumerate(rendered):
        r = clip.result
        summary.append((f"{k + 1}. {clip.title}:  {'success' if r['success'] else 'FAIL — ' + r['reason']}",
                        12, demo.OK if r["success"] else demo.BAD))
    end = demo.title_card(summary, y0=0.80, small="Success rates over 50 seeds with Wilson intervals: see the report")
    for _ in range(int(4.0 * demo.FPS)):
        writer.append_data(end)
    writer.close()
    with open(os.path.join(a.out, f"{a.name}_clips.md"), "w", encoding="utf-8") as f:
        f.write("| # | Clip | Condition | Seed | Verdict (primary detector) |\n|---|---|---|---|---|\n")
        for k, (n, clip, d) in enumerate(rendered):
            r = clip.result
            f.write(f"| {k + 1} | {clip.title} | {CLIPS[n - 1][1]} | {r['seed']} | "
                    f"{'success' if r['success'] else 'FAIL: ' + r['reason']} |\n")
    print("->", path)


if __name__ == "__main__":
    main()
