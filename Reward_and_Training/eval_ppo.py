"""
eval_ppo.py — evaluate a trained §03 policy next to the §02 baselines.

The learned controller runs through the IDENTICAL harness as the baselines
(run_baselines.evaluate): same env stack, action-repeat, seed bank, noise,
metrics and the independent §04 SuccessDetector.

    python eval_ppo.py --model runs/<run>/best_model.zip
    python eval_ppo.py --model runs/<run>/best_model.zip --episodes 50 --action-noise 0.05 --plot
    python eval_ppo.py --model ... --no-baselines           # only the learned policy
"""
from __future__ import annotations

import argparse
import os
import sys

from run_baselines import evaluate, load_script, plot, make_raw_env, check_simulator
from flip_policy import LearnedFlipController


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to best_model.zip / final_model.zip")
    ap.add_argument("--config", default=None, help="config.json (default: next to the model)")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=10_000)
    ap.add_argument("--action-noise", type=float, default=0.0)
    ap.add_argument("--obs-noise", type=float, default=0.0)
    ap.add_argument("--wind", type=float, default=0.0, help="median wind speed [m/s], env's RANDOMSINE model")
    ap.add_argument("--params", default="scripted_params.json")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--pid-handoff", action="store_true",
                    help="also evaluate the hybrid: PPO flips, the §02 PID holds position afterwards")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass

    probe = LearnedFlipController(a.model, a.config)
    k = int(probe.rc["action_repeat"])
    env = make_raw_env(action_repeat=k)
    check_simulator(env)
    env.close()

    names = [] if a.no_baselines else ["random", "pid_hover", "scripted_flip"]
    names.append("ppo_flip")
    factories = {"ppo_flip": lambda env: LearnedFlipController(a.model, a.config)}
    if a.pid_handoff:
        names.append("ppo_flip+pid")
        factories["ppo_flip+pid"] = lambda env: LearnedFlipController(a.model, a.config, pid_handoff=True)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.model)), "eval")
    print(f"evaluating {names} — {a.episodes} episodes, seeds {a.seed0}..{a.seed0 + a.episodes - 1}, "
          f"action noise {a.action_noise}, gyro noise {a.obs_noise}, wind {a.wind} m/s")
    evaluate(names, a.episodes, a.seed0, k, a.action_noise, a.obs_noise,
             load_script(a.params), out, factories=factories, wind=a.wind)
    if a.plot:
        plot(out, names)
    print(f"results -> {out}/summary.md")


if __name__ == "__main__":
    main()
