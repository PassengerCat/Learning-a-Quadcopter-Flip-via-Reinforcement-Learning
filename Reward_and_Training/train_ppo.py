"""
train_ppo.py — §03 first training run for the Quadcopter Flip (PPO / Stable-Baselines3).

Trains a PPO policy on the wired flip task (`flip_task_env.make_flip_task_env`) with
parallel `SubprocVecEnv` workers, and logs flip-specific diagnostics via an eval
callback that uses the *independent* success criterion (never the reward).

Run-1 config (locked): s_alive = 0, gamma_shape = 1, all other reward defaults.

Requirements
------------
    numpy<2, scipy<1.13, gymnasium, stable-baselines3, tensorboard
    `flip_env.py`, `flip_reward.py`, `flip_task_env.py` importable, and the fork's
    `quad_velocity_env.py` on PYTHONPATH (same requirement as §01).

Examples
--------
    python train_ppo.py --smoke                      # 3k steps, 1 env: sanity only
    python train_ppo.py --timesteps 3000000 --n-envs 8
    python train_ppo.py --s-alive 0.005              # an ablation
    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# --- make the §01/§03 modules importable ------------------------------------ #
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.join(_HERE, "..", "Environment_Task_Wrapper")):
    _cand = os.path.abspath(_cand)
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "stable-baselines3 is required: pip install 'stable-baselines3>=2.0' tensorboard\n"
        f"(import error: {e})"
    )

from flip_task_env import make_flip_task_env
from flip_reward import FlipRewardConfig, SuccessConfig


# --------------------------------------------------------------------------- #
# Eval callback: independent success + flip diagnostics -> TensorBoard          #
# --------------------------------------------------------------------------- #
class FlipEvalCallback(BaseCallback):
    """Every `eval_freq` steps, run `n_episodes` deterministic episodes on a fresh
    env and log flip metrics. Success is judged by the SAME geometric criterion as
    `SuccessDetector` (full revolution + upright cone + slow, held N decisions),
    read from the reward's tracked/cached quantities — the training reward is never
    consulted for success."""

    def __init__(self, eval_freq: int, n_episodes: int, env_kwargs: dict,
                 succ: SuccessConfig, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.env_kwargs = env_kwargs
        self.succ = succ
        self._next = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next:
            return True
        self._next += self.eval_freq
        self._run_eval()
        return True

    def _run_eval(self):
        env = make_flip_task_env(**self.env_kwargs)
        rw = env.unwrapped_reward
        s = self.succ
        succ = inv = crash = 0
        revs, alphas, drifts, rets = [], [], [], []
        for ep in range(self.n_episodes):
            obs, _ = env.reset(seed=10_000 + ep)
            done = False
            hold = 0
            ep_ret = 0.0
            reached = succeeded = crashed = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, _ = env.step(action)
                ep_ret += float(r)
                reached = reached or rw.passed_inverted
                ok = (rw.revolution >= s.full_rev_rad
                      and rw.last_tilt < s.cone_rad
                      and rw.last_speed < s.v_thr
                      and rw.last_wnorm < s.w_thr)
                hold = hold + 1 if ok else 0
                if hold >= s.hold_steps:
                    succeeded = True
                crashed = bool(terminated)
                done = terminated or truncated
            succ += int(succeeded); inv += int(reached); crash += int(crashed)
            revs.append(rw.revolution); alphas.append(rw.alpha)
            drifts.append(abs(rw.alpha - rw.revolution)); rets.append(ep_ret)
        env.close()
        n = self.n_episodes
        log = self.logger
        log.record("flip/success_rate", succ / n)
        log.record("flip/reached_inverted_rate", inv / n)
        log.record("flip/crash_rate", crash / n)
        log.record("flip/mean_revolution_rad", float(np.mean(revs)))
        log.record("flip/mean_alpha_rad", float(np.mean(alphas)))
        log.record("flip/alpha_vs_revolution_drift", float(np.mean(drifts)))
        log.record("flip/mean_return", float(np.mean(rets)))
        if self.verbose:
            print(f"[eval @ {self.num_timesteps:>8}] success={succ}/{n} "
                  f"inverted={inv}/{n} crash={crash}/{n} "
                  f"rev̄={np.mean(revs):.2f} drift̄={np.mean(drifts):.3f} ret̄={np.mean(rets):.2f}")


# --------------------------------------------------------------------------- #
def build_env_kwargs(args) -> dict:
    return dict(
        action_repeat=args.action_repeat,
        reward_cfg=FlipRewardConfig(s_alive=args.s_alive, gamma_shape=args.gamma_shape),
        episode_seconds=args.episode_seconds,
    )


def make_vec(n_envs: int, env_kwargs: dict, seed: int):
    def thunk(rank: int):
        def _init():
            return make_flip_task_env(seed=seed + rank, **env_kwargs)
        return _init
    cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = cls([thunk(i) for i in range(n_envs)])
    return VecMonitor(venv)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--action-repeat", type=int, default=4)
    p.add_argument("--episode-seconds", type=float, default=10.0)
    p.add_argument("--s-alive", type=float, default=0.0)         # run-1 = 0
    p.add_argument("--gamma-shape", type=float, default=1.0)     # difference form
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--eval-freq", type=int, default=100_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--smoke", action="store_true", help="tiny run (3k steps, 1 env) to sanity-check the pipeline")
    args = p.parse_args()

    if args.smoke:
        args.timesteps, args.n_envs, args.eval_freq, args.eval_episodes = 3000, 1, 1500, 3

    env_kwargs = build_env_kwargs(args)
    venv = make_vec(args.n_envs, env_kwargs, args.seed)

    run_name = f"flip_ppo_salive{args.s_alive}_seed{args.seed}_{int(time.time())}"
    model = PPO(
        "MlpPolicy", venv,
        n_steps=1024, batch_size=2048, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=args.seed, verbose=1, tensorboard_log=args.logdir,
    )

    eval_cb = FlipEvalCallback(
        eval_freq=args.eval_freq, n_episodes=args.eval_episodes,
        env_kwargs=env_kwargs, succ=SuccessConfig(),
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.timesteps // (args.n_envs * 10), 1),
        save_path=os.path.join(args.logdir, run_name), name_prefix="ppo_flip",
    )

    model.learn(total_timesteps=args.timesteps, callback=[eval_cb, ckpt_cb],
                tb_log_name=run_name, progress_bar=False)  # set True if tqdm+rich installed

    out = os.path.join(args.logdir, run_name, "ppo_flip_final.zip")
    model.save(out)
    venv.close()
    print(f"\nSaved model -> {out}")


if __name__ == "__main__":
    main()
