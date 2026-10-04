"""MTR-PPO training: pure-RL PPO with the multiplicative tracking reward (no behaviour cloning,
no demonstrations).

    python MTR_PPO/train_mtr.py --dry-run                     # pipeline check, no learning
    python MTR_PPO/train_mtr.py --smoke                       # ~minutes, tiny learning run

The final recipe (all options below), from a fresh network. The reported models (runs 13 and
14) are the final models of these two runs; --action-mode motors gives the motors model:

    python MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --kernels dense --ref-start-prob 0.5 \
        --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1 --axis-term --axis-limit-deg 40 \
        --axis-penalty 2 --timesteps 3000000

Each training run's options are recorded in its config.json (Logs/MTR_PPO/runs/).

PPO (Stable-Baselines3) with the PPO track's asymmetric actor-critic (the actor reads the 25
tracking observations, the critic also the 26 privileged ones), on make_mtr_env workers:
  * the initial-state curriculum grows linearly from 0 to 1 over the first
    --curriculum-frac of the timesteps (as in GEAR, "the randomization range expands progressively");
  * the entropy coefficient falls linearly from --ent-start to --ent-end;
  * --kernels narrow|dense: GEAR's reward kernels (default) or the wider "dense" kernels;
  * --ref-start-prob p: a fraction p of training episodes starts ON the reference at a random
    loop phase (reference state initialisation); the rest start at hover (default 0: all hover);
  * --backtrack-deg d: during the flip, end the episode when the revolution falls more than d
    degrees below the largest reached (default 0: off);
  * --alive-bonus b: after the flip (HOVER), b per second while airborne, weighted by
    uprightness (default 0: off); --alive-speed-k k also weights it by 1 / (1 + k |v|^2), so it
    pays for stopping (default 0: no speed weighting);
  * --axis-term: multiply the reward by a pitch-axis kernel, so that rolling out of the flip
    plane or turning the heading costs reward (default off);
  * --axis-limit-deg d, --axis-penalty P: during the flip, the first time the pitch axis deviates
    by more than d degrees, the reward drops once by P (d = 0 = off, default; P = 2). Nothing
    is terminated: the environment's termination conditions stay as the brief requires;
  * --init-model path: start from a saved model (fine-tuning, e.g. a second stage for the
    recovery) instead of a fresh network. The network, its action spread and the observation
    layout come from the file; the PPO settings, schedules and environment from this command.
    Use --curriculum-frac 0 to start with the full initial-state randomisation;
  * every --eval-freq steps the deployed controller (mtr_controller.MTRController) flies
    --eval-episodes nominal episodes in the universal evaluation harness (Evaluation/evaluate.py),
    omega = --eval-omega. Success is judged by the PPO track's detector (the reported definition),
    with the §04 verdict logged alongside. best_model.zip = best (PPO-track success, §04 success,
    -crash rate). Rewards are never used for selection.
Output: <logdir>/<tag>_<action_mode>_seed<seed>_<time>/ (default logdir MTR_PPO/runs/, ignored by
git) with config.json, eval_log.csv, best_model.zip, final_model.zip, checkpoints/ and
TensorBoard logs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_RT = os.path.join(_REPO, "Reward_and_Training")   # the PPO track's modules
for _p in (_HERE, _RT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch as th                                                                   # noqa: E402
from stable_baselines3 import PPO                                                    # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback      # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor  # noqa: E402

from mtr_env import N_PRIV, make_mtr_env                                       # noqa: E402
from mtr_controller import MTRController                                        # noqa: E402
from flip_policy import AsymmetricActorCriticPolicy, model_to_dict                   # noqa: E402

for _p in (os.path.join(_REPO, "Evaluation"), _REPO):
    if _p not in sys.path:
        sys.path.append(_p)
import evaluate as harness                                                           # noqa: E402

INFO_KEYS = ("flip_success", "flip_inverted", "crashed", "flip_rev", "mtr_task", "mtr_ref_start",
             "mtr_backtrack", "mtr_axis_violation")


def linear(start: float, end: float):
    """SB3 schedule: progress_remaining goes 1 -> 0."""
    return lambda p: end + (start - end) * p


def curriculum_scale(t: int, total: int, frac: float) -> float:
    return 1.0 if frac <= 0 else float(np.clip(t / (frac * max(1, total)), 0.0, 1.0))


class ScheduleCallback(BaseCallback):
    """Entropy coefficient (linear) and initial-state curriculum scale, per rollout."""

    def __init__(self, ent_start: float, ent_end: float, total: int, curriculum_frac: float):
        super().__init__()
        self.ent_start, self.ent_end, self.total, self.frac = ent_start, ent_end, max(1, total), curriculum_frac

    def _on_rollout_start(self) -> None:
        t = self.model.num_timesteps
        self.model.ent_coef = self.ent_start + (self.ent_end - self.ent_start) * min(1.0, t / self.total)
        self.scale = curriculum_scale(t, self.total, self.frac)
        self.training_env.env_method("set_curriculum", self.scale)

    def _on_rollout_end(self) -> None:
        self.logger.record("train/ent_coef_sched", self.model.ent_coef)
        self.logger.record("train/curriculum", self.scale)
        buf = list(self.model.ep_info_buffer or [])
        if buf and "flip_success" in buf[0]:
            for k in INFO_KEYS:
                self.logger.record(f"train_flip/{k}", float(np.mean([b[k] for b in buf])))

    def _on_step(self) -> bool:
        return True


class HarnessEvalCallback(BaseCallback):
    """Deployed controller in the universal harness (nominal benchmark), both detectors."""

    def __init__(self, run_cfg: dict, run_dir: str, eval_freq: int, n_episodes: int,
                 seconds: float, seed0: int = 10_000):
        super().__init__()
        self.run_cfg, self.run_dir = run_cfg, run_dir
        self.eval_freq, self.n_episodes, self.seconds, self.seed0 = eval_freq, n_episodes, seconds, seed0
        self._next = eval_freq
        self.best = None
        self.csv_path = os.path.join(run_dir, "eval_log.csv")

    def _on_training_start(self) -> None:
        self._next = self.num_timesteps + self.eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.eval_freq
            self.evaluate()
        return True

    def _on_training_end(self) -> None:
        self.evaluate()

    def evaluate(self) -> dict:
        ctrl = MTRController(self.model, self.run_cfg)
        res = harness.evaluate(ctrl, range(self.seed0, self.seed0 + self.n_episodes),
                               episode_seconds=self.seconds)
        n = len(res)
        row = dict(timesteps=int(self.model.num_timesteps),
                   success_ppo_track=sum(r.ppo_track["success"] for r in res) / n,
                   success_section04=sum(r.success for r in res) / n,
                   crash_rate=sum(r.terminated for r in res) / n,
                   passed_inverted=sum(r.ppo_track["passed_inverted"] for r in res) / n,
                   revolution_deg=float(np.mean([np.degrees(r.ppo_track["revolution_rad"]) for r in res])),
                   final_tilt_deg=float(np.mean([r.final_tilt_deg for r in res])),
                   final_omega=float(np.mean([r.final_omega for r in res])))
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        log = getattr(self.model, "_logger", None)
        if log is not None:
            for k, v in row.items():
                if k != "timesteps":
                    log.record(f"eval/{k}", v)
            log.dump(self.model.num_timesteps)
        key = (row["success_ppo_track"], row["success_section04"], -row["crash_rate"])
        if self.best is None or key > self.best:
            self.best = key
            self.model.save(os.path.join(self.run_dir, "best_model.zip"))
        print(f"[eval @ {row['timesteps']:>9,}] success {row['success_ppo_track']*100:5.1f}% (PPO-track) / "
              f"{row['success_section04']*100:5.1f}% (§04)  inverted {row['passed_inverted']*100:5.1f}%  "
              f"crash {row['crash_rate']*100:5.1f}%  rev {row['revolution_deg']:+7.1f}°", flush=True)
        return row


def make_vec(n_envs: int, env_kwargs: dict, seed: int):
    def thunk(rank: int):
        def _init():
            th.set_num_threads(1)
            env = make_mtr_env(**env_kwargs)
            env.reset(seed=seed + 1000 * rank)
            return env
        return _init
    cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    return VecMonitor(cls([thunk(i) for i in range(n_envs)]), info_keywords=INFO_KEYS)


def check_init_config(init_model: str, run_cfg: dict) -> None:
    """The saved model must come from a run with the same action mode and actor observation."""
    try:
        from mtr_controller import _find_config, load_config
        src = load_config(_find_config(init_model))
    except FileNotFoundError:
        return                                   # no config next to it: the spaces check remains
    for k in ("action_mode", "n_actor", "n_priv"):
        if src.get(k) != run_cfg[k]:
            raise ValueError(f"--init-model was trained with {k}={src.get(k)!r}, this run has {run_cfg[k]!r}")


def build_model(a, venv, run_cfg: dict) -> PPO:
    """A fresh PPO model, or (--init-model) a saved one with this command's PPO settings."""
    ppo = dict(n_steps=a.n_steps, batch_size=a.batch_size, n_epochs=a.n_epochs,
               learning_rate=linear(a.lr, a.lr_end), gamma=a.gamma, gae_lambda=a.gae_lambda,
               clip_range=a.clip, ent_coef=a.ent_start, vf_coef=0.5, max_grad_norm=0.5,
               verbose=0, tensorboard_log=a.logdir)
    if a.init_model is None:
        return PPO(AsymmetricActorCriticPolicy, venv, seed=a.seed, device="cpu", **ppo,
                   policy_kwargs=dict(n_actor_obs=run_cfg["n_actor"], net_arch=dict(pi=a.pi_arch, vf=a.vf_arch),
                                      activation_fn=th.nn.Tanh, log_std_init=a.log_std_init))
    check_init_config(a.init_model, run_cfg)
    saved = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": 0.2}   # replaced by ppo
    model = PPO.load(a.init_model, env=venv, device="cpu", custom_objects=saved, **ppo)   # checks the spaces
    model.set_random_seed(a.seed)
    return model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--action-mode", choices=["ctbr", "motors"], default="ctbr")
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="mtr")
    p.add_argument("--logdir", default=os.path.join(_HERE, "runs"))
    p.add_argument("--episode-seconds", type=float, default=6.0)
    p.add_argument("--omega-min", type=float, default=4.5)
    p.add_argument("--omega-max", type=float, default=5.5)
    p.add_argument("--radius", type=float, default=0.6)
    p.add_argument("--eval-omega", type=float, default=5.0)
    p.add_argument("--curriculum-frac", type=float, default=0.5, help="fraction of timesteps to reach full randomisation")
    p.add_argument("--max-tilt-deg", type=float, default=10.0)
    p.add_argument("--max-rate", type=float, default=1.0)
    p.add_argument("--max-vel", type=float, default=0.5)
    p.add_argument("--kernels", choices=["narrow", "dense"], default="narrow",
                   help="reward kernel widths: GEAR's k values (narrow) or the wider 'dense' set")
    p.add_argument("--ref-start-prob", type=float, default=0.0,
                   help="fraction of training episodes that start on the reference (0 = all at hover)")
    p.add_argument("--backtrack-deg", type=float, default=0.0,
                   help="end a training episode that turns back more than this many degrees during the flip (0 = off)")
    p.add_argument("--alive-bonus", type=float, default=0.0,
                   help="post-flip survival bonus per second, weighted by uprightness (0 = off)")
    p.add_argument("--alive-speed-k", type=float, default=0.0,
                   help="weight the survival bonus by 1 / (1 + k |v|^2) (0 = no speed weighting)")
    p.add_argument("--axis-term", action="store_true",
                   help="multiply the reward by the pitch-axis kernel (pure pitch flip, heading kept)")
    p.add_argument("--axis-limit-deg", type=float, default=0.0,
                   help="FLIP: one-off penalty when the pitch axis first deviates beyond this [deg] (0 = off)")
    p.add_argument("--axis-penalty", type=float, default=2.0, help="size of that one-off penalty (return units)")
    p.add_argument("--init-model", default=None,
                   help="start from this saved model (.zip) instead of a fresh network (fine-tuning)")
    p.add_argument("--rate-max", type=float, default=20.0, help="CTBR roll/pitch rate limit [rad/s]")
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=3e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-start", type=float, default=0.005)
    p.add_argument("--ent-end", type=float, default=0.0)
    p.add_argument("--log-std-init", type=float, default=-1.0)
    p.add_argument("--pi-arch", type=int, nargs="+", default=[128, 128])
    p.add_argument("--vf-arch", type=int, nargs="+", default=[256, 256])
    p.add_argument("--eval-freq", type=int, default=200_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-seconds", type=float, default=10.0)
    p.add_argument("--smoke", action="store_true", help="tiny learning run to check the pipeline (starts learning)")
    p.add_argument("--dry-run", action="store_true", help="build everything, evaluate the untrained policy once, no learning")
    a = p.parse_args(argv)
    if a.smoke:
        a.timesteps, a.n_envs, a.n_steps, a.batch_size = 4096, min(2, a.n_envs), 1024, 512
        a.eval_freq, a.eval_episodes, a.eval_seconds, a.tag = 2048, 2, 4.0, a.tag + "_smoke"
    if a.dry_run:
        a.n_envs, a.eval_episodes, a.eval_seconds, a.tag = min(2, a.n_envs), 2, 4.0, a.tag + "_dryrun"
    if not (0 < a.omega_min <= a.eval_omega <= a.omega_max):
        p.error("need 0 < omega-min <= eval-omega <= omega-max")
    if not 0.0 <= a.ref_start_prob <= 1.0:
        p.error("--ref-start-prob must be in [0, 1]")
    if not 0.0 <= a.backtrack_deg < 360.0:
        p.error("--backtrack-deg must be in [0, 360)")
    if not 0.0 <= a.axis_limit_deg < 180.0:
        p.error("--axis-limit-deg must be in [0, 180)")
    if not 0.0 <= a.axis_penalty < float("inf"):
        p.error("--axis-penalty must be finite and >= 0")
    if not 0.0 <= a.alive_bonus < float("inf"):
        p.error("--alive-bonus must be finite and >= 0")
    if not 0.0 <= a.alive_speed_k < float("inf"):
        p.error("--alive-speed-k must be finite and >= 0")
    if a.init_model is not None:
        a.init_model = os.path.abspath(a.init_model)
        if not os.path.isfile(a.init_model):
            p.error(f"--init-model: no such file {a.init_model}")
    return a


def main(argv=None):
    a = parse_args(argv)
    th.set_num_threads(1)
    env_kwargs = dict(action_mode=a.action_mode, episode_seconds=a.episode_seconds,
                      omega_range=(a.omega_min, a.omega_max), radius=a.radius, max_tilt_deg=a.max_tilt_deg,
                      max_rate=a.max_rate, max_vel=a.max_vel, rate_max=a.rate_max,
                      ref_start_prob=a.ref_start_prob, kernels=a.kernels, backtrack_deg=a.backtrack_deg,
                      alive_bonus=a.alive_bonus, alive_speed_k=a.alive_speed_k, axis_term=a.axis_term,
                      axis_limit_deg=a.axis_limit_deg, axis_penalty=a.axis_penalty)
    probe = make_mtr_env(**env_kwargs)
    run_cfg = dict(action_mode=a.action_mode, action_repeat=4,
                   ctbr=probe.mapper.config() if probe.mapper is not None else None,
                   quad_model=model_to_dict(probe.quad_model), n_actor=probe.n_actor, n_priv=N_PRIV,
                   flip_direction=float(probe.reward.cfg.flip_direction), eps_full=float(probe.reward.cfg.eps_full),
                   radius=a.radius, eval_omega=a.eval_omega, env_kwargs=env_kwargs, args=vars(a))
    probe.close()
    run_name = f"{a.tag}_{a.action_mode}_seed{a.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(a.logdir, run_name)
    os.makedirs(run_dir, exist_ok=False)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    venv = make_vec(a.n_envs, env_kwargs, a.seed)
    model = build_model(a, venv, run_cfg)
    eval_cb = HarnessEvalCallback(run_cfg, run_dir, a.eval_freq, a.eval_episodes, a.eval_seconds)
    print(f"run -> {run_dir}\n{a.action_mode}: actor obs {run_cfg['n_actor']}, privileged {N_PRIV}, "
          f"envs {a.n_envs}, {a.timesteps:,} steps", flush=True)
    if a.dry_run:
        eval_cb.model = model
        eval_cb.evaluate()
        venv.close()
        print("dry run: no learning; config, evaluation and untrained best_model written", flush=True)
        return run_dir
    callbacks = [ScheduleCallback(a.ent_start, a.ent_end, a.timesteps, a.curriculum_frac), eval_cb,
                 CheckpointCallback(save_freq=max(a.timesteps // (a.n_envs * 10), 1),
                                    save_path=os.path.join(run_dir, "checkpoints"), name_prefix="ppo_mtr")]
    t0 = time.time()
    model.learn(total_timesteps=a.timesteps, callback=callbacks, tb_log_name=run_name)
    model.save(os.path.join(run_dir, "final_model.zip"))
    venv.close()
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {run_dir}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
