"""
train_ppo.py — §03 PPO training for the Quadcopter Flip (Stable-Baselines3).

What it does
------------
* Behaviour-cloning warm start (default): DART-style demonstrations from
  `ctbr_expert.CTBRExpert` initialise the actor (and the critic, on returns-to-go);
  PPO then fine-tunes on the true reward. `--bc-episodes 0` = pure RL from scratch.
* Parallel CPU workers (SubprocVecEnv) running `flip_task_env.make_flip_task_env`.
* Asymmetric actor-critic (flip_policy.AsymmetricActorCriticPolicy): the actor sees
  only the deployable observation, the critic additionally sees privileged state.
* Entropy-coefficient and learning-rate schedules (linear).
* Periodic evaluation through the DEPLOYED controller (flip_policy.LearnedFlipController)
  on the §02 harness (same env, seeds, noise and metrics as the baselines), with the
  independent §04 SuccessDetector as the success judge — never the reward.
  The best model by (success, -altitude loss) is kept as best_model.zip.
* Everything needed to reload the policy is written to <run>/config.json.

The defaults are the final configuration (run-9 in reward_log.md).

Examples
--------
    python train_ppo.py --smoke                              # ~3 min pipeline check
    python train_ppo.py --timesteps 1000000                  # final config (BC + PPO)
    python train_ppo.py --bc-episodes 0 --lr 3e-4 --log-std-init -1.0 \
        --ent-start 0.005 --demo-prob 0.5 --tag pure_rl      # ablation: no demonstrations
    python train_ppo.py --symmetric --tag symmetric          # ablation: no privileged critic
    python train_ppo.py --timesteps 0 --tag bc_only          # ablation: behaviour cloning only
    python train_ppo.py --resume models/ppo_flip.zip --timesteps 300000 --wind-prob 0.7 --wind-max 5 \
        --lam-pos 0.05 --k-up-g 2 --k-up-p 0.1 --eval-wind 3 --tag wind   # run-10/11 (wind fine-tune)
    tensorboard --logdir runs/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.join(_HERE, "..", "Environment_Task_Wrapper")):
    _cand = os.path.abspath(_cand)
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
except Exception as e:  # pragma: no cover
    raise SystemExit("stable-baselines3 is required: pip install -r requirements.txt\n"
                     f"(import error: {e})")

from flip_env import ObsConfig
from flip_reward import FlipRewardConfig, SuccessConfig
from flip_task_env import make_flip_task_env, N_PRIV
from flip_policy import AsymmetricActorCriticPolicy, LearnedFlipController, model_to_dict
from run_baselines import make_raw_env, check_simulator, rollout, quick_metrics, summarise
from baselines import QuadModel
from flip_reward import SuccessDetector


# --------------------------------------------------------------------------- #
# Schedules                                                                    #
# --------------------------------------------------------------------------- #
def linear(start: float, end: float):
    """SB3 schedule: progress_remaining goes 1 -> 0."""
    return lambda p: end + (start - end) * p


class EntropySchedule(BaseCallback):
    def __init__(self, start: float, end: float, total: int):
        super().__init__()
        self.start, self.end, self.total = start, end, max(1, total)
        self.t0 = 0

    def _on_training_start(self) -> None:
        self.t0 = self.num_timesteps                          # schedule over THIS learn() call

    def _on_step(self) -> bool:
        frac = min(1.0, (self.num_timesteps - self.t0) / self.total)
        self.model.ent_coef = self.start + (self.end - self.start) * frac
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("train/ent_coef_sched", self.model.ent_coef)


# --------------------------------------------------------------------------- #
# Training-episode flip statistics (stochastic policy, from VecMonitor)        #
# --------------------------------------------------------------------------- #
class TrainFlipStats(BaseCallback):
    def _on_rollout_end(self) -> None:
        buf = list(self.model.ep_info_buffer)
        if not buf or "flip_success" not in buf[0]:
            return
        for k in ("flip_success", "flip_inverted", "crashed", "flip_rev"):
            self.logger.record(f"train_flip/{k}", float(np.mean([b[k] for b in buf])))

    def _on_step(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Evaluation through the deployed controller + §04 detector                    #
# --------------------------------------------------------------------------- #
class FlipEvalCallback(BaseCallback):
    def __init__(self, eval_freq: int, n_episodes: int, run_cfg: dict, run_dir: str,
                 eval_seconds: float, seed0: int = 10_000, verbose: int = 1, wind: float = 0.0):
        super().__init__(verbose)
        self.wind = float(wind)
        self.eval_freq, self.n_episodes = eval_freq, n_episodes
        self.run_cfg, self.run_dir = run_cfg, run_dir
        self.eval_seconds, self.seed0 = eval_seconds, seed0
        self._next = eval_freq
        self.best = (-1.0, -np.inf, -np.inf, -np.inf)
        self.csv_path = os.path.join(run_dir, "eval_log.csv")

    def _on_training_start(self) -> None:
        self._next = self.num_timesteps + self.eval_freq     # also correct when resuming

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.eval_freq
            self.evaluate()
        return True

    def _on_training_end(self) -> None:
        self.evaluate()

    def evaluate(self) -> dict:
        rc = self.run_cfg
        env = make_raw_env(action_repeat=rc["action_repeat"], episode_seconds=self.eval_seconds)
        ctrl = LearnedFlipController(self.model, rc)
        model = QuadModel.from_env(env)
        det = SuccessDetector(SuccessConfig(**rc["success_cfg"]))
        rows = []
        for i in range(self.n_episodes):
            tr = rollout(env, ctrl, self.seed0 + i, detector=det, wind=self.wind)
            rows.append(quick_metrics(tr, axis="pitch", hover_action=model.hover_action))
        env.close()
        s = summarise(rows)
        sd = s["success_detector_rate"]
        alt = s["altitude_loss_m"]["mean"] or 0.0
        inv = float(np.mean([r["passed_inverted"] for r in rows]))
        crash = s["termination_rate"]
        rot = s["rotation_deg"]["mean"]
        try:
            log = self.logger
        except AttributeError:          # before learn(): the model has no logger yet
            log = _NullLogger()
        log.record("eval/success_detector", sd)
        log.record("eval/success_quick", s["success_rate"])
        log.record("eval/passed_inverted", inv)
        log.record("eval/crash_rate", crash)
        log.record("eval/rotation_deg", rot)
        log.record("eval/altitude_loss_m", alt)
        log.record("eval/final_tilt_deg", s["final_tilt_deg"]["mean"])
        log.dump(self.num_timesteps)
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timesteps", "success_detector", "success_quick", "passed_inverted",
                            "crash_rate", "rotation_deg", "altitude_loss_m", "final_tilt_deg"])
            w.writerow([self.num_timesteps, sd, s["success_rate"], inv, crash, rot, alt,
                        s["final_tilt_deg"]["mean"]])
        key = (sd, -crash, -alt, -float(s["final_tilt_deg"]["mean"] or 0.0))
        if key > self.best:
            self.best = key
            self.model.save(os.path.join(self.run_dir, "best_model.zip"))
        if self.verbose:
            print(f"[eval @ {self.num_timesteps:>9,} | wind {self.wind:g} m/s] §04 success {sd*100:5.1f}%  inverted {inv*100:5.1f}%  "
                  f"crash {crash*100:5.1f}%  rot {rot:+7.1f}°  alt-loss {alt:.2f} m  "
                  f"(best {self.best[0]*100:.0f}%)", flush=True)
        return s


# --------------------------------------------------------------------------- #
# Behaviour-cloning warm start (DART-style demonstrations from ctbr_expert)     #
# --------------------------------------------------------------------------- #
def collect_demos(env_kwargs: dict, n_episodes: int, seed: int, noise_levels, gamma: float,
                  episode_seconds: float):
    """Roll out the CTBR expert. Executed actions get Gaussian noise (DART), labels are
    the expert's CLEAN actions, so the clone also learns to correct perturbations.
    Returns (obs, actions, returns-to-go) with obs exactly as the policy sees them."""
    from ctbr_expert import CTBRExpert
    from baselines import control_dt_from_env
    kw = dict(env_kwargs, demo_prob=0.0, episode_seconds=episode_seconds)   # demos keep the wind
    env = make_flip_task_env(**kw)
    rate_max = env.mapper.config()["rate_max_rp"]
    ex = CTBRExpert(env.quad_model, control_dt_from_env(env.env), rate_max_rp=rate_max)
    rng = np.random.default_rng(seed)
    O, A, G, n_succ = [], [], [], 0
    for ep in range(n_episodes):
        sigma = noise_levels[ep % len(noise_levels)]
        o, _ = env.reset(seed=seed + ep)
        ex.reset()
        raw = env._last_raw
        obs, acts, rews = [], [], []
        while True:
            a = ex.act(raw)
            obs.append(o); acts.append(a)
            a_exec = np.clip(a + rng.normal(0, sigma, 4), -1, 1).astype(np.float32)
            o, r, te, tr, info = env.step(a_exec)
            raw = env._last_raw
            rews.append(r)
            if te or tr:
                break
        n_succ += int(info["flip_success"] > 0)
        g, ret = 0.0, []
        for r in reversed(rews):
            g = r + gamma * g
            ret.append(g)
        O += obs; A += acts; G += ret[::-1]
    env.close()
    print(f"demos: {n_episodes} expert episodes, {len(O):,} samples, "
          f"expert §04 success {n_succ}/{n_episodes}", flush=True)
    return (np.asarray(O, np.float32), np.asarray(A, np.float32), np.asarray(G, np.float32))


def bc_pretrain(model, O, A, G, epochs: int, lr: float, batch: int = 512, seed: int = 0):
    """Regress the actor MEAN onto the expert actions and the critic onto the
    demonstrations' returns-to-go (so PPO's first updates see a sensible baseline)."""
    pol = model.policy
    opt = th.optim.Adam(pol.parameters(), lr=lr)
    Ot, At, Gt = (th.as_tensor(x) for x in (O, A, G))
    n = len(Ot)
    g = th.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = th.randperm(n, generator=g)
        lp = lv = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            mean = pol.get_distribution(Ot[idx]).distribution.mean
            v = pol.predict_values(Ot[idx]).squeeze(-1)
            loss_pi = th.mean((mean - At[idx]) ** 2)
            loss_v = th.mean((v - Gt[idx]) ** 2)
            loss = loss_pi + 0.1 * loss_v
            opt.zero_grad(); loss.backward(); opt.step()
            lp += float(loss_pi) * len(idx); lv += float(loss_v) * len(idx)
        if ep == 0 or (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"  BC epoch {ep + 1:3d}/{epochs}: actor MSE {lp / n:.4f}  critic MSE {lv / n:.3f}", flush=True)


# --------------------------------------------------------------------------- #
def make_vec(n_envs: int, env_kwargs: dict, seed: int):
    def thunk(rank: int):
        def _init():
            th.set_num_threads(1)
            env = make_flip_task_env(**env_kwargs)
            env.reset(seed=seed + 1000 * rank)
            return env
        return _init
    cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = cls([thunk(i) for i in range(n_envs)])
    return VecMonitor(venv, info_keywords=("flip_success", "flip_inverted", "crashed", "flip_rev"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--action-repeat", type=int, default=4, help="must match §01/§02")
    p.add_argument("--episode-seconds", type=float, default=6.0, help="training episode length")
    p.add_argument("--eval-seconds", type=float, default=10.0, help="evaluation episode length")
    p.add_argument("--init-max-steps", type=int, default=15, help="randomised start: up to N noisy hover decisions")
    p.add_argument("--init-noise", type=float, default=0.15)
    p.add_argument("--action-mode", choices=["ctbr", "motors"], default="ctbr",
                   help="ctbr: thrust + body-rate commands through a fixed rate loop (default); "
                        "motors: raw motor commands (ablation)")
    p.add_argument("--rate-max", type=float, default=20.0, help="CTBR roll/pitch rate limit [rad/s]")
    p.add_argument("--demo-prob", type=float, default=0.3,
                   help="fraction of TRAINING episodes that start mid-flip from the §02 scripted flip")
    p.add_argument("--demo-max-steps", type=int, default=45)
    p.add_argument("--wind-prob", type=float, default=0.0,
                   help="fraction of training episodes with wind (run-10/11 ablation: 0.7)")
    p.add_argument("--pos-integral", action="store_true",
                   help="depth exp. 5: give the actor the integral of the position error (memory)")
    p.add_argument("--wind-max", type=float, default=0.0,
                   help="max median wind speed [m/s] (run-10/11 ablation: 5)")
    p.add_argument("--bc-episodes", type=int, default=120,
                   help="behaviour-cloning warm start from ctbr_expert (0 = pure RL from scratch)")
    p.add_argument("--bc-epochs", type=int, default=40)
    p.add_argument("--bc-lr", type=float, default=1e-3)
    p.add_argument("--bc-noise", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2],
                   help="DART execution-noise levels cycled over demo episodes")
    # reward (FlipRewardConfig) knobs used in ablations
    p.add_argument("--s-alive", type=float, default=0.1, help="survival bonus per second")
    p.add_argument("--over-rot-slope", type=float, default=1.0, help="0 = flat cap beyond 360° (run-1)")
    p.add_argument("--w-prog", type=float, default=3.0, help="'new record' rotation progress weight")
    p.add_argument("--alive-always", action="store_true",
                   help="pay s_alive also before the flip (run-3..5 behaviour)")
    p.add_argument("--gamma-shape", type=float, default=1.0)
    p.add_argument("--b-hold", type=float, default=None)
    p.add_argument("--b-crash", type=float, default=None)
    p.add_argument("--lam-pos", type=float, default=0.005,
                   help="horizontal position penalty per metre per second (run-10/11 ablation: 0.05)")
    p.add_argument("--k-up-g", type=float, default=5.0, help="b_upright attitude sharpness (run-11: 2.0)")
    p.add_argument("--k-up-p", type=float, default=0.0, help="b_upright 'near start' sharpness (run-11: 0.1)")
    # PPO
    p.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4, help="fine-tuning LR (use ~3e-4 from scratch)")
    p.add_argument("--lr-end", type=float, default=3e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-start", type=float, default=0.0)
    p.add_argument("--ent-end", type=float, default=0.0)
    p.add_argument("--log-std-init", type=float, default=-1.6,
                   help="exploration std (log) — after BC this is the fine-tuning noise")
    p.add_argument("--pi-arch", type=int, nargs="+", default=[128, 128])
    p.add_argument("--vf-arch", type=int, nargs="+", default=[256, 256])
    p.add_argument("--symmetric", action="store_true", help="ablation: critic sees only the actor obs")
    # bookkeeping
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--eval-freq", type=int, default=200_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-wind", type=float, default=0.0,
                   help="median wind [m/s] in the periodic evaluation (model selection)")
    p.add_argument("--resume", type=str, default=None, help="path to a .zip to continue from")
    p.add_argument("--reset-log-std", type=float, default=None,
                   help="with --resume: re-inject exploration by setting log_std to this value")
    p.add_argument("--smoke", action="store_true", help="tiny run to sanity-check the pipeline")
    p.add_argument("--skip-if-done", action="store_true",
                   help="exit at once if <logdir>/<tag>_seed<seed>_*/final_model.zip exists (resumable batches)")
    args = p.parse_args()

    if args.skip_if_done:
        import glob
        done = glob.glob(os.path.join(args.logdir, f"{args.tag}_seed{args.seed}_*", "final_model.zip"))
        if done:
            print(f"[skip] {args.tag} seed {args.seed} already finished: {done[0]}")
            return

    if args.smoke:
        args.timesteps, args.n_envs, args.n_steps, args.batch_size = 4096, min(2, args.n_envs), 1024, 512
        args.eval_freq, args.eval_episodes, args.eval_seconds = 2048, 2, 4.0
        args.bc_episodes, args.bc_epochs = min(args.bc_episodes, 4), 3
    th.set_num_threads(1)

    # --- configs ---------------------------------------------------------------
    rcfg = FlipRewardConfig(s_alive=args.s_alive, gamma_shape=args.gamma_shape,
                            over_rot_slope=args.over_rot_slope, w_prog=args.w_prog,
                            alive_after_flip=not args.alive_always)
    if args.b_hold is not None:
        rcfg.b_hold = args.b_hold
    if args.b_crash is not None:
        rcfg.b_crash = args.b_crash
    rcfg.lam_pos = args.lam_pos
    rcfg.k_up_g, rcfg.k_up_p = args.k_up_g, args.k_up_p
    scfg = SuccessConfig(flip_direction=rcfg.flip_direction)
    env_kwargs = dict(action_repeat=args.action_repeat, reward_cfg=rcfg, success_cfg=scfg,
                      episode_seconds=args.episode_seconds,
                      init_steps=(0, args.init_max_steps), init_noise=args.init_noise,
                      action_mode=args.action_mode, ctbr_kwargs=dict(rate_max_rp=args.rate_max),
                      demo_prob=args.demo_prob, demo_max_steps=args.demo_max_steps,
                      wind_prob=args.wind_prob, wind_max=args.wind_max,
                      pos_integral=args.pos_integral)

    probe = make_flip_task_env(**env_kwargs)
    check_simulator(probe.env)
    n_actor, obs_cfg = probe.n_actor, probe.builder.cfg
    quad_model = model_to_dict(probe.quad_model)
    ctbr_cfg = probe.mapper.config() if probe.mapper is not None else None
    sim_dt = float(probe.base.dt)
    probe.close()

    run_name = f"{args.tag}_seed{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(args.logdir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    run_cfg = {
        "action_repeat": args.action_repeat, "n_actor": n_actor, "n_priv": N_PRIV,
        "obs_cfg": asdict(obs_cfg), "flip_direction": rcfg.flip_direction, "eps_full": rcfg.eps_full,
        "action_mode": args.action_mode, "ctbr": ctbr_cfg, "quad_model": quad_model,
        "pos_integral": bool(args.pos_integral), "sim_dt": sim_dt,
        "reward_cfg": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                       for k, v in asdict(rcfg).items()},
        "success_cfg": {k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in asdict(scfg).items()},
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    # --- model -----------------------------------------------------------------
    venv = make_vec(args.n_envs, env_kwargs, args.seed)
    policy_kwargs = dict(n_actor_obs=n_actor,
                         net_arch=dict(pi=args.pi_arch, vf=args.vf_arch),
                         activation_fn=th.nn.Tanh, log_std_init=args.log_std_init)
    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu",
                         custom_objects={"learning_rate": linear(args.lr, args.lr_end),
                                         "lr_schedule": linear(args.lr, args.lr_end)})
        model.tensorboard_log = args.logdir
        model.ent_coef = args.ent_start
        if args.reset_log_std is not None:
            with th.no_grad():
                model.policy.log_std.fill_(args.reset_log_std)
    else:
        model = PPO(AsymmetricActorCriticPolicy, venv,
                    n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
                    learning_rate=linear(args.lr, args.lr_end), gamma=args.gamma,
                    gae_lambda=args.gae_lambda, clip_range=args.clip, ent_coef=args.ent_start,
                    vf_coef=0.5, max_grad_norm=0.5, policy_kwargs=policy_kwargs,
                    seed=args.seed, verbose=0, tensorboard_log=args.logdir, device="cpu")
    if args.symmetric:
        _make_critic_symmetric(model, n_actor)

    if args.bc_episodes > 0 and not args.resume:
        if args.action_mode != "ctbr":
            raise SystemExit("--bc-episodes needs --action-mode ctbr (the expert speaks CTBR)")
        O, A, G = collect_demos(env_kwargs, args.bc_episodes, 50_000 + args.seed, args.bc_noise,
                                args.gamma, args.episode_seconds)
        bc_pretrain(model, O, A, G, args.bc_epochs, args.bc_lr, seed=args.seed)
        with th.no_grad():
            model.policy.log_std.fill_(args.log_std_init)
        model.save(os.path.join(run_dir, "bc_model.zip"))
        print("behaviour-cloned policy (before any PPO update):", flush=True)

    eval_cb = FlipEvalCallback(args.eval_freq, args.eval_episodes, run_cfg, run_dir, args.eval_seconds,
                               wind=args.eval_wind)
    if args.bc_episodes > 0 and not args.resume:
        eval_cb.model = model
        eval_cb.evaluate()                              # BC-only reference point (timestep 0)
    callbacks = [
        EntropySchedule(args.ent_start, args.ent_end, args.timesteps),
        TrainFlipStats(),
        eval_cb,
        CheckpointCallback(save_freq=max(args.timesteps // (args.n_envs * 10), 1),
                           save_path=os.path.join(run_dir, "checkpoints"), name_prefix="ppo_flip"),
    ]
    print(f"run -> {run_dir}\nactor obs {n_actor}, privileged {N_PRIV}, envs {args.n_envs}, "
          f"{args.timesteps:,} steps", flush=True)
    t0 = time.time()
    if args.timesteps > 0:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name=run_name,
                    reset_num_timesteps=not bool(args.resume))
    model.save(os.path.join(run_dir, "final_model.zip"))
    venv.close()
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {run_dir}\n"
          f"  best_model.zip  (highest §04 success in periodic evaluation)\n"
          f"  final_model.zip, config.json, eval_log.csv, checkpoints/")


class _NullLogger:
    def record(self, *a, **k): pass
    def dump(self, *a, **k): pass


def _make_critic_symmetric(model, n_actor: int) -> None:
    """Ablation helper: zero the privileged input weights of the critic's first layer
    and freeze them, so the critic effectively sees only the actor observation."""
    first = model.policy.mlp_extractor.value_net[0]
    with th.no_grad():
        first.weight[:, n_actor:] = 0.0
    mask = th.ones_like(first.weight)
    mask[:, n_actor:] = 0.0
    first.weight.register_hook(lambda g: g * mask)


if __name__ == "__main__":
    main()
