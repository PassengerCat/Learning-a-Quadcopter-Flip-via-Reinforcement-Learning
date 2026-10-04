"""MTR-PPO: training environment.

The PPO track's FlipTaskEnv (unchanged: action-repeat 4, CTBR or direct motor actions,
asymmetric critic with privileged state, independent success detector for logging), with
  * the multiplicative tracking reward (mtr_reward.MultiplicativeTrackingReward) as reward_fn;
  * the tracking actor observation (mtr_observation.TrackingObsBuilder, 25 values) instead of the PPO
    track's actor observation; the critic's privileged part (26 values) is unchanged;
  * per-episode command randomisation: omega ~ U[omega_range] (as GEAR randomises the command);
  * initial-state randomisation with a curriculum (as in GEAR, the range "expands progressively"):
    tilt <= scale * max_tilt_deg about a random horizontal axis, body rates and velocity
    uniform in +-scale * max_rate / max_vel per axis. `set_curriculum(scale)` sets scale in
    [0, 1]; the trainer raises it during training. Evaluation always starts at nominal hover;
  * optional starts ON the reference (reference state initialisation): with probability
    ref_start_prob an episode starts at loop phase theta ~ U[0, 2*pi - eps) in exactly the
    reference state (position, attitude, velocity omega*r, body rates omega), with the loop
    anchored at the nominal start position and the completed phase counted as revolution.
    No expert actions are involved. ref_start_prob = 0 reproduces the hover-start environment
    exactly (the random stream is unchanged).
  * optional termination on turning back (backtrack_deg > 0): during FLIP, the episode ends
    when the revolution falls more than backtrack_deg below the largest revolution reached
    (rev_max - rev > backtrack_deg). The reward is positive, so ending the episode forfeits the
    rest of it: swinging forward and back through the lower half of the loop (run 3) stops
    paying. It applies during FLIP only; backtrack_deg = 0 (default) switches it off and
    reproduces the environment of runs 1-3 exactly.
  * optional post-flip survival bonus (alive_bonus, per second; see mtr_reward), optionally
    weighted by speed (alive_speed_k). 0 = off.
  * optional pitch-axis reward term (axis_term; see mtr_reward). Off by default.
  * optional pitch-axis penalty (axis_limit_deg, axis_penalty; see mtr_reward): a one-off
    penalty the first time the flip leaves the pitch axis. A reward change only; it terminates nothing.
No demonstration starts, no noisy-hover init steps, no wind.

    env = make_mtr_env(action_mode="ctbr")      # or "motors"
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np
from gymnasium import spaces

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_RT = os.path.join(_REPO, "Reward_and_Training")   # the PPO track's modules
_SIM = os.environ.get("QUAD_SIM_DIR", os.path.join(_RT, "Quadcopter_SimCon", "Simulation"))
for _p in (_HERE, _RT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if os.path.isdir(_SIM) and _SIM not in sys.path:
    sys.path.append(_SIM)                 # the simulator's own packages (quadFiles, utils)

from flip_env import ObsConfig, make_flip_env                                  # noqa: E402
from flip_reward import SuccessConfig                                          # noqa: E402
from flip_task_env import FlipTaskEnv, N_PRIV                                  # noqa: E402
from mtr_observation import TrackingObsBuilder                                         # noqa: E402
from mtr_reference import FLIP, FlipCommand, LoopReference, yaw                         # noqa: E402
from mtr_reward import MTRRewardConfig, MultiplicativeTrackingReward               # noqa: E402


def perturbed_start(rng: np.random.Generator, scale: float, max_tilt_deg: float,
                    max_rate: float, max_vel: float) -> dict:
    """Random start: tilt about a random horizontal axis, uniform rates and velocity."""
    axis = rng.uniform(0.0, 2.0 * np.pi)
    tilt = np.deg2rad(rng.uniform(0.0, scale * max_tilt_deg))
    quat = np.array([np.cos(tilt / 2), np.sin(tilt / 2) * np.cos(axis), np.sin(tilt / 2) * np.sin(axis), 0.0])
    vel = rng.uniform(-scale * max_vel, scale * max_vel, 3)
    omega = rng.uniform(-scale * max_rate, scale * max_rate, 3)
    return dict(quat=quat, vel=vel, omega=omega)


def apply_start(base_env, start: dict) -> np.ndarray:
    """Overwrite the freshly reset vehicle state and restart the integrator from it
    (as the evaluation harness does for its perturbed-start condition)."""
    q = base_env.quad
    if "pos" in start:
        q.state[0:3] = start["pos"]
    q.state[3:7], q.state[7:10], q.state[10:13] = start["quat"], start["vel"], start["omega"]
    q.pos, q.quat, q.vel, q.omega = q.state[0:3], q.state[3:7], q.state[7:10], q.state[10:13]
    q.extended_state()
    q.integrator.set_initial_value(q.state, base_env.t)
    return base_env._get_obs()


class MTRFlipEnv(FlipTaskEnv):
    def __init__(self, raw_env, obs_cfg: ObsConfig, reward: MultiplicativeTrackingReward,
                 success_cfg: Optional[SuccessConfig] = None, action_mode: str = "ctbr",
                 ctbr_kwargs: Optional[dict] = None, omega_range: Tuple[float, float] = (4.5, 5.5),
                 radius: float = 0.6, max_tilt_deg: float = 10.0, max_rate: float = 1.0,
                 max_vel: float = 0.5, curriculum: float = 0.0, ref_start_prob: float = 0.0,
                 backtrack_deg: float = 0.0):
        super().__init__(raw_env, obs_cfg, reward, success_cfg, init_steps=(0, 0), init_noise=0.0,
                         action_mode=action_mode, ctbr_kwargs=ctbr_kwargs, demo_prob=0.0)
        lo, hi = map(float, omega_range)
        if not 0.0 < lo <= hi:
            raise ValueError(f"invalid omega_range {omega_range}")
        self.omega_range, self.radius = (lo, hi), float(radius)
        self.max_tilt_deg, self.max_rate, self.max_vel = float(max_tilt_deg), float(max_rate), float(max_vel)
        self.set_curriculum(curriculum)
        if not 0.0 <= float(ref_start_prob) <= 1.0:
            raise ValueError(f"ref_start_prob must be in [0, 1], got {ref_start_prob}")
        self.ref_start_prob = float(ref_start_prob)
        self.ref_start = False
        if not 0.0 <= float(backtrack_deg) < 360.0:
            raise ValueError(f"backtrack_deg must be in [0, 360), got {backtrack_deg}")
        self.backtrack_rad = float(np.deg2rad(backtrack_deg))
        d = reward.cfg.flip_direction
        self.mtr_obs = TrackingObsBuilder(d, reward.cfg.eps_full)
        self.n_actor = self.mtr_obs.dim
        lim = obs_cfg.clip
        self.observation_space = spaces.Box(-lim, lim, (self.n_actor + self.n_priv,), np.float32)
        self.command = FlipCommand(omega=lo, radius=self.radius, direction=d)

    def set_curriculum(self, scale: float) -> None:
        self.curriculum = float(np.clip(scale, 0.0, 1.0))

    def _obs(self, raw) -> np.ndarray:
        self.builder.build(raw)                       # keeps the PPO track's tracker (info) current
        return np.concatenate([self.mtr_obs.build(raw), self._privileged()]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        rng = self.np_random
        self.command = FlipCommand(omega=float(rng.uniform(*self.omega_range)), radius=self.radius,
                                   direction=self.reward.cfg.flip_direction)
        self.ref_start = self.ref_start_prob > 0.0 and rng.random() < self.ref_start_prob
        anchor, rev0 = None, 0.0
        if self.ref_start:
            q = self.base.quad
            anchor = (np.array(q.pos, dtype=float), yaw(q.quat))
            rev0 = float(rng.uniform(0.0, 2.0 * np.pi - self.reward.cfg.eps_full))
            ref = LoopReference(self.command)
            ref.anchor(*anchor)
            start = ref.state_at(rev0)
        else:
            start = perturbed_start(rng, self.curriculum, self.max_tilt_deg, self.max_rate, self.max_vel)
        raw = apply_start(self.base, start)
        q = self.base.quad
        self.reward.reset(self.command)
        self.reward.begin(q.pos, q.quat, rev0, anchor)
        self.builder.reset()
        self.builder.tracker.rev = rev0               # the PPO track's tracker (logging only)
        self.detector.reset()
        self.detector.update(None, env=self.base)
        self.mtr_obs.reset(self.command, anchor, rev0)
        self._last_raw = np.asarray(raw, dtype=np.float64)
        return self._obs(raw), {}

    def step(self, action):
        self.mtr_obs.set_action(action)
        obs, r, terminated, truncated, info = super().step(action)
        rw = self.reward
        backtrack = (self.backtrack_rad > 0.0 and not terminated and rw.task == FLIP
                     and rw.rev_max - rw.revolution > self.backtrack_rad)
        if backtrack:
            terminated = True
        info.update(mtr_task=float(self.mtr_obs.task), mtr_omega=float(self.command.omega),
                    mtr_ref_start=float(self.ref_start), mtr_backtrack=float(backtrack),
                    mtr_axis_violation=float(rw.axis_violated))
        return obs, r, terminated, truncated, info


def make_mtr_env(action_mode: str = "ctbr", action_repeat: int = 4, episode_seconds: float = 6.0,
                  omega_range: Tuple[float, float] = (4.5, 5.5), radius: float = 0.6,
                  max_tilt_deg: float = 10.0, max_rate: float = 1.0, max_vel: float = 0.5,
                  curriculum: float = 0.0, rate_max: float = 20.0, max_abs_z: float = 5.0,
                  ref_start_prob: float = 0.0, kernels: str = "narrow", backtrack_deg: float = 0.0,
                  alive_bonus: float = 0.0, alive_speed_k: float = 0.0, axis_term: bool = False,
                  axis_limit_deg: float = 0.0, axis_penalty: float = 2.0,
                  seed: Optional[int] = None) -> MTRFlipEnv:
    """One fully wired environment (one reward and one detector instance per env)."""
    if not 0.0 <= float(alive_bonus) < np.inf:
        raise ValueError(f"alive_bonus must be finite and >= 0, got {alive_bonus}")
    if not 0.0 <= float(alive_speed_k) < np.inf:
        raise ValueError(f"alive_speed_k must be finite and >= 0, got {alive_speed_k}")
    if not 0.0 <= float(axis_limit_deg) < 180.0:
        raise ValueError(f"axis_limit_deg must be in [0, 180), got {axis_limit_deg}")
    if not 0.0 <= float(axis_penalty) < np.inf:
        raise ValueError(f"axis_penalty must be finite and >= 0, got {axis_penalty}")
    reward = MultiplicativeTrackingReward(MTRRewardConfig(kernels=kernels, b_alive_hover=float(alive_bonus),
                                                 alive_speed_k=float(alive_speed_k),
                                                 axis_term=bool(axis_term),
                                                 axis_limit_deg=float(axis_limit_deg),
                                                 axis_penalty=float(axis_penalty)))
    flip = make_flip_env(action_repeat=action_repeat, obs_cfg=ObsConfig(), reward_fn=reward,
                         max_abs_z=max_abs_z, episode_seconds=episode_seconds, seed=None)
    env = MTRFlipEnv(flip.env, flip.cfg, reward, SuccessConfig(flip_direction=reward.cfg.flip_direction),
                          action_mode=action_mode, ctbr_kwargs=dict(rate_max_rp=rate_max),
                          omega_range=omega_range, radius=radius, max_tilt_deg=max_tilt_deg,
                          max_rate=max_rate, max_vel=max_vel, curriculum=curriculum,
                          ref_start_prob=ref_start_prob, backtrack_deg=backtrack_deg)
    if seed is not None:
        env.reset(seed=seed)
    return env


__all__ = ["MTRFlipEnv", "make_mtr_env", "N_PRIV", "perturbed_start", "apply_start"]
