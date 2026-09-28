"""
flip_task_env.py — §03 wiring for the Quadcopter Flip task.

Composes the (unchanged) §01 environment with the §03 reward:

    QuadcopterVelocityEnv  (course fork, untouched)
      └─ ActionRepeat       (§01, via make_flip_env(...).env — same config as §01/§02)
           └─ FlipTaskEnv   (this file)
                · installs FlipReward as the env's reward_fn
                · policy observation = [ actor | privileged ]
                    actor      = ActorObsBuilder(raw)  (transform_observation ++ flip features)
                                 — the SAME builder the deployed controller uses
                    privileged = noise-free state, reward phase/angle, time left (critic only)
                · optional randomised start (a few noisy hover decisions before the
                  episode "starts") so the policy does not memorise one trajectory
                · crash penalty also on integrator failure (ActionRepeat swallows it)
                · action_mode "ctbr" (default): the policy outputs collective thrust + body
                  rates, a fixed P rate loop + the §02 mixer turn them into motor commands;
                  "motors": the policy outputs the 4 motor commands directly
                · demonstration starts (demo_prob): the §02 scripted flip drives the first
                  U[1, demo_max_steps] decisions of some training episodes, so the policy
                  also starts mid-flip (reverse curriculum). Evaluation always starts at hover.
                · wind randomisation (wind_prob, wind_max) via the env's own enable_random_wind
                · an independent SuccessDetector (never read by the reward) reports
                  info["flip_success"] for training-time logging

Usage
-----
    from flip_task_env import make_flip_task_env
    env = make_flip_task_env(action_repeat=4)
"""
from __future__ import annotations

import os
import random
import sys
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- make §01 (flip_env) and the fork sim (quad_velocity_env) importable ----- #
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.environ.get("QUAD_SIM_DIR"), os.path.join(_HERE, "..", "Environment_Task_Wrapper"), _HERE):
    if _cand:
        _cand = os.path.abspath(_cand)
        if os.path.isdir(_cand) and _cand not in sys.path:
            sys.path.insert(0, _cand)

from flip_env import make_flip_env, ObsConfig, quat_to_rotation_matrix          # noqa: E402
from flip_reward import FlipReward, FlipRewardConfig, SuccessDetector, SuccessConfig  # noqa: E402
from flip_policy import ActorObsBuilder, CTBRMapper                              # noqa: E402
from baselines import QuadModel, ScriptedFlipController, FlipScript, control_dt_from_env  # noqa: E402

N_PRIV = 26   # see FlipTaskEnv._privileged


class FlipTaskEnv(gym.Wrapper):
    """Wraps the RAW-observation action-repeat env (make_flip_env(...).env)."""

    def __init__(self, raw_env: gym.Env, obs_cfg: ObsConfig, reward: FlipReward,
                 success_cfg: Optional[SuccessConfig] = None,
                 init_steps: Tuple[int, int] = (0, 0), init_noise: float = 0.0,
                 action_mode: str = "ctbr", ctbr_kwargs: Optional[dict] = None,
                 demo_prob: float = 0.0, demo_max_steps: int = 45,
                 wind_prob: float = 0.0, wind_max: float = 0.0, pos_integral: bool = False):
        super().__init__(raw_env)
        # Wind domain randomisation through the env's OWN API (enable_random_wind):
        # with probability wind_prob an episode gets the fork's RANDOMSINE wind with a
        # median speed U[0, wind_max] m/s, any heading, +-10 deg elevation.
        self.wind_prob, self.wind_max = float(wind_prob), float(wind_max)
        self.wind_speed = 0.0
        # Demonstration starts (reverse curriculum): with probability demo_prob the
        # §02 scripted flip drives the vehicle for U[1, demo_max_steps] decisions and
        # the policy takes over mid-manoeuvre. The env is untouched (the start state
        # is produced by actions only); the reward/tracker keep the rotation already
        # made, but nothing earned during the demo is given to the policy.
        self.demo_prob = float(demo_prob)
        self.demo_max_steps = int(demo_max_steps)
        self._demo = None
        if action_mode not in ("ctbr", "motors"):
            raise ValueError(action_mode)
        self.action_mode = action_mode
        self.quad_model = QuadModel.from_env(raw_env)
        self.mapper = CTBRMapper(self.quad_model, **(ctbr_kwargs or {})) if action_mode == "ctbr" else None
        self._last_raw = None
        if self.demo_prob > 0:
            d = reward.cfg.flip_direction
            self._demo = ScriptedFlipController(self.quad_model, control_dt_from_env(raw_env),
                                                FlipScript(settle_time=0.0, direction=int(d)))
        self.reward = reward
        self.base = raw_env.unwrapped
        d, eps = reward.cfg.flip_direction, reward.cfg.eps_full
        self.builder = ActorObsBuilder(obs_cfg, d, eps, pos_integral=pos_integral,
                                       dt=control_dt_from_env(raw_env))
        self.detector = SuccessDetector(success_cfg or SuccessConfig(flip_direction=d))
        self.init_steps = (int(init_steps[0]), int(init_steps[1]))
        self.init_noise = float(init_noise)
        b = self.base
        self.hover_action = 2.0 * (b.hover_w - b.min_w) / (b.max_w - b.min_w) - 1.0
        self.n_actor = self.builder.dim
        self.n_priv = N_PRIV
        lim = obs_cfg.clip
        self.observation_space = spaces.Box(-lim, lim, (self.n_actor + self.n_priv,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    # -- privileged (critic-only) state ---------------------------------------
    def _privileged(self) -> np.ndarray:
        q = self.base.quad
        cfg = self.builder.cfg
        motors = 2.0 * (np.asarray(q.wMotor) - cfg.motor_min) / (cfg.motor_max - cfg.motor_min) - 1.0
        rw = self.reward
        t_frac = self.base.steps / max(1, self.base.max_steps)
        v = np.concatenate([
            np.asarray(q.pos) / cfg.pos_scale,                               # 3
            quat_to_rotation_matrix(np.asarray(q.quat)).reshape(-1),         # 9
            np.asarray(q.vel) / cfg.vel_scale,                               # 3
            np.asarray(q.omega) / cfg.rate_scale,                            # 3
            motors,                                                          # 4
            [np.clip(rw.alpha / (2 * np.pi), -2, 2), 1.0 if rw.phase == "RECOVER" else 0.0,
             1.0 if rw.passed_inverted else 0.0, t_frac],                     # 4
        ]).astype(np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=cfg.clip, neginf=-cfg.clip)
        return np.clip(v, -cfg.clip, cfg.clip)

    def _obs(self, raw) -> np.ndarray:
        return np.concatenate([self.builder.build(raw), self._privileged()]).astype(np.float32)

    # -- gym API --------------------------------------------------------------
    def _demo_start(self, raw):
        """Run the scripted flip for a random number of decisions; returns (raw, ok)."""
        n = int(self.np_random.integers(1, self.demo_max_steps + 1))
        self._demo.reset()
        for _ in range(n):
            raw, _, te, tr, _ = self.env.step(self._demo.act(raw))
            self.builder.build(raw)                       # tracker follows the manoeuvre
            self.detector.update(None, env=self.base)
            if te or tr:
                return raw, False
        self.reward._prev_action = None                   # smoothness restarts at hand-off
        self.reward.smooth_action = None
        return raw, True

    def _sample_wind(self) -> None:
        """Called right after env.reset (t = 0, nothing simulated yet)."""
        if self.wind_prob <= 0 or self.wind_max <= 0:
            return
        on = self.np_random.random() < self.wind_prob
        speed = float(self.np_random.uniform(0.0, self.wind_max)) if on else 0.0
        # the fork's wind model draws from Python's global RNG: seed it from ours
        random.seed(int(self.np_random.integers(2 ** 31)))
        self.base.enable_random_wind(on and speed > 0, magnitude=max(speed, 1e-3),
                                     heading_deg=180.0, elevation_deg=10.0)
        self.wind_speed = speed

    def reset(self, *, seed=None, options=None):
        raw, info = self.env.reset(seed=seed, options=options)
        self._sample_wind()
        if self._demo is not None and self.np_random.random() < self.demo_prob:
            self.reward.reset()
            self.builder.reset()
            self.detector.reset()
            self.builder.build(raw)
            self.detector.update(None, env=self.base)
            raw, ok = self._demo_start(raw)
            if ok:
                self.demo_episode = True
                self._last_raw = np.asarray(raw, dtype=np.float64)
                return self._obs(raw), info
            raw, info = self.env.reset()
        self.demo_episode = False
        last_a = None
        lo, hi = self.init_steps
        n = int(self.np_random.integers(lo, hi + 1)) if hi > 0 else 0
        for _ in range(n):   # randomised start: noisy hover; none of it is rewarded or counted
            a = np.clip(self.hover_action + self.np_random.normal(0.0, self.init_noise, 4), -1, 1)
            last_a = a.astype(np.float32)
            raw, _, te, tr, _ = self.env.step(last_a)
            if te or tr:
                raw, info = self.env.reset()
                last_a = None
                break
        self.reward.reset()
        self.builder.reset(last_action=None if self.mapper is not None else last_a)
        self._last_raw = np.asarray(raw, dtype=np.float64)
        self.detector.reset()
        self.detector.update(None, env=self.base)
        return self._obs(raw), info

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        self.builder.set_action(a)
        motor_a = self.mapper.motor_action(a, self._last_raw[13:16]) if self.mapper is not None else a
        if self.mapper is not None:
            self.reward.smooth_action = a       # smoothness on the policy's command
        raw, r, terminated, truncated, info = self.env.step(motor_a)
        self._last_raw = np.asarray(raw, dtype=np.float64)
        if "integrator_error" in info and not self.reward._done_latched:
            r -= self.reward.cfg.b_crash           # the env could not even integrate: a crash
            self.reward._done_latched = True
        ok = self.detector.update(None, env=self.base)   # true state, independent of the reward
        obs = self._obs(raw)
        info = dict(info)
        info.update(flip_success=float(ok and not terminated),
                    flip_rev=float(self.builder.tracker.rev),
                    flip_inverted=float(self.reward.passed_inverted),
                    crashed=float(terminated))
        return obs, float(r), terminated, truncated, info


# --------------------------------------------------------------------------- #
# Factories                                                                    #
# --------------------------------------------------------------------------- #
def make_flip_task_env(action_repeat: int = 4,
                       obs_cfg: Optional[ObsConfig] = None,
                       reward_cfg: Optional[FlipRewardConfig] = None,
                       success_cfg: Optional[SuccessConfig] = None,
                       max_abs_z: float = 5.0,
                       episode_seconds: float = 10.0,
                       init_steps: Tuple[int, int] = (0, 0),
                       init_noise: float = 0.0,
                       action_mode: str = "ctbr",
                       ctbr_kwargs: Optional[dict] = None,
                       demo_prob: float = 0.0,
                       demo_max_steps: int = 45,
                       wind_prob: float = 0.0,
                       wind_max: float = 0.0,
                       pos_integral: bool = False,
                       seed: Optional[int] = None) -> FlipTaskEnv:
    """One fully-wired flip task env (one reward + one detector instance per env)."""
    reward = FlipReward(reward_cfg or FlipRewardConfig())
    flip = make_flip_env(action_repeat=action_repeat, obs_cfg=obs_cfg or ObsConfig(),
                         reward_fn=reward, max_abs_z=max_abs_z,
                         episode_seconds=episode_seconds, seed=None)
    env = FlipTaskEnv(flip.env, flip.cfg, reward, success_cfg, init_steps, init_noise,
                      action_mode, ctbr_kwargs, demo_prob, demo_max_steps, wind_prob, wind_max,
                      pos_integral)
    env.unwrapped_reward = reward          # handle for logging
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_vec_task_env(n_envs: int = 8, asynchronous: bool = True, **kwargs):
    """Parallel gymnasium vector env; each worker gets its own reward instance."""
    def _thunk():
        return make_flip_task_env(**kwargs)
    if asynchronous and n_envs > 1:
        return gym.vector.AsyncVectorEnv([_thunk for _ in range(n_envs)])
    return gym.vector.SyncVectorEnv([_thunk for _ in range(n_envs)])


def make_success_detector(**overrides) -> SuccessDetector:
    """An independent success detector for evaluation (§04)."""
    return SuccessDetector(SuccessConfig(**overrides)) if overrides else SuccessDetector()
