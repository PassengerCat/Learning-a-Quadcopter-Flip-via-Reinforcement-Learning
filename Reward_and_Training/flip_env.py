"""
flip_env.py — Environment & Task Wrapper for the Quadcopter Flip project (Section 1).

Turns the velocity-tracking QuadcopterVelocityEnv into a *flip* task:

  1. Disables ONLY the roll/pitch termination (max_tilt_rad = inf), keeping the
     altitude, non-finite, and max-duration terminations required by the brief.
  2. Provides a custom reward hook (reward_fn) — the actual flip reward is
     designed in Section 3; here we install a neutral placeholder.
  3. Action-repeat (frame-skip): the policy decides at dt*k seconds instead of
     every sim step, summing reward over the k sub-steps. Guards the sim's
     integrator, which can raise under wild exploratory actions.
  4. Observation transform with FIXED physical scales + clipping, and a choice of
     attitude representation (projected gravity / quaternion / rotation matrix).
     The SAME transform (transform_observation) must be used at deployment inside
     Controller.act — it is a pure function of the raw observation.

Requires: numpy<2, scipy<1.13, gymnasium. (The bobzwik dynamics break on numpy 2.x.)
Must be importable next to quad_velocity_env.py (same folder or on PYTHONPATH).

Empirically verified against the env (seed-checked):
    motor speeds in [75, 925], hover ~523; initial state = origin, level
    (quat = [1,0,0,0], scalar-first, body->world, NED), zero vel/omega, hover motors.
    Projected gravity at upright = [0,0,+1]; fully inverted = [0,0,-1].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from quad_velocity_env import QuadcopterVelocityEnv


# --------------------------------------------------------------------------- #
# Attitude helpers (quaternion is scalar-first [w,x,y,z], body->world, NED)    #
# In NED, world "down" is +z, so gravity direction in world = [0,0,1].         #
# --------------------------------------------------------------------------- #
def quat_to_projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Gravity (world down = [0,0,1] in NED) expressed in the body frame.
    Upright -> [0,0,+1]; fully inverted -> [0,0,-1]. Loses only yaw (irrelevant
    to the flip), giving a continuous, singularity-free 'how inverted am I' signal."""
    w, x, y, z = quat
    return np.array([
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ], dtype=np.float32)


def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Full body->world rotation matrix (9 numbers), continuous & complete."""
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Observation configuration & the pure, deployable transform                  #
# --------------------------------------------------------------------------- #
@dataclass
class ObsConfig:
    attitude: str = "gravity"            # "gravity" (3) | "quat" (4) | "rotmat" (9)
    include_prev_action: bool = True     # append last decision (4); pairs w/ action-rate penalty
    pos_scale: float = 5.0               # ~max_abs_z; position ~O(meters)
    vel_scale: float = 10.0              # linear velocity scale (m/s)
    rate_scale: float = 20.0            # body-rate scale (rad/s) — KEEP magnitude (flip needs it)
    clip: float = 10.0                   # post-normalisation clip
    # Motor normalisation uses the env's known [min_w, max_w] (set by make_flip_env).
    motor_min: float = 75.0
    motor_max: float = 925.0

    def obs_dim(self) -> int:
        att = {"gravity": 3, "quat": 4, "rotmat": 9}[self.attitude]
        # pos(3) + attitude + vel(3) + rates(3) + motors(4) [+ prev_action(4)]
        return 3 + att + 3 + 3 + 4 + (4 if self.include_prev_action else 0)


# Indices into the raw 20-dim "full" observation of QuadcopterVelocityEnv:
#   [0:3] target_vel (dropped) | [3:6] pos | [6:10] quat | [10:13] lin_vel
#   [13:16] body rates | [16:20] motor speeds
def transform_observation(raw_obs: np.ndarray,
                          prev_action: np.ndarray,
                          cfg: ObsConfig) -> np.ndarray:
    """Pure function: raw env observation -> normalised policy observation.
    Use this identical function in training AND in Controller.act at deployment."""
    raw = np.asarray(raw_obs, dtype=np.float32)
    pos   = raw[3:6]
    quat  = raw[6:10]
    vel   = raw[10:13]
    rates = raw[13:16]
    motor = raw[16:20]

    if cfg.attitude == "gravity":
        att = quat_to_projected_gravity(quat)
    elif cfg.attitude == "quat":
        att = quat.copy()
    elif cfg.attitude == "rotmat":
        att = quat_to_rotation_matrix(quat).reshape(-1)
    else:
        raise ValueError(f"unknown attitude representation: {cfg.attitude}")

    motor_norm = 2.0 * (motor - cfg.motor_min) / (cfg.motor_max - cfg.motor_min) - 1.0

    parts = [
        pos / cfg.pos_scale,
        att,                          # already O(1)
        vel / cfg.vel_scale,
        rates / cfg.rate_scale,       # magnitude preserved
        motor_norm,
    ]
    if cfg.include_prev_action:
        parts.append(np.asarray(prev_action, dtype=np.float32))

    obs = np.concatenate(parts).astype(np.float32)
    return np.clip(obs, -cfg.clip, cfg.clip) # Clipping here might be too aggressive but well'see :P


# --------------------------------------------------------------------------- #
# Placeholder reward — the real flip reward is designed in Section 3.          #
# Signature required by the env: (target_velocity, full_state, action, env).  #
# --------------------------------------------------------------------------- #
def placeholder_reward(target_velocity, full_state, action, env) -> float: # Afto einai gia meta !
    """Neutral reward so we never accidentally use the env's default (which
    penalises pitch and fights the flip). REPLACE in Section 3."""
    return 0.0


# --------------------------------------------------------------------------- #
# Action-repeat wrapper (with integrator guard)                               #
# --------------------------------------------------------------------------- #
class ActionRepeat(gym.Wrapper):
    """Hold each policy action for k sim steps; sum reward; stop early on
    termination/truncation. Guards the sim integrator, which can raise on wild
    actions early in training — treated as a terminal 'unstable' step."""

    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        assert k >= 1
        self.k = int(k)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        obs = None
        info: dict = {}
        for _ in range(self.k):
            try:
                obs, reward, terminated, truncated, info = self.env.step(action)
            except Exception as e:                       # integrator divergence
                if obs is None:
                    obs = self.observation_space.sample() * 0.0
                info = {**info, "integrator_error": repr(e)}
                terminated = True
                break
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


# --------------------------------------------------------------------------- #
# Observation wrapper (tracks previous *decision* action)                     #
# --------------------------------------------------------------------------- #
class FlipObservation(gym.Wrapper):
    """Applies transform_observation and tracks the previous policy action at
    decision granularity. Sits OUTSIDE ActionRepeat so 'previous action' is the
    previous decision."""

    def __init__(self, env: gym.Env, cfg: ObsConfig):
        super().__init__(env)
        self.cfg = cfg
        self._prev_action = np.zeros(4, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-cfg.clip, high=cfg.clip, shape=(cfg.obs_dim(),), dtype=np.float32
        )

    def reset(self, **kwargs):
        raw, info = self.env.reset(**kwargs)
        self._prev_action = np.zeros(4, dtype=np.float32)
        return transform_observation(raw, self._prev_action, self.cfg), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        raw, reward, terminated, truncated, info = self.env.step(action)
        obs = transform_observation(raw, self._prev_action, self.cfg)
        self._prev_action = action
        return obs, reward, terminated, truncated, info


# --------------------------------------------------------------------------- #
# Factories                                                                   #
# --------------------------------------------------------------------------- #
def make_flip_env(action_repeat: int = 4,
                  obs_cfg: Optional[ObsConfig] = None,
                  reward_fn=placeholder_reward,
                  max_abs_z: float = 5.0,
                  episode_seconds: float = 10.0,
                  seed: Optional[int] = None) -> gym.Env:
    """Build one flip environment: flip config + action-repeat + obs transform.

    Flip config: tilt termination OFF (max_tilt_rad=inf) while keeping altitude,
    non-finite and duration terminations; target velocity fixed to zero."""
    base = QuadcopterVelocityEnv(
        obs_mode="full",
        target_velocity=(0.0, 0.0, 0.0), # Target velocity being fixed to zero could be helpful for the flip task, but it could also hurt us. KEEP IN MIND
        terminate_on_unstable=True,     # keep non-finite + altitude termination
        max_tilt_rad=np.inf,            # disable ONLY the roll/pitch gate
        max_abs_z=max_abs_z,
        episode_seconds=episode_seconds,
        reward_fn=reward_fn,
    )
    cfg = obs_cfg or ObsConfig()
    # Pin motor normalisation to this env's true motor limits.
    cfg.motor_min = float(base.min_w)
    cfg.motor_max = float(base.max_w)

    env = ActionRepeat(base, k=action_repeat)
    env = FlipObservation(env, cfg)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_vec_env(n_envs: int = 8, asynchronous: bool = True, **kwargs): # saving time, keeping us sane :)
    """Parallel envs (sim is CPU-bound Python). Library-agnostic: returns a
    gymnasium vector env, usable by a from-scratch loop or wrapped for SB3."""
    def _thunk():
        return make_flip_env(**kwargs)
    if asynchronous and n_envs > 1:
        return gym.vector.AsyncVectorEnv([_thunk for _ in range(n_envs)])
    return gym.vector.SyncVectorEnv([_thunk for _ in range(n_envs)])
