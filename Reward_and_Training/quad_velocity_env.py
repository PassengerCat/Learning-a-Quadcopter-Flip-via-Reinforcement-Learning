# Simulation/quad_velocity_env.py
"""
Gymnasium-compatible velocity-tracking environment for bobzwik/Quadcopter_SimCon.

Usage from inside Simulation/:
    import gymnasium as gym
    from quad_velocity_env import QuadcopterVelocityEnv

    env = QuadcopterVelocityEnv(obs_mode="full")
    obs, info = env.reset(options={"target_velocity": [1.0, 0.0, 0.0]})
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Observation modes:
    "full":
        target vx, vy, vyaw
        position x, y, z
        quaternion q0, q1, q2, q3
        linear velocity vx, vy, vz
        body rates p, q, r
        motor speeds wM1..wM4

    "realistic":
        target vx, vy, vyaw
        z
        quaternion q0, q1, q2, q3
        body rates p, q, r
        linear velocity vx, vy, vz
        motor speeds wM1..wM4

Noisy observations are applied after constructing the observation.
Actions are normalized motor commands in [-1, 1]^4 and mapped to motor speed commands.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Make this robust whether imported from Simulation/ or project root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from quadFiles.quad import Quadcopter
from utils.windModel import Wind
import config


@dataclass
class RewardWeights:
    vel_xy: float = 1.0
    yaw_rate: float = 0.35
    z: float = 0.10
    roll_pitch: float = 0.05
    ang_rate: float = 0.01
    action: float = 0.002
    action_rate: float = 0.002
    alive: float = 0.02


class QuadcopterVelocityEnv(gym.Env):
    """
    Velocity-following task.

    Target command:
        target_velocity = [vx_target, vy_target, vyaw_target]

    The reward penalizes:
        horizontal velocity error
        yaw-rate error
        altitude drift
        excessive roll/pitch
        excessive angular rates
        action magnitude and action slew

    The environment itself gives the policy motor commands, not high-level velocity commands.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        obs_mode: str = "full",
        dt: float = 0.005,
        episode_seconds: float = 10.0,
        orient: str = "NED",
        observation_noise_std: Optional[Dict[str, float]] = None,
        random_wind: bool = False,
        wind_magnitude: float = 2.0,
        wind_heading_deg: float = 0.0,
        wind_elevation_deg: float = 0.0,
        wind_type: str = "RANDOMSINE",
        target_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        max_target_xy_speed: float = 5.0,
        max_target_yaw_rate: float = np.deg2rad(150.0),
        reward_weights: RewardWeights = RewardWeights(),
        reward_fn: Optional[
            Callable[[np.ndarray, np.ndarray, np.ndarray, "QuadcopterVelocityEnv"], float]
        ] = None,
        terminate_on_unstable: bool = True,
        max_abs_z: float = 5.0,
        max_tilt_rad: float = np.deg2rad(80.0),
    ):
        super().__init__()

        assert obs_mode in {"full", "realistic"}
        assert orient in {"NED", "ENU"}

        # The repo uses config.orient globally inside the dynamics.
        config.orient = orient

        self.obs_mode = obs_mode
        self.dt = float(dt)
        self.episode_seconds = float(episode_seconds)
        self.max_steps = int(np.ceil(self.episode_seconds / self.dt))
        self.target_velocity = np.asarray(target_velocity, dtype=np.float32)
        self.max_target_xy_speed = float(max_target_xy_speed)
        self.max_target_yaw_rate = float(max_target_yaw_rate)
        self.reward_weights = reward_weights
        self.reward_fn = reward_fn

        self.random_wind = bool(random_wind)
        self.wind_magnitude = float(wind_magnitude)
        self.wind_heading_deg = float(wind_heading_deg)
        self.wind_elevation_deg = float(wind_elevation_deg)
        self.wind_type = wind_type.upper()

        self.terminate_on_unstable = bool(terminate_on_unstable)
        self.max_abs_z = float(max_abs_z)
        self.max_tilt_rad = float(max_tilt_rad)

        self.quad: Optional[Quadcopter] = None
        self.wind: Optional[Wind] = None
        self.t = 0.0
        self.steps = 0
        self.prev_action = np.zeros(4, dtype=np.float32)

        # Noise defaults are intentionally modest. Set values to 0.0 to disable per group.
        self.noise_std = {
            "target": 0.0,
            "pos": 0.002,
            "z": 0.002,
            "quat": 0.001,
            "vel": 0.02,
            "omega": 0.005,
            "motor": 1.0,
        }
        if observation_noise_std is not None:
            self.noise_std.update(observation_noise_std)

        # Temporary quad to discover parameters and observation size.
        q = Quadcopter(0.0)
        self.min_w = float(q.params["minWmotor"])
        self.max_w = float(q.params["maxWmotor"])
        self.hover_w = float(q.params["w_hover"])

        # Normalized motor speed command action.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        obs_dim = self._obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _obs_dim(self) -> int:
        if self.obs_mode == "full":
            # target 3 + pos 3 + quat 4 + vel 3 + omega 3 + motors 4
            return 20
        # target 3 + z 1 + quat 4 + omega 3 + vel 3 + motors 4
        return 18

    def _make_wind(self) -> Wind:
        if not self.random_wind or self.wind_magnitude <= 0.0:
            return Wind("NONE")

        if self.wind_type == "FIXED":
            # magnitude, heading deg, elevation deg
            return Wind("FIXED", self.wind_magnitude, self.wind_heading_deg, self.wind_elevation_deg)

        if self.wind_type == "SINE":
            return Wind("SINE", self.wind_magnitude, self.wind_heading_deg, self.wind_elevation_deg)

        # RANDOMSINE arguments:
        # velW_max, velW_min, qW1_max, qW1_min, qW2_max, qW2_min
        return Wind(
            "RANDOMSINE",
            self.wind_magnitude,
            0.0,
            self.wind_heading_deg,
            -self.wind_heading_deg,
            self.wind_elevation_deg,
            -self.wind_elevation_deg,
        )

    def set_target_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        target = np.asarray([vx, vy, vyaw], dtype=np.float32)
        target[:2] = np.clip(target[:2], -self.max_target_xy_speed, self.max_target_xy_speed)
        target[2] = np.clip(target[2], -self.max_target_yaw_rate, self.max_target_yaw_rate)
        self.target_velocity = target

    def enable_random_wind(self, enabled: bool = True, magnitude: Optional[float] = None, heading_deg: Optional[float] = None, elevation_deg: Optional[float] = None) -> None:
        self.random_wind = bool(enabled)
        if magnitude is not None:
            self.wind_magnitude = float(magnitude)
        if heading_deg is not None:
            self.wind_heading_deg = float(heading_deg)
        if elevation_deg is not None:
            self.wind_elevation_deg = float(elevation_deg)
        self.wind = self._make_wind()

    def _scale_action_to_motor_cmd(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        return self.min_w + 0.5 * (action + 1.0) * (self.max_w - self.min_w)

    def _get_true_state_parts(self) -> Dict[str, np.ndarray]:
        assert self.quad is not None

        return {
            "target": self.target_velocity.astype(np.float32),
            "pos": self.quad.pos.astype(np.float32),
            "z": np.asarray([self.quad.pos[2]], dtype=np.float32),
            "quat": self.quad.quat.astype(np.float32),
            "vel": self.quad.vel.astype(np.float32),
            "vz": np.asarray([self.quad.vel[2]], dtype=np.float32),
            "omega": self.quad.omega.astype(np.float32),
            "motor": self.quad.wMotor.astype(np.float32),
        }

    def _add_noise(self, key: str, x: np.ndarray) -> np.ndarray:
        std = float(self.noise_std.get(key, 0.0))
        if std <= 0.0:
            return x
        return x + self.np_random.normal(0.0, std, size=x.shape).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        s = self._get_true_state_parts()

        if self.obs_mode == "full":
            parts = [
                self._add_noise("target", s["target"]),
                self._add_noise("pos", s["pos"]),
                self._add_noise("quat", s["quat"]),
                self._add_noise("vel", s["vel"]),
                self._add_noise("omega", s["omega"]),
                self._add_noise("motor", s["motor"]),
            ]
        else:
            # Realistic: no ground-truth x/y.
            # Keeps orientation, z, body rates, linear velocity, and motor speeds.
            parts = [
                self._add_noise("target", s["target"]),
                self._add_noise("z", s["z"]),
                self._add_noise("quat", s["quat"]),
                self._add_noise("omega", s["omega"]),
                self._add_noise("vel", s["vel"]),
                self._add_noise("motor", s["motor"]),
            ]

        obs = np.concatenate(parts).astype(np.float32)

        # Noise can slightly de-normalize quaternion observation. Keep it sane.
        if self.obs_mode == "full":
            q_slice = slice(6, 10)
        else:
            q_slice = slice(4, 8)

        q = obs[q_slice]
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-8:
            obs[q_slice] = q / q_norm

        return obs

    def _default_reward(self, action: np.ndarray) -> float:
        assert self.quad is not None
        w = self.reward_weights

        vx_t, vy_t, vyaw_t = self.target_velocity
        vel_xy = self.quad.vel[:2]
        yaw_rate = self.quad.omega[2]

        vel_err = vel_xy - np.asarray([vx_t, vy_t], dtype=np.float32)
        yaw_rate_err = yaw_rate - vyaw_t

        roll, pitch, _yaw = self.quad.euler

        action_slew = action - self.prev_action

        cost = (
            w.vel_xy * float(np.dot(vel_err, vel_err))
            + w.yaw_rate * float(yaw_rate_err * yaw_rate_err)
            + w.z * float(self.quad.pos[2] * self.quad.pos[2])
            + w.roll_pitch * float(roll * roll + pitch * pitch)
            + w.ang_rate * float(np.dot(self.quad.omega, self.quad.omega))
            + w.action * float(np.dot(action, action))
            + w.action_rate * float(np.dot(action_slew, action_slew))
        )

        return float(w.alive - cost)

    def _is_unstable(self) -> bool:
        assert self.quad is not None

        if not np.all(np.isfinite(self.quad.state)):
            return True

        roll, pitch, _yaw = self.quad.euler
        if abs(self.quad.pos[2]) > self.max_abs_z:
            return True

        if abs(roll) > self.max_tilt_rad or abs(pitch) > self.max_tilt_rad:
            return True

        return False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        options = options or {}

        if "target_velocity" in options:
            self.set_target_velocity(*options["target_velocity"])
        elif options.get("sample_target_velocity", False):
            vx = self.np_random.uniform(-self.max_target_xy_speed, self.max_target_xy_speed)
            vy = self.np_random.uniform(-self.max_target_xy_speed, self.max_target_xy_speed)
            vyaw = self.np_random.uniform(-self.max_target_yaw_rate, self.max_target_yaw_rate)
            self.set_target_velocity(vx, vy, vyaw)

        if "random_wind" in options:
            self.random_wind = bool(options["random_wind"])
        if "wind_magnitude" in options:
            self.wind_magnitude = float(options["wind_magnitude"])
        if "wind_heading_deg" in options:
            self.wind_heading_deg = float(options["wind_heading_deg"])
        if "wind_elevation_deg" in options:
            self.wind_elevation_deg = float(options["wind_elevation_deg"])

        self.t = 0.0
        self.steps = 0
        self.prev_action = np.zeros(4, dtype=np.float32)

        self.quad = Quadcopter(self.t)
        self.wind = self._make_wind()

        obs = self._get_obs()
        info = {
            "target_velocity": self.target_velocity.copy(),
            "wind_enabled": self.random_wind,
            "wind_magnitude": self.wind_magnitude,
            "obs_mode": self.obs_mode,
        }
        return obs, info

    def step(self, action: np.ndarray):
        assert self.quad is not None
        assert self.wind is not None

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        motor_cmd = self._scale_action_to_motor_cmd(action)

        self.quad.update(self.t, self.dt, motor_cmd, self.wind)
        self.t += self.dt
        self.steps += 1

        obs = self._get_obs()

        if self.reward_fn is None:
            reward = self._default_reward(action)
        else:
            # reward_fn(target_velocity, full_state, normalized_action, env)
            reward = float(self.reward_fn(
                self.target_velocity.copy(),
                self.quad.state.copy(),
                action.copy(),
                self,
            ))

        terminated = bool(self.terminate_on_unstable and self._is_unstable())
        truncated = bool(self.steps >= self.max_steps)

        vel_err = self.quad.vel[:2] - self.target_velocity[:2]
        yaw_rate_err = self.quad.omega[2] - self.target_velocity[2]

        info = {
            "t": self.t,
            "target_velocity": self.target_velocity.copy(),
            "velocity": self.quad.vel.copy(),
            "omega": self.quad.omega.copy(),
            "velocity_error_xy": vel_err.copy(),
            "yaw_rate_error": float(yaw_rate_err),
            "position": self.quad.pos.copy(),
            "quat": self.quad.quat.copy(),
            "motor_cmd": motor_cmd.copy(),
            "wind_enabled": self.random_wind,
            "wind_magnitude": self.wind_magnitude,
        }

        self.prev_action = action.copy()
        return obs, reward, terminated, truncated, info
