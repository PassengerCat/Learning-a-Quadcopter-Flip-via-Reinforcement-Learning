"""
flip_task_env.py — §03 wiring for the Quadcopter Flip task.

Composes the (unchanged) §01 environment with the §03 reward and success detector:

  * installs `FlipReward` as the env's `reward_fn`;
  * appends the flip features [sin α, cos α, progress, phase] to the policy
    observation (REQUIRED — the shaping potential depends on α and phase, so the
    policy must observe them or the task is non-Markov);
  * resets the reward together with the env.

§01 (`flip_env.py`) is imported and used as-is; nothing there is edited.

Layout / imports
----------------
This file expects to import `flip_env` (and, transitively, the fork's
`quad_velocity_env`). It inserts the sibling `Environment_Task_Wrapper/` folder
onto sys.path automatically; `quad_velocity_env.py` must still be importable the
same way §01 required (same folder or on PYTHONPATH).

Usage
-----
    from flip_task_env import make_flip_task_env, make_vec_task_env
    env = make_flip_task_env(action_repeat=4)          # single env
    venv = make_vec_task_env(n_envs=8)                 # parallel (training)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- make §01 (flip_env) and the fork sim (quad_velocity_env) importable ----- #
# Without editing §01. The fork's `Simulation/` folder can be pointed to via the
# QUAD_SIM_DIR env var; otherwise set it on PYTHONPATH (same requirement as §01).
_HERE = os.path.dirname(os.path.abspath(__file__))
_cands = [
    os.environ.get("QUAD_SIM_DIR"),                       # e.g. ...\Quadcopter_SimCon\Simulation
    os.path.join(_HERE, "..", "Environment_Task_Wrapper"),
    _HERE,
]
for _cand in _cands:
    if not _cand:
        continue
    _cand = os.path.abspath(_cand)
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from flip_env import make_flip_env, ObsConfig                     # noqa: E402
from flip_reward import (                                          # noqa: E402
    FlipReward, FlipRewardConfig, SuccessDetector, SuccessConfig,
)


# --------------------------------------------------------------------------- #
# Observation wrapper: append the flip features, and reset the reward          #
# --------------------------------------------------------------------------- #
class FlipFeatureObservation(gym.Wrapper):
    """Appends `reward.obs_features()` to the §01 observation and keeps the shared
    reward instance in sync with episode resets. Sits OUTSIDE the §01 wrappers."""

    def __init__(self, env: gym.Env, reward: FlipReward):
        super().__init__(env)
        self.reward = reward
        base = env.observation_space
        assert isinstance(base, spaces.Box) and len(base.shape) == 1
        n = FlipReward.N_OBS_FEATURES
        low = np.concatenate([base.low, np.array([-1.0, -1.0, 0.0, 0.0], np.float32)])
        high = np.concatenate([base.high, np.array([1.0, 1.0, 1.0, 1.0], np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment(self, obs):
        return np.concatenate([np.asarray(obs, np.float32), self.reward.obs_features()]).astype(np.float32)

    def reset(self, **kwargs):
        self.reward.reset()                          # clear α, phase, milestones first
        obs, info = self.env.reset(**kwargs)         # §01 reset (does not touch the reward)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info


# --------------------------------------------------------------------------- #
# Factories                                                                    #
# --------------------------------------------------------------------------- #
def make_flip_task_env(action_repeat: int = 4,
                       obs_cfg: Optional[ObsConfig] = None,
                       reward_cfg: Optional[FlipRewardConfig] = None,
                       max_abs_z: float = 5.0,
                       episode_seconds: float = 10.0,
                       seed: Optional[int] = None) -> gym.Env:
    """Build one fully-wired flip task env (single reward instance per env)."""
    reward = FlipReward(reward_cfg or FlipRewardConfig())
    env = make_flip_env(
        action_repeat=action_repeat,
        obs_cfg=obs_cfg,
        reward_fn=reward,                # FlipReward.__call__ matches the hook
        max_abs_z=max_abs_z,
        episode_seconds=episode_seconds,
        seed=None,                       # seed after wrapping, via reset()
    )
    env = FlipFeatureObservation(env, reward)
    env.unwrapped_reward = reward        # handle for logging (reward.alpha / .revolution / .phase)
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
    """Convenience: an independent success detector for the eval loop (§04)."""
    return SuccessDetector(SuccessConfig(**overrides)) if overrides else SuccessDetector()
