"""Run the §02 baselines (random, PID hover, scripted flip) in the universal harness.

The baselines are the PPO track's own, loaded unchanged from Reward_and_Training/baselines.py
(by file path, so a copy elsewhere on sys.path, e.g. in the simulator folder, cannot shadow it).

The baselines follow the PPO track's interface: `act(raw_obs) -> action`, one decision
per `k` simulator steps (their ActionRepeat, k = 4 -> 50 Hz). The harness calls a
controller every 5 ms step and expects `(action, info)`. `DecisionRateAdapter` bridges
the two: it calls the wrapped controller on steps 0, k, 2k, ... and holds its action in
between, exactly as ActionRepeat does, and returns the controller's diagnostics as info.

    from harness_adapter import make_baseline
    ctrl = make_baseline("scripted_flip")          # or "random", "pid_hover"
    results = evaluate(ctrl, seeds)                 # Evaluation/evaluate.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (HERE, REPO / "Evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import evaluate as harness                                    # noqa: E402  (puts the simulator on sys.path)

BASELINES_PATH = REPO / "Reward_and_Training" / "baselines.py"


def _load_baselines():
    name = "ppo_track_baselines"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BASELINES_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module                  # registered first: dataclasses need it
        spec.loader.exec_module(module)
    return sys.modules[name]


_baselines = _load_baselines()
BASELINES, make_controller = _baselines.BASELINES, _baselines.make_controller

ACTION_REPEAT = 4          # the baselines' decision period: 4 x 5 ms = 20 ms


class DecisionRateAdapter:
    """Harness interface (reset / act -> (action, info)) for a controller that decides every k steps."""

    def __init__(self, controller, k: int = ACTION_REPEAT):
        if int(k) < 1:
            raise ValueError("k must be >= 1")
        self.controller, self.k = controller, int(k)
        self.name = getattr(controller, "name", type(controller).__name__)
        self.reset()

    def reset(self, seed=None):
        self.controller.reset(seed=seed)
        self._i, self._action = 0, None

    def act(self, observation):
        if self._i % self.k == 0:
            self._action = np.asarray(self.controller.act(np.asarray(observation, dtype=np.float64)),
                                      dtype=np.float32)
        self._i += 1
        return self._action, dict(getattr(self.controller, "diagnostics", {}) or {})


class _ActionRepeatView:
    """What baselines.make_controller reads from an env: the base env (model, limits, dt)
    and the action-repeat factor `k` in the wrapper chain."""

    def __init__(self, base_env, k: int):
        self.k, self.env, self.unwrapped = int(k), base_env, base_env


def make_baseline(name: str, *, seed: int = 0, k: int = ACTION_REPEAT, **kwargs) -> DecisionRateAdapter:
    """A §02 baseline configured exactly as in the PPO track (decision period k x dt, default
    FlipScript / PIDGains), wrapped for the harness."""
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; choose from {BASELINES}")
    env = harness.make_benchmark_env(1.0)
    try:
        ctrl = make_controller(name, _ActionRepeatView(env, k), seed=seed, **kwargs)
    finally:
        env.close()
    return DecisionRateAdapter(ctrl, k)
