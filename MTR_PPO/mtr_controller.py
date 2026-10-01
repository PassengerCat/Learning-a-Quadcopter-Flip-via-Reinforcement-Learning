"""MTR-PPO: deployed policy behind the course interface, for the evaluation harness.

    ctrl = MTRController("MTR_PPO/models/ctbr_seed0/final_model.zip")   # config.json next to it
    ctrl.reset(seed=0)
    action, info = ctrl.act(raw_observation)                          # called every 5 ms

The policy decides once every `action_repeat` calls (as in training, 50 Hz) and the motor
command is held in between. The actor observation is built by the same TrackingObsBuilder as in
training, for the nominal command (omega from the run config, default 5 rad/s). A CTBR policy
is mapped to motor commands by the same fixed rate loop, once per decision, with the rates of
that decision's observation, exactly as the training environment does. The critic's
privileged input is not needed: the actor reads only the first n_actor values.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional, Union

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RT = os.path.join(os.path.dirname(_HERE), "Reward_and_Training")   # the PPO track's modules
_SIM = os.environ.get("QUAD_SIM_DIR", os.path.join(_RT, "Quadcopter_SimCon", "Simulation"))
for _p in (_HERE, _RT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if os.path.isdir(_SIM) and _SIM not in sys.path:
    sys.path.append(_SIM)                 # the simulator's own packages (quadFiles, utils)

from mtr_observation import TrackingObsBuilder                                   # noqa: E402
from mtr_reference import FlipCommand                                        # noqa: E402
from flip_policy import CTBRMapper, model_from_dict                      # noqa: E402

CONFIG_KEYS = ("action_mode", "action_repeat", "ctbr", "quad_model", "n_actor", "n_priv",
               "flip_direction", "eps_full", "radius", "eval_omega")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    missing = [k for k in CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"run config {path} lacks {missing}")
    return cfg


def _find_config(model_path: str) -> str:
    d = os.path.dirname(os.path.abspath(model_path))
    for cand in (os.path.join(d, "config.json"), os.path.join(os.path.dirname(d), "config.json")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"no config.json next to {model_path} or one level up")


class MTRController:
    name = "ppo_mtr"

    def __init__(self, model: Union[str, object], run_config: Optional[Union[str, dict]] = None,
                 omega: Optional[float] = None, device: str = "cpu"):
        if isinstance(model, str):
            from stable_baselines3 import PPO
            path = model
            model = PPO.load(path, device=device,
                             custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0,
                                             "clip_range": lambda _: 0.2})
            run_config = run_config if run_config is not None else _find_config(path)
        if isinstance(run_config, str):
            run_config = load_config(run_config)
        if run_config is None:
            raise ValueError("a run config is required")
        self.model, self.rc = model, run_config
        self.k = int(run_config["action_repeat"])
        self.n_actor, self.n_priv = int(run_config["n_actor"]), int(run_config["n_priv"])
        if self.n_actor != TrackingObsBuilder.dim:
            raise ValueError(f"run config n_actor {self.n_actor} != TrackingObsBuilder.dim {TrackingObsBuilder.dim}")
        d = float(run_config["flip_direction"])
        self.command = FlipCommand(omega=float(omega if omega is not None else run_config["eval_omega"]),
                                   radius=float(run_config["radius"]), direction=d)
        self.builder = TrackingObsBuilder(d, float(run_config["eps_full"]))
        self.mapper = None
        if run_config["action_mode"] == "ctbr":
            self.mapper = CTBRMapper(model_from_dict(run_config["quad_model"]), **run_config["ctbr"])
        elif run_config["action_mode"] != "motors":
            raise ValueError(f"unknown action_mode {run_config['action_mode']!r}")
        self.reset()

    def reset(self, seed=None) -> None:
        self.builder.reset(self.command)
        self._i = 0
        self._motor = np.zeros(4, np.float32)
        self._policy_action = np.zeros(4, np.float32)

    def decide(self, raw_obs) -> np.ndarray:
        """One decision: raw observation -> motor command (also updates the builder)."""
        raw = np.asarray(raw_obs, dtype=np.float64)
        obs = np.concatenate([self.builder.build(raw), np.zeros(self.n_priv, np.float32)])
        a, _ = self.model.predict(obs, deterministic=True)
        a = np.clip(np.asarray(a, np.float32), -1.0, 1.0)
        self.builder.set_action(a)
        self._policy_action = a
        motor = self.mapper.motor_action(a, raw[13:16]) if self.mapper is not None else a
        return np.asarray(motor, np.float32)

    def act(self, observation):
        if self._i % self.k == 0:
            self._motor = self.decide(observation)
        self._i += 1
        return self._motor, dict(task=int(self.builder.task), rev=float(self.builder.tracker.rev))
