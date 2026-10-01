"""MTR-PPO: actor observation, built from the raw 20-value simulator observation (deployable).

    [ p_rel (3) | v_rel / 5 (3) | w_rel / 10 (3) | R_rel (9) | previous action (4) |
      task one-hot [FLIP, HOVER] (2) | commanded omega / 5 (1) ]            -> 25 values

As in GEAR (arXiv 2602.10997), the observation is the body-frame relative state (18), the previous
action (4) and the command (task one-hot + scalar parameter). The scales only bring the entries to O(1).

The task switches from FLIP to HOVER when the geometric revolution, accumulated once per
decision from the observed quaternion (the PPO track's FlipProgressTracker), reaches
2*pi - eps_full. The reference is fixed at the first observation after reset().
The SAME builder is used by the training environment and by the deployed controller.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RT = os.path.join(os.path.dirname(_HERE), "Reward_and_Training")   # the PPO track's modules
for _p in (_HERE, _RT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mtr_reference import FLIP, HOVER, FlipCommand, LoopReference        # noqa: E402
from flip_policy import FlipProgressTracker                           # noqa: E402

V_SCALE, W_SCALE, OMEGA_SCALE, CLIP = 5.0, 10.0, 5.0, 10.0
OBS_DIM = 3 + 3 + 3 + 9 + 4 + 2 + 1


class TrackingObsBuilder:
    """reset(command) at episode start, build(raw) every decision, set_action(a) with the
    action about to be applied (the next observation carries it)."""

    dim = OBS_DIM

    def __init__(self, direction: float = +1.0, eps_full: float = 0.15):
        self.direction, self.eps_full = float(direction), float(eps_full)
        self.reset(FlipCommand(direction=self.direction))

    def reset(self, command: Optional[FlipCommand] = None, anchor=None, rev0: float = 0.0) -> None:
        """anchor=(p0, psi) and rev0 only for training starts on the reference; the deployed
        controller always starts at hover (anchor from the first observation, rev0 = 0)."""
        self.command = command or self.command
        if self.command.direction != self.direction:
            raise ValueError("command direction must equal the builder's direction")
        self.ref = LoopReference(self.command)
        if anchor is not None:
            self.ref.anchor(*anchor)
        self.tracker = FlipProgressTracker(self.direction, self.eps_full)
        self.tracker.rev = float(rev0)
        self.last_action = np.zeros(4, np.float32)

    def set_action(self, action) -> None:
        self.last_action = np.clip(np.asarray(action, np.float32), -1.0, 1.0).copy()

    @property
    def task(self) -> int:
        return HOVER if self.tracker.done else FLIP

    def build(self, raw_obs) -> np.ndarray:
        raw = np.asarray(raw_obs, dtype=np.float64)
        pos, quat, vel, omega = raw[3:6], raw[6:10], raw[10:13], raw[13:16]
        if not self.ref.started:
            self.ref.start(pos, quat)
        self.tracker.update(quat)
        task = self.task
        s = self.ref.relative_state(task, pos, quat, vel, omega)
        onehot = np.array([1.0, 0.0] if task == FLIP else [0.0, 1.0])
        obs = np.concatenate([s["p"], s["v"] / V_SCALE, s["w"] / W_SCALE, s["R"].reshape(-1),
                              self.last_action, onehot, [self.command.omega / OMEGA_SCALE]])
        obs = np.nan_to_num(obs, nan=0.0, posinf=CLIP, neginf=-CLIP)
        return np.clip(obs, -CLIP, CLIP).astype(np.float32)
