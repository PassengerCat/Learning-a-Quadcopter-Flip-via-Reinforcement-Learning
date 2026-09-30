"""Three-phase pitch-flip controller with tunable parameters (the MAP-Elites genome).

Course interface: reset(seed=None), act(observation) -> (action, info).
It reads ONLY the 20-value benchmark observation
    [0:3] target vel | [3:6] pos | [6:10] quat (w,x,y,z) | [10:13] vel | [13:16] body rates | [16:20] motors
plus fixed vehicle constants (mass, inertia, motor mixer, motor limits) taken
once from the simulator's parameter set -- model knowledge, not simulator state.

Phases (Lupashin-style):
  climb   : collective F_climb_g * m g for t_climb seconds, body rates damped;
  rotate  : collective F_rot_g * m g, pitch rate driven to min(q_peak, sqrt(2 a_brake (2pi - rev)))
            so it brakes smoothly into 360 deg;
  recover : geometric thrust-vector controller back to hover at the start position.
It decides every `decision_steps` simulator steps (4 x 5 ms = 50 Hz) and holds
its action in between; the pitch rotation `rev` is integrated at every call.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SIM_DIR = Path(os.environ.get("QUAD_SIM_DIR",
                              REPO / "Reward_and_Training" / "Quadcopter_SimCon" / "Simulation"))
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))


@dataclass
class FlipParams:
    t_climb: float = 0.20      # s of climb before rotating
    F_climb_g: float = 2.2     # climb collective, multiples of m*g
    q_peak: float = 20.0       # rad/s peak pitch rate
    a_brake: float = 70.0      # rad/s^2 braking profile into 360 deg
    F_rot_g: float = 0.9       # collective during rotation, multiples of m*g
    Kp_att: float = 60.0       # recovery attitude gain
    Kd_att: float = 12.0       # rate damping (all phases)
    Kv: float = 4.0            # recovery vertical velocity gain
    Kz: float = 2.0            # recovery altitude gain
    Kvxy: float = 2.0          # recovery horizontal velocity gain
    Kpxy: float = 0.8          # recovery horizontal position gain

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)], float)

    @classmethod
    def from_array(cls, x) -> "FlipParams":
        return cls(*[float(v) for v in x])


class VehicleModel:
    """Constant physical parameters of the simulated quadrotor."""

    def __init__(self, params: dict):
        self.m, self.g = float(params["mB"]), float(params["g"])
        self.I = np.asarray(params["IB"], float)
        self.mixer_inv = np.asarray(params["mixerFMinv"], float)
        self.w_min, self.w_max = float(params["minWmotor"]), float(params["maxWmotor"])

    @classmethod
    def from_simulator(cls) -> "VehicleModel":
        from quadFiles.quad import Quadcopter
        return cls(Quadcopter(0.0).params)

    def motor_action(self, F: float, M) -> np.ndarray:
        """Collective thrust F [N] and body moments M [Nm] -> normalized motor commands."""
        w2 = self.mixer_inv @ np.array([F, M[0], M[1], M[2]])
        w = np.sqrt(np.clip(w2, self.w_min ** 2, self.w_max ** 2))
        return np.clip(2.0 * (w - self.w_min) / (self.w_max - self.w_min) - 1.0, -1.0, 1.0)


def _rotmat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _pitch_increment(q_prev, q) -> float:
    """Signed rotation about body y between two consecutive attitudes (small, so unambiguous)."""
    w0, x0, y0, z0 = q_prev
    w1, x1, y1, z1 = q
    dw = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1            # conj(q_prev) * q, w and y parts
    dy = w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1
    if dw < 0.0:                                          # q and -q are the same attitude
        dw, dy = -dw, -dy
    return 2.0 * np.arctan2(dy, dw)


class ThreePhaseFlip:
    def __init__(self, params: FlipParams | None = None, model: VehicleModel | None = None,
                 dt: float = 0.005, decision_steps: int = 4):
        self.p = params if params is not None else FlipParams()
        self.model = model if model is not None else VehicleModel.from_simulator()
        self.dt, self.decision_steps = float(dt), int(decision_steps)
        self.reset()

    def reset(self, seed=None):
        self.k, self.rev, self.q_prev = 0, 0.0, None
        self.phase, self.action = "climb", np.zeros(4)

    def act(self, observation):
        obs = np.asarray(observation, float)
        pos, q, vel, om = obs[3:6], obs[6:10] / np.linalg.norm(obs[6:10]), obs[10:13], obs[13:16]
        if self.q_prev is not None:
            self.rev += _pitch_increment(self.q_prev, q)
        self.q_prev = q
        if self.k % self.decision_steps == 0:
            self.action = self._decide(self.k * self.dt, pos, q, vel, om)
        self.k += 1
        return self.action.astype(np.float32), {"phase": self.phase, "rev": self.rev}

    def _decide(self, t, pos, q, vel, om):
        p, m, g, I = self.p, self.model.m, self.model.g, self.model.I
        if self.phase == "climb" and t >= p.t_climb:
            self.phase = "rotate"
        if self.phase == "rotate" and self.rev >= 2 * np.pi - 0.15:
            self.phase = "recover"
        if self.phase == "climb":
            return self.model.motor_action(p.F_climb_g * m * g, I @ (-p.Kd_att * om))
        if self.phase == "rotate":
            q_des = min(p.q_peak, np.sqrt(2.0 * p.a_brake * max(2 * np.pi - self.rev, 0.0)))
            alpha = np.array([-p.Kd_att * om[0], 40.0 * (q_des - om[1]), -p.Kd_att * om[2]])
            return self.model.motor_action(p.F_rot_g * m * g, I @ alpha)
        # recover: tilt the thrust vector towards the acceleration that returns to hover at the start
        R = _rotmat(q)
        a_xy = -p.Kvxy * vel[:2] - p.Kpxy * pos[:2]
        n = np.linalg.norm(a_xy)
        if n > 0.5 * g:
            a_xy *= 0.5 * g / n
        f_w = m * (np.array([a_xy[0], a_xy[1], -p.Kv * vel[2] - p.Kz * pos[2]]) - np.array([0.0, 0.0, g]))
        zb_des = -f_w / np.linalg.norm(f_w)
        F = max(float(-f_w @ R[:, 2]), 0.1 * m * g)
        e = np.cross(np.array([0.0, 0.0, 1.0]), R.T @ zb_des)
        return self.model.motor_action(F, I @ (p.Kp_att * e - p.Kd_att * om))
