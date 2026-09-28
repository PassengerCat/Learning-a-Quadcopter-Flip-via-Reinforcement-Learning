"""
ctbr_expert.py — a hand-written flip controller that speaks the POLICY's action language
(collective thrust + body rates, see flip_policy.CTBRMapper).

Two uses:
  1. Sanity check that the CTBR action space can express a complete flip + recovery
     (if this expert fails, no policy in that space can succeed).
  2. Demonstrations for behaviour-cloning warm-start of PPO (train_ppo.py --bc-episodes),
     the standard remedy when on-policy exploration cannot discover a manoeuvre that
     only pays off once fully committed to (see reward_log.md, runs 5-8).

The expert only produces actions in [-1, 1]^4 exactly like the policy; the same
CTBRMapper turns them into motor commands.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from baselines import QuadModel, quat_to_rotmat, wrap_pi
from flip_policy import FlipProgressTracker


@dataclass
class CTBRExpertParams:
    direction: float = +1.0
    climb_time: float = 0.28       # s, strong climb before the flip (altitude to spend)
    climb_thrust: float = 0.75     # collective action during the climb
    flip_rate: float = 1.0         # pitch-rate action during the flip (x rate_max)
    flip_thrust: float = -0.8      # collective action while rotating (low thrust)
    release_deg: float = 300.0     # stop commanding rotation at this flip angle
    # recovery (reduced-attitude P -> rate command, velocity/altitude PD -> thrust)
    kp_att: float = 8.0            # rad/s per rad
    kp_vel: float = 2.0
    kp_pos: float = 1.0
    kp_z: float = 2.0
    kd_z: float = 2.5


class CTBRExpert:
    """Stateful expert: reset(), then act(raw_obs) once per decision -> CTBR action."""

    def __init__(self, model: QuadModel, dt: float, rate_max_rp: float = 20.0,
                 rate_max_yaw: float = 4.0, params: Optional[CTBRExpertParams] = None):
        self.m, self.dt = model, float(dt)
        self.rmax = np.array([rate_max_rp, rate_max_rp, rate_max_yaw])
        self.p = params or CTBRExpertParams()
        self.T_hover = model.mass * model.g
        self.reset()

    def reset(self, seed=None) -> None:
        self.t = 0.0
        self.phase = "climb"
        self.tracker = FlipProgressTracker(self.p.direction)
        self.hold_pos = None

    def _thrust_action(self, T: float) -> float:
        Th, Tmax, Tmin = self.T_hover, self.m.max_thrust, self.m.min_thrust
        return float(np.clip((T - Th) / ((Tmax - Th) if T >= Th else (Th - Tmin)), -1, 1))

    def _recover(self, pos, vel, R) -> np.ndarray:
        p, m = self.p, self.m
        if self.hold_pos is None:
            self.hold_pos = pos.copy()
        # desired acceleration (NED), then thrust vector; tilt-limited
        acc = np.zeros(3)
        acc[:2] = p.kp_vel * (p.kp_pos * (self.hold_pos[:2] - pos[:2]) - vel[:2])
        acc[:2] = np.clip(acc[:2], -6, 6)
        acc[2] = p.kp_z * (self.hold_pos[2] - pos[2]) - p.kd_z * vel[2]
        F = m.mass * (acc - np.array([0, 0, m.g]))
        F[2] = min(F[2], -0.3 * m.mass * m.g)
        z_des = -F / np.linalg.norm(F)
        zb = R.T @ z_des
        e3 = np.array([0.0, 0.0, 1.0])
        axis = np.cross(e3, zb)
        n = np.linalg.norm(axis)
        ang = np.arccos(np.clip(zb[2], -1, 1))
        axis = axis / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
        rate = p.kp_att * ang * axis
        yaw = np.arctan2(R[1, 0], R[0, 0])
        rate[2] = 2.0 * wrap_pi(0.0 - yaw) * max(0.0, R[2, 2])
        T = float(F @ (-R[:, 2]))
        a = np.zeros(4)
        a[0] = self._thrust_action(max(T, 0.2 * self.T_hover))
        a[1:] = np.clip(rate / self.rmax, -1, 1)
        return a

    def act(self, raw_obs) -> np.ndarray:
        o = np.asarray(raw_obs, dtype=np.float64)
        pos, quat, vel = o[3:6], o[6:10], o[10:13]
        R = quat_to_rotmat(quat)
        self.tracker.update(quat)
        p = self.p
        if self.phase == "climb" and self.t >= p.climb_time:
            self.phase = "flip"
        if self.phase == "flip" and np.degrees(self.tracker.rev) >= p.release_deg:
            self.phase = "recover"
        if self.phase == "climb":
            a = self._recover(pos, vel, R)          # attitude hold ...
            a[0] = p.climb_thrust                   # ... with a strong climb
        elif self.phase == "flip":
            a = np.array([p.flip_thrust, 0.0, p.direction * p.flip_rate, 0.0])
        else:
            a = self._recover(pos, vel, R)
        self.t += self.dt
        return np.clip(a, -1, 1).astype(np.float32)
