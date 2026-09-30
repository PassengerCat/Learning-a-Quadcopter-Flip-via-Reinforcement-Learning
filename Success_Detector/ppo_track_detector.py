"""The PPO track's success detector, used as the evaluation criterion for all controllers.

Verbatim copy of the detector in the PPO track's Reward_and_Training/flip_reward.py
(GitHub snapshot pulled 2026-09-29, sha256 0e84d979f14a518a39cfead3e0e254a5206add0b9b28d85173cce33b415dc8c4):
SuccessConfig, SuccessDetector and the geometry they use. Only the module docstring and the
imports are new; every copied definition is unchanged, so verdicts are bit-identical to hers.
Copied rather than imported so the harness does not depend on the reward module.

Criterion (her defaults): the gravity direction sweeps a signed revolution >= 2*pi - 0.15 in
the body x-z plane, the tilt reaches >= 150 deg at some update, and afterwards the vehicle is
upright (tilt < 15 deg) and slow (|v| < 0.4 m/s, |w| < 0.8 rad/s) for 20 consecutive updates.
Success is latched: once reached it stays True. It is meant to be updated once per decision
(every 4 simulator steps = 20 ms), so 20 updates = 0.4 s.

    det = SuccessDetector()
    det.reset(); det.update(None, env=env)          # at reset
    ok = det.update(None, env=env)                  # after every decision (4 sim steps)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return a unit quaternion; falls back to identity if degenerate."""
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def projected_gravity(q: np.ndarray) -> np.ndarray:
    """
    Projected-gravity unit vector g = (gx, gy, gz) from a scalar-first quaternion.
    Matches the sim's verified convention: upright -> [0,0,1], inverted -> [0,0,-1].
        gx = 2(xz - wy),  gy = 2(yz + wx),  gz = 1 - 2(x^2 + y^2)
    """
    w, x, y, z = quat_normalize(q)
    gx = 2.0 * (x * z - w * y)
    gy = 2.0 * (y * z + w * x)
    gz = 1.0 - 2.0 * (x * x + y * y)
    return np.array([gx, gy, gz])


def tilt_angle(gz: float) -> float:
    """Geodesic tilt from upright: Theta = arccos(gz), in radians, 0..pi."""
    return float(np.arccos(np.clip(gz, -1.0, 1.0)))


def gravity_pitch_angle(q: np.ndarray) -> Optional[float]:
    """Pitch angle of the vehicle measured from GRAVITY, in the body x-z plane:
        theta = atan2(-gx, gz)      (0 upright, +-pi inverted, + = nose up)
    Returns None when gravity is (almost) perpendicular to that plane (|g_xz| small,
    e.g. rolled 90 deg), where the angle is undefined."""
    g = projected_gravity(q)
    if g[0] * g[0] + g[2] * g[2] < 0.09:          # |g_xz| < 0.3
        return None
    return float(np.arctan2(-g[0], g[2]))


def gravity_pitch_increment(q_prev: np.ndarray, q_curr: np.ndarray) -> float:
    """Signed change of `gravity_pitch_angle` between two consecutive attitudes.

    Unlike integrating body rate q (or the body-y part of the quaternion delta),
    this is a GEOMETRIC flip measure: it only accumulates 2*pi if gravity really
    sweeps once around the body x-z plane, i.e. the vehicle goes upright ->
    inverted -> upright about its pitch axis. Coning (tilted + yawing), which makes
    integrated q grow without bound, leaves it oscillating around zero.
    (Found in run-4: the policy 'earned' 9 rad of body-y rotation while never
    tilting past 92 deg.)"""
    a0, a1 = gravity_pitch_angle(q_prev), gravity_pitch_angle(q_curr)
    if a0 is None or a1 is None:
        return 0.0
    return float((a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi)


class StateView:
    """
    Lightweight, robust extraction of (pos, quat, vel, omega) from whatever the
    env hands the reward hook. Prefers `env.quad` attributes when available,
    otherwise parses `full_state` by length (20-dim full obs or 21-dim raw).
    """

    __slots__ = ("pos", "quat", "vel", "omega")

    def __init__(self, full_state: Optional[np.ndarray], env=None):
        quad = getattr(env, "quad", None)
        if quad is not None and hasattr(quad, "quat"):
            self.pos = np.asarray(quad.pos, dtype=float)
            self.quat = np.asarray(quad.quat, dtype=float)
            self.vel = np.asarray(quad.vel, dtype=float)
            self.omega = np.asarray(quad.omega, dtype=float)
            return

        s = np.asarray(full_state, dtype=float).ravel()
        if s.size >= 21 and _looks_like_raw(s):
            # raw sim state: [pos(3) | quat(4) | vel(3) | omega(3) | motors...]
            self.pos = s[0:3]
            self.quat = s[3:7]
            self.vel = s[7:10]
            self.omega = s[10:13]
        elif s.size >= 16:
            # 20-dim "full" obs: [tvel(3) | pos(3) | quat(4) | vel(3) | pqr(3) | motors]
            self.pos = s[3:6]
            self.quat = s[6:10]
            self.vel = s[10:13]
            self.omega = s[13:16]
        else:
            raise ValueError(f"Cannot parse state of size {s.size}; pass env.quad or a 20/21-dim vector.")


def _looks_like_raw(s: np.ndarray) -> bool:
    """Heuristic: raw layout has a unit quaternion at [3:7]."""
    if s.size < 7:
        return False
    return abs(float(np.linalg.norm(s[3:7])) - 1.0) < 1e-3


@dataclass
class SuccessConfig:
    flip_direction: float = +1.0
    full_rev_rad: float = 2.0 * np.pi - 0.15   # a genuine 360 (small margin)
    cone_rad: float = np.deg2rad(15.0)         # upright tolerance
    v_thr: float = 0.4                         # m/s
    w_thr: float = 0.8                         # rad/s
    hold_steps: int = 20                       # consecutive decisions the criterion must hold
    inverted_tilt_rad: float = np.deg2rad(150.0)   # must pass through at least this tilt


class SuccessDetector:
    """
    Geometric, reward-independent success test. Success = the gravity direction
    swept a full signed revolution in the body x-z plane (gravity_pitch_increment:
    drift-free and immune to coning), the vehicle passed through an inverted
    attitude (tilt >= inverted_tilt_rad), AND it then holds upright + slow for
    `hold_steps` consecutive updates.

    Call once per decision (post action-repeat) with the same `full_state`/`env`
    the policy sees. `reset()` at episode start.
    """

    def __init__(self, config: Optional[SuccessConfig] = None):
        self.cfg = config or SuccessConfig()
        self.reset()

    def reset(self) -> None:
        self._rev = 0.0
        self._q_prev: Optional[np.ndarray] = None
        self._rev_done = False
        self._inverted = False
        self._hold = 0
        self._succeeded = False

    def update(self, full_state, env=None) -> bool:
        cfg = self.cfg
        sv = StateView(full_state, env)
        if self._q_prev is not None:
            self._rev += cfg.flip_direction * gravity_pitch_increment(self._q_prev, sv.quat)
        self._q_prev = sv.quat.copy()
        if tilt_angle(float(projected_gravity(sv.quat)[2])) >= cfg.inverted_tilt_rad:
            self._inverted = True

        if not self._rev_done and self._rev >= cfg.full_rev_rad and self._inverted:
            self._rev_done = True

        if self._rev_done:
            g = projected_gravity(sv.quat)
            upright = tilt_angle(float(g[2])) < cfg.cone_rad
            slow = (np.linalg.norm(sv.vel) < cfg.v_thr
                    and np.linalg.norm(sv.omega) < cfg.w_thr)
            self._hold = self._hold + 1 if (upright and slow) else 0
        else:
            self._hold = 0

        if self._hold >= cfg.hold_steps:
            self._succeeded = True
        return self._succeeded

    @property
    def succeeded(self) -> bool:
        return self._succeeded

    @property
    def revolution(self) -> float:
        return self._rev

    @property
    def passed_inverted(self) -> bool:
        return self._inverted
