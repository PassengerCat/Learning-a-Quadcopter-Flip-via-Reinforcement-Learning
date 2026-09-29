"""
baselines.py — §02 Baseline Controllers for the Quadcopter Flip project.

Three rule-based controllers behind ONE shared interface (``Controller``):

  * ``RandomController``        — uniform random motor commands, fixed seed.
                                  Fixes the floor of every metric.
  * ``PIDHoverController``      — cascaded PID (position -> velocity -> thrust
                                  vector -> attitude -> body rate -> mixer).
                                  Lower bound (never flips) AND the recovery
                                  block used by the scripted flip.
  * ``ScriptedFlipController``  — the required rule-based baseline: PID settle,
                                  pre-pulse climb bias, open-loop feed-forward
                                  torque pulse, coast, brake pulse, and hand-off
                                  to the PID once the integrated rotation passes
                                  a trigger angle.

Shared interface (identical for the learned policy of §03)::

    ctrl.reset(seed=None)             # call at every episode start
    action = ctrl.act(raw_obs)        # raw 20-dim env observation -> (4,) in [-1, 1]
    ctrl.diagnostics                  # dict for logging ONLY (phase, angle, ...)

Controllers consume the RAW observation of ``QuadcopterVelocityEnv`` (obs_mode
"full"); a learned controller applies ``flip_env.transform_observation`` inside
its own ``act``. Every controller is therefore evaluated on exactly the same
input by the §05 harness.

Conventions (verified against the bobzwik dynamics the env wraps):
    NED world frame, FRD body frame, quaternion scalar-first body->world.
    Motor 1 front-left, numbered clockwise (1 FL, 2 FR, 3 RR, 4 RL).
    Motor command is the motor-speed set-point, linear in the action:
        w_cmd = min_w + (a + 1)/2 * (max_w - min_w).
    Positive pitch rate q (omega_y) = nose up.

Nothing here touches the environment's reward, termination, seeds or scoring:
the controllers only read observations and return actions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict, replace
from typing import Dict, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Raw observation layout (QuadcopterVelocityEnv, obs_mode="full")              #
# --------------------------------------------------------------------------- #
SL_POS = slice(3, 6)
SL_QUAT = slice(6, 10)
SL_VEL = slice(10, 13)
SL_RATE = slice(13, 16)
SL_MOTOR = slice(16, 20)

AXIS_INDEX = {"roll": 0, "pitch": 1}


# --------------------------------------------------------------------------- #
# Small math helpers                                                          #
# --------------------------------------------------------------------------- #
def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Body->world rotation matrix from a scalar-first quaternion (float64).
    Same formula as flip_env.quat_to_rotation_matrix; re-normalises q so that
    integrator drift never leaks into the controllers."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def tilt_angle(R: np.ndarray) -> float:
    """Angle between body z and world z (0 = upright, pi = fully inverted)."""
    return float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))


def wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def split_obs(obs: np.ndarray):
    o = np.asarray(obs, dtype=np.float64)
    return o[SL_POS], o[SL_QUAT], o[SL_VEL], o[SL_RATE], o[SL_MOTOR]


# --------------------------------------------------------------------------- #
# Vehicle model (read from the simulator when available)                      #
# --------------------------------------------------------------------------- #
@dataclass
class QuadModel:
    mass: float = 1.2
    g: float = 9.81
    kTh: float = 1.076e-5          # thrust  = kTh * w^2   [N]
    kTo: float = 1.632e-7          # torque  = kTo * w^2   [N m]
    dxm: float = 0.16              # arm, x  [m]
    dym: float = 0.16              # arm, y  [m]
    inertia: np.ndarray = field(default_factory=lambda: np.diag([0.0123, 0.0123, 0.0224]))
    w_min: float = 75.0
    w_max: float = 925.0
    w_hover: float = 0.0           # 0 -> derived from mass, g, kTh

    def __post_init__(self):
        if self.w_hover <= 0:
            self.w_hover = float(np.sqrt(self.mass * self.g / (4.0 * self.kTh)))

    @staticmethod
    def from_env(env) -> "QuadModel":
        """Pull physical parameters from the wrapped simulator (env.unwrapped.quad.params).
        Falls back to the bobzwik defaults for anything that is missing."""
        base = getattr(env, "unwrapped", env)
        m = QuadModel()
        p = getattr(getattr(base, "quad", None), "params", None) or {}
        m.mass = float(p.get("mB", m.mass))
        m.g = float(p.get("g", m.g))
        m.kTh = float(p.get("kTh", m.kTh))
        m.kTo = float(p.get("kTo", m.kTo))
        m.dxm = float(p.get("dxm", m.dxm))
        m.dym = float(p.get("dym", m.dym))
        if "IB" in p:
            m.inertia = np.asarray(p["IB"], dtype=np.float64)
        m.w_min = float(getattr(base, "min_w", p.get("minWmotor", m.w_min)))
        m.w_max = float(getattr(base, "max_w", p.get("maxWmotor", m.w_max)))
        m.w_hover = float(getattr(base, "hover_w", p.get("w_hover", np.sqrt(m.mass * m.g / 4 / m.kTh))))
        return m

    @property
    def hover_action(self) -> float:
        return 2.0 * (self.w_hover - self.w_min) / (self.w_max - self.w_min) - 1.0

    @property
    def max_thrust(self) -> float:
        return 4.0 * self.kTh * self.w_max ** 2

    @property
    def min_thrust(self) -> float:
        return 4.0 * self.kTh * self.w_min ** 2

    def max_torque(self, axis: int) -> float:
        """Largest body torque on roll (0) or pitch (1): two motors at max, two at min."""
        arm = self.dym if axis == 0 else self.dxm
        return 2.0 * self.kTh * (self.w_max ** 2 - self.w_min ** 2) * arm


def control_dt_from_env(env) -> float:
    """Decision period seen by a controller = sim dt x action-repeat k.
    Walks the wrapper chain looking for ActionRepeat.k and the base env's dt."""
    k, e = 1, env
    while e is not None:
        if hasattr(e, "k") and isinstance(getattr(e, "k"), int):
            k *= int(e.k)
        e = getattr(e, "env", None)
    base = getattr(env, "unwrapped", env)
    for name in ("dt", "Ts", "sim_dt", "ts"):
        v = getattr(base, name, None)
        if isinstance(v, (int, float)) and v > 0:
            return float(v) * k
    if getattr(base, "max_steps", None) and getattr(base, "episode_seconds", None):
        return float(base.episode_seconds) / float(base.max_steps) * k
    return 0.005 * k  # bobzwik default Ts


# --------------------------------------------------------------------------- #
# Mixer with saturation priority (roll/pitch > collective > yaw)              #
# --------------------------------------------------------------------------- #
class Mixer:
    """Maps a desired collective thrust T [N] and body torques tau [N m] to the
    four normalised motor actions.

    Squared motor speeds are linear in the wrench:  [T, Mx, My, Mz] = B @ w^2.
    When the request is infeasible we keep the *direction* of the roll/pitch
    torque (attitude authority first), then place the collective as close to the
    request as the remaining headroom allows, then add as much yaw as fits.
    This is what keeps the recovery well-behaved after a flip."""

    def __init__(self, model: QuadModel):
        self.m = model
        kTh, kTo, dx, dy = model.kTh, model.kTo, model.dxm, model.dym
        # Rows: thrust, roll (Mx), pitch (My), yaw (Mz). Columns: motors 1..4.
        self.B = np.array([[kTh, kTh, kTh, kTh],
                           [dy * kTh, -dy * kTh, -dy * kTh, dy * kTh],
                           [dx * kTh, dx * kTh, -dx * kTh, -dx * kTh],
                           [-kTo, kTo, -kTo, kTo]])
        self.Binv = np.linalg.inv(self.B)
        self.lo = model.w_min ** 2
        self.hi = model.w_max ** 2

    def wrench(self, w: np.ndarray) -> np.ndarray:
        """Forward model: motor speeds -> [T, Mx, My, Mz]."""
        return self.B @ (np.asarray(w, dtype=np.float64) ** 2)

    def w2_from_wrench(self, T: float, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=np.float64)
        rp = self.Binv @ np.array([0.0, tau[0], tau[1], 0.0])
        yw = self.Binv @ np.array([0.0, 0.0, 0.0, tau[2]])
        span_max = self.hi - self.lo

        # 1) roll/pitch must fit inside the motor range (scale, keep direction)
        span = rp.max() - rp.min()
        if span > span_max:
            rp *= span_max / span
        # 2) collective: as requested, but shifted so that roll/pitch fits
        c = float(T) / (4.0 * self.m.kTh)
        c = np.clip(c, self.lo - rp.min(), self.hi - rp.max())
        w2 = c + rp
        # 3) yaw: largest s in [0, 1] that keeps every motor in range
        s = 1.0
        for i in range(4):
            if yw[i] > 0:
                s = min(s, (self.hi - w2[i]) / yw[i])
            elif yw[i] < 0:
                s = min(s, (self.lo - w2[i]) / yw[i])
        w2 = w2 + max(s, 0.0) * yw
        return np.clip(w2, self.lo, self.hi)

    def action(self, T: float, tau: np.ndarray) -> np.ndarray:
        w = np.sqrt(self.w2_from_wrench(T, tau))
        a = 2.0 * (w - self.m.w_min) / (self.m.w_max - self.m.w_min) - 1.0
        return np.clip(a, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Shared controller interface                                                 #
# --------------------------------------------------------------------------- #
class Controller(ABC):
    """Interface shared by every baseline and by the learned policy (§03).

    ``act`` maps the RAW env observation to an action in [-1, 1]^4.
    ``diagnostics`` is for logging/visualisation only and must never feed back
    into the environment, reward, termination or scoring."""

    name: str = "controller"

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        ...

    @property
    def diagnostics(self) -> Dict:
        return {}

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.act(obs)


# --------------------------------------------------------------------------- #
# 1) Random policy — the floor                                                #
# --------------------------------------------------------------------------- #
class RandomController(Controller):
    """i.i.d. motor commands. ``uniform``: U[-1, 1]^4 (the brief's floor).
    ``hover_gaussian``: N(hover, sigma) clipped — a slightly less degenerate
    floor, useful for the report's sanity checks. Reproducible via the seed."""

    name = "random"

    def __init__(self, seed: int = 0, mode: str = "uniform", sigma: float = 0.35,
                 hover_action: float = 0.0):
        if mode not in ("uniform", "hover_gaussian"):
            raise ValueError(mode)
        self.base_seed = int(seed)
        self.mode = mode
        self.sigma = float(sigma)
        self.hover_action = float(hover_action)
        self.rng = np.random.default_rng(self.base_seed)

    def reset(self, seed: Optional[int] = None) -> None:
        # Derive the stream from (base_seed, episode seed) -> reproducible AND
        # different per episode.
        ss = np.random.SeedSequence([self.base_seed] + ([] if seed is None else [int(seed)]))
        self.rng = np.random.default_rng(ss)

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.mode == "uniform":
            a = self.rng.uniform(-1.0, 1.0, size=4)
        else:
            a = self.hover_action + self.rng.normal(0.0, self.sigma, size=4)
        return np.clip(a, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# 2) Cascaded PID — hover lower bound and recovery block                      #
# --------------------------------------------------------------------------- #
@dataclass
class PIDGains:
    # outer loops (world frame, NED)
    kp_pos: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.5]))
    kp_vel: np.ndarray = field(default_factory=lambda: np.array([2.2, 2.2, 4.0]))
    ki_vel: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 1.5]))
    vel_limit_xy: float = 2.5           # m/s
    vel_limit_z: float = 2.5            # m/s
    acc_limit_xy: float = 5.0           # m/s^2
    max_tilt_deg: float = 35.0          # tilt of the commanded thrust vector
    min_thrust_frac: float = 0.15       # of hover thrust (keeps attitude authority)
    # attitude (reduced attitude + separate yaw)
    kp_att: float = 7.0                 # rad/s per rad
    kp_yaw: float = 2.0
    rate_limit_rp: float = 12.0         # rad/s
    rate_limit_yaw: float = 3.0
    # body-rate loop -> angular acceleration
    kp_rate: np.ndarray = field(default_factory=lambda: np.array([16.0, 16.0, 8.0]))
    ki_rate: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 2.0]))
    i_limit_vel: float = 3.0
    i_limit_rate: float = 2.0


class CascadedPID:
    """Position -> velocity -> thrust vector -> attitude -> rate -> torque.

    The attitude loop uses the *reduced-attitude* error (rotation that takes the
    body z axis onto the desired thrust axis), which is well defined for any
    attitude short of exactly inverted — so it can catch the vehicle straight out
    of a flip. Gyroscopic coupling omega x I omega is fed forward in the rate loop.
    """

    def __init__(self, model: QuadModel, dt: float, gains: Optional[PIDGains] = None):
        self.m = model
        self.dt = float(dt)
        self.k = gains or PIDGains()
        self.mixer = Mixer(model)
        self.reset()

    def reset(self, pos_ref=None, yaw_ref: float = 0.0) -> None:
        self.pos_ref = None if pos_ref is None else np.asarray(pos_ref, dtype=np.float64).copy()
        self.yaw_ref = float(yaw_ref)
        self.i_vel = np.zeros(3)
        self.i_rate = np.zeros(3)
        self.last = {}

    def set_reference(self, pos_ref, yaw_ref: Optional[float] = None, reset_integrators: bool = True):
        self.pos_ref = np.asarray(pos_ref, dtype=np.float64).copy()
        if yaw_ref is not None:
            self.yaw_ref = float(yaw_ref)
        if reset_integrators:
            self.i_vel[:] = 0.0
            self.i_rate[:] = 0.0

    # -- outer loop: desired force in world frame ----------------------------
    def desired_force(self, pos, vel) -> np.ndarray:
        k, m = self.k, self.m
        v_ref = k.kp_pos * (self.pos_ref - pos)
        n = np.linalg.norm(v_ref[:2])
        if n > k.vel_limit_xy:
            v_ref[:2] *= k.vel_limit_xy / n
        v_ref[2] = np.clip(v_ref[2], -k.vel_limit_z, k.vel_limit_z)
        e_v = v_ref - vel
        self.i_vel = np.clip(self.i_vel + e_v * self.dt, -k.i_limit_vel, k.i_limit_vel)
        acc = k.kp_vel * e_v + k.ki_vel * self.i_vel
        n = np.linalg.norm(acc[:2])
        if n > k.acc_limit_xy:
            acc[:2] *= k.acc_limit_xy / n
        F = m.mass * (acc - np.array([0.0, 0.0, m.g]))   # thrust must cancel gravity (+z in NED)
        # keep a minimum upward component and bound the tilt of the thrust vector
        F[2] = min(F[2], -k.min_thrust_frac * m.mass * m.g)
        f_xy_max = abs(F[2]) * np.tan(np.deg2rad(k.max_tilt_deg))
        n = np.linalg.norm(F[:2])
        if n > f_xy_max:
            F[:2] *= f_xy_max / n
        return F

    # -- inner loops ----------------------------------------------------------
    def rate_to_torque(self, rate_des, rate) -> np.ndarray:
        k = self.k
        e = rate_des - rate
        self.i_rate = np.clip(self.i_rate + e * self.dt, -k.i_limit_rate, k.i_limit_rate)
        alpha = k.kp_rate * e + k.ki_rate * self.i_rate
        I = self.m.inertia
        return I @ alpha + np.cross(rate, I @ rate)

    def attitude_rates(self, R, z_des_world) -> np.ndarray:
        """Body-rate command that rotates body z onto z_des (+ yaw hold)."""
        k = self.k
        zb = R.T @ z_des_world                      # desired z axis in body frame
        e3 = np.array([0.0, 0.0, 1.0])
        c = np.clip(e3 @ zb, -1.0, 1.0)
        angle = np.arccos(c)
        axis = np.cross(e3, zb)
        n = np.linalg.norm(axis)
        axis = axis / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])  # exactly inverted: pick pitch
        rate = k.kp_att * angle * axis
        rate[:2] = np.clip(rate[:2], -k.rate_limit_rp, k.rate_limit_rp)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        # yaw correction only once roughly upright (yaw is ill-defined when inverted)
        w_yaw = np.clip((R[2, 2] - 0.5) / 0.5, 0.0, 1.0)
        rate[2] = np.clip(w_yaw * k.kp_yaw * wrap_pi(self.yaw_ref - yaw), -k.rate_limit_yaw, k.rate_limit_yaw)
        return rate

    def step(self, obs) -> np.ndarray:
        pos, quat, vel, rate, _ = split_obs(obs)
        if self.pos_ref is None:
            self.pos_ref = pos.copy()
        R = quat_to_rotmat(quat)
        F = self.desired_force(pos, vel)
        z_des = -F / np.linalg.norm(F)              # body z points DOWN, thrust along -z
        T = float(F @ (-R[:, 2]))                   # project onto current thrust axis
        T = np.clip(T, self.k.min_thrust_frac * self.m.mass * self.m.g, self.m.max_thrust)
        rate_des = self.attitude_rates(R, z_des)
        tau = self.rate_to_torque(rate_des, rate)
        self.last = {"T": T, "tau": tau, "rate_des": rate_des}
        return self.mixer.action(T, tau)


class PIDHoverController(Controller):
    """Holds the pose it wakes up in (the reset pose). Never flips: the lower
    bound on flip metrics, the upper bound on smoothness/altitude hold."""

    name = "pid_hover"

    def __init__(self, model: QuadModel, dt: float, gains: Optional[PIDGains] = None):
        self.pid = CascadedPID(model, dt, gains)

    def reset(self, seed: Optional[int] = None) -> None:
        self.pid.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.pid.step(obs)

    @property
    def diagnostics(self) -> Dict:
        return {"phase": "hover"}


# --------------------------------------------------------------------------- #
# 3) Scripted feed-forward flip — the required rule-based baseline            #
# --------------------------------------------------------------------------- #
@dataclass
class FlipScript:
    """All tunables of the scripted flip, in physical units. Defaults were tuned
    with ``run_baselines.py --tune`` on a seed bank disjoint from evaluation."""
    axis: str = "pitch"               # "pitch" (rotation about body y) or "roll"
    direction: int = +1               # +1 nose-up (back-flip), -1 nose-down
    settle_time: float = 0.50         # s   PID hover before anything happens
    # pre-pulse bias: climb so the flip has altitude to spend
    climb_time: float = 0.30          # s
    climb_thrust_frac: float = 0.90   # of max collective thrust
    # open-loop torque pulse
    pulse_torque_frac: float = 1.00   # of max body torque on the flip axis
    pulse_time: float = 0.06          # s
    flip_thrust_frac: float = 0.20    # collective during the rotation, of hover thrust
    # coast, then an opposite (braking) pulse
    brake_angle_deg: float = 280.0    # start braking at this integrated angle
    brake_torque_frac: float = 0.80
    brake_time: float = 0.08          # s (ends early at the hand-off trigger)
    # hand-off trigger to the PID recovery block
    handoff_angle_deg: float = 320.0  # integrated angle ...
    handoff_tilt_deg: float = 60.0    # ... OR tilt from upright once past inverted
    off_axis_damping: float = 0.15    # s; rate damping on the two non-flip axes
    max_flip_time: float = 1.5        # s   safety net: hand off no matter what

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "FlipScript":
        keys = FlipScript.__dataclass_fields__.keys()
        return FlipScript(**{k: v for k, v in d.items() if k in keys})


class ScriptedFlipController(Controller):
    """Phase machine:

        SETTLE --t--> CLIMB --t--> PULSE --t--> COAST --angle--> BRAKE --angle/tilt--> RECOVER

    PULSE/COAST/BRAKE are *open loop on the flip axis* (pure feed-forward torque
    through the mixer); only the two other axes receive light rate damping so the
    rotation stays in one plane. The rotation is tracked by integrating the body
    rate on the flip axis — the same quantity the §04 detector integrates — and
    RECOVER hands the vehicle to the cascaded PID, holding the hand-off position.
    """

    name = "scripted_flip"
    PHASES = ("settle", "climb", "pulse", "coast", "brake", "recover")

    def __init__(self, model: QuadModel, dt: float, script: Optional[FlipScript] = None,
                 gains: Optional[PIDGains] = None):
        self.m = model
        self.dt = float(dt)
        self.s = script or FlipScript()
        if self.s.axis not in AXIS_INDEX:
            raise ValueError(f"axis must be one of {list(AXIS_INDEX)}")
        self.ax = AXIS_INDEX[self.s.axis]
        self.pid = CascadedPID(model, dt, gains)
        self.mixer = self.pid.mixer
        self.reset()

    # -- bookkeeping ----------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        self.pid.reset()
        self.phase = "settle"
        self.t = 0.0
        self.t_phase = 0.0
        self.angle = 0.0                 # integrated rotation on the flip axis [rad], signed by direction
        self.prev_rate = None
        self.passed_inverted = False
        self.flip_start_pos = None
        self.handoff_t = None

    def _goto(self, phase: str) -> None:
        self.phase = phase
        self.t_phase = 0.0

    @property
    def diagnostics(self) -> Dict:
        return {"phase": self.phase, "phase_id": self.PHASES.index(self.phase),
                "flip_angle_deg": float(np.degrees(self.angle)),
                "passed_inverted": self.passed_inverted}

    # -- open-loop wrench during the manoeuvre --------------------------------
    def _flip_wrench(self, rate, torque_frac: float, sign: float):
        s, m = self.s, self.m
        tau = np.zeros(3)
        tau[self.ax] = sign * s.direction * torque_frac * m.max_torque(self.ax)
        # light damping on the off-axes (feedback, but NOT on the flip axis)
        I = np.diag(m.inertia)
        for j in range(3):
            if j != self.ax:
                tau[j] = -I[j] * rate[j] / max(s.off_axis_damping, 1e-3)
        T = s.flip_thrust_frac * m.mass * m.g
        return T, tau

    # -- main -----------------------------------------------------------------
    def act(self, obs: np.ndarray) -> np.ndarray:
        s, m = self.s, self.m
        pos, quat, vel, rate, _ = split_obs(obs)
        R = quat_to_rotmat(quat)

        # integrate rotation on the flip axis (trapezoidal), from the pulse onward
        w = rate[self.ax] * s.direction
        if self.phase in ("pulse", "coast", "brake") and self.prev_rate is not None:
            self.angle += 0.5 * (w + self.prev_rate) * self.dt
        self.prev_rate = w
        if R[2, 2] < -0.5:               # tilt > 120 deg: we are through the inverted half
            if self.phase in ("pulse", "coast", "brake"):
                self.passed_inverted = True
        tilt = np.degrees(tilt_angle(R))

        # ---- phase transitions ----
        if self.phase == "settle" and self.t_phase >= s.settle_time:
            self._goto("climb" if s.climb_time > 0 else "pulse")
            self.flip_start_pos = pos.copy()
        if self.phase == "climb" and self.t_phase >= s.climb_time:
            self._goto("pulse")
        if self.phase == "pulse" and self.t_phase >= s.pulse_time:
            self._goto("coast")
        if self.phase == "coast" and np.degrees(self.angle) >= s.brake_angle_deg:
            self._goto("brake")
        if self.phase in ("pulse", "coast", "brake"):
            handoff = (np.degrees(self.angle) >= s.handoff_angle_deg
                       or (self.passed_inverted and tilt <= s.handoff_tilt_deg
                           and np.degrees(self.angle) >= 270.0))
            if self.phase == "brake" and self.t_phase >= s.brake_time:
                handoff = handoff or np.degrees(self.angle) >= 270.0
            flip_elapsed = self.t - (s.settle_time + s.climb_time)
            if handoff or flip_elapsed >= s.max_flip_time:
                self._goto("recover")
                self.handoff_t = self.t
                ref = pos.copy()
                self.pid.set_reference(ref, yaw_ref=0.0, reset_integrators=True)

        # ---- phase outputs ----
        if self.phase == "settle":
            a = self.pid.step(obs)
        elif self.phase == "climb":
            # keep attitude with the PID inner loops, but command a strong climb
            rate_des = self.pid.attitude_rates(R, np.array([0.0, 0.0, 1.0]))
            tau = self.pid.rate_to_torque(rate_des, rate)
            a = self.mixer.action(s.climb_thrust_frac * m.max_thrust, tau)
        elif self.phase == "pulse":
            a = self.mixer.action(*self._flip_wrench(rate, s.pulse_torque_frac, +1.0))
        elif self.phase == "coast":
            a = self.mixer.action(*self._flip_wrench(rate, 0.0, +1.0))
        elif self.phase == "brake":
            a = self.mixer.action(*self._flip_wrench(rate, s.brake_torque_frac, -1.0))
        else:  # recover
            a = self.pid.step(obs)

        self.t += self.dt
        self.t_phase += self.dt
        return a


# --------------------------------------------------------------------------- #
# Factory — one entry point for the §05 evaluation harness                    #
# --------------------------------------------------------------------------- #
BASELINES = ("random", "pid_hover", "scripted_flip")


def make_controller(name: str, env, *, seed: int = 0, script: Optional[FlipScript] = None,
                    gains: Optional[PIDGains] = None, **kwargs) -> Controller:
    """Build a baseline controller configured for ``env`` (model + decision period
    are read from the env, so the same call works for any action-repeat)."""
    model = QuadModel.from_env(env)
    dt = control_dt_from_env(env)
    if name == "random":
        return RandomController(seed=seed, hover_action=model.hover_action, **kwargs)
    if name == "pid_hover":
        return PIDHoverController(model, dt, gains)
    if name == "scripted_flip":
        return ScriptedFlipController(model, dt, script, gains)
    raise ValueError(f"unknown baseline '{name}', choose from {BASELINES}")
