"""
flip_reward.py
==============

Reward function and independent success detector for the Quadcopter Flip RL task
(Intelligent Control final project, §03).

The vehicle starts from hover, performs a full 360 deg rotation about the body
pitch axis (body-y), passes through inverted, and recovers to a stable upright
hover. This module implements the reward architecture agreed in §03:

    r = r_shape  +  r_task  +  r_reg

  r_shape : potential-based shaping (Ng, Harada & Russell 1999). Dense; drives
            rotation progress (a tent: 0 -> w_rot at 360 deg, decreasing beyond),
            then upright/hover in RECOVER.
  r_task  : the "true" objective -- one-off milestones (inverted, full turn), a
            bounded "new record" progress term (w_prog), dense post-flip terms
            (b_hold: upright & slow cone; b_upright: smooth upright/slow pull) and a
            crash penalty.
  r_reg   : regularizers -- action smoothness (on the policy's own action),
            off-axis rates, horizontal position, altitude drift, and a survival
            bonus paid after the flip (alive_after_flip).

Every term that differs from the first §03 design was added in response to a
measured failure (reward runs 1-9, Table 6 of the report).

Design invariants (do NOT break):
  * The reward NEVER reads the success signal. `SuccessDetector` is a *separate*
    object; success is defined geometrically and independently (brief requirement).
  * Rotation progress (shaping, milestones, phase, success) is measured
    GEOMETRICALLY from the gravity direction in the body x-z plane
    (`gravity_pitch_increment`): drift-free, and immune to coning, which fooled
    both the integrated body rate and the body-y quaternion increment in run-4.
  * The shaping potential is a deterministic function of (augmented) state, so the
    telescoping argument holds for r_shape. `alpha` (or its trig encoding) must
    therefore be part of the policy observation (flip_policy.FlipProgressTracker
    reproduces it from raw observations). The w_prog / b_upright terms are NOT
    potentials -- deliberate, bounded, goal-aligned exceptions.

State layout
------------
`reward_fn(target_velocity, full_state, action, env)` is called once per sim
substep (the env sums the returns over an action-repeat block). `full_state` is
parsed by `StateView`, which supports:
  * the 20-dim "full" observation:
        [0:3] target_vel | [3:6] pos | [6:10] quat(w,x,y,z) | [10:13] lin_vel |
        [13:16] body rates (p,q,r) | [16:20] motor speeds
  * the 21-dim raw sim state:
        [0:3] pos | [3:7] quat | [7:10] lin_vel | [10:13] omega | [13:21] motors
  * falling back to `env.quad.{pos,quat,vel,omega}` when present.
Quaternion is scalar-first [w, x, y, z], body->world, NED. Upright -> gz = +1.

Usage
-----
    from flip_reward import FlipReward, SuccessDetector, FlipRewardConfig

    reward = FlipReward(FlipRewardConfig())          # one instance PER env
    env.reward_fn = reward                            # matches the hook signature
    # in the wrapper's reset():  reward.reset()

    detector = SuccessDetector()                      # one instance PER env
    # in the wrapper's reset():  detector.reset()
    # each step (once per decision):  ok = detector.update(full_state, env)

For vectorized envs, give every sub-env its own FlipReward / SuccessDetector.

Author: (Intelligent Control project). License: MIT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ----------------------------------------------------------------------------- #
# Geometry helpers (all pure functions)
# ----------------------------------------------------------------------------- #

_UP = np.array([0.0, 0.0, 1.0])  # projected-gravity target for "upright" (NED, verified)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return a unit quaternion; falls back to identity if degenerate."""
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate of a scalar-first quaternion [w,x,y,z]."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-first quaternions (a then b)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


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


def incremental_pitch(q_prev: np.ndarray, q_curr: np.ndarray) -> float:
    """
    Signed incremental rotation about body-y (pitch) between two consecutive
    quaternions, in radians. Small between steps, so it is unambiguous (< pi) --
    accumulating it yields a drift-free, singularity-free revolution count.

    We take the body-frame delta dq = conj(q_prev) * q_curr, enforce the
    shortest-path hemisphere (double-cover), and read the y-component rotation as
    2*atan2(dq_y, dq_w).
    """
    q_prev = quat_normalize(q_prev)
    q_curr = quat_normalize(q_curr)
    dq = quat_mul(quat_conj(q_prev), q_curr)
    if dq[0] < 0.0:            # shortest path: q and -q are the same rotation
        dq = -dq
    return 2.0 * float(np.arctan2(dq[2], dq[0]))   # dq = [w, x, y, z] -> y is index 2


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


# ----------------------------------------------------------------------------- #
# State parsing
# ----------------------------------------------------------------------------- #

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


# ----------------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------------- #

@dataclass
class FlipRewardConfig:
    """All reward weights / gains. Defaults = the §03 first-run configuration."""

    # --- timing / dynamics ---
    dt: float = 0.005                 # sim substep (s); overridden by env.dt if present
    gamma: float = 0.99               # PPO discount (informational; used only if gamma_shape mirrors it)
    gamma_shape: float = 1.0          # discount INSIDE the shaping term F = gamma_shape*Phi' - Phi.
                                      # 1.0 = "difference form": constant Phi -> 0 reward, so holding at
                                      # the goal is not bled by (gamma-1)*Phi. Set = gamma for the strict
                                      # Ng-Harada-Russell form (then raise b_hold to beat the bleed).
    flip_direction: float = +1.0      # +1 => reward positive pitch rotation

    # --- potential Phi ---
    w_rot: float = 1.0                # height of the rotation-progress ramp (0 -> w_rot over 0..2pi)
    w_set: float = 1.0                # height of the settle (recover) term
    eps_full: float = 0.15            # margin (rad) below 2*pi that counts as "rotation complete"
    over_rot_slope: float = 1.0       # beyond 2*pi the rotation potential DECREASES (tent):
                                      # one extra full turn gives back the whole w_rot. 0 = old
                                      # behaviour (flat cap). Added after run-1, whose policy
                                      # learned to spin continuously ("spin-and-crash").

    # --- settle(s) Gaussian sharpness (per-quantity) ---
    k_g: float = 5.0                  # attitude  e^(-k_g * ||g - g_up||^2)
    k_v: float = 0.5                  # linear velocity (softened vs hover-tuned 5.0 for earlier gradient)
    k_w: float = 0.1                  # body-rate magnitude (softened vs 0.5)
    k_p: float = 1.0                  # horizontal position

    # --- exploration aid: "new record" rotation progress (added after run-3) ---
    w_prog: float = 1.0               # paid ONLY when alpha exceeds its previous episode maximum,
                                      # capped at 2*pi -> total <= w_prog per episode; wobbling
                                      # earns nothing twice. Not a potential (breaks strict policy
                                      # invariance) but aligned with the goal: it rewards trying
                                      # ever-larger rotations, which the potential alone does not
                                      # (tilt-and-return nets zero shaping).

    # --- task (sparse-ish) ---
    b_inv: float = 0.5                # one-off, first time |revolution| passes pi (inverted)
    b_full: float = 2.0              # one-off, first time |revolution| passes 2*pi - eps_full
    b_hold: float = 1.0               # dense, per SECOND, while RECOVER & upright-cone & slow
    b_upright: float = 1.0            # dense, per SECOND, in RECOVER: exp(-k_g|g-up|^2)*exp(-k_up_v|v|^2)
    k_up_v: float = 0.05              #   *exp(-k_up_w|w|^2). Smooth pull towards upright & slow from
    k_up_w: float = 0.02              #   ANY post-flip state (run-7 flew off tilted at 14 m/s: the
                                      #   binary hold bonus gave no gradient that far from hover)
    k_up_g: float = 5.0               # attitude sharpness inside b_upright (run-9 = k_g = 5; run-11
                                      #   tried 2.0: holding position in wind needs ~10-15 deg of lean)
    k_up_p: float = 0.0               # ... *exp(-k_up_p |xy - xy0|^2) "near where you started"
                                      #   (run-9 = 0; run-11 tried 0.1)
    b_crash: float = 5.0              # one-off penalty on unstable termination

    # thresholds for the *reward-internal* hold bonus (NOT the official success test)
    hold_cone_rad: float = np.deg2rad(20.0)
    hold_v_thr: float = 0.5           # m/s
    hold_w_thr: float = 1.0           # rad/s

    # --- regularizers ---
    lam_dact: float = 0.005           # action-smoothness ||a_t - a_{t-1}||^2 (per decision);
                                      # 0.02 made exploration noise alone cost ~3.5/episode (run-3)
    lam_off: float = 0.01             # off-axis rate penalty (p^2 + r^2), per SECOND
    lam_pos: float = 0.005            # in-place L1 horizontal position, per SECOND
    lam_z: float = 0.05               # altitude drift |z - z0|, per SECOND (run-3 policy climbed out)
    s_alive: float = 0.0              # survival bonus per SECOND (first run = 0; ablate {0,0.005,0.02})
    alive_after_flip: bool = False    # pay s_alive only in RECOVER: hovering WITHOUT flipping then
                                      # earns nothing (run-5 settled on exactly that)

    def validate(self) -> None:
        assert self.dt > 0 and self.gamma > 0
        assert self.flip_direction in (+1.0, -1.0)
        for name in ("k_g", "k_v", "k_w", "k_p"):
            assert getattr(self, name) >= 0.0, f"{name} must be >= 0"


# ----------------------------------------------------------------------------- #
# Reward
# ----------------------------------------------------------------------------- #

# phase codes
_FLIP, _RECOVER = 0, 1


class FlipReward:
    """
    Stateful, per-env reward callable matching the hook
    `reward_fn(target_velocity, full_state, action, env) -> float`.

    Call `reset()` at the start of each episode (from the env wrapper's reset).
    One instance per environment; never share across parallel envs.
    """

    def __init__(self, config: Optional[FlipRewardConfig] = None):
        self.cfg = config or FlipRewardConfig()
        self.cfg.validate()
        self.reset()

    # -- episode lifecycle ---------------------------------------------------- #
    def reset(self) -> None:
        self._alpha = 0.0             # geometric flip angle (gravity-based), used by the shaping
        self._rev = 0.0               # same measure, used for phase transitions / milestones
        self._phi_prev: Optional[float] = None
        self._q_prev: Optional[np.ndarray] = None
        self._prev_action: Optional[np.ndarray] = None
        self._max_alpha = 0.0         # episode maximum of alpha (for w_prog)
        self._z0: Optional[float] = None
        self._xy0: Optional[np.ndarray] = None     # start position (for b_upright "home")
        self._phase = _FLIP
        self._hit_inv = False         # milestone latch: passed pi
        self._hit_full = False        # milestone latch: passed 2pi - eps
        self._paid_inv = False        # b_inv paid at most once per episode
        self._paid_full = False       # b_full paid at most once per episode
        self._done_latched = False    # crash penalty paid at most once
        # cached geometry from the last call (for logging / eval only)
        self._last_gz = 1.0
        self._last_speed = 0.0
        self._last_wnorm = 0.0
        # When the policy does not command the motors directly (CTBR), the wrapper sets
        # the policy's own action here so the smoothness term penalises what the
        # policy controls, not the inner loop's motor activity.
        self.smooth_action: Optional[np.ndarray] = None

    # -- main hook ------------------------------------------------------------ #
    def __call__(self, target_velocity, full_state, action, env=None) -> float:
        cfg = self.cfg
        dt = float(getattr(env, "dt", cfg.dt))
        sv = StateView(full_state, env)
        d = cfg.flip_direction
        if self._xy0 is None:
            self._xy0 = np.array(sv.pos[0:2], dtype=float)

        # --- accumulate the geometric flip angle ----------------------------- #
        if self._q_prev is not None:
            inc = d * gravity_pitch_increment(self._q_prev, sv.quat)
            self._rev += inc
            self._alpha = self._rev
        self._q_prev = sv.quat.copy()

        # cache geometry for logging / the eval-time success check (not used by r)
        self._last_gz = float(projected_gravity(sv.quat)[2])
        self._last_speed = float(np.linalg.norm(sv.vel))
        self._last_wnorm = float(np.linalg.norm(sv.omega))

        # --- geometric phase machine (transitions on physical conditions) --- #
        if not self._hit_inv and self._rev >= np.pi:
            self._hit_inv = True
        if self._phase == _FLIP and self._rev >= (2.0 * np.pi - cfg.eps_full):
            self._phase = _RECOVER
            self._hit_full = True

        # --- shaping: r_shape = gamma * Phi(s') - Phi(s) -------------------- #
        phi = self._potential(sv)
        if self._phi_prev is None:            # first call after reset: no jump
            r_shape = 0.0
        else:
            r_shape = cfg.gamma_shape * phi - self._phi_prev
        self._phi_prev = phi

        # --- task ----------------------------------------------------------- #
        r_task = 0.0
        new_max = min(max(self._max_alpha, self._alpha), 2.0 * np.pi)
        if new_max > self._max_alpha:
            r_task += cfg.w_prog * (new_max - self._max_alpha) / (2.0 * np.pi)
            self._max_alpha = new_max
        # milestones fire exactly once (the latches were just set above)
        if self._hit_inv and not getattr(self, "_paid_inv", False):
            r_task += cfg.b_inv
            self._paid_inv = True
        if self._hit_full and not getattr(self, "_paid_full", False):
            r_task += cfg.b_full
            self._paid_full = True
        # dense hold bonus while recovered, upright and slow (reward-internal thresholds)
        if self._phase == _RECOVER:
            g = projected_gravity(sv.quat)
            upright = tilt_angle(float(g[2])) < cfg.hold_cone_rad
            slow = (np.linalg.norm(sv.vel) < cfg.hold_v_thr
                    and np.linalg.norm(sv.omega) < cfg.hold_w_thr)
            if upright and slow:
                r_task += cfg.b_hold * dt
            if cfg.b_upright:
                att = np.exp(-cfg.k_up_g * float(np.sum((g - _UP) ** 2)))
                home = np.exp(-cfg.k_up_p * float(np.sum((sv.pos[0:2] - self._xy0) ** 2)))
                r_task += cfg.b_upright * dt * att * home \
                    * np.exp(-cfg.k_up_v * float(np.sum(sv.vel ** 2))) \
                    * np.exp(-cfg.k_up_w * float(np.sum(sv.omega ** 2)))
        # crash penalty (paid once); env exposes termination via _is_unstable/terminated
        if not self._done_latched and _is_terminated(env):
            r_task -= cfg.b_crash
            self._done_latched = True

        # --- regularizers --------------------------------------------------- #
        act = np.asarray(action if self.smooth_action is None else self.smooth_action,
                         dtype=float).ravel()
        if self._prev_action is None:
            self._prev_action = act.copy()
        # action-smoothness: within an action-repeat block the action is constant,
        # so ||a - a_prev||^2 is non-zero only at the decision boundary -> fires
        # once per decision automatically.
        r_reg = -cfg.lam_dact * float(np.sum((act - self._prev_action) ** 2))
        self._prev_action = act.copy()

        p, r_yaw = float(sv.omega[0]), float(sv.omega[2])
        r_reg += -cfg.lam_off * (p * p + r_yaw * r_yaw) * dt
        r_reg += -cfg.lam_pos * float(np.sum(np.abs(sv.pos[0:2]))) * dt
        if self._z0 is None:
            self._z0 = float(sv.pos[2])
        r_reg += -cfg.lam_z * abs(float(sv.pos[2]) - self._z0) * dt
        if not cfg.alive_after_flip or self._phase == _RECOVER:
            r_reg += cfg.s_alive * dt

        total = r_shape + r_task + r_reg
        if not np.isfinite(total):     # defensive: never poison the buffer
            return 0.0
        return float(total)

    # -- potential ------------------------------------------------------------ #
    def _potential(self, sv: StateView) -> float:
        cfg = self.cfg
        # rotation-progress ramp 0..2pi, then a tent: extra spin is paid back
        a, two_pi = self._alpha, 2.0 * np.pi
        if a <= two_pi:
            rot = cfg.w_rot * max(a, 0.0) / two_pi
        else:
            rot = cfg.w_rot * max(0.0, 1.0 - cfg.over_rot_slope * (a - two_pi) / two_pi)
        # settle only contributes in RECOVER
        if self._phase == _RECOVER:
            rot += cfg.w_set * self._settle(sv)
        return float(rot)

    def _settle(self, sv: StateView) -> float:
        cfg = self.cfg
        g = projected_gravity(sv.quat)
        att = np.exp(-cfg.k_g * float(np.sum((g - _UP) ** 2)))
        vel = np.exp(-cfg.k_v * float(np.sum(sv.vel ** 2)))
        omg = np.exp(-cfg.k_w * float(np.sum(sv.omega ** 2)))
        pos = np.exp(-cfg.k_p * float(np.sum(sv.pos[0:2] ** 2)))
        return 0.25 * (att + vel + omg + pos)     # soft-AND (average), for learnability

    # -- introspection (for logging / debugging; NOT used by the policy) ------ #
    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def revolution(self) -> float:
        return self._rev

    @property
    def phase(self) -> str:
        return "RECOVER" if self._phase == _RECOVER else "FLIP"

    @property
    def passed_inverted(self) -> bool:
        return self._hit_inv

    @property
    def last_tilt(self) -> float:
        """Tilt angle Θ = arccos(gz) from the most recent call (rad)."""
        return tilt_angle(self._last_gz)

    @property
    def last_speed(self) -> float:
        return self._last_speed

    @property
    def last_wnorm(self) -> float:
        return self._last_wnorm

    # -- observation features (MUST be appended to the policy obs) ------------ #
    N_OBS_FEATURES = 4

    def obs_features(self) -> np.ndarray:
        """Return [sin α, cos α, progress, phase] to concatenate onto the policy
        observation. Required for the shaping to be Markov: the potential Φ depends
        on the accumulated angle and phase, so the policy must observe them.
          sin α, cos α : angular position within a turn (periodic, unambiguous)
          progress     : clip(α,0,2π)/2π  -> monotone 0..1, disambiguates 0 from 2π
          phase        : 0.0 = FLIP, 1.0 = RECOVER
        """
        a = self._alpha
        progress = float(np.clip(a, 0.0, 2.0 * np.pi) / (2.0 * np.pi))
        phase_flag = 1.0 if self._phase == _RECOVER else 0.0
        return np.array([np.sin(a), np.cos(a), progress, phase_flag], dtype=np.float32)


# ----------------------------------------------------------------------------- #
# Independent success detector (must NOT be read by the reward)
# ----------------------------------------------------------------------------- #

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


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #

def _is_terminated(env) -> bool:
    """Best-effort read of the env's unstable-termination flag."""
    if env is None:
        return False
    for attr in ("_terminated", "terminated"):
        if bool(getattr(env, attr, False)):
            return True
    fn = getattr(env, "_is_unstable", None)
    if callable(fn):
        try:
            return bool(fn())                         # QuadcopterVelocityEnv._is_unstable(self)
        except TypeError:
            try:
                quad = getattr(env, "quad", None)
                return bool(fn(quad.state if quad is not None else None))
            except Exception:
                return False
        except Exception:
            return False
    return False


def make_flip_reward(**overrides) -> FlipReward:
    """Factory: FlipReward with config field overrides, e.g. make_flip_reward(s_alive=0.005)."""
    return FlipReward(FlipRewardConfig(**overrides))
