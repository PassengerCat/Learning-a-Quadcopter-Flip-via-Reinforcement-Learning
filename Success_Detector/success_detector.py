"""
success_detector.py
===================

§04 — Independent geometric SUCCESS CRITERION for the Quadcopter Flip
(Intelligent Control final project).

This module is the AUTHORITATIVE, reward-independent definition of "did the
vehicle perform the maneuver". The brief requires that success is defined
*geometrically* and NEVER by the reward. Nothing here imports or reads a reward.
It is also policy-independent: the criterion
is fixed up front from the physics of the task, not fitted to whatever a trained
policy happens to do.

It is fully SELF-CONTAINED (numpy only), so it can be used by the evaluation
harness (§05) or run stand-alone on a logged trajectory.

What counts as success (all must hold):
  0. started_upright    — the episode starts upright (tilt < cone).
  1. reached_inverted   — the vehicle genuinely went upside down at least once
                          (min projected-gravity gz <= inverted_gz, i.e. tilt >= 170deg,
                          also checked along the arc between samples; not exactly
                          180deg because the attitude is sampled every 5 ms while it
                          turns 4-9deg per step). Guards against faking a "revolution".
  2. completed_rotation — a full signed 360deg about the pitch axis, measured by a
                          accumulated local quaternion pitch increments.
  3. recovered + held   — afterwards, upright (tilt < cone) AND slow (|v|,|w| small)
                          held for `hold_steps` consecutive decisions (a real,
                          stabilized hover, not a momentary pass-through).
  4. no crash           — the episode never terminated unstable.
  5. in region          — never left the |z| altitude budget (nor went non-finite).
  6. one turn only      — net revolution within the cone of 360deg at the finish,
                          maximum revolution <= 390deg, |pitch| travel <= 420deg
                          during the maneuver (until upright again after the turn).

Two entry points:
  * `StreamingSuccessDetector` — call `.update(state, terminated=...)` once per
    step during a rollout (online).
  * `evaluate_episode(states, terminated=...)` — judge a whole logged trajectory
    at once (offline; for the §05 harness and failure-case reporting). Returns a
    `SuccessResult` that says not just pass/fail but WHY.

State parsing accepts, in order of preference:
  * an explicit (pos, quat, vel, omega) — pass a dict or a StateLike;
  * the 21-dim raw sim state  [pos(3) | quat(4) | vel(3) | omega(3) | motors(8)];
  * the 20-dim "full" obs     [tvel(3) | pos(3) | quat(4) | vel(3) | rates(3) | motor(4)].
Quaternion is scalar-first [w,x,y,z], body->world, NED. Upright -> gz=+1.

Author: (Intelligent Control project). License: MIT.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

# ----------------------------------------------------------------------------- #
# Quaternion geometry (pure)                                                     #
# ----------------------------------------------------------------------------- #

_UP = np.array([0.0, 0.0, 1.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    return np.array([1.0, 0.0, 0.0, 0.0]) if n < 1e-12 else q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def projected_gravity(q: np.ndarray) -> np.ndarray:
    """Projected-gravity unit vector (gx,gy,gz). Upright -> [0,0,1], inverted -> [0,0,-1].
        gx = 2(xz - wy),  gy = 2(yz + wx),  gz = 1 - 2(x^2 + y^2)."""
    w, x, y, z = quat_normalize(q)
    return np.array([2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)])


def tilt_angle(gz: float) -> float:
    """Geodesic tilt from upright: Theta = arccos(gz), radians in [0, pi]."""
    return float(np.arccos(np.clip(gz, -1.0, 1.0)))


def min_gz_on_arc(q0: np.ndarray, q1: np.ndarray, n: int = 8) -> float:
    """Lowest projected-gravity gz along the shortest rotation from q0 to q1 (slerp,
    n sub-steps). Makes the inversion check independent of the sampling rate: at
    50 Hz a flip turns ~23 deg between samples, so the sampled tilt alone can miss 170 deg."""
    q0, q1 = quat_normalize(q0), quat_normalize(q1)
    c = float(np.dot(q0, q1))
    if c < 0.0:                       # q and -q are the same attitude: take the short way
        q1, c = -q1, -c
    theta = float(np.arccos(min(c, 1.0)))
    best = 1.0
    for s in np.linspace(0.0, 1.0, n + 1):
        if theta < 1e-9:
            q = q0
        else:
            q = (np.sin((1 - s) * theta) * q0 + np.sin(s * theta) * q1) / np.sin(theta)
        best = min(best, float(projected_gravity(q)[2]))
    return best


def incremental_pitch(q_prev: np.ndarray, q_curr: np.ndarray) -> float:
    """Signed incremental pitch (body-y) rotation between consecutive quaternions
    (rad). Assumes less than pi rotation between samples. This avoids Euler
    singularities; it is a local body-y measure, not a general fixed-axis winding
    number for arbitrary 3D tumbles. dq = conj(q_prev)*q_curr, shortest-path."""
    q_prev = quat_normalize(q_prev)
    q_curr = quat_normalize(q_curr)
    if float(np.dot(q_prev, q_curr)) < 0.0:      # double-cover: shortest path
        q_curr = -q_curr
    dq = quat_mul(quat_conj(q_prev), q_curr)
    return float(2.0 * np.arctan2(dq[2], dq[0]))  # rotation about body-y


# ----------------------------------------------------------------------------- #
# State parsing (self-contained; no env / reward dependency)                     #
# ----------------------------------------------------------------------------- #

def parse_state(state, env=None):
    """Return (pos, quat, vel, omega) from any supported representation.

    Preference: explicit env.quad -> dict/attributes -> 21-dim raw -> 20-dim obs.
    """
    # 1) live env with a .quad (online use inside the sim)
    if env is not None:
        quad = getattr(env, "quad", None)
        if quad is not None and all(hasattr(quad, a) for a in ("pos", "quat", "vel", "omega")):
            return (np.asarray(quad.pos, float), np.asarray(quad.quat, float),
                    np.asarray(quad.vel, float), np.asarray(quad.omega, float))
    # 2) mapping / object carrying the four fields
    if isinstance(state, dict):
        return (np.asarray(state["pos"], float), np.asarray(state["quat"], float),
                np.asarray(state["vel"], float), np.asarray(state["omega"], float))
    if all(hasattr(state, a) for a in ("pos", "quat", "vel", "omega")):
        return (np.asarray(state.pos, float), np.asarray(state.quat, float),
                np.asarray(state.vel, float), np.asarray(state.omega, float))
    # 3) flat vector
    x = np.asarray(state, dtype=float).ravel()
    if x.size >= 21:                              # raw sim state
        return x[0:3], x[3:7], x[7:10], x[10:13]
    if x.size >= 20:                              # 20-dim "full" obs
        return x[3:6], x[6:10], x[10:13], x[13:16]
    raise ValueError(f"Cannot parse state of size {x.size}; expected env.quad, "
                     f"a (pos,quat,vel,omega) mapping, or a >=20-dim vector.")


# ----------------------------------------------------------------------------- #
# Configuration                                                                  #
# ----------------------------------------------------------------------------- #

@dataclass
class SuccessConfig:
    flip_direction: float = +1.0                 # +1 => a positive (body-y) revolution
    full_rev_rad: float = 2.0 * np.pi - 0.15     # a genuine 360deg (small margin)
    cone_rad: float = np.deg2rad(15.0)           # upright tolerance (tilt < cone)
    v_thr: float = 0.4                           # m/s  : "slow" linear speed
    w_thr: float = 0.8                           # rad/s: "slow" body-rate magnitude
    hold_steps: int = 20                         # consecutive decisions upright+slow must hold
    # --- hardening (all default ON) ---
    require_passed_inverted: bool = True         # must have genuinely gone upside down
    inverted_gz: float = float(np.cos(np.deg2rad(170.0)))  # gz <= this is "inverted" (tilt >= 170deg; was 120deg)
    forbid_crash: bool = True                    # any unstable termination => failure
    enforce_region: bool = True                  # leaving |z| budget => failure
    region_abs_z: float = 5.0                    # matches env max_abs_z (NED, metres)
    require_initial_upright: bool = True         # PDF p.6: a flip starts upright
    max_rev_rad: float = 2.0 * np.pi + np.deg2rad(30.0)
    max_pitch_travel_rad: float = 2.0 * np.pi + np.deg2rad(60.0)
    # PDF gives no numeric overshoot limit. Allow 30deg overshoot and the
    # corresponding 30deg return (420deg total travel), but never a second turn.
    travel_until_upright: bool = True
    # The travel budget measures the MANEUVER: |pitch| travel is counted until the first
    # sample that has completed the revolution AND is upright again (tilt < cone). Hover
    # jitter afterwards (e.g. under actuator/sensor noise) is not a rotation; a second turn
    # after that point is still caught by max_rev_rad and by the final one-turn check, and
    # stability by the final hold. False = count over the whole episode (behaviour before
    # 2026-09-29).

    def validate(self) -> None:
        assert self.flip_direction in (+1.0, -1.0)
        assert 0 < self.full_rev_rad
        assert self.full_rev_rad < 2.0 * np.pi < self.max_rev_rad < 4.0 * np.pi
        assert self.max_rev_rad <= self.max_pitch_travel_rad < 4.0 * np.pi
        assert 0.0 < self.cone_rad < np.pi / 2.0
        assert self.v_thr > 0.0 and self.w_thr > 0.0 and self.region_abs_z > 0.0
        assert self.hold_steps >= 1
        assert -1.0 <= self.inverted_gz <= 1.0


# ----------------------------------------------------------------------------- #
# Result (rich: says WHY, for reporting failure cases)                           #
# ----------------------------------------------------------------------------- #

@dataclass
class SuccessResult:
    success: bool = False
    # per-criterion booleans
    reached_inverted: bool = False
    completed_rotation: bool = False
    recovered_and_held: bool = False
    crashed: bool = False
    left_region: bool = False
    task: str = "flip"
    started_upright: bool = False
    excessive_rotation: bool = False
    within_one_turn: bool = False
    # diagnostics
    revolution_rad: float = 0.0
    max_revolution_rad: float = 0.0
    pitch_travel_rad: float = 0.0          # travel counted against the budget (see travel_until_upright)
    pitch_travel_total_rad: float = 0.0    # diagnostic: |pitch| travel over the whole episode
    min_gz: float = 1.0
    final_tilt_deg: float = 180.0
    best_hold: int = 0
    steps: int = 0

    def reason(self) -> str:
        """One-line human-readable explanation (esp. for failures)."""
        if self.success:
            return f"SUCCESS: {self.task} + final upright hover, no crash, in region."
        why = []
        if self.crashed:
            why.append("crashed (unstable termination)")
        if self.left_region:
            why.append("left |z| region")
        if self.task == "flip":
            if not self.started_upright:
                why.append("did not start upright")
            if not self.reached_inverted:
                why.append(f"never inverted (min gz={self.min_gz:+.2f})")
            if not self.completed_rotation:
                why.append(f"rotation incomplete ({np.degrees(self.revolution_rad):.0f}deg)")
            if self.excessive_rotation:
                why.append("exceeded one-flip rotation/travel budget")
            if self.completed_rotation and not self.within_one_turn:
                why.append("did not finish near one signed revolution")
        if not self.recovered_and_held:
            why.append(f"did not finish with held upright hover (best hold={self.best_hold}, "
                       f"final tilt={self.final_tilt_deg:.0f}deg)")
        return "FAIL: " + "; ".join(why) if why else "FAIL: criteria not met."

    def as_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- #
# Streaming (online) detector                                                    #
# ----------------------------------------------------------------------------- #

class StreamingSuccessDetector:
    """Online success detector. `reset()` per episode, then `update(state, terminated)`
    once per decision (post action-repeat) with the same state the policy sees.
    Feed the reset state with count_hold=False before the first action. Success
    requires a dwell ending at the current/final sample (PDF p.6); crash, region
    and excess-rotation violations latch as failures for the whole episode."""

    def __init__(self, config: Optional[SuccessConfig] = None):
        self.cfg = config or SuccessConfig()
        self.cfg.validate()
        self.reset()

    def reset(self) -> None:
        self._q_prev: Optional[np.ndarray] = None
        self._rev = 0.0
        self._max_rev = 0.0
        self._pitch_travel = 0.0
        self._pitch_travel_total = 0.0
        self._maneuver_done = False
        self._excessive_rotation = False
        self._started_upright = None
        self._min_gz = 1.0
        self._hit_inverted = False
        self._rev_done = False
        self._crashed = False
        self._left_region = False
        self._hold = 0
        self._best_hold = 0
        self._steps = 0
        self._final_tilt = np.pi

    def update(self, state, terminated: Optional[bool] = None, env=None,
               *, count_hold: bool = True) -> bool:
        cfg = self.cfg
        self._steps += 1
        pos, quat, vel, omega = parse_state(state, env)
        d = cfg.flip_direction

        # non-finite anywhere -> treat as crash-out
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))
                and np.all(np.isfinite(vel)) and np.all(np.isfinite(omega))
                and np.linalg.norm(quat) > 1e-12):
            self._crashed = True
            self._hold = 0
            return False

        # Accumulate local geometric pitch increments and absolute travel.
        arc_gz = 1.0
        if self._q_prev is not None:
            arc_gz = min_gz_on_arc(self._q_prev, quat)   # tilt reached BETWEEN samples
            increment = d * incremental_pitch(self._q_prev, quat)
            self._rev += increment
            self._pitch_travel_total += abs(increment)
            if not (cfg.travel_until_upright and self._maneuver_done):
                self._pitch_travel += abs(increment)
        self._max_rev = max(self._max_rev, self._rev)
        self._excessive_rotation = bool(
            self._excessive_rotation or self._max_rev > cfg.max_rev_rad + 1e-9
            or self._pitch_travel > cfg.max_pitch_travel_rad + 1e-9)
        self._q_prev = quat.copy()

        g = projected_gravity(quat)
        gz = float(g[2])
        self._min_gz = min(self._min_gz, gz, arc_gz)
        self._final_tilt = tilt_angle(gz)
        if self._started_upright is None:
            self._started_upright = self._final_tilt < cfg.cone_rad

        # crash / region latches (any occurrence dooms the episode)
        if terminated is None and env is not None:
            unstable = getattr(env, "_is_unstable", None)
            terminated = (bool(unstable()) if callable(unstable) else
                          bool(getattr(env, "terminated", False)
                               or getattr(env, "_terminated", False)))
        if terminated:
            self._crashed = True
        if cfg.enforce_region and abs(float(pos[2])) > cfg.region_abs_z:
            self._left_region = True

        # milestones
        if min(gz, arc_gz) <= cfg.inverted_gz:
            self._hit_inverted = True
        # _rev already accumulates in the intended direction (d * increment),
        # so it is >= 0 for a correct flip; compare directly to +full_rev_rad.
        if not self._rev_done and self._rev >= cfg.full_rev_rad:
            self._rev_done = True
        if self._rev_done and self._final_tilt < cfg.cone_rad:
            self._maneuver_done = True                  # travel budget stops counting after this sample

        # dwell: upright + slow, only after a completed (and genuine) flip
        ready = (self._rev_done and self._within_one_turn
                 and (self._hit_inverted or not cfg.require_passed_inverted)
                 and (self._started_upright or not cfg.require_initial_upright))
        if ready and not self._blocked and count_hold:
            upright = self._final_tilt < cfg.cone_rad
            slow = (float(np.linalg.norm(vel)) < cfg.v_thr
                    and float(np.linalg.norm(omega)) < cfg.w_thr)
            self._hold = self._hold + 1 if (upright and slow) else 0
        elif count_hold or self._blocked:
            self._hold = 0
        self._best_hold = max(self._best_hold, self._hold)

        return self.succeeded

    # -- read-outs ------------------------------------------------------------ #
    @property
    def succeeded(self) -> bool:
        return bool(self._hold >= self.cfg.hold_steps and not self._blocked)

    @property
    def _blocked(self) -> bool:
        return bool((self.cfg.forbid_crash and self._crashed)
                    or (self.cfg.enforce_region and self._left_region)
                    or self._excessive_rotation)

    @property
    def _within_one_turn(self) -> bool:
        return bool(abs(self._rev - 2.0 * np.pi) <= self.cfg.cone_rad)

    @property
    def revolution(self) -> float:
        return self._rev

    @property
    def passed_inverted(self) -> bool:
        return self._hit_inverted

    def result(self) -> SuccessResult:
        return SuccessResult(
            success=self.succeeded,
            task="flip",
            started_upright=bool(self._started_upright),
            excessive_rotation=self._excessive_rotation,
            within_one_turn=self._within_one_turn,
            reached_inverted=self._hit_inverted,
            completed_rotation=self._rev_done,
            recovered_and_held=(self._hold >= self.cfg.hold_steps),
            crashed=self._crashed,
            left_region=self._left_region,
            revolution_rad=self._rev,
            max_revolution_rad=self._max_rev,
            pitch_travel_rad=self._pitch_travel,
            pitch_travel_total_rad=self._pitch_travel_total,
            min_gz=self._min_gz,
            final_tilt_deg=float(np.degrees(self._final_tilt)),
            best_hold=self._best_hold,
            steps=self._steps,
        )


# ----------------------------------------------------------------------------- #
# Offline (batch) evaluation over a logged trajectory                            #
# ----------------------------------------------------------------------------- #

def evaluate_episode(states: Sequence,
                     terminated: Optional[Sequence[bool]] = None,
                     config: Optional[SuccessConfig] = None) -> SuccessResult:
    """Judge a whole episode from a logged sequence of states (any representation
    accepted by `parse_state`). `terminated[i]` (optional) marks the unstable-
    termination step. Returns a `SuccessResult` (use `.reason()` for why)."""
    det = StreamingSuccessDetector(config)
    n = len(states)
    for i in range(n):
        term = bool(terminated[i]) if terminated is not None else False
        det.update(states[i], terminated=term)
    return det.result()
