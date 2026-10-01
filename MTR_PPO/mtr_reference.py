"""MTR-PPO: reference for one pitch flip followed by hover.

The flip is described by desired *invariants* in the body frame, following GEAR
(arXiv 2602.10997), whose maneuver description and relative state are adapted here. For the flip (a loop
in the vertical plane of heading psi_des, pitch rate omega, radius r):

    vehicle -> loop centre  = r along the thrust axis        (body, NED/FRD: [0, 0, -r])
    velocity                = [d*omega*r, 0, 0]              (body)
    body rates              = [0, d*omega, 0]                (body)

with d = +1 for a nose-up flip. These are exactly the kinematics of a loop at constant rate:
the body-frame vector to a fixed centre, rotated by the body rates, reproduces the velocity.
For hover the vehicle should be at the start position, at rest.

Relative state (the policy observation and the reward's errors, as in GEAR), all in the body frame:

    p_rel = R^T (p_des - p),  v_rel = R^T v_des - R^T v,  w_rel = w_des - w,  R_rel = R^T R_z(psi_des)

where R is body->world. For the flip, p_des = c - r * t_hat is the point of the circle where the
vehicle should be for its current attitude (c = loop centre, t_hat = thrust direction in world);
for hover p_des = p0. The loop centre is fixed at the start: c = p0 + r * up. `state_at(theta)`
gives the reference state at loop phase theta (for starts on the reference). The flip ends
when the geometric revolution reaches 2*pi - eps (the same measure as the success detectors);
then the HOVER reference applies.

Conventions: NED world (z down), body FRD, scalar-first quaternion body->world (simulator).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UP = np.array([0.0, 0.0, -1.0])        # NED: up is -z
FLIP, HOVER = 0, 1                     # task indices (one-hot order in the observation)


@dataclass(frozen=True)
class FlipCommand:
    omega: float = 5.0                 # commanded pitch rate [rad/s]
    radius: float = 0.6                # loop radius [m]
    direction: float = +1.0            # +1 nose-up (the detectors' positive revolution)

    def __post_init__(self):
        if not (self.omega > 0 and self.radius > 0 and self.direction in (1.0, -1.0)):
            raise ValueError(f"invalid flip command {self}")


def rotation_matrix(quat) -> np.ndarray:
    """Body->world rotation matrix from a scalar-first quaternion (normalised first)."""
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw(quat) -> float:
    """Heading psi (ZYX Euler) from a scalar-first quaternion."""
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def rot_z(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class LoopReference:
    """Desired state of the flip-then-hover task; `start` fixes p0, psi_des and the centre."""

    def __init__(self, command: FlipCommand = FlipCommand()):
        self.command = command
        self.p0 = self.centre = None
        self.psi = 0.0

    def start(self, pos, quat) -> None:
        """Anchor the task at the current (hover) state: p0 = pos, psi_des = its heading."""
        self.anchor(pos, yaw(quat))

    def anchor(self, p0, psi: float) -> None:
        """Anchor the task explicitly: hover target / loop bottom p0 and flip-plane heading psi."""
        self.p0 = np.array(p0, dtype=float)
        self.psi = float(psi)
        self.centre = self.p0 + self.command.radius * UP

    def state_at(self, theta: float) -> dict:
        """Reference state at loop phase theta (0 = bottom, upright): pos, quat, vel, body rates."""
        if not self.started:
            raise RuntimeError("call start() or anchor() first")
        c = self.command
        half = 0.5 * c.direction * theta
        quat = quat_mul(np.array([np.cos(self.psi / 2), 0.0, 0.0, np.sin(self.psi / 2)]),
                        np.array([np.cos(half), 0.0, np.sin(half), 0.0]))
        R = rotation_matrix(quat)
        return dict(pos=self.centre + c.radius * R[:, 2], quat=quat,
                    vel=R @ np.array([c.direction * c.omega * c.radius, 0.0, 0.0]),
                    omega=np.array([0.0, c.direction * c.omega, 0.0]))

    @property
    def started(self) -> bool:
        return self.p0 is not None

    def relative_state(self, task: int, pos, quat, vel, omega) -> dict:
        """Relative state for the given task (FLIP or HOVER), all body-frame."""
        if not self.started:
            raise RuntimeError("call start() first")
        R = rotation_matrix(quat)
        p, v, w = (np.asarray(a, dtype=float) for a in (pos, vel, omega))
        c = self.command
        if task == FLIP:
            thrust_world = -R[:, 2]                                     # thrust along body -z
            p_des = self.centre - c.radius * thrust_world
            v_des_body = np.array([c.direction * c.omega * c.radius, 0.0, 0.0])
            w_des = np.array([0.0, c.direction * c.omega, 0.0])
        elif task == HOVER:
            p_des, v_des_body, w_des = self.p0, np.zeros(3), np.zeros(3)
        else:
            raise ValueError(f"unknown task {task}")
        return dict(p=R.T @ (p_des - p), v=v_des_body - R.T @ v, w=w_des - w,
                    R=R.T @ rot_z(self.psi))
