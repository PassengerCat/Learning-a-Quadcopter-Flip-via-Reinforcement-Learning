"""MTR-PPO: multiplicative tracking reward for one flip followed by hover.

    r = r_pos * r_lin * r_ang * r_cmd * r_task,          H(x; k) = 1 / (1 + k x)

    r_pos = sum_{k in {1, 10}}        H(|p_rel|^2; k)
    r_lin = sum_{k in {1, 10, 100}}   H(|v_rel|^2; k)
    r_ang = sum_{k in {0.1, 1, 10}}   H(|w_rel|^2; k)
    r_cmd = sum_{k in {1, 10}}        H((a_ach - a_cmd)^2; k)
    r_task = 2 during the flip, 1 during hover

The structure and the "narrow" k values are those of GEAR (arXiv 2602.10997: basic tracking
terms, command adherence and the flip's task term). The relative state comes from
mtr_reference.LoopReference. Commanded attribute:
during FLIP the pitch rate (a_ach = d*q, a_cmd = omega); during HOVER the number of turns
(a_ach = revolution / 2*pi, a_cmd = 1), which also discourages a second turn.

The task switches from FLIP to HOVER when the geometric revolution (gravity direction in the
body x-z plane, the same measure as the success detectors) reaches 2*pi - eps_full. It is
updated at every simulator step, together with the largest revolution reached so far
(rev_max; the training environment can end an episode that turns back, see mtr_env).
There is no crash penalty: a crash ends the episode and with
it the (positive) reward, as in GEAR.

Kernel sets (MTRRewardConfig.kernels): "narrow" = the k values above (GEAR's); "dense" = wider
kernels with the same structure, one coarse and one fine kernel per term, the coarse one at half
value at the typical error of a hover start: r_lin k in {0.1, 1} (half value at ~3.2 and 1 m/s),
r_ang and r_cmd k in {0.04, 0.25} (half value at 5 and 2 rad/s), r_pos unchanged {1, 10}.
From hover (0 % of the loop's speed and rate) "narrow" gives 0.01 % of the maximum and "dense"
3.2 %; at 50 %: 0.23 % vs 20 %; at 75 %: 2.7 % vs 55 %.

Optional post-flip survival bonus (b_alive_hover, per second, default 0 = off): during HOVER only,
i.e. once the flip is complete, every simulator step also earns b_alive_hover * u * dt with
u = (1 + cos(tilt)) / 2 (1 upright, 0 inverted). It pays for staying airborne and upright after
the flip; it cannot reward not flipping, since it only exists after the loop is complete.
With alive_speed_k = k > 0 the bonus is also multiplied by 1 / (1 + k |v|^2) (v = velocity), so
it pays for stopping, not just for staying upright (k = 1: 86 % at 0.4 m/s, 50 % at 1 m/s,
7.5 % at 3.5 m/s). k = 0 (default) leaves it as above.

Scaling: the product is divided by its maximum (flip, zero error) and multiplied by the step
time, so a perfect second of flight earns 1 (flip) or 0.5 (hover). A constant factor does not
change the optimal policy; it keeps returns on the scale the PPO settings were tuned for.

The class also provides what the PPO track's FlipTaskEnv reads from its reward (alpha, phase,
passed_inverted, cfg.flip_direction/eps_full/b_crash, reset, smooth_action, _prev_action,
_done_latched), so it can be installed in that environment unchanged.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RT = os.path.join(os.path.dirname(_HERE), "Reward_and_Training")   # the PPO track's modules
for _p in (_HERE, _RT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mtr_reference import FLIP, HOVER, FlipCommand, LoopReference         # noqa: E402
from flip_reward import gravity_pitch_increment, projected_gravity, tilt_angle  # noqa: E402

K_POS, K_LIN, K_ANG, K_CMD = (1.0, 10.0), (1.0, 10.0, 100.0), (0.1, 1.0, 10.0), (1.0, 10.0)   # GEAR's values
KERNEL_SETS = {
    "narrow": dict(pos=K_POS, lin=K_LIN, ang=K_ANG, cmd=K_CMD),
    "dense": dict(pos=(1.0, 10.0), lin=(0.1, 1.0), ang=(0.04, 0.25), cmd=(0.04, 0.25)),
}


def kernel_sum(x: float, ks) -> float:
    """sum_k H(x; k) with H(x; k) = 1 / (1 + k x), x >= 0."""
    return float(sum(1.0 / (1.0 + k * x) for k in ks))


@dataclass
class MTRRewardConfig:
    flip_direction: float = +1.0
    eps_full: float = 0.15            # revolution margin below 2*pi that completes the flip
    kernels: str = "narrow"          # "narrow" (GEAR's k values) or "dense" (wider kernels)
    r_task_flip: float = 2.0
    r_task_hover: float = 1.0
    b_crash: float = 0.0              # read by FlipTaskEnv on integrator failure; GEAR has none
    b_alive_hover: float = 0.0        # post-flip survival bonus per second (HOVER only); 0 = off
    alive_speed_k: float = 0.0        # bonus also weighted by 1 / (1 + k |v|^2); 0 = no speed weighting
    inverted_tilt_rad: float = np.deg2rad(150.0)   # privileged critic flag only

    @property
    def max_product(self) -> float:
        k = self.k
        return len(k["pos"]) * len(k["lin"]) * len(k["ang"]) * len(k["cmd"]) * self.r_task_flip

    @property
    def k(self) -> dict:
        if self.kernels not in KERNEL_SETS:
            raise ValueError(f"unknown kernel set {self.kernels!r}; choose from {list(KERNEL_SETS)}")
        return KERNEL_SETS[self.kernels]


class MultiplicativeTrackingReward:
    """Per-environment reward hook: reward_fn(target_velocity, full_state, action, env)."""

    def __init__(self, config: Optional[MTRRewardConfig] = None,
                 command: Optional[FlipCommand] = None):
        self.cfg = config or MTRRewardConfig()
        self.cfg.k                              # validates the kernel set
        self.command = command or FlipCommand(direction=self.cfg.flip_direction)
        if self.command.direction != self.cfg.flip_direction:
            raise ValueError("command direction must equal cfg.flip_direction")
        self.reset()

    # -- episode lifecycle ----------------------------------------------------
    def reset(self, command: Optional[FlipCommand] = None) -> None:
        if command is not None:
            if command.direction != self.cfg.flip_direction:
                raise ValueError("command direction must equal cfg.flip_direction")
            self.command = command
        self.ref = LoopReference(self.command)
        self._q_prev = None
        self._rev = 0.0
        self._rev_max = 0.0
        self._task = FLIP
        self._inverted = False
        self._done_latched = False
        self._prev_action = None           # FlipTaskEnv compatibility (no smoothness term here)
        self.smooth_action = None
        self.last_terms: dict = {}

    def begin(self, pos, quat, rev0: float = 0.0, anchor=None) -> None:
        """Fix the reference at the episode's start (call after every reset). For a start on the
        reference, pass anchor=(p0, psi) and the loop phase already completed as rev0."""
        if anchor is None:
            self.ref.start(pos, quat)
        else:
            self.ref.anchor(*anchor)
        self._q_prev = np.array(quat, dtype=float)
        self._rev = float(rev0)
        self._rev_max = self._rev
        self._inverted = self._rev >= self.cfg.inverted_tilt_rad       # tilt = phase on the loop

    # -- reward ---------------------------------------------------------------
    def __call__(self, target_velocity, full_state, action, env=None) -> float:
        q = env.quad
        pos, quat = np.asarray(q.pos, float), np.asarray(q.quat, float)
        vel, omega = np.asarray(q.vel, float), np.asarray(q.omega, float)
        if not self.ref.started:
            self.begin(pos, quat)
        self._rev += self.cfg.flip_direction * gravity_pitch_increment(self._q_prev, quat)
        self._q_prev = quat.copy()
        self._rev_max = max(self._rev_max, self._rev)
        tilt = tilt_angle(float(projected_gravity(quat)[2]))
        if tilt >= self.cfg.inverted_tilt_rad:
            self._inverted = True
        if self._task == FLIP and self._rev >= 2.0 * np.pi - self.cfg.eps_full:
            self._task = HOVER
        value = self.value(self._task, pos, quat, vel, omega, self._rev)
        if self._task == HOVER and self.cfg.b_alive_hover > 0.0:
            alive = self.cfg.b_alive_hover * 0.5 * (1.0 + np.cos(tilt))
            if self.cfg.alive_speed_k > 0.0:
                alive /= 1.0 + self.cfg.alive_speed_k * float(vel @ vel)
            self.last_terms["alive"] = alive
            value += alive
        dt = float(getattr(env, "dt", 0.005))
        r = value * dt
        return float(r) if np.isfinite(r) else 0.0

    def value(self, task: int, pos, quat, vel, omega, rev: float) -> float:
        """Normalised reward rate in [0, 1] for one state (1 = perfect flip tracking)."""
        c, cfg = self.command, self.cfg
        s = self.ref.relative_state(task, pos, quat, vel, omega)
        if task == FLIP:
            a_err = cfg.flip_direction * float(omega[1]) - c.omega
            r_task = cfg.r_task_flip
        else:
            a_err = rev / (2.0 * np.pi) - 1.0
            r_task = cfg.r_task_hover
        k = cfg.k
        terms = dict(pos=kernel_sum(float(s["p"] @ s["p"]), k["pos"]),
                     lin=kernel_sum(float(s["v"] @ s["v"]), k["lin"]),
                     ang=kernel_sum(float(s["w"] @ s["w"]), k["ang"]),
                     cmd=kernel_sum(a_err * a_err, k["cmd"]), task=r_task)
        self.last_terms = terms
        return float(np.prod(list(terms.values())) / cfg.max_product)

    # -- what FlipTaskEnv and logging read -----------------------------------------
    @property
    def alpha(self) -> float:
        return self._rev

    @property
    def revolution(self) -> float:
        return self._rev

    @property
    def rev_max(self) -> float:
        """Largest revolution reached so far in the episode (the phase for reference starts)."""
        return self._rev_max

    @property
    def phase(self) -> str:
        return "FLIP" if self._task == FLIP else "RECOVER"

    @property
    def task(self) -> int:
        return self._task

    @property
    def passed_inverted(self) -> bool:
        return self._inverted
