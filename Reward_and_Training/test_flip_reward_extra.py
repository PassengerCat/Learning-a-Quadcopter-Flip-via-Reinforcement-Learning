"""
Extra checks for flip_reward.py:
  * reset cleanliness between episodes
  * negative flip direction
  * bounded (not exploding) potential hand-off at the FLIP->RECOVER transition
  * hold bonus accrues during recover
  * end-to-end wiring smoke test against a MOCK env that mimics the documented API

Run:  python test_flip_reward_extra.py
"""

import numpy as np
from flip_reward import (
    FlipReward, FlipRewardConfig, SuccessDetector, SuccessConfig, projected_gravity,
)

DT = 0.005
K = 4


def quat_pitch(theta):
    return np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])


def full_obs(pos, quat, vel, omega, motors=None):
    if motors is None:
        motors = np.full(4, 523.0)
    return np.concatenate([np.zeros(3), pos, quat, vel, omega, motors]).astype(float)


def smoothstep(theta_end, n, hold=0):
    t = np.linspace(0, 1, n)
    th = theta_end * (3 * t**2 - 2 * t**3)
    th = np.concatenate([th, np.full(hold, theta_end)])
    return th, np.gradient(th, DT)


def run_flip(rw, theta, thetadot, vel_end=None):
    total = 0.0
    n = len(theta)
    for i, (th, thd) in enumerate(zip(theta, thetadot)):
        # optional: let velocity decay to zero over the tail so settle can rise
        vel = np.zeros(3)
        total += rw(np.zeros(3), full_obs(np.zeros(3), quat_pitch(th), vel,
                                          np.array([0, thd, 0])), np.zeros(4), env=None)
    return total


def test_reset_between_episodes():
    rw = FlipReward(FlipRewardConfig())
    th, thd = smoothstep(2 * np.pi, 300, hold=50)
    run_flip(rw, th, thd)
    assert rw._paid_full and rw.phase == "RECOVER"
    rw.reset()
    assert rw.alpha == 0.0 and rw.revolution == 0.0
    assert not rw._paid_inv and not rw._paid_full and rw.phase == "FLIP"
    # second episode must re-fire milestones (proves no carryover)
    run_flip(rw, th, thd)
    assert rw._paid_full
    print("[ok] reset: state fully cleared, milestones re-fire next episode")


def test_negative_direction():
    rw = FlipReward(FlipRewardConfig(flip_direction=-1.0))
    th, thd = smoothstep(-2 * np.pi, 300)     # rotate the other way
    run_flip(rw, th, thd)
    assert abs(rw.alpha - 2 * np.pi) < 0.1, rw.alpha        # direction sign makes it positive
    assert abs(rw.revolution - 2 * np.pi) < 0.1, rw.revolution
    assert rw.phase == "RECOVER"
    print("[ok] negative flip_direction accumulates correctly")


def test_transition_handoff_bounded():
    """At FLIP->RECOVER the settle term switches on, so the POTENTIAL steps up by
    <= w_set*settle -> a small, bounded, intentional shaping hand-off (honest note:
    this is a step, not a smooth continuation; it still preserves policy-invariance
    because Phi is a deterministic function of the augmented state). We isolate the
    shaping by zeroing all task/reg terms and verify the jump is bounded by ~w_set."""
    cfg = FlipRewardConfig(b_inv=0.0, b_full=0.0, b_hold=0.0, b_crash=0.0,
                           lam_dact=0.0, lam_off=0.0, lam_pos=0.0, s_alive=0.0)
    rw = FlipReward(cfg)
    th, thd = smoothstep(2 * np.pi, 400)
    shaping = []
    for t, d in zip(th, thd):
        shaping.append(rw(np.zeros(3), full_obs(np.zeros(3), quat_pitch(t),
                       np.zeros(3), np.array([0, d, 0])), np.zeros(4), env=None))
    jump = max(shaping)
    # bound: gamma_shape*w_set*settle(<=1) + the tiny (gamma_shape-1)*w_rot rotation term
    assert jump <= cfg.gamma_shape * cfg.w_set + 0.05, jump
    print(f"[ok] pure-shaping hand-off bounded: max step = {jump:.3f} (<= ~w_set); "
          f"NOTE full reward also adds b_full at that step by design")


def test_hold_bonus_accrues():
    cfg = FlipRewardConfig()
    rw = FlipReward(cfg)
    th, thd = smoothstep(2 * np.pi, 300)
    run_flip(rw, th, thd)                       # now in RECOVER
    # hold perfectly upright & still for 100 substeps
    acc = 0.0
    for _ in range(100):
        acc += rw(np.zeros(3), full_obs(np.zeros(3), quat_pitch(2 * np.pi),
                  np.zeros(3), np.zeros(3)), np.zeros(4), env=None)
    # b_hold * dt * 100 should be present (plus a little settle shaping)
    assert acc > cfg.b_hold * DT * 100 * 0.5, acc
    print(f"[ok] hold bonus accrues in RECOVER: 100-step recover reward = {acc:.3f}")


# ----------------------------- wiring smoke test ---------------------------- #
class _MockQuad:
    def __init__(self):
        self.pos = np.zeros(3); self.quat = quat_pitch(0.0)
        self.vel = np.zeros(3); self.omega = np.zeros(3)


class _MockEnv:
    """Minimal stand-in for QuadcopterVelocityEnv: exposes dt, quad, _terminated,
    a reward_fn hook, and an action-repeat step loop — mirrors the documented API."""
    def __init__(self, reward, detector, k=K):
        self.dt = DT; self.k = k
        self.quad = _MockQuad(); self._terminated = False
        self.reward_fn = reward; self.detector = detector

    def reset(self):
        self._terminated = False
        self.reward_fn.reset(); self.detector.reset()

    def step(self, action, theta, thetadot):
        """Advance one DECISION = k substeps; reward summed over substeps (as §01)."""
        r = 0.0
        for _ in range(self.k):
            self.quad.quat = quat_pitch(theta)
            self.quad.omega = np.array([0.0, thetadot, 0.0])
            r += self.reward_fn(np.zeros(3), None, action, env=self)  # full_state=None -> reads env.quad
        ok = self.detector.update(None, env=self)
        return r, ok


def test_wiring_mock_env_episode():
    env = _MockEnv(FlipReward(FlipRewardConfig()), SuccessDetector(SuccessConfig(hold_steps=10)))
    env.reset()
    th, thd = smoothstep(2 * np.pi, 120, hold=40)     # per-decision angle schedule
    total, succeeded = 0.0, False
    for t, d in zip(th, thd):
        r, ok = env.step(np.zeros(4), t, d)
        assert np.isfinite(r)
        total += r
        succeeded = succeeded or ok
    # then hold to let the detector's hold-counter fill
    for _ in range(15):
        r, ok = env.step(np.zeros(4), 2 * np.pi, 0.0)
        succeeded = succeeded or ok
    assert env.reward_fn.phase == "RECOVER"
    assert succeeded, "mock episode should reach success"
    print(f"[ok] wiring smoke test: full episode ran via env.reward_fn, success={succeeded}, "
          f"return={total:.2f}")


if __name__ == "__main__":
    for t in (test_reset_between_episodes, test_negative_direction,
              test_transition_handoff_bounded, test_hold_bonus_accrues,
              test_wiring_mock_env_episode):
        t()
    print("\nAll extra checks passed.")
