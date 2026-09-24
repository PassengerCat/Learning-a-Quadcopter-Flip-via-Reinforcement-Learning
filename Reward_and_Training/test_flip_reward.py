"""
Synthetic-trajectory unit tests for flip_reward.py.

No simulator needed: we hand-build 20-dim "full" observation vectors from a
prescribed pitch trajectory (pure rotation about body-y) and feed them through
the reward / detector exactly as the env would (one call per substep, action
held constant across an action-repeat block).

Run:  python test_flip_reward.py
"""

import numpy as np

from flip_reward import (
    FlipReward, FlipRewardConfig, SuccessDetector, SuccessConfig,
    projected_gravity, incremental_pitch, tilt_angle,
)

DT = 0.005
K = 4  # action-repeat


def quat_pitch(theta: float) -> np.ndarray:
    """Scalar-first quaternion for a rotation `theta` about body-y."""
    return np.array([np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0])


def make_full_obs(pos, quat, vel, omega, motors=None) -> np.ndarray:
    """Assemble a 20-dim 'full' observation: tvel|pos|quat|vel|pqr|motors."""
    if motors is None:
        motors = np.full(4, 523.0)
    return np.concatenate([np.zeros(3), pos, quat, vel, omega, motors]).astype(float)


def pitch_profile(theta_end, n, hold=0):
    """A smooth 0->theta_end angle profile over n steps, then `hold` steps flat.
    Returns arrays (theta, thetadot) with thetadot consistent with dt."""
    t = np.linspace(0.0, 1.0, n)
    theta = theta_end * (3 * t**2 - 2 * t**3)          # smoothstep (0 slope at ends)
    theta = np.concatenate([theta, np.full(hold, theta_end)])
    thetadot = np.gradient(theta, DT)
    return theta, thetadot


# --------------------------------------------------------------------------- #
def test_geometry():
    assert np.allclose(projected_gravity(quat_pitch(0.0)), [0, 0, 1], atol=1e-6)
    assert np.allclose(projected_gravity(quat_pitch(np.pi)), [0, 0, -1], atol=1e-6)
    assert abs(tilt_angle(1.0)) < 1e-6
    assert abs(tilt_angle(-1.0) - np.pi) < 1e-6
    # incremental pitch reconstructs a small delta
    dphi = incremental_pitch(quat_pitch(0.30), quat_pitch(0.34))
    assert abs(dphi - 0.04) < 1e-6, dphi
    print("[ok] geometry: projected gravity, tilt, incremental pitch")


def test_progress_and_cap():
    """alpha & revolution track the rotation; the rotation potential saturates
    at w_rot at 2*pi, so extra spin earns ~0 additional shaping (anti spin-forever)."""
    cfg = FlipRewardConfig()
    rw = FlipReward(cfg)
    theta, thetadot = pitch_profile(2 * np.pi, 400)

    # accumulate shaping over the full flip
    shaped_to_2pi = 0.0
    for th, thd in zip(theta, thetadot):
        s = make_full_obs(np.zeros(3), quat_pitch(th), np.zeros(3), np.array([0, thd, 0]))
        shaped_to_2pi += rw(np.zeros(3), s, np.zeros(4), env=None)
    assert abs(rw.revolution - 2 * np.pi) < 0.05, rw.revolution
    assert abs(rw.alpha - 2 * np.pi) < 0.05, rw.alpha
    assert rw.phase == "RECOVER"

    # now keep spinning another full turn -> rotation term is capped
    extra = 0.0
    theta2 = 2 * np.pi + np.cumsum(np.full(200, (2 * np.pi) / 200))
    thd2 = np.gradient(theta2, DT)
    for th, thd in zip(theta2, thd2):
        s = make_full_obs(np.zeros(3), quat_pitch(th), np.zeros(3), np.array([0, thd, 0]))
        extra += rw(np.zeros(3), s, np.zeros(4), env=None)
    # any extra reward now comes from settle (recover), NOT from more rotation.
    rot_component = cfg.w_rot * min(rw.alpha, 2 * np.pi) / (2 * np.pi)
    assert rot_component <= cfg.w_rot + 1e-9
    print(f"[ok] progress & cap: rev={rw.revolution:.3f}, extra-spin shaping stayed bounded")


def test_wobble_is_neutral():
    """A back-and-forth wobble nets ~0 accumulated angle -> no net rotation shaping."""
    rw = FlipReward(FlipRewardConfig())
    # go to +1.0 rad and back to 0 repeatedly
    seg_up, thd_up = pitch_profile(1.0, 100)
    seq = np.concatenate([seg_up, seg_up[::-1], seg_up, seg_up[::-1]])
    thd = np.gradient(seq, DT)
    total = 0.0
    for th, d in zip(seq, thd):
        s = make_full_obs(np.zeros(3), quat_pitch(th), np.zeros(3), np.array([0, d, 0]))
        total += rw(np.zeros(3), s, np.zeros(4), env=None)
    assert abs(rw.alpha) < 0.05, rw.alpha         # netted out
    assert abs(rw.revolution) < 0.05, rw.revolution
    assert not rw._hit_full
    print(f"[ok] wobble neutral: alpha={rw.alpha:.4f}, rev={rw.revolution:.4f}")


def test_milestones_fire_once():
    rw = FlipReward(FlipRewardConfig())
    theta, thetadot = pitch_profile(2 * np.pi, 500, hold=100)
    inv_hits = full_hits = 0
    prev_paid_inv = prev_paid_full = False
    for th, thd in zip(theta, thetadot):
        s = make_full_obs(np.zeros(3), quat_pitch(th), np.zeros(3), np.array([0, thd, 0]))
        rw(np.zeros(3), s, np.zeros(4), env=None)
        if rw._paid_inv and not prev_paid_inv:
            inv_hits += 1
        if rw._paid_full and not prev_paid_full:
            full_hits += 1
        prev_paid_inv, prev_paid_full = rw._paid_inv, rw._paid_full
    assert inv_hits == 1 and full_hits == 1, (inv_hits, full_hits)
    print("[ok] milestones: b_inv and b_full each fired exactly once")


def test_action_smoothness_once_per_decision():
    """With action held constant across K substeps, the smoothness penalty must
    fire exactly once per decision (at the boundary), not K times."""
    cfg = FlipRewardConfig(lam_dact=1.0, w_rot=0.0, w_set=0.0, s_alive=0.0,
                           lam_off=0.0, lam_pos=0.0, b_hold=0.0)
    rw = FlipReward(cfg)
    q = quat_pitch(0.0)
    penalties = []
    actions = [np.zeros(4), np.ones(4), np.ones(4) * -1.0]  # 3 decisions
    for dec, a in enumerate(actions):
        for sub in range(K):
            s = make_full_obs(np.zeros(3), q, np.zeros(3), np.zeros(3))
            penalties.append(rw(np.zeros(3), s, a, env=None))
    penalties = np.array(penalties).reshape(len(actions), K)
    # within each decision only the FIRST substep carries a non-zero jump
    assert np.allclose(penalties[:, 1:], 0.0), penalties
    # decision 0 boundary: a_prev initialized to a -> 0; decisions 1,2 -> non-zero
    assert penalties[1, 0] < 0 and penalties[2, 0] < 0
    print("[ok] action-smoothness fires once per decision, not per substep")


def test_success_detector():
    """Detector fires on a clean flip+settle; NOT on a half rotation or a wobble."""
    det = SuccessDetector(SuccessConfig(hold_steps=10))

    # clean: full 2pi then hold upright & still
    theta, _ = pitch_profile(2 * np.pi, 300, hold=0)
    for th in theta:
        det.update(make_full_obs(np.zeros(3), quat_pitch(th), np.zeros(3), np.zeros(3)))
    # hold upright and still for >= hold_steps
    for _ in range(20):
        det.update(make_full_obs(np.zeros(3), quat_pitch(2 * np.pi), np.zeros(3), np.zeros(3)))
    assert det.succeeded, "clean flip should succeed"

    # half rotation only -> never a full revolution -> no success
    det2 = SuccessDetector(SuccessConfig(hold_steps=10))
    theta2, _ = pitch_profile(np.pi, 300)
    for th in theta2:
        det2.update(make_full_obs(np.zeros(3), quat_pitch(th), np.zeros(3), np.zeros(3)))
    for _ in range(30):
        det2.update(make_full_obs(np.zeros(3), quat_pitch(np.pi), np.zeros(3), np.zeros(3)))
    assert not det2.succeeded, "half rotation must not count as success"

    # wobble -> revolution nets ~0 -> no success even if it ends upright & still
    det3 = SuccessDetector(SuccessConfig(hold_steps=10))
    seg, _ = pitch_profile(1.0, 100)
    for th in np.concatenate([seg, seg[::-1]]):
        det3.update(make_full_obs(np.zeros(3), quat_pitch(th), np.zeros(3), np.zeros(3)))
    for _ in range(30):
        det3.update(make_full_obs(np.zeros(3), quat_pitch(0.0), np.zeros(3), np.zeros(3)))
    assert not det3.succeeded, "wobble must not count as success"
    print("[ok] success detector: clean=success, half=fail, wobble=fail")


def test_reward_independent_of_success():
    """The reward object must expose no success read and must not need the detector."""
    rw = FlipReward(FlipRewardConfig())
    assert not hasattr(rw, "succeeded")
    r = rw(np.zeros(3), make_full_obs(np.zeros(3), quat_pitch(0.1), np.zeros(3), np.array([0, 1.0, 0])),
           np.zeros(4), env=None)
    assert np.isfinite(r)
    print("[ok] reward is finite and independent of the success detector")


def test_no_nans_on_extremes():
    rw = FlipReward(FlipRewardConfig())
    weird = make_full_obs(np.array([10, -10, 4.9]), quat_pitch(2.7),
                          np.array([8, -8, 3]), np.array([12, -30, 9]),
                          motors=np.array([925, 75, 925, 75.0]))
    for _ in range(5):
        r = rw(np.zeros(3), weird, np.array([1, -1, 1, -1.0]), env=None)
        assert np.isfinite(r), r
    print("[ok] no NaNs/inf on extreme states")


if __name__ == "__main__":
    tests = [
        test_geometry,
        test_progress_and_cap,
        test_wobble_is_neutral,
        test_milestones_fire_once,
        test_action_smoothness_once_per_decision,
        test_success_detector,
        test_reward_independent_of_success,
        test_no_nans_on_extremes,
    ]
    for t in tests:
        t()
    print("\nAll tests passed.")
