"""
test_baselines.py — checks for §02. Run with:  pytest -q test_baselines.py

The sign tests (test_mixer_signs_match_simulator) are the important ones on a
new machine: they confirm that the motor numbering / frame conventions assumed
by the mixer match the simulator actually installed.
"""
import numpy as np
import pytest

from baselines import (Mixer, QuadModel, FlipScript, RandomController, control_dt_from_env,
                       make_controller, quat_to_rotmat, tilt_angle, BASELINES)
from run_baselines import make_raw_env, rollout, quick_metrics, check_simulator


@pytest.fixture(scope="module")
def env():
    e = make_raw_env(action_repeat=4)
    check_simulator(e)          # fails loudly on an incompatible numpy/scipy install
    yield e
    e.close()


# ---------------------------------------------------------------- mixer ---- #
def test_mixer_roundtrip():
    m = QuadModel()
    mix = Mixer(m)
    T, tau = m.mass * m.g, np.array([0.05, -0.08, 0.01])
    w = np.sqrt(mix.w2_from_wrench(T, tau))
    np.testing.assert_allclose(mix.wrench(w), [T, *tau], rtol=1e-9, atol=1e-9)


def test_mixer_saturation_keeps_direction_and_bounds():
    m = QuadModel()
    mix = Mixer(m)
    a = mix.action(0.0, np.array([0.0, 50.0, 0.0]))       # absurd pitch request
    assert a.shape == (4,) and a.dtype == np.float32
    assert np.all(a >= -1) and np.all(a <= 1)
    wr = mix.wrench(m.w_min + (a + 1) / 2 * (m.w_max - m.w_min))
    assert wr[2] > 0.95 * m.max_torque(1)                  # full authority, right sign
    assert abs(wr[1]) < 1e-6                               # no roll leak


def test_hover_action_gives_hover_thrust():
    m = QuadModel()
    a = Mixer(m).action(m.mass * m.g, np.zeros(3))
    np.testing.assert_allclose(a, m.hover_action, atol=1e-4)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_mixer_signs_match_simulator(env, axis):
    """Command a torque on one axis for ~0.06 s from hover; the body rate on that
    axis must move with the commanded sign and dominate the other two."""
    m = QuadModel.from_env(env)
    mix = Mixer(m)
    tau = np.zeros(3)
    tau[axis] = 0.3 if axis < 2 else 0.05
    env.reset(seed=0)
    for _ in range(3):
        obs, *_ = env.step(mix.action(m.mass * m.g, tau))
    rate = obs[13:16]
    assert rate[axis] > 0.05, f"axis {axis}: rate {rate}"
    assert abs(rate[axis]) > 3 * np.max(np.abs(np.delete(rate, axis)))


# ------------------------------------------------------------ interface ---- #
def test_model_and_dt_read_from_env(env):
    m = QuadModel.from_env(env)
    base = env.unwrapped
    assert m.w_min == pytest.approx(base.min_w) and m.w_max == pytest.approx(base.max_w)
    assert m.w_hover == pytest.approx(base.hover_w)
    dt = control_dt_from_env(env)
    assert 0 < dt < 0.1


@pytest.mark.parametrize("name", BASELINES)
def test_interface_contract(env, name):
    c = make_controller(name, env, seed=3)
    obs, _ = env.reset(seed=1)
    c.reset(seed=1)
    for _ in range(5):
        a = c.act(obs)
        assert isinstance(a, np.ndarray) and a.shape == (4,) and a.dtype == np.float32
        assert np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0)
        assert isinstance(c.diagnostics, dict)
        obs, *_ = env.step(a)


def test_random_is_reproducible_and_varies_per_episode():
    obs = np.zeros(20, dtype=np.float32)
    c = RandomController(seed=7)
    c.reset(seed=1); a1 = [c.act(obs) for _ in range(10)]
    c.reset(seed=1); a2 = [c.act(obs) for _ in range(10)]
    c.reset(seed=2); a3 = [c.act(obs) for _ in range(10)]
    np.testing.assert_array_equal(a1, a2)
    assert not np.allclose(a1, a3)


# ------------------------------------------------------------- closed loop - #
def test_pid_hover_holds_position(env):
    c = make_controller("pid_hover", env)
    tr = rollout(env, c, seed=0, max_seconds=5.0)
    obs = tr["obs"]
    assert not tr["terminated"]
    assert np.max(np.linalg.norm(obs[:, 3:6], axis=1)) < 0.05
    assert np.degrees(max(tilt_angle(quat_to_rotmat(q)) for q in obs[:, 6:10])) < 3.0


def test_pid_recovers_from_large_tilt(env):
    """Knock the vehicle to a large tilt with an open-loop pulse, then let the
    PID (the recovery block) catch it."""
    m = QuadModel.from_env(env)
    mix = Mixer(m)
    c = make_controller("pid_hover", env)
    obs, _ = env.reset(seed=0)
    for _ in range(50):                                 # full pitch torque until tilt > 60 deg
        obs, *_ = env.step(mix.action(m.mass * m.g, np.array([0, m.max_torque(1), 0])))
        if np.degrees(tilt_angle(quat_to_rotmat(obs[6:10]))) > 60:
            break
    assert np.degrees(tilt_angle(quat_to_rotmat(obs[6:10]))) > 60
    c.reset()
    for _ in range(int(2.5 / control_dt_from_env(env))):
        obs, r, te, tr, _ = env.step(c.act(obs))
        assert not te
    assert np.degrees(tilt_angle(quat_to_rotmat(obs[6:10]))) < 5
    assert np.linalg.norm(obs[13:16]) < 0.5


@pytest.mark.parametrize("script", [FlipScript(), FlipScript(direction=-1), FlipScript(axis="roll")],
                         ids=["backflip", "frontflip", "roll"])
def test_scripted_flip_completes(env, script):
    c = make_controller("scripted_flip", env, script=script)
    tr = rollout(env, c, seed=0, max_seconds=5.0)
    m = quick_metrics(tr, axis=script.axis)
    assert m["success_quick"], m
    assert abs(m["rotation_deg"]) == pytest.approx(360, abs=30)
    assert m["altitude_loss_m"] < 1.0


def test_pid_and_random_never_count_as_flips(env):
    for name in ("pid_hover", "random"):
        c = make_controller(name, env, seed=0)
        m = quick_metrics(rollout(env, c, seed=0, max_seconds=5.0))
        assert not m["success_quick"]