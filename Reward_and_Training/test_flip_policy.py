"""
test_flip_policy.py — §03 checks on the real simulator.  Run:  pytest -q test_flip_policy.py

The key guarantee tested here: the observation the policy is TRAINED on is
bit-for-bit the observation the DEPLOYED controller builds from raw observations.
"""
import os
import tempfile

import numpy as np
import gymnasium as gym
import pytest
import torch as th
from stable_baselines3 import PPO

from flip_env import make_flip_env
from flip_reward import FlipReward, FlipRewardConfig
from flip_task_env import FlipTaskEnv, make_flip_task_env, N_PRIV
from flip_policy import (FlipProgressTracker, ActorObsBuilder, AsymmetricActorCriticPolicy,
                         LearnedFlipController, N_FLIP_FEATURES)
from run_baselines import check_simulator


def quat_pitch(t):
    return np.array([np.cos(t / 2), 0.0, np.sin(t / 2), 0.0])


# ---------------------------------------------------------------- tracker -- #
def test_tracker_full_turn_and_wobble():
    tr = FlipProgressTracker()
    for t in np.linspace(0, 2 * np.pi, 60):            # ~0.1 rad per decision
        tr.update(quat_pitch(t))
    assert abs(tr.rev - 2 * np.pi) < 1e-6 and tr.done
    f = tr.features()
    assert f.shape == (N_FLIP_FEATURES,) and f[2] == pytest.approx(1.0) and f[3] == 1.0
    tr.reset()
    for t in np.concatenate([np.linspace(0, 1, 20), np.linspace(1, 0, 20)]):
        tr.update(quat_pitch(t))
    assert abs(tr.rev) < 1e-6 and not tr.done


def test_tracker_negative_direction():
    tr = FlipProgressTracker(direction=-1.0)
    for t in np.linspace(0, -2 * np.pi, 60):
        tr.update(quat_pitch(t))
    assert tr.rev == pytest.approx(2 * np.pi, abs=1e-6) and tr.done


# ------------------------------------------------ train/deploy consistency -- #
class _Recorder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.raws = []

    def reset(self, **kw):
        o, i = self.env.reset(**kw)
        self.raws = [np.array(o)]
        return o, i

    def step(self, a):
        o, r, te, tr, i = self.env.step(a)
        self.raws.append(np.array(o))
        return o, r, te, tr, i


def _task_env_with_recorder(**kw):
    reward = FlipReward(FlipRewardConfig())
    flip = make_flip_env(action_repeat=4, reward_fn=reward)
    rec = _Recorder(flip.env)
    return FlipTaskEnv(rec, flip.cfg, reward, **kw), rec


def test_training_obs_equals_deployed_obs():
    env, rec = _task_env_with_recorder()
    check_simulator(env.env)
    obs0, _ = env.reset(seed=3)
    rng = np.random.default_rng(0)
    actions = [np.clip(0.06 + rng.normal(0, 0.3, 4), -1, 1).astype(np.float32) for _ in range(60)]
    train_obs = [obs0]
    for a in actions:
        o, r, te, tr, _ = env.step(a)
        train_obs.append(o)
        if te or tr:
            break
    # deployment side: rebuild from the recorded RAW observations only
    b = ActorObsBuilder(env.builder.cfg)
    b.reset()
    for i, raw in enumerate(rec.raws[:len(train_obs)]):
        if i > 0:
            b.set_action(actions[i - 1])
        dep = b.build(raw)
        np.testing.assert_array_equal(dep, train_obs[i][:env.n_actor])


def test_randomised_start_resets_reward_and_tracker():
    env, _ = _task_env_with_recorder(init_steps=(5, 5), init_noise=0.2)
    o1, _ = env.reset(seed=1)
    assert env.reward.alpha == 0.0 and env.builder.tracker.rev == 0.0
    o2, _ = env.reset(seed=2)
    assert not np.allclose(o1[:env.n_actor], o2[:env.n_actor])        # different starts
    o1b, _ = env.reset(seed=1)
    np.testing.assert_allclose(o1[:env.n_actor], o1b[:env.n_actor])   # reproducible


def test_observation_bounds_and_info():
    env = make_flip_task_env()
    o, _ = env.reset(seed=0)
    assert o.shape == env.observation_space.shape == (env.n_actor + N_PRIV,)
    for _ in range(20):
        o, r, te, tr, info = env.step(np.array([1, 1, -1, -1], np.float32))   # hard pitch
        assert np.all(np.isfinite(o)) and np.all(np.abs(o) <= env.builder.cfg.clip)
        assert np.isfinite(r)
        for k in ("flip_success", "flip_rev", "flip_inverted", "crashed"):
            assert k in info
        if te or tr:
            break


# ---------------------------------------------------------- policy/model -- #
@pytest.fixture(scope="module")
def tiny_model():
    env = make_flip_task_env()
    model = PPO(AsymmetricActorCriticPolicy, env, n_steps=64, batch_size=64, device="cpu",
                policy_kwargs=dict(n_actor_obs=env.n_actor, net_arch=dict(pi=[32, 32], vf=[32, 32])))
    return model, env


def test_actor_ignores_privileged_critic_uses_it(tiny_model):
    model, env = tiny_model
    o, _ = env.reset(seed=0)
    x = th.as_tensor(np.stack([o, o]))
    x[1, env.n_actor:] += 1.0                                   # change privileged part only
    with th.no_grad():
        d = model.policy.get_distribution(x).distribution.mean
        v = model.policy.predict_values(x)
    assert th.allclose(d[0], d[1])
    assert not th.allclose(v[0], v[1])


def _run_cfg(env):
    from dataclasses import asdict
    return {"action_repeat": 4, "n_actor": env.n_actor, "n_priv": N_PRIV,
            "obs_cfg": asdict(env.builder.cfg), "flip_direction": 1.0, "eps_full": 0.15}


def test_controller_interface_and_decision_hold(tiny_model):
    model, env = tiny_model
    c = LearnedFlipController(model, _run_cfg(env), decision_every=4)
    raw = np.zeros(20, np.float32); raw[6] = 1.0
    c.reset()
    acts = [c.act(raw) for _ in range(8)]
    for a in acts:
        assert a.shape == (4,) and a.dtype == np.float32 and np.all(np.abs(a) <= 1)
    assert np.array_equal(acts[0], acts[3]) and np.array_equal(acts[4], acts[7])


def test_save_load_roundtrip(tiny_model):
    model, env = tiny_model
    rc = _run_cfg(env)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.zip")
        model.save(path)
        import json
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(rc, f)
        c_mem = LearnedFlipController(model, rc)
        c_disk = LearnedFlipController(path)                 # finds config.json next to it
        raw = np.zeros(20, np.float32); raw[6] = 1.0
        np.testing.assert_allclose(c_mem.act(raw), c_disk.act(raw), atol=1e-6)


# ------------------------------------------------ anti-hacking (run-4 finding) -- #
def _quat_from_R(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def _coning_quats(tilt_deg=60.0, n=400, turns=3):
    """Body tilted by a fixed angle while yawing about world z: coning."""
    from scipy.spatial.transform import Rotation
    qs = []
    for psi in np.linspace(0, 2 * np.pi * turns, n):
        R = Rotation.from_euler("z", psi).as_matrix() @ Rotation.from_euler("y", np.deg2rad(tilt_deg)).as_matrix()
        qs.append(_quat_from_R(R))
    return qs


def test_coning_is_not_a_flip():
    from flip_reward import SuccessDetector, SuccessConfig, FlipReward, FlipRewardConfig
    qs = _coning_quats()
    tr = FlipProgressTracker()
    for q in qs:
        tr.update(q)
    assert abs(tr.rev) < 0.5 and not tr.done
    rw = FlipReward(FlipRewardConfig())
    full = lambda q: np.concatenate([np.zeros(6), q, np.zeros(6), np.full(4, 523.0)])
    for q in qs:
        rw(np.zeros(3), full(q), np.zeros(4), env=None)
    assert abs(rw.alpha) < 0.5 and rw.phase == "FLIP" and not rw.passed_inverted
    det = SuccessDetector(SuccessConfig(hold_steps=5))
    for q in qs + [np.array([1.0, 0, 0, 0])] * 20:
        det.update(full(q))
    assert not det.succeeded


def test_detector_requires_inversion_and_accepts_real_flip():
    from flip_reward import SuccessDetector, SuccessConfig
    full = lambda q: np.concatenate([np.zeros(6), q, np.zeros(6), np.full(4, 523.0)])
    det = SuccessDetector(SuccessConfig(hold_steps=5))
    for t in np.linspace(0, 2 * np.pi, 80):
        det.update(full(quat_pitch(t)))
    for _ in range(10):
        det.update(full(quat_pitch(2 * np.pi)))
    assert det.succeeded and det.passed_inverted


# --------------------------------------------------------- CTBR interface -- #
def test_ctbr_zero_action_hovers_and_rate_tracks():
    env = make_flip_task_env(action_mode="ctbr")
    env.reset(seed=0)
    for _ in range(50):
        env.step(np.zeros(4, np.float32))
    q = env.base.quad
    assert np.linalg.norm(q.omega) < 0.1 and abs(q.pos[2]) < 0.1
    for _ in range(10):
        env.step(np.array([0, 0, 0.5, 0], np.float32))          # q command = 10 rad/s
    assert env.base.quad.omega[1] == pytest.approx(10.0, abs=1.5)


def test_ctbr_expert_flips_and_recovers():
    from ctbr_expert import CTBRExpert
    from baselines import control_dt_from_env
    env = make_flip_task_env(action_mode="ctbr", episode_seconds=4.0)
    ex = CTBRExpert(env.quad_model, control_dt_from_env(env.env))
    env.reset(seed=7)
    ex.reset()
    info = {}
    for _ in range(200):
        _, _, te, tr, info = env.step(ex.act(env._last_raw))
        assert not te
        if tr:
            break
    assert info["flip_success"] == 1.0 and info["flip_rev"] > 2 * np.pi - 0.2


def test_deployed_ctbr_controller_matches_training_env(tiny_model):
    """Same policy, same seed: the env (training path) and the controller
    (deployment path, raw env + internal CTBR mapper) produce the same motors."""
    from dataclasses import asdict
    from flip_policy import model_to_dict
    model, _ = tiny_model
    env = make_flip_task_env(action_mode="ctbr")
    rc = {"action_repeat": 4, "n_actor": env.n_actor, "n_priv": N_PRIV,
          "obs_cfg": asdict(env.builder.cfg), "flip_direction": 1.0, "eps_full": 0.15,
          "action_mode": "ctbr", "ctbr": env.mapper.config(), "quad_model": model_to_dict(env.quad_model)}
    ctrl = LearnedFlipController(model, rc)
    o, _ = env.reset(seed=5)
    ctrl.reset()
    for _ in range(20):
        a, _ = model.predict(o, deterministic=True)
        m_env = env.mapper.motor_action(np.clip(a, -1, 1), env._last_raw[13:16])
        m_ctrl = ctrl.act(env._last_raw)
        np.testing.assert_allclose(m_env, m_ctrl, atol=1e-5)
        o, *_ = env.step(a)


# ------------------------------------------------------------------ wind -- #
def test_wind_randomisation_is_reproducible_and_pushes_the_vehicle():
    env = make_flip_task_env(wind_prob=1.0, wind_max=5.0)
    speeds, drifts = [], []
    for s in (1, 2, 1):
        env.reset(seed=s)
        speeds.append(env.wind_speed)
        for _ in range(100):                                   # zero CTBR action: level, no position hold
            env.step(np.zeros(4, np.float32))
        drifts.append(np.linalg.norm(env.base.quad.pos[:2]))
    assert speeds[0] == speeds[2] and speeds[0] != speeds[1]  # per-seed, reproducible
    assert drifts[0] == pytest.approx(drifts[2]) and max(drifts) > 0.3
    env0 = make_flip_task_env()                                # default: no wind
    env0.reset(seed=1)
    assert not env0.base.random_wind


def test_hybrid_pid_handoff_after_flip():
    """PPO flies the flip; once upright and still the §02 PID takes over and holds the start."""
    from run_baselines import make_raw_env, rollout
    from flip_reward import SuccessDetector, SuccessConfig
    env = make_raw_env()
    c = LearnedFlipController("models/ppo_flip.zip", pid_handoff=True)
    det = SuccessDetector(SuccessConfig())
    tr = rollout(env, c, 10000, detector=det, max_seconds=4.0, wind=3.0)
    assert bool(tr["detector_success"]) and not bool(tr["terminated"])
    assert 2 in set(tr["phase"].tolist())                        # the PID phase was reached
    assert np.linalg.norm(tr["obs"][-1][3:5]) < 0.5              # held near the start despite wind


def test_position_integral_feature():
    env = make_flip_task_env(pos_integral=True, wind_prob=1.0, wind_max=4.0)
    assert env.n_actor == 24 + 3
    o, _ = env.reset(seed=3)
    for _ in range(60):
        o, *_ = env.step(np.zeros(4, np.float32))
    integ = o[24:27]
    pos = env.base.quad.pos
    assert np.any(np.abs(integ) > 1e-3)
    assert np.sign(integ[0]) == np.sign(pos[0]) or abs(pos[0]) < 0.05     # integral follows the drift
