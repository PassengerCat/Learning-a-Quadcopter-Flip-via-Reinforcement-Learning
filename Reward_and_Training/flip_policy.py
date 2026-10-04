"""
flip_policy.py — deployable side of §03 (everything the learned controller needs at run time).

  * FlipProgressTracker  — geometric (quaternion) revolution count, updated once per
                           decision from the RAW observation. Drift-free, and exactly
                           reproducible at deployment (no access to reward internals).
  * ActorObsBuilder      — raw obs -> policy observation:
                           flip_env.transform_observation(raw, last_action) ++ tracker features.
                           The SAME object builds the observation in training
                           (flip_task_env.FlipTaskEnv) and at deployment (LearnedFlipController),
                           so train/deploy observations are identical by construction.
  * AsymmetricActorCriticPolicy — SB3 policy whose actor sees only the deployable
                           observation while the critic also sees privileged state
                           (noise-free state, reward phase, time left).
  * LearnedFlipController — the trained policy behind the §02 `Controller` interface
                           (reset / act(raw_obs) / diagnostics), so §05 evaluates it with
                           exactly the same harness as the baselines.

Observation layout fed to the network:  [ actor (n_actor) | privileged (n_priv) ]
At deployment the privileged block is zero-filled; the actor never reads it.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Union

import numpy as np

from flip_env import ObsConfig, transform_observation
from flip_reward import gravity_pitch_increment

N_FLIP_FEATURES = 4        # [sin rev, cos rev, progress, phase]


# --------------------------------------------------------------------------- #
# Progress tracker (deployable)                                               #
# --------------------------------------------------------------------------- #
class FlipProgressTracker:
    """Geometric flip angle (flip_reward.gravity_pitch_increment — the same measure
    the reward and the §04 detector use), accumulated once per decision from the raw
    quaternion. Per-decision rotation must stay below pi (157 rad/s at 50 Hz)."""

    def __init__(self, direction: float = +1.0, eps_full: float = 0.15):
        self.direction = float(direction)
        self.eps_full = float(eps_full)
        self.reset()

    def reset(self) -> None:
        self.rev = 0.0
        self.q_prev: Optional[np.ndarray] = None
        self.done = False                       # latched: a full revolution happened

    def update(self, quat: np.ndarray) -> None:
        q = np.asarray(quat, dtype=np.float64)
        if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-8:
            return
        if self.q_prev is not None:
            self.rev += self.direction * gravity_pitch_increment(self.q_prev, q)
        self.q_prev = q.copy()
        if self.rev >= 2.0 * np.pi - self.eps_full:
            self.done = True

    def features(self) -> np.ndarray:
        a = self.rev
        progress = float(np.clip(a, 0.0, 2.0 * np.pi) / (2.0 * np.pi))
        return np.array([np.sin(a), np.cos(a), progress, 1.0 if self.done else 0.0], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Actor observation (deployable)                                              #
# --------------------------------------------------------------------------- #
class ActorObsBuilder:
    """raw 20-dim obs -> actor observation. Call ``reset()`` at episode start,
    ``build(raw)`` on every decision's observation, and ``set_action(a)`` with the
    action that is about to be applied (so the NEXT observation carries it)."""

    def __init__(self, obs_cfg: ObsConfig, direction: float = +1.0, eps_full: float = 0.15,
                 pos_integral: bool = False, dt: float = 0.02,
                 i_scale: float = 5.0, i_limit: float = 20.0):
        """pos_integral=True appends the time-integral of the position error w.r.t. the
        episode's start position (3 values, m·s / i_scale). This gives a memoryless
        policy the INTEGRAL state that a PID uses to reject steady wind (depth
        experiment 5 of the run_all.bat study). Anti-windup: clipped to ±i_limit m·s."""
        self.cfg = obs_cfg
        self.tracker = FlipProgressTracker(direction, eps_full)
        self.pos_integral = bool(pos_integral)
        self.dt, self.i_scale, self.i_limit = float(dt), float(i_scale), float(i_limit)
        self.reset()

    @property
    def dim(self) -> int:
        return self.cfg.obs_dim() + N_FLIP_FEATURES + (3 if self.pos_integral else 0)

    def reset(self, last_action: Optional[np.ndarray] = None) -> None:
        self.tracker.reset()
        self._pos0 = None
        self._int = np.zeros(3)
        self.last_action = (np.zeros(4, np.float32) if last_action is None
                            else np.asarray(last_action, np.float32).copy())

    def set_action(self, action: np.ndarray) -> None:
        self.last_action = np.clip(np.asarray(action, np.float32), -1.0, 1.0).copy()

    def build(self, raw_obs: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_obs, dtype=np.float32)
        self.tracker.update(raw[6:10])
        base = transform_observation(raw, self.last_action, self.cfg)
        parts = [base, self.tracker.features()]
        if self.pos_integral:
            pos = np.asarray(raw[3:6], dtype=np.float64)
            if self._pos0 is None:
                self._pos0 = pos.copy()
            self._int = np.clip(self._int + (pos - self._pos0) * self.dt, -self.i_limit, self.i_limit)
            parts.append((self._int / self.i_scale).astype(np.float32))
        return np.concatenate(parts).astype(np.float32)


# --------------------------------------------------------------------------- #
# Action interface: collective thrust + body rates (CTBR) -> motor commands   #
# --------------------------------------------------------------------------- #
from baselines import QuadModel, Mixer          # noqa: E402  (§02: same mixer as the baselines)


def model_to_dict(m: QuadModel) -> Dict:
    return {"mass": m.mass, "g": m.g, "kTh": m.kTh, "kTo": m.kTo, "dxm": m.dxm, "dym": m.dym,
            "inertia_diag": [float(x) for x in np.diag(m.inertia)],
            "w_min": m.w_min, "w_max": m.w_max, "w_hover": m.w_hover}


def model_from_dict(d: Dict) -> QuadModel:
    d = dict(d)
    I = np.diag(d.pop("inertia_diag"))
    return QuadModel(inertia=I, **d)


class CTBRMapper:
    """Policy action a in [-1,1]^4 = [collective, p_cmd, q_cmd, r_cmd]  ->  motor action.

        T     = T_hover + a0 * (T_max - T_hover)   (a0 >= 0)
              = T_hover + a0 * (T_hover - T_min)   (a0 <  0)      -> a0 = 0 is hover
        ω_cmd = a[1:4] * (rate_max_rp, rate_max_rp, rate_max_yaw)
        τ     = I · kp ⊙ (ω_cmd − ω) + ω × Iω                     (fixed P rate loop)
        motors = Mixer(T, τ)                                        (§02 mixer, attitude-first)

    A fixed, stateless inner loop: the learned policy still decides the WHOLE
    manoeuvre (when/how hard to rotate, thrust, recovery); the loop only turns rate
    commands into motor speeds. Deterministic given (a, ω), so training and
    deployment apply exactly the same mapping."""

    def __init__(self, model: QuadModel, rate_max_rp: float = 20.0, rate_max_yaw: float = 4.0,
                 kp_rate=(18.0, 18.0, 8.0)):
        self.m = model
        self.mixer = Mixer(model)
        self.rate_max = np.array([rate_max_rp, rate_max_rp, rate_max_yaw], dtype=np.float64)
        self.kp = np.asarray(kp_rate, dtype=np.float64)
        self.T_hover = model.mass * model.g
        self.T_min, self.T_max = model.min_thrust, model.max_thrust

    def config(self) -> Dict:
        return {"rate_max_rp": float(self.rate_max[0]), "rate_max_yaw": float(self.rate_max[2]),
                "kp_rate": [float(x) for x in self.kp]}

    def motor_action(self, action: np.ndarray, omega: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        c = a[0]
        T = self.T_hover + c * ((self.T_max - self.T_hover) if c >= 0 else (self.T_hover - self.T_min))
        w = np.nan_to_num(np.asarray(omega, dtype=np.float64))
        rate_cmd = a[1:4] * self.rate_max
        I = self.m.inertia
        tau = I @ (self.kp * (rate_cmd - w)) + np.cross(w, I @ w)
        return self.mixer.action(T, tau)


# --------------------------------------------------------------------------- #
# Asymmetric actor-critic for Stable-Baselines3                               #
# --------------------------------------------------------------------------- #
try:
    import torch as th
    from torch import nn
    from stable_baselines3.common.policies import ActorCriticPolicy

    def _mlp(n_in: int, sizes: List[int], act) -> nn.Sequential:
        layers, last = [], n_in
        for h in sizes:
            layers += [nn.Linear(last, h), act()]
            last = h
        return nn.Sequential(*layers)

    class AsymmetricMlpExtractor(nn.Module):
        """Actor MLP reads features[:, :n_actor]; critic MLP reads everything."""

        def __init__(self, feature_dim: int, n_actor: int, pi: List[int], vf: List[int], act):
            super().__init__()
            assert 0 < n_actor <= feature_dim
            self.n_actor = n_actor
            self.policy_net = _mlp(n_actor, pi, act)
            self.value_net = _mlp(feature_dim, vf, act)
            self.latent_dim_pi = pi[-1] if pi else n_actor
            self.latent_dim_vf = vf[-1] if vf else feature_dim

        def forward(self, features):
            return self.forward_actor(features), self.forward_critic(features)

        def forward_actor(self, features):
            return self.policy_net(features[..., : self.n_actor])

        def forward_critic(self, features):
            return self.value_net(features)

    class AsymmetricActorCriticPolicy(ActorCriticPolicy):
        """PPO policy with a privileged critic. Pass ``n_actor_obs`` in policy_kwargs;
        ``net_arch=dict(pi=[..], vf=[..])``."""

        def __init__(self, *args, n_actor_obs: int, **kwargs):
            self.n_actor_obs = int(n_actor_obs)
            super().__init__(*args, **kwargs)

        def _build_mlp_extractor(self) -> None:
            arch = self.net_arch
            if isinstance(arch, dict):
                pi, vf = list(arch.get("pi", [])), list(arch.get("vf", []))
            else:
                pi = vf = list(arch or [])
            self.mlp_extractor = AsymmetricMlpExtractor(self.features_dim, self.n_actor_obs,
                                                        pi, vf, self.activation_fn)

        def _get_constructor_parameters(self) -> Dict:
            data = super()._get_constructor_parameters()
            data["n_actor_obs"] = self.n_actor_obs
            return data

except ImportError:  # deployment without torch/SB3 is not supported for the PPO policy
    AsymmetricActorCriticPolicy = None  # type: ignore


# --------------------------------------------------------------------------- #
# Deployable controller (§02 interface)                                       #
# --------------------------------------------------------------------------- #
from baselines import Controller   # noqa: E402  (§02 shared interface)


def load_run_config(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class LearnedFlipController(Controller):
    """Trained PPO policy behind the shared interface.

    ``act(raw_obs)`` expects the RAW env observation and returns MOTOR actions
    (for a CTBR policy the fixed rate loop is applied inside). ``decision_every`` = how many
    ``act`` calls share one decision: 1 when the env already applies action-repeat
    (make_flip_env / §05 harness), k when ``act`` is called at the sim rate (dt).

    ``pid_handoff=True`` = the hybrid of the methodology: the learned policy flies the
    flip and brings the vehicle back upright; once the flip is complete AND the vehicle
    is upright (< handoff_tilt_deg) and still (|ω| < handoff_rate), the §02 cascaded PID
    (the "recovery block") takes over and holds the START position. The PID's integral
    action is what a memoryless policy lacks against steady wind.
    """

    name = "ppo_flip"

    def __init__(self, model: Union[str, "object"], run_config: Optional[Union[str, Dict]] = None,
                 decision_every: int = 1, device: str = "cpu", pid_handoff: bool = False,
                 handoff_tilt_deg: float = 15.0, handoff_rate: float = 1.0):
        if isinstance(model, str):
            from stable_baselines3 import PPO
            path = model
            model = PPO.load(path, device=device,
                             custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0,
                                             "clip_range": lambda _: 0.2})
            if run_config is None:   # next to the model, or one level up (checkpoints/)
                d = os.path.dirname(os.path.abspath(path))
                for cand in (os.path.join(d, "config.json"), os.path.join(os.path.dirname(d), "config.json")):
                    if os.path.exists(cand):
                        run_config = cand
                        break
        if isinstance(run_config, str):
            run_config = load_run_config(run_config)
        if run_config is None:
            raise ValueError("run config (config.json written by train_ppo.py) is required")
        self.model = model
        self.rc = run_config
        cfg = ObsConfig(**run_config["obs_cfg"])
        self.builder = ActorObsBuilder(
            cfg, run_config["flip_direction"], run_config["eps_full"],
            pos_integral=bool(run_config.get("pos_integral", False)),
            dt=float(run_config.get("sim_dt", 0.005)) * int(run_config["action_repeat"]))
        self.n_actor = int(run_config["n_actor"])
        self.n_priv = int(run_config["n_priv"])
        assert self.builder.dim == self.n_actor, (self.builder.dim, self.n_actor)
        self.action_mode = run_config.get("action_mode", "motors")
        self.mapper = None
        if self.action_mode == "ctbr":
            self.mapper = CTBRMapper(model_from_dict(run_config["quad_model"]), **run_config["ctbr"])
        self.decision_every = int(decision_every)
        self.pid = None
        if pid_handoff:
            from baselines import CascadedPID
            dt = float(run_config.get("sim_dt", 0.005)) * int(run_config["action_repeat"])
            self.pid = CascadedPID(model_from_dict(run_config["quad_model"]), dt)
            self.name = "ppo_flip+pid"
        self.handoff_tilt = np.deg2rad(handoff_tilt_deg)
        self.handoff_rate = float(handoff_rate)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        self.builder.reset()
        self._calls = 0
        self._held = np.zeros(4, np.float32)
        self._pid_active = False
        self._start_pos = None

    def policy_obs(self, raw_obs) -> np.ndarray:
        a = self.builder.build(raw_obs)
        return np.concatenate([a, np.zeros(self.n_priv, np.float32)])

    def act(self, raw_obs: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_obs, dtype=np.float64)
        if self._start_pos is None:
            self._start_pos = raw[3:6].copy()
        if self._calls % self.decision_every == 0 and self.pid is not None:
            if not self._pid_active:
                R22 = 1.0 - 2.0 * (raw[7] ** 2 + raw[8] ** 2)          # cos(tilt) from the quaternion
                if (self.builder.tracker.done and np.arccos(np.clip(R22, -1, 1)) < self.handoff_tilt
                        and np.linalg.norm(raw[13:16]) < self.handoff_rate):
                    self._pid_active = True
                    self.pid.reset()
                    self.pid.set_reference(self._start_pos, yaw_ref=0.0)
            if self._pid_active:
                self.builder.tracker.update(raw[6:10])                   # keep tracking for logs
                self._held = self.pid.step(raw)
                self._calls += 1
                return self._held.copy()
        if self._calls % self.decision_every == 0:
            obs = self.policy_obs(raw_obs)
            action, _ = self.model.predict(obs, deterministic=True)
            a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
            self.builder.set_action(a)          # the obs carries the POLICY's own last action
            self._held = (self.mapper.motor_action(a, np.asarray(raw_obs)[13:16])
                          if self.mapper is not None else a)
        self._calls += 1
        return self._held.copy()

    @property
    def diagnostics(self) -> Dict:
        t = self.builder.tracker
        if getattr(self, "_pid_active", False):
            return {"phase": "pid_hold", "phase_id": 2, "flip_angle_deg": float(np.degrees(t.rev))}
        return {"phase": "recover" if t.done else "flip", "phase_id": int(t.done),
                "flip_angle_deg": float(np.degrees(t.rev))}
