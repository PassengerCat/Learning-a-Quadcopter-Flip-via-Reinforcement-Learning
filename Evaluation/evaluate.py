"""Universal evaluation harness (§05) — episode loop and per-episode metrics.

Runs ANY controller that has the course interface

    reset(seed=None)
    act(observation) -> (action, info)

on the official benchmark environment and judges every episode with the
independent §04 success detector. Rewards are never used. Every episode is also judged,
on the same trajectory, by the PPO track's detector (Success_Detector/ppo_track_detector.py),
updated once per 20 ms decision as in her pipeline; its verdict is stored in
EpisodeResult.ppo_track and never changes the §04 verdict in EpisodeResult.success.

Benchmark conditions (project brief, "Initial Benchmark Conditions"):
full-state observation, no wind, no observation noise, nominal hover start,
tilt termination disabled; ground/region, numerical failure and the 10 s
horizon still end an episode.

Stress conditions (robustness, brief: "only after a successful flip policy has been
obtained"): a Condition switches single disturbances on, one field per disturbance,
so that each can be measured alone. The default Condition() is the nominal benchmark.
  obs_noise_scale    Gaussian observation noise, as a multiple of the simulator's own
                     default std per group (1 = simulator defaults). It changes only what
                     the controller sees; the dynamics and the §04 verdict use the true state.
  wind_speed         horizontal wind [m/s]; the direction of the wind velocity is drawn uniformly per episode,
                     built with the simulator's own Wind model ("FIXED"); it acts through the
                     simulator's per-axis quadratic drag, a_i = Cd u_i |u_i| / m with u = wind minus
                     vehicle velocity and Cd = 0.1 (0.06-0.08 m/s^2 at 1 m/s, depending on direction).
  wind_gusts         use the simulator's "SINE" model instead: the same mean speed and heading
                     plus its fixed sinusoidal gusts (speed +-1.5/1.1/0.8 m/s, heading +-18 deg,
                     elevation +-4.8 deg).
  init_tilt_deg,     perturbed hover start ("approximately stable hover", brief p.1): tilt about a
  init_rate,         random horizontal axis, uniform in [0, init_tilt_deg] (< the detector's 15 deg
  init_vel           upright cone); body rates and linear velocity uniform in +-init_rate [rad/s]
                     and +-init_vel [m/s] per axis. Position, yaw and motor speeds stay nominal.
  action_noise,      controller-side noise as in the PPO track's stress tests (run_baselines.rollout):
  gyro_noise         Gaussian noise added to the 4 motor commands (then clipped to [-1, 1]) and to
                     the body rates the controller observes (obs[13:16], rad/s). One sample per
                     20 ms decision (NOISE_BLOCK_STEPS = 4 simulator steps), held for the block, drawn
                     from default_rng(SeedSequence([12345, seed])) in the same order (gyro 3, then
                     action 4). The env, the dynamics and the detector are untouched; the applied
                     (noisy, clipped) command is what the env receives and what the effort counts.
Noise is drawn from the environment's generator, which env.reset(seed) re-seeds; the wind
heading and the initial perturbation from separate generators seeded with (WIND_STREAM, seed)
and (INIT_STREAM, seed). An episode is therefore
reproducible from its seed and independent of the order of episodes. The simulator's own
random_wind option is not used: its "RANDOMSINE" draws from Python's global `random`
(not reproducible per episode) and, with the default heading range, always blows along +x.

The controller is called at EVERY simulator step (dt = 5 ms). A controller
that decides at a lower rate (e.g. 50 Hz) holds its own action between calls.

Per-episode metrics (brief, "Evaluation metrics"); they never change the verdict:
  final_tilt_deg     angle between body z-axis and vertical at the last step
  final_omega        |body rates| at the last step [rad/s]
  max_alt_loss_m     deepest point below the start altitude [m] (0 if never below)
  t_flip_s           first time the §04 rotation reaches 360 deg - 0.15 rad [s] (None if never)
  t_recovered_s      start of the final 0.4 s upright-and-slow hold [s] (successes only)
  control_effort     integral of sum_i (a_i - a_hover)^2 dt over the clipped actions [s]

Optional trajectory (record=True), one row per simulator state including t = 0:
  t, pos, quat, vel, omega (true simulator state; equal to the noiseless observation),
  action (the clipped command applied during the step that ENDS at that row; row 0 = NaN),
  rev (the detector's signed pitch revolution) [rad]

PPO-track verdict (EpisodeResult.ppo_track): success (latched), passed_inverted, revolution_rad,
t_flip_s (first update with the revolution done), t_recovered_s (start of the 0.4 s hold that
latched success), success_then_terminated (latched, then the episode still ended in failure).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SIM_DIR = Path(os.environ.get("QUAD_SIM_DIR",
                              REPO / "Reward_and_Training" / "Quadcopter_SimCon" / "Simulation"))
for _p in (str(SIM_DIR), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quad_velocity_env import QuadcopterVelocityEnv, Wind                            # noqa: E402
from Success_Detector.success_detector import StreamingSuccessDetector, SuccessConfig  # noqa: E402
from Success_Detector.ppo_track_detector import SuccessDetector as PPOTrackDetector    # noqa: E402

DT = 0.005                 # simulator step [s]
EPISODE_SECONDS = 10.0     # benchmark horizon [s]
HOLD_SECONDS = 0.4         # final upright-and-slow dwell required by the detector [s]
PPO_TRACK_UPDATE_STEPS = 4 # the PPO track's detector is updated once per decision (4 x 5 ms)

# observation_noise_std=None does NOT disable noise in QuadcopterVelocityEnv:
# the simulator then keeps its default noise. Zeros switch every group off.
NOISE_OFF = {k: 0.0 for k in ("target", "pos", "z", "quat", "vel", "omega", "motor")}


@dataclass(frozen=True)
class Condition:
    """Evaluation condition; the default is the nominal benchmark."""
    name: str = "nominal"
    obs_noise_scale: float = 0.0      # x simulator default observation-noise std (0 = off)
    wind_speed: float = 0.0           # mean horizontal wind [m/s], random heading per episode (0 = off)
    wind_gusts: bool = False          # add the simulator's sinusoidal gusts (needs wind_speed > 0)
    init_tilt_deg: float = 0.0        # max initial tilt [deg] (0 = level start)
    init_rate: float = 0.0            # max initial body rate per axis [rad/s]
    init_vel: float = 0.0             # max initial linear velocity per axis [m/s]
    action_noise: float = 0.0         # std of the noise on the normalized motor commands
    gyro_noise: float = 0.0           # std of the noise on the observed body rates [rad/s]

    def __post_init__(self):
        for f in ("obs_noise_scale", "wind_speed", "init_tilt_deg", "init_rate", "init_vel",
                  "action_noise", "gyro_noise"):
            v = getattr(self, f)
            if not (np.isfinite(v) and v >= 0.0):
                raise ValueError(f"{f} must be finite and >= 0, got {v}")
        if self.wind_gusts and self.wind_speed == 0.0:
            raise ValueError("wind_gusts needs wind_speed > 0")
        if self.init_tilt_deg >= np.degrees(SuccessConfig().cone_rad):
            raise ValueError("init_tilt_deg must stay below the detector's upright cone "
                             f"({np.degrees(SuccessConfig().cone_rad):.0f} deg), or no flip can start upright")

    @property
    def perturbs_start(self) -> bool:
        return self.init_tilt_deg > 0.0 or self.init_rate > 0.0 or self.init_vel > 0.0

    @property
    def controller_noise(self) -> bool:
        return self.action_noise > 0.0 or self.gyro_noise > 0.0


NOMINAL = Condition()
OBS_NOISE = Condition("obs_noise", obs_noise_scale=1.0)     # the simulator's default noise
WIND = Condition("wind", wind_speed=1.0)                    # steady 1 m/s, random heading
PERTURBED_START = Condition("perturbed_start", init_tilt_deg=5.0, init_rate=0.2, init_vel=0.2)
STRESS = Condition("stress", action_noise=0.15, gyro_noise=0.2)   # the PPO track's "stress" point
WIND_STREAM = 7_000_001        # separate the wind and start-state generators from every other stream
INIT_STREAM = 7_000_002
CTRL_NOISE_STREAM = 12345      # as in the PPO track's rollout: SeedSequence([12345, seed])
NOISE_BLOCK_STEPS = 4          # one noise sample per 20 ms decision (their action-repeat)


def episode_wind(condition: Condition, seed) -> Wind:
    """The simulator Wind object for one episode: heading uniform in [0, 360) deg from
    np.random.default_rng((WIND_STREAM, seed)); no wind for wind_speed = 0."""
    if condition.wind_speed == 0.0:
        return Wind("NONE")
    rng = np.random.default_rng(None if seed is None else (WIND_STREAM, int(seed)))
    heading_deg = float(rng.uniform(0.0, 360.0))
    return Wind("SINE" if condition.wind_gusts else "FIXED", condition.wind_speed, heading_deg, 0.0)


def initial_perturbation(condition: Condition, seed) -> dict:
    """Perturbed start for one episode, from np.random.default_rng((INIT_STREAM, seed)):
    quat (wxyz, tilt about a random horizontal axis), vel [m/s], omega [rad/s]."""
    rng = np.random.default_rng(None if seed is None else (INIT_STREAM, int(seed)))
    axis_angle = rng.uniform(0.0, 2.0 * np.pi)
    tilt = np.deg2rad(rng.uniform(0.0, condition.init_tilt_deg))
    quat = np.array([np.cos(tilt / 2), np.sin(tilt / 2) * np.cos(axis_angle),
                     np.sin(tilt / 2) * np.sin(axis_angle), 0.0])
    vel = rng.uniform(-condition.init_vel, condition.init_vel, 3)
    omega = rng.uniform(-condition.init_rate, condition.init_rate, 3)
    return dict(quat=quat, vel=vel, omega=omega)


def apply_initial_perturbation(env: QuadcopterVelocityEnv, start: dict) -> np.ndarray:
    """Overwrite the freshly reset vehicle state and restart the integrator from it.
    Returns the observation of the perturbed start (drawn again, with the env's noise)."""
    q = env.quad
    q.state[3:7], q.state[7:10], q.state[10:13] = start["quat"], start["vel"], start["omega"]
    q.pos, q.quat, q.vel, q.omega = q.state[0:3], q.state[3:7], q.state[7:10], q.state[10:13]
    q.extended_state()
    q.integrator.set_initial_value(q.state, env.t)
    return env._get_obs()


@lru_cache(maxsize=1)
def _simulator_default_noise() -> tuple:
    env = QuadcopterVelocityEnv(obs_mode="full", observation_noise_std=None)
    try:
        return tuple(sorted(env.noise_std.items()))
    finally:
        env.close()


def simulator_default_noise() -> dict:
    """The simulator's own default observation-noise std per group (read, not copied)."""
    return dict(_simulator_default_noise())


def observation_noise(condition: Condition) -> dict:
    if condition.obs_noise_scale == 0.0:
        return dict(NOISE_OFF)
    return {k: condition.obs_noise_scale * v for k, v in simulator_default_noise().items()}


def make_benchmark_env(episode_seconds: float = EPISODE_SECONDS,
                       condition: Condition = NOMINAL) -> QuadcopterVelocityEnv:
    """The benchmark environment used for every controller (nominal unless `condition` says otherwise).
    The condition is stored on the env; run_episode applies its per-episode parts (wind) after reset."""
    env = QuadcopterVelocityEnv(obs_mode="full", dt=DT, episode_seconds=episode_seconds,
                                observation_noise_std=observation_noise(condition), random_wind=False,
                                terminate_on_unstable=True, max_abs_z=5.0, max_tilt_rad=np.inf)
    env.benchmark_condition = condition
    return env


def detector_config(dt: float = DT) -> SuccessConfig:
    """§04 thresholds, with the dwell expressed in simulator steps (0.4 s / 5 ms = 80)."""
    return SuccessConfig(hold_steps=int(np.ceil(HOLD_SECONDS / dt - 1e-9)))


@dataclass
class EpisodeResult:
    seed: int
    success: bool          # the §04 verdict, nothing else
    reason: str            # detector's one-line explanation
    terminated: bool       # episode ended by failure (not by the horizon)
    steps: int
    duration_s: float
    final_tilt_deg: float
    final_omega: float
    max_alt_loss_m: float
    t_flip_s: float | None
    t_recovered_s: float | None
    control_effort: float
    detector: dict         # full SuccessResult
    ppo_track: dict = field(default_factory=dict)   # the PPO track's verdict on the same episode
    trajectory: dict | None = field(default=None, compare=False, repr=False)


def hover_action(env: QuadcopterVelocityEnv) -> float:
    """Normalized command that maps to the hover motor speed (about +0.054)."""
    return 2.0 * (env.hover_w - env.min_w) / (env.max_w - env.min_w) - 1.0


def tilt_deg(quat) -> float:
    """Angle between the body z-axis and the vertical, from a scalar-first quaternion."""
    w, x, y, z = np.asarray(quat, float) / np.linalg.norm(quat)
    gz = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(gz, -1.0, 1.0))))


def _check_action(action) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32)
    if a.shape != (4,) or not np.all(np.isfinite(a)):
        raise ValueError(f"controller must return 4 finite motor commands, got {a!r}")
    return a


class _Recorder:
    """Collects the per-step trajectory; stacked into arrays at the end."""

    def __init__(self):
        self.rows = {k: [] for k in ("t", "pos", "quat", "vel", "omega", "action", "rev")}

    def add(self, env, action, rev):
        q = env.quad
        for k, v in (("t", env.t), ("pos", q.pos), ("quat", q.quat), ("vel", q.vel),
                     ("omega", q.omega), ("action", action), ("rev", rev)):
            self.rows[k].append(np.array(v, dtype=float, copy=True))

    def arrays(self) -> dict:
        return {k: np.stack(v) for k, v in self.rows.items()}


def run_episode(controller, seed: int, env: QuadcopterVelocityEnv | None = None,
                cfg: SuccessConfig | None = None, *, record: bool = False,
                episode_seconds: float = EPISODE_SECONDS,
                condition: Condition = NOMINAL) -> EpisodeResult:
    """One benchmark episode. The environment clips actions to [-1, 1] itself.
    `episode_seconds` and `condition` are used only when no environment is passed in."""
    own_env = env is None
    env = env if env is not None else make_benchmark_env(episode_seconds, condition)
    cfg = cfg if cfg is not None else detector_config(env.dt)
    try:
        controller.reset(seed=seed)
        obs, _ = env.reset(seed=seed)
        condition = getattr(env, "benchmark_condition", NOMINAL)
        if condition.wind_speed > 0.0:                  # otherwise keep the env's own (no) wind
            env.wind = episode_wind(condition, seed)
        if condition.perturbs_start:
            obs = apply_initial_perturbation(env, initial_perturbation(condition, seed))
        detector = StreamingSuccessDetector(cfg)
        detector.update(env.quad, count_hold=False)     # start state: upright check only
        ptd = PPOTrackDetector()
        ptd.reset()
        ptd.update(None, env=env)                       # start state, as at her env.reset
        pt_flip = pt_success = None
        a_hover = hover_action(env)
        z0 = float(env.quad.pos[2])                     # NED: z grows downwards
        max_drop = effort = 0.0
        t_flip = t_success = None
        rec = _Recorder() if record else None
        if rec:
            rec.add(env, np.full(4, np.nan), detector.revolution)
        terminated = truncated = False
        noisy = condition.controller_noise
        if noisy:
            nrng = np.random.default_rng(np.random.SeedSequence([CTRL_NOISE_STREAM, int(seed)]))
            gyro_n, act_n, k = np.zeros(3), np.zeros(4), 0
        while not (terminated or truncated):
            if noisy:
                if k % NOISE_BLOCK_STEPS == 0:
                    if condition.gyro_noise > 0.0:
                        gyro_n = nrng.normal(0.0, condition.gyro_noise, 3)
                    if condition.action_noise > 0.0:
                        act_n = nrng.normal(0.0, condition.action_noise, 4)
                k += 1
                seen = np.asarray(obs, dtype=np.float64).copy()
                seen[13:16] += gyro_n
                action = _check_action(controller.act(seen)[0])
                sent = np.clip(action.astype(np.float64) + act_n, -1.0, 1.0).astype(np.float32)
                applied = sent.astype(float)
                obs, _reward, terminated, truncated, _ = env.step(sent)
            else:
                action = _check_action(controller.act(obs)[0])
                applied = np.clip(action.astype(float), -1.0, 1.0)      # what the env applies, in float64
                obs, _reward, terminated, truncated, _ = env.step(action)
            detector.update(env.quad, terminated=bool(terminated))
            if env.steps % PPO_TRACK_UPDATE_STEPS == 0 or terminated or truncated:
                ptd.update(None, env=env)               # once per decision, and at the last state
                if pt_flip is None and ptd.passed_inverted and ptd.revolution >= ptd.cfg.full_rev_rad:
                    pt_flip = float(env.t)
                if pt_success is None and ptd.succeeded:
                    pt_success = float(env.t)
            if rec:
                rec.add(env, applied, detector.revolution)
            effort += float(np.sum((applied - a_hover) ** 2)) * env.dt
            max_drop = max(max_drop, float(env.quad.pos[2]) - z0)
            if t_flip is None and detector.result().completed_rotation:
                t_flip = float(env.t)
            if detector.succeeded:                     # start of the current successful streak
                t_success = t_success if t_success is not None else float(env.t)
            else:
                t_success = None
        result = detector.result()
        t_recovered = (t_success - (cfg.hold_steps - 1) * env.dt
                       if result.success and t_success is not None else None)
        return EpisodeResult(seed=int(seed), success=bool(result.success), reason=result.reason(),
                             terminated=bool(terminated), steps=int(env.steps),
                             duration_s=float(env.t),
                             final_tilt_deg=tilt_deg(env.quad.quat),
                             final_omega=float(np.linalg.norm(env.quad.omega)),
                             max_alt_loss_m=max_drop, t_flip_s=t_flip, t_recovered_s=t_recovered,
                             control_effort=effort, detector=result.as_dict(),
                             ppo_track=dict(
                                 success=bool(ptd.succeeded), passed_inverted=bool(ptd.passed_inverted),
                                 revolution_rad=float(ptd.revolution), t_flip_s=pt_flip,
                                 t_recovered_s=(None if pt_success is None else pt_success
                                                - (ptd.cfg.hold_steps - 1) * PPO_TRACK_UPDATE_STEPS * env.dt),
                                 success_then_terminated=bool(ptd.succeeded and terminated)),
                             trajectory=rec.arrays() if rec else None)
    finally:
        if own_env:
            env.close()


def evaluate(controller, seeds, *, episode_seconds: float = EPISODE_SECONDS,
             record: bool = False, condition: Condition = NOMINAL) -> list[EpisodeResult]:
    """Run the same controller on a fixed list of seeds (identical for every controller)."""
    env = make_benchmark_env(episode_seconds, condition)
    try:
        return [run_episode(controller, s, env, record=record) for s in seeds]
    finally:
        env.close()
