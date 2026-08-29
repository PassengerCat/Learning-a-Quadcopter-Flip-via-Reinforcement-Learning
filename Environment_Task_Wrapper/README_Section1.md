# Section 1 — Environment & Task Wrapper (Quadcopter Flip)

This folder implements Stage 1 of the project: turning the velocity-tracking
`QuadcopterVelocityEnv` into a **flip** task, cleanly and reproducibly.

## Files
- `flip_env.py` — the wrapper stack + factories + the deployable observation transform.
- `calibrate_flip_env.py` — one-off sanity + normalisation calibration.

## Dependencies (important)
```
numpy<2      # the bobzwik dynamics BREAK on numpy 2.x (SystemError in the integrator)
scipy<1.13
gymnasium
```
The simulator is CPU-bound pure Python.

## Setup
`flip_env.py` must be importable next to `quad_velocity_env.py` — put it in the
repo's `Simulation/` folder (or add that folder to `PYTHONPATH`). Then:
```python
from flip_env import make_flip_env, make_vec_env, ObsConfig
env = make_flip_env(action_repeat=4)          # single env
venv = make_vec_env(n_envs=8, action_repeat=4)  # parallel (for on-policy training)
```

## What it does — and the decisions behind it (all verified against the env)

**Disable ONLY the roll/pitch termination.** `max_tilt_rad=np.inf` is a plain
constructor argument, so no monkeypatch is needed. `terminate_on_unstable=True`
and `max_abs_z` are kept, so the altitude, non-finite and max-duration
terminations required by the brief remain active. Calibration confirms: **0 tilt
terminations, altitude termination still fires.**

**Custom reward via the `reward_fn=` hook** (no patch). A neutral
`placeholder_reward` is installed so we never accidentally use the env's default
reward, which penalises pitch and would fight the flip. **The real flip reward is
designed in Section 3** and dropped in here.

**Target velocity fixed to (0,0,0)** — already the env default; the 3 target
fields are dropped from the policy observation.

**Action-repeat k=4 (frame-skip).** The policy decides at 50 Hz (dt·k = 20 ms)
instead of 200 Hz; reward is summed over the k sub-steps; an episode stops early
if a sub-step ends. This shortens the horizon 2000 → ~500 decisions, improves
exploration (a perturbation persists long enough to move the vehicle), and
smooths control. It does **not** change the sim integration dt (fidelity kept).
Ablation knob: `action_repeat ∈ {2, 4, 8}`.

**Observation: fixed physical scales + clip, not running mean/std.** Chosen
because the flip regime appears late in training, so running statistics would
mis-scale exactly the states that matter; fixed scales are stationary and
trivially reproducible at deployment. Scales (see `ObsConfig`): pos/5, vel/10,
body-rates/20 (**magnitude kept** — the flip needs pitch-rate magnitude, so we do
*not* normalise rates to a unit direction), motors mapped to [-1,1] by the known
[min_w, max_w], everything clipped to ±10.

**Attitude representation = projected gravity vector (default).** Gravity in the
body frame (`quat_to_projected_gravity`), verified: upright → [0,0,+1], inverted
→ [0,0,-1]. It drops only yaw (irrelevant to the flip → yaw-invariance for free)
and hands the network the "how inverted am I" signal continuously and linearly.
Ablation knob: `ObsConfig(attitude=...)` ∈ {`"gravity"` (3), `"quat"` (4),
`"rotmat"` (9)}.

**The observation transform is a pure function** (`transform_observation`). Use
the *same* function inside `Controller.act` at deployment so training and
evaluation see identical inputs (a common source of silent eval failures).

**No RNN/stacking** — full state is Markov. Previous decision-action is appended
to the observation by default (`include_prev_action=True`); pair it with an
action-rate penalty in the Section 3 reward, or turn it off.

**Integrator guard.** `ActionRepeat` wraps `env.step` in try/except and treats a
raised integrator error as a terminal unstable step. With numpy<2 the sim is
robust even under extreme actions, so this is cheap defensive insurance rather
than a frequently-needed path.

## Verified env facts (numpy<2)
- obs "full" = 20-dim: target(3) | pos(3) | quat(4, scalar-first, body→world, NED)
  | lin_vel(3) | body rates(3) | motor speeds(4).
- motor speeds ∈ [75, 925], hover ≈ 523 (hover action ≈ +0.054).
- reset → origin, level (quat [1,0,0,0]), zero vel/omega, hover motors.
- dt=0.005, 2000 steps/episode (10 s). `reward_fn(target_velocity, full_state, action, env)`.

## Open (later sections)
Reward design (§3), algorithm choice (PPO vs off-policy), the meaningful competing
method (§2/§3), success detector (§4). The wrapper is decoupled from all of these.
