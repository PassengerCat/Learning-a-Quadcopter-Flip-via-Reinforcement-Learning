# §03 — Reward, Wiring & Training

Everything for the flip reward and the first PPO run. **§01 (`Environment_Task_Wrapper/flip_env.py`)
is not modified** — the wiring is additive (composition).

## Files (put this folder next to `Environment_Task_Wrapper/`)
- `flip_reward.py` — `FlipReward` (reward), `SuccessDetector` (independent, geometric), `FlipRewardConfig`.
- `flip_task_env.py` — `make_flip_task_env` / `make_vec_task_env`: §01 env + reward + the flip-feature obs wrapper.
- `train_ppo.py` — PPO (SB3) + parallel envs + eval/logging callback.
- `test_flip_reward.py`, `test_flip_reward_extra.py` — unit tests (import only `flip_reward`; run anywhere).

## How the wiring works (no §01 edits)
`make_flip_task_env` builds the §01 env with `reward_fn = FlipReward(...)`, then wraps it in
`FlipFeatureObservation`, which:
1. **appends 4 features** to the policy observation — `[sin α, cos α, progress, phase]` — because the
   shaping potential depends on the accumulated angle `α` and the phase, so the policy MUST observe them
   (else the task is non-Markov). With `attitude="gravity"`, obs dim goes **20 → 24**.
2. calls `reward.reset()` on every episode reset so the shared reward stays in sync.

Two rotation measures: `α = ∫q dt` (integral) feeds the shaping; a drift-free **quaternion revolution
count** drives phase transitions and success — so integration drift never corrupts a decision.

## Setup (confirmed working — validated end-to-end on the real LAR sim)
Deps are in `requirements.txt`: `numpy<2`, `scipy<1.13`, `gymnasium`, **`matplotlib`**
(the fork's `utils/display.py` imports it), `stable-baselines3>=2.0`, `tensorboard`.

```bash
# 1) a clean venv (normal CPython 3.10/3.11 — NOT the MSYS2 one; torch installs cleanly there)
py -3.11 -m venv .venv
.venv\Scripts\activate                       # Windows  (bash: source .venv/bin/activate)
pip install -r requirements.txt

# 2) get the fork simulator and point at it (it is NOT a pip package)
git clone https://github.com/upatras-lar/Quadcopter_SimCon
set QUAD_SIM_DIR=C:\...\Quadcopter_SimCon\Simulation     # Windows (bash: export QUAD_SIM_DIR=.../Simulation)
```
`flip_task_env` / `train_ppo` add `QUAD_SIM_DIR` (and the sibling `Environment_Task_Wrapper/`) to
`sys.path` automatically, so `import quad_velocity_env` / `import flip_env` just work.

## Run it
```bash
# from the Reward_and_Training/ folder, venv active, QUAD_SIM_DIR set
python train_ppo.py --smoke                 # 3k steps, 1 env — sanity-check the whole pipeline first
python train_ppo.py --timesteps 3000000 --n-envs 8
tensorboard --logdir runs/
```
Run-1 config is locked in the defaults: **`s_alive = 0`, `gamma_shape = 1`**, everything else default.
Ablations: `--s-alive 0.005` / `0.02`; edit `FlipRewardConfig` for `k_v`,`k_ω`, `w_rot`, dual-bandwidth.

## What to watch (logged to TensorBoard by the eval callback)
`flip/success_rate` (independent detector) · `flip/reached_inverted_rate` · `flip/crash_rate` ·
`flip/mean_revolution_rad` · `flip/mean_alpha_rad` · **`flip/alpha_vs_revolution_drift`** (should stay
small — confirms `λ_off` keeps `α ≈ true angle`) · `flip/mean_return`.

Stall check: with `s_alive=0` there is no pay for hovering, so if it still never rotates, raise `w_rot`
or check exploration. Break-even (for later `s_alive` ablations): `s_alive=0.02` → stall below `q*≈6.3`
rad/s; `0.005` → `q*≈1.6`. Keep `w_rot` dominant over `s_alive`.

## Verified
- `test_flip_reward.py` (8) + `test_flip_reward_extra.py` (5): progress/cap, wobble-neutral, milestones,
  action-smoothness once/decision, detector (clean vs half vs wobble), reset, negative direction,
  bounded transition hand-off, hold-bonus accrual, no-NaN.
- End-to-end wiring smoke (mock sim, in the cloud): obs → 24-dim, reward flows through the hook,
  scripted flip → `revolution≈2π`, `phase=RECOVER`, success fires, `α`-drift ≈ 0.12.

## full_state layout — CONFIRMED
The LAR env calls `reward_fn(target_velocity, self.quad.state, action, self)`, i.e. `full_state` is the
**21-dim raw sim state** `[pos(3) | quat(4) | vel(3) | omega(3) | motors(8)]`, and `env.quad` is present.
`StateView` reads `env.quad.{pos,quat,vel,omega}` directly (falling back to parsing the 21-dim vector),
so attitude is correct (upright → `gz=+1`). Verified on the real sim: obs → 24-dim, reward flows,
crash-detection fires, and in a `--smoke` run `alpha_vs_revolution_drift ≈ 0.018` (α tracks the geometric
revolution tightly).

## Design finding: `gamma_shape = 1.0`
Discounted shaping `γΦ′−Φ` with `γ<1` bleeds `(γ−1)Φ` per step while sitting at a constant-`Φ` goal
(a finite-training nuisance, NOT a correctness bug — the Ng optimum is preserved with `γ_shape=γ`). We
use the **difference form** `Φ′−Φ` (`gamma_shape=1`) so holding the hover isn't bled. Your PPO discount
stays `γ=0.99`, untouched. Set `gamma_shape=0.99` for the strict Ng form (then raise `b_hold`).
