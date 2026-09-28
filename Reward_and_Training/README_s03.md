# §03 — Reinforcement-Learning Loop (Quadcopter Flip)

A PPO policy learns the full manoeuvre: climb, a 360° pitch flip through inverted, and recovery to hover. It is tested on the real course simulator (`upatras-lar/Quadcopter_SimCon`). The simulator is never modified.

## Files

Put all of these in the fork's `Simulation/` folder, next to `quad_velocity_env.py`, `flip_env.py` (§01), and `baselines.py` / `run_baselines.py` (§02, updated versions included here).

| File | Role |
|---|---|
| `flip_reward.py` | Reward (`FlipReward`) and the independent success detector (`SuccessDetector`, §04). |
| `flip_policy.py` | **Deployable** parts: progress tracker, observation builder, CTBR mapper, asymmetric actor-critic, and `LearnedFlipController` (the §02 `Controller` interface). |
| `flip_task_env.py` | Training wiring: reward hook, actor + privileged observation, randomised and demonstration starts, optional wind randomisation. |
| `ctbr_expert.py` | Hand-written flip controller in the policy's action space. It provides the behaviour-cloning demonstrations. |
| `train_ppo.py` | BC warm start → PPO fine-tuning, with periodic evaluation through the deployed controller and the §04 detector. |
| `visualize_flip.py` | Report figures and an animation: side-view filmstrip, phases/commands plot, and a side-by-side GIF (scripted vs PPO, same seed). |
| `run_all.bat` | **Depth experiments in one command** (seeds, ablations, memory experiment, analyses). See `README_depth.md`. |
| `aggregate_runs.py` | Seeds and ablations → tables (mean ± std), learning curves, ablation bars, on a fresh seed bank. |
| `robustness_sweep.py` | Success, stillness and drift as functions of noise level and wind speed. |
| `optimal_flip_analysis.py` | Learned vs scripted vs time-optimal (bang-bang) flip: timing, phase portrait, thrust while inverted. |
| `failure_analysis.py` | Coning, noise and wind failures, each with a mechanism and a control experiment. |
| `eval_ppo.py` | Evaluates a trained policy next to the §02 baselines with the identical harness. |
| `run_baselines.py` | §02 runner, now with the §04 detector column and support for extra controllers. |
| `test_flip_reward.py`, `test_flip_reward_extra.py` | Your original reward tests (all still pass). |
| `test_flip_policy.py` | 15 new tests. They include train/deploy observation identity, anti-coning, CTBR consistency, expert feasibility, wind randomisation and the PID hand-off. |
| `reward_log.md` | Log of every reward iteration (runs 1–11, including the wind experiments). It is the input for §06. |

## Run

```bash
pip install -r requirements.txt            # Python 3.11/3.12
pytest -q test_flip_policy.py              # ~10 s
python test_flip_reward.py && python test_flip_reward_extra.py
python train_ppo.py --smoke                # ~3 min pipeline check
python train_ppo.py --timesteps 1000000    # final config
python eval_ppo.py --model runs/<run>/best_model.zip --episodes 50 --action-noise 0.05 --plot
tensorboard --logdir runs/
```

`--n-envs` defaults to *cores − 1*. The simulator is CPU-bound (≈200 decisions/s per core; slower during aggressive manoeuvres), so more cores means proportionally faster training.

## Architecture

```
raw obs (20) ──► ActorObsBuilder ──► actor MLP 128×128 ──► a = [thrust, p, q, r] ∈ [-1,1]^4
                 transform_observation (§01)                 │
                 + [sin θ, cos θ, progress, phase]           ▼
                   (geometric flip angle θ)            CTBRMapper: fixed P rate loop + §02 mixer
privileged (26) ─────────────────────► critic MLP 256×256   │
 noise-free state, reward phase, time left                  ▼
                                                     4 motor commands → env
```

- **CTBR actions (collective thrust + body rates).** The policy still decides the whole manoeuvre: when to rotate, how hard, and how much thrust. A fixed, stateless rate loop turns rate commands into motor speeds. This is the standard choice for learned agile flight: it learns faster and transfers better than raw motor commands. `--action-mode motors` is kept as an ablation.
- **Asymmetric critic.** The critic also sees privileged, noise-free state. At deployment that block is zero-filled, and the actor never reads it (tested).
- **Train = deploy.** The same `ActorObsBuilder` and `CTBRMapper` objects are used in training and inside `LearnedFlipController`. A test checks bit-for-bit identity.
- **Geometric flip angle.** θ = atan2(−g_x, g_z) is the angle of gravity in the body x-z plane. The reward, the observation features and the §04 detector all use it. It cannot be inflated by coning (see run 4).

## Training recipe (final configuration)

1. **Behaviour cloning, DART-style.** 120 episodes of `ctbr_expert` are recorded with execution noise σ ∈ {0, 0.05, 0.1, 0.2} while the labels stay clean. The actor mean is regressed on the expert actions and the critic on the returns-to-go.
2. **PPO fine-tuning** on the true reward: lr 1e-4 → 3e-5, exploration std e^−1.6, γ 0.99, GAE 0.95, clip 0.2. 30% of training episodes start mid-flip from the §02 scripted flip, and all start from a randomised hover.
3. **Model selection** by the §04 detector on deterministic evaluation episodes, never by the training return.

Why a warm start (see `reward_log.md`): pure PPO learned to hover (run 5) or to "pump" to ~100° and back (run 8). It never committed to the part of the manoeuvre that only pays off once complete. The expert scores a return of ≈17 against ≈1 for those policies, so the reward was right and exploration was the bottleneck.

## Results

The final model is in `models/ppo_flip.zip` with its `config.json`. It was trained with BC warm start plus 200k PPO steps, on 2 CPU cores in about 1 h. Every controller runs through the **same harness, env, seeds and noise**. Success is judged by the independent **§04 detector**: a full geometric 360°, passing through inverted, then upright (<15°) and slow (|v| < 0.4 m/s, |ω| < 0.8 rad/s) for 0.4 s, with no crash.

**Nominal (50 episodes, actuator noise σ = 0.05, the env's default sensor noise)**

| Controller | Success §04 (95% CI) | Final tilt [°] | Final ‖ω‖ [rad/s] | Altitude loss [m] | Recovery [s] | Control effort |
|---|---|---|---|---|---|---|
| random | 0% (0–7) | 94.4 | 11.52 | 5.02 (crash) | — | 1.346 |
| pid_hover | 0% (0–7) | 1.6 | 0.31 | 0.02 | — | 0.013 |
| scripted_flip (§02) | 100% (93–100) | 1.6 | 0.31 | 0.01 | **0.40** | 0.216 |
| BC only | 100% (93–100) | 1.4 | 0.31 | 0.08 | 0.83 | 0.271 |
| **PPO (BC + fine-tune)** | **100% (93–100)** | **0.6** | **0.16** | **0.00** | 0.51 | **0.208** |

**Stress (20 episodes, actuator noise σ = 0.15, gyro noise σ = 0.2 rad/s)**

| Controller | Success §04 (95% CI) | Final tilt [°] | Final ‖ω‖ [rad/s] | Recovery [s] | Control effort |
|---|---|---|---|---|---|
| pid_hover | 0% (0–16) | 4.7 | 0.95 | — | 0.198 |
| scripted_flip (§02) | 20% (8–42) | 4.7 | 0.95 | 3.09 | 0.389 |
| BC only | 30% (15–52) | 4.5 | 0.98 | 3.12 | 0.381 |
| **PPO (BC + fine-tune)** | **100% (84–100)** | **2.0** | **0.42** | **0.52** | **0.352** |

Reading these tables:
- **Nominal conditions.** Both the scripted flip and the PPO policy complete the flip every time. PPO ends more precisely (tilt 0.6° vs 1.6°, ‖ω‖ 0.16 vs 0.31) with less effort. It also flips faster: it is inverted at ≈0.45 s, against ≈1.1 s for the scripted flip, and it climbs less (≈1.4 m vs 2.3 m; see `results/nominal/baselines_timeseries.png`).
- **Stress.** This is where the methods separate. The open-loop scripted flip and the pure imitation both fail to settle (20% / 30%). The PPO policy, fine-tuned on the true reward under the env's sensor noise and randomised starts, still succeeds 100% of the time and recovers in 0.5 s.
- **Ablation, BC vs BC + PPO.** The imitation alone already flips. RL fine-tuning is what gives precision (recovery 0.83 → 0.51 s), efficiency (effort 0.271 → 0.208) and robustness (stress success 30% → 100%).
- **Ablation, pure RL** (runs 1–8 in `reward_log.md`). Without demonstrations PPO never completed a flip in 0.2–0.5M steps: it spun, coned, hovered or pumped. That is a result worth reporting, not hiding.

### All conditions, including wind (20 episodes each, same seeds; `results/final/`)

Wind is the env's own RANDOMSINE model (`enable_random_wind`): median speed as given, gusts of ±1–3 m/s on top, random heading. **PPO + PID** is the methodology's hybrid (`--pid-handoff`): the learned policy flies the flip and brings the vehicle upright, then the §02 PID "recovery block" holds the start position.

| Condition | Metric | Scripted (§02) | PPO (§03) | PPO + PID |
|---|---|---|---|---|
| Nominal (actuator σ 0.05) | §04 success | 100% | 100% | 100% |
| | horizontal drift | 0.18 m | 0.31 m | **0.03 m** |
| Stress (actuator σ 0.15, gyro σ 0.2 rad/s) | §04 success | 20% | **100%** | 20% |
| | final ‖ω‖ | 0.95 rad/s | **0.42 rad/s** | 0.95 rad/s |
| Wind, 3 m/s | §04 success | 100% | 100% | 100% |
| | horizontal drift | 0.13 m | 1.04 m | **0.01 m** |
| Wind, 5 m/s | §04 success | 100% | 100% | 100% |
| | horizontal drift | 0.21 m | 1.86 m | **0.04 m** |

**The trade-off, stated plainly.**
- **Every controller completes the flip in every condition except stress.** Under stress, the PID-based ones (scripted, and PPO + PID after the hand-off) jitter above the detector's 0.8 rad/s stillness threshold.
- **The learned policy is the most robust to actuator and sensor noise.** It is the only controller that settles under stress.
- **It does not hold position against steady wind.** It is memoryless (no integral action), so it drifts downwind by 1–2 m while staying upright and still.
- **Trying to *learn* position hold did not work in our budget.** Fine-tuning with wind (runs 10–11, `results/ablation_wind_finetune/`) did not help: drift at 5 m/s went from 2.0 m to 2.9 m, and success fell to 90%.
- **The hybrid removes the drift but brings back the PID's sensitivity to gyro noise.** The choice depends on which disturbance matters more. That is a fair, reportable conclusion.

Reproduce:

```bash
python eval_ppo.py --model models/ppo_flip.zip --episodes 50 --action-noise 0.05 --plot --out results/nominal
python eval_ppo.py --model models/ppo_flip.zip --episodes 20 --action-noise 0.15 --obs-noise 0.2 --out results/stress
python eval_ppo.py --model models/bc_flip.zip  --episodes 50 --action-noise 0.05 --no-baselines --out results/nominal_bc
python eval_ppo.py --model models/ppo_flip.zip --pid-handoff --episodes 20 --wind 5 --out results/final/wind5
```

## Changes to the files you sent (and why)

| File | Change | Reason |
|---|---|---|
| `flip_reward.py` | Rotation measured from **gravity in the body x-z plane** (`gravity_pitch_increment`). The integrated rate / body-y quaternion increment is no longer used. | These are fooled by coning (run 4). |
| | Tent potential beyond 360° (`over_rot_slope`). | Spin-and-crash (run 1). |
| | `w_prog` "new record" term, dense `b_upright`, `lam_z`, `alive_after_flip`. | Exploration and recovery gradients (runs 3–8). |
| | `lam_dact` 0.02 → 0.005, applied to the policy action (`smooth_action`). | Runs 2–3. |
| | `SuccessDetector`: geometric angle **and** mandatory inversion (tilt ≥ 150°). | Coning would have passed the old detector. |
| `flip_task_env.py` | Rewritten. The observation is built from the raw obs by the deployable builder (not from reward internals). Adds privileged critic obs, CTBR actions, randomised and demonstration starts, crash penalty on integrator failure, and the detector in `info`. | Train/deploy mismatch; missing asymmetric critic. |
| `train_ppo.py` | Rewritten. Adds asymmetric policy, BC warm start, LR/entropy schedules, evaluation through the deployed controller + §04 detector, best-model selection, resume, ablation flags. | The old eval read reward internals, and pure PPO did not learn the flip. |
| `test_flip_reward*.py` | Unchanged, all pass. | |
| `requirements.txt` | Adds torch/pytest and the Python-version note. | |

**§01 note.** In `flip_env.FlipObservation.step` the observation carries the action from *two* decisions ago, because `_prev_action` is updated after the observation is built. §03 no longer uses that wrapper, so nothing here depends on it. If you keep it for anything else, swap the two lines.
