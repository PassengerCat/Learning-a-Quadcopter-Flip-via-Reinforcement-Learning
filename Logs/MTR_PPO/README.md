# MTR-PPO logs

- `checks/`: one log per development step (steps 1–10, plus step 11 for the rename). Each records checks run in the cloud with a temporary script that was deleted after it passed. No check involved training.
- `runs/`: one folder per training run, executed by the user. Each holds `config.json` (absolute local paths shortened to `<repo>`), `eval_log.csv` (harness evaluation during training), `training_curves.txt` (from TensorBoard) and, where needed, a replay of the policy.

| Run | Start | Added | Steps | Outcome |
|---|---|---|---|---|
| 1 | fresh | narrow (GEAR's) kernels, hover starts | 1.1M | no learning signal |
| 2 | fresh | + reference starts | 0.3M | reward too narrow |
| 3 | fresh | dense kernels | 2.95M | swing exploit, no flip |
| 4 | fresh | + termination on turning back | 1.5M used | flip learned, no recovery |
| 5 | run 4 at 1.5M | + survival bonus | 1.0M | survives, drifts away |
| 6 | run 5 final | + speed weighting | 1.0M | flip and recovery: the reported model |

## Names used before the rename (step 11)

The method was developed under the working name "GEAR tracking". Runs 1–6 and checks 1–10 were produced under the earlier names. Their raw files (`config.json`, TensorBoard tags, check logs) keep those names. The code is otherwise identical: step 11 checks it bit for bit.

| Earlier name | Current name |
|---|---|
| `Reward_and_Training/gear_tracking/` | `MTR_PPO/` |
| `reference.py`, `tracking_reward.py`, `observation.py`, `tracking_env.py`, `controller.py`, `train_tracking.py` | `mtr_reference.py`, `mtr_reward.py`, `mtr_observation.py`, `mtr_env.py`, `mtr_controller.py`, `train_mtr.py` |
| `GearReference`, `GearTrackingReward`, `GearRewardConfig`, `GearObsBuilder`, `GearTrackingEnv`, `make_gear_env`, `GearTrackingController` | `LoopReference`, `MultiplicativeTrackingReward`, `MTRRewardConfig`, `TrackingObsBuilder`, `MTRFlipEnv`, `make_mtr_env`, `MTRController` |
| kernel set `gear` | kernel set `narrow` |
| info / TensorBoard keys `gear_task`, `gear_omega`, `gear_ref_start`, `gear_backtrack` | `mtr_task`, `mtr_omega`, `mtr_ref_start`, `mtr_backtrack` |
| run tags `gear`, `gear_rsi`, `gear_dense`, `gear_bt`, `gear_ft`, `gear_ft2`; checkpoints `ppo_gear_*` | default tag `mtr`; checkpoints `ppo_mtr_*` |
| check logs `g1`–`g10` | `checks/step01`–`step10` |
