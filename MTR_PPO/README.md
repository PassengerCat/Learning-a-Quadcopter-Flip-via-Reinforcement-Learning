# MTR-PPO: Multiplicative Tracking Reward PPO

Pure-RL PPO for one 360° pitch flip followed by recovery to hover. There is no behaviour cloning and there are no demonstrations. The policy is trained against a body-frame tracking reference with a multiplicative reward. The reward structure (product of kernel sums H(x; k) = 1/(1 + kx) over position, velocity, rate and command errors, times a task factor) and the reference invariants are adapted from GEAR (*Multi-Task Reinforcement Learning of Drone Aerobatics by Exploiting Geometric Symmetries*, arXiv 2602.10997). The additions that made the flip learnable on this vehicle are our own (see below).

## Files

| File | Contents |
|---|---|
| `mtr_reference.py` | `FlipCommand`, `LoopReference`: the loop (ω, r) and hover references and the body-frame relative state |
| `mtr_reward.py` | `MultiplicativeTrackingReward`, `MTRRewardConfig`: the reward, kernel sets, post-flip survival bonus |
| `mtr_observation.py` | `TrackingObsBuilder`: the 25-value actor observation (shared by training and deployment) |
| `mtr_env.py` | `MTRFlipEnv`, `make_mtr_env`: the training environment on the PPO track's `FlipTaskEnv` |
| `mtr_controller.py` | `MTRController`: the trained policy behind the course interface (evaluation harness) |
| `train_mtr.py` | the training script (`--help` lists every option) |
| `models/ctbr_seed0/` | the reported model (CTBR actions) and its run configuration |
| `runs/` | training outputs (ignored by git) |

The PPO track's modules in `Reward_and_Training/` are used unchanged: `FlipTaskEnv`, the asymmetric actor-critic policy, the CTBR rate loop and her success detector.

## Method

| Component | Setting |
|---|---|
| Reference | FLIP: loop of radius r = 0.6 m at pitch rate ω (training ω ~ U[4.5, 5.5] rad/s, evaluation 5 rad/s). HOVER: the start position, at rest. FLIP switches to HOVER when the geometric revolution reaches 2π − 0.15. |
| Reward | r = r_pos · r_lin · r_ang · r_cmd · r_task, normalised so that perfect flip tracking earns 1/s and perfect hover 0.5/s. "Dense" kernels: one coarse and one fine kernel per term (GEAR's narrow set gives no learning signal from hover). |
| Reference state initialisation | half of the training episodes start on the loop at a random phase |
| Termination on turning back | during FLIP, the episode ends if the revolution falls 30° below its maximum (removes a swing exploit) |
| Post-flip survival bonus | during HOVER only: 0.5/s × (1 + cos tilt)/2 × 1/(1 + \|v\|²) (pays for stopping upright) |
| Actions | CTBR (collective thrust and body rates, through the PPO track's fixed rate loop) or direct motor commands; 50 Hz decisions |
| Observation | actor 25 values (relative state, previous action, task, ω); critic also the 26 privileged values |

## Commands

    python MTR_PPO/train_mtr.py --dry-run
    python MTR_PPO/train_mtr.py --action-mode motors --seed 0 --kernels dense --ref-start-prob 0.5 --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1
    python Evaluation/final_comparison.py --name FINAL_PT_MTR --detector ppo_track --seed0 10000 --episodes 50 --conditions nominal ppo_track_nominal ppo_track_stress --controllers ppo_mtr

The reported CTBR model was trained in three stages with `--init-model` (`models/ctbr_seed0/README.md`). The development history, every check and every run are in `Documentation/CHANGELOG_EXPERIMENTS.md` and `Logs/MTR_PPO/`.
