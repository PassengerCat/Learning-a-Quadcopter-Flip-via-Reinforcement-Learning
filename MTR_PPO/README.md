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
| `models/ctbr_seed0/` | the reported model with CTBR actions (controller `ppo_mtr`) and its run configuration |
| `models/motors_seed0/` | the reported model with direct motor commands (controller `ppo_mtr_motors`) and its run configuration |
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
| Pitch-axis term (optional, `--axis-term`) | multiplies the product by Σ_{k∈{1,10}} H(\|R e_y − R_z(ψ) e_y\|²; k), so the flip stays a pure pitch flip and the heading is kept |
| Pitch-axis penalty (optional, `--axis-limit-deg d --axis-penalty P`) | during FLIP, the first time the pitch axis deviates by more than d (roll and heading together), the reward drops once by P; nothing is terminated |
| Actions | CTBR (collective thrust and body rates, through the PPO track's fixed rate loop) or direct motor commands; 50 Hz decisions |
| Observation | actor 25 values (relative state, previous action, task, ω); critic also the 26 privileged values |

## Commands

    python MTR_PPO/train_mtr.py --dry-run
    python MTR_PPO/train_mtr.py --action-mode motors --seed 0 --kernels dense --ref-start-prob 0.5 --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1
    python Evaluation/final_comparison.py --name FINAL_PT_MTR --detector ppo_track --seed0 10000 --episodes 50 --conditions nominal ppo_track_nominal ppo_track_stress --controllers ppo_mtr
    python Evaluation/final_comparison.py --name FINAL_PT_MTR_MOTORS --detector ppo_track --seed0 10000 --episodes 50 --conditions nominal ppo_track_nominal ppo_track_stress --controllers ppo_mtr_motors

The reported models (runs 13 and 14) were each trained from a fresh network with every component from the first step:

    python MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --kernels dense --ref-start-prob 0.5 --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1 --axis-term --axis-limit-deg 40 --axis-penalty 2 --timesteps 3000000 --tag mtr_ctbr_clean
    python MTR_PPO/train_mtr.py --action-mode motors --seed 0 --kernels dense --ref-start-prob 0.5 --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1 --axis-term --axis-limit-deg 40 --axis-penalty 2 --timesteps 3000000 --tag mtr_motors_clean

Both reported models are the final models of their runs (`models/ctbr_seed0/README.md`, `models/motors_seed0/README.md`). The development history (runs 1–14), every check and every run are in `Documentation/CHANGELOG_EXPERIMENTS.md` and `Logs/MTR_PPO/`.
