# Learning a Quadcopter Flip

Team project for the Intelligent Control course (University of Patras). A simulated quadrotor starts from hover, performs one complete 360° flip about its pitch axis, and recovers to a stable hover. We compare hand-designed, optimised and learned controllers. All of them are judged by the same evaluation harness, on the same seeds and disturbances, with a geometric success criterion that is independent of any reward.

The simulator is the UPatras LAR fork of `Quadcopter_SimCon` (`QuadcopterVelocityEnv`). It is used unchanged and is not part of this repository.

## Controllers

| Controller | Method | Code |
|---|---|---|
| Scripted flip | rule-based open-loop flip followed by a PID hover | `Reward_and_Training/baselines.py` |
| Three-phase | climb, rotate at a peak pitch rate, brake and recover; hand-tuned parameters | `Controllers/three_phase_flip.py` |
| MAP-Elites | the three-phase controller with parameters optimised by MAP-Elites | `MAP_Elites/`, `Controllers/map_elites_final.json` |
| BC-PPO | PPO with CTBR actions, warm-started by behaviour cloning from a hand-written expert | `Reward_and_Training/` |
| MTR-PPO | pure PPO with a multiplicative tracking reward, CTBR actions or direct motor commands | `MTR_PPO/` |

## Results

50 episodes per condition (seeds 10000–10049). Nominal: no disturbances. Noisy nominal: actuator noise σ = 0.05 and the simulator's default sensor noise. Stress: actuator noise σ = 0.15 and gyroscope noise 0.2 rad/s. Times, altitude loss and control effort are from the nominal condition.

| Controller | Nominal | Noisy nominal | Stress | Flip [s] | Recovery [s] | Altitude loss [m] | Effort |
|---|---|---|---|---|---|---|---|
| Scripted flip | 50/50 | 50/50 | 17/50 | 1.30 | 1.48 | 0.00 | 2.13 |
| Three-phase (default) | 50/50 | 50/50 | 14/50 | 0.62 | 0.84 | 0.00 | 1.12 |
| MAP-Elites | 50/50 | 50/50 | 15/50 | 1.52 | 2.30 | 0.00 | 0.69 |
| BC only | 50/50 | 50/50 | 13/50 | 0.68 | 3.44 | 0.07 | 2.63 |
| BC-PPO | 50/50 | 50/50 | 50/50 | 0.60 | 1.62 | 0.00 | 1.98 |
| MTR-PPO (CTBR) | 50/50 | 50/50 | 39/50 | 0.48 | 0.84 | 0.10 | 18.52 |
| MTR-PPO (motors) | 50/50 | 50/50 | 50/50 | 0.50 | 0.74 | 0.17 | 0.90 |

In every failure the flip itself was completed; what failed was the still hover afterwards. Raw results: `Logs/final_comparison/FINAL_PT_ALL/`. Figures: `Logs/final_comparison/report_figures/`. Demo videos: `Visualization/videos/`.

## Repository

| Folder | Contents |
|---|---|
| `Environment_Task_Wrapper/` | the flip task on top of the benchmark environment (termination, observation, action repeat) |
| `Success_Detector/` | the geometric success detector used for every reported number (`ppo_track_detector.py`) and a stricter variant |
| `Evaluation/` | evaluation harness, final comparison of all controllers, report figures |
| `Baselines/` | adapter that runs the baseline controllers in the harness |
| `Controllers/` | three-phase flip controller and the selected MAP-Elites genome |
| `MAP_Elites/` | MAP-Elites problem, archive, search loop, analysis and robustness screening |
| `Reward_and_Training/` | BC-PPO: reward, expert, policy, training, evaluation and analyses; delivered models in `models/` |
| `MTR_PPO/` | MTR-PPO: reference, reward, observation, environment, controller, training; reported models in `models/` |
| `Visualization/` | demo video renderers and the rendered videos |
| `Logs/` | final results, MTR-PPO training logs (runs 1–14), MAP-Elites archives and analyses |

Each script documents its options in its docstring (`python <script> --help`).

## Setup

Python 3.11 on CPU is enough.

```powershell
py -3.11 -m venv flip_env
flip_env\Scripts\activate
pip install -r Reward_and_Training/requirements.txt

git clone https://github.com/upatras-lar/Quadcopter_SimCon Reward_and_Training/Quadcopter_SimCon
$env:QUAD_SIM_DIR = "$PWD\Reward_and_Training\Quadcopter_SimCon\Simulation"
$env:PYTHONUTF8 = "1"
```

`numpy<2` is required: the simulator's dynamics break on numpy 2.x. Run every command from the repository root unless stated otherwise. Outputs go to `runs/` folders, which git ignores.

## Reproducing the results

Final comparison, figures and videos:

```powershell
python Evaluation/final_comparison.py --name FINAL_PT_ALL --detector ppo_track --seed0 10000 --episodes 50 --conditions nominal ppo_track_nominal ppo_track_stress --controllers pid_hover scripted_flip three_phase_default map_elites_final bc_only bc_ppo ppo_mtr ppo_mtr_motors
python Evaluation/plot_comparison.py --runs FINAL_PT_ALL --name success_rates --controllers scripted_flip three_phase_default map_elites_final bc_only bc_ppo ppo_mtr ppo_mtr_motors
python Evaluation/flip_analysis.py --name flips_nominal
python Visualization/make_controller_videos.py
```

MTR-PPO, the reported models (run 13 with CTBR actions, run 14 with motor commands, each the final model after 3M steps):

```powershell
python MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --kernels dense --ref-start-prob 0.5 --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1 --axis-term --axis-limit-deg 40 --axis-penalty 2 --timesteps 3000000
python MTR_PPO/train_mtr.py --action-mode motors --seed 0 --kernels dense --ref-start-prob 0.5 --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1 --axis-term --axis-limit-deg 40 --axis-penalty 2 --timesteps 3000000
python MTR_PPO/plot_training.py --name mtr_final "CTBR (run 13)=Logs/MTR_PPO/runs/run13_ctbr_seed0_scratch_all_components" "motors (run 14)=Logs/MTR_PPO/runs/run14_motors_seed0_scratch_all_components"
```

BC-PPO, from the `Reward_and_Training/` folder (the defaults are the final configuration; `run_all.bat` runs the seeds, ablations and analyses):

```powershell
python train_ppo.py --timesteps 1000000
python eval_ppo.py --model models/ppo_flip.zip --episodes 50 --action-noise 0.05 --plot
```

MAP-Elites. ME1 uses the default genome box (`t_climb` up to 0.6 s), ME2 widens it to 1.0 s; each configuration was searched with seeds 0, 1 and 2 (`ME2`, `ME2_s1`, `ME2_s2`). The reported genome comes from `ME2_s1`, as recorded in `Controllers/map_elites_final.json`:

```powershell
python MAP_Elites/map_elites.py --name ME2_s1 --seed 1 --budget 5000 --bound t_climb=0,1.0
python MAP_Elites/analysis.py --compare ME1=ME1,ME1_s1,ME1_s2 ME2=ME2,ME2_s1,ME2_s2
python MAP_Elites/robustness.py --name R1 --runs ME1 ME2 ME3 ME1_s1 ME1_s2 ME2_s1 ME2_s2 --top 10 --episodes 10
```
