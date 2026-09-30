# Documentation: evaluation, success detectors and MAP-Elites

- [Changelog](CHANGELOG_EXPERIMENTS.md): every change, check and result, in order.
- [Methods for the report](map_elites/METHODS_FOR_REPORT.md): each verified method, how it was checked, and where the evidence is.
- `../Logs/`: the check logs and run analyses referred to above.

## Folders

| Folder | Contents |
|---|---|
| `Success_Detector/` | the PPO track's detector (used for reported numbers) and the stricter §04 detector ([README](../Success_Detector/README_Section4.md)) |
| `Controllers/` | three-phase flip controller; the selected MAP-Elites genome and its loader |
| `Evaluation/` | the evaluation harness, the final comparison and the demo renderer |
| `MAP_Elites/` | problem, archive, search loop, analysis figures, robustness screening |
| `Baselines/harness_adapter.py` | runs the PPO track's §02 baselines in the harness |

## Setup

Same environment as the PPO track (`Reward_and_Training/requirements.txt`). The simulator is not in the repository: clone `https://github.com/upatras-lar/Quadcopter_SimCon` and set `QUAD_SIM_DIR` to its `Simulation/` folder (or place the clone at `Reward_and_Training/Quadcopter_SimCon`). Run the commands from the repository root. Outputs go to `runs/` folders, which git ignores.

## Commands

    python MAP_Elites/map_elites.py --name ME2 --seed 0 --budget 5000 --bound t_climb=0,1.0
    python MAP_Elites/analysis.py --name ME2
    python MAP_Elites/robustness.py --name R1 --runs ME1 ME2 --top 10
    python Evaluation/final_comparison.py --name FINAL_PT --detector ppo_track --seed0 10000 --episodes 50 --conditions nominal ppo_track_nominal ppo_track_stress --controllers map_elites_final three_phase_default scripted_flip
    python Evaluation/render_episode.py --controller map_elites_final --condition nominal --seed 0

Each script documents its options in its docstring (`--help`).
