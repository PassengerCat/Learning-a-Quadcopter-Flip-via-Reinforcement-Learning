# MTR-PPO, direct motor commands, seed 0

`final_model.zip` is the policy reported for MTR-PPO with direct motor commands (controller `ppo_mtr_motors` in `Evaluation/final_comparison.py`). `config.json` is its run configuration, which `mtr_controller.MTRController` reads.

The model is the final model of run 14: a fresh network trained for 3M steps with every component of the method from the first step (dense kernels, reference starts with p = 0.5, randomised hover starts over the first half of training, termination on turning back by 30°, survival bonus 0.5/s weighted by 1/(1 + |v|²), command randomisation, pitch-axis term, one-off pitch-axis penalty at 40°).

Training command, curves and evaluation during training: `Logs/MTR_PPO/runs/run14_motors_seed0_scratch_all_components/`. The earlier motor models (runs 7, 8 and 10–12, the latter three fine-tuning stages with the pitch-axis penalty) are documented in `Logs/MTR_PPO/runs/` and in `Documentation/CHANGELOG_EXPERIMENTS.md`.
