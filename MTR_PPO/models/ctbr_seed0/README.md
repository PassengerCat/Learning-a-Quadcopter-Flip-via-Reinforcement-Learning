# MTR-PPO, CTBR actions, seed 0

`final_model.zip` is the policy reported for MTR-PPO with CTBR actions (controller `ppo_mtr` in `Evaluation/final_comparison.py`). `config.json` is its run configuration, which `mtr_controller.MTRController` reads.

The model comes from run 13: a fresh network trained for 3M steps with every component of the method from the first step (dense kernels, reference starts with p = 0.5, randomised hover starts over the first half of training, termination on turning back by 30°, survival bonus 0.5/s weighted by 1/(1 + |v|²), command randomisation, pitch-axis term, one-off pitch-axis penalty at 40°).

The reported file is the **final model after 3M steps**, chosen by the same rule as the motors model (the final model of the run, no checkpoint selection). Known behaviour: from 1.8M steps on, the evaluation during training shows a final body rate of about 0.3 rad/s in the noise-free hover, a small oscillation that the earlier checkpoints (e.g. at 1.5M) do not have; it costs control effort but stays below the detector's 0.8 rad/s threshold.

Training command, curves and evaluation during training: `Logs/MTR_PPO/runs/run13_ctbr_seed0_scratch_all_components/`. The earlier CTBR models (runs 1–6 and 9) are documented in `Logs/MTR_PPO/runs/` and in `Documentation/CHANGELOG_EXPERIMENTS.md`.
