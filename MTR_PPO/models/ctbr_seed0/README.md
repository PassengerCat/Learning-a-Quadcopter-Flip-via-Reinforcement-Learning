# MTR-PPO, CTBR actions, seed 0

`final_model.zip` is the policy reported for MTR-PPO (controller `ppo_mtr` in `Evaluation/final_comparison.py`). `config.json` is its run configuration, which `mtr_controller.MTRController` reads.

The model was trained in three stages. Each stage continued from the previous one with `--init-model`:

| stage | run | steps | added | from |
|---|---|---|---|---|
| 1 | run 4 | 1.5M | dense kernels, reference starts (p = 0.5), termination on turning back (30°) | fresh network |
| 2 | run 5 | 1.0M | post-flip survival bonus 0.5/s | stage 1, checkpoint at 1.5M |
| 3 | run 6 | 1.0M | survival bonus weighted by 1/(1 + \|v\|²) | stage 2, final model |

That is 3.5M environment steps in total. Each stage's command, training curves and evaluation are in `Logs/MTR_PPO/runs/` (runs 4–6).
