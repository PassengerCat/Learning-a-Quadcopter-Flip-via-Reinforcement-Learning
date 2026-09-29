# Baseline results (20 episodes/controller, seeds 10000–10019, action-repeat 4, action noise σ=0.05, gyro noise σ=0.0, wind 0.0 m/s)

| Controller | Success §04 (95% CI) | Success quick (95% CI) | Rotation [°] | Final tilt [°] | Final ‖ω‖ [rad/s] | Altitude loss [m] | Horizontal drift [m] | Recovery time [s] | Control effort | Terminated |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0% (0–16) | 0% (0–16) | -53 ± 415 | 97.5 ± 19.4 | 12.21 ± 6.34 | 5.02 ± 0.01 | 2.01 ± 1.27 | — | 1.372 ± 0.095 | 100% |
| pid_hover | 0% (0–16) | 0% (0–16) | 0 ± 1 | 1.5 ± 0.4 | 0.31 ± 0.08 | 0.02 ± 0.00 | 0.03 ± 0.01 | — | 0.013 ± 0.001 | 0% |
| scripted_flip | 100% (84–100) | 100% (84–100) | 360 ± 1 | 1.6 ± 0.4 | 0.32 ± 0.08 | 0.00 ± 0.00 | 0.18 ± 0.08 | 0.39 ± 0.06 | 0.216 ± 0.004 | 0% |
| ppo_flip | 100% (84–100) | 100% (84–100) | 360 ± 1 | 0.6 ± 0.1 | 0.16 ± 0.02 | 0.00 ± 0.00 | 0.31 ± 0.04 | 0.51 ± 0.01 | 0.207 ± 0.004 | 0% |
| ppo_flip+pid | 100% (84–100) | 100% (84–100) | 360 ± 1 | 1.6 ± 0.4 | 0.32 ± 0.08 | 0.02 ± 0.01 | 0.03 ± 0.01 | 0.97 ± 0.02 | 0.245 ± 0.009 | 0% |

Success here is the provisional quick-check (min_rotation_deg=330.0, inverted_tilt_deg=150.0, final_tilt_deg=15.0, final_rate=1.0, max_alt_loss_m=3.0, settle_window_s=0.5); the §04 detector is authoritative.
