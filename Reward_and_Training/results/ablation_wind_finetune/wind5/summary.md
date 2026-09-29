# Baseline results (20 episodes/controller, seeds 10000–10019, action-repeat 4, action noise σ=0.0, gyro noise σ=0.0, wind 5.0 m/s)

| Controller | Success §04 (95% CI) | Success quick (95% CI) | Rotation [°] | Final tilt [°] | Final ‖ω‖ [rad/s] | Altitude loss [m] | Recovery time [s] | Control effort | Terminated |
|---|---|---|---|---|---|---|---|---|---|
| random | 0% (0–16) | 0% (0–16) | -50 ± 405 | 100.2 ± 22.8 | 12.12 ± 6.37 | 5.02 ± 0.01 | — | 1.366 ± 0.092 | 100% |
| pid_hover | 0% (0–16) | 0% (0–16) | -0 ± 4 | 3.0 ± 2.4 | 0.05 ± 0.02 | 0.01 ± 0.00 | — | 0.000 ± 0.000 | 0% |
| scripted_flip | 100% (84–100) | 100% (84–100) | 360 ± 4 | 3.0 ± 2.4 | 0.05 ± 0.02 | 0.00 ± 0.00 | 0.72 ± 0.59 | 0.211 ± 0.004 | 0% |
| ppo_flip | 90% (70–97) | 100% (84–100) | 359 ± 3 | 2.8 ± 2.2 | 0.10 ± 0.06 | 0.00 ± 0.00 | 0.50 ± 0.01 | 0.143 ± 0.001 | 0% |

Success here is the provisional quick-check (min_rotation_deg=330.0, inverted_tilt_deg=150.0, final_tilt_deg=15.0, final_rate=1.0, max_alt_loss_m=3.0, settle_window_s=0.5); the §04 detector is authoritative.
