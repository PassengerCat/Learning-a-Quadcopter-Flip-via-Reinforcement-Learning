# Baseline results (50 episodes/controller, seeds 10000–10049, action-repeat 4, action noise σ=0.05, gyro noise σ=0.0, wind 0.0 m/s)

| Controller | Success §04 (95% CI) | Success quick (95% CI) | Rotation [°] | Final tilt [°] | Final ‖ω‖ [rad/s] | Altitude loss [m] | Horizontal drift [m] | Recovery time [s] | Control effort | Terminated |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0% (0–7) | 0% (0–7) | -48 ± 406 | 94.4 ± 18.2 | 11.52 ± 5.45 | 5.02 ± 0.01 | 2.03 ± 1.14 | — | 1.346 ± 0.092 | 100% |
| pid_hover | 0% (0–7) | 0% (0–7) | -0 ± 1 | 1.6 ± 0.4 | 0.31 ± 0.07 | 0.02 ± 0.00 | 0.03 ± 0.01 | — | 0.013 ± 0.000 | 0% |
| scripted_flip | 100% (93–100) | 100% (93–100) | 360 ± 1 | 1.6 ± 0.4 | 0.31 ± 0.08 | 0.01 ± 0.00 | 0.17 ± 0.08 | 0.40 ± 0.06 | 0.216 ± 0.004 | 0% |
| ppo_flip | 100% (93–100) | 100% (93–100) | 360 ± 1 | 0.6 ± 0.2 | 0.16 ± 0.02 | 0.00 ± 0.00 | 0.32 ± 0.05 | 0.51 ± 0.01 | 0.208 ± 0.005 | 0% |

Success here is the provisional quick-check (min_rotation_deg=330.0, inverted_tilt_deg=150.0, final_tilt_deg=15.0, final_rate=1.0, max_alt_loss_m=3.0, settle_window_s=0.5); the §04 detector is authoritative.
