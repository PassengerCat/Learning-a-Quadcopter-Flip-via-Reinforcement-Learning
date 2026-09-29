# Baseline results (20 episodes/controller, seeds 10000–10019, action-repeat 4, action noise σ=0.15, gyro noise σ=0.2, wind 0.0 m/s)

| Controller | Success §04 (95% CI) | Success quick (95% CI) | Rotation [°] | Final tilt [°] | Final ‖ω‖ [rad/s] | Altitude loss [m] | Horizontal drift [m] | Recovery time [s] | Control effort | Terminated |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0% (0–16) | 0% (0–16) | -41 ± 425 | 94.1 ± 17.8 | 12.74 ± 6.71 | 5.02 ± 0.01 | 2.27 ± 1.11 | — | 1.410 ± 0.100 | 100% |
| pid_hover | 0% (0–16) | 0% (0–16) | 1 ± 3 | 4.7 ± 1.5 | 0.95 ± 0.22 | 0.05 ± 0.01 | 0.09 ± 0.05 | — | 0.198 ± 0.006 | 0% |
| scripted_flip | 20% (8–42) | 55% (34–74) | 361 ± 3 | 4.7 ± 1.5 | 0.95 ± 0.22 | 0.01 ± 0.01 | 0.35 ± 0.15 | 3.09 ± 2.09 | 0.389 ± 0.011 | 0% |
| ppo_flip | 100% (84–100) | 100% (84–100) | 360 ± 2 | 2.0 ± 0.7 | 0.42 ± 0.06 | 0.00 ± 0.00 | 0.41 ± 0.13 | 0.52 ± 0.03 | 0.352 ± 0.009 | 0% |
| ppo_flip+pid | 20% (8–42) | 55% (34–74) | 360 ± 3 | 4.7 ± 1.5 | 0.95 ± 0.22 | 0.05 ± 0.01 | 0.09 ± 0.05 | 3.29 ± 1.82 | 0.406 ± 0.012 | 0% |

Success here is the provisional quick-check (min_rotation_deg=330.0, inverted_tilt_deg=150.0, final_tilt_deg=15.0, final_rate=1.0, max_alt_loss_m=3.0, settle_window_s=0.5); the §04 detector is authoritative.
