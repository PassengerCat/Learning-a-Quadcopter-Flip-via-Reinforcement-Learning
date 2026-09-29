# Baseline results (20 episodes/controller, seeds 10000–10019, action-repeat 4, action noise σ=0.15, gyro noise σ=0.2)

| Controller | Success §04 (95% CI) | Success quick (95% CI) | Rotation [°] | Final tilt [°] | Final ‖ω‖ [rad/s] | Altitude loss [m] | Recovery time [s] | Control effort | Terminated |
|---|---|---|---|---|---|---|---|---|---|
| ppo_flip | 30% (15–52) | 45% (26–66) | 361 ± 4 | 4.5 ± 1.4 | 0.98 ± 0.22 | 0.13 ± 0.04 | 3.12 ± 1.53 | 0.381 ± 0.014 | 0% |

Success here is the provisional quick-check (min_rotation_deg=330.0, inverted_tilt_deg=150.0, final_tilt_deg=15.0, final_rate=1.0, max_alt_loss_m=3.0, settle_window_s=0.5); the §04 detector is authoritative.
