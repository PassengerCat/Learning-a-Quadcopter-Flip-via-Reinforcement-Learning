# Baseline results (50 episodes/controller, seeds 10000–10049, action-repeat 4, action noise σ=0.05, gyro noise σ=0.0)

| Controller | Success §04 (95% CI) | Success quick (95% CI) | Rotation [°] | Final tilt [°] | Final ‖ω‖ [rad/s] | Altitude loss [m] | Recovery time [s] | Control effort | Terminated |
|---|---|---|---|---|---|---|---|---|---|
| ppo_flip | 100% (93–100) | 100% (93–100) | 360 ± 1 | 1.4 ± 0.4 | 0.31 ± 0.07 | 0.08 ± 0.02 | 0.83 ± 0.09 | 0.271 ± 0.004 | 0% |

Success here is the provisional quick-check (min_rotation_deg=330.0, inverted_tilt_deg=150.0, final_tilt_deg=15.0, final_rate=1.0, max_alt_loss_m=3.0, settle_window_s=0.5); the §04 detector is authoritative.
