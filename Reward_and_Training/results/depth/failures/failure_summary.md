# Failure analysis

## A. Reward hacking by coning (run-4)

- Integrated pitch rate after 4 s: **661°**, i.e. "almost two flips" by the run-4 reward.
- Gravity-based flip angle: **70°**. Maximum tilt: **86°**. The vehicle never went upside down.
- Mean yaw rate -1.26 rad/s. The vehicle is tilted and spinning about the vertical (a cone). In the body frame this makes q oscillate with a non-zero mean, so ∫q dt grows without bound. The body-y quaternion increment behaves the same way.
- **Consequence:** the flip is now measured from the gravity direction, θ = atan2(−g_x, g_z), which only reaches 360° if gravity sweeps once around the body x-z plane. The §04 detector also requires an explicit inversion (tilt ≥ 150°). The coning policy scores 0 on both.

## B. Stress test: why the scripted flip fails (and PPO does not)

Actuator σ = 0.15, gyro σ = 0.2 rad/s. Seed 10000: scripted FAIL, PPO success. The scripted flip does complete the rotation (≈360°); what it misses is the stillness condition afterwards (upper plot).

Control experiment (hover only, same noise, mean ‖ω‖ for t ≥ 3 s, and % of decisions below 0.8 rad/s):

| Controller | mean ‖ω‖ [rad/s] | time below 0.8 rad/s |
|---|---|---|
| PID hover (§02) | 0.97 ± 0.10 | 42% |
| CTBR loop, zero command | 0.72 ± 0.05 | 62% |
| PPO policy (§03) | 0.43 ± 0.04 | 94% |

- **The flip is not the problem; the hover after it is.** Even without any flip, the §02 PID hover sits at ‖ω‖ ≈ 0.97 rad/s and is below the 0.8 rad/s threshold only 42% of the time. The detector needs 0.4 s in a row, which rarely happens.
- **Where the jitter comes from.** The bare rate loop with a zero command already reaches ≈ 0.72 rad/s. The noisy gyro enters the P rate loop directly, and actuator noise excites the airframe. The PID's outer loops (attitude, velocity and position, with integrators) add to it (0.72 → 0.97).
- **The PPO policy goes below the bare loop** (≈ 0.43 rad/s, 94% of the time under the threshold). It uses the same rate loop, so its rate *commands* must actively cancel part of the disturbance, rather than simply being low-gain. It was trained under the env's sensor noise, with a reward that pays for being still after the flip, so it learned exactly this. A PID tuned for nominal conditions was never asked to.

## C. Steady wind (5 m/s): drift of the memoryless policy

| Controller | §04 | final drift [m] |
|---|---|---|
| Scripted flip (§02) | success | 0.28 |
| PPO policy (§03) | success | 3.67 |
| PPO + PID hold | success | 0.05 |

- **Why PPO drifts.** A constant force needs a constant counter-lean. A controller that only sees the current state (position error, velocity) settles where the lean it commands balances the wind, which is a non-zero position error. This is the classical steady-state error of proportional control.
- **Why the PID does not.** It removes that error with its integrator.
- **Two ways out.** The hybrid (PID after the flip). Or giving the policy the integral as an input: depth experiment 5 (`--pos-integral`), whose result is in `results/depth/aggregate/`.
