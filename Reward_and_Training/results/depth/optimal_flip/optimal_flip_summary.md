# How close is the learned flip to the time-optimal flip?

Vehicle: τ_max = 2.93 N·m, I_yy = 0.0123 kg·m² → **α_max = 238 rad/s²**. Policy rate limit (CTBR): 20 rad/s.

| Flip | Rest-to-rest rotation [s] | 10°→350° [s] | Peak pitch rate [rad/s] |
|---|---|---|---|
| Theory: bang-bang, torque-limited | 0.325 | 0.248 | 38.6 |
| Theory: bang-bang, rate cap 20 rad/s | 0.398 | 0.322 | 20.0 |
| Simulator bang-bang (motor dynamics) | 0.365 | 0.255 | 35.0 |
| PPO policy (§03) | 0.640 | 0.300 | 21.9 |
| Scripted flip (§02) | 0.500 | 0.420 | 14.9 |

| Controller | Mean thrust/weight while inverted (135°–225°) |
|---|---|
| Scripted flip (§02) | 0.20 |
| PPO policy (§03) | 0.19 |

## Reading it

- **The benchmark.** The torque-limited optimum (0.32 s) needs a peak rate of 39 rad/s. The policy can only command 20 rad/s, so the fair benchmark is the rate-capped bang-bang: 0.40 s rest-to-rest, 0.32 s for 10°→350°. Flown open-loop in the simulator (motor lag included) the unconstrained bang-bang takes 0.37 s.
- **Getting there.** The PPO flip reaches 350° in 0.30 s. That is as fast as the rate-capped bound, and faster than the scripted flip (0.42 s). The policy saturates its pitch-rate command and holds it, which is bang-bang *in rate* (the plateau in flip_phase_portrait.png).
- **Stopping.** The optimal profile brakes so as to stop exactly at 360°. The policy does not brake early: it overshoots by ≈24° and then corrects, so its full rest-to-rest rotation takes 0.64 s, against 0.40 s for the bound. The learned solution trades a precise stop for speed through the inverted phase, which is the dangerous part (thrust points down, altitude is lost). The correction happens upright, where it is cheap.
- **Thrust.** Both controllers **cut collective thrust while inverted** (PPO 0.19, scripted 0.20 × weight), the second feature of the optimal flip. Thrust pointing down would accelerate the vehicle towards the ground. Nobody told the policy this; it follows from the reward (a crash costs −5 and altitude drift is penalised).
- **The scripted flip** is slower by design: its rotation rate (~15 rad/s) comes from a hand-tuned pulse and a coast phase, not from an optimisation.
- **Why is the policy fast at all?** The reward has no explicit time term. But the post-flip terms (b_upright, b_hold, the alive bonus) are paid **per second after the flip**, and γ = 0.99 per 0.02 s (a half-life of ≈1.4 s) discounts later rewards. Finishing earlier is worth more, so the policy learned to approach the physical limit.
