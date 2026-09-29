# Aggregated results (20 episodes per run and condition, fresh seeds 30000+, final models, §04 detector)

Mean ± std over training seeds (n = number of seeds). Conditions: nominal = actuator σ 0.05; stress = actuator σ 0.15 + gyro σ 0.2 rad/s; wind5 = 5 m/s median wind.

| Configuration | n | Success nominal | Success stress | Success wind 5 | Drift wind 5 [m] | Recovery nominal [s] | Effort nominal |
|---|---|---|---|---|---|---|---|
| Final configuration | 3 | 100% ± 0 | 100% ± 0 | 88% ± 8 | 2.63 ± 0.12 | 0.50 ± 0.03 | 0.187 ± 0.011 |
| − PPO (BC only) | 1 | 100% | 25% | 90% | 0.31 | 0.84 | 0.277 |
| − BC (pure RL) | 1 | 10% | 15% | 5% | 14.78 | 0.53 | 0.168 |
| − asymmetric critic | 1 | 100% | 60% | 35% | 10.06 | 0.50 | 0.223 |
| − 'new record' term | 1 | 100% | 100% | 75% | 4.19 | 0.52 | 0.178 |
| − demonstration starts | 1 | 100% | 100% | 75% | 3.08 | 0.51 | 0.190 |
| − CTBR (motor commands) | 1 | 0% | 0% | 0% | 17.17 | 0.66 | 0.111 |
| Wind training (control) | 1 | 100% | 100% | 95% | 2.23 | 0.47 | 0.169 |
| Wind training + position integral | 1 | 100% | 100% | 85% | 6.09 | 0.35 | 0.160 |

What each row changes (relative to the final configuration):

- **Final configuration** (`final`): BC warm start + PPO, CTBR, asymmetric critic, all reward terms
- **− PPO (BC only)** (`bc_only`): imitation of the expert, no RL fine-tuning
- **− BC (pure RL)** (`no_bc`): PPO from scratch, same reward and action space
- **− asymmetric critic** (`symmetric`): critic sees only what the actor sees
- **− 'new record' term** (`no_prog`): w_prog = 0
- **− demonstration starts** (`no_demo`): every training episode starts at hover
- **− CTBR (motor commands)** (`motors`): policy outputs the 4 motor commands; pure RL (the expert speaks CTBR)
- **Wind training (control)** (`wind_ctrl`): 70% windy episodes, stronger position terms
- **Wind training + position integral** (`wind_integral`): same, plus ∫(p − p0) dt in the actor's input

Per-seed values: `per_run.csv`. Learning curves: `learning_curves.png`. Bars: `ablation_bars.png`.

How to read it: a component matters if removing it changes the result by more than the seed-to-seed spread (the ± column). With 3 seeds, treat differences smaller than ~2 std as inconclusive, and say so.
