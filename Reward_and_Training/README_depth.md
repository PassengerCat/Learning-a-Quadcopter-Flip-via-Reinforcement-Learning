# Depth experiments — what they answer and how to use them in the report

One command runs everything (from `Simulation\`, with the `.venv` active):

```
run_all.bat
```

It can be stopped and restarted at any time: finished trainings are skipped (`--skip-if-done`). Everything lands in `results\depth\`. The budget is 11 trainings of 300k steps plus the analyses, roughly one night on a typical laptop (more cores means faster).

| # | Question | Script | Output folder |
|---|---|---|---|
| 1 | Is the result reproducible, or was it one lucky seed? | `train_ppo.py` × 3 seeds + `aggregate_runs.py` | `results\depth\aggregate\` |
| 2 | Which design choice actually matters? | 6 ablation runs + `aggregate_runs.py` | `results\depth\aggregate\` |
| 3 | Where does each controller break, and how gracefully? | `robustness_sweep.py` | `results\depth\robustness\` |
| 4 | How close is the learned flip to the time-optimal flip? | `optimal_flip_analysis.py` | `results\depth\optimal_flip\` |
| 5 | Is the wind drift really caused by missing integral action? | 2 runs (`wind_ctrl`, `wind_integral`) + `aggregate_runs.py` | `results\depth\aggregate\` |
| 6 | Why do the failures happen (mechanism, not just symptoms)? | `failure_analysis.py` | `results\depth\failures\` |

---

## 1 + 2. Seeds and ablations (`aggregate_runs.py`)

**Design (write this in the methods section).**
- **One component removed at a time.** Every ablation is the final configuration minus exactly one component, with the same budget (300k steps) and the same evaluation.
- **Evaluation of the final model.** We evaluate each run's **final** model, not the "best" checkpoint. The best checkpoint was chosen on the periodic-evaluation seeds, which would bias the comparison in its favour.
- **Fresh seeds.** The evaluation uses a **fresh seed bank (30000+)** that was never used for training, BC demonstrations or model selection.
- **Metric.** Success is judged only by the independent §04 detector.
- **Seeds.** The final configuration uses 3 training seeds. The ablations use 1 seed each (add `--seed 1 2` in `run_all.bat` if time allows).

**The rows.**

| Tag | Removes | Expected (from the reward log) |
|---|---|---|
| `bc_only` | PPO fine-tuning | flips in nominal conditions, weaker under stress (30% in our run) |
| `no_bc` | the behaviour-cloning warm start | no flip within the budget (runs 1–8) |
| `symmetric` | privileged information in the critic | unknown: this is a real test |
| `no_prog` | the "new record" reward term | with BC probably small; the term mattered for pure RL |
| `no_demo` | demonstration (mid-flip) starts | unknown |
| `motors` | the CTBR action space (pure RL, since the expert speaks CTBR) | no flip (run 1) |

`motors` changes two things at once (action space *and* no BC, because the expert only produces CTBR actions). Compare it with `no_bc`, not with `final`, and say so in the report.

**How to write it up.**
- **What counts as a difference.** A component matters if removing it changes a column by more than the seed-to-seed spread of `final` (its ± value). With 3 seeds, call differences below about 2 standard deviations *inconclusive*.
- **Negative results.** Report them, e.g. "the asymmetric critic made no measurable difference at this budget". That is what depth looks like.
- **Learning curves** (`learning_curves.png`) show sample efficiency. Point 0 is the policy right after BC, so you can see how much of the final performance comes from imitation and how much from RL.

## 3. Robustness curves (`robustness_sweep.py`)

The sweep covers actuator noise σ from 0 to 0.3 (gyro σ = 4/3 of it) and wind from 0 to 8 m/s. It runs the scripted flip, PPO and PPO + PID on the same seeds (40000+), with 95% Wilson intervals.

- **Report the breaking point, not one number.** For example: "the scripted flip drops below 50% at σ ≈ …, PPO stays above 90% up to σ ≈ …".
- **Look at the lower-left panel** (final ‖ω‖ against the 0.8 rad/s threshold). The PID-based controllers fail on *stillness*, not on the flip.
- **Wind:** the difference is mostly in *drift* (lower-right panel).

Preliminary result on our machine (10 episodes per point, `results\depth\robustness\`):

| | scripted | PPO | PPO + PID |
|---|---|---|---|
| noise: last level with ≥ 90% success | σ = 0.10 | **σ = 0.20** | σ = 0.10 |
| noise σ = 0.30 | 0% | 30% | 0% |
| wind: last level with ≥ 90% success | 8 m/s | 4 m/s | 8 m/s |
| drift at 8 m/s | 1.17 m | 6.14 m | **0.49 m** |

The two disturbances split the controllers cleanly. Final ‖ω‖ grows linearly with the noise level for all three, but with about half the slope for PPO. That is why PPO crosses the 0.8 rad/s stillness threshold only at σ ≈ 0.27, against ≈ 0.12 for the PID-based controllers. Under noise the hybrid's curve lies exactly on the scripted one, because after the flip both run the same PID.

## 4. Learned flip vs time-optimal flip (`optimal_flip_analysis.py`)

Already computed on the delivered model (`results\depth\optimal_flip\`):

| Flip | Rest-to-rest rotation [s] | 10°→350° [s] | Peak q [rad/s] |
|---|---|---|---|
| Theory: bang-bang, torque-limited (α_max = 238 rad/s²) | 0.325 | 0.248 | 38.6 |
| Theory: bang-bang, rate cap 20 rad/s (the policy's limit) | 0.398 | 0.322 | 20.0 |
| Simulator bang-bang (includes motor lag) | 0.365 | 0.255 | 35.0 |
| PPO policy | 0.640 | 0.300 | 21.9 |
| Scripted flip | 0.500 | 0.420 | 14.9 |

**What to say about it.**
- **Speed through the flip.** The policy is bang-bang *in rate*: it saturates its pitch-rate command and holds it. It gets through the inverted phase as fast as the rate-capped optimum.
- **The trade it makes.** It does not brake early. It overshoots ≈25° and corrects while upright, so a precise stop is traded for less time spent upside down.
- **Thrust.** Both controllers cut thrust to ≈0.2 × weight while inverted, as the optimal flip does. Nobody told the policy to.
- **The motor gap.** The motor dynamics cost the bang-bang ≈40 ms against the ideal (0.365 vs 0.325 s). This is a concrete number for the gap between the model and the simulator.
- **Why speed emerges without a time penalty.** The post-flip rewards are paid per second, and γ = 0.99 per decision discounts them.

The figures are `flip_phase_portrait.png` (q vs θ, the key picture) and `flip_thrust_vs_angle.png`.

## 5. Memory experiment (`wind_ctrl` vs `wind_integral`)

- **Hypothesis:** the policy drifts in steady wind because it is memoryless. Proportional-only feedback has a steady-state error, and a PID removes it with its integrator.
- **Test:** give the actor ∫(p − p₀) dt as an extra input (`--pos-integral`), train both arms with the same wind randomisation and reward, and compare *drift at 5 m/s* in `summary.md`.
- **Reading the outcome:**
  - *Drift drops clearly* → hypothesis supported. The trade-off with the PID hybrid disappears, and the policy stays noise-robust.
  - *No change* → the missing state was not the bottleneck at this budget. That is also a valid, reportable result; then say what you would try next (longer training, recurrent policy).
- **Preliminary run** (our 2-core machine, integral arm only, 300k steps; evaluated on 20 fresh episodes at 5 m/s):

  | Policy | success at 5 m/s | drift at 5 m/s | nominal success |
  |---|---|---|---|
  | delivered model (run-9, never saw wind) | 95% | 1.97 m | 100% |
  | wind fine-tune, no integral (runs 10–11, earlier) | 90% | 2.9 m | 100% |
  | wind training **with** integral input (this experiment) | 85% | 4.53 m | 100% |

  So far the hypothesis is **not** supported. Training in strong wind made the policy *worse* at holding position, with or without the integral input. During training the periodic evaluation at 5 m/s even dipped to 67% with crashes before recovering. Your run adds the proper control arm (`wind_ctrl`, identical settings without the integral), and only that comparison is conclusive.
- **Possible explanations to discuss if it stays negative:**
  - The integral input needs far more than 300k steps to be exploited.
  - The stronger position terms fight the demonstrations, since the expert does not use an integral.
  - Wind gusts make the value targets very noisy.

  A cleaner next step is a *residual* policy on top of a PI position loop.

## 6. Failure analysis (`failure_analysis.py`)

Already computed on the delivered model (`results\depth\failures\`):

- **A · Coning (run-4 policy, `models\ablation\run4_coning.zip`).**
  - The integrated pitch rate reached **661°**, while the gravity-based flip angle was **70°** and the tilt never exceeded **86°**.
  - This is the reward hack that made us measure the flip geometrically and require an explicit inversion in the detector.
- **B · Noise.** Hover only, same noise and seeds; mean ‖ω‖ and share of time below the 0.8 rad/s threshold:

  | Controller | mean ‖ω‖ | below 0.8 rad/s |
  |---|---|---|
  | §02 PID | 0.97 | 42% |
  | bare CTBR rate loop, zero command | 0.72 | 62% |
  | PPO policy | 0.43 | 94% |

  The flip is not what fails; the hover after it is. The rate loop alone explains part of the jitter, and the PID's outer loops add to it. The policy ends *below* the bare loop, so it actively counteracts the disturbance.
- **C · Wind (5 m/s).** Final drift: scripted 0.28 m, PPO 3.67 m, PPO + PID 0.05 m. This is the classical steady-state error of proportional control (see item 5).

---

### Suggested structure for the report's analysis section

1. **Reproducibility:** seeds table and learning curves.
2. **What matters:** ablation table.
3. **What was learned:** optimal-flip comparison and the thrust-cut observation.
4. **Robustness:** curves, breaking points and the stillness mechanism.
5. **Failure cases:** coning, noise and wind, each with its mechanism.
6. **Limitations and next steps:** the memory experiment result, and sim-only validation.
