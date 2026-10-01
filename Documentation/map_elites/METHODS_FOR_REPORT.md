# Methods for the report: evaluation and MAP-Elites (verified)

This page lists the methods used so far for the comparison part of the project, how each one was checked, and where the evidence is. It is a source for the report's *Methods* and *Experiments* sections. Every result below was produced **without any training**.

## 1. Universal evaluation harness (`Evaluation/`)

- **Benchmark conditions (brief):**
  - `QuadcopterVelocityEnv` with full-state observation, no wind, nominal hover start and a 10 s horizon.
  - **Observation noise explicitly zero.** Passing `observation_noise_std=None` keeps the simulator's default noise.
  - Tilt termination disabled; |z| ≤ 5 m and numerical failure still end an episode.
- **Interface:** any controller with `reset(seed)` and `act(obs) -> (action, info)`. It is called every 5 ms simulator step; slower controllers hold their own action.
- **Judge:** a geometric detector decides success; rewards are never used. Every episode is judged by both detectors (section 14); the reported numbers use the PPO track's detector.
- **Per-episode metrics:**

  | Metric | Definition |
  |---|---|
  | Final tilt | Tilt at the last step |
  | Final \|ω\| | Body-rate magnitude at the last step |
  | Maximum altitude loss | Largest drop below the start altitude |
  | Flip time | First time rev ≥ 360° − 0.15 rad |
  | Recovery time | Start of the final 0.4 s hold |
  | Control effort | ∫ Σᵢ (aᵢ − a_hover)² dt |

- **Per-controller summary:**
  - success rate with a **95 % Wilson score interval**, which stays meaningful at 0/n and n/n;
  - mean, min and max of each metric;
  - failure modes, each episode classified by its first failed criterion: crash/region, over-rotation, never inverted, rotation incomplete, or no final hold.
- **Stress conditions:** a `Condition` switches on one disturbance at a time; the default is nominal. Observation noise uses the simulator's own default std per group (scale 1). It affects only what the controller sees: an open-loop controller is unaffected, and the verdict uses the true state. Each episode's noise is fixed by its seed. Log: `Logs/evaluation_harness/step5_obs_noise_checks_log.txt`.
- **Wind:** steady horizontal wind (1 m/s) from a direction drawn per episode seed, built with the simulator's own wind model. Optionally, the simulator's sinusoidal gusts are added. It acts through the simulator's per-axis quadratic drag. Log: `Logs/evaluation_harness/step6_wind_checks_log.txt`.
- **Perturbed start:** tilt ≤ 5° about a random horizontal axis, and body rates and velocity ≤ 0.2 per axis, drawn per episode seed. The simulator's integrator restarts from that state. Log: `Logs/evaluation_harness/step7_perturbed_start_checks_log.txt`.
- **Verification:** every metric recomputes exactly from the recorded trajectory; results are identical for identical seeds; a hand-computed control effort matches (motors off: 4·(1 + a_hover)²·t). Logs: `Logs/evaluation_harness/`.

## 2. Success detector (§04) changes

- **Inversion threshold 170°** (was 120°): gz ≤ cos 170°.
- **Why not exactly 180°:** with 5 ms samples the vehicle turns 4–9° per step, so exact inversion is never sampled. In the genome study, every successful flip reached a sampled tilt of at least 175.7°.
- **Inversion checked along the arc between samples** (slerp, 8 sub-steps). The verdict is then independent of the sampling rate: at 50 Hz the default flip is sampled at only 169.6°.
- **Verification:**
  - a synthetic flip reaching only 165° is rejected;
  - the scripted flips pass at both 200 Hz and 50 Hz;
  - all 101 earlier successes are kept.

  Log: `Logs/success_detector/`.

## 3. Controller parameterisation (MAP-Elites genome)

- **Controller:** a three-phase controller (`Controllers/three_phase_flip.py`) with Lupashin-style phases:
  1. climb,
  2. braked pitch rotation, with q_des = min(q_peak, √(2·a_brake·(2π − rev))),
  3. geometric thrust-vector recovery.
- **Inputs:** only the 20-value observation, plus fixed vehicle constants.
- **Genome:** 11 parameters.
- **Verification:** its pitch increment equals the detector's to 4·10⁻¹⁶, and it succeeds in 10/10 episodes in the harness. Log: `Logs/map_elites/`.

## 4. Exploratory genome sampling (choosing descriptor ranges)

- **Latin hypercube sampling (LHS), 400 samples, seed 0.** Each parameter's range is split into 400 equal strata and each stratum is used exactly once, which covers every parameter evenly. Each sample is run for one nominal 5 s episode.
- **One-at-a-time sweeps:** 11 parameters × 9 values around the default, to see which parameter moves which descriptor.
- **Robust ranges:** the 1st–99th percentiles of the successful samples.
- **Uncertainty of the ranges:** a bootstrap with 1000 resamples with replacement, reporting the mean ± standard deviation of p1 and p99. A convergence check compares the first 200 samples with all 400.
- **Independence of the descriptors:** Spearman rank correlation, which is unit-free and robust to outliers.
- **Results:**
  - 101/400 successes;
  - duration 0.24–0.75 s, climb 0.03–3.67 m;
  - ρ = +0.19 (p = 0.06), so the descriptors are nearly independent;
  - duration is driven by q_peak and a_brake, climb by t_climb and F_climb; the recovery gains move neither descriptor.

  Data, analysis and figure: `Logs/map_elites/descriptor_sampling/`.

## 5. MAP-Elites problem definition (`MAP_Elites/problem.py`)

- **Validity:** the §04 verdict on a nominal 5 s search episode. Elites are re-checked at 10 s later. Invalid genomes get no descriptors and no fitness and never enter the archive. This closes the **crash loophole**, where a flip that crashes shares a cell with a stable flip, and the **do-nothing loophole**, where hovering never qualifies.
- **Descriptors:**
  - rotation duration = t(rev ≥ 351.4°) − t(rev ≥ 10°), range 0.20–0.80 s;
  - maximum climb = max(z₀ − z), range 0–3 m;
  - a 10 × 10 grid, with out-of-range values placed in the edge cells.
- **Fitness:** f = −(control effort + 1.0 · maximum altitude loss).
- **Genome box:** as in the sampling study, except a_brake is 20–150 and Kd_att is 6–25. No sampled success lay outside these ranges. A run may override single limits (`--bound NAME=LOW,HIGH`); the full box is stored in the run's `config.json`. With the default box the code reproduces ME1 exactly (`Logs/map_elites/bounds_override_checks_log.txt`).
- **Known open risk:** "barely valid" flips that sit just above the detector's thresholds. The plan is to re-check the final elites under small perturbations.
- **Verification:**
  - descriptors reproduce the sampling study exactly;
  - the default, slow and fast flips are valid, with durations 0.345, 0.460 and 0.265 s;
  - a flip that then crashes has the same descriptors as the default flip but is invalid;
  - hover and over-rotation are invalid.

  Log: `Logs/map_elites/m2_problem_checks_log.txt`.

## 6. MAP-Elites archive (`MAP_Elites/archive.py`)

- **Grid:** 10 × 10 over the two descriptors. A cell is [lo + k·w, lo + (k+1)·w); out-of-range values go to the edge cells.
- **Cell edges:** values exactly on an edge belong to the upper cell, with a 1e−9 floating-point guard. This matters because durations are exact multiples of 5 ms, and several of them coincide with cell edges.
- **One elite per cell.** A valid candidate enters an empty cell or replaces the incumbent only if its fitness is strictly higher, so ties keep the incumbent. Invalid candidates never enter.
- **Summary statistics:**
  - coverage: filled cells / 100;
  - best and mean fitness;
  - QD-score = Σ over filled cells of (f − f_floor), with f_floor = −30 below the worst possible fitness, about −27.2. Every elite therefore adds a positive amount, and the score rises with both coverage and quality.
- **Persistence:** the archive is saved to and loaded from JSON. Loading verifies that each elite's stored cell matches its descriptors.
- **Verification:**
  - cell mapping is correct on edges, centres and out-of-range values;
  - the insertion rule behaves as specified (new, worse, tie, better, invalid);
  - the QD-score is monotone over 500 random insertions;
  - save and load give identical cells, counts and statistics;
  - the real default flip enters cell (2, 2).

  Log: `Logs/map_elites/m3_archive_checks_log.txt`.

## 7. MAP-Elites loop (`MAP_Elites/map_elites.py`)

- **Initialisation:** the hand-tuned default genome plus 100 Latin-hypercube genomes.
- **Iteration:**
  - choose 20 parents uniformly at random among the elites;
  - Gaussian mutation with σ = 10 % of each gene's range, clipped to the genome box;
  - evaluate the children in parallel and insert them in submission order.
- **Budget:** 5000 evaluations in total, counting the initial ones.
- **Reproducibility:** one random generator lives in the main process, so a given seed yields the same run for any number of workers.
- **Checkpoints and logging:**
  - every 500 evaluations and at the end, the run saves the archive, the generator state and the history;
  - a stopped run resumes exactly;
  - coverage, best and mean fitness and QD-score are logged per generation for the convergence plots.
- **Verification:**
  - 18,000 mutations all stay in the box, and the step spread matches σ;
  - serial and parallel runs give identical archives and histories;
  - a different seed gives a different run;
  - evaluations are fully accounted for;
  - only §04 successes are stored;
  - a run killed after a checkpoint and resumed matches the uninterrupted run exactly;
  - the command line works under the "spawn" start method used on Windows.

  Log: `Logs/map_elites/m4_loop_checks_log.txt`.

## 8. Analysis figures (`MAP_Elites/analysis.py`)

- **Archive heatmap:** the fitness of each cell on a one-hue sequential scale, where darker means better. Cells where no valid flip was found are neutral grey, and the default controller is marked with a star.
- **Convergence:** two panels, each with its own single axis: cells filled, and QD-score, both against evaluations.
- **Verification:**
  - pixel checks confirm the orientation (x = duration, y = climb), that darker means higher fitness, and that empty cells are grey;
  - the default marker lands in its cell;
  - an empty archive renders;
  - the convergence lines plot the history exactly;
  - `load_run` returns exactly the saved archive, history and config.

  Log: `Logs/map_elites/m5a_analysis_checks_log.txt`.

## 9. Isolated genome-box experiments (ME2, ME3)

- **Design:** one limit changed per run, with everything else identical to ME1 (seed 0, budget 5000). The full genome box is stored in each run's `config.json` (`--bound NAME=LOW,HIGH`). With the default box, the code reproduces ME1 exactly.
- **ME2 (t_climb ≤ 1.0 s instead of 0.6 s):**
  - better in 67/100 cells (Wilcoxon signed-rank, p = 6·10⁻⁷);
  - best fitness −0.702 against −0.758;
  - the limit is still active: 25 elites sit at 1.0 s.
- **ME3 (a_brake ≥ 10 instead of 20):** no effect (53 cells better, 46 worse, p = 0.82). The old limit was not binding.
- **Comparison points:** at 2001 evaluations ME2 is not yet better per cell. The widened box needs more budget, so both runs were compared at 5000.

  Logs: `Logs/map_elites/ME2_ME3_analysis/`.

## 10. Search-seed replicates

- **Design:** ME1 and ME2 repeated with search seeds 1 and 2, so each configuration has 3 runs.
- **Result:** every ME2 run is better than every ME1 run on best fitness, median cell fitness and QD-score. With 3 vs 3 runs this is the strongest possible result: one-sided Mann-Whitney p = 0.05.
  - best fitness −0.699 ± 0.006 against −0.752 ± 0.006;
  - median cell fitness −0.904 ± 0.021 against −1.021 ± 0.011.
- **Where the gain is:** it grows with climb height, from +0.01 in the lowest climb row to +0.29 in the highest.

  Figure and log: `Logs/map_elites/seed_replicates/`.

## 11. Stress conditions and the one-turn rule

- **Harness conditions:** `Condition` switches one disturbance at a time:
  - observation noise, at the simulator's defaults;
  - wind, through the simulator's own wind model, with the direction seeded per episode;
  - a perturbed start: tilt ≤ 5° and rates and velocity ≤ 0.2;
  - the PPO track's controller-side noise: motor 0.05 / 0.15 and gyro 0.2 rad/s, one sample per 20 ms, from their exact generator.

  Nominal results are unchanged by every addition.
- **§04 one-turn rule:** the 420° |pitch|-travel budget now counts only the maneuver, i.e. until the first upright sample after the completed revolution. Under noise, post-flip hover jitter (about 2° RMS) had summed to 100–300° over 8 s and failed every flip. A second turn is still caught by the maximum-revolution limit (≤ 390°) and by the final one-turn check.

  Logs: `Logs/evaluation_harness/step5`–`step8`, `Logs/success_detector/travel_window_checks_log.txt`.

## 12. Robustness screening and final selection

- **Screening R1:** 71 candidates (the default + the top-10 elites of 7 runs), all on the same seeds.
  - the mild conditions saturate: 63/71 reach 30/30;
  - every failure is over-rotation;
  - all 5 leaders reach 150/150 over 50 seeds.
- **Selection rule (agreed):** the MAP-Elites controller is optimised offline for the nominal task, so stress results are only logged. The final controller is the elite with the best nominal fitness: **ME2_s1 cell (4,5)**.
  - fitness −0.692, against −1.117 for the default (−38 % cost);
  - maximum revolution 364.9°;
  - no altitude loss.
- **Trade-off:** it is slow. It climbs for 1 s, completes the turn at 1.5 s and settles at 2.3 s; the default completes at 0.6 s and settles at 0.84 s.

  Files: `Controllers/map_elites_final.json`, `map_elites_controller.py`. Logs: `Logs/map_elites/R1_robustness/`.

## 13. Baselines and the final comparison

- **Baselines:** the §02 baselines of the PPO track (random, PID hover, scripted flip) run in the same harness through a decision-rate adapter that holds each action for 4 steps. Their rollouts are reproduced **bit-exactly** over 100 decisions.
- **Final comparison:** `Evaluation/final_comparison.py` runs all controllers on the same seeds with the same detector (FINAL: §04):
  - nominal (primary);
  - motor-noise 0.05 and stress, logged only.

  Log: `Logs/evaluation_harness/step9`, `step10`.

## 14. Success definition for the reported numbers

- **Decision (user):** all controllers are judged with the PPO track's detector, so that MAP-Elites, the scripted flip and PPO share one definition. The PPO track's own measurements are not re-run.
- **Her criterion:** geometric revolution ≥ 360° − 8.6°, tilt ≥ 150° at some 20 ms update, then upright (< 15°) and slow (|v| < 0.4 m/s, |ω| < 0.8 rad/s) for 20 updates (0.4 s); latched. The §04 verdict is kept next to it for every episode.
- **Conditions:** her "nominal" and "stress" keep the simulator's default sensor noise on (her environment passes `observation_noise_std=None`), so `ppo_track_nominal` (motor 0.05) and `ppo_track_stress` (motor 0.15 + gyro 0.2 rad/s) include it.
- **Verification:** `Success_Detector/ppo_track_detector.py` is a verbatim copy (text and behaviour identical). With it, the harness reproduces her published scripted-flip results on her seeds 10000–10019: 20 % under stress, 100 % under motor noise 0.05, with matching rotation, final tilt and |ω|.

  Logs: `Logs/success_detector/ppo_track_copy_checks_log.txt`, `Logs/evaluation_harness/step12`, `step13`.
- **Result FINAL_PT** (50 episodes per cell, seeds 10000–10049; `Logs/final_comparison/FINAL_PT/`). Parity inside the run: the scripted flip scores 4/20 and 20/20 on her seeds 10000–10019, as in her tables.

  | Condition | MAP-Elites | Default three-phase | Scripted flip |
  |---|---|---|---|
  | nominal (brief) | 50/50 | 50/50 | 50/50 |
  | PPO-track nominal | 50/50 | 50/50 | 50/50 |
  | PPO-track stress | 15/50 | 14/50 | 17/50 |

  The §04 verdict is 0/50 for all three under stress, because none holds still at the end.
