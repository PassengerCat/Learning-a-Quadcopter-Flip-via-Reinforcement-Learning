# Changelog: evaluation harness, success detectors and MAP-Elites

Every change, check and result of the evaluation harness (`Evaluation/`), the success detectors (`Success_Detector/`), the controllers (`Controllers/`) and MAP-Elites (`MAP_Elites/`), in order. Planned experiments, executed checks and results are marked as such. The PPO track keeps its own log in `Reward_and_Training/reward_log.md`.

## 2026-09-27 — Evaluation harness step 1

**Evaluation harness, step 1 (executed; no training):** added `Evaluation/evaluate.py` (≈100 lines).

- It runs any controller with the course interface (`reset(seed)`, `act(obs) -> (action, info)`) on the benchmark `QuadcopterVelocityEnv`: full observation, no wind, observation noise explicitly zero, tilt termination disabled, |z| ≤ 5 m, 10 s.
- The controller is called every 5 ms simulator step. Each episode is judged only by the independent §04 detector, with the 0.4 s dwell as 80 steps.
- **Note:** `observation_noise_std=None`, as written in the brief's example, keeps the simulator's default noise on. Zeros are required.
- Checks, run outside the project ([log](../Logs/evaluation_harness/step1_checks_log.txt)):
  - observation equals the true state (max error 4e−11);
  - constant hover survives 10 s and fails with "never inverted";
  - motors off crashes at 1.12 s;
  - a random controller crashes;
  - identical seeds give identical results;
  - invalid actions are rejected;
  - about 1.9 s wall-clock per 10 s episode.
- Next steps: episode metrics, then a results file/table, then an observation-only scripted controller.

## 2026-09-27 — Evaluation harness step 2: per-episode metrics (executed; no training)

`Evaluation/evaluate.py` now also reports, per episode, the brief's evaluation metrics. The success verdict is unchanged.

- final tilt (degrees);
- final |ω| (rad/s);
- maximum altitude loss below the start altitude (metres);
- flip time, the first time the §04 rotation reaches 360° − 0.15 rad;
- recovery time, the start of the final 0.4 s hold (successes only);
- control effort, ∫ Σᵢ(aᵢ − a_hover)² dt over the clipped actions.

Checks, run outside the project ([log](../Logs/evaluation_harness/step2_metrics_checks_log.txt)):

- Constant hover gives all metrics 0 and no flip time.
- Motors off: effort 4.9777 matches the hand value 4·(1+0.0541)²·1.12 s, and altitude loss is 5.01 m at termination.
- The scripted flip, wrapped outside the project, passes the detector at 200 Hz: flip at 0.605 s, recovered at 0.835 s, effort 1.117, no altitude loss.
- Random verdicts are identical to step 1.

## 2026-09-27 — Evaluation harness step 3: summaries, result files, comparison table (executed; no training)

Added `Evaluation/report.py` (≈100 lines). It works on the harness's episode results and never re-runs or re-judges an episode.

- **Per controller:**
  - success count, rate and 95% Wilson interval, which stays meaningful for 0/n and n/n;
  - mean, min and max of each metric (flip and recovery times over the episodes that have one);
  - failure-mode counts.
- **Failure modes:** each episode is classified by its first failed criterion: crash/region, over-rotation, never inverted, rotation incomplete, or no final hold.
- **Output files:** `episodes.csv` (one row per episode) and `summary.json`, plus a plain-text comparison table.

Checks, run outside the project ([log](../Logs/evaluation_harness/step3_report_checks_log.txt)):

- Wilson intervals: 0/3 → [0, 0.5615], 3/3 → [0.4385, 1].
- For random, constant hover, motors off and the check-only scripted flip wrapper, the CSV, JSON and table agree on success counts, mean effort and failure-mode totals.
- Constant hover is classified "never inverted"; random and motors off as "crash/region"; the scripted flip scores 3/3.

## 2026-09-27 — MAP-Elites M1: parameterized three-phase flip controller (executed; no training)

**Design decisions (user):**

- The MAP-Elites genome is option A: the parameters of a three-phase controller (climb, braked rotation, geometric recovery), not neural-network weights.
- The candidate behaviour descriptors are flip time and maximum altitude deviation.
- **Risk to handle in M2:** a "crash loophole", where MAP-Elites could treat a flip that crashes like a stable flip. The fitness and archive must separate crashed from stably recovered solutions.
- Test policy: every new file gets tests that run outside the project and are deleted once they pass. Only their logs are kept.

**Added `Controllers/three_phase_flip.py`:**

- `ThreePhaseFlip` implements the course interface and reads only the 20-value observation. It also uses fixed vehicle constants (mass, inertia, mixer, motor limits) read once from the simulator's parameter set.
- It integrates pitch rotation from the observed quaternions, decides at 50 Hz and holds its action in between.
- `FlipParams` (11 values) converts to and from a vector for MAP-Elites.

**Checks, run outside the project ([log](../Logs/map_elites/m1_controller_checks_log.txt)):**

- The pitch increment matches the §04 detector's implementation to 4e−16 on 20,000 random attitude pairs. A sign error in the first draft of the quaternion product was fixed before this check ran; the check guards against it.
- The controller source contains no simulator access.
- Default parameters succeed in 10/10 episodes in the harness. Metrics are identical to the privileged check wrapper: flip at 0.605 s, recovery at 0.835 s, effort 1.1172.
- Back-to-back episodes are identical, so reset works.
- The parameter vector round-trips.
- Parameters change behaviour: q_peak 14 gives a 0.72 s flip, q_peak 28 gives 0.525 s; both succeed.

## 2026-09-27 — MAP-Elites design decisions and evaluation harness step 4: trajectory recording, episode length (executed; no training)

**Decisions (user):**

- Keep the two proposed behaviour descriptors, rotation duration × maximum climb, and revisit them later if needed.
- Search episodes last 5 s.
- The archive design keeps the "crash loophole" risk in view (see M1).
- A further risk to keep in view: the effort-plus-altitude-loss fitness could favour a "do nothing" solution such as stable hover with zero altitude loss.

**Harness change (`Evaluation/evaluate.py`):**

- `run_episode(..., record=True)` and `evaluate(..., record=True)` optionally return the per-step trajectory: t, position, quaternion, velocity, body rates, applied (clipped) action, and the detector's pitch revolution, including the t = 0 state.
- `evaluate(..., episode_seconds=...)` sets the episode length.
- Control effort is now accumulated in float64. It was previously computed in float32, which changed values only at the 1e-8 relative level.

**Checks, run outside the project ([log](../Logs/evaluation_harness/step4_trajectory_checks_log.txt)).** The controllers were the default and slow three-phase flips, constant hover, motors off, and a controller that sends out-of-range commands.

- Recording changes no result field.
- Array lengths are steps+1 on a 5 ms time base; row 0's action is NaN and every applied action lies within [−1, 1].
- Recorded state equals the observation each controller received, up to float32 observation rounding.
- Every harness metric recomputes exactly from the trajectory: altitude loss, effort, flip time, final tilt and |ω|.
- Repeated runs give identical arrays.
- Out-of-range commands are recorded as their clipped values.
- Step-2 and M1 numbers are unchanged.
- 5 s episodes give 1000 steps with the same verdicts and timings as 10 s. Pitch travel grows by only 0.03–0.06° between 5 and 10 s for the scripted flips.
- Crash trajectories end at termination.
- Result equality ignores the trajectory.

## 2026-09-27 — MAP-Elites: exploratory sampling of the controller genome for descriptor bounds (executed; no training)

**Decisions (user):**

- Keep the 10×10 grid.
- Risk to keep in view: minimal-effort fitness may exploit "barely valid" rotations, sitting just above the detector's thresholds.
- Open question raised: why §04 counts 120° tilt as inverted rather than about 180°.

**Descriptor definitions, both computed from the harness trajectory at 5 ms resolution:**

- rotation duration = t(detector revolution ≥ 351.4°) − t(revolution ≥ 10°);
- max climb = max(z₀ − z) ≥ 0 (NED).

**Sampling:**

- 400 Latin-hypercube samples, seed 0, over physically bounded genome ranges: t_climb 0–0.6 s, F_climb 1–3 mg, q_peak 8–35 rad/s, a_brake 20–200 rad/s², F_rot 0.3–1.5 mg, Kp_att 20–120, Kd_att 4–25, Kv 1–8, Kz 0.5–4, Kvxy 0.5–4, Kpxy 0.2–2.
- Plus 11×9 one-at-a-time sweeps around the default and the default itself.
- 500 nominal 5 s episodes in 553 s on 2 CPU cores. Outputs: [samples, analysis, figure](../Logs/map_elites/descriptor_sampling/).

**Results:**

- **Success rate:** 101/400 (25%). Failures: 264 exceeded the one-flip budget (over-rotation), 27 crashed, 4 had no final hold. The feasible region depends mostly on a_brake: the success rate by tercile is 0.58, 0.16 and 0.02, and the largest successful a_brake is 145. Kd_att matters too: the lowest successful value is 6.4.
- **Descriptor ranges among successes:**
  - duration 0.24–0.75 s (p1 0.246 ± 0.008, p99 0.659 ± 0.058 by bootstrap);
  - climb 0.03–3.67 m (p1 0.028 ± 0.002, p99 3.15 ± 0.40), heavily skewed: median 0.49 m, 9/101 above 2 m.
- **Convergence:** bounds from the first 200 versus all 400 samples differ by up to about 11% in the tails. The tails remain uncertain, so edge cells must be checked after the MAP-Elites run.
- **Correlation:** Spearman ρ(duration, climb) = +0.19 (p = 0.06), so the axes are close to independent.
- **One-at-a-time sweeps:** duration is set by q_peak and a_brake; climb by t_climb and F_climb. The recovery gains change neither descriptor, so they act only on recovery quality.
- **Inversion:** every successful flip reached a sampled maximum tilt of at least 175.7° (median 178.6°). At 5 ms sampling an exact 180° is never sampled: the rotation advances 4–9° per step.

## 2026-09-27 — §04 detector: inversion threshold raised from 120° to 170°, inversion checked between samples (executed; no training)

**Decision (user):** "inverted" should mean essentially upside down, so the threshold is now 170° (gz ≤ cos 170° ≈ −0.985). It is not exactly 180° because the attitude is sampled while the vehicle turns 4–9° per 5 ms step. Evidence from the genome sampling: every successful flip reached a sampled tilt of at least 175.7°.

**Change (`Success_Detector/success_detector.py`, README):**

- `SuccessConfig.inverted_gz` is now cos 170°.
- The inversion check also uses the lowest gz along the shortest rotation between consecutive samples (`min_gz_on_arc`, slerp, 8 sub-steps). The verdict no longer depends on the sampling rate. Without this, 50 Hz evaluation (the PPO track's decision rate) would reject real flips: the default scripted flip is sampled at only 169.6° and the fast one at 164.2°.
- Nothing else in the detector changed.

**Impact:** every user of the default `SuccessConfig` now applies the stricter rule. This includes the evaluation harness.

**Checks, run outside the project ([log](../Logs/success_detector/inversion_170deg_checks_log.txt)):**

- The arc minimum is exact on the analytic cases: 160°→200° gives gz = −1, sign-invariant. On 3000 random steps of up to 46° it is within 1e−3 of a 2001-point reference.
- **Synthetic flip with a mid-flip roll excursion:**

  | Roll excursion | Maximum tilt | 170° rule | Old 120° rule |
  |---|---|---|---|
  | 5° | ≈175° | pass | pass |
  | 15° | ≈165° | **fails, "never inverted"** | passes |
  | 40° | ≈140° | fails, rotation incomplete (317°) | fails the same way |

- The default, slow and fast scripted flips succeed at 200 Hz and, via the arc check, when fed at 50 Hz.
- Harness numbers are unchanged (0.605 s / 0.835 s / 1.1172).
- All 400 Latin-hypercube samples were re-run: 101/101 successes kept, 0 verdicts changed.
- The detector's built-in self-tests (7 scenarios) pass, and so do the 7 pre-cleanup unit tests, run from outside the project.

## 2026-09-27 — MAP-Elites M2: problem definition (executed; no training) + methods page for the report

**Decisions (user):**

- Descriptor ranges are rotation duration 0.20–0.80 s and maximum climb 0–3 m, on a 10×10 grid.
- The genome box narrows a_brake to 20–150 and Kd_att to 6–25.
- The methods used and verified so far are kept for the report.

**Added `MAP_Elites/problem.py`:**

- the genome box;
- `descriptors(trajectory)`;
- `fitness = −(control effort + 1.0 · altitude loss)`;
- `evaluate_genome(x)`, which runs one nominal 5 s episode through the harness.

Validity is only the §04 verdict. Invalid genomes get no descriptors and no fitness, which closes the crash and do-nothing loopholes by construction.

**Checks, run outside the project ([log](../Logs/map_elites/m2_problem_checks_log.txt)):**

- Out-of-box or wrong-length genomes are rejected.
- Default, slow and fast flips are valid:
  - durations 0.345, 0.460 and 0.265 s;
  - fitness −1.117, −1.130 and −1.439. The slow flip's effort is lower, 0.971, but it loses 0.16 m of altitude.
- Descriptors and validity reproduce the sampling study on 13 samples. Two samples were skipped because they lie outside the narrowed box.
- The default flip followed by motors-off at 1.5 s gets the *same* descriptors (0.345 s, 0.633 m) but is invalid (crash). Hover and an over-rotating sample are invalid.
- Repeated evaluation is identical.

**Added [`Documentation/map_elites/METHODS_FOR_REPORT.md`](map_elites/METHODS_FOR_REPORT.md)**, a summary of every verified method so far with pointers to its evidence. It is linked from the documentation index.

## 2026-09-27 — MAP-Elites M3: archive (executed; no training)

Added `MAP_Elites/archive.py`:

- a 10×10 grid with one elite per cell;
- strictly-better replacement, so ties keep the incumbent;
- invalid candidates rejected;
- coverage, best and mean fitness, and a QD-score with a fixed fitness floor of −30;
- a fitness grid for heatmaps;
- verified JSON save and load.

The methods page gained section 6.

**Checks, run outside the project ([log](../Logs/map_elites/m3_archive_checks_log.txt)).** They found and fixed two small bugs before commit:

1. **Floating-point cell edges.** Durations that fall exactly on a cell edge were placed one cell too low; for example 0.26 s landed in cell 0 instead of 1. Durations are multiples of 5 ms and several edges coincide with them, so this would have happened in real runs. Fixed with a 1e−9 guard. All 9 interior edges per axis are now tested.
2. **Statistics depended on insertion order.** The mean fitness differed in the last bit after a save/load. Statistics are now computed in sorted cell order.

Also verified:

- every cell centre maps to its own cell, and out-of-range values go to edge cells;
- new, worse, tie, better and invalid candidates behave as specified;
- the QD-score is monotone over 500 random insertions;
- `random_elite` is deterministic for a given seed;
- save and load are exact, and a tampered file is rejected;
- the real default flip enters cell (2, 2).

## 2026-09-27 — MAP-Elites M4: the loop (executed checks; the full run is for the user to start)

**Decisions (user):**

- initialisation: default genome + 100 LHS genomes;
- uniform random parent selection;
- Gaussian mutation with σ = 10 % of each range, clipped to the box;
- batches of 20;
- budget 5000;
- checkpoints every 500;
- fixed seed.

**Added `MAP_Elites/map_elites.py`**, with the loop, checkpoint/resume and a command-line interface:

- start: `python MAP_Elites/map_elites.py --name ME1 --budget 5000 --workers N --seed 0`
- continue: `--resume`

Outputs go to `MAP_Elites/runs/<name>/`: config, history, archive checkpoints, final archive and state.

**Checks, run outside the project ([log](../Logs/map_elites/m4_loop_checks_log.txt)):**

- Mutations stay in the box, and the step spread equals σ (ratio 0.99–1.01).
- Initial genomes are deterministic and in the box.
- Workers 1 and 2 give identical archives and histories.
- Accounting: 40 evaluations; the counts sum to 40; all stored elites are §04 successes.
- A different seed gives a different archive.
- A run killed at 36 evaluations, with its last checkpoint at 31, resumed to exactly the uninterrupted result.
- Resuming with changed search settings is refused, and an all-invalid initialisation raises.
- The command line works under the Windows-style "spawn" start method, refuses to overwrite a run, and resumes to a larger budget.

**Small-run observation, not a result:** in 40 evaluations, 17 cells were filled, and the best fitness (−1.032) already beat the default (−1.117). Throughput in the cloud was about 1.4 s per evaluation per worker.

**Fix made during development:** the checkpoint trigger originally required the evaluation count to be an exact multiple of 500. Batch boundaries do not align with that, because the default genome makes 101 initial evaluations, so it now fires whenever the count crosses a multiple.

## 2026-09-27 — MAP-Elites M5a: archive heatmap and convergence figures (executed checks; no training)

Added `MAP_Elites/analysis.py`. Running `python MAP_Elites/analysis.py --name <run>` writes `figures/archive_heatmap.png` and `figures/convergence.png` into the run folder and prints the archive statistics. It never re-runs the search, but it does evaluate the default genome once to place its marker.

**Checks, run outside the project ([log](../Logs/map_elites/m5a_analysis_checks_log.txt)):**

- The heatmap orientation is correct, verified from pixel colours: a filled cell is coloured and its transpose is empty grey.
- A better cell is darker.
- The default marker lands in cell (2,2), and out-of-range marks are clamped.
- An empty archive renders.
- The convergence figure has two panels, each with one series, and the data is plotted exactly.
- `load_run` round-trips a tiny real run: 100 evaluations, test scale, not a result.

A readability fix made during review: the "default controller" label now has a background, so it stays legible over dark cells.

## 2026-09-27 — MAP-Elites run ME1 (executed by the user): results and analysis

**Run:** `MAP_Elites/runs/ME1`, seed 0, budget 5000, 12 workers, about 38 min. Final counts: 100 new, 453 improved, 3511 rejected, 936 invalid (18.7 %). Figures are in `runs/ME1/figures/`. The re-validation data is in [ME1_analysis](../Logs/map_elites/ME1_analysis/).

**Illumination:**

- All **100/100 cells** hold a valid §04 flip, so every combination of rotation duration (0.245–0.76 s) and climb (0–2.94 m) in the grid is achievable. No descriptor fell outside the bounds.
- Coverage reached 100 % at 1921 evaluations, and the QD-score reached 90 / 95 / 99 % of its final value at 841 / 1181 / 1921 evaluations.
- The last 3000 evaluations added no new cells, only 137 small improvements (QD-score +0.3 %). About 2000 evaluations suffice for this grid.
- The invalid-child rate stayed at 17–20 % throughout.

**Quality:**

- The best elite scores fitness −0.758, **32 % below** the hand-tuned default's control effort (−1.117). It is in cell (4,3): duration 0.44 s, climb 1.0 m, no altitude loss, recovered at 1.63 s.
- In the default controller's own cell (2,2), the elite scores −0.789 (−29 %). The median over cells is −1.025.
- The worst cells are slow flips with little climb, which lose up to 2.6 m of altitude (fitness down to −4.3).

**How cost was reduced (genome trends):**

- Longer, gentler climbs: t_climb sits at its 0.6 s upper bound in 35 elites (median 0.57 s), and the median climb thrust is 1.69 mg against 2.2 mg for the default.
- Softer braking: a_brake sits at its lower bound of 20 in 24 elites (median 43 against 70 for the default).
- These box limits are therefore active. This follows from the effort definition, which penalises deviation from the hover command.

**Verification (cloud, no training):**

- Re-running all 100 elites (5 s) reproduces the Windows fitness and descriptors to about 5e−12.
- **10 s benchmark re-validation:** 100/100 still succeed, in the same cells, with unchanged fitness. The minimum sampled inversion is 179.4°.

**Risk found ("barely valid"):**

- Pitch travel at 10 s has a median of 382°, but 14 elites exceed 400° and 2 exceed 415°, against the 420° budget. The best elite uses 396°.
- *[Corrected 2026-09-28]* Spearman(fitness, travel) = −0.19 (p = 0.06) means that **better** elites use slightly **less** pitch travel. The original wording here had the direction reversed. See the ME2/ME3 entry.
- A robustness check is therefore required before a final controller is chosen (M5b, pending the user's decisions).

## 2026-09-28 — MAP-Elites: genome-box overrides from the command line (for ME2 / ME3)

**Why:** in ME1, t_climb sits at its 0.6 s upper bound in 35 elites and a_brake at its lower bound of 20 in 24 elites. Two isolated follow-up runs will widen one limit each (ME2: t_climb upper 0.6 → 1.0; ME3: a_brake lower 20 → 10), 2000 evaluations each, compared with ME1's checkpoint at 2001 evaluations.

**Change (`MAP_Elites/problem.py`, `MAP_Elites/map_elites.py`):**

- `problem.genome_box(overrides)` returns the validated box: unknown names, non-finite or empty ranges, and boxes that exclude the default genome are rejected. `GENOME_BOUNDS` itself is unchanged.
- `Config.genome_bounds` holds the run's full box; it is written to `config.json`. Initialisation (LHS), mutation (clip and sigma) and the genome check in `evaluate_genome` all use it.
- CLI: `--bound NAME=LOW,HIGH` (repeatable) for new runs. It is refused together with `--resume`, and resume refuses a changed box. An ME1-style `config.json` without the key loads as the original box.

**Checks** (log: `Logs/map_elites/bounds_override_checks_log.txt`):

- With the default box the new code reproduces the old code exactly (archive and history of a 141-evaluation run identical).
- Override runs keep every genome in the box, are identical for 1 and 2 workers and under the Windows `spawn` start method, and resume exactly.
- Nine malformed or forbidden `--bound` uses exit with a clear message.

**Known side effects of widening a limit (part of the experiment, not bugs):**

- The mutation step of that gene grows with its range (sigma = 10 % of range; t_climb 0.06 → 0.10 s).
- Latin-hypercube initial samples spread over the wider range.
- Observed in the checks: the default genome with t_climb = 0.9 crashes (climbs out of the region), and with a_brake = 10 it succeeds with a 0.875 s rotation, above the 0.80 s grid edge, so slow flips would be placed in the edge cells.

**Planned (the user runs):**

    python MAP_Elites/map_elites.py --name ME2 --budget 2000 --bound t_climb=0,1.0
    python MAP_Elites/map_elites.py --name ME3 --budget 2000 --bound a_brake=10,150

## 2026-09-28 — MAP-Elites runs ME2 and ME3 (executed by the user): widened genome limits, one at a time

**Runs:** seed 0, **budget 5000** (raised from the planned 2000 so the final archives compare 1:1 with ME1; the 2001 checkpoints give the early comparison), 12 workers.

- ME2: only `t_climb` upper 0.6 → 1.0.
- ME3: only `a_brake` lower 20 → 10.

Full numbers: `Logs/map_elites/ME2_ME3_analysis/analysis_log.txt`. Elite re-validation: `revalidation_5s_10s.json` in the same folder.

**Results at 5000 evaluations (ME1 / ME2 / ME3):**

- Best fitness: −0.758 / **−0.702** / −0.743.
- Median cell fitness: −1.025 / **−0.914** / −1.010.
- QD-score: 2872 / 2881 / 2843. Cells filled: 100 / 100 / 99.

**ME2 improves the archive:**

- 67/100 cells are better than ME1 (median +0.059, Wilcoxon p = 6·10⁻⁷).
- The limit is still active: 25 elites sit at t_climb = 1.0.
- The gain needs budget. At 2001 evaluations there is no difference (48/48, p = 0.23), all cells are filled only at 3841 (ME1: 1921), and 29 % of children are invalid (ME1: 19 %).
- Mechanism: a longer, gentler climb lowers the quadratic effort. The best ME2 flip ends later (t_flip 1.55 s against 1.07 s) and higher (climb 2.0 m). Flip time is not part of the fitness.

**ME3 shows no effect:**

- 53 cells better, 46 worse, p = 0.82.
- a_brake < 20 in 21 elites but none at 10, so ME1's lower bound of 20 was not limiting the search.

**Verification (cloud):**

- All elites reproduce at 5 s (|Δfitness| ≤ 1.5·10⁻⁹).
- At 10 s: ME2 99/100 (cell (0,0) exceeds the 420° one-flip travel budget), ME3 99/99.
- **Travel margin:** Spearman(fitness, travel) ≈ −0.45 in both runs, so better elites travel less. The top-10 elites of every run use 370–396°. Correction of the ME1 wording above.

**Limitation:** one search seed per configuration.

**Planned:** repeat ME1 and ME2 with seeds 1 and 2 (the user runs them) to measure run-to-run variance before claiming that ME2 > ME1.

## 2026-09-28 — Evaluation harness step 5: stress condition 1, observation noise

**Why:** the robustness screening of MAP-Elites elites (and later PPO) tests one disturbance at a time. The brief's order is nominal first, disturbances only after a working flip.

**Change (`Evaluation/evaluate.py`):**

- A frozen `Condition` dataclass with one field per disturbance. The default `Condition()` (= `NOMINAL`) is the benchmark.
- First field: `obs_noise_scale`, a multiple of the simulator's **own** default observation-noise std per group, read from `QuadcopterVelocityEnv(observation_noise_std=None)` and not copied. `OBS_NOISE` is scale 1: pos 0.002 m, quat 0.001, vel 0.02 m/s, omega 0.005 rad/s, motor 1.0 rad/s, target 0.
- `make_benchmark_env`, `run_episode` and `evaluate` take `condition=`. MAP-Elites is untouched and stays nominal.

**Checks** (log: `Logs/evaluation_harness/step5_obs_noise_checks_log.txt`):

- The nominal default is identical to the previous harness (5 s and 10 s).
- Measured noise matches the configured std per group. The quaternion is 0.866×, because the simulator re-normalises the noisy quaternion and removes the radial part: √(3/4).
- An open-loop controller gives identical results with and without noise, so noise reaches only the controller, never the dynamics or the §04 verdict.
- Episodes are reproducible per seed, independent of episode order, and leave the global RNG untouched.
- Informative: the default three-phase flip succeeds in 10/10 episodes (10 s, seeds 0–9) under `OBS_NOISE`.

**Next:** wind (step 6), then a perturbed initial state (step 7), each alone.

## 2026-09-28 — Evaluation harness step 6: stress condition 2, wind

**Change (`Evaluation/evaluate.py`):**

- New `Condition.wind_speed` field: horizontal wind [m/s].
  - Its direction is drawn uniformly per episode from `np.random.default_rng((WIND_STREAM, seed))`.
  - It is built with the simulator's own `Wind` class ("FIXED").
- Optional `wind_gusts`: the simulator's "SINE" model, with the same mean and direction plus its fixed sinusoidal gusts.
- Preset `WIND` = steady 1 m/s.
- The condition is stored on the env (`benchmark_condition`), and `run_episode` sets the wind right after `reset(seed)`.

**Why not the simulator's `random_wind`:** its "RANDOMSINE" model draws from Python's global `random`, so episodes are not reproducible per seed. With the default heading range it also always blows along +x.

**Physics note:** the simulator's wind acts only through per-axis quadratic drag, a_i = Cd·u_i|u_i|/m with Cd = 0.1 and m = 1.2 kg. At 1 m/s this is only 0.06–0.08 m/s², depending on direction. The simulator reports the body velocity one step late (`quad.vel`); this does not affect the harness.

**Checks** (log: `Logs/evaluation_harness/step6_wind_checks_log.txt`):

- Nominal and `OBS_NOISE` episodes are identical to the previous harness.
- Invalid values are rejected.
- Directions are uniform over 200 seeds and reproducible per seed. The global numpy and Python RNGs are untouched, and drawing the wind does not consume the env's noise generator.
- The measured acceleration equals the simulator's drag formula within 1 % for three directions.
- Episodes are order-independent.
- Informative: the default flip succeeds in 10/10 episodes both at 1 m/s and with gusts around a 2 m/s mean (speed 0–4.9 m/s).

**Implication:** the mild wind level barely disturbs this vehicle. A stronger level belongs to the later "very intense" condition.

## 2026-09-28 — Evaluation harness step 7: stress condition 3, perturbed initial state

**Change (`Evaluation/evaluate.py`):**

- New `Condition` fields `init_tilt_deg`, `init_rate` and `init_vel`.
  - After `reset(seed)`, the vehicle state is overwritten with a draw from `np.random.default_rng((INIT_STREAM, seed))`: tilt about a random horizontal axis, uniform in [0, init_tilt_deg]; body rates and linear velocity uniform in ±init_rate and ±init_vel per axis.
  - The simulator's integrator is restarted from that state, and the first observation is recomputed.
  - Position, yaw and motor speeds stay nominal.
- Preset `PERTURBED_START`: tilt ≤ 5°, rates ≤ 0.2 rad/s, velocity ≤ 0.2 m/s, i.e. an "approximately stable hover" (brief p.1).
- Tilts of 15° or more are rejected, because the §04 detector requires an upright start (15° cone).

**Checks** (log: `Logs/evaluation_harness/step7_perturbed_start_checks_log.txt`):

- `NOMINAL`, `OBS_NOISE` and `WIND` episodes are identical to the previous harness.
- Invalid values are rejected.
- Over 2000 draws, the ranges and uniformity are correct and the draws are reproducible per seed. The global numpy and Python RNGs are untouched.
- The episode starts exactly at the drawn state, both in the true state and in the controller's first observation. The first integrated step moves the position by v·dt and rotates the attitude by the initial rates.
- Episodes are order-independent. All three disturbances together are reproducible.
- Informative: the default flip succeeds in 10/10 episodes under `PERTURBED_START` (10 s).

**Simulator property found:** `env.step` calls `quad.update(env.t, dt)` before advancing `env.t`. The scipy `ode.integrate(t, step)` call therefore integrates only up to the *current* time: the first step is a no-op, and every reported state lags one dt. This is the root cause of the velocity lag noted in step 6. It is the same for every controller and harmless for the comparison. The simulator is not modified.

**Stress-condition set now complete for the mild level:** `OBS_NOISE`, `WIND`, `PERTURBED_START`, each switchable alone. **Next:** robustness screening of the elites (M5b-2), after the seed runs.

## 2026-09-28 — MAP-Elites M5b-2 step 1: robustness screening engine (`MAP_Elites/robustness.py`)

**What it does:** takes as candidates the default genome plus the top-K elites of each listed run, by nominal search fitness, without duplicates. Every candidate is run for 10 s on the harness under `nominal` (1 deterministic episode) and under `obs_noise`, `wind` and `perturbed_start` with the same episode seeds 0…E−1 for all candidates (a paired comparison). Only the §04 detector judges success.

**Output:** one row per episode in `MAP_Elites/runs/robustness/<name>/episodes.csv`, next to `candidates.json` and `config.json`.

- Episodes are appended in chunks, so a stopped screening resumes where it stopped.
- Another candidate set or other settings under the same name are refused.
- A controller error (e.g. a non-finite motor command) is recorded as a failed episode instead of stopping the run.

**Checks** (log: `Logs/map_elites/m5b2_screening_checks_log.txt`):

- Candidate selection and order are correct.
- A job equals a direct harness episode under all four conditions.
- Nominal episodes are seed-independent.
- 1 worker gives the same result as 2 workers.
- Stop and resume gives the same result as an uninterrupted run.
- The CLI works under the Windows `spawn` start method.

**Next (step 2):** ranking by robust success rate (Wilson intervals), a confirmation stage with more episodes for the leaders, and a summary table.

## 2026-09-28 — MAP-Elites M5b-2 step 2: ranking, confirmation and selection (`MAP_Elites/robustness.py`)

**Selection rule (as agreed):**

- A candidate must succeed in the nominal 10 s episode.
- Candidates are ranked by robust success rate: successes over all `obs_noise`, `wind` and `perturbed_start` episodes, with a Wilson 95 % interval.
- Ties are broken by nominal 10 s fitness, −(effort + altitude loss).

**Two stages:**

1. Stage 1 ranks every candidate on seeds 0…E−1 only.
2. The top N ("leaders", default 5) then also run seeds E…M−1 (default M = 50) and are ranked again on all M seeds. Rank 1 is the selected controller.

Stage 1 never uses the extra seeds, so the leaders do not change when a screening resumes. `--confirm 0` selects directly from stage 1.

**Outputs:** `summary.json` (both rankings with per-condition counts, intervals and failure modes, the leaders, and the selected id and genome), `ranking.csv`, and a printed table.

**Checks** (log appended to `Logs/map_elites/m5b2_screening_checks_log.txt`):

- Synthetic rows confirm the rule: a nominal failure goes last, 29/30 ranks below 30/30 even with better fitness, and ties are broken by fitness.
- Only the leaders receive the extra seeds.
- A resume adds 0 episodes and gives an identical summary.
- A real end-to-end CLI run works (3 candidates, 2 leaders, 2 workers).

**Note for the analysis:** at the mild levels every tested flip has succeeded so far (the default in 10/10 under each condition). If most elites saturate at 100 %, the ranking falls back to fitness. The pitch-travel margin, e.g. 409° for ME1's best elite in one disturbed episode, and the later "very intense" level then become the discriminating evidence.

**Planned run (the user starts it after the seed runs):**

    python MAP_Elites/robustness.py --name R1 --runs ME1 ME2 ME3 ME1_s1 ME1_s2 ME2_s1 ME2_s2 --top 10 --episodes 10 --workers 12

## 2026-09-29 — MAP-Elites seed replicates (executed by the user): ME1 vs ME2 over search seeds 0, 1 and 2

**Runs:** `ME1_s1`, `ME1_s2`, `ME2_s1` and `ME2_s2` (budget 5000, 12 workers), plus the existing seed-0 runs. The only difference within each group is the search seed. Details: `Logs/map_elites/seed_replicates/analysis_log.txt`.

**Result at 5000 evaluations (mean ± sd over 3 seeds):**

| | ME1 | ME2 |
|---|---|---|
| Best fitness | −0.752 ± 0.006 | **−0.699 ± 0.006** |
| Median cell fitness | −1.021 ± 0.011 | **−0.904 ± 0.021** |
| QD-score | 2871.5 ± 0.2 | **2881.2 ± 1.9** |

- Every ME2 run beats every ME1 run on all three measures. With 3 vs 3 runs this is the strongest possible result: one-sided Mann-Whitney p = 0.05.
- Per cell, ME2 wins in 67, 79 and 68 of 100 cells for seeds 0, 1 and 2 (all p < 10⁻⁵).
- The gain grows with climb height: +0.01 in the lowest climb row, +0.29 in the highest.

**Conclusion:** widening t_climb to 1.0 s gives a real, seed-independent improvement (best −7 %, median cell −11 % cost). It needs more evaluations: all 100 cells are filled at 2681–3841 evaluations, against 1921–3101 for ME1. ME1's QD-score is almost identical across seeds, so the search converges for that genome box.

**Limitation:** 3 seeds per configuration.

## 2026-09-29 — Robustness screening R1 (executed by the user): mild conditions saturate

**Run:** 71 candidates (default + top-10 of ME1, ME2, ME3 and the four seed runs), 10 s episodes, 2801 episodes. Details and copies of `summary.json` / `ranking.csv`: `Logs/map_elites/R1_robustness/`.

**Stage 1 (seeds 0–9):**

- Nominal 71/71, `obs_noise` 699/710, `wind` 710/710, `perturbed_start` 694/710.
- **Every failure is over-rotation**, i.e. more than 420° of pitch travel.
- 63/71 candidates reach 30/30. The failing ones already travel further in the nominal episode (Spearman 0.50), and disturbances add a median of 19° of travel (maximum 47°).

**Confirmation:** 5 ME2-family leaders, 50 seeds each: all 150/150 (Wilson 98–100 %).

**Selection:** the agreed rule selects `ME2_s1:4,5` (fitness −0.692). But the leaders differ by only 1.6 % in fitness, while their worst-case travel margin to 420° ranges from 1° to 27°. The selected elite keeps 4.7°; `ME2_s2:5,7` keeps 26.9° at fitness −0.703.

**Conclusion:** at the mild levels the rule reduces to nominal fitness, and the 1 m/s wind has no effect. The final choice needs a discriminating test (stronger disturbance) or an explicit margin criterion. **Decision pending (user).**

## 2026-09-29 — Evaluation harness step 8: controller-side action and gyro noise (PPO-track definition)

**Change (`Evaluation/evaluate.py`):** new `Condition` fields `action_noise` and `gyro_noise`, reproducing `run_baselines.rollout` of the PPO track:

- Gaussian noise is added to the 4 motor commands (then clipped to [−1, 1]) and to the observed body rates `obs[13:16]`.
- One sample is drawn per 20 ms decision and held for 4 simulator steps.
- The generator is `default_rng(SeedSequence([12345, seed]))`, in the same draw order: gyro first, then action.

The env, the dynamics and the detector are untouched. The noisy, clipped command is what the env receives and what the effort counts. Preset `STRESS` = action 0.15 + gyro 0.2 rad/s (the PPO track's "stress" point).

**Checks** (log: `Logs/evaluation_harness/step8_controller_noise_checks_log.txt`):

- All earlier conditions are identical to the previous harness.
- The noise sequence equals the PPO track's generator exactly, block-held over 4 steps, with clipping. Only the body rates are observed noisily.
- The measured standard deviations are 0.149 and 0.196, against the target 0.15 and 0.2.
- Episodes are reproducible per seed, independent of episode order, and leave the global RNGs untouched.

**Finding (informative runs):** the default flip and two ME2 elites score **0/10** under action noise 0.05 and under `STRESS`. The flip itself stays a clean single turn: net rotation ≈ 360°, maximum revolution ≤ 371°. The failures come from the §04 **pitch-travel budget of 420° counted over the whole episode**: noise-induced hover jitter after the flip adds 110–310° of |pitch| travel. This is the open detector question noted earlier, now shown to decide every noisy result. It likely also explains part of R1's "over-rotation" failures. **Decision needed (user) before any noise comparison:** how the one-turn rule should treat post-flip jitter.

**Also noted for the report (PPO track):** the delivered PPO acts in CTBR mode (collective thrust + body rates → fixed rate loop → motors), while the brief specifies a policy acting directly on the four motors. Their direct-motor ablation reached 0 %.

## 2026-09-29 — §04 detector: the 420° travel budget counts only the maneuver (decided by the user)

**Why:** under actuator noise 0.05, the pitch deviation after the flip stays below 5° (RMS about 2°). Summed over 40–50 back-and-forth wobbles in the remaining 8 s, however, it added 90–130° to the whole-episode |pitch| travel and failed every flip. Travel then measured the episode length, not the rotation.

**Change (`Success_Detector/success_detector.py`, `README_Section4.md`):**

- |pitch| travel is counted until the first sample that has completed the revolution and is upright again (tilt < 15°).
- Protection that remains unchanged:
  - maximum revolution ≤ 390° over the whole episode;
  - net revolution within 15° of 360° at the finish;
  - the final 0.4 s upright-and-slow hold.
- `pitch_travel_total_rad` keeps the whole-episode value as a diagnostic.
- `travel_until_upright=False` restores the old rule.

**Checks** (log: `Logs/success_detector/travel_window_checks_log.txt`):

- Synthetic flips: only "flip + long jitter" changes (FAIL → PASS). A double flip, a further 40° turn and back, a reverse turn, and rocking before completion all still FAIL.
- With the flag off, the result equals the old detector exactly.
- Nominal 10 s episodes of 4 controllers get identical verdicts and reasons to the old detector.
- Every old success remains a success, since the counted travel ≤ the old travel. The ME archives and R1 therefore stay valid; some old failures may now pass.

**Informative (10 s, seeds 0–9):**

| Controller | Action noise 0.05 (PPO track's "nominal") | `STRESS` |
|---|---|---|
| Default flip | 9/10 | 0/10 |
| ME2_s1:4,5 | 10/10 | 0/10 |
| ME2_s2:5,7 | 10/10 | 0/10 |

Under `STRESS`, the failures are now real:

- the final hold is not met (final |ω| ≈ 0.8–1.0 rad/s);
- some flips never reach 170° of tilt (157–169°).

## 2026-09-29 — MAP-Elites: final controller packaged (`Controllers/map_elites_final.json`, `map_elites_controller.py`)

**Selection (user-agreed rule, stress tests only logged):** best nominal fitness → **ME2_s1 cell (4,5)**.

- Re-checked with the maneuver-only travel budget, on a 10 s nominal episode: success, fitness −0.692 (default −1.117).
- Travel counted during the maneuver: 352°. Maximum revolution: 364.9° (limit 390°).
- No altitude loss; final |ω| ≈ 0.
- Trade-off to report: the flip is slow. It climbs for 1.0 s (1.5 m), completes the turn at 1.5 s and settles at 2.3 s; the default completes at 0.6 s and settles at 0.84 s. The fitness rewards low effort, not speed.

**Package:**

- The JSON holds the genome plus its provenance (run, seed, genome box, cell, descriptors, search fitness, selection rule).
- `load_final_controller()` returns the `ThreePhaseFlip` with the course interface (`reset`/`act`).
- `load_params()` rejects missing, extra or non-finite genes.

**Checks** (log: `Logs/map_elites/final_controller_checks_log.txt`):

- The genome equals the archive elite exactly.
- A 10 s nominal success with fitness −0.6924, reproducible.
- Malformed files are rejected.

## 2026-09-29 — Harness step 9: the §02 baselines (random, PID hover, scripted flip) run in the universal harness

**Change:**

- `Baselines/baselines.py` is copied unchanged from the PPO track (GitHub `Reward_and_Training/baselines.py`). This also resolves the missing import of `Baselines/run_baselines.py`.
- New `Baselines/harness_adapter.py`:
  - `DecisionRateAdapter` calls a controller that decides every k = 4 steps (their ActionRepeat, 50 Hz) on steps 0, 4, 8, …, holds its action in between, and returns its diagnostics as `info`.
  - `make_baseline(name)` builds each baseline through their own `make_controller`, with the same decision period (0.02 s), vehicle model, default `FlipScript` (their tuned values) and default `PIDGains`.

**Checks** (log: `Logs/evaluation_harness/step9_baselines_adapter_checks_log.txt`):

- Adapter timing and output types are correct. Invalid arguments are rejected.
- The controllers are configured identically to their `make_controller` on their action-repeat-4 env.
- **Bit-exact parity:** over 100 decisions, their `ActionRepeat` rollout and our harness + adapter give identical actions, positions and quaternions for all three baselines. This was run with the simulator's default sensor noise, which their env always has.
- Informative, on the nominal 10 s benchmark with our §04 detector (seeds 0–4): scripted flip 5/5, PID hover 0/5 (never inverted), random 0/5 (crash).

## 2026-09-29 — Harness step 10: final comparison script (`Evaluation/final_comparison.py`)

**What it does:** runs every controller on the same benchmark, the same seeds and the same §04 detector:

- **Controllers:** `map_elites_final`, `three_phase_default`, `scripted_flip`, `pid_hover`, `random`.
- **Conditions:** `nominal` (the brief's benchmark, primary result); `action_noise_005` and `stress` (the PPO track's points, logged only, as agreed).
- **Parallelism:** jobs are blocks of seeds, with one env per job.
- **Output:** results sorted by seed in `Evaluation/runs/<name>/<condition>/` (`episodes.csv` and `summary.json` via `report.write_results`), plus `summary.txt` with the comparison tables and `config.json`.
- The nominal benchmark is deterministic, so deterministic controllers repeat one episode (stated in the docstring). The PPO controller will be added once its wrapper exists.

**Checks** (log: `Logs/evaluation_harness/step10_final_comparison_checks_log.txt`):

- The CLI runs under the Windows `spawn` start method with 2 workers (2 episodes, 5 controllers, 2 conditions).
- Parallel results equal direct serial harness calls.
- An existing run name, invalid counts and unknown controllers are refused.

**Planned run (the user starts it):**

    python Evaluation/final_comparison.py --name FINAL --episodes 50 --workers 12

## 2026-09-29 — MAP-Elites analysis: seed-comparison figure (`analysis.py --compare`)

**Change:** `python MAP_Elites/analysis.py --compare ME1=ME1,ME1_s1,ME1_s2 ME2=ME2,ME2_s1,ME2_s2` writes `MAP_Elites/runs/comparison/seed_comparison.png`, with three panels and one measure each (no shared y-axis):

- QD-score against evaluations, one line per seed;
- final best fitness per seed;
- final median cell fitness per seed, with the mean shown as a bar.

The two configurations are blue and orange (a validated colour-vision-deficiency-safe pair), with circle and square markers as secondary encoding. `--name` works as before, and exactly one of `--name` or `--compare` is required.

**Checks:**

- The figure rendered from the six real runs and was inspected visually. Direct labels that overlapped at 5000 evaluations were removed, and a truncated title was shortened.
- Malformed or duplicate labels, and giving both or neither option, are rejected (exit code 2).
- The `--name ME1` path still produces the heatmap and the convergence figure.

Figure copy: `Logs/map_elites/seed_replicates/seed_comparison.png`.

## 2026-09-29 — Methods page updated for the report (sections 9–13)

`Documentation/map_elites/METHODS_FOR_REPORT.md` gains five sections, each with its evidence path:

- genome-box experiments (ME2, ME3);
- search-seed replicates;
- stress conditions and the maneuver-only one-turn rule;
- robustness screening and final selection;
- baselines and the final comparison.

Text only; no code change.

## 2026-09-29 — Demo videos: `Evaluation/render_episode.py` (brief deliverable: successes and failures)

**What it does:** renders one normal harness episode (any controller or condition from `final_comparison.py`, any seed) as a side-view GIF of the pitch plane, written to `Evaluation/runs/videos/`.

- **Axes:** x forward and height above start.
- **Drawing:**
  - the body arm, enlarged, with the front rotor marked;
  - the thrust direction;
  - the trail of past positions;
  - time and flip angle, and the §04 verdict on the last frame.
- `--still` also saves a PNG of the inverted moment.
- Note: the verdict refers to the rendered episode length (`--seconds`, default 4 s).

**Checks** (log: `Logs/evaluation_harness/step11_render_checks_log.txt`):

- The body axes equal the simulator's `quat2Dcm` for 200 random quaternions, and a +90° pitch points the nose up.
- The frame count matches duration × fps.
- Invalid arguments are rejected.
- Visual check of the still.
- Example outputs: final elite nominal (SUCCESS); final elite under stress (FAIL: never inverted).

**Suggested demo set (the user renders):**

    python Evaluation/render_episode.py --controller map_elites_final --condition nominal --seed 0 --still
    python Evaluation/render_episode.py --controller three_phase_default --condition nominal --seed 0
    python Evaluation/render_episode.py --controller map_elites_final --condition stress --seed 0 --seconds 6
    python Evaluation/render_episode.py --controller random --condition nominal --seed 0 --seconds 2

## 2026-09-30 — Final comparison FINAL (executed by the user): MAP-Elites vs default three-phase vs scripted flip

**Run:** `final_comparison.py --name FINAL --episodes 50 --workers 12 --controllers map_elites_final three_phase_default scripted_flip`.

- Seeds 0–49, 10 s episodes, §04 detector with the maneuver-only travel budget.
- Random and PID hover were left out at the user's decision: the comparison is between the MAP-Elites controller, the scripted flip and PPO, which will be added once its wrapper exists.
- Copies of all results: `Logs/final_comparison/FINAL/`.
- Verified in the cloud: 50 distinct seeds per controller and condition. Nominal episodes are identical across seeds, as expected for a deterministic benchmark.

**Nominal (the brief's benchmark), 50 episodes each:**

| Controller | Success (95 % CI) | Effort | t_flip [s] | t_rec [s] | Altitude loss |
|---|---|---|---|---|---|
| MAP-Elites final (ME2_s1 4,5) | 50/50 (93–100 %) | **0.69** | 1.50 | 2.29 | 0.00 |
| Default three-phase (hand-tuned) | 50/50 (93–100 %) | 1.12 | **0.61** | **0.83** | 0.00 |
| Scripted flip (§02) | 50/50 (93–100 %) | 2.13 | 1.28 | 1.47 | 0.00 |

All three are 100 % reliable. MAP-Elites needs 38 % less effort than its hand-tuned starting point and 68 % less than the scripted flip. It is also the slowest to flip and settle, the effort/speed trade-off noted at selection.

**Logged conditions (not used for selection):**

- **Motor noise 0.05:** MAP-Elites 50/50; default and scripted 48/50 each (2× "no final hold"). MAP-Elites effort 0.80, against 1.22 and 2.15.
- **Stress (motor 0.15 + gyro 0.2 rad/s):**
  - MAP-Elites 0/50 (33 no final hold, 16 never inverted, 1 over-rotation);
  - default 0/50;
  - scripted 1/50 (46 no final hold).
- **Reading of the stress results:** the mean final |ω| is 0.78–0.90 rad/s, right at the detector's 0.8 rad/s stillness threshold. None of these controllers can hold still under 0.2 rad/s gyro noise, and some flips stop short of 170°. The PPO track reported scripted 20 % under the same noise with its lenient 150° / no-final-hold detector.

## 2026-09-30 — PPO-track success detector copied into Success_Detector/ (user decision: all measurements use her detector)

**Decision (user):** from now on every controller is judged with the PPO track's detector, so that our numbers and hers use the same success definition. Her measurements are not re-run or changed.

**Change:** new file `Success_Detector/ppo_track_detector.py`. It is a verbatim copy of `SuccessConfig`, `SuccessDetector` and the geometry they use, from her `Reward_and_Training/flip_reward.py` (snapshot 2026-09-29, sha256 in the file). Only the docstring and imports are new. It is copied rather than imported so the harness does not depend on a reward module.

**Her criterion, for the record:**

- signed geometric revolution ≥ 2π − 0.15;
- tilt ≥ 150° at some update;
- afterwards upright (< 15°) and slow (|v| < 0.4 m/s, |ω| < 0.8 rad/s) for 20 consecutive updates, one update per decision (20 ms), so 0.4 s;
- success is latched once reached. There is no one-turn limit and no crash or region check inside the detector.

**Checks (cloud, script deleted):**

- every copied definition is textually identical to the source;
- on 300 random attitude episodes (400 updates each, 90 successes) the verdict, revolution, hold counter and inverted flag are identical at every update.

Log: `Logs/success_detector/ppo_track_copy_checks_log.txt`. The §04 detector (`success_detector.py`) is unchanged.

## 2026-09-30 — Harness step 12: every episode is also judged by the PPO track's detector

**Change (`Evaluation/evaluate.py`):**

- her detector is reset and updated at the start state, then once per decision (every 4 simulator steps = 20 ms) and at the last state. It reads the true simulator state, as in her pipeline;
- its verdict is stored in `EpisodeResult.ppo_track`: success (latched), passed_inverted, revolution, t_flip_s, t_recovered_s, and success_then_terminated (latched but the episode still failed; her detector has no crash check, so this is reported separately);
- `EpisodeResult.success` is still the §04 verdict. Nothing else in the episode loop changed.

**Checks (cloud, scripts deleted):**

- 27 episodes (3 controllers × 3 conditions × 3 seeds) are bit-identical to the unchanged harness in every field.
- Against the FINAL run the verdicts, failure modes and terminations are identical. Metrics differ by ≤ 3.3e-7 relative, which is Windows vs Linux floating point: the unchanged harness gives the same Linux values.
- Parity with her published scripted-flip results (seeds 10000–10019, 10 s): 20 % under stress and 100 % under motor noise 0.05. Rotation and, with her metric definition (mean of the last 0.5 s), final tilt and |ω| match her tables.

**Finding:** her conditions keep the simulator's default sensor noise ON, because her `make_flip_env` passes `observation_noise_std=None`. Our `action_noise_005` and `stress` conditions switch it off. Parity holds only with it on (without it: 5/20 instead of 4/20 under stress). So the FINAL noisy conditions were close to hers but not identical; the re-measurement will use her exact conditions.

Log: `Logs/evaluation_harness/step12_ppo_track_detector_checks_log.txt`.

## 2026-09-30 — Harness step 13: choose the detector in the final comparison; the PPO track's exact conditions

**Change:**

- `Evaluation/report.py`: `detector="section04"` (default) or `"ppo_track"` decides whose verdict counts as success. With `ppo_track`, the failure modes and the flip/recovery times follow her detector. The episodes CSV keeps the §04 verdict next to hers, and the table lists the §04 count on the same episodes.
- Her failure modes, in order: crash/region, never inverted (< 150°), rotation incomplete, no 0.4 s hold.
- `Evaluation/final_comparison.py`: `--detector`, and two new conditions that are her definitions exactly:
  - `ppo_track_nominal`: motor noise 0.05 + the simulator's default sensor noise;
  - `ppo_track_stress`: motor 0.15 + gyro 0.2 rad/s + sensor noise.
- Default conditions and detector stay as in FINAL, so the FINAL command still reproduces FINAL.

**Checks (cloud, deleted):**

- the default path writes byte-identical results to the code before this commit; config.json only gains the `detector` key;
- a 2-episode `ppo_track` run gives the expected columns and counts;
- failure modes, metric sources and summary counts were checked on 6 constructed cases, and an unknown detector is rejected.

Log: `Logs/evaluation_harness/step13_detector_option_checks_log.txt`.

## 2026-09-30 — Repository: the components move into the team repository (branch `eval-map-elites`)

**What is included:** `Success_Detector/`, `Controllers/`, `Evaluation/`, `MAP_Elites/`, `Baselines/harness_adapter.py`, the logs of every check and run analysis in `Logs/`, the FINAL results, and this documentation. Nothing of the PPO track is changed.

**Changes made while moving (no change of any verdict or number):**

- `Baselines/harness_adapter.py` loads the PPO track's `Reward_and_Training/baselines.py` directly, by file path, instead of a copy in `Baselines/`. A `baselines.py` elsewhere on `sys.path` (for example in the simulator folder) cannot shadow it.
- `Success_Detector/success_detector.py`: the unused recovery-only diagnostic class, the compatibility alias and the embedded self-tests were removed, and the module docstring now lists all criteria. Checked:
  - the removed self-tests still pass against the cleaned module (7 cases);
  - on 400 random attitude trajectories (61 successes) every `SuccessResult` field and the reason text are identical to before.
- `Success_Detector/README_Section4.md` rewritten: both detectors and their criteria.

**Checks of the whole branch against the previous working copy (cloud, same simulator, no training):**

- `final_comparison.py --detector ppo_track`, 5 controllers (including random and PID hover through the adapter) × nominal and `ppo_track_stress` × 2 seeds: `summary.txt`, `config.json` and every `episodes.csv` / `summary.json` byte-identical;
- `map_elites.py`, 30 evaluations, seed 0: identical archive and generator state, and a history identical except for wall-clock time; `robustness.py` on it: identical candidates, episodes, ranking and summary;
- `analysis.py` figures and `render_episode.py` GIF/PNG are produced.

Log: `Logs/repository_branch_checks_log.txt`.

**Setup note:** the simulator is not part of the repository (`.gitignore`). Clone `upatras-lar/Quadcopter_SimCon` and either set `QUAD_SIM_DIR` to its `Simulation/` folder, as for the PPO track, or place the clone at `Reward_and_Training/Quadcopter_SimCon`, the default path of the harness.

## 2026-09-30 — Planned experiment: final PPO attempt with a GEAR-style tracking reward (pure RL)

**Decision (user):** one last PPO experiment. It is pure RL, with no behaviour cloning and no demonstration starts. It uses the trajectory-tracking reward of GEAR ("Multi-Task Reinforcement Learning of Drone Aerobatics by Exploiting Geometric Symmetries", arXiv 2602.10997), and follows its recipe as closely as this vehicle and our compute allow. The same pipeline is run first with CTBR actions (as in GEAR), then with direct motor commands. New files go in `MTR_PPO/`; no file of the PPO track is changed. Training is started by the user.

**GEAR recipe, as implemented:**

- **Reference.** Maneuvers are defined by body-frame invariants. Flip: loop centre at r along the thrust axis, velocity [ωr, 0, 0], body rates [0, ω, 0]. Hover: the start position, at rest.
- **Task sequence.** Our task is one flip and then hover, so the command switches from FLIP to HOVER when the geometric revolution reaches 360° − 8.6°. Because the switch is triggered by completing the turn rather than by time, a vehicle that only hovers stays in FLIP and earns almost nothing.
- **Relative state.** p_rel = Rᵀ(p_des − p), v_rel, ω_rel, R_rel = Rᵀ R_z(ψ_des), all in the body frame.
- **Actor observation.** The relative state (18), the previous action (4), and the command: task one-hot and ω.
- **Reward.** r = r_pos · r_lin · r_ang · r_cmd · r_task, with H(x; k) = 1/(1 + kx) and GEAR's k sets ({1, 10}, {1, 10, 100}, {0.1, 1, 10}, {1, 10}). r_task = 2 for the flip.
- **r_cmd.** Commanded against achieved attribute: pitch rate against ω during the flip, number of turns against 1 afterwards.
- **Episodes.** No crash penalty: termination ends the positive reward.
- **Randomisation.** The command ω is randomised per episode. Initial states are randomised with a range that expands during training (curriculum).

**Deviations, and why:**

- **ω range.** 4.5–5.5 rad/s with r = 0.6 m, instead of GEAR's 4–6 rad/s. With thrust/weight ≈ 3.1, no single radius makes both ends of 4–6 feasible: at ω = 4 the top of the loop needs r ≥ 0.61 m, at ω = 6 the bottom needs r ≤ 0.57 m. Evaluation uses ω = 5.
- **Network and platform.** A plain MLP actor with the PPO track's privileged critic, instead of GEAR's equivariant networks, which serve multi-task generalisation. 50 Hz decisions (action repeat 4) instead of 100 Hz. CPU environments instead of 2048 GPU environments.
- **Reward scaling.** The reward is divided by its maximum and multiplied by the step time, so that returns are on the scale of the PPO track's settings. A constant factor does not change the optimal policy.

**Evaluation:** the universal harness on the nominal benchmark, judged by the PPO track's detector with the §04 verdict alongside, during and after training.

## 2026-09-30 — MTR-PPO step 1: reference (`MTR_PPO/mtr_reference.py`)

**Added:** `FlipCommand` (ω, r, direction) and `LoopReference`. `start()` fixes the start position, ψ_des and the loop centre (r above the start). `relative_state(task, …)` returns GEAR's body-frame p_rel, v_rel, ω_rel and R_rel for FLIP or HOVER.

**Checks (cloud, script deleted):**

- the rotation matrix equals the PPO track's `quat_to_rotation_matrix`;
- an ideal loop has zero FLIP relative state at every point, for 5 headings and both directions;
- integrating the desired velocity over one loop returns to the start (4.5e-16 m), so the invariants are kinematically consistent;
- start and hover states give the expected values; invalid commands are rejected.

**Feasibility on this vehicle (r = 0.6 m):**

| ω [rad/s] | thrust needed at the top | thrust needed at the bottom | loop duration |
|---|---|---|---|
| 4.5 | 0.24 g | 2.24 g | 1.40 s |
| 5.0 | 0.53 g | 2.53 g | 1.26 s |
| 5.5 | 0.85 g | 2.85 g | 1.14 s |

Available thrust is 0.02–3.13 g.

Log: `Logs/MTR_PPO/checks/step01_reference_checks_log.txt`.

## 2026-09-30 — MTR-PPO step 2: reward (`MTR_PPO/mtr_reward.py`)

**Added:** `MultiplicativeTrackingReward`, a per-environment reward hook with the PPO track's signature.

- r = r_pos · r_lin · r_ang · r_cmd · r_task with GEAR's kernels and k sets.
- r_cmd: during FLIP, pitch rate against ω; during HOVER, turns against 1.
- r_task = 2 during FLIP and 1 during HOVER. The switch to HOVER happens at a geometric revolution of 360° − 8.6°, updated every 5 ms.
- Normalised so that perfect flip tracking earns 1 per second and perfect hover 0.5 per second.
- No crash penalty.
- It provides what the PPO track's `FlipTaskEnv` reads from a reward (alpha, phase, passed_inverted, cfg fields, reset), so it can be installed there unchanged.

**Checks (cloud, script deleted):**

- kernel sums are exact, and 500 random states equal an independent recomputation;
- perfect flip = 1/s, perfect hover = 0.5/s;
- the reward decreases strictly with position, velocity and rate errors;
- **hovering while FLIP is commanded earns 8.7e-5/s (0.009 %)**, so doing nothing is not rewarded;
- an ideal loop fed through the hook scores 1.000/s during FLIP, switches to HOVER at 352.4°, and scores 0.5/s at rest afterwards;
- right after the switch, a vehicle still at loop speed scores about 0.002/s, so the policy must brake.

Log: `Logs/MTR_PPO/checks/step02_reward_checks_log.txt`.

## 2026-09-30 — MTR-PPO step 3: observation and training environment (`MTR_PPO/mtr_observation.py`, `mtr_env.py`)

**Added:**

- **`TrackingObsBuilder` (deployable).** The actor observation is 25 values: p_rel, v_rel/5, ω_rel/10, R_rel (9), the previous action (4), the task one-hot [FLIP, HOVER] and ω/5. The task switch uses the PPO track's `FlipProgressTracker`, once per decision.
- **`MTRFlipEnv`.** A subclass of the PPO track's `FlipTaskEnv`, whose code is not changed. It keeps action repeat 4, CTBR or motors, the privileged critic (26 values) and her detector for logging. It installs the multiplicative tracking reward and the tracking actor observation.
- **Randomisation.** ω ~ U[4.5, 5.5] per episode. The initial-state curriculum has scale s ∈ [0, 1]: tilt ≤ 10°·s about a random horizontal axis, rates ≤ 1 rad/s·s, velocity ≤ 0.5 m/s·s. No demonstration starts, no noisy-hover initial steps, no wind.
- **`make_mtr_env(action_mode=...)`.**

**Checks (cloud, script deleted):**

- actor 25 + privileged 26 = 51 observations, and both action modes run;
- the start observation matches the reference, and ω covers [4.50, 5.48] over 40 resets;
- at curriculum 1 the start ranges are respected and used (tilt ≤ 9.87°, rates ≤ 0.999, velocity ≤ 0.500), and reward, detector and trackers restart at the perturbed state;
- the start is applied exactly as the harness applies its perturbed start;
- the env's actor observation equals an independent builder over 110 decisions;
- seeded runs are reproducible;
- motors off ends the episode with no negative reward.

**Real simulator (6 s, informative):**

- PID hover: return 0.001, it never leaves FLIP;
- the PPO track's scripted flip: return 0.20. It switches to HOVER at 1.3 s and succeeds by her detector, but it ends 2.09 m above the start, so the hover term pays only 0.05/s.

Log: `Logs/MTR_PPO/checks/step03_env_checks_log.txt`.

## 2026-09-30 — MTR-PPO step 4: deployed controller (`MTR_PPO/mtr_controller.py`)

**Added:** `MTRController`, which runs a trained policy with the course interface (`reset`, `act -> (action, info)`) in the evaluation harness.

- It loads the model and the run's `config.json`.
- It decides once every `action_repeat` calls and holds the motor command in between.
- It builds the observation with the same `TrackingObsBuilder` as training, for the nominal command (ω = 5 rad/s).
- A CTBR action goes through the same rate loop, once per decision. The privileged critic input is not needed.

**Checks (cloud, untrained policies only, script deleted):**

- for both CTBR and motors, the controller, fed only the raw observations, gives exactly the actions and motor commands of the training environment over a whole episode (300 and 104 decisions);
- it runs in the harness under nominal and the PPO track's stress, producing both verdicts;
- malformed configs are rejected.

Log: `Logs/MTR_PPO/checks/step04_controller_checks_log.txt`.

## 2026-09-30 — MTR-PPO step 5: training script (`MTR_PPO/train_mtr.py`), ready for the user to run

**Added:** pure-RL PPO (Stable-Baselines3) with the PPO track's asymmetric actor-critic on `make_mtr_env` workers.

- **Schedules.** The curriculum scale grows linearly 0 → 1 over the first 50 % of the steps. The entropy coefficient falls 0.005 → 0 and the learning rate 3e-4 → 3e-5.
- **Other settings.** log-std −1.0; the PPO track's pure-RL settings for everything not specified by GEAR.
- **Evaluation during training.** Every 200k steps, 10 nominal 10 s episodes of the deployed controller in the universal harness (ω = 5). Success is judged by the PPO track's detector, with the §04 verdict alongside. `best_model.zip` is the best by (PPO-track success, §04 success, −crash rate).
- **Options.** `--action-mode ctbr|motors`, `--dry-run` (builds everything and evaluates the untrained policy once, without learning), and `--smoke` (a tiny learning run, started by the user).

**Checks (cloud, no learning):**

- `--dry-run` for both action modes with 2 subprocess workers; each run folder holds only config, evaluation log and the untrained model;
- the saved model and config reload into the deployed controller;
- the schedules set the entropy and the curriculum in the workers and log them;
- the evaluation callback writes the CSV and keeps `best_model.zip`;
- out-of-range ω options are rejected.

A small robustness fix was made during the checks: the callbacks read the step count from the model, and tolerate an empty episode buffer before learning starts.

Log: `Logs/MTR_PPO/checks/step05_training_script_checks_log.txt`.

**Planned runs (started by the user):** CTBR seeds 0 and 1, then motors seeds 0 and 1, with 3M steps each.

## 2026-09-30 — Final comparison FINAL_PT (executed by the user): the PPO track's detector and conditions

**Run:** `final_comparison.py --name FINAL_PT --detector ppo_track --seed0 10000 --episodes 50 --workers 12 --conditions nominal ppo_track_nominal ppo_track_stress --controllers map_elites_final three_phase_default scripted_flip`.

- Seeds 10000–10049, which is the PPO track's seed bank. The first 20 seeds are exactly her evaluation episodes.
- 10 s episodes.
- Success by her detector, with the §04 verdict alongside.
- Copies of all results: `Logs/final_comparison/FINAL_PT/`.

| Condition | MAP-Elites | Default three-phase | Scripted flip |
|---|---|---|---|
| nominal (brief, no noise) | 50/50 | 50/50 | 50/50 |
| PPO-track nominal (motor 0.05 + sensor noise) | 50/50 | 50/50 | 50/50 |
| PPO-track stress (motor 0.15 + gyro 0.2 + sensor noise) | 15/50 (30 %) | 14/50 (28 %) | 17/50 (34 %) |

**Parity inside the real run:** on her seeds 10000–10019 the scripted flip scores 4/20 under stress and 20/20 under her nominal. Both equal her published tables. No episode latched success and then crashed.

**§04 on the same episodes:**

- identical under nominal;
- 47–49/50 under her nominal;
- 0/50 under stress, where every stress failure by her detector is "no 0.4 s hold".

The difference between the detectors is the final-hold rule. Under 0.2 rad/s gyro noise the vehicles pass |ω| < 0.8 for some 0.4 s window, but not at the end of the episode.

**Effort and timing (her nominal):** MAP-Elites 0.81, against 1.22 (default) and 2.16 (scripted). MAP-Elites is the slowest to flip and settle (1.52 s / 2.65 s).

## 2026-09-30 — MTR-PPO run 1 (executed by the user): CTBR, seed 0, all episodes start at hover. Stopped at ~1.1M steps: no learning signal

**Run:** `MTR_PPO/train_mtr.py --action-mode ctbr --seed 0` (defaults: 3M steps, 11 workers). It was stopped at about 1.1M steps, on the evidence below.

**Evaluation (10 nominal episodes, every 200k steps):** 0 % success with both detectors, 0 % inverted, 100 % terminated at 200k, 400k, 600k, 800k and 1M steps.

**Training curves (TensorBoard):**

- mean episode return 0.0002–0.0004 throughout; a flip followed by hover is worth about 3.5;
- every training episode terminates after 2–4 s;
- critic explained variance about −2 and value loss 3e-5, so there is nothing to predict.

**Behaviour of the 900k checkpoint (cloud replay, no training):** the policy climbs at full thrust while tilting to about 70° the wrong way. It leaves the altitude region upward (z = −5 m) after about 1.1 s. The best instantaneous reward found was 0.5 % of the maximum.

**Diagnosis.** The multiplicative reward is a narrow peak. From hover it gives 0.009 % of the maximum; at 50 % of the loop's speed and rate it still gives only 0.23 %, and it becomes noticeable (2.7 %, then 19 %) only at 75–90 %. Random exploration from hover never gets close enough, so the gradient is effectively zero. With no crash penalty (as in GEAR), ending the episode early costs nothing when the reward is already about zero, so leaving the region is not discouraged. This is not a code error: it is the hover-start value measured in step 2 (8.7e-5/s).

Evidence: `Logs/MTR_PPO/runs/run1_ctbr_seed0_hover_starts/`, containing `eval_log.csv`, `config.json` and `training_curves.txt`.

## 2026-09-30 — MTR-PPO step 6: starts on the reference (reference state initialisation)

**Decision (user):** GEAR need not be followed exactly. One final attempt adds random initialisation on the reference, with a longer run than 1M steps.

**Change.**

- New option `--ref-start-prob p`, default 0. A fraction p of the training episodes starts at a random loop phase θ ~ U[0, 360° − 8.6°), in exactly the reference state: position on the circle, attitude, velocity ωr and body rates ω.
- The loop is anchored at the nominal start position, which is also the hover target.
- The completed phase θ counts as revolution already made, so the switch to HOVER comes when the loop closes.
- The remaining episodes start at hover as before. No expert actions are used, so this is not behaviour cloning.
- Evaluation still starts at hover, so learning the entry into the loop from hover is what the evaluation tests.
- Code: `LoopReference.anchor()` and `state_at(θ)`; `begin(..., rev0, anchor)` in the reward; `reset(..., anchor, rev0)` in the observation builder; the start option in `mtr_env.py` and `train_tracking.py`.

**Checks (cloud, script deleted):**

- with p = 0, observations, rewards and commands are bit-identical to the previous version, so run 1 stays reproducible;
- `state_at(θ)` lies exactly on the reference for 800 random phases, headings and commands, in both directions;
- with p = 1 the reward rate is exactly 1 at the start state, the phase covers 1.5°–348.7°, and the trackers and revolution start at the phase;
- with p = 0.5, 47.5 % of 400 resets start on the reference;
- **learning signal:** under the initial random policy, the mean reward rate over the first 0.2 s is 0.0002/s from hover and 0.099/s from reference starts, about 500 times more;
- the controller parity of step 4 still holds, and invalid p is rejected.

Log: `Logs/MTR_PPO/checks/step06_reference_starts_checks_log.txt`.

**Planned run (user):** `MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --ref-start-prob 0.5 --tag gear_rsi`, 3M steps. Motors and a second seed only if CTBR learns.

## 2026-09-30 — MTR-PPO run 2 (executed by the user): CTBR, seed 0, `--ref-start-prob 0.5`, narrow (GEAR's) kernels. Stopped at ~0.3M steps: the reward is too narrow

**Run:** `MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --ref-start-prob 0.5 --tag gear_rsi`, read at about 0.29M steps and then stopped.

**Training curves:**

- **Return:** the mean episode return rose from 0.0003 (run 1) to about 0.011, about 40 times more, but stayed flat from the first rollouts on.
- **Episodes:** about 95 % still end by leaving the region, after about 3 s.
- **Starts:** half the episodes started on the reference, as intended (`mtr_ref_start` ≈ 0.5).
- **Evaluation at 200k (from hover):** 0 % inverted, 100 % terminated.

**Reading.** Starting on the loop gives reward at the start, but the policy leaves the loop within a few tenths of a second and the reward falls back to about zero. GEAR's k values (up to 100 on velocity and 10 on rate) make every term collapse for small deviations, and the product collapses with them. The user judged the reward too narrow and asked for a much denser one.

Evidence: `Logs/MTR_PPO/runs/run2_ctbr_seed0_ref_starts_narrow_kernels/`.

## 2026-09-30 — MTR-PPO step 7: wider ("dense") reward kernels (`--kernels dense`)

**Decision (user):** make the reward much denser, keeping its structure.

**Change.**

- The kernel k values become a setting: `MTRRewardConfig.kernels` and the CLI option `--kernels narrow|dense`. The default `narrow` is GEAR's set, so runs 1 and 2 stay reproducible.
- `dense` keeps the multiplicative GEAR structure and the same terms, with one coarse and one fine kernel per term. The coarse kernel has half value at the typical error of a hover start:
  - r_lin k ∈ {0.1, 1}, half value at about 3.2 and 1 m/s;
  - r_ang and r_cmd k ∈ {0.04, 0.25}, half value at 5 and 2 rad/s;
  - r_pos unchanged at {1, 10}.

**Reward rate against closeness to the loop** (fraction of the loop's speed and rate reached):

| Kernels | 0 % | 25 % | 50 % | 75 % | 90 % | 100 % |
|---|---|---|---|---|---|---|
| narrow | 0.01 % | 0.04 % | 0.23 % | 2.7 % | 19 % | 100 % |
| dense | 3.2 % | 7.7 % | 19.9 % | 54.7 % | 89.0 % | 100 % |

Doing nothing during FLIP still earns little: 0.032/s, about 0.19 per 6 s episode, against about 3.4 for a flip followed by hover. Staying airborne now pays, since the reward is never close to zero while the vehicle flies, so leaving the region has a cost.

**Checks (cloud, script deleted):**

- `narrow` is bit-identical to the previous version, with and without reference starts;
- `dense` equals an independent recomputation on 500 random states, reproduces the table above, and gives perfect hover = 0.5/s;
- unknown kernel sets are rejected.

**Learning signal:** under the initial random policy, the mean reward rate over the first 0.2 s is:

| Kernels | hover starts | reference starts |
|---|---|---|
| narrow | 0.0004/s | 0.11/s |
| dense | 0.039/s | 0.29/s |

Log: `Logs/MTR_PPO/checks/step07_dense_kernels_checks_log.txt`.

**Planned run (user):** `MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --ref-start-prob 0.5 --kernels dense --tag gear_dense`, 3M steps.

## 2026-09-30 — MTR-PPO run 3 (executed by the user): CTBR, seed 0, `--ref-start-prob 0.5`, `--kernels dense`. The policy exploits the reward

**Run:** `MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --ref-start-prob 0.5 --kernels dense --tag gear_dense`, read at about 2.3M of 3M steps; it then ran on to about 2.95M, with the same behaviour (return 2.55, harness evaluation 0 % success up to 2.8M).

**Training curves (mean per 0.2M steps):**

| steps | 0–0.2M | 0.8M | 1.2M | 1.6M | 2.0M | 2.2M+ |
|---|---|---|---|---|---|---|
| episode return | 0.06 | 0.51 | 1.31 | 1.94 | 2.30 | 2.40 |
| crashed | 0.95 | 0.02 | 0 | 0 | 0 | 0 |
| passed inverted | 0.44 | 0.39 | 0.34 | 0.31 | 0.34 | 0.30 |
| flip success | 0 | 0 | 0 | 0 | 0 | 0 |

The dense kernels fixed the learning signal: crashes went to zero, the explained variance rose to about 0.96, and the return kept rising. The fraction of episodes that pass inverted fell instead of rising, and the harness evaluation (10 nominal episodes from hover) gave 0 % success at every point up to 2.8M.

**Replay of the 2.1M checkpoint (cloud, deterministic):** from hover the policy pitches forward at about +5 rad/s, as the reference asks, through the lower half of the loop (about −100° to +90°), where the reward rate is 55–80 % of the maximum. Near +90° it snaps back at −12 to −21 rad/s, losing reward for only about 0.15 s, and repeats about five times per 6 s episode. None of 4 hover starts passes 180°. Reference starts above about 150° complete the loop and hover at the start position; starts at 65–100° swing back.

**Reading.** The FLIP reference is parametrised by the vehicle's attitude, not by time: every attitude has a point on the loop, and being there with the loop's velocity and rate earns the same reward whether the vehicle is advancing or has turned back. Nothing rewards progress, so the easy lower half pays about 0.4/s indefinitely, and the policy settled there. Its action spread (std) fell from 0.36 to 0.14, so the remaining steps would only entrench it.

Evidence: `Logs/MTR_PPO/runs/run3_ctbr_seed0_ref_starts_dense_kernels/` (eval log, config, training curves, policy replay).

## 2026-09-30 — MTR-PPO step 8: termination on turning back (`--backtrack-deg`)

**Decision (user):** end an episode that turns back during the flip (option 1 of two; the alternative, a time-driven reference, was judged likelier to make the reward sparse again).

**Change.**

- `MultiplicativeTrackingReward` keeps the largest revolution reached in the episode (`rev_max`, starting at the start phase).
- `MTRFlipEnv(backtrack_deg=…)` and the CLI option `--backtrack-deg`: during FLIP, the episode ends when the revolution falls more than `backtrack_deg` below `rev_max`. The reward is positive and there is no penalty; ending the episode forfeits the rest of it, so swinging back stops paying. It never applies during HOVER. The step info carries `mtr_backtrack` (logged as `train_flip/mtr_backtrack`).
- Default 0 = off, so runs 1–3 stay reproducible. The evaluation harness and the deployed controller are unchanged.

**Checks (cloud, scripts deleted):**

- with the option off (0 or default), rewards, observations and terminations are bit-identical to the previous version, from hover and from reference starts;
- the run-3 policy with `--backtrack-deg 30`: all 4 hover starts and the reference starts at 66–103° end by backtracking after 0.2–0.3 s (return 0.02–0.10 instead of about 2.4); the start at 148.5° completes the loop and hovers, unchanged (return 2.486); over 40 reference starts, 22 complete the loop and no episode ends by this rule during HOVER;
- under random actions, 87 % of hover-start episodes end by backtracking (mean length 1.4 s instead of 3.0 s), and the reward rate over the first 0.2 s is unchanged;
- invalid values are rejected by the environment and the CLI; a trainer dry run (no learning) writes the setting to `config.json`.

Log: `Logs/MTR_PPO/checks/step08_backtrack_termination_checks_log.txt`.

**Risk:** a policy that never starts the flip is not terminated and earns about 0.19 per episode, more than a forward swing that is cut off (0.02–0.04). The reference starts, where completing the loop earns about 2.5, are what must pull the policy over the top. Watch `train_flip/flip_inverted` in the first 0.6–0.8M steps.

**Planned run (user):** `MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --ref-start-prob 0.5 --kernels dense --backtrack-deg 30 --tag gear_bt`, 3M steps.

## 2026-10-01 — MTR-PPO run 4 (executed by the user): CTBR, seed 0, `--ref-start-prob 0.5`, `--kernels dense`, `--backtrack-deg 30`. The flip is learned, the recovery is not

**Run:** `MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --ref-start-prob 0.5 --kernels dense --backtrack-deg 30 --tag gear_bt`, read at about 1.37M of 3M steps.

**Training curves (mean per 0.1M steps):**

| steps | 0.1M | 0.3M | 0.5M | 0.8M | 1.1M | 1.3M+ |
|---|---|---|---|---|---|---|
| passed inverted | 0.48 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 |
| loop completed (`mtr_task`) | 0.22 | 0.47 | 0.95 | 0.98 | 0.99 | 1.00 |
| ended by backtracking | 0.55 | 0.01 | 0 | ~0 | ~0 | 0 |
| crashed | 0.45 | 0.99 | 0.99 | 0.98 | 0.97 | 0.96 |
| episode length (s) | 1.1 | 1.5 | 2.0 | 2.8 | 3.1 | 3.2 |

The swing of run 3 disappeared within 0.3M steps. From then on nearly every episode completes the loop, and nearly every episode then leaves the region. The harness evaluation (10 nominal episodes from hover) gave 100 % inverted, 0 % success and 100 % crashes at every point up to 1.4M.

**Replay of the 1.2M checkpoint (cloud, deterministic):**

- **The flip works.** From hover the policy flies a full loop in 1.1–1.2 s (the reference at 5 rad/s takes about 1.26 s), climbing about 1.2 m and passing inverted at about 12 rad/s.
- **The recovery fails.** At the end of the loop it still turns at 3–4 rad/s and moves at about 3 m/s. It does not brake, keeps rotating (up to about 1.4 turns), loses height, and leaves the region 5 m below about 2 s after the flip.
- The 1.5M checkpoint behaves the same.

**Reading.** After the flip the HOVER reward is tiny while the vehicle is far and fast (0.3–3 % of the maximum), and there is no crash penalty, so crashing costs almost nothing. The episode length grows only slowly (2.8 → 3.2 s over the last 0.5M steps).

Evidence: `Logs/MTR_PPO/runs/run4_ctbr_seed0_backtrack30/` (eval log, config, training curves, policy replay).

## 2026-10-01 — MTR-PPO step 9: post-flip survival bonus (`--alive-bonus`) and fine-tuning start (`--init-model`)

**Decision (user):** reward staying airborne after the flip, and continue from the run-4 policy, which already flips, instead of starting again.

**Change.**

- **Survival bonus.** `MTRRewardConfig.b_alive_hover`, `make_mtr_env(alive_bonus=…)` and the CLI option `--alive-bonus b`. During HOVER only, i.e. once the loop is complete, every simulator step also earns b · u · dt, with u = (1 + cos tilt)/2 (1 upright, 0 inverted). With b = 0.5 a perfect hover earns 1/s instead of 0.5/s, and flying upright but off target still earns about 0.5/s instead of about 0.02/s. The bonus only exists after the flip, so it cannot reward not flipping (a crash penalty could).
- **Fine-tuning start.** CLI option `--init-model path`: the trainer starts from a saved model instead of a fresh network.
  - The network, its action spread and the observation layout come from the file.
  - The PPO settings, schedules, number of workers and environment come from the command.
  - A model from a run with a different action mode or observation is rejected (read from the `config.json` next to the file).
  - Use `--curriculum-frac 0` to start with the full initial-state randomisation.
- **Defaults** 0 and none, so runs 1–4 stay reproducible. The evaluation harness and the deployed controller are unchanged.

**Checks (cloud, scripts deleted):**

- with the bonus off, rewards, observations and terminations are bit-identical to step 8;
- the bonus equals an independent recomputation from the quaternion at every simulator step (max error 3.5e-18); it is exactly 0 during FLIP, and the dynamics are unchanged;
- the run-4 policy's returns from hover rise from 0.08–0.14 to 0.57–0.65, against about 6 for a flip followed by perfect hover and about 0.19 for hovering without flipping;
- a model loaded with `--init-model` has the file's weights and actions and this command's PPO settings;
- a trainer dry run with the run-4 checkpoint (no learning) reproduces run 4's own evaluation at 1.2M exactly;
- invalid values, a missing file and a mismatched action mode are rejected.

Log: `Logs/MTR_PPO/checks/step09_alive_bonus_init_model_checks_log.txt`.

**Planned run (user): run 5, second stage.** From the run-4 checkpoint at 1.5M (the curriculum was complete there), 1M steps, bonus 0.5/s, full randomisation from the start, learning rate continuing from where run 4 was (about 1.5e-4 → 3e-5):

`MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --ref-start-prob 0.5 --kernels dense --backtrack-deg 30 --alive-bonus 0.5 --init-model <run 4>/checkpoints/ppo_gear_1499960_steps.zip --curriculum-frac 0 --timesteps 1000000 --lr 1.5e-4 --tag gear_ft`


## 2026-10-01 — MTR-PPO run 5 (executed by the user): fine-tuning from run 4 with the survival bonus. Flip and survival learned; the vehicle does not stop

**Run:** `MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --ref-start-prob 0.5 --kernels dense --backtrack-deg 30 --alive-bonus 0.5 --init-model <run 4>/checkpoints/ppo_gear_1499960_steps.zip --curriculum-frac 0 --timesteps 1000000 --lr 1.5e-4 --tag gear_ft`, complete (1M steps, about 47 min).

**Training curves (mean per 0.1M steps):**

| steps | 0.1M | 0.3M | 0.5M | 0.7M | 1.0M |
|---|---|---|---|---|---|
| crashed | 0.96 | 0.51 | 0.09 | 0.05 | 0 |
| episode length (s) | 3.3 | 5.2 | 5.9 | 6.0 | 6.0 |
| passed inverted | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| ended by backtracking | ~0 | 0 | 0 | 0 | 0 |
| episode return | 1.0 | 2.0 | 2.4 | 2.6 | 2.76 |

**Harness evaluation** (10 nominal episodes from hover, 10 s): crashes fell from 100 % (0.2M, 0.4M) to 0 % (from 0.6M on); at the end the final tilt is 2–5° and the final rate 0.02–0.05 rad/s. Success is 0 % with both detectors at every point.

**Why it fails (final model in the harness, cloud):** the flip is clean (360° by 0.9 s). The vehicle then levels out, but keeps flying horizontally at about 3.5 m/s and ends about 25 m away after 10 s. The PPO-track detector needs |v| < 0.4 m/s for 0.4 s, so it never confirms the recovery. The §04 detector also counts the wobble of 1.5–4 s (tilt up to 40°) towards its rotation budget. All 10 nominal episodes are identical (deterministic policy, nominal start).

**Reading.** The survival bonus of step 9 pays for being upright, not for stopping, and training episodes end only on leaving the region vertically. Flying away upright earns almost the full bonus (0.5/s), while the HOVER tracking term is tiny far from the start.

Evidence: `Logs/MTR_PPO/runs/run5_ctbr_seed0_finetune_alive_bonus/` (eval log, config, training curves, harness trace).

## 2026-10-01 — MTR-PPO step 10: speed-weighted survival bonus (`--alive-speed-k`)

**Decision (user):** make the survival bonus pay for stopping.

**Change.** `MTRRewardConfig.alive_speed_k`, `make_mtr_env(alive_speed_k=…)` and the CLI option `--alive-speed-k k`. The bonus of step 9 is also multiplied by 1/(1 + k|v|²). The weighting is smooth, so the reward does not become sparse. With k = 1:

| speed | share of the bonus |
|---|---|
| 0.4 m/s (detector limit) | 86 % |
| 1 m/s | 50 % |
| 3.5 m/s (run-5 drift) | 7.5 % |

Default 0, so runs 1–5 stay reproducible. The evaluation harness and the deployed controller are unchanged.

**Checks (cloud, scripts deleted):**

- with k = 0 (or the default), rewards, observations and terminations are bit-identical to step 9;
- with k = 1 the bonus equals an independent recomputation at every simulator step (max error 1.7e-18), is exactly 0 during FLIP, and the dynamics are unchanged;
- HOVER reward rate, upright, k = 0 → 1: at the start position at rest 1.00 → 1.00/s; 1 m away at 1 m/s 0.60 → 0.35/s; 10 m away at 3.5 m/s 0.50 → 0.04/s;
- the run-5 policy's return per 6 s episode falls from 2.6–2.7 to 0.49–0.57 (it drifts), against about 6 for a flip followed by a stop near the start;
- invalid values are rejected; a trainer dry run (no learning) with the run-5 model writes the setting to `config.json`.

Log: `Logs/MTR_PPO/checks/step10_alive_speed_weighting_checks_log.txt`.

**Planned run (user): run 6, third stage.** From the run-5 final model, bonus 0.5/s with k = 1, full randomisation, 1M steps:

`MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --ref-start-prob 0.5 --kernels dense --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1 --init-model <run 5>/final_model.zip --curriculum-frac 0 --timesteps 1000000 --lr 1e-4 --tag gear_ft2`

## 2026-10-01 — MTR-PPO run 6 (executed by the user): fine-tuning from run 5 with the speed-weighted bonus. The flip and the recovery are learned

**Run:** `MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --ref-start-prob 0.5 --kernels dense --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1 --init-model <run 5>/final_model.zip --curriculum-frac 0 --timesteps 1000000 --lr 1e-4 --tag gear_ft2`, complete (1M steps).

**Training:** no crash and no backtracking throughout, every episode completes the loop; the return rose from 0.73 to 4.25 (a flip followed by perfect hover earns about 6). The harness evaluation during training (10 nominal episodes) gave PPO-track success 100 % from the first point (0.2M) on; §04 success only at 0.6M (`best_model.zip`).

**Harness evaluation of the two models** (cloud; same seeds 10000–10049, conditions and detector as `FINAL_PT`; 50 episodes each):

| condition | final (1.0M): PPO-track | §04 | best (0.6M): PPO-track | §04 | best of FINAL_PT (PPO-track) |
|---|---|---|---|---|---|
| nominal | 50/50 | 0/50 | 50/50 | 50/50 | 50/50 (all three) |
| ppo_track_nominal | 50/50 | 0/50 | 50/50 | 42/50 | 50/50 (all three) |
| ppo_track_stress | 36/50 [0.58, 0.83] | 0/50 | 34/50 [0.54, 0.79] | 0/50 | 17/50 (scripted_flip) |

Final model, nominal: flip by 0.82 s, recovered by 2.16 s, altitude loss 0.06 m. Under stress the failures are all "no 0.4 s hold" (14/50); there are no crashes.

**Why the final model fails §04:** it passes inverted at about 169.8° (min g_z = −0.984), just short of §04's 170°, so §04 never registers the inversion and its other checks follow from that. The best model reaches about 172° (min g_z = −0.991) and passes §04 in nominal. The reported definition is the PPO-track detector (150°).

Evidence: `Logs/MTR_PPO/runs/run6_ctbr_seed0_finetune_speed_weighted/` (eval log, config, training curves, §04 vs PPO-track on one nominal episode, `harness_FINAL_PT_conditions/`).

## 2026-10-01 — MTR-PPO step 11: the method is named MTR-PPO, gets its own folder, and joins the final comparison

**Decision (user):** name the method MTR-PPO (Multiplicative Tracking Reward PPO), state that the reward structure is adapted from GEAR, give the method an organised folder, and make the trained policy a controller of the final comparison.

**Change.**

- **Folder.** `Reward_and_Training/gear_tracking/` becomes `MTR_PPO/` at the repository root, next to `MAP_Elites/`. Modules: `mtr_reference.py`, `mtr_reward.py`, `mtr_observation.py`, `mtr_env.py`, `mtr_controller.py`, `train_mtr.py`, plus a README. The PPO track's modules are still imported from `Reward_and_Training/` and are unchanged.
- **Names.** Classes, kernel set (`gear` → `narrow`), info and TensorBoard keys (`gear_*` → `mtr_*`), default tag (`mtr`) and checkpoint prefix (`ppo_mtr_*`). The entries above use the current names. Runs 1–6 were executed under the earlier names, and their commands keep the tags they ran with. The mapping is in `Logs/MTR_PPO/README.md`.
- **Model.** The run-6 final model is `MTR_PPO/models/ctbr_seed0/final_model.zip`, with its run configuration and a README of the three training stages.
- **Controller.** `Evaluation/final_comparison.py` accepts `--controllers ppo_mtr`. The default controller set is unchanged, so `FINAL` and `FINAL_PT` rerun as before.
- **Logs.** `Logs/gear_tracking/` becomes `Logs/MTR_PPO/`, with `checks/` (step01–step11) and `runs/` (run1–run6). Absolute local paths in the run configurations are shortened to `<repo>`.

**Checks (cloud, scripts deleted):**

- the renamed environment is bit-identical to the previous code (rewards, observations, terminations, info values) for CTBR with narrow kernels, dense kernels with reference starts, the final recipe, and motors with the final recipe;
- `ppo_mtr` in the final comparison reproduces the earlier evaluation of the run-6 model episode by episode (150/150 episodes identical);
- trainer dry runs (no learning) work in CTBR and motors mode; a CTBR model is rejected as the start of a motors run;
- the reference checks of step 1 run on the renamed module.

Log: `Logs/MTR_PPO/checks/step11_rename_checks_log.txt`.

**Result, `FINAL_PT_MTR`** (same seeds, conditions and detector as `FINAL_PT`; `Logs/final_comparison/FINAL_PT_MTR/`):

| condition | ppo_mtr | MAP-Elites | three-phase default | scripted flip |
|---|---|---|---|---|
| nominal | 50/50 | 50/50 | 50/50 | 50/50 |
| ppo_track_nominal | 50/50 | 50/50 | 50/50 | 50/50 |
| ppo_track_stress | **36/50** [0.58, 0.83] | 15/50 | 14/50 | 17/50 |

ppo_mtr in nominal: flip by 0.82 s, recovered by 2.16 s, altitude loss 0.06 m. Under stress its 14 failures are all "no 0.4 s hold", with no crashes. The §04 verdict is 0/50 in every condition, because the policy passes inverted at about 169.8°, just short of §04's 170° (step 11 does not change this; see run 6).

**Next (user):** the motors run with the final recipe from a fresh network, for the comparison of action spaces:

`python MTR_PPO/train_mtr.py --action-mode motors --seed 0 --kernels dense --ref-start-prob 0.5 --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1 --timesteps 3000000 --tag mtr_motors`

## 2026-10-01 — MTR-PPO run 7 (executed by the user): direct motor commands, final recipe from a fresh network. The flip is learned, with roll

**Run:** `python MTR_PPO/train_mtr.py --action-mode motors --seed 0 --kernels dense --ref-start-prob 0.5 --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1 --timesteps 3000000 --tag mtr_motors`, read at about 1.26M steps.

**Training curves (mean per 0.1M steps):**

| steps | 0.1M | 0.3M | 0.4M | 0.6M | 1.0M | 1.2M+ |
|---|---|---|---|---|---|---|
| crashed | 0.90 | 0.90 | 0.21 | 0 | 0 | 0 |
| episode length (s) | 1.2 | 4.1 | 5.7 | 6.0 | 6.0 | 6.0 |
| passed inverted | 0.83 | 0.99 | 1.0 | 1.0 | 1.0 | 1.0 |
| episode return | 0.08 | 0.49 | 1.35 | 3.82 | 4.98 | 5.08 |

**Harness evaluation during training** (10 nominal episodes): all crash at 0.2M. From 0.4M on, the primary (PPO-track) detector gives 100 %, and the final tilt and rate fall to 0.001° and 0.0003 rad/s at 1.2M. The strict (§04) detector gives 0 %.

With the final recipe from the start and no demonstrations, the policy learns the flip and the recovery with direct motor commands in about 0.4M steps. The ablation of the PPO track with motor commands (its own reward) gave 0 %.

**Why the strict detector fails (1.2M checkpoint, one nominal episode, cloud):**

- The primary detector passes it: flip by 0.52 s, recovered by 1.04 s, altitude loss 0.11 m.
- The strict detector reports "rotation incomplete (325°)". The policy rolls during the flip (|ω_x| up to 9.8 rad/s, ∫ω_x ≈ −154°) and ends with its heading turned by 32°.
- Gravity still sweeps a full circle in the body x-z plane, so the geometric revolution is 360°. The pitch component of the quaternion increments, which the strict detector counts, is only 325°.
- The brief asks for a rotation about the pitch axis. Nothing in the reward asks the rotation to stay about one axis: the reference is defined by the thrust direction, and the rate term only weighs the roll rate among the three rate errors.

## 2026-10-01 — MTR-PPO step 12: pitch-axis reward term (`--axis-term`)

**Decision (user):** discourage the roll and heading change of the motors policy with a quick fix, and continue from its checkpoint.

**Change.** In a pure pitch flip the body y-axis (the pitch axis) stays fixed in the world, along the normal of the flip plane, y_des = R_z(ψ_des) e_y. The option `--axis-term` (`MTRRewardConfig.axis_term`, `make_mtr_env(axis_term=…)`) multiplies the reward product by r_axis = Σ_{k∈{1,10}} H(‖R e_y − y_des‖²; k), with ‖R e_y − y_des‖² = 2(1 − R_rel[1,1]).

- It applies in FLIP and in HOVER, so rolling out of the flip plane and turning the heading both cost reward.
- The coarse kernel is at half value at 60° of axis deviation, the fine one at 18°.
- The maximum product includes the term's maximum (2), so a perfect flip still earns 1/s.
- It is the same kind of term as GEAR's: a body-frame invariant of the manoeuvre in a kernel factor.
- Default off, so runs 1–7 stay reproducible.

**Checks (cloud, scripts deleted):**

- off, it is bit-identical to step 11;
- on the ideal loop the axis error is 0 and the reward rate exactly 1/s, for 4 headings and both directions;
- the axis error equals an independent computation on 500 random attitudes;
- the run-7 policy deviates by 48–58° from the pitch axis during its flips, and its return per episode falls from 4.8–5.2 to 3.8–3.95 with the term on;
- a trainer dry run (no learning) loads the run-7 checkpoint with the term on.

Log: `Logs/MTR_PPO/checks/step12_axis_term_checks_log.txt`.

**Planned run (user): run 8.** Continue run 7 from its latest checkpoint with `--axis-term`, full randomisation and a lower learning rate, 1M steps:

`python MTR_PPO/train_mtr.py --action-mode motors --seed 0 --kernels dense --ref-start-prob 0.5 --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1 --axis-term --init-model <run 7>/checkpoints/ppo_mtr_1799952_steps.zip --curriculum-frac 0 --timesteps 1000000 --lr 1e-4 --tag mtr_motors_axis`

## 2026-10-01 — MTR-PPO run 8 (executed by the user): motors, fine-tuning from run 7 with the pitch-axis term. Heading kept, roll remains; reported as the motors model

**Run:** the planned command of step 12, from the run-7 checkpoint at 1.8M (run 7 was stopped at about 2.0M steps), 1M steps, 0.76 h on the user's machine.

**Training curves:** the episode return rises from 4.23 to 5.22, the evaluation (10 nominal episodes every 0.2M) gives 100 % by the primary detector at every point with a final tilt below 0.003°, and the exploration std falls from 0.168 to 0.149.

**Harness evaluation of the final model** (`Evaluation/final_comparison.py --name FINAL_PT_MTR_MOTORS --detector ppo_track --seed0 10000 --conditions nominal ppo_track_nominal ppo_track_stress --controllers ppo_mtr_motors`, 50 episodes per condition, primary detector):

| condition | success | 95 % CI | alt loss | t_flip | t_rec | effort |
|---|---|---|---|---|---|---|
| nominal | 50/50 | [0.93, 1.00] | 0.20 m | 0.46 s | 0.62 s | 0.82 |
| ppo_track_nominal | 50/50 | [0.93, 1.00] | 0.21 m | 0.45 s | 0.63 s | 1.09 |
| ppo_track_stress | 49/50 | [0.90, 1.00] | 0.23 m | 0.45 s | 2.15 s | 6.02 |

The one failure under stress has no 0.4 s hold. Against the CTBR model (FINAL_PT_MTR): stress 49/50 against 36/50, flip 0.46 s against 0.82 s and recovery 0.62 s against 2.16 s in nominal, but an altitude loss of 0.20 m against 0.06 m. Under stress the control effort rises from 0.82 to 6.02.

**Off-axis rotation (cloud check, one nominal episode, script deleted):** the axis term removed the heading change (yaw at the end −0.2° against 32° in run 7), but not the roll: over the flip ∫ω_x = −165° and ∫ω_y = 320°, a rotation about an axis about 27° from the pitch axis, with the body y-axis up to 55° from its start direction. The same check shows that the CTBR model (run 6, trained without the axis term) rotates about the pitch axis (∫ω_x = 2°) but turns its heading during the flip (∫ω_z = 112°, heading 133.5° at the end). Log: `Logs/MTR_PPO/runs/run8_motors_seed0_axis_term/off_axis_rotation_check.txt`.

**Decision (user):** keep the run-8 final model as the reported motors model and report the roll explicitly.

**Added:**

- `MTR_PPO/models/motors_seed0/` (`final_model.zip`, sha256 prefix 0409d7ed341839c8; `config.json` with paths shortened to `<repo>`; README with the two stages);
- controller `ppo_mtr_motors` in `Evaluation/final_comparison.py` (not in the default set, like `ppo_mtr`);
- `Logs/MTR_PPO/runs/run7_motors_seed0_scratch/` and `run8_motors_seed0_axis_term/` (`config.json`, `eval_log.csv`, `training_curves.txt`; run 8 also the off-axis check);
- `Logs/final_comparison/FINAL_PT_MTR_MOTORS/`.

The evaluation above was first run with a scratch script, then repeated with `final_comparison.py` after adding the controller; both give the same numbers.

## 2026-10-01 — MTR-PPO run 9 (executed by the user): CTBR, fine-tuning from run 6 with the pitch-axis term. Heading partly kept, roll appears, less robust; not adopted

**Run:** `python MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --kernels dense --ref-start-prob 0.5 --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1 --axis-term --init-model MTR_PPO/models/ctbr_seed0/final_model.zip --curriculum-frac 0 --timesteps 1000000 --lr 1e-4 --tag mtr_ctbr_axis`, 1M steps, 0.77 h. The off-axis check of run 8 had shown that the CTBR model turns its heading during the flip (∫ω_z = 112°, heading 133.5° at the end).

**Result.** The episode return rises from 2.78 to 3.06, the evaluation during training stays at 100 % (primary detector) and the exploration std falls from 0.147 to 0.117. Final model, one nominal episode: ∫ω_x = −38° (run 6: 2°), ∫ω_z = 92° (112°), heading at the end 99.4° (133.5°), largest axis deviation 97° (128°). Harness evaluation (scratch script, the conditions of FINAL_PT_MTR, 50 episodes, primary detector): nominal 50/50, ppo_track_nominal 50/50, ppo_track_stress 29/50 [0.44, 0.71] against 36/50 for run 6; recovery 1.42 s against 2.16 s in nominal.

**Reading.** As with the motors in run 8, the soft term redistributes the deviation between heading and roll instead of removing it, and robustness under stress drops.

**Outcome:** run 6 stays the reported CTBR model; run 9 is reported as a negative result. The user moved on to a last attempt with the motors policy (step 13).

## 2026-10-01 — MTR-PPO step 13: one-off pitch-axis penalty (`--axis-limit-deg`, `--axis-penalty`)

**Decision (user):** a last attempt to remove the roll of the motors policy. Ending the episode on a large roll would be the strongest signal, but the brief allows termination only for ground contact, leaving the region, numerical failure and the time limit, and explicitly not for large roll or pitch angles. The change is therefore a reward change only.

**Change.** During FLIP, the first time in an episode that the body y-axis is more than d degrees from the flip plane's normal R_z(ψ_des) e_y (arccos(R_rel[1,1]) > d), the reward of that step drops once by P (`MTRRewardConfig.axis_limit_deg`, `axis_penalty`; `make_mtr_env(axis_limit_deg=…, axis_penalty=…)`; CLI `--axis-limit-deg d --axis-penalty P`, P = 2 by default, d = 0 = off). The deviation is the angle between the two axes, so roll and heading count together and a pure pitch rotation costs nothing; unlike the soft axis term, the policy cannot trade one for the other. A one-off penalty was preferred to withholding the rest of the episode's reward, because the critic cannot observe such a latch, while a penalty appears in the step where it happens. P = 2 is about the difference between the run-8 return (5.3) and a flip that loses half of it; a flip with roll still pays more than no flip. The step info carries `mtr_axis_violation` (logged as `train_flip/mtr_axis_violation`).

**Checks (cloud, scripts deleted):** off, it is bit-identical to step 12; the deviation equals an independent computation on 500 random attitudes, and single-axis rotations give 0 for pitch and the rotation angle for roll and yaw; the run-8 policy is penalised exactly once, at 0.22 s, from hover starts with d = 40° (return 5.35 → 3.35) and never from reference starts or with d = 60°; a trainer dry run loads the run-8 model with the options on. Log: `Logs/MTR_PPO/checks/step13_axis_penalty_checks_log.txt`.

**Planned run (started by Claude on the user's machine at the user's request): run 10.** Continue the reported motors model with the penalty at 40°, 1M steps:

`python MTR_PPO/train_mtr.py --action-mode motors --seed 0 --kernels dense --ref-start-prob 0.5 --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1 --axis-term --axis-limit-deg 40 --axis-penalty 2 --init-model MTR_PPO/models/motors_seed0/final_model.zip --curriculum-frac 0 --timesteps 1000000 --lr 1e-4 --tag mtr_motors_axislim`

## 2026-10-03 — MAP-Elites: the primary detector as validity (`--validity`), and the LHS study re-judged (`descriptor_sampling.py`)

**Decision (user):** the report describes only the primary (PPO-track) detector, so the MAP-Elites measurements (Latin-hypercube study, archives, screening) are repeated with it as the validity check. ME1–ME3, the seed replicates and R1 keep their §04 results; the new runs get new names.

**Change:**

- `problem.py`: `is_valid(result, validity)` and `evaluate_genome(..., validity=...)` with
  - `section04` (default, unchanged behaviour);
  - `ppo_track`: the primary detector's latched success, and the episode did not end in a crash or region exit (the crash loophole stays closed);
  - `--single-turn-check` (with `ppo_track` only): additionally a check that the vehicle turned only once, i.e. the primary detector's final revolution is at most 390° (the §04 limit). The primary detector does not count the turns, so without the check a double flip is valid. It is a flag rather than a third detector because it only adds this one test to the primary verdict.
- `map_elites.py`: `--validity` for a new run, stored in `config.json` and fixed on `--resume`; old `config.json` files without the field load as `section04`.
- `robustness.py`: `--detector ppo_track` judges the screening with the primary detector (counted as in `final_comparison.py`); a §04 screening writes the same `config.json` as before.
- New `descriptor_sampling.py`: re-runs the 401 genomes of the original study (read from `Logs/map_elites/descriptor_sampling/samples.json`) with a chosen validity and writes samples, percentiles, convergence, Spearman correlation and the descriptor figure.

**Checks** (log: `Logs/map_elites/primary_validity_checks_log.txt`): the §04 verdicts of the first 21 genomes reproduce the original study except one, which the documented 2026-09-29 travel-window change explains; with the primary detector 46/60 LHS genomes are valid (§04 at the time: about a quarter), 2 of them double flips (720°), which the single-turn check rejects; resume, refusal of a changed validity and old configs work; the screening writes primary verdicts.

**Decision (user):** primary detector with the single-turn check. **Runs (the user starts them),** the same configurations as before with the suffix `_pt`:

    python MAP_Elites/descriptor_sampling.py --name LHS_pt --validity ppo_track --single-turn-check --workers 12
    python MAP_Elites/map_elites.py --name ME1_pt --validity ppo_track --single-turn-check --seed 0 --budget 5000 --workers 12
    (likewise ME1_pt_s1, ME1_pt_s2 with --seed 1, 2; ME2_pt, ME2_pt_s1, ME2_pt_s2 with --bound t_climb=0,1.0; ME3_pt with --bound a_brake=10,150)
    python MAP_Elites/robustness.py --name R1_pt --detector ppo_track --runs ME1_pt ME2_pt ME3_pt ME1_pt_s1 ME1_pt_s2 ME2_pt_s1 ME2_pt_s2 --top 10 --episodes 10 --workers 12

## 2026-10-02 — MTR-PPO runs 10–12: pitch-axis penalty by fine-tuning the motors model

All three runs continue the previous model with `--axis-term --axis-limit-deg d --axis-penalty 2 --curriculum-frac 0 --lr 1e-4`. Evaluation: harness, 50 episodes, seeds 10000–10049, primary detector (§04 verdict on the same episodes in brackets), conditions of FINAL_PT_MTR. Logs (config, evaluation during training, training curves): `Logs/MTR_PPO/runs/run10_…`, `run11_…`, `run12_…`.

| run | from | d | steps | nominal | ppo_track_nominal | ppo_track_stress |
|---|---|---|---|---|---|---|
| 10 | run 8 final (trained in the cloud workspace) | 40° | 1M | 50/50 (0/50) | 50/50 (0/50) | 48/50 (0/50) |
| 11 | run 10 final | 40° | 1M | 50/50 (0/50) | 50/50 (0/50) | 49/50 (0/50) |
| 12 | run 11 final | 15° | 2M | 50/50 (50/50) | 50/50 (49/50) | 48/50 (0/50) |

**Reading.** The penalty reduces the off-axis rotation only slowly when it is added to a model that already flies an off-axis flip; run 12 is the first fine-tuned motors model that the §04 detector also accepts in nominal conditions. This motivated training from scratch with all components (runs 13–14).

## 2026-10-03 — MTR-PPO runs 13–14: all components from the first step (reported models)

**Runs (executed by the user):**

    python MTR_PPO/train_mtr.py --action-mode ctbr --seed 0 --kernels dense --ref-start-prob 0.5 --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1 --axis-term --axis-limit-deg 40 --axis-penalty 2 --timesteps 3000000 --tag mtr_ctbr_clean
    python MTR_PPO/train_mtr.py --action-mode motors --seed 0 --kernels dense --ref-start-prob 0.5 --backtrack-deg 30 --alive-bonus 0.5 --alive-speed-k 1 --axis-term --axis-limit-deg 40 --axis-penalty 2 --timesteps 3000000 --tag mtr_motors_clean

**Model selection (decision, user).** Both reported models are the final models of their runs, the same rule for both action spaces. Choosing the CTBR checkpoint at 1.5M (no hover oscillation) was considered and rejected as a post-hoc choice; it would not have changed the test success rates (table below). From 1.8M steps on, the CTBR evaluation during training shows a final body rate of about 0.3 rad/s in the noise-free hover; this is reported as a result.

**Evaluation** (harness, 50 episodes, seeds 10000–10049, primary detector; §04 verdict in brackets; cloud evaluation with the run files):

| model | nominal | ppo_track_nominal | ppo_track_stress |
|---|---|---|---|
| run 13 CTBR, checkpoint at 1.5M (not reported) | 50/50 (50/50) | 50/50 (50/50) | 37/50 (0/50) |
| run 13 CTBR, checkpoint at 3.0M (2999920 steps) | 50/50 (50/50) | 50/50 (50/50) | 38/50 (1/50) |
| **run 13 CTBR, final model (reported)** | 50/50 (50/50) | 50/50 (50/50) | 39/50 (1/50) |
| **run 14 motors, final model (reported)** | 50/50 (50/50) | 50/50 (50/50) | 50/50 (1/50) |

In nominal flight the final CTBR model ends with a body rate of 0.31 rad/s and a control effort of 18.5 (1.5M checkpoint: 0.00 rad/s, 2.4); the hover oscillation is visible in the effort, not in the success rate.

Off-axis rotation in one nominal episode (seed 10000): largest deviation of the pitch axis about 9° (run 13) and 8° (run 14), heading at the end 0.5° and 0.3°, against about 27° of axis tilt for run 8 and a heading change of about 130° for run 6.

**Added:** the two models replace the contents of `MTR_PPO/models/ctbr_seed0/` (run 13 final model, sha256 prefix 73acb74ce9eaf66c) and `MTR_PPO/models/motors_seed0/` (run 14 final, 302b851a49588c8e), with new READMEs. Both zips are copies whose stored `tensorboard_log` path is shortened to `<repo>` (all other entries byte-identical); both `config.json` files have their paths shortened the same way. The motors copy reproduces the earlier evaluation exactly (seeds 10000–10001, nominal and ppo_track_stress); the final CTBR model was evaluated (table above) from its path-shortened copy, through the `ppo_mtr` controller; `Logs/MTR_PPO/runs/run9_…` to `run14_…`; controllers `bc_ppo` and `bc_only` in `Evaluation/final_comparison.py` (the PPO track's delivered policy and its warm start, through `LearnedFlipController` and the baselines' `DecisionRateAdapter`; parity with the earlier scratch evaluation checked on seeds 10000–10001 under ppo_track_stress).

## 2026-10-03 — Report figures: `plot_comparison.py`, `flip_analysis.py`, `plot_training.py`

Three scripts that only read results or run single recorded nominal episodes; outputs go to `Evaluation/runs/figures/` and `MTR_PPO/runs/figures/` (PNG and PDF, plus text and LaTeX tables).

- `Evaluation/plot_comparison.py --runs <final-comparison runs>`: success rate per controller and condition (small multiples), with the 95% Wilson interval for the noisy conditions and pass/fail for the deterministic nominal condition.
- `Evaluation/flip_analysis.py`: one nominal episode per controller; time series of rotation, altitude change, pitch-axis deviation and body rate, and the off-axis table (integrated body rates over the flip, largest axis deviation, final heading). This makes the earlier scratch off-axis checks reproducible from the repository.
- `MTR_PPO/plot_training.py LABEL=PATH ...`: return, evaluation success and final body rate during training, from a training run folder or from `Logs/MTR_PPO/runs/`.

Checks: `Logs/evaluation_harness/report_figures_checks_log.txt`.
