# §04 — Success detectors

Success is decided geometrically from the true simulator state, never by a reward. Two detectors are kept; the evaluation harness (`Evaluation/`) runs both on every episode.

## `ppo_track_detector.py` — the detector used for the reported numbers

A verbatim copy of the PPO track's detector (`Reward_and_Training/flip_reward.py`), so that every controller is judged by the same definition as the PPO results. Updated once per 20 ms decision.

- a signed revolution of the gravity direction in the body x-z plane ≥ 360° − 8.6°;
- tilt ≥ 150° at some update;
- afterwards upright (tilt < 15°) and slow (|v| < 0.4 m/s, |ω| < 0.8 rad/s) for 20 consecutive updates (0.4 s);
- success is latched once reached. There is no one-turn limit and no crash or region check inside the detector; the harness reports "latched, then crashed" separately.

## `success_detector.py` — the stricter §04 detector (reported alongside)

`StreamingSuccessDetector` (online, one update per 5 ms step) and `evaluate_episode` (offline, a whole logged trajectory). All criteria must hold:

0. **starts upright** (tilt < 15°);
1. **reached inverted**: tilt ≥ 170°, also checked along the arc between samples (slerp, 8 sub-steps), so the verdict does not depend on the sampling rate;
2. **completed rotation**: a signed revolution ≥ 360° − 8.6°, from accumulated quaternion pitch increments;
3. **recovered and held**: upright (< 15°) and slow (|v| < 0.4 m/s, |ω| < 0.8 rad/s) for the last 0.4 s of the episode;
4. **no crash**;
5. **in region**: |z| ≤ 5 m, finite state;
6. **one turn only**: final revolution within 15° of 360°, maximum revolution ≤ 390°, |pitch| travel ≤ 420° during the maneuver (until the first upright sample after the revolution).

`SuccessResult.reason()` gives a one-line explanation of a failure.

Checks: `Logs/success_detector/`.
