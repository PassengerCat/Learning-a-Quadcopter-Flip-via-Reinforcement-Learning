"""MAP-Elites problem definition (M2): search space, behaviour descriptors, fitness.

Genome     : the 11 FlipParams of Controllers/three_phase_flip.py, inside GENOME_BOUNDS
             (a run may override single limits; see genome_box).
Validity   : ONLY an independent detector's verdict on a nominal episode, never the fitness.
             validity="section04" (default, runs ME1-ME3 and the seed replicates): the §04
             detector (flip + final stable hover, no crash, in region).
             validity="ppo_track": the PPO track's (primary) detector, i.e. the report's success
             criterion (latched flip + 0.4 s upright-and-slow hold), AND the episode did not end
             in a crash or region exit afterwards, so the crash loophole stays closed.
             single_turn_check=True (with "ppo_track"): additionally, the final revolution of the
             primary detector must be at most 390 deg, i.e. the vehicle turned once. The primary
             detector alone does not count the turns, so a double flip would otherwise be valid.
             (The §04 detector already enforces one turn.)
             Invalid genomes get no descriptors, no fitness and never enter the archive -- a
             crashing or unfinished flip cannot occupy a cell ("crash loophole"), and hovering
             never qualifies ("do-nothing loophole").
Descriptors: behaviour of the flip, measured on the recorded trajectory (5 ms resolution):
               rotation_duration = t(rev >= 351.4 deg) - t(rev >= 10 deg)            [s]
               max_climb         = max(z0 - z) over the episode, NED so up is -z      [m]
Fitness    : quality inside a cell, higher is better:
               f = -(control_effort + ALT_LOSS_WEIGHT * max_alt_loss)
All bounds come from the 400-sample Latin-hypercube study (Logs/map_elites/descriptor_sampling).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "Controllers", REPO / "Evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import evaluate as harness                                       # noqa: E402
from three_phase_flip import FlipParams, ThreePhaseFlip          # noqa: E402

SEARCH_EPISODE_SECONDS = 5.0

# Genome box (a_brake and Kd_att narrowed after the sampling study: no success above 145 / below 6.4)
GENOME_BOUNDS = {
    "t_climb": (0.0, 0.6), "F_climb_g": (1.0, 3.0), "q_peak": (8.0, 35.0), "a_brake": (20.0, 150.0),
    "F_rot_g": (0.3, 1.5), "Kp_att": (20.0, 120.0), "Kd_att": (6.0, 25.0), "Kv": (1.0, 8.0),
    "Kz": (0.5, 4.0), "Kvxy": (0.5, 4.0), "Kpxy": (0.2, 2.0)}
assert list(GENOME_BOUNDS) == [f.name for f in fields(FlipParams)], "bounds must follow FlipParams order"
GENOME_LOW = np.array([b[0] for b in GENOME_BOUNDS.values()])
GENOME_HIGH = np.array([b[1] for b in GENOME_BOUNDS.values()])


def genome_box(bounds: dict | None = None) -> dict:
    """Validated genome box as {name: (low, high)} in FlipParams order.
    `bounds` may override any subset of GENOME_BOUNDS (a run changing one limit, e.g. ME2).
    Rejects unknown names, non-finite or empty ranges, and boxes that exclude the default
    genome (the default seeds every run's initialisation)."""
    box = dict(GENOME_BOUNDS)
    for name, lh in (bounds or {}).items():
        if name not in box:
            raise ValueError(f"unknown genome parameter {name!r}; expected one of {list(box)}")
        lo, hi = (float(v) for v in lh)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"{name}: need finite low < high, got ({lo}, {hi})")
        box[name] = (lo, hi)
    default = FlipParams().as_array()
    for (name, (lo, hi)), d in zip(box.items(), default):
        if not lo <= d <= hi:
            raise ValueError(f"{name}: box ({lo}, {hi}) excludes the default value {d}")
    return box


def box_arrays(box: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(low, high) arrays of a genome box (default: GENOME_BOUNDS)."""
    box = genome_box(box)
    return np.array([b[0] for b in box.values()]), np.array([b[1] for b in box.values()])

# Descriptor space: 10 x 10 grid; values outside the range fall into the edge cells
DESCRIPTOR_NAMES = ("rotation_duration_s", "max_climb_m")
DESCRIPTOR_BOUNDS = ((0.20, 0.80), (0.0, 3.0))
GRID_SHAPE = (10, 10)

REV_START_RAD = np.deg2rad(10.0)
ALT_LOSS_WEIGHT = 1.0          # effort ~0.8-3.4 and altitude loss ~0-2.8 m in the sampling study


VALIDITY_DETECTORS = ("section04", "ppo_track")
ONE_TURN_MAX_RAD = 2.0 * np.pi + np.deg2rad(30.0)     # the §04 detector's max_rev_rad


def is_valid(result, validity: str = "section04", single_turn_check: bool = False) -> bool:
    """Validity of one harness episode under the chosen detector (see the module docstring)."""
    if validity == "section04":
        return bool(result.success)
    if validity == "ppo_track":
        ok = bool(result.ppo_track["success"]) and not bool(result.terminated)
        if single_turn_check:
            ok = ok and float(result.ppo_track["revolution_rad"]) <= ONE_TURN_MAX_RAD
        return ok
    raise ValueError(f"unknown validity {validity!r}; choose from {VALIDITY_DETECTORS}")


@dataclass
class Evaluation:
    valid: bool                        # detector success on the nominal search episode (see is_valid)
    descriptors: tuple | None          # (rotation_duration_s, max_climb_m) if valid
    fitness: float | None              # higher is better, if valid
    result: "harness.EpisodeResult"    # full harness result (metrics + trajectory)


def descriptors(trajectory: dict, rev_done_rad: float) -> tuple[float | None, float]:
    """(rotation duration, max climb) from a harness trajectory. Duration is None if the
    rotation never reached rev_done_rad (the detector's own completion threshold)."""
    t, rev, z = trajectory["t"], trajectory["rev"], trajectory["pos"][:, 2]
    start, done = np.nonzero(rev >= REV_START_RAD)[0], np.nonzero(rev >= rev_done_rad)[0]
    duration = float(t[done[0]] - t[start[0]]) if len(done) else None
    return duration, float(max(0.0, np.max(z[0] - z)))


def fitness(result) -> float:
    return -(result.control_effort + ALT_LOSS_WEIGHT * result.max_alt_loss_m)


def evaluate_genome(x, episode_seconds: float = SEARCH_EPISODE_SECONDS, box: dict | None = None,
                    validity: str = "section04", single_turn_check: bool = False) -> Evaluation:
    """Run one nominal episode of the controller defined by genome x and score it.
    `box` is the run's genome box (default GENOME_BOUNDS); x must lie inside it.
    `validity` picks the detector that decides validity, `single_turn_check` adds the one-turn
    check to the primary detector (see is_valid)."""
    x = np.asarray(x, float)
    low, high = box_arrays(box)
    if x.shape != low.shape or np.any(x < low - 1e-12) or np.any(x > high + 1e-12):
        raise ValueError(f"genome outside the genome box: {x}")
    result = harness.evaluate(ThreePhaseFlip(FlipParams.from_array(x)), [0],
                              episode_seconds=episode_seconds, record=True)[0]
    if not is_valid(result, validity, single_turn_check):
        return Evaluation(False, None, None, result)
    rev_done = harness.detector_config().full_rev_rad     # 2*pi - 0.15, the same for both detectors
    return Evaluation(True, descriptors(result.trajectory, rev_done), fitness(result), result)
