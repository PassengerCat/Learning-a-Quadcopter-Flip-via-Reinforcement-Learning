"""Final comparison (M5c): every controller, same benchmark, same seeds, same §04 detector.

Controllers
    map_elites_final     the selected MAP-Elites genome (Controllers/map_elites_final.json)
    three_phase_default  the same controller with the hand-tuned default genome
    scripted_flip        §02 rule-based flip (PPO track, via Baselines/harness_adapter.py)
    pid_hover            §02 cascaded PID that never flips (floor for "does nothing")
    random               §02 uniform random motor commands (floor)
    ppo_mtr              MTR-PPO, CTBR actions (MTR_PPO/models/ctbr_seed0/final_model.zip)
Conditions
    nominal              the brief's benchmark: no noise, no wind, nominal start (primary result)
    action_noise_005     motor-command noise 0.05, simulator sensor noise off   (run FINAL)
    stress               motor 0.15 + gyro 0.2 rad/s, sensor noise off          (run FINAL)
    ppo_track_nominal    the PPO track's "nominal" exactly: motor noise 0.05 + the simulator's
                         default sensor noise (her env keeps it on)
    ppo_track_stress     the PPO track's "stress" exactly: motor 0.15 + gyro 0.2 + sensor noise
Default controllers: the first five (the FINAL set). Default conditions: nominal,
action_noise_005, stress (the FINAL set).
Detector (--detector): section04 (default, as in FINAL) or ppo_track (the PPO track's
detector on the same episodes; the §04 verdict is kept in the CSV and under the table).

The nominal benchmark is deterministic, so deterministic controllers repeat the same
episode on every seed; the seeds still matter for random and for the noisy conditions.

    python Evaluation/final_comparison.py --name FINAL --episodes 50 --workers 12
    python Evaluation/final_comparison.py --name FINAL_PT --detector ppo_track --seed0 10000 \
        --conditions nominal ppo_track_nominal ppo_track_stress --controllers map_elites_final scripted_flip
    python Evaluation/final_comparison.py --name FINAL_PT_MTR --detector ppo_track --seed0 10000 \
        --conditions nominal ppo_track_nominal ppo_track_stress --controllers ppo_mtr
Output: Evaluation/runs/<name>/<condition>/ (episodes.csv, summary.json) and summary.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (HERE, REPO / "Controllers", REPO / "Baselines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import evaluate as harness                                   # noqa: E402
from report import DETECTORS, format_table, write_results   # noqa: E402

CONTROLLERS = ("map_elites_final", "three_phase_default", "scripted_flip", "pid_hover", "random", "ppo_mtr")
DEFAULT_CONTROLLERS = CONTROLLERS[:5]          # the FINAL set; ppo_mtr only on request
MTR_MODELS = {"ppo_mtr": REPO / "MTR_PPO" / "models" / "ctbr_seed0" / "final_model.zip"}
CONDITIONS = {"nominal": harness.NOMINAL,
              "action_noise_005": harness.Condition("action_noise_005", action_noise=0.05),
              "stress": harness.STRESS,
              "ppo_track_nominal": harness.Condition("ppo_track_nominal", obs_noise_scale=1.0,
                                                     action_noise=0.05),
              "ppo_track_stress": harness.Condition("ppo_track_stress", obs_noise_scale=1.0,
                                                    action_noise=0.15, gyro_noise=0.2)}
DEFAULT_CONDITIONS = ("nominal", "action_noise_005", "stress")


def make_controller(name: str):
    if name == "map_elites_final":
        from map_elites_controller import load_final_controller
        return load_final_controller()
    if name == "three_phase_default":
        from three_phase_flip import FlipParams, ThreePhaseFlip
        return ThreePhaseFlip(FlipParams())
    if name in ("scripted_flip", "pid_hover", "random"):
        from harness_adapter import make_baseline
        return make_baseline(name)
    if name in MTR_MODELS:
        sys.path.insert(0, str(REPO / "MTR_PPO"))
        from mtr_controller import MTRController
        return MTRController(str(MTR_MODELS[name]))           # its config.json sits next to it
    raise ValueError(f"unknown controller {name!r}; choose from {CONTROLLERS}")


def run_job(job):
    """Worker: one controller, one condition, a block of seeds (one env reused)."""
    name, cond, seeds = job
    results = harness.evaluate(make_controller(name), seeds, condition=CONDITIONS[cond])
    return name, cond, results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="output folder under Evaluation/runs/")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--controllers", nargs="+", default=list(DEFAULT_CONTROLLERS), choices=CONTROLLERS)
    ap.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--detector", default="section04", choices=DETECTORS,
                    help="whose verdict counts as success (both are stored)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--block", type=int, default=5, help="seeds per job")
    a = ap.parse_args(argv)
    if a.episodes < 1 or a.block < 1:
        ap.error("--episodes and --block must be >= 1")
    out = HERE / "runs" / a.name
    if out.exists():
        ap.error(f"{out} already exists; use another --name")
    seeds = list(range(a.seed0, a.seed0 + a.episodes))
    blocks = [seeds[i:i + a.block] for i in range(0, len(seeds), a.block)]
    jobs = [(n, c, b) for c in a.conditions for n in a.controllers for b in blocks]
    t0 = time.time()
    print(f"{len(a.controllers)} controllers x {len(a.conditions)} conditions x {a.episodes} episodes "
          f"on {a.workers} workers ...", flush=True)
    collected = {c: {n: [] for n in a.controllers} for c in a.conditions}
    with Pool(a.workers) as pool:
        for name, cond, results in pool.imap_unordered(run_job, jobs):
            collected[cond][name].extend(results)
    report = []
    for cond in a.conditions:
        per = {n: sorted(collected[cond][n], key=lambda r: r.seed) for n in a.controllers}
        meta = dict(condition=cond, **vars(CONDITIONS[cond]), seeds=[seeds[0], seeds[-1]])
        if a.detector != "section04":
            meta["detector"] = a.detector
        summary = write_results(per, out / cond, meta=meta, detector=a.detector)
        det = "" if a.detector == "section04" else f", detector {a.detector}"
        report += [f"== {cond} ({a.episodes} episodes, seeds {seeds[0]}-{seeds[-1]}{det}) ==",
                   format_table(summary), ""]
    text = "\n".join(report)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    (out / "config.json").write_text(json.dumps(vars(a), indent=1), encoding="utf-8")
    print(text)
    print(f"({time.time() - t0:.0f} s) -> {out}")


if __name__ == "__main__":
    main()
