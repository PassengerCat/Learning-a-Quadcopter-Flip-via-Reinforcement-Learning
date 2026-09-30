"""Evaluation harness (§05), step 3: summaries, result files and the comparison table.

Works on the EpisodeResult lists returned by evaluate.evaluate(); it never
re-runs or re-judges an episode. `detector` picks whose verdict counts as success:
"section04" (EpisodeResult.success, the default) or "ppo_track" (EpisodeResult.ppo_track,
the PPO track's detector on the same episode). Both are stored with every episode.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

METRICS = ("final_tilt_deg", "final_omega", "max_alt_loss_m", "t_flip_s", "t_recovered_s",
           "control_effort")
DETECTORS = ("section04", "ppo_track")
PPO_TRACK_FULL_REV = 2.0 * np.pi - 0.15      # her SuccessConfig.full_rev_rad


def _check(detector: str) -> None:
    if detector not in DETECTORS:
        raise ValueError(f"unknown detector {detector!r}; choose from {DETECTORS}")


def succeeded(r, detector: str = "section04") -> bool:
    _check(detector)
    return bool(r.success) if detector == "section04" else bool(r.ppo_track["success"])


def metric(r, m: str, detector: str = "section04"):
    """Metric value; flip and recovery times come from the chosen detector."""
    _check(detector)
    if detector == "ppo_track" and m in ("t_flip_s", "t_recovered_s"):
        return r.ppo_track[m]
    return getattr(r, m)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n episodes (sensible also for 0/n, n/n)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def failure_mode(r, detector: str = "section04") -> str:
    """First failed criterion, in the order a flip can go wrong."""
    _check(detector)
    if detector == "ppo_track":
        p = r.ppo_track
        if p["success"]:
            return "success"
        if r.terminated:
            return "crash/region"
        if not p["passed_inverted"]:
            return "never inverted"
        if p["revolution_rad"] < PPO_TRACK_FULL_REV:
            return "rotation incomplete"
        return "no 0.4 s hold"
    d = r.detector
    if r.success:
        return "success"
    if d["crashed"] or d["left_region"]:
        return "crash/region"
    if d["excessive_rotation"]:
        return "over-rotation"
    if not d["reached_inverted"]:
        return "never inverted"
    if not d["completed_rotation"]:
        return "rotation incomplete"
    return "no final hold"


def summarize(results, detector: str = "section04") -> dict:
    """Success rate with interval, per-metric mean/min/max, and failure-mode counts."""
    n, k = len(results), sum(succeeded(r, detector) for r in results)
    lo, hi = wilson(k, n)
    metrics = {}
    for m in METRICS:
        v = np.array([metric(r, m, detector) for r in results if metric(r, m, detector) is not None], float)
        metrics[m] = dict(n=int(v.size), mean=float(v.mean()) if v.size else None,
                          min=float(v.min()) if v.size else None,
                          max=float(v.max()) if v.size else None)
    modes: dict[str, int] = {}
    for r in results:
        modes[failure_mode(r, detector)] = modes.get(failure_mode(r, detector), 0) + 1
    out = dict(episodes=n, successes=int(k), success_rate=k / n if n else 0.0,
               success_ci95=[lo, hi], metrics=metrics, failure_modes=modes)
    if detector == "ppo_track":                  # cross-reference: the other verdict, same episodes
        out.update(detector=detector, section04_successes=int(sum(r.success for r in results)),
                   success_then_terminated=int(sum(r.ppo_track["success_then_terminated"] for r in results)))
    return out


def write_results(all_results: dict, out_dir, meta: dict | None = None,
                  detector: str = "section04") -> dict:
    """all_results = {controller_name: [EpisodeResult, ...]} -> episodes.csv + summary.json.
    With detector="ppo_track", success/failure_mode/t_flip_s/t_recovered_s follow her detector
    and extra columns keep the §04 verdict and her revolution for every episode."""
    _check(detector)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["controller", "seed", "success", "failure_mode", "terminated", "duration_s", *METRICS,
            "revolution_deg", "pitch_travel_deg"]
    if detector == "ppo_track":
        cols += ["detector", "success_section04", "failure_mode_section04", "ppo_track_revolution_deg",
                 "ppo_track_success_then_terminated"]
    with open(out / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for name, results in all_results.items():
            for r in results:
                row = {c: getattr(r, c) for c in cols if hasattr(r, c)}
                row.update(controller=name, failure_mode=failure_mode(r),
                           revolution_deg=float(np.degrees(r.detector["revolution_rad"])),
                           pitch_travel_deg=float(np.degrees(r.detector["pitch_travel_rad"])))
                if detector == "ppo_track":
                    row.update({m: metric(r, m, detector) for m in ("t_flip_s", "t_recovered_s")})
                    row.update(detector=detector, success=succeeded(r, detector),
                               failure_mode=failure_mode(r, detector), success_section04=r.success,
                               failure_mode_section04=failure_mode(r),
                               ppo_track_revolution_deg=float(np.degrees(r.ppo_track["revolution_rad"])),
                               ppo_track_success_then_terminated=r.ppo_track["success_then_terminated"])
                w.writerow(row)
    summary = {name: summarize(results, detector) for name, results in all_results.items()}
    (out / "summary.json").write_text(json.dumps(dict(meta=meta or {}, controllers=summary), indent=2),
                                      encoding="utf-8")
    return summary


def format_table(summary: dict) -> str:
    """Plain-text comparison table (mean over episodes; flip/recovery times over those that have one)."""
    def mean(s, m, fmt):
        v = s["metrics"][m]["mean"]
        width = len(fmt.format(0.0))
        return f"{'—':>{width}}" if v is None else fmt.format(v)
    head = (f"{'controller':18s} {'success':>9s} {'95% CI':>13s} {'alt loss':>9s} {'fin tilt':>9s} "
            f"{'fin |w|':>8s} {'t_flip':>7s} {'t_rec':>7s} {'effort':>7s}  failure modes")
    lines = [head, "-" * len(head)]
    for name, s in summary.items():
        lo, hi = s["success_ci95"]
        modes = ", ".join(f"{k} {v}" for k, v in s["failure_modes"].items() if k != "success")
        lines.append(f"{name:18s} {s['successes']:>4d}/{s['episodes']:<4d} "
                     f"[{lo:4.2f},{hi:4.2f}]  {mean(s, 'max_alt_loss_m', '{:7.2f} m')} "
                     f"{mean(s, 'final_tilt_deg', '{:7.1f}°')} {mean(s, 'final_omega', '{:8.2f}')} "
                     f"{mean(s, 't_flip_s', '{:7.2f}')} {mean(s, 't_recovered_s', '{:7.2f}')} "
                     f"{mean(s, 'control_effort', '{:7.2f}')}  {modes or '—'}")
    notes = [f"{name}: §04 verdict on the same episodes {s['section04_successes']}/{s['episodes']}"
             + (f"; {s['success_then_terminated']} latched successes then crashed"
                if s["success_then_terminated"] else "")
             for name, s in summary.items() if s.get("detector") == "ppo_track"]
    if notes:
        lines += ["(success, failure modes, t_flip and t_rec by the PPO track's detector)", *notes]
    return "\n".join(lines)
