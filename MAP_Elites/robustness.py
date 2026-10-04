"""Robustness screening of MAP-Elites elites (M5b-2), step 1: run the candidates under each condition.

Candidates: the default hand-tuned genome plus the top-K elites (by nominal search fitness)
of every listed run; identical genomes are kept once. Every candidate is run for 10 s on the
harness under each stress condition with the SAME episode seeds (paired comparison):

    nominal           1 episode (deterministic)
    obs_noise         simulator default observation noise        seeds 0..E-1
    wind              steady 1 m/s, random direction per seed    seeds 0..E-1
    perturbed_start   tilt <= 5 deg, rates/velocity <= 0.2       seeds 0..E-1

One detector judges success: the §04 detector (--detector section04, the default and the
setting of R1) or the PPO track's primary detector (--detector ppo_track, counted exactly as in
Evaluation/final_comparison.py). The pitch-travel column is always the §04 measurement. Episodes are written to episodes.csv in chunks, so a
stopped screening resumes where it stopped (same candidates required).

Selection (agreed rule): a candidate must succeed in the nominal 10 s episode; candidates are
ranked by their robust success rate (successes / episodes over the three stochastic conditions)
and ties are broken by nominal 10 s fitness -(effort + altitude loss). The stage-1 ranking uses
seeds 0..E-1 only; its top N ("leaders") then run seeds E..M-1 as well (confirmation) and are
ranked again on all M seeds. Rank 1 of the confirmation is the selected controller.

    python MAP_Elites/robustness.py --name R1 --runs ME1 ME2 ME3 --top 20 --episodes 10 --workers 12
    python MAP_Elites/robustness.py --name R1_pt --detector ppo_track --runs ME1_pt ME2_pt --top 10
Output: MAP_Elites/runs/robustness/<name>/ (candidates.json, config.json, episodes.csv,
        summary.json, ranking.csv)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import problem as P                                # noqa: E402  (also puts Evaluation/ and Controllers/ on the path)
import evaluate as harness                         # noqa: E402
from report import DETECTORS, failure_mode, succeeded, wilson   # noqa: E402
from three_phase_flip import FlipParams, ThreePhaseFlip   # noqa: E402

EPISODE_SECONDS = 10.0
CONDITIONS = {"nominal": harness.NOMINAL, "obs_noise": harness.OBS_NOISE,
              "wind": harness.WIND, "perturbed_start": harness.PERTURBED_START}
COLUMNS = ["candidate", "condition", "seed", "success", "failure_mode", "control_effort",
           "max_alt_loss_m", "pitch_travel_deg", "final_tilt_deg", "t_recovered_s"]


def select_candidates(runs_dir, runs, top: int) -> list[dict]:
    """Default genome + top-`top` elites of each run by stored fitness; duplicates removed."""
    cands, seen = [], set()

    def add(cid, genome, **meta):
        key = tuple(float(v) for v in genome)
        if key not in seen:
            seen.add(key)
            cands.append(dict(id=cid, genome=list(key), **meta))

    add("default", FlipParams().as_array(), run=None, cell=None, search_fitness=None, descriptors=None)
    for run in runs:
        elites = json.loads((Path(runs_dir) / run / "archive_final.json").read_text(encoding="utf-8"))["elites"]
        for e in sorted(elites, key=lambda e: (-e["fitness"], e["cell"]))[:top]:
            add(f"{run}:{e['cell'][0]},{e['cell'][1]}", e["genome"], run=run, cell=e["cell"],
                search_fitness=e["fitness"], descriptors=e["descriptors"])
    return cands


def jobs_for(cands, episodes: int, detector: str = "section04") -> list[tuple]:
    jobs = []
    for c in cands:
        for name in CONDITIONS:
            for seed in ([0] if name == "nominal" else range(episodes)):
                jobs.append((c["id"], tuple(c["genome"]), name, int(seed), detector))
    return jobs


def run_job(job) -> dict:
    """Worker: one 10 s episode of one candidate under one condition."""
    cid, genome, name, seed, detector = job
    try:
        r = harness.run_episode(ThreePhaseFlip(FlipParams.from_array(np.array(genome))), seed,
                                episode_seconds=EPISODE_SECONDS, condition=CONDITIONS[name])
    except (ValueError, FloatingPointError) as e:          # e.g. a non-finite motor command
        return dict(candidate=cid, condition=name, seed=seed, success=False,
                    failure_mode=f"error: {e}"[:120], control_effort=None, max_alt_loss_m=None,
                    pitch_travel_deg=None, final_tilt_deg=None, t_recovered_s=None)
    return dict(candidate=cid, condition=name, seed=seed, success=succeeded(r, detector),
                failure_mode=failure_mode(r, detector), control_effort=r.control_effort,
                max_alt_loss_m=r.max_alt_loss_m,
                pitch_travel_deg=float(np.degrees(r.detector["pitch_travel_rad"])),
                final_tilt_deg=r.final_tilt_deg, t_recovered_s=r.t_recovered_s)


def read_episodes(path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(r, seed=int(r["seed"]), success=r["success"] == "True") for r in csv.DictReader(f)]


def screen(out_dir, cands, episodes: int, workers: int = 1, chunk: int = 240, log=print,
           subset=None, detector: str = "section04") -> list[dict]:
    """Run every (candidate, condition, seed) not yet in episodes.csv; returns all episode rows.
    `subset` (candidate ids) restricts the new episodes to those candidates (confirmation stage)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cpath = out / "candidates.json"
    if cpath.exists() and json.loads(cpath.read_text(encoding="utf-8")) != cands:
        raise SystemExit(f"{out} holds a screening of different candidates; use another --name")
    cpath.write_text(json.dumps(cands, indent=1), encoding="utf-8")
    epath = out / "episodes.csv"
    done = {(r["candidate"], r["condition"], r["seed"]) for r in read_episodes(epath)}
    run_cands = cands if subset is None else [c for c in cands if c["id"] in set(subset)]
    todo = [j for j in jobs_for(run_cands, episodes, detector) if (j[0], j[2], j[3]) not in done]
    log(f"{len(run_cands)} candidates, {len(done)} episodes done, {len(todo)} to run")
    new_file = not epath.exists()
    t0 = time.time()
    pool = Pool(workers) if workers > 1 else None
    try:
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            rows = pool.map(run_job, part) if pool else [run_job(j) for j in part]
            with open(epath, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                if new_file:
                    w.writeheader()
                    new_file = False
                w.writerows(rows)
            log(f"episodes {min(i + chunk, len(todo))}/{len(todo)}  ({time.time() - t0:.0f} s)")
    finally:
        if pool:
            pool.close()
            pool.join()
    return read_episodes(epath)


STOCHASTIC = [k for k in CONDITIONS if k != "nominal"]


def rank(rows, cands, max_seed=None) -> list[dict]:
    """Per-candidate summary, sorted by the selection rule. Only seeds < max_seed are used
    (None = all), so a ranking never depends on episodes added later for other candidates."""
    rows = [r for r in rows if max_seed is None or r["seed"] < max_seed]
    out = []
    for c in cands:
        mine = [r for r in rows if r["candidate"] == c["id"]]
        if not mine:
            continue
        nom = next((r for r in mine if r["condition"] == "nominal"), None)
        nominal_ok = bool(nom and nom["success"])
        fit10 = (-(float(nom["control_effort"]) + float(nom["max_alt_loss_m"]))
                 if nom and nom["control_effort"] not in (None, "") else float("-inf"))
        per = {}
        for name in STOCHASTIC:
            rs = [r for r in mine if r["condition"] == name]
            k, n = sum(r["success"] for r in rs), len(rs)
            per[name] = dict(k=k, n=n, rate=k / n if n else 0.0, ci95=list(wilson(k, n)),
                             modes=sorted({r["failure_mode"] for r in rs if not r["success"]}))
        k = sum(v["k"] for v in per.values())
        n = sum(v["n"] for v in per.values())
        travel = [float(r["pitch_travel_deg"]) for r in mine if r["pitch_travel_deg"] not in (None, "")]
        out.append(dict(id=c["id"], run=c["run"], cell=c["cell"], nominal_ok=nominal_ok, fitness_10s=fit10,
                        robust_k=k, robust_n=n, robust_rate=k / n if n else 0.0, robust_ci95=list(wilson(k, n)),
                        worst_rate=min((v["rate"] for v in per.values()), default=0.0),
                        max_travel_deg=max(travel, default=None), conditions=per))
    out.sort(key=lambda d: (not d["nominal_ok"], -d["robust_rate"], -d["fitness_10s"], d["id"]))
    return out


def format_ranking(ranking, limit=None) -> str:
    head = f"{'#':>3} {'candidate':<14} {'nom':>3} {'robust':>13} {'95% CI':>13} " + \
           " ".join(f"{k:>15}" for k in STOCHASTIC) + f" {'fit 10s':>8} {'travel':>7}"
    lines = [head, "-" * len(head)]
    for i, d in enumerate(ranking[:limit], 1):
        lo, hi = d["robust_ci95"]
        conds = " ".join(f"{d['conditions'][k]['k']:>7}/{d['conditions'][k]['n']:<7}" for k in STOCHASTIC)
        travel = f"{d['max_travel_deg']:7.1f}" if d["max_travel_deg"] is not None else f"{'-':>7}"
        lines.append(f"{i:>3} {d['id']:<14} {'yes' if d['nominal_ok'] else 'NO':>3} "
                     f"{d['robust_k']:>5}/{d['robust_n']:<3}{d['robust_rate']:5.0%} {lo:6.0%}-{hi:<6.0%} "
                     f"{conds} {d['fitness_10s']:8.3f} {travel}")
    return "\n".join(lines)


def select(out_dir, cands, episodes, confirm, confirm_episodes, workers=1, log=print,
           detector: str = "section04") -> dict:
    """Stage 1 (all candidates, seeds < episodes) -> leaders -> confirmation (seeds < confirm_episodes)."""
    rows = screen(out_dir, cands, episodes, workers, log=log, detector=detector)
    stage1 = rank(rows, cands, max_seed=episodes)
    leaders = [d["id"] for d in stage1 if d["nominal_ok"]][:confirm]
    if leaders and confirm_episodes > episodes:
        rows = screen(out_dir, cands, confirm_episodes, workers, log=log, subset=leaders, detector=detector)
    if confirm == 0:                                   # no confirmation: select from stage 1
        final = [d for d in stage1 if d["nominal_ok"]]
    else:
        final = rank(rows, [c for c in cands if c["id"] in set(leaders)],
                     max_seed=max(episodes, confirm_episodes))
    summary = dict(stage1=stage1, leaders=leaders, confirmation=final,
                   selected=final[0]["id"] if final else None,
                   selected_genome=next((c["genome"] for c in cands if final and c["id"] == final[0]["id"]), None))
    out = Path(out_dir)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(out / "ranking.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "rank", "candidate", "nominal_ok", "robust_k", "robust_n", "robust_rate",
                    *[f"{k}_k" for k in STOCHASTIC], "fitness_10s", "max_travel_deg"])
        for stage, rk in (("stage1", stage1), ("confirmation", final)):
            for i, d in enumerate(rk, 1):
                w.writerow([stage, i, d["id"], d["nominal_ok"], d["robust_k"], d["robust_n"], d["robust_rate"],
                            *[d["conditions"][k]["k"] for k in STOCHASTIC], d["fitness_10s"], d["max_travel_deg"]])
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="output folder under MAP_Elites/runs/robustness/")
    ap.add_argument("--runs", nargs="+", required=True, help="MAP-Elites runs to take elites from")
    ap.add_argument("--top", type=int, default=20, help="elites per run, by nominal search fitness")
    ap.add_argument("--episodes", type=int, default=10, help="episodes per stochastic condition (stage 1)")
    ap.add_argument("--confirm", type=int, default=5, help="stage-1 leaders re-tested (0 = no confirmation)")
    ap.add_argument("--confirm-episodes", type=int, default=50, help="episodes per condition for the leaders")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--detector", choices=DETECTORS, default="section04",
                    help="detector that judges success (default section04, as in R1)")
    a = ap.parse_args()
    if a.top < 1 or a.episodes < 1 or a.confirm < 0:
        ap.error("--top and --episodes must be >= 1 and --confirm >= 0")
    if a.confirm and a.confirm_episodes < a.episodes:
        ap.error("--confirm-episodes must be >= --episodes")
    out = HERE / "runs" / "robustness" / a.name
    cands = select_candidates(HERE / "runs", a.runs, a.top)
    cfg = dict(runs=a.runs, top=a.top, episodes=a.episodes, confirm=a.confirm,
               confirm_episodes=a.confirm_episodes, episode_seconds=EPISODE_SECONDS,
               conditions={k: vars(v) for k, v in CONDITIONS.items()})
    if a.detector != "section04":          # old screenings (R1) keep their config.json unchanged
        cfg["detector"] = a.detector
    cfg_path = out / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text(encoding="utf-8")) != json.loads(json.dumps(cfg)):
        raise SystemExit(f"{out} was screened with other settings; use another --name")
    out.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    summary = select(out, cands, a.episodes, a.confirm, a.confirm_episodes, a.workers, detector=a.detector)
    print("\nStage 1 (seeds 0-%d), top 15:" % (a.episodes - 1))
    print(format_ranking(summary["stage1"], 15))
    if summary["confirmation"]:
        print("\nConfirmation (seeds 0-%d):" % (max(a.episodes, a.confirm_episodes) - 1))
        print(format_ranking(summary["confirmation"]))
    print("\nselected:", summary["selected"], "->", out / "summary.json")


if __name__ == "__main__":
    main()
