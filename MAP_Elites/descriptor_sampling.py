"""Latin-hypercube descriptor study, re-judged with a chosen detector.

The original study (Logs/map_elites/descriptor_sampling/samples.json) evaluated 400
Latin-hypercube genomes (seed 0) plus the default genome on nominal 5 s episodes with the §04
detector, and fixed the descriptor ranges and the genome box of problem.py from it. This script
re-runs exactly the same 401 genomes (read from that file, so the sample is identical) and
judges validity with problem.is_valid(result, validity), e.g. the PPO track's primary detector.

Outputs in MAP_Elites/runs/descriptor_sampling/<name>/:
    samples.json   every genome with its validity, descriptors, effort and altitude loss
    analysis.txt   success count, failure modes, descriptor percentiles (p1/p99), convergence
                   (first 200 vs all 400 samples) and the Spearman correlation of the descriptors
    descriptor_sampling.png   descriptors of the valid samples with the grid ranges

    python MAP_Elites/descriptor_sampling.py --name LHS_pt --validity ppo_track --single-turn-check --workers 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import problem as P                                # noqa: E402  (also puts Evaluation/ on the path)
import evaluate as harness                         # noqa: E402
from report import failure_mode                    # noqa: E402
from three_phase_flip import FlipParams, ThreePhaseFlip   # noqa: E402

SOURCE = REPO / "Logs" / "map_elites" / "descriptor_sampling" / "samples.json"
REPORT_DETECTOR = {"section04": "section04", "ppo_track": "ppo_track"}


def run_one(x, validity: str, single_turn_check: bool = False) -> dict:
    """One nominal 5 s search episode, judged as in problem.evaluate_genome (box not enforced:
    the study's sampling box is wider than the search box)."""
    r = harness.evaluate(ThreePhaseFlip(FlipParams.from_array(np.asarray(x, float))), [0],
                         episode_seconds=P.SEARCH_EPISODE_SECONDS, record=True)[0]
    valid = P.is_valid(r, validity, single_turn_check)
    dur, climb = P.descriptors(r.trajectory, harness.detector_config().full_rev_rad)
    if valid:
        mode = "success"
    elif r.terminated:
        mode = "crash/region"
    elif single_turn_check and r.ppo_track["success"]:
        mode = "more than one turn"
    else:
        mode = failure_mode(r, REPORT_DETECTOR[validity])
    return dict(x=list(map(float, x)), valid=bool(valid), failure_mode=mode,
                rot_duration=dur, max_climb=climb, effort=float(r.control_effort),
                alt_loss=float(r.max_alt_loss_m),
                ppo_track_revolution_deg=float(np.degrees(r.ppo_track["revolution_rad"])),
                max_revolution_deg=float(np.degrees(r.detector.get("max_revolution_rad", np.nan))))


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def analyse(rows, log) -> dict:
    from scipy.stats import spearmanr
    lhs = [r for r in rows if r["kind"] == "lhs"]
    ok = [r for r in lhs if r["valid"] and r["rot_duration"] is not None]
    log(f"default genome: valid={rows[0]['valid']} duration={rows[0]['rot_duration']} climb={rows[0]['max_climb']:.3f}")
    log(f"LHS: {len(ok)}/{len(lhs)} valid ({len(ok) / len(lhs):.1%})")
    log("failure modes: " + str(Counter(r["failure_mode"] for r in lhs if not r["valid"]).most_common()))
    multi = [r for r in ok if r["ppo_track_revolution_deg"] > 390.0]
    log(f"valid samples whose final revolution exceeds 390 deg (more than one turn): {len(multi)}")
    out = dict(n=len(lhs), valid=len(ok))
    for key in ("rot_duration", "max_climb", "effort", "alt_loss"):
        v = np.array([r[key] for r in ok])
        out[key] = dict(min=float(v.min()), p1=pct(v, 1), p50=pct(v, 50), p99=pct(v, 99), max=float(v.max()))
        log(f"  {key:<12} min/p1/p50/p99/max = " + " ".join(f"{out[key][k]:.3f}" for k in ("min", "p1", "p50", "p99", "max")))
    first = [r for r in ok if lhs.index(r) < 200]
    for key in ("rot_duration", "max_climb"):
        a, b = [r[key] for r in first], [r[key] for r in ok]
        log(f"convergence {key:<12} first 200: [{pct(a, 1):.3f}, {pct(a, 99):.3f}]   all 400: [{pct(b, 1):.3f}, {pct(b, 99):.3f}]")
    rho, p = spearmanr([r["rot_duration"] for r in ok], [r["max_climb"] for r in ok])
    out["spearman"] = dict(rho=float(rho), p=float(p))
    log(f"Spearman(duration, climb) among valid samples = {rho:+.3f} (p={p:.3f})")
    return out


def figure(rows, path, validity):  # validity: text for the title
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lhs = [r for r in rows if r["kind"] == "lhs"]
    ok = [r for r in lhs if r["valid"] and r["rot_duration"] is not None]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter([r["rot_duration"] for r in ok], [r["max_climb"] for r in ok], s=14, alpha=0.75,
               label=f"valid LHS samples ({len(ok)}/{len(lhs)})")
    d = rows[0]
    if d["valid"]:
        ax.scatter([d["rot_duration"]], [d["max_climb"]], marker="*", s=140, color="k", label="default genome")
    (x0, x1), (y0, y1) = P.DESCRIPTOR_BOUNDS
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=1.2, color="gray",
                               label="archive grid ranges"))
    ax.set_xlabel("rotation duration $d_1$ [s]")
    ax.set_ylabel("maximum climb $d_2$ [m]")
    ax.set_title(f"Latin-hypercube study (validity: {validity})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="output folder under MAP_Elites/runs/descriptor_sampling/")
    ap.add_argument("--validity", choices=P.VALIDITY_DETECTORS, default="ppo_track")
    ap.add_argument("--single-turn-check", action="store_true",
                    help="with --validity ppo_track: valid only if the vehicle turned once")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N genomes (quick check)")
    a = ap.parse_args()
    if a.single_turn_check and a.validity != "ppo_track":
        ap.error("--single-turn-check needs --validity ppo_track")
    out = HERE / "runs" / "descriptor_sampling" / a.name
    out.mkdir(parents=True, exist_ok=True)
    src = json.loads(SOURCE.read_text(encoding="utf-8"))["samples"]
    src = [s for s in src if s["kind"] == "default"] + [s for s in src if s["kind"] == "lhs"]
    if a.limit:
        src = src[:a.limit]
    work = partial(run_one, validity=a.validity, single_turn_check=a.single_turn_check)
    xs = [s["x"] for s in src]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            res = pool.map(work, xs, chunksize=4)
    else:
        res = [work(x) for x in xs]
    rows = [dict(r, kind=s["kind"]) for s, r in zip(src, res)]
    (out / "samples.json").write_text(json.dumps(dict(validity=a.validity, single_turn_check=a.single_turn_check, source=str(SOURCE.relative_to(REPO)),
                                                      samples=rows), indent=0), encoding="utf-8")
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log(f"validity: {a.validity}{' + single-turn check' if a.single_turn_check else ''}; genomes: {len(rows)} (default + LHS) from {SOURCE.relative_to(REPO)}")
    summary = analyse(rows, log)
    (out / "analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    figure(rows, out / "descriptor_sampling.png", a.validity + (" + single-turn check" if a.single_turn_check else ""))
    print("written to", out)


if __name__ == "__main__":
    main()
