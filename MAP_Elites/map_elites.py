"""MAP-Elites loop (M4), as in lecture 9: fill a behaviour grid with the best flip per cell.

  1. initialise: the hand-tuned default genome + N_INIT Latin-hypercube genomes;
  2. repeat until the budget is spent:
       pick BATCH elites uniformly at random, mutate each (Gaussian, sigma = SIGMA_FRAC
       of every gene's range, clipped to the genome box), evaluate them in parallel,
       and try to insert each child into the archive (problem.py decides validity,
       descriptors and fitness; archive.py the insertion rule).

Every evaluation is one nominal 5 s episode. Validity is decided by the §04 detector
(--validity section04, the default and the setting of ME1-ME3) or by the PPO track's primary
detector (--validity ppo_track, the report's success criterion), optionally with a check that the
vehicle turned only once (--single-turn-check); see problem.py. Both are stored in config.json and
fixed for the life of the run.
All randomness lives in one generator in the main process and results are inserted in
submission order, so a run is reproducible for a given seed whatever the worker count.
Checkpoints (archive + generator state + history) allow a stopped run to resume exactly.

The genome box defaults to problem.GENOME_BOUNDS; single limits can be overridden per run
(--bound NAME=LOW,HIGH, repeatable). The full box is stored in the run's config.json and is
fixed for the life of the run (resume keeps it).

Run (the user starts full runs):
    python MAP_Elites/map_elites.py --name ME1 --budget 5000 --workers 8 --seed 0
    python MAP_Elites/map_elites.py --name ME1 --resume        # continue after a stop
    python MAP_Elites/map_elites.py --name ME2 --budget 2000 --bound t_climb=0,1.0
    python MAP_Elites/map_elites.py --name ME1_pt --validity ppo_track --single-turn-check --budget 5000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import problem as P                    # noqa: E402
from archive import Archive            # noqa: E402


@dataclass
class Config:
    seed: int = 0
    budget: int = 5000          # total evaluations, including the initial ones
    n_init: int = 100           # Latin-hypercube genomes (plus the default genome)
    batch: int = 20             # children per generation
    sigma_frac: float = 0.10    # mutation std as a fraction of each gene's range
    workers: int = 1
    checkpoint_every: int = 500
    genome_bounds: dict = field(default_factory=lambda: dict(P.GENOME_BOUNDS))   # full box of this run
    validity: str = "section04"   # detector that decides validity (problem.VALIDITY_DETECTORS)
    single_turn_check: bool = False   # with ppo_track: also require that the vehicle turned only once

    def __post_init__(self):
        if self.validity not in P.VALIDITY_DETECTORS:
            raise ValueError(f"unknown validity {self.validity!r}; choose from {P.VALIDITY_DETECTORS}")
        # validated, completed from GENOME_BOUNDS, FlipParams order, float tuples
        # (so a box read back from config.json compares equal to the one that was saved)
        self.genome_bounds = {k: (float(lo), float(hi)) for k, (lo, hi) in P.genome_box(self.genome_bounds).items()}


def evaluate_for_archive(x, box: dict | None = None, validity: str = "section04",
                         single_turn_check: bool = False) -> dict:
    """Worker: evaluate one genome, return only what the archive and the logs need."""
    ev = P.evaluate_genome(x, box=box, validity=validity, single_turn_check=single_turn_check)
    r = ev.result
    return dict(valid=ev.valid, descriptors=ev.descriptors, fitness=ev.fitness,
                meta=dict(reason=r.reason.split(";")[0], control_effort=r.control_effort,
                          max_alt_loss_m=r.max_alt_loss_m, t_flip_s=r.t_flip_s,
                          t_recovered_s=r.t_recovered_s))


def initial_genomes(cfg: Config, rng) -> np.ndarray:
    from scipy.stats import qmc
    low, high = P.box_arrays(cfg.genome_bounds)
    lhs = qmc.LatinHypercube(d=len(low), seed=rng).random(cfg.n_init)
    return np.vstack([P.FlipParams().as_array(), qmc.scale(lhs, low, high)])


def mutate(parent, cfg: Config, rng) -> np.ndarray:
    low, high = P.box_arrays(cfg.genome_bounds)
    step = rng.normal(0.0, cfg.sigma_frac * (high - low))
    return np.clip(np.asarray(parent, float) + step, low, high)


class MapElites:
    def __init__(self, cfg: Config, out_dir):
        self.cfg, self.out = cfg, Path(out_dir)
        self.rng = np.random.default_rng(cfg.seed)
        self.archive, self.evals, self.history = Archive(), 0, []
        self.pending_init = None

    # -- one batch: evaluate in order, insert in order ------------------------------------ #
    def _evaluate(self, genomes, pool):
        work = partial(evaluate_for_archive, box=self.cfg.genome_bounds, validity=self.cfg.validity,
                       single_turn_check=self.cfg.single_turn_check)
        results = pool.map(work, list(genomes)) if pool else [work(g) for g in genomes]
        for g, res in zip(genomes, results):
            self.archive.try_insert(g, res["descriptors"], res["fitness"], res["meta"])
        self.evals += len(genomes)
        self._last_batch = len(genomes)
        self.history.append(dict(evals=self.evals, time_s=round(time.time() - self._t0, 2),
                                 **self.archive.stats()))

    def run(self, log=print):
        cfg, self._t0 = self.cfg, time.time()
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=1), encoding="utf-8")
        pool = Pool(cfg.workers) if cfg.workers > 1 else None
        try:
            if self.evals == 0:
                self.pending_init = initial_genomes(cfg, self.rng)
            while self.pending_init is not None and len(self.pending_init):
                take = min(cfg.batch, len(self.pending_init), cfg.budget - self.evals)
                if take <= 0:
                    break
                batch, self.pending_init = self.pending_init[:take], self.pending_init[take:]
                self._evaluate(batch, pool)
                self._maybe_checkpoint(log)
            while self.evals < cfg.budget:
                if not self.archive.cells:
                    raise RuntimeError("no valid genome after initialisation -- nothing to mutate")
                n = min(cfg.batch, cfg.budget - self.evals)
                children = np.array([mutate(self.archive.random_elite(self.rng)["genome"], cfg, self.rng)
                                     for _ in range(n)])
                self._evaluate(children, pool)
                self._maybe_checkpoint(log)
        finally:
            if pool:
                pool.close()
                pool.join()
        self.checkpoint("final")
        return self.archive

    # -- checkpoints ------------------------------------------------------------------------ #
    def _maybe_checkpoint(self, log):
        s = self.history[-1]
        log(f"evals {s['evals']:5d}  filled {s['filled']:3d}  best {s['best_fitness']}  "
            f"qd {s['qd_score']:.1f}  ({s['time_s']:.0f} s)")
        every, before = self.cfg.checkpoint_every, self.evals - self._last_batch
        if self.evals // every > before // every:        # crossed a multiple of checkpoint_every
            self.checkpoint(f"{self.evals:05d}")

    def checkpoint(self, tag: str):
        self.archive.save(self.out / f"archive_{tag}.json")
        state = dict(evals=self.evals, rng_state=self.rng.bit_generator.state, archive=f"archive_{tag}.json",
                     pending_init=None if self.pending_init is None else self.pending_init.tolist())
        (self.out / "state.json").write_text(json.dumps(state), encoding="utf-8")
        (self.out / "history.json").write_text(json.dumps(self.history, indent=0), encoding="utf-8")

    @classmethod
    def resume(cls, out_dir, cfg: Config | None = None) -> "MapElites":
        out = Path(out_dir)
        saved = Config(**json.loads((out / "config.json").read_text(encoding="utf-8")))
        cfg = cfg or saved
        keep = ("seed", "n_init", "batch", "sigma_frac", "genome_bounds", "validity", "single_turn_check")
        if any(getattr(cfg, k) != getattr(saved, k) for k in keep):
            raise ValueError("resume must keep " + ", ".join(keep))
        me = cls(cfg, out)
        state = json.loads((out / "state.json").read_text(encoding="utf-8"))
        me.archive = Archive.load(out / state["archive"])
        me.evals = int(state["evals"])
        me.rng.bit_generator.state = state["rng_state"]
        me.pending_init = None if state["pending_init"] is None else np.array(state["pending_init"])
        me.history = json.loads((out / "history.json").read_text(encoding="utf-8"))
        return me


def parse_bounds(specs) -> dict:
    """['t_climb=0,1.0', ...] -> {'t_climb': (0.0, 1.0), ...}; checked by problem.genome_box."""
    out = {}
    for spec in specs or []:
        name, sep, rng = spec.partition("=")
        parts = rng.split(",")
        if not sep or len(parts) != 2:
            raise ValueError(f"--bound {spec!r}: expected NAME=LOW,HIGH")
        name = name.strip()
        if name in out:
            raise ValueError(f"--bound {name} given twice")
        try:
            out[name] = (float(parts[0]), float(parts[1]))
        except ValueError:
            raise ValueError(f"--bound {spec!r}: LOW and HIGH must be numbers") from None
    P.genome_box(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run folder under MAP_Elites/runs/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--bound", action="append", metavar="NAME=LOW,HIGH",
                    help="override one genome limit for a new run (repeatable)")
    ap.add_argument("--validity", choices=P.VALIDITY_DETECTORS, default=None,
                    help="detector that decides validity for a new run (default section04); "
                         "a resumed run keeps the one in its config.json")
    ap.add_argument("--single-turn-check", action="store_true",
                    help="with --validity ppo_track: a genome is valid only if the vehicle turned once")
    a = ap.parse_args()
    out = HERE / "runs" / a.name
    try:
        overrides = parse_bounds(a.bound)
    except ValueError as e:
        ap.error(str(e))
    if a.resume and overrides:
        ap.error("--bound cannot be used with --resume: the genome box is fixed by the run's config.json")
    if a.resume and (a.validity is not None or a.single_turn_check):
        ap.error("--validity and --single-turn-check cannot be used with --resume: they are fixed by the run's config.json")
    if a.single_turn_check and a.validity != "ppo_track":
        ap.error("--single-turn-check needs --validity ppo_track (the §04 detector already checks one turn)")
    if a.resume:
        saved = json.loads((out / "config.json").read_text(encoding="utf-8"))
        me = MapElites.resume(out, Config(**{**saved, "budget": a.budget, "workers": a.workers}))
    else:
        if (out / "state.json").exists():
            raise SystemExit(f"{out} already has a run; use --resume or another --name")
        me = MapElites(Config(seed=a.seed, budget=a.budget, workers=a.workers,
                              genome_bounds={**P.GENOME_BOUNDS, **overrides},
                              validity=a.validity or "section04",
                              single_turn_check=a.single_turn_check), out)
        print("validity:", me.cfg.validity, "+ single-turn check" if me.cfg.single_turn_check else "")
        if overrides:
            print("genome box overrides:", overrides)
    archive = me.run()
    print("final:", archive.stats())


if __name__ == "__main__":
    main()
