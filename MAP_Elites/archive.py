"""MAP-Elites archive (M3): a 10 x 10 grid over the two behaviour descriptors.

Each cell keeps at most one elite: the best valid genome whose descriptors fall in it.
Insertion rule: a valid candidate enters an empty cell, or replaces the elite if its
fitness is STRICTLY higher (ties keep the incumbent, so the archive never churns).
Invalid candidates (no descriptors / no finite fitness) never enter.
Descriptor values outside the bounds are placed in the nearest edge cell.

QD-score = sum over filled cells of (fitness - FITNESS_FLOOR). FITNESS_FLOOR is a fixed
lower bound of the fitness, so every elite adds a positive amount and the score grows
with both coverage and quality (with negative fitness, a plain sum would fall as the
archive fills).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from problem import DESCRIPTOR_BOUNDS, DESCRIPTOR_NAMES, GRID_SHAPE

# fitness = -(effort + altitude loss) >= -(4 motors * (1 + 0.054)^2 * 5 s + 5 m) ~ -27.2
FITNESS_FLOOR = -30.0


class Archive:
    def __init__(self, bounds=DESCRIPTOR_BOUNDS, shape=GRID_SHAPE):
        self.bounds = tuple(tuple(float(v) for v in b) for b in bounds)
        self.shape = tuple(int(n) for n in shape)
        if len(self.bounds) != len(self.shape) or any(lo >= hi for lo, hi in self.bounds):
            raise ValueError("need one (low < high) bound per grid dimension")
        self.cells: dict[tuple, dict] = {}
        self.counts = dict(new=0, improved=0, rejected=0, invalid=0)

    def cell_of(self, descriptors) -> tuple:
        """Grid index of a descriptor vector; out-of-range values go to the edge cells.
        Cells are [lo + k w, lo + (k+1) w). The 1e-9 guard keeps values that lie exactly on a
        cell edge in the upper cell despite floating-point rounding -- this matters because
        rotation durations are exact multiples of 5 ms and cell edges are multiples of 0.06 s."""
        idx = []
        for v, (lo, hi), n in zip(descriptors, self.bounds, self.shape):
            k = int(np.floor((float(v) - lo) / (hi - lo) * n + 1e-9))
            idx.append(min(max(k, 0), n - 1))
        return tuple(idx)

    def try_insert(self, genome, descriptors, fitness, meta: dict | None = None) -> str:
        """Returns 'new', 'improved', 'rejected' (worse or equal) or 'invalid'."""
        if (descriptors is None or fitness is None or not np.isfinite(fitness)
                or len(descriptors) != len(self.shape) or not np.all(np.isfinite(descriptors))):
            self.counts["invalid"] += 1
            return "invalid"
        cell = self.cell_of(descriptors)
        incumbent = self.cells.get(cell)
        if incumbent is not None and not fitness > incumbent["fitness"]:
            self.counts["rejected"] += 1
            return "rejected"
        self.cells[cell] = dict(genome=[float(v) for v in genome],
                                descriptors=[float(v) for v in descriptors],
                                fitness=float(fitness), meta=dict(meta or {}))
        outcome = "new" if incumbent is None else "improved"
        self.counts[outcome] += 1
        return outcome

    def random_elite(self, rng) -> dict:
        if not self.cells:
            raise ValueError("archive is empty")
        keys = sorted(self.cells)                       # deterministic order for a given rng
        return self.cells[keys[rng.integers(len(keys))]]

    def stats(self) -> dict:
        # sorted cell order: identical floating-point sums however the archive was built or loaded
        f = np.array([self.cells[c]["fitness"] for c in sorted(self.cells)], float)
        n_cells = int(np.prod(self.shape))
        return dict(filled=len(self.cells), coverage=len(self.cells) / n_cells,
                    best_fitness=float(f.max()) if f.size else None,
                    mean_fitness=float(f.mean()) if f.size else None,
                    qd_score=float(np.sum(f - FITNESS_FLOOR)), **self.counts)

    def fitness_grid(self) -> np.ndarray:
        """Fitness per cell (NaN where empty), for heatmaps."""
        g = np.full(self.shape, np.nan)
        for cell, e in self.cells.items():
            g[cell] = e["fitness"]
        return g

    # -- persistence -------------------------------------------------------------------- #
    def to_dict(self) -> dict:
        return dict(descriptor_names=list(DESCRIPTOR_NAMES), bounds=[list(b) for b in self.bounds],
                    shape=list(self.shape), counts=dict(self.counts), fitness_floor=FITNESS_FLOOR,
                    elites=[dict(cell=list(c), **e) for c, e in sorted(self.cells.items())])

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Archive":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        a = cls(d["bounds"], d["shape"])
        a.counts = {k: int(v) for k, v in d["counts"].items()}
        for e in d["elites"]:
            cell = tuple(e.pop("cell"))
            if a.cell_of(e["descriptors"]) != cell:
                raise ValueError(f"stored cell {cell} does not match its descriptors")
            a.cells[cell] = e
        return a
