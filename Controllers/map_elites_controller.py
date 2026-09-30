"""The final MAP-Elites flip controller, as a ready-to-use course controller.

    from map_elites_controller import load_final_controller
    ctrl = load_final_controller()          # ThreePhaseFlip with the selected genome
    ctrl.reset(seed=0)
    action, info = ctrl.act(observation)    # course interface: reset / act

The selected genome and its provenance (run, cell, fitness, selection rule) are stored in
map_elites_final.json next to this file; the controller itself is three_phase_flip.py.
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from three_phase_flip import FlipParams, ThreePhaseFlip   # noqa: E402

FINAL_PATH = HERE / "map_elites_final.json"


def load_params(path=FINAL_PATH) -> FlipParams:
    """FlipParams from a stored genome; every FlipParams field must be present, finite, and nothing else."""
    genome = json.loads(Path(path).read_text(encoding="utf-8"))["genome"]
    names = [f.name for f in fields(FlipParams)]
    if sorted(genome) != sorted(names):
        raise ValueError(f"genome keys {sorted(genome)} do not match FlipParams fields {sorted(names)}")
    values = [float(genome[n]) for n in names]
    if not np.all(np.isfinite(values)):
        raise ValueError("genome contains non-finite values")
    return FlipParams(*values)


def load_final_controller(path=FINAL_PATH, **kwargs) -> ThreePhaseFlip:
    """The selected MAP-Elites controller (kwargs go to ThreePhaseFlip, e.g. dt)."""
    return ThreePhaseFlip(load_params(path), **kwargs)
