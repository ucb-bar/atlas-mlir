"""Compute engine timing by simulating control cones of CIRCT HW IR; target data lives in targets/."""
from .control import ControlCircuit
from .ir import load_document
from .runner import extract
from .spec import available_engines, load_spec, load_target, variant
