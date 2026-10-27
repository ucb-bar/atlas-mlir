"""Shared native validation setup; observations never select timing facts."""
from contextlib import closing, contextmanager
from io import StringIO
from pathlib import Path
import sys
from unittest.mock import Mock

from atlas_derive_scratchpad import source_path


def safe_path(value):
    return source_path(value, ValueError, "symlink paths are not supported", absolute=True)


def load_model(model, message="imported native model differs from --model", error=ValueError):
    root = safe_path(model)
    sys.path.insert(0, str(root))
    import npu_model
    import torch
    if safe_path(Path(npu_model.__file__).parent.parent) != root:
        raise error(message)
    torch.set_num_threads(1)
    return root


def instructions(assembly):
    from npu_model.util.converter import stream_to_instrs
    return list(stream_to_instrs(StringIO(assembly if isinstance(assembly, str) else "\n".join(assembly))))


def uops(assembly):
    from npu_model.software.instruction import Uop
    return [Uop(insn, pc=index * 4) for index, insn in enumerate(instructions(assembly))]


def native_state(cfg):
    from npu_model.hardware.arch_state import ArchState
    return closing(ArchState(cfg.arch_state_config))


def native_unit(cfg, state, engine, name=None, logger=None):
    from npu_model.hardware.dma import DmaExecutionUnit
    from npu_model.hardware.lsu import LoadStoreUnit
    from npu_model.hardware.mxu import MatrixExecutionUnitInner, MatrixExecutionUnitSystolic
    from npu_model.hardware.vpu import VectorExecutionUnit
    from npu_model.hardware.xlu import CrossLaneExecutionUnit
    from npu_model.logging.logger import Logger
    cls = dict(dma=DmaExecutionUnit, lsu=LoadStoreUnit, vlsu=LoadStoreUnit,
               scalar_lsu=LoadStoreUnit, vpu=VectorExecutionUnit,
               xlu=CrossLaneExecutionUnit, mxu0=MatrixExecutionUnitSystolic, mxu1=MatrixExecutionUnitInner)[engine]
    return cls(name or engine, Mock(spec=Logger) if logger is None else logger, state, config=cfg)


@contextmanager
def native_core(cfg, assembly=None, logger=None):
    from npu_model.hardware.core import Core
    from npu_model.logging.logger import Logger
    from npu_model.software.program import InstantiableProgram
    core = Core(config=cfg, logger=Mock(spec=Logger) if logger is None else logger)
    try:
        if assembly is not None:
            core.load_program(InstantiableProgram(instructions(assembly)))
        yield core
    finally:
        core.close()
