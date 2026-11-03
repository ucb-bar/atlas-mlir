"""Physical Atlas instruction execution against an explicit, pinned RTL model.

This is functional execution, not a Core timing measurement. No arithmetic is
reimplemented here: each assembled physical instruction executes on ArchState.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import time

from atlas_path_guard import checked

MODEL_PIN = "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5"
UNARY_TABLES = ('cos', 'cube', 'exp', 'exp2', 'log', 'rcp', 'relu', 'sin', 'sqrt', 'square', 'tanh')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@dataclass
class Execution:
    arena: bytes
    output: bytes
    elapsed_seconds: float
    instructions: int
    program_words: int
    program_sha256: str
    observed: dict
    halt_pc: int
    model: dict
    implementation_source_sha256: dict
    cycles: None = None

    def metadata(self):
        return {key: value for key, value in asdict(self).items() if key not in ("arena", "output")}


class Executor:
    def __init__(self, model_root):
        self.root = checked(model_root)
        self._revision()
        # Refuse an already imported alternate model before importing its children.
        self.binding()
        sys.path.insert(0, str(self.root))
        import torch
        from npu_model.configs import isa_definition  # Register the physical ISA.
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.hardware.arch_state import ArchState
        from npu_model.isa import SBType, UJType
        from npu_model.isa_patterns import TensorComputeBinary, TensorComputeUnary
        from npu_model.isa_types import EXU
        from npu_model.util.converter import input_to_program

        torch.set_num_threads(1)
        self.torch, self.config, self.state = torch, DefaultHardwareConfig, ArchState
        self.branches, self.vector_ops, self.vector = (SBType, UJType), (TensorComputeBinary, TensorComputeUnary), EXU.VECTOR
        self.binary, self.parse = TensorComputeBinary, input_to_program
        self.sources = self.binding()
        self.implementation_sources = {str(path): sha(path.read_bytes()) for path in
                                       (checked(__file__), checked(Path(__file__).with_name('atlas_path_guard.py')))}

    def _revision(self):
        revision = subprocess.check_output(["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True).strip()
        if revision != MODEL_PIN:
            raise ValueError(f"npu_model revision is {revision}, expected {MODEL_PIN}")
        return revision

    def binding(self, extra_sources=()):
        imported = {}
        for name, module in list(sys.modules.items()):
            if name == "npu_model" or name.startswith("npu_model."):
                filename = getattr(module, "__file__", None)
                if filename:
                    path = checked(filename)
                    if not path.is_relative_to(self.root):
                        raise ValueError(f"model imported outside selected clone: {name}")
                    imported[str(path.relative_to(self.root))] = path
        tables = {f'npu_model/configs/data/{name}.bin.gz': checked(self.root / f'npu_model/configs/data/{name}.bin.gz')
                  for name in UNARY_TABLES}
        curve = 'npu_model/configs/memory_curves/ee290sim_vcs_probe.json'
        extra = {}
        for filename in extra_sources:
            path = checked(filename)
            if not path.is_relative_to(self.root):
                raise ValueError('recorded model dependency is outside selected original clone')
            extra[str(path.relative_to(self.root))] = path
        paths = dict(sorted({**extra, **imported, **tables, curve: checked(self.root / curve)}.items()))
        objects = [f'{MODEL_PIN}:{name}' for name in paths]
        raw = subprocess.run(['git', '-C', str(self.root), 'cat-file', '--batch'],
                             input=('\n'.join(objects) + '\n').encode(), stdout=subprocess.PIPE, check=True).stdout
        cursor, sources = 0, {}
        for name, path in paths.items():
            end = raw.index(b'\n', cursor)
            header = raw[cursor:end].split()
            if len(header) != 3 or header[1] != b'blob':
                raise ValueError('model dependency absent from pinned revision: ' + name)
            size = int(header[2])
            original = raw[end + 1:end + 1 + size]
            actual = path.read_bytes()
            if actual != original:
                raise ValueError('model dependency differs from pinned revision: ' + name)
            sources[name] = sha(actual)
            cursor = end + size + 2
        return {"revision": self._revision(), "root": str(self.root), "numerics": "rtl",
                "imported_source_sha256": {name: sources[name] for name in imported},
                "unary_table_sha256": {name: sources[name] for name in tables},
                "curve_sha256": sources[curve], "source_sha256": sources}

    def _bookend(self):
        current = self.binding()
        if (any(current['source_sha256'].get(name) != value for name, value in self.sources['source_sha256'].items())
                or any(sha(checked(path).read_bytes()) != value for path, value in self.implementation_sources.items())):
            raise ValueError("execution model or helper source bookend changed")
        self.sources = current
        return current

    def assemble(self, source):
        self._bookend()
        program = self.parse(io.StringIO(source))
        for pc, insn in enumerate(program.instructions):
            if insn.exu == self.vector and isinstance(insn, self.vector_ops) and insn.mnemonic != "vmov":
                if int(insn.vd) % 2:
                    raise ValueError(f"odd VPU pair destination at word {pc}: {insn}")
            if insn.exu == self.vector and isinstance(insn, self.binary):
                a, b = int(insn.vs1), int(insn.vs2)
                if a != b and a % 32 == b % 32:
                    raise ValueError(f"binary VPU source read-port alias at word {pc}: {insn}")
        words = program.assemble()
        if len(words) > 32768:
            raise ValueError("assembled program exceeds physical IMEM")
        self._bookend()
        return program, sha(b"".join(word.to_bytes(4, "little") for word in words))

    def execute(self, source, arena, output_offset, output_bytes, *, expected_arena=None, requested_dma=None,
                program_sha256=None, max_instructions=5_000_000, seed=42, dram_base=0x90000000):
        if (output_offset < 0 or output_bytes < 0 or output_offset + output_bytes > len(arena)
                or dram_base < 0 or dram_base + len(arena) > 1 << 32):
            raise ValueError("output region or staged DMA arena is outside bounds")
        if expected_arena is not None and len(expected_arena) != len(arena):
            raise ValueError("expected full arena size differs")
        if requested_dma is not None and (set(requested_dma) != {"load_bytes", "store_bytes"}
                or any(type(value) is not int or value < 0 for value in requested_dma.values())):
            raise ValueError("invalid requested DMA byte counts")
        program, digest = self.assemble(source)
        if program_sha256 is not None and digest != program_sha256:
            raise ValueError("assembled program binary identity differs")
        config = replace(self.config().arch_state_config, numerics="rtl", randomize_init=True, init_seed=seed)
        state = self.state(config)
        started = time.perf_counter()
        pc, pending, executed = 0, None, 0
        events = Counter(load_bytes=0, store_bytes=0)
        try:
            state.write_dram(dram_base, self.torch.frombuffer(bytearray(arena), dtype=self.torch.uint8))
            while 0 <= pc < len(program.instructions):
                if executed >= max_instructions:
                    raise RuntimeError(f"functional instruction bound reached at word {pc}")
                insn = program.instructions[pc]
                if pending is not None and (isinstance(insn, self.branches) or insn.mnemonic == "jalr"):
                    raise ValueError(f"branch in delay slot at word {pc}")
                state.execute_pc, state.pc, state.npc, state.redirect_requested = pc, pc, pc + 1, False
                if insn.mnemonic in ("dma.load", "dma.store"):
                    events["load_bytes" if insn.mnemonic == "dma.load" else "store_bytes"] += state.read_xrf(insn.rs2)
                insn.exec(state)
                executed += 1
                if state.halted:
                    break
                next_pc = pending if pending is not None else pc + 1
                pending = state.npc if state.redirect_requested else None
                pc = next_pc
            if not state.halted or state.halt_reason != "ecall":
                raise AssertionError(f"run did not reach ECALL: pc={state.execute_pc}")
            actual = bytes(state.read_dram(dram_base, len(arena)).tolist())
            end = output_offset + output_bytes
            if actual[:output_offset] != arena[:output_offset] or actual[end:] != arena[end:]:
                raise AssertionError("input or output guard bytes changed")
            if expected_arena is not None and actual != expected_arena:
                raise AssertionError("full staged arena differs from expected")
            if requested_dma is not None and dict(events) != requested_dma:
                raise AssertionError(f"DMA byte counts differ: {dict(events)} != {requested_dma}")
            model = self._bookend()
            return Execution(actual, actual[output_offset:end], time.perf_counter() - started, executed,
                             len(program), digest, dict(events), state.execute_pc, model, dict(self.implementation_sources))
        finally:
            state.close()
