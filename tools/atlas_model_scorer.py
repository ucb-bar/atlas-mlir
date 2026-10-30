"""Execute verified words on the evidence-selected model; no timing surrogate."""
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from atlas_derive_curve import resolve_curve_path
from atlas_model_semantics import check_model_import, source_pins
from atlas_model_timing import configuration, encoded, sha256
from check_program_decode import decode_binary


# Compiler policy: bound native scoring work.
MAX_CYCLES = 200000
TORCH_THREADS = 1


def fifo43_accept(cycle):
    """Synthetic FIFO43 target: four accepted cycles per eleven-cycle period."""
    return cycle % 11 < 4


def bind_fifo43_configuration(cfg):
    # Synthetic target matching the standalone memory responder.
    cfg.dma_memory_backend = "fixed"
    cfg.dma_memory_params = dict(latency=43, cycles_per_beat=1,
                                 accept_fn=fifo43_accept)


def bind_curve_configuration(cfg, evidence):
    """Run exactly the configured curve bytes whose digest the provider retained."""
    path = resolve_curve_path(cfg)
    curve_bytes = path.read_bytes()
    if sha256(curve_bytes) != evidence.get("dma_curve_sha256"):
        raise ValueError("scoring DMA curve differs from provider evidence")
    # Bind checked bytes to prevent a second path resolution/read during Core init.
    cfg.dma_memory_params = dict(cfg.dma_memory_params, curves=json.loads(curve_bytes))


def model_instruction(row, inventory, isa):
    """Bind independently decoded words to model instructions and require identical re-encoding."""
    from npu_model.isa import SBType, SType, UJType, UType, VIType
    from npu_model.isa_patterns import ScalarComputeShamt

    mode = row['mode_id']
    variant = next(v for v in inventory['variants'] if v['id'] == mode)
    name = mode.removesuffix('_ANY') if mode.startswith('DMA_') else variant['model_classes'][0]
    cls, word = getattr(isa, name), row['word_u32']
    operands = {'rd': (word >> 7) & 31, 'rs1': (word >> 15) & 31, 'rs2': (word >> 20) & 31,
                'vd': (word >> 7) & 63, 'vs1': (word >> 13) & 63, 'vs2': (word >> 19) & 63,
                'es1': (word >> 13) & 63, 'ch': (word >> 12) & 7}
    if issubclass(cls, ScalarComputeShamt):
        immediate = (word >> 20) & 31
    elif issubclass(cls, SBType):
        immediate = row['fields']['offset_bytes']
    elif issubclass(cls, UJType):
        immediate = row['fields']['offset']
    elif issubclass(cls, SType):
        immediate = ((word >> 25) << 5) | ((word >> 7) & 31)
    elif issubclass(cls, VIType):
        immediate = word >> 16
    elif issubclass(cls, UType):
        immediate = word >> 12
    else:
        immediate = word >> 20
    operands['imm'] = immediate
    instruction = cls(**{name: operands[name] for name in inspect.signature(cls).parameters})
    if instruction.to_bytecode() != word:
        raise ValueError(f'npu_model instruction differs from word {row["word_index"]}')
    return instruction


@dataclass(frozen=True)
class ModelScorer:
    model: Path
    regions: tuple[Path, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "model", Path(self.model).absolute())
        object.__setattr__(self, "regions", tuple(
            json.loads(Path(path).absolute().read_text()) for path in self.regions))

    def __call__(self, document, profile="curve"):
        if sys.flags.optimize:
            raise ValueError("model scoring requires Python assertions; disable -O/PYTHONOPTIMIZE")
        evidence = document.get("model_evidence", {})
        if evidence:
            facts = Path(evidence["op_timing"]).read_bytes()
            if sha256(facts) != evidence["op_timing_sha256"]:
                raise ValueError("scoring facts differ from provider evidence")
            identity = json.loads(facts).get("model", {})
            if source_pins(self.model, mask_timing=False) != identity.get("source_pins"):
                raise ValueError("scoring model source differs from provider facts")
        if str(self.model) not in sys.path:
            sys.path.insert(0, str(self.model))
        check_model_import(self.model)
        cfg, config = configuration()
        if evidence and sha256(encoded(config)) != identity.get("configuration_sha256"):
            raise ValueError("scoring model configuration differs from provider facts")
        if profile == "fifo43":
            bind_fifo43_configuration(cfg)
        elif profile == "curve":
            bind_curve_configuration(cfg, evidence)
        else:
            raise ValueError("unsupported memory profile")
        import torch
        torch.set_num_threads(TORCH_THREADS)
        from npu_model.configs import isa_definition as isa
        from npu_model.hardware.core import Core
        from npu_model.logging.logger import Logger
        from npu_model.software.program import InstantiableProgram

        words = [row["word_u32"] for row in document["instructions"]]
        inventory = json.loads((Path(__file__).parents[1] / "docs/selected-variant-inventory.json").read_text())
        decoded = decode_binary(inventory, b"".join(word.to_bytes(4, "little") for word in words))
        program = InstantiableProgram([model_instruction(row, inventory, isa)
                                       for row in decoded["instructions"]])
        logger = SimpleNamespace(**{name: lambda *args, **kwargs: None
                                  for name in dir(Logger) if name.startswith("log_")})
        core = Core(cfg, logger)
        try:
            if core.ignore_runtime_errors is not False:
                raise ValueError("model must propagate runtime errors")
            core.load_program(program)
            state = core.arch_state
            for region in self.regions:
                initial = torch.tensor(list(bytes.fromhex(region["initial_hex"])), dtype=torch.uint8)
                state.write_dram(region["address"], initial)
            first = None
            while not state.halted and core.cycle_count < MAX_CYCLES:
                if core.is_finished():
                    raise ValueError("program exhausted before ECALL")
                core.tick()
                if first is None and core.issued_uop is not None:
                    first = core.cycle_count
            if first is None or not state.halted or state.halt_reason != "ecall":
                raise ValueError(f'model did not reach ECALL within {MAX_CYCLES} cycles')
            if any(unit.has_in_flight for unit in core.exus) or any(state.flags):
                raise ValueError("ECALL before engines and DMA drained")
            for region in self.regions:
                if "expected_hex" in region:
                    expected = bytes.fromhex(region["expected_hex"])
                    if bytes(state.read_dram(region["address"], len(expected)).tolist()) != expected:
                        raise ValueError("candidate DRAM outputs, inputs or guards differ from expected bytes")
            return core.cycle_count - first + 1
        finally:
            core.close()
