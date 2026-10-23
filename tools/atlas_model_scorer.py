"""Execute verified words on the evidence-selected model; no timing surrogate."""
from dataclasses import dataclass
from functools import lru_cache
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from atlas_derive_curve import resolve_curve_path
from atlas_model_facts import safe_path
from atlas_model_profiles import bind_synthetic_configuration, scoring_configuration_receipt, synthetic_targets
from atlas_model_schema import LOCK
from atlas_model_semantics import check_model_semantics
from atlas_model_capabilities import check_configuration
from atlas_search_policy import POLICY, fixed_execution_contract
from atlas_model_timing import configuration, encoded, numerical_data, sha256, source_digest
from check_program_decode import decode_binary


@lru_cache(maxsize=1)
def _model_content(root):
    root = safe_path(root)
    check_model_semantics(root)
    sys.path.insert(0, str(root))
    import npu_model
    import torch
    if safe_path(Path(npu_model.__file__).parent.parent) != root:
        raise ValueError("imported npu_model differs from --npu-model")
    torch.set_num_threads(POLICY["engineering"]["scorer"]["torch_threads"])
    source, _ = source_digest(root)
    numerical, _ = numerical_data(root)
    return root, source, numerical


def selected_model(root, *, include_configuration=False):
    root, source, numerical = _model_content(root)
    cfg, values = configuration()
    check_configuration(cfg)
    identity = root, source, numerical, sha256(encoded(values))
    return (*identity, cfg) if include_configuration else identity


def bind_curve_configuration(cfg, evidence):
    """Run exactly the configured curve bytes whose digest the provider retained."""
    path = resolve_curve_path(cfg)
    curve_bytes = path.read_bytes()
    # Older providers admitted only the locked curve, and did not retain a
    # separate digest. Omission must not admit arbitrary replacement bytes.
    expected = evidence.get("dma_curve_sha256", LOCK["curve_sha256"])
    if sha256(curve_bytes) != expected:
        raise ValueError("scoring DMA curve differs from provider evidence")
    # Pass the checked bytes through the backend's supported dict input. This
    # prevents a second implicit path resolution or file read during Core init.
    cfg.dma_memory_params = dict(cfg.dma_memory_params, curves=json.loads(curve_bytes))


def model_instruction(row, inventory, isa):
    """Bind a decoded instruction and require an identical re-encoded word.

    npu_model executes instruction objects. Decode uses the existing RTL
    inventory, independently of the emitter and its JSON operand annotations.
    """
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
        object.__setattr__(self, "model", safe_path(self.model))
        object.__setattr__(self, "regions", tuple(json.loads(safe_path(path).read_text()) for path in self.regions))
        # Snapshot named targets once, including when this scorer is pickled for workers.
        object.__setattr__(self, "_targets", synthetic_targets())
        object.__setattr__(self, "_scorer_policy", dict(POLICY["engineering"]["scorer"]))
        object.__setattr__(self, "_execution_contract", fixed_execution_contract(POLICY["execution_contract"]))

    def configured_profile(self, document, profile):
        """Bind and identify the exact configuration used for native scoring."""
        root, source, numerical, config_hash, cfg = selected_model(self.model, include_configuration=True)
        evidence = document.get("model_evidence", {})
        expected = (evidence.get("source_sha256"), evidence.get("numerical_data_sha256"), evidence.get("configuration_sha256"))
        if (source, numerical, config_hash) != expected:
            raise ValueError("scoring model source, numerical data or configuration differs from provider evidence")
        target = self._targets.get(profile)
        if target is not None:
            bind_synthetic_configuration(cfg, target)
        elif profile == "curve":
            bind_curve_configuration(cfg, evidence)
        else:
            raise ValueError("unsupported memory profile")
        identity = dict(source_sha256=source, numerical_data_sha256=numerical, configuration_sha256=config_hash)
        receipt = scoring_configuration_receipt(cfg, profile, identity, synthetic_target=target)
        return cfg, receipt

    def profile_receipt(self, document, profile):
        return self.configured_profile(document, profile)[1]

    def runtime_receipt(self):
        """Identify the actual scoring interpreter and imported numerical libraries."""
        import platform
        import numpy
        import torch

        _, _, _, _, cfg = selected_model(self.model, include_configuration=True)
        curve = resolve_curve_path(cfg)
        from atlas_model_timing import configuration_values
        values = configuration_values(cfg)
        # The filename is location evidence; the selected bytes are semantic.
        values["dma_memory_params"] = dict(values["dma_memory_params"],
                                           curves={"sha256": sha256(curve.read_bytes())})
        return {"python": platform.python_version(), "python_implementation": platform.python_implementation(),
                "torch": str(torch.__version__), "numpy": str(numpy.__version__),
                "python_optimization": sys.flags.optimize,
                "curve_path": str(curve), "configuration": values}

    def __call__(self, document, profile="curve"):
        if self._scorer_policy["assertions_required"] and sys.flags.optimize:
            raise ValueError("model scoring requires Python assertions; disable -O/PYTHONOPTIMIZE")
        cfg, _ = self.configured_profile(document, profile)
        import torch
        torch.set_num_threads(self._scorer_policy["torch_threads"])
        from npu_model.configs import isa_definition as isa
        from npu_model.hardware.core import Core
        from npu_model.logging.logger import Logger
        from npu_model.software.program import InstantiableProgram

        words = [row["word_u32"] for row in document["instructions"]]
        inventory = json.loads((Path(__file__).parents[1] / "docs/selected-variant-inventory.json").read_text())
        decoded = decode_binary(inventory, b"".join(word.to_bytes(4, "little") for word in words))
        program = InstantiableProgram([model_instruction(row, inventory, isa) for row in decoded["instructions"]])
        logger = SimpleNamespace(**{name: lambda *args, **kwargs: None for name in dir(Logger) if name.startswith("log_")})
        core = Core(cfg, logger)
        try:
            if core.ignore_runtime_errors is not self._execution_contract["ignore_runtime_errors"]:
                raise ValueError("model must propagate runtime errors")
            core.load_program(program)
            state = core.arch_state
            for region in self.regions:
                state.write_dram(region["address"], torch.tensor(list(bytes.fromhex(region["initial_hex"])), dtype=torch.uint8))
            first = None
            while not state.halted and core.cycle_count < self._scorer_policy["max_cycles"]:
                if core.is_finished():
                    raise ValueError("program exhausted before ECALL")
                core.tick()
                if first is None and core.issued_uop is not None:
                    first = core.cycle_count
            if first is None or not state.halted or state.halt_reason != "ecall":
                raise ValueError(f'model did not reach ECALL within {self._scorer_policy["max_cycles"]} cycles')
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
