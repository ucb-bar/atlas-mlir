"""Selected Atlas DMA macro binding for Merlin's checked native candidate DAG.

One descriptor means a DMA launch followed by its channel wait. The native
selector owns the graph, physical placement, ordering, and feedback; this OOT
module only materializes the selected physical macro operations. This is a
diagnostic movement subset, not a complete Atlas target compiler.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import importlib.util
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from merlin.semantic_compiler.allocate import StorageBank
from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.rules import InstructionDescriptor
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.snapshot import NativeSnapshot, NativeTargetProfile
from merlin.semantic_compiler.verify import check_selection


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class MovementContract:
    record: dict[str, object]

    @classmethod
    def load(cls) -> MovementContract:
        path = importlib.resources.files("atlas_native_support").joinpath("movement_contract.json")
        record = json.loads(path.read_text())
        expected = {
            "schema", "scope", "rtl_revision", "source_sha256", "block_bytes", "dma_beat_bytes",
            "scalar_word_bytes", "dram_window_bytes", "vmem_capacity_bytes", "dram_program_base",
            "load_channel", "store_channel",
        }
        if set(record) != expected or record["schema"] != "atlas.native_movement_contract.v1":
            raise ValueError("unrecognized selected movement contract")
        for name in ("block_bytes", "dma_beat_bytes", "scalar_word_bytes", "dram_window_bytes",
                     "vmem_capacity_bytes", "dram_program_base", "load_channel", "store_channel"):
            if type(record[name]) is not int or record[name] < 0:
                raise ValueError(f"invalid selected target geometry: {name}")
        block = record["block_bytes"]
        if not record["dma_beat_bytes"] or not record["scalar_word_bytes"] or (
            not block or block % record["dma_beat_bytes"] or block % record["scalar_word_bytes"]
        ):
            raise ValueError("movement block must align to DMA beats and scalar words")
        if any(record[name] % block for name in ("dram_window_bytes", "vmem_capacity_bytes")):
            raise ValueError("selected storage capacities must divide into movement blocks")
        if record["load_channel"] == record["store_channel"] or any(
            not 0 <= record[name] < 8 for name in ("load_channel", "store_channel")
        ):
            raise ValueError("selected DMA channels must be distinct and in range")
        if not isinstance(record["source_sha256"], dict) or set(record["source_sha256"]) != {
            "src/main/scala/atlas/scalar/Instructions.scala",
            "src/main/scala/atlas/scalar/ScalarCore.scala",
            "src/main/scala/atlas/common/VmemParams.scala",
            "src/main/scala/atlas/common/DmaParams.scala",
            "src/main/scala/diplomatic/memory/DMA.scala",
            "src/main/scala/diplomatic/top/AtlasCore.scala",
            "baremetal/assembler.py",
        } or any(not isinstance(value, str) or len(value) != 64 for value in record["source_sha256"].values()):
            raise TypeError("selected RTL/assembler source identities are absent")
        return cls(record)

    @property
    def identity(self) -> str:
        return "atlas-movement-" + _sha(_canonical(self.record))

    @property
    def block_bytes(self) -> int:
        return int(self.record["block_bytes"])

    def verify_source(self, rtl_root: Path) -> Path:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=rtl_root,
                                  text=True, capture_output=True, check=True).stdout.strip()
        if revision != self.record["rtl_revision"]:
            raise ValueError("Atlas RTL checkout is not the selected revision")
        for relative, digest in self.record["source_sha256"].items():
            source = rtl_root / relative
            if _sha(source.read_bytes()) != digest:
                raise ValueError(f"selected Atlas source changed: {relative}")
        return rtl_root / "baremetal/assembler.py"


def movement_profile(contract: MovementContract) -> NativeTargetProfile:
    block = contract.block_bytes
    raw = "raw-byte-copy"
    descriptors = (
        InstructionDescriptor(
            "dma_load_wait", "identity", ("external",), "vmem", "i8", raw, (1,),
            input_dtypes=("i8",), input_numerical_policies=(raw,),
            value_preserving_copy=True,
        ),
        InstructionDescriptor(
            "dma_store_wait", "identity", ("vmem",), "external", "i8", raw, (1,),
            input_dtypes=("i8",), input_numerical_policies=(raw,),
            value_preserving_copy=True,
        ),
    )
    banks = (
        StorageBank("external", "dram", contract.record["dram_window_bytes"] // block, "block"),
        StorageBank("vmem", "vmem", contract.record["vmem_capacity_bytes"] // block, "block"),
    )
    return NativeTargetProfile(contract.identity, descriptors, banks)


def _load_selected_assembler(path: Path):
    spec = importlib.util.spec_from_file_location("selected_atlas_assembler", path)
    if spec is None or spec.loader is None:
        raise ValueError("selected Atlas assembler is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "assemble", None)):
        raise TypeError("selected Atlas assembler lacks assemble")
    return module


def _verify_native_dependencies(snapshot: NativeSnapshot) -> None:
    requirements_path = importlib.resources.files("atlas_native_support").joinpath("requirements.json")
    requirements = json.loads(requirements_path.read_text())
    if requirements.get("schema") != "atlas.native_movement_requirements.v1" or requirements.get("act_required"):
        raise ValueError("selected native dependency manifest is invalid")
    if snapshot.manifest["source_revision"] != requirements["merlin_revision"] or (
        snapshot.manifest["cargo_lock_sha256"] != requirements["merlin_egg_cargo_lock_sha256"]
    ) or version("z3-solver") != requirements["z3_solver_version"]:
        raise ValueError("native compiler or solver dependency differs from selected pins")


def compile_movement(
    snapshot: NativeSnapshot,
    request: KernelRequest,
    *,
    fixed_inputs: dict[str, int],
    fixed_outputs: tuple[int, ...],
    rtl_root: Path,
    destination: Path,
    limits: SearchLimits | None = None,
    contract: MovementContract | None = None,
) -> dict[str, object]:
    """Compile a typed raw-byte region without runtime data or reference outputs."""
    if destination.exists():
        raise FileExistsError(f"fresh compilation output required: {destination}")
    contract = contract or MovementContract.load()
    expected_profile = movement_profile(contract)
    _verify_native_dependencies(snapshot)
    if snapshot.profile.digest() != expected_profile.digest() or request.target_identity != contract.identity:
        raise ValueError("native snapshot/request differs from selected Atlas contract")
    assembler_path = contract.verify_source(rtl_root)
    if request.lowering_policy != "strict-native" or request.constants:
        raise ValueError("diagnostic movement compiler admits strict-native regions without constants")
    for node in request.nodes:
        if node.type.shape != (contract.block_bytes,) or node.type.dtype != "i8" or (
            node.type.numerical_policy != "raw-byte-copy"
        ):
            raise ValueError("movement region needs one raw byte block per value")
        if node.op not in {"input", "identity"}:
            raise ValueError("movement region contains unadmitted computation")
    if len(fixed_outputs) != len(request.outputs) or set(fixed_inputs) != {
        node.id for node in request.nodes if node.op == "input"
    }:
        raise ValueError("movement ABI must bind all ordered inputs and outputs")
    effective_limits = limits or SearchLimits()
    selected = snapshot.select(request, fixed_inputs=fixed_inputs, fixed_outputs=fixed_outputs,
                               limits=effective_limits)
    if selected.status != "selected" or selected.graph is None or selected.allocation is None or (
        selected.candidate is None or selected.rules is None or selected.exploration is None
    ):
        raise RuntimeError(f"native movement selection failed: {selected.status}: {selected.reason}")
    replay = check_selection(
        request, expected_profile.descriptors, selected.rules, selected.exploration,
        selected.candidate, selected.graph, selected.allocation, expected_profile.banks,
        fixed_inputs=fixed_inputs, fixed_outputs=fixed_outputs,
    )
    if not replay.valid or replay.fingerprint != selected.check_fingerprint:
        raise ValueError("native selected graph changed before Atlas materialization")

    addresses = selected.allocation.addresses
    graph = selected.graph
    block = contract.block_bytes
    word_bytes = int(contract.record["scalar_word_bytes"])
    dram_base = int(contract.record["dram_program_base"])
    lines = ["LI x5, 0", "DMA.CONFIG x5, 0", f"LI x2, {block}"]
    for value_id in selected.allocation.order:
        value = graph.value(value_id)
        if value.kind != "instruction" or len(value.children) != 1:
            raise ValueError("unqualified selected movement instruction")
        descriptor = selected.rules.symbols[value.symbol]["descriptor"]
        child = graph.value(value.children[0])
        if descriptor["name"] == "dma_load_wait" and (child.storage, value.storage) == (
            "external", "vmem"
        ):
            dram_address = dram_base + block * addresses[child.id]
            vmem_word = block * addresses[value_id] // word_bytes
            lines.extend((f"LI x6, {vmem_word}", f"LI x1, {dram_address}",
                          f"DMA.LOAD x6, x1, x2, {contract.record['load_channel']}",
                          f"DMA.WAIT {contract.record['load_channel']}"))
        elif descriptor["name"] == "dma_store_wait" and (child.storage, value.storage) == (
            "vmem", "external"
        ):
            dram_address = dram_base + block * addresses[value_id]
            vmem_word = block * addresses[child.id] // word_bytes
            lines.extend((f"LI x6, {vmem_word}", f"LI x1, {dram_address}",
                          f"DMA.STORE x1, x6, x2, {contract.record['store_channel']}",
                          f"DMA.WAIT {contract.record['store_channel']}"))
        else:
            raise ValueError("selected instruction has no Atlas movement binding")
    lines.extend(("LI x1, 1", "CSRRW x0, 0xC10, x1", "ECALL"))
    assembly = "\n".join(lines) + "\n"
    words = tuple(_load_selected_assembler(assembler_path).assemble(assembly))
    if not words or len(words) * 4 > 0x20000:
        raise ValueError("selected movement program exceeds Atlas instruction memory")
    binary = b"".join(word.to_bytes(4, "little") for word in words)
    execution_plan = {
        "schema": "atlas.native_movement_execution_plan.v1",
        "scope": "AtlasCore diagnostic; external host launch and completion management required",
        "engine": "merlin_native",
        "target_identity": contract.identity,
        "request_digest": request.digest(),
        "program": {"file": "program.bin", "sha256": _sha(binary),
                    "word_count": len(words), "word_bytes": 4, "endianness": "little"},
        "inputs": [
            {"source": node.id, "byte_address": dram_base + block * fixed_inputs[node.id],
             "byte_length": block, "preserve": True}
            for node in request.nodes if node.op == "input"
        ],
        "outputs": [
            {"source": output, "position": index,
             "byte_address": dram_base + block * fixed_outputs[index],
             "byte_length": block}
            for index, output in enumerate(request.outputs)
        ],
        "constants": [],
        "completion": "DMA channel waits followed by Atlas halt",
    }
    plan_bytes = _canonical(execution_plan) + b"\n"
    manifest: dict[str, object] = {
        "schema": "atlas.native_movement_compilation.v1",
        "status": "diagnostic_program",
        "engine": "merlin_native",
        "scope": "raw 128-byte DMA movement on selected AtlasCore; no full target compiler claim",
        "target_identity": contract.identity,
        "request_digest": request.digest(),
        "candidate_digest": selected.candidate.digest(),
        "check_fingerprint": replay.fingerprint,
        "source_revision": contract.record["rtl_revision"],
        "contract_sha256": _sha(_canonical(contract.record)),
        "assembler_sha256": _sha(assembler_path.read_bytes()),
        "binary_sha256": _sha(binary),
        "execution_plan_sha256": _sha(plan_bytes),
        "instruction_words": len(words),
        "selected_instructions": len(selected.allocation.order),
        "candidate_attempts": selected.candidate_attempts,
        "ordering_attempts": selected.ordering_attempts,
        "search_limits": asdict(effective_limits),
        "fixed_inputs": fixed_inputs,
        "fixed_outputs": fixed_outputs,
        "addresses": addresses,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="atlas-native-movement-", dir=destination.parent) as temp:
        root = Path(temp) / "compilation"
        root.mkdir()
        (root / "program.S").write_text(assembly)
        (root / "program.bin").write_bytes(binary)
        (root / "execution_plan.json").write_bytes(plan_bytes)
        (root / "request.json").write_bytes(_canonical(request.record()) + b"\n")
        (root / "selected_graph.json").write_bytes(_canonical({
            "values": [asdict(value) for value in graph.values],
            "outputs": graph.outputs,
            "order": selected.allocation.order,
        }) + b"\n")
        (root / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
        root.rename(destination)
    return manifest


def verify_movement_artifact(
    snapshot: NativeSnapshot,
    destination: Path,
    *,
    rtl_root: Path,
    expected_request_digest: str,
    contract: MovementContract | None = None,
) -> dict[str, object]:
    """Rebuild from the exact compiler inputs and compare every published stage byte."""
    manifest = json.loads((destination / "manifest.json").read_text())
    request = KernelRequest.from_record(json.loads((destination / "request.json").read_text()))
    if request.digest() != expected_request_digest or manifest.get("request_digest") != expected_request_digest:
        raise ValueError("compiled request changed from evaluator-selected input")
    if manifest.get("engine") != "merlin_native" or manifest.get("status") != "diagnostic_program":
        raise ValueError("compiled engine or stage status changed")
    inputs = manifest["fixed_inputs"]
    outputs = tuple(manifest["fixed_outputs"])
    limits = SearchLimits(**manifest["search_limits"])
    if not isinstance(inputs, dict) or not all(type(value) is int for value in inputs.values()) or (
        not all(type(value) is int for value in outputs)
    ):
        raise ValueError("compiled fixed I/O ABI changed")
    with tempfile.TemporaryDirectory(prefix="atlas-native-recheck-", dir=destination.parent) as temp:
        replay_path = Path(temp) / "replay"
        replay = compile_movement(
            snapshot, request, fixed_inputs=inputs, fixed_outputs=outputs,
            rtl_root=rtl_root, destination=replay_path, contract=contract, limits=limits,
        )
        for name in ("manifest.json", "request.json", "selected_graph.json", "program.S",
                     "program.bin", "execution_plan.json"):
            if (destination / name).read_bytes() != (replay_path / name).read_bytes():
                raise ValueError(f"compiled movement artifact changed after verification: {name}")
    return replay
