"""Raw affine layout device contracts preserve every byte and reject mutations."""
from dataclasses import replace
import itertools
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from atlas_mlir_native.binding import DRAM_BASE, TARGET_IDENTITY, _run
from atlas_mlir_native.layout_device import compile_source, prepare_c_package, prepare_source, virtual_program
from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.semantic_compiler.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))


def request(op="reshape", shapes=((3, 7),), shape=(21,), dtype="bf16", mapping=None, axis=None):
    policy = "signed-i8" if dtype == "i8" else "raw-bit-pattern"
    inputs = tuple(SemanticNode(f"input_{i}", "input", (), TensorType(s, dtype, policy), effect="input") for i, s in enumerate(shapes))
    maps = () if mapping is None else (mapping, pointwise_identity_maps(len(shape), 1)[0])
    output = SemanticNode("result", op, tuple(n.id for n in inputs), TensorType(shape, dtype, policy),
                          attrs=() if axis is None else (("axis", axis),), index_maps=maps)
    return KernelRequest((*inputs, output), (output.id,), ("external",), tuple((n.id, "external") for n in inputs), TARGET_IDENTITY, "diagnostic")


def cases():
    immutable = request(shapes=((3,),), shape=(1, 3), dtype="f32")
    constant = replace(immutable.nodes[0], op="constant", effect="constant")
    immutable = replace(immutable, nodes=(constant, immutable.nodes[-1]), input_storages=(),
                        constants=(ConstantBinding(constant.id, "f32-le-raw", "3412c07f0000008001000000"),))
    return {
        "immutable_nan_zero_subnormal": immutable,
        "reverse_bytes": request("gather", ((64,),), (64,), dtype="i8", mapping=IndexMap(1, ((-1,),), (63,))),
        "small_scalar_reshape": request(shapes=((3,),), shape=(1, 3), dtype="f32"),
        "dense_dma": request(shapes=((3, 32),), shape=(96,)),
        "permuted_heads": request("transpose", ((2, 3, 32),), (3, 2, 32), mapping=IndexMap(3, ((0, 1, 0), (1, 0, 0), (0, 0, 1)), (0, 0, 0))),
        "broadcast": request("gather", ((3, 32),), (4, 3, 32), mapping=IndexMap(3, ((0, 1, 0), (0, 0, 1)), (0, 0))),
        "slice": request("gather", ((2, 64),), (2, 32), mapping=IndexMap(2, ((1, 0), (0, 1)), (0, 32))),
        "concat": request("concat", ((2, 1, 32), (2, 2, 32)), (2, 3, 32), axis=1),
        "xlu": request("transpose", ((32, 64),), (64, 32), dtype="i8", mapping=IndexMap(2, ((0, 1), (1, 0)), (0, 0))),
        "ragged_scalar_transpose": request("transpose", ((5, 8),), (8, 5), dtype="i8", mapping=IndexMap(2, ((0, 1), (1, 0)), (0, 0))),
    }


def reference(source, raw):
    result = source.nodes[-1]
    width = prepare_source(source).width
    if result.op in {"reshape", "copy"}:
        return raw[0]
    output = bytearray()
    axis = dict(result.attrs).get("axis")
    for index in itertools.product(*(range(n) for n in result.type.shape)):
        if result.op == "concat":
            coordinate = index[axis]
            for operand, node_id in enumerate(result.inputs):
                shape = source.node(node_id).type.shape
                if coordinate < shape[axis]:
                    location = (*index[:axis], coordinate, *index[axis+1:])
                    break
                coordinate -= shape[axis]
        else:
            operand = 0
            mapping = result.index_maps[0]
            location = tuple(offset + sum(c * i for c, i in zip(row, index)) for row, offset in zip(mapping.coefficients, mapping.offsets))
            shape = source.node(result.inputs[0]).type.shape
        flat = 0
        for extent, coordinate in zip(shape, location):
            flat = flat * extent + coordinate
        output.extend(raw[operand][flat*width:(flat+1)*width])
    return bytes(output)


class LayoutDeviceRecognitionTest(unittest.TestCase):
    def test_modes_are_selected_from_semantics(self):
        selected = {name: [s["mode"] for s in prepare_source(source).segments] for name, source in cases().items()}
        self.assertEqual(selected["xlu"], ["xlu"])
        self.assertEqual(selected["small_scalar_reshape"], ["scalar"])
        self.assertEqual(selected["ragged_scalar_transpose"], ["scalar"])
        self.assertEqual(selected["concat"], ["dma", "dma"])
        self.assertEqual(selected["broadcast"], ["dma"])

    def test_refuses_changed_type_maps_attrs_and_incomplete_source(self):
        source = cases()["broadcast"]
        x, y = source.nodes
        changed = (replace(source, nodes=(x, replace(y, attrs=(("unknown", 1),)))),
                   replace(source, nodes=(x, replace(y, type=replace(y.type, dtype="f32")))) ,
                   replace(source, nodes=(x, replace(y, index_maps=(replace(y.index_maps[0], offsets=(0, 32)), y.index_maps[-1])))))
        for region in changed:
            with self.assertRaises(ValueError):
                prepare_source(region)

    def test_relocated_window_rejects_invalid_bases(self):
        source = cases()["dense_dma"]
        for base in (True, -32, 1, 0x100000000, 0xffffffe0, 0x40000, 0x07ffffe0, 0x08010000, 0x0800ffe0):
            with self.subTest(base=base), self.assertRaises(ValueError):
                prepare_source(source, io_memory_base=base)

    def test_zero_byte_layout_refused_without_tools(self):
        for shapes in (((0,),), ((1,),)):
            with self.assertRaisesRegex(ValueError, "positive static dimensions"):
                source = request(shapes=shapes, shape=(0,))
                prepare_source(source, io_memory_base=0x08000000)

    def test_large_ragged_transpose_rejects_default_budget(self):
        source = request("transpose", ((513, 64),), (64, 513), dtype="i8", mapping=IndexMap(2, ((0, 1), (1, 0)), (0, 0)))
        with self.assertRaisesRegex(ValueError, "default proof work budget"):
            prepare_source(source)


@unittest.skipUnless((BIN / "atlas-opt").is_file(), "select built Atlas tools")
class LayoutDeviceCompilationTest(unittest.TestCase):
    def test_all_modes_compile_export_and_replay(self):
        with tempfile.TemporaryDirectory(prefix="atlas-layout-device-") as directory:
            for label, source in cases().items():
                with self.subTest(label=label):
                    compiled = Path(directory) / label
                    compile_source(None, source, target_source=ROOT, destination=compiled, limits=None)
                    machine = (compiled / "atlas-machine.mlir").read_text()
                    for contract in ("dma", "mxu", "tile", "cfg", "source_memory", "buffer"):
                        self.assertIn("atlas.virtual_" + contract + "_contract", machine)
                    self.assertNotIn("atlas.layout_device_contract", machine)
                    virtual_text = (compiled / "atlas-virtual.mlir").read_text()
                    self.assertNotIn("atlas.virtual_layout_device", virtual_text)
                    if label == "xlu":
                        self.assertIn("atlas.virtual_xlu_transpose", virtual_text)
                    record, files = prepare_c_package(source, compiled, "layout_" + label, target_source=ROOT)
                    self.assertEqual(record["launches_per_invocation"], 1)
                    self.assertEqual(record["input_node_ids"], [key for key, _ in source.input_storages])
                    if shutil.which("cc"):
                        package = Path(directory) / (label + "_c")
                        package.mkdir()
                        for name, payload in files.items():
                            path = package / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(payload)
                        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", str(package / ("layout_" + label + ".c")), "-I", str(package)], check=True, capture_output=True)
                    virtual = compiled / "atlas-virtual.mlir"
                    virtual.write_text(virtual.read_text() + "\n")
                    manifest = json.loads((compiled / "manifest.json").read_text())
                    from hashlib import sha256
                    manifest["stage_artifacts"][virtual.name] = sha256(virtual.read_bytes()).hexdigest()
                    (compiled / "manifest.json").write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, "source or execution plan"):
                        prepare_c_package(source, compiled, "layout_" + label, target_source=ROOT)

    def test_scratchpad_compile_and_memory_holes(self):
        source = cases()["dense_dma"]
        with tempfile.TemporaryDirectory(prefix="atlas-layout-base-") as directory:
            compiled = Path(directory) / "compiled"
            compile_source(None, source, target_source=ROOT, destination=compiled,
                           limits=None, io_memory_base=0x0800e800)
            record, _ = prepare_c_package(source, compiled, "relocated", target_source=ROOT,
                                           io_memory_base=0x0800e800)
            self.assertEqual(record["layout_device_plan"]["input_addresses"], [0x0800e800])
            with self.assertRaises(ValueError):
                prepare_c_package(source, compiled, "relocated", target_source=ROOT)
            for base in (0x40000, 0x07ffffe0, 0x08010000, 0x0800ffe0):
                with self.subTest(base=base), self.assertRaises(ValueError):
                    compile_source(None, source, target_source=ROOT,
                                   destination=Path(directory) / str(base), limits=None,
                                   io_memory_base=base)

    def test_malformed_scalar_slot_accesses_reject_before_lowering(self):
        virtual = virtual_program(prepare_source(cases()["reverse_bytes"]))
        mutations = (
            ('kind = "lbu", byte_offset = 31 : i32', 'kind = "lbu", byte_offset = -1 : i32'),
            ('kind = "lbu", byte_offset = 31 : i32', 'kind = "lbu", byte_offset = 1024 : i32'),
            ('kind = "lbu", byte_offset = 31 : i32', 'kind = "lw", byte_offset = 31 : i32'),
            ('kind = "sb", byte_offset = 0 : i32', 'kind = "unknown", byte_offset = 0 : i32'),
        )
        for old, new in mutations:
            with self.subTest(mutation=new):
                self.assertIn(old, virtual)
                with self.assertRaises(RuntimeError):
                    _run([str(BIN / "atlas-opt"), "--atlas-compile", "-"], input_text=virtual.replace(old, new, 1))

    def test_changed_machine_and_pending_dma_rejected(self):
        with tempfile.TemporaryDirectory(prefix="atlas-layout-mutation-") as directory:
            compiled = Path(directory) / "compiled"
            compile_source(None, cases()["dense_dma"], target_source=ROOT, destination=compiled, limits=None)
            machine = (compiled / "atlas-machine.mlir").read_text()
            # Independent source contracts reject changed DMA effects and missing completion.
            no_wait = re.sub(r'"atlas.dma_wait"(\([^)]*\)) <\{channel = [0-9]+ : i32\}>',
                             r'"atlas.delay"\1 <{cycles = 1 : i32}>', machine, count=1)
            self.assertNotEqual(no_wait, machine)
            for changed in (machine.replace('direction = "store"', 'direction = "load"', 1), no_wait):
                with self.assertRaises(RuntimeError):
                    _run([str(BIN / "atlas-opt"), "--verify-atlas-generated-schedule", "-"], input_text=changed)

    @unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "select unchanged npu_model")
    def test_raw_bits_and_guards_on_unchanged_model(self):
        import dataclasses
        import torch
        import npu_model
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from tools.check_program_decode import decode_binary
        from tools.model_instructions import model_instruction
        model_root = Path(npu_model.__file__).resolve().parents[1]
        self.assertEqual(subprocess.check_output(["git", "-C", str(model_root), "rev-parse", "HEAD"], text=True).strip(), "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5")
        torch.set_num_threads(1)
        inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
        with tempfile.TemporaryDirectory(prefix="atlas-layout-model-") as directory:
            for label, source in cases().items():
                with self.subTest(label=label):
                    plan = prepare_source(source)
                    compiled = Path(directory) / label
                    compile_source(None, source, target_source=ROOT, destination=compiled, limits=None)
                    decoded = decode_binary(inventory, (compiled / "program.bin").read_bytes())
                    program = InstantiableProgram([model_instruction(row, inventory, isa) for row in decoded["instructions"]])
                    program.dram_base = DRAM_BASE
                    initial = bytearray((i * 17 + 23) % 256 for i in range(plan.workspace_bytes + 64))
                    raw = []
                    bindings = {b.node_id: bytes.fromhex(b.data_hex) for b in source.constants}
                    for node, address, size in zip(plan.operands, plan.input_addresses, plan.input_bytes):
                        start = address - DRAM_BASE
                        if node.effect == "constant":
                            initial[start:start+size] = bindings[node.id]
                        raw.append(bytes(initial[start:start+size]))
                        initial[start+size:start+(size+31)//32*32] = bytes((size+31)//32*32-size)
                    program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
                    config = DefaultHardwareConfig()
                    config.arch_state_config = dataclasses.replace(config.arch_state_config, numerics="rtl", randomize_init=True, init_seed=173)
                    simulation = Simulation(config, LoggerConfig(filename=str(Path(directory) / (label + ".json"))), program, verbose=False)
                    state = simulation.core.arch_state
                    try:
                        state.vmem.fill_(0xcd)
                        simulation.run()
                        self.assertTrue(state.halted)
                        self.assertEqual(simulation.get_stats().runtime_errors, 0)
                        actual = bytes(state.read_dram(DRAM_BASE, len(initial)).tolist())
                        offset = plan.output_address - DRAM_BASE
                        self.assertEqual(actual[:offset], bytes(initial[:offset]))
                        self.assertEqual(actual[offset:offset+plan.output_bytes], reference(source, raw))
                        self.assertEqual(actual[plan.workspace_bytes:], bytes(initial[plan.workspace_bytes:]))
                    finally:
                        state.close()


if __name__ == "__main__":
    unittest.main()
