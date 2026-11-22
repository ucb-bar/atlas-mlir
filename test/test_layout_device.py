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
from merlin.compiler.semantic.target_binding import NativeCompilationError
from atlas_mlir_native.options import CodegenOptions, codegen_options
from merlin.compiler.semantic.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps


import pytest
from atlas_mlir_native.options import current_codegen_options, codegen_options


@pytest.fixture(autouse=True)
def device_kv_enabled():
    with codegen_options(replace(current_codegen_options(), device_kv_quantizer=True)):
        yield

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
        "ragged_xlu": request("transpose", ((2, 3, 32),), (2, 32, 3), dtype="i8", mapping=IndexMap(3, ((1, 0, 0), (0, 0, 1), (0, 1, 0)), (0, 0, 0))),
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
        self.assertEqual(selected["ragged_xlu"], ["ragged_xlu"])
        self.assertEqual(selected["small_scalar_reshape"], ["scalar"])
        self.assertEqual(selected["ragged_scalar_transpose"], ["scalar"])
        self.assertEqual(selected["concat"], ["dma", "dma"])
        self.assertEqual(selected["broadcast"], ["dma"])

    def test_ragged_xlu_packs_arbitrary_bytes_without_padding(self):
        from atlas_mlir_native.layout_device import _ragged_transpose_words, _ragged_transpose_tile_words
        for rows in (1, 2, 3, 31, 33, 34, 35, 65, 66, 163, 255, 400, 511):
            raw = bytes((i * 37 + i // 64 * 11) % 256 for i in range(rows * 64))
            actual = bytearray()
            for column_base in (0, 32):
                caches = []
                for first in range(0, rows, 32):
                    cache = bytearray(1024)
                    for column in range(32):
                        for row in range(min(32, rows - first)):
                            cache[column * 32 + row] = raw[(first + row) * 64 + column_base + column]
                    caches.append(cache)
                for _, parts in _ragged_transpose_words(rows):
                    word = 0
                    for index, offset, shift, mask in parts:
                        value = int.from_bytes(caches[index][offset:offset + 4], "little")
                        word |= ((value << shift) if shift >= 0 else (value >> -shift)) & mask
                    actual.extend(word.to_bytes(4, "little"))
                grouped = bytearray()
                for group in range(8):
                    for _, parts in _ragged_transpose_words(rows, 4):
                        word = 0
                        for index, offset, shift, mask in parts:
                            offset += group * 128
                            value = int.from_bytes(caches[index][offset:offset + 4], "little")
                            word |= ((value << shift) if shift >= 0 else (value >> -shift)) & mask
                        grouped.extend(word.to_bytes(4, "little"))
                self.assertEqual(bytes(grouped), bytes(actual[-32 * rows:]))
                tiled = bytearray(32 * rows)
                for tile, cache in enumerate(caches):
                    for group in range(8):
                        for destination, parts, mask in _ragged_transpose_tile_words(rows, min(32, rows - tile * 32)):
                            value = 0
                            for offset, shift, part_mask in parts:
                                offset += group * 128
                                word = int.from_bytes(cache[offset:offset + 4], "little")
                                value |= ((word << shift) if shift >= 0 else (word >> -shift)) & part_mask
                            destination += group * 4 * rows + tile * 32
                            tile_rows = min(32, rows - tile * 32)
                            preserve = mask != 0xffffffff and (tile_rows != 32 or rows % 32 < 3 or not mask & 1)
                            previous = int.from_bytes(tiled[destination:destination + 4], "little") if preserve else 0
                            tiled[destination:destination + 4] = ((previous & (0xffffffff ^ mask)) | value).to_bytes(4, "little")
                self.assertEqual(bytes(tiled), bytes(actual[-32 * rows:]))
            expected = bytes(raw[row * 64 + column] for column in range(64) for row in range(rows))
            self.assertEqual(bytes(actual), expected)

    def test_ragged_xlu_keeps_complete_prefix_in_counted_device_image(self):
        with codegen_options(CodegenOptions(device_kv_quantizer=True, io_memory_bytes=67108864)):
            from atlas_mlir_native.restoration_budget import _source_loop_trips
            for rows, groups in ((65, 16), (163, 16), (255, 2), (400, 2)):
                source = request("transpose", ((2, groups, rows, 64),), (2, groups, 64, rows), dtype="i8",
                    mapping=IndexMap(4, ((1,0,0,0),(0,1,0,0),(0,0,0,1),(0,0,1,0)), (0,0,0,0)))
                plan = prepare_source(source)
                self.assertEqual(plan.segments[0]["mode"], "ragged_xlu")
                text = virtual_program(plan)
                trips = [trips for _, trips in _source_loop_trips(text)]
                self.assertCountEqual(trips, [2, 2, groups, 8, 8, rows, rows, rows // 32])
                self.assertEqual(text.count('"atlas.virtual_xlu_transpose"'), 2)
                # Only full-tile and tail-tile assembly/packing bodies remain.
                self.assertLess(text.count('"atlas.virtual_slot_load"'), 800)
                # Semantic eligibility requires the unchanged prefix permutation.
                changed = replace(source.nodes[-1].index_maps[0], coefficients=((0,1,0,0),(1,0,0,0),(0,0,0,1),(0,0,1,0)))
                swapped = request("transpose", ((2, 2, rows, 64),), (2, 2, 64, rows), dtype="i8", mapping=changed)
                with self.assertRaisesRegex(ValueError, "default proof work budget"):
                    prepare_source(swapped)

    def test_ragged_xlu_image_does_not_grow_with_tile_count(self):
        counts = []
        with codegen_options(CodegenOptions(device_kv_quantizer=True, io_memory_bytes=67108864)):
            for rows in (65, 513):
                source = request("transpose", ((rows, 64),), (64, rows), dtype="i8",
                    mapping=IndexMap(2, ((0,1),(1,0)), (0,0)))
                text = virtual_program(prepare_source(source))
                counts.append((text.count("atlas.virtual_slot_load"), text.count("atlas.virtual_slot_store"),
                               text.count("atlas.virtual_xlu_transpose"), text.count("atlas.virtual_slot")))
        self.assertEqual(counts[0], counts[1])

    def test_ragged_provider_classifies_only_explicit_lowered_image_policy(self):
        from unittest.mock import patch
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.layout_device import prove_plan
        source = request("transpose", ((163, 64),), (64, 163), dtype="i8",
            mapping=IndexMap(2, ((0,1),(1,0)), (0,0)))
        diagnostic = "Atlas tool rejected selected program: lowered program exceeds the 32768-word policy limit"
        plan = prepare_source(source)
        for status, reason, expected in (("compile_error", diagnostic, "resource_limit"),
                ("compile_error", "unsupported virtual operation", "compile_error"),
                ("search_timeout", diagnostic, "search_timeout")):
            with self.subTest(status=status, reason=reason), patch("atlas_mlir_native.layout_device._run",
                    side_effect=NativeCompilationError(status, reason)):
                with self.assertRaises(NativeCompilationError) as failure:
                    prove_plan(plan)
                self.assertEqual(failure.exception.status, expected)
        with patch("atlas_mlir_native.layout_device._run",
                side_effect=NativeCompilationError("compile_error", diagnostic)):
            self.assertFalse(AtlasApplicationAdapter._supports_device_source(source, source.nodes[-1], "layout_device"))

    def test_ragged_provider_resource_refusal_preserves_host_layout(self):
        from unittest.mock import patch
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.already_quantized import POLICY
        from atlas_mlir_native.application_precision import CONTRACTION_POLICY
        from atlas_mlir_native.row_softmax import SOFTMAX_POLICY
        from atlas_mlir_native.options import CodegenOptions, codegen_options
        source = request("transpose", ((80, 65, 64),), (80, 64, 65), dtype="i8",
            mapping=IndexMap(3, ((1,0,0),(0,0,1),(0,1,0)), (0,0,0)))
        policy = dict(schema="atlas.diagnostic_application_precision.v1", source_sha256="0" * 64,
            target_identity=TARGET_IDENTITY, contraction=CONTRACTION_POLICY, row_softmax=SOFTMAX_POLICY,
            host_support=["transpose", "reshape", "copy"], already_quantized=POLICY)
        adapter = AtlasApplicationAdapter(policy, original_source_sha256="0" * 64)
        with codegen_options(CodegenOptions(device_kv_quantizer=True, io_memory_bytes=67108864)):
            # Shape recognition stays tool-free even for an oversized prefix.
            with patch("atlas_mlir_native.layout_device.prove_plan") as proof:
                self.assertEqual(prepare_source(source).segments[0]["mode"], "ragged_xlu")
                proof.assert_not_called()
            refusal = NativeCompilationError("resource_limit", "complete prefix exceeds default instruction bound")
            with patch("atlas_mlir_native.layout_device.prove_plan", side_effect=refusal):
                self.assertFalse(adapter._supports_device_source(source, source.nodes[-1], "layout_device"))
                selected, = adapter.plan(source).selected
                self.assertEqual((selected.placement, dict(selected.metadata)["kind"]), ("host", "layout"))
            for status in ("search_timeout", "compile_error", "tool_unavailable"):
                with self.subTest(status=status), patch("atlas_mlir_native.layout_device.prove_plan",
                        side_effect=NativeCompilationError(status, "native tool failure")):
                    with self.assertRaises(NativeCompilationError) as failure:
                        adapter._supports_device_source(source, source.nodes[-1], "layout_device")
                    self.assertEqual(failure.exception.status, status)
            with patch("atlas_mlir_native.layout_device.prove_plan", return_value=None) as proof:
                self.assertTrue(adapter._supports_device_source(source, source.nodes[-1], "layout_device"))
                proof.assert_called_once()
            with patch("atlas_mlir_native.layout_device.prove_plan", side_effect=AssertionError("unexpected preflight")):
                for name in ("xlu", "dense_dma"):
                    existing = cases()[name]
                    self.assertTrue(adapter._supports_device_source(existing, existing.nodes[-1], "layout_device"))
                # Byte-granular scalar transport stays on the checked host layout, including convolution gathers:
                # a stride-2 1x1 shortcut and a 5x5 stride-2 im2col window over BF16 NCHW images.
                shortcut = request("gather", ((1, 4, 8, 8),), (1, 4, 4, 4),
                    mapping=IndexMap(4, ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)), (0, 0, 0, 0)))
                window = request("gather", ((1, 1, 9, 9),), (1, 1, 5, 3, 5, 3),
                    mapping=IndexMap(6, ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 2, 0, 0), (0, 0, 0, 0, 1, 2)), (0, 0, 0, 0)))
                for existing in (cases()["small_scalar_reshape"], cases()["ragged_scalar_transpose"], shortcut, window):
                    self.assertEqual({s["mode"] for s in prepare_source(existing).segments}, {"scalar"})
                    self.assertFalse(adapter._supports_device_source(existing, existing.nodes[-1], "layout_device"))

    def test_layout_proof_errors_are_not_resource_refusals(self):
        from unittest.mock import patch
        from atlas_mlir_native.binding import RTL_REVISION
        from atlas_mlir_native.layout_device import prove_plan
        plan = prepare_source(cases()["ragged_xlu"])
        emitted = json.dumps(dict(selected_rtl_revision=RTL_REVISION, instructions=[]))
        with patch("atlas_mlir_native.layout_device._native_pipeline", return_value="mandatory"):
            for error in (ValueError("unsupported bound structure"), KeyError("missing field"), IndexError("missing instruction")):
                with self.subTest(error=type(error).__name__), patch("atlas_mlir_native.layout_device._run",
                        side_effect=["machine", emitted]), patch("atlas_mlir_native.restoration_budget.instruction_bound", side_effect=error):
                    with self.assertRaises(NativeCompilationError) as failure:
                        prove_plan(plan)
                    self.assertEqual(failure.exception.status, "compile_error")
            for invalid in ("invalid program JSON", "[]"):
                with self.subTest(emitted=invalid), patch("atlas_mlir_native.layout_device._run", side_effect=["machine", invalid]):
                    with self.assertRaises(NativeCompilationError) as failure:
                        prove_plan(plan)
                    self.assertEqual(failure.exception.status, "compile_error")
            with patch("atlas_mlir_native.layout_device._run", side_effect=["machine", emitted]), patch(
                    "atlas_mlir_native.restoration_budget.instruction_bound",
                    return_value=dict(admitted=False, instruction_bound=1000001)):
                with self.assertRaises(NativeCompilationError) as failure:
                    prove_plan(plan)
                self.assertEqual(failure.exception.status, "resource_limit")

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

    def test_ragged_xlu_necessary_policy_bounds_reject_before_proof(self):
        from unittest.mock import patch
        def transpose(rows, heads=1):
            return request("transpose", ((heads, rows, 64),), (heads, 64, rows), dtype="i8",
                mapping=IndexMap(3, ((1,0,0),(0,0,1),(0,1,0)), (0,0,0)))
        options = CodegenOptions(device_kv_quantizer=True, io_memory_bytes=67108864)
        with codegen_options(options), patch("atlas_mlir_native.layout_device.prove_plan") as proof:
            for source in (transpose(20001), transpose(65, 200)):
                with self.subTest(shape=source.nodes[-1].type.shape), self.assertRaisesRegex(
                        ValueError, "selected image or instruction budget"):
                    prepare_source(source)
            proof.assert_not_called()
        with codegen_options(replace(options, program_size_limit_bytes=2048)):
            with self.assertRaisesRegex(ValueError, "selected image or instruction budget"):
                prepare_source(transpose(65))
        with codegen_options(replace(options, execution_instruction_limit=1000)):
            with self.assertRaisesRegex(ValueError, "selected image or instruction budget"):
                prepare_source(transpose(65))

    def test_ragged_xlu_admission_uses_completed_policy_proofs(self):
        from unittest.mock import patch
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from merlin.compiler.lower.c.layout import emit_layout_c
        for rows in (65, 163, 255, 400, 511, 513):
            source = request("transpose", ((rows, 64),), (64, rows), dtype="i8",
                mapping=IndexMap(2, ((0, 1), (1, 0)), (0, 0)))
            with self.subTest(rows=rows):
                self.assertEqual(prepare_source(source).segments[0]["mode"], "ragged_xlu")
                with patch("atlas_mlir_native.layout_device.prove_plan", return_value=None) as proof:
                    self.assertTrue(AtlasApplicationAdapter._supports_device_source(source, source.nodes[-1], "layout_device"))
                    proof.assert_called_once()
                for reason in ("lowered image exceeds selected program budget",
                               "layout worst-case instruction bound exceeds 1000000"):
                    with patch("atlas_mlir_native.layout_device.prove_plan",
                            side_effect=NativeCompilationError("resource_limit", reason)) as proof:
                        self.assertFalse(AtlasApplicationAdapter._supports_device_source(source, source.nodes[-1], "layout_device"))
                        proof.assert_called_once()
                record, _ = emit_layout_c(source, "bounded_transpose")
                self.assertEqual(record["source_request_digest"], source.digest())


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

    def test_ragged_xlu_native_default_bounds_and_source_contracts(self):
        with codegen_options(CodegenOptions(device_kv_quantizer=True, io_memory_bytes=67108864)):
            from atlas_mlir_native.binding import NATIVE_CHECKS
            with tempfile.TemporaryDirectory(prefix="atlas-ragged-xlu-") as directory:
                for heads, rows in ((32, 65), (15, 163), (1, 255), (1, 400)):
                    source = request("transpose", ((heads, rows, 64),), (heads, 64, rows), dtype="i8",
                        mapping=IndexMap(3, ((1,0,0),(0,0,1),(0,1,0)), (0,0,0)))
                    compiled = Path(directory) / str(rows)
                    compile_source(None, source, target_source=ROOT, destination=compiled, limits=None)
                    bound = json.loads((compiled / "layout-instruction-bound.json").read_text())
                    self.assertTrue(bound["admitted"])
                    from atlas_mlir_native.application import AtlasApplicationAdapter
                    self.assertTrue(AtlasApplicationAdapter._supports_device_source(source, source.nodes[-1], "layout_device"))
                    machine = (compiled / "atlas-machine.mlir").read_text()
                    for contract in ("dma", "mxu", "tile", "cfg", "source_memory", "buffer"):
                        self.assertIn("atlas.virtual_" + contract + "_contract", machine)
                    # The retained source must authenticate the exact scalar word
                    # shifts, independent of the unchanged machine instructions.
                    altered, count = re.subn(r'constant = 8 : i32', 'constant = 9 : i32', machine, count=1)
                    self.assertEqual(count, 1)
                    with self.assertRaises(RuntimeError):
                        _run([str(BIN / "atlas-opt"), *NATIVE_CHECKS, "-"], input_text=altered)

    def test_large_ragged_offsets_keep_generated_host_route(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from merlin.compiler.lower.c.layout import emit_layout_c
        for heads, rows, columns in ((1, 673, 64), (1, 700, 64),
                                     (2, 1000, 64), (1, 9000, 32)):
            source = request("transpose", ((heads, rows, columns),),
                (heads, columns, rows), dtype="i8",
                mapping=IndexMap(3, ((1,0,0),(0,0,1),(0,1,0)), (0,0,0)))
            with self.subTest(shape=source.nodes[0].type.shape), codegen_options(
                    CodegenOptions(device_kv_quantizer=True, io_memory_bytes=67108864)):
                self.assertFalse(AtlasApplicationAdapter._supports_device_source(
                    source, source.nodes[-1], "layout_device"))
                record, files = emit_layout_c(source, "large_transpose")
                self.assertEqual(record["source_request_digest"], source.digest())
                if shutil.which("cc"):
                    with tempfile.TemporaryDirectory(prefix="atlas-large-host-layout-") as directory:
                        for name, payload in files.items():
                            path = Path(directory) / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(payload.encode() if isinstance(payload, str) else payload)
                        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                            "-fsyntax-only", str(Path(directory) / "large_transpose.c"),
                            "-I", directory], check=True, capture_output=True)

    def test_counted_slab_initialization_requires_complete_coverage(self):
        source = request("transpose", ((65, 64),), (64, 65), dtype="i8",
            mapping=IndexMap(2, ((0,1),(1,0)), (0,0)))
        virtual = virtual_program(prepare_source(source))
        # The initialization is the first counted loop; later loops pack and
        # publish the same slab and must not supply its initial footprint.
        mutations = (
            ("arith.constant 65 : i32", "arith.constant 64 : i32"),
            ("arith.constant 8 : i32", "arith.constant 9 : i32"),
            ('kind = "sw", byte_offset = 28 : i32', 'kind = "sw", byte_offset = 24 : i32'),
            ('kind = "sw", byte_offset = 28 : i32', 'kind = "sw", byte_offset = 32 : i32'),
        )
        for old, new in mutations:
            with self.subTest(mutation=new):
                self.assertIn(old, virtual)
                with self.assertRaisesRegex(NativeCompilationError,
                        "scalar slot load reads an uninitialized byte"):
                    _run([str(BIN / "atlas-opt"), "--atlas-compile", "-"],
                         input_text=virtual.replace(old, new, 1))

    def test_ragged_xlu_oversized_complete_prefix_is_ineligible_before_compilation(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from merlin.compiler.lower.c.layout import emit_layout_c
        from atlas_mlir_native.options import CodegenOptions, codegen_options
        source = request("transpose", ((80, 65, 64),), (80, 64, 65), dtype="i8",
            mapping=IndexMap(3, ((1,0,0),(0,0,1),(0,1,0)), (0,0,0)))
        with codegen_options(CodegenOptions(device_kv_quantizer=True, io_memory_bytes=67108864)):
            self.assertFalse(AtlasApplicationAdapter._supports_device_source(source, source.nodes[-1], "layout_device"))
        record, _ = emit_layout_c(source, "oversized_transpose")
        self.assertEqual(record["source_request_digest"], source.digest())

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
        self.assertEqual(subprocess.check_output(["git", "-C", str(model_root), "rev-parse", "HEAD"], text=True).strip(), "b97b2f5a97e432141c4ef8504bec396651d77a8f")
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


def test_adapter_names_a_layout_slice_the_planner_refuses(tmp_path):
    """A staged column slice of a head-repeating gather leaves only scalar copies past the proof budget.

    The adapter names that refusal, so staging rejects only the slice and tries its next option.
    """
    from types import SimpleNamespace

    from atlas_mlir_native.application import AtlasApplicationAdapter

    mapping = IndexMap(5, ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)), (0,) * 5)
    source = request("gather", ((1, 5, 1, 227, 8),), (1, 5, 3, 227, 8), mapping=mapping)
    with pytest.raises(ValueError, match="default proof work budget"):
        prepare_source(source)
    layout = SimpleNamespace(metadata=(("kind", "layout_device"),))
    with pytest.raises(NativeCompilationError, match="layout_device: ragged scalar layout") as refused:
        AtlasApplicationAdapter.compile_region(None, None, source, layout, "slice", tmp_path, ROOT, None)
    assert refused.value.status == "unsupported_semantics" and not (tmp_path / "slice").exists()
