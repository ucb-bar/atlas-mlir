"""The model link must cover exactly the selected source-bound device calls."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_model_link import (
    RUNTIME_UNDEFINED,
    _check_link_symbols,
    _check_native_reset_words,
    _render_bindings,
    _render_entry_adapter,
    _selection,
    main,
)
from atlas_model_tile_bridge import render_wrapper
from check_program_decode import decode_binary
from check_variant_inventory import load_inventory


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModelLinkTest(unittest.TestCase):
    def test_selected_zero_start_requires_encoded_reset(self) -> None:
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            probe_path = Path(temporary)
            artifact = probe_path / "regions" / "region-0000"
            artifact.mkdir(parents=True)
            inventory = load_inventory()
            plan = artifact / "execution_plan.json"
            plan.write_text('{"operation":"mxu0_reset"}\n')

            def write_image(matmul_word: int) -> str:
                binary = struct.pack("<III", 0x00002077, matmul_word, 0x10001077)
                (artifact / "program.bin").write_bytes(binary)
                program = decode_binary(inventory, binary)
                physical = artifact / "physical-program.json"
                physical.write_text(json.dumps(program))
                digest = hashlib.sha256(binary).hexdigest()
                (artifact / "manifest.json").write_text(json.dumps({
                    "schema": "atlas.native_tile_compilation.v6",
                    "artifact_kind": "program", "target_identity": "test-rtl",
                    "target_inventory_sha256": _sha(ROOT / "docs/selected-variant-inventory.json"),
                    "engine": "merlin_native", "request_digest": "request",
                    "binary_sha256": digest,
                    "execution_plan_sha256": _sha(plan),
                    "stage_artifacts": {"physical-program.json": _sha(physical)},
                }))
                return digest

            digest = write_image(0x14000077)
            probe = {
                "target_identity": "test-rtl",
                "regions": [{"region_id": "matmul_0", "artifact": "regions/region-0000",
                             "request_digest": "request", "binary_sha256": digest}],
                "host_bridge": {
                    "external_symbols": ["forward$kernel_0"],
                    "bindings": [{"symbol": "forward$kernel_0", "region_id": "matmul_0"}],
                },
            }
            host_check = {"entry_dataflow": {
                "initialization_semantics": "source_drop_and_host_initializer_checked",
                "native_reset_semantics": "not_checked",
                "native_zero_initialization_obligation_count": 1,
            }}
            checked = _check_native_reset_words(probe_path, probe, host_check)
            self.assertEqual(checked[0]["reset_mode"], "VMATMUL_MXU0")

            probe["regions"][0]["binary_sha256"] = write_image(0x18000077)
            with self.assertRaisesRegex(ValueError, "unique encoded reset"):
                _check_native_reset_words(probe_path, probe, host_check)
            host_check["entry_dataflow"]["native_zero_initialization_obligation_count"] = 0
            with self.assertRaisesRegex(ValueError, "positive-zero source starts"):
                _check_native_reset_words(probe_path, probe, host_check)

    def test_exact_selected_wrapper_and_renderer(self) -> None:
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary)
            host_object = root / "host.o"
            host_object.write_bytes(b"selected host object")
            directory = root / "wrapper"
            directory.mkdir()
            symbol = "forward$kernel_1"
            binary = hashlib.sha256(struct.pack("<I", 0x00000073)).hexdigest()
            header, source, binder = render_wrapper(symbol, 2, (0, 1))
            contents = {
                "atlas_model_tile_wrapper.h": header,
                "atlas_model_tile_wrapper.c": source,
                "atlas_program.h": (
                    f'#define ATLAS_PROGRAM_SHA256 "{binary}"\n'
                    "#define ATLAS_PROGRAM_BITSTREAM_BOUND 0\n"
                    "#define ATLAS_PROGRAM_WORD_COUNT 1u\n"
                    "static const uint32_t atlas_program_words[ATLAS_PROGRAM_WORD_COUNT] = {\n"
                    "  0x00000073u,\n};\n"
                ),
                "atlas_captured_data.h": "selected data",
            }
            for name, value in contents.items():
                (directory / name).write_text(value)
            wrapper = {
                "schema": "atlas.diagnostic_model_tile_wrapper.v1",
                "status": "diagnostic_unbound", "engine": "merlin_native",
                "region_id": "matmul_0", "source_operation_id": "op0",
                "outlined_symbol": symbol, "binding_function": binder,
                "constant_operand": 2, "transpose_input": [0, 1],
                "bitstream_bound": False,
                "native_request_digest": "request", "native_binary_sha256": binary,
                "host_bridge_sha256": "bridge",
                "linked": False, "executed": False,
                "files": {name: _sha(directory / name) for name in contents},
            }
            (directory / "manifest.json").write_text(json.dumps(wrapper))
            probe = {
                "schema": "merlin.native_model_region_probe.v1",
                "engine": "merlin_native", "status": "incomplete",
                "source_operation_count": 4,
                "regions": [{
                    "region_id": "matmul_0", "request_digest": "request",
                    "binary_sha256": binary, "status": "emitted",
                    "static_inputs_status": "packed_or_runtime",
                }],
                "host_bridge": {
                    "schema": "merlin.diagnostic_model_host_bridge.v1",
                    "status": "typed_declarations_only",
                    "sha256": "bridge", "external_symbols": [symbol],
                    "required_native_unmet": [],
                    "bindings": [{"symbol": symbol, "region_id": "matmul_0",
                                  "operation_id": "op0",
                                  "input_types": ["tensor<32x32xf32>"] * 2,
                                  "result_types": ["tensor<32x32xf32>"]}],
                },
            }
            host_check = {
                "schema": "merlin.native_host_object_check.v1",
                "status": "object_symbols_checked",
                "selected_device_symbols": [symbol],
                "required_native_unmet": [],
                "host_object_sha256": _sha(host_object),
                "entry_dataflow": {
                    "schema": "merlin.host_entry_dataflow_check.v2",
                    "status": "source_to_host_body_checked",
                    "operation_count": 4,
                    "native_external_call_count": 1,
                    "host_defined_call_count": 1,
                    "host_linalg_body_correspondence_checked_count": 1,
                    "checked_operand_edges": 3,
                },
                "source_route_accounting": {
                    "schema": "merlin.model_source_route_accounting.v1",
                    "operation_count": 4,
                    "state_counts": {
                        "native_external_unexecuted": 1,
                        "host_body_numerics_unqualified": 1,
                        "entry_support_numerics_unqualified": 1,
                        "model_output_unexecuted": 1,
                    },
                    "operations": [{"operation_ordinal": index} for index in range(4)],
                },
                "entry_abi": {
                    "source_input_count": 1, "source_result_count": 1,
                    "input_descriptor_positions": [0],
                    "result_descriptor_positions": [1],
                    "descriptor_types": ["memref<2xf32>", "memref<3xf32>"],
                },
            }
            rows = _selection(probe, host_check, host_object, [(directory, wrapper)])
            self.assertEqual([row["symbol"] for row in rows], [symbol])
            stale_wiring = copy.deepcopy(host_check)
            stale_wiring["entry_dataflow"]["operation_count"] = 5
            with self.assertRaisesRegex(ValueError, "source-to-entry wiring"):
                _selection(probe, stale_wiring, host_object, [(directory, wrapper)])
            missing_body = copy.deepcopy(host_check)
            missing_body["entry_dataflow"]["host_linalg_body_correspondence_checked_count"] = 0
            with self.assertRaisesRegex(ValueError, "source-to-entry wiring"):
                _selection(probe, missing_body, host_object, [(directory, wrapper)])
            missing_counts = copy.deepcopy(host_check)
            del missing_counts["entry_dataflow"]["host_defined_call_count"]
            del missing_counts["entry_dataflow"]["host_linalg_body_correspondence_checked_count"]
            with self.assertRaisesRegex(ValueError, "source-to-entry wiring"):
                _selection(probe, missing_counts, host_object, [(directory, wrapper)])
            stale_routes = copy.deepcopy(host_check)
            stale_routes["source_route_accounting"]["state_counts"]["native_external_unexecuted"] = 0
            with self.assertRaisesRegex(ValueError, "source route accounting"):
                _selection(probe, stale_routes, host_object, [(directory, wrapper)])
            with self.assertRaisesRegex(ValueError, "source-to-entry wiring"):
                _selection(probe, {key: value for key, value in host_check.items()
                                   if key != "entry_dataflow"}, host_object, [(directory, wrapper)])
            with self.assertRaisesRegex(ValueError, "cover exactly"):
                _selection(probe, host_check, host_object, [])
            with self.assertRaisesRegex(ValueError, "one selected native call"):
                _selection(probe, host_check, host_object,
                           [(directory, wrapper), (directory, wrapper)])
            image_header = directory / "atlas_program.h"
            image_header.write_text(contents["atlas_program.h"].replace("0x00000073u", "0x00000013u"))
            wrapper["files"]["atlas_program.h"] = _sha(image_header)
            with self.assertRaisesRegex(ValueError, "words differ"):
                _selection(probe, host_check, host_object, [(directory, wrapper)])
            image_header.write_text(contents["atlas_program.h"])
            wrapper["files"]["atlas_program.h"] = _sha(image_header)
            (directory / "atlas_model_tile_wrapper.c").write_text(source + "/* changed */\n")
            wrapper["files"]["atlas_model_tile_wrapper.c"] = _sha(
                directory / "atlas_model_tile_wrapper.c")
            with self.assertRaisesRegex(ValueError, "differs from the selected Atlas renderer"):
                _selection(probe, host_check, host_object, [(directory, wrapper)])

    def test_linked_symbols_reject_host_substitution_and_unknown_runtime(self) -> None:
        rows = [
            "_mlir_ciface_forward T 0 10",
            "_mlir_ciface_forward$kernel_1 T 10 10",
            "atlas_invoke_checked_entry T 20 10",
            "atlas_constant_0 r 0 20",
            "atlas_constant_0 r 20 20",
            *(f"{name} U" for name in sorted(RUNTIME_UNDEFINED)),
        ]
        self.assertEqual(_check_link_symbols("\n".join(rows), ["forward$kernel_1"], "forward"),
                         sorted(RUNTIME_UNDEFINED))
        with self.assertRaisesRegex(ValueError, "lacks a defined"):
            _check_link_symbols("\n".join(rows).replace(
                "_mlir_ciface_forward$kernel_1 T", "_mlir_ciface_forward$kernel_1 U"),
                ["forward$kernel_1"], "forward")
        with self.assertRaisesRegex(ValueError, "unexpected unresolved"):
            _check_link_symbols("\n".join(rows + ["surprise U"]),
                                ["forward$kernel_1"], "forward")

    def test_checked_entry_adapter_uses_input_then_result_descriptors(self) -> None:
        abi = {
            "source_input_count": 2, "source_result_count": 1,
            "descriptor_types": ["memref<2x3xf32>", "memref<3x4xf32>", "memref<2x4xf32>"],
        }
        header, source = _render_entry_adapter("forward", abi)
        self.assertIn("_mlir_ciface_forward(&inputs[0], &inputs[1], &results[0]);", source)
        with self.assertRaisesRegex(ValueError, "static rank-one to rank-three"):
            _render_entry_adapter("forward", {**abi, "descriptor_types": [
                "memref<2x3xf32>", "memref<?x4xf32>", "memref<2x4xf32>"
            ]})
        with self.assertRaisesRegex(ValueError, "supported C symbol"):
            _render_entry_adapter("forward;bad", abi)
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            work = Path(temporary)
            (work / "atlas_model_entry.h").write_text(header)
            (work / "atlas_model_entry.c").write_text(source)
            (work / "entry_test.c").write_text("""
#include "atlas_model_entry.h"
static float a[6], b[12], c[8];
static int called;
void _mlir_ciface_forward(const atlas_f32_memref2 *x,
                          const atlas_f32_memref2 *y,
                          atlas_f32_memref2 *z) {
  if (x->aligned == a && y->aligned == b && z->aligned == c) ++called;
  *z = (atlas_f32_memref2){c, c, 0, {2, 4}, {4, 1}};
}
int main(void) {
  atlas_f32_memref2 in[2] = {
      {a, a, 0, {2, 3}, {3, 1}}, {b, b, 0, {3, 4}, {4, 1}}};
  atlas_f32_memref2 out[1] = {{0}};
  if (atlas_invoke_checked_entry(in, 2, out, 1) != 3 || called) return 4;
  out[0] = (atlas_f32_memref2){c, c, 0, {2, 4}, {4, 1}};
  if (atlas_invoke_checked_entry(in, 2, out, 1) || called != 1 || out[0].aligned != c)
    return 1;
  if (atlas_invoke_checked_entry(in, 1, out, 1) != 1 || called != 1)
    return 2;
  in[1].sizes[1] = 5;
  if (atlas_invoke_checked_entry(in, 2, out, 1) != 2 || called != 1)
    return 3;
  return 0;
}
""")
            executable = work / "entry_test"
            subprocess.run([
                compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                "-I", str(work), "-I", str(ROOT / "runtime"),
                str(work / "atlas_model_entry.c"), str(work / "entry_test.c"),
                "-o", str(executable),
            ], check=True, capture_output=True)
            subprocess.run([str(executable)], check=True, capture_output=True)

    def test_mixed_rank_entry_adapter_preserves_descriptor_order(self) -> None:
        abi = {
            "source_input_count": 2, "source_result_count": 1,
            "descriptor_types": ["memref<8xf32>", "memref<2x8xf32>", "memref<2x4xf32>"],
        }
        header, source = _render_entry_adapter("forward", abi)
        self.assertIn("const atlas_f32_memref1 *input_0", header)
        self.assertIn("_mlir_ciface_forward(call->input_0, call->input_1, call->result_0)", source)
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            work = Path(temporary)
            (work / "atlas_model_entry.h").write_text(header)
            (work / "atlas_model_entry.c").write_text(source)
            (work / "entry_test.c").write_text("""
#include "atlas_model_entry.h"
static float vector[8], matrix[16], output[8];
static int calls;
void _mlir_ciface_forward(const atlas_f32_memref1 *x,
                          const atlas_f32_memref2 *y,
                          atlas_f32_memref2 *z) {
  if (x->aligned == vector && y->aligned == matrix && z->aligned == output) ++calls;
  *z = (atlas_f32_memref2){output, output, 0, {2, 4}, {4, 1}};
}
int main(void) {
  atlas_f32_memref1 x = {vector, vector, 0, {8}, {1}};
  atlas_f32_memref2 y = {matrix, matrix, 0, {2, 8}, {8, 1}};
  atlas_f32_memref2 z = {0};
  atlas_model_entry_call call = {&x, &y, &z};
  if (atlas_invoke_checked_entry(&call) != 3 || calls) return 5;
  z = (atlas_f32_memref2){output, output, 0, {2, 4}, {4, 1}};
  if (atlas_invoke_checked_entry(&call) || calls != 1 || z.aligned != output)
    return 1;
  x.sizes[0] = 7;
  if (atlas_invoke_checked_entry(&call) != 2 || calls != 1) return 2;
  x.sizes[0] = 8;
  x.strides[0] = INT64_MAX;
  if (atlas_invoke_checked_entry(&call) != 2 || calls != 1) return 4;
  call.input_0 = 0;
  if (atlas_invoke_checked_entry(&call) != 1 || calls != 1) return 3;
  return 0;
}
""")
            executable = work / "entry_test"
            subprocess.run([
                compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                "-I", str(work), "-I", str(ROOT / "runtime"),
                str(work / "atlas_model_entry.c"), str(work / "entry_test.c"),
                "-o", str(executable),
            ], check=True, capture_output=True)
            subprocess.run([str(executable)], check=True, capture_output=True)

    def test_rank_three_entry_adapter_checks_shapes_and_calls_host(self) -> None:
        abi = {
            "source_input_count": 2, "source_result_count": 1,
            "descriptor_types": [
                "memref<2x3x4xf32>", "memref<2x4x5xf32>",
                "memref<2x3x5xf32>",
            ],
        }
        header, source = _render_entry_adapter("forward", abi)
        self.assertIn("atlas_f32_memref3 *result_0", header)
        self.assertIn("_mlir_ciface_forward(call->input_0, call->input_1, call->result_0)", source)
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            work = Path(temporary)
            (work / "atlas_model_entry.h").write_text(header)
            (work / "atlas_model_entry.c").write_text(source)
            (work / "entry_test.c").write_text("""
#include "atlas_model_entry.h"
static float lhs[24], rhs[40], output[30];
static int calls;
void _mlir_ciface_forward(const atlas_f32_memref3 *x,
                          const atlas_f32_memref3 *y,
                          atlas_f32_memref3 *z) {
  if (x->aligned == lhs && y->aligned == rhs && z->aligned == output) ++calls;
}
int main(void) {
  atlas_f32_memref3 x = {lhs, lhs, 0, {2, 3, 4}, {12, 4, 1}};
  atlas_f32_memref3 y = {rhs, rhs, 0, {2, 4, 5}, {20, 5, 1}};
  atlas_f32_memref3 z = {output, output, 0, {2, 3, 5}, {15, 5, 1}};
  atlas_model_entry_call call = {&x, &y, &z};
  if (atlas_invoke_checked_entry(&call) || calls != 1) return 1;
  x.sizes[0] = 3;
  if (atlas_invoke_checked_entry(&call) != 2 || calls != 1) return 2;
  x.sizes[0] = 2;
  z.strides[0] = INT64_MAX;
  if (atlas_invoke_checked_entry(&call) != 3 || calls != 1) return 3;
  return 0;
}
""")
            executable = work / "entry_test"
            subprocess.run([
                compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                "-I", str(work), "-I", str(ROOT / "runtime"),
                str(work / "atlas_model_entry.c"), str(work / "entry_test.c"),
                "-o", str(executable),
            ], check=True, capture_output=True)
            subprocess.run([str(executable)], check=True, capture_output=True)

    def test_model_binder_checks_shared_host_map_and_order(self) -> None:
        header, source = _render_bindings([
            {"binder": "atlas_bind_tile_000000000001"},
            {"binder": "atlas_bind_tile_000000000002"},
        ])
        self.assertIn("int atlas_bind_model_tiles", header)
        self.assertIn("map.imem_base != other_atlas_tile_map_000000000002.imem_base", source)
        self.assertIn("atlas_host_init(host, map, mmio, memory)", source)
        self.assertLess(source.index("atlas_bind_tile_000000000001(host"),
                        source.index("atlas_bind_tile_000000000002(host"))

    def test_existing_output_is_not_overwritten_on_failure(self) -> None:
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            output = Path(temporary) / "old"
            output.mkdir()
            receipt = output / "manifest.json"
            receipt.write_bytes(b"older result\n")
            args = ["--probe", "absent", "--host-check", "absent",
                    "--host-object", "absent", "--wrapper", "absent",
                    "--merlin-runtime-include", "absent", "--clang", "absent",
                    "--nm", "absent", "--out", str(output)]
            self.assertEqual(main(args), 1)
            self.assertEqual(receipt.read_bytes(), b"older result\n")


if __name__ == "__main__":
    unittest.main()
