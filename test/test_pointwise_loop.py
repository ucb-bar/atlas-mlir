"""Execute affine pointwise loops with independent BF16 arithmetic and guards."""

import ctypes
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_looped_mnk import ROOT, build_snapshot


def check_source_c_call(
    self, package, initial, binary, execute, values, wanted_bytes, invalid_values=(), *, source_dtype="f32",
):
    from atlas_mlir_native.binding import DRAM_BASE

    # Exercise the exported source C ABI, including its
    # packing and repeated launches through actual model execution.
    client_source = """#include "pointwise.h"
    int call(atlas_host_read32 rd,atlas_host_write32 wr,atlas_tile_write put,atlas_tile_read get,
             const float *const *inputs,float *output) {
      atlas_host host; pointwise_context raw; pointwise_source_context source;
      unsigned char scratch[POINTWISE_WORKSPACE_BYTES], conversion[POINTWISE_SOURCE_WORKSPACE_BYTES];
      if (atlas_host_init(&host,(atlas_host_map){0x20000,0x40000,131072},
          (atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0}) ||
          pointwise_bind(&raw,&host,(atlas_tile_memory_io){put,get,0},scratch,sizeof(scratch),8) ||
          pointwise_source_bind(&source,&raw,conversion,sizeof(conversion))) return -1;
      return pointwise_source_run(&source,inputs,output,0);
    }
    """
    if source_dtype == "bf16":
        client_source = client_source.replace("const float *const *inputs,float *output",
                                              "const uint8_t *const *inputs,uint8_t *output")
    elif source_dtype != "f32":
        raise ValueError("unsupported source test dtype")
    (package / "client.c").write_text(client_source)
    subprocess.run(
        [
            shutil.which("cc"),
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            "pointwise.c",
            "client.c",
            "-o",
            "client.so",
        ],
        cwd=package,
        check=True,
        capture_output=True,
    )
    client = ctypes.CDLL(str(package / "client.so"))
    rd_type = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t)
    wr_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
    mem_type = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    )
    memory, words, errors = bytearray(initial), {}, []
    polls = None

    def read32(_, address):
        nonlocal polls
        if address == 0x40004:
            return len(words)
        if address != 0x40008:
            errors.append("unexpected CSR read")
            return 3
        if polls is None:
            return 5
        polls += 1
        return 0 if polls == 1 else 3 if errors else 5

    def write32(_, address, word):
        nonlocal polls
        try:
            if 0x20000 <= address < 0x40000:
                words[(address - 0x20000) // 4] = word
            else:
                self.assertEqual((address, word), (0x40018, 1))
                self.assertEqual(
                    b"".join(words[i].to_bytes(4, "little") for i in range(len(words))),
                    binary,
                )
                self.assertEqual(memory, initial)
                memory[:] = execute(memory)
                polls = 0
        except (AssertionError, RuntimeError, ValueError, KeyError, OSError) as error:
            errors.append(repr(error))

    def transfer(address, pointer, size, write):
        at = address - DRAM_BASE
        if not 0 <= at <= len(memory) - size:
            return 1
        if write:
            memory[at : at + size] = ctypes.string_at(pointer, size)
        else:
            ctypes.memmove(pointer, bytes(memory[at : at + size]), size)
        return 0

    callbacks = (
        rd_type(read32),
        wr_type(write32),
        mem_type(lambda _, a, p, n: transfer(a, p, n, True)),
        mem_type(lambda _, a, p, n: transfer(a, p, n, False)),
    )
    client.call.argtypes = [
        rd_type,
        wr_type,
        mem_type,
        mem_type,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    for _ in range(2):
        memory[:] = initial
        words.clear()
        polls = None
        result = ctypes.create_string_buffer(bytes([0xA7]) * (len(wanted_bytes) + 32))
        before = [value.tobytes() for value in values]
        inputs = (ctypes.c_void_p * len(values))(*(value.ctypes.data for value in values))
        self.assertEqual(
            client.call(
                *callbacks,
                inputs,
                ctypes.addressof(result) + 16,
            ),
            0,
        )
        self.assertEqual(errors, [])
        self.assertEqual(result.raw[16 : 16 + len(wanted_bytes)], wanted_bytes)
        self.assertEqual(result.raw[:16], bytes([0xA7]) * 16)
        self.assertEqual(
            result.raw[16 + len(wanted_bytes) : -1],
            bytes([0xA7]) * 16,
        )
        self.assertEqual([value.tobytes() for value in values], before)
    for rejected in invalid_values:
        memory[:] = initial
        words.clear()
        polls = None
        result = ctypes.create_string_buffer(bytes([0xA7]) * (len(wanted_bytes) + 32))
        inputs = (ctypes.c_void_p * len(rejected))(*(value.ctypes.data for value in rejected))
        self.assertEqual(client.call(*callbacks, inputs, ctypes.addressof(result) + 16), 2)
        self.assertEqual(errors, [])
        self.assertEqual(words, {})
        self.assertEqual(memory, initial)
        self.assertEqual(result.raw[:-1], bytes([0xA7]) * (len(wanted_bytes) + 32))


@unittest.skipUnless(
    os.environ.get("MERLIN_NATIVE_ROOT")
    and os.environ.get("ATLAS_OOT_BIN_DIR")
    and os.environ.get("ATLAS_RUN_NPU_MODEL") == "1",
    "select native tools and npu_model",
)
class PointwiseLoopTest(unittest.TestCase):
    def test_pointwise_loops_broadcast_views_and_shared_operands(self):
        self._check_pointwise_cases(bf16_sources=False)

    def test_bf16_sources_preserve_raw_codes_and_shared_operands(self):
        self._check_pointwise_cases(bf16_sources=True)

    def test_pressure_broadcast_reuses_tiles_in_one_launch(self):
        self._check_pointwise_cases(bf16_sources=False, pressure=True)

    def test_fused_silu_executes_without_an_intermediate_buffer(self):
        self._check_pointwise_cases(bf16_sources=False, silu=True)

    def test_shared_operand_uses_one_buffer_and_one_load(self):
        self._check_pointwise_cases(bf16_sources=False, shared=True)

    def test_exact_cast_broadcast_fusion_executes_through_compact_pointwise(self):
        self._check_pointwise_cases(bf16_sources=False, affine_cast=True)

    def test_large_io_window_executes_in_one_launch_per_invocation(self):
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options

        with codegen_options(CodegenOptions(io_memory_bytes=1048576)):
            self._check_pointwise_cases(bf16_sources=False, large=True)

    def _check_pointwise_cases(self, *, bf16_sources, pressure=False, silu=False, shared=False, affine_cast=False,
                               large=False):
        import numpy as np
        import torch
        from atlas_mlir_native.binding import DRAM_BASE, dram_slots
        from atlas_mlir_native.layout import pack_bf16_pair_halves
        from atlas_mlir_native.pointwise import (
            BF16_POINTWISE_POLICY,
            compile_pointwise,
            input_spec,
            plan_abi,
            prepare_plan,
            prepare_source,
        )
        from atlas_mlir_native.pointwise_c import prepare_c_package
        from merlin.semantic_compiler.model import (
            IndexMap,
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.pointwise_tiling import (
            materialize_pointwise_input,
        )
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from test_spill_model_execution import MODEL_REVISION, model_instruction

        from tools.check_program_decode import decode_binary

        origin = json.loads(importlib.metadata.distribution("npu-model").read_text("direct_url.json") or "{}")
        self.assertTrue(
            MODEL_REVISION in origin.get("url", "") or origin.get("vcs_info", {}).get("commit_id") == MODEL_REVISION
        )
        torch.set_num_threads(1)
        spec = input_spec()
        spec.encode(np.array([0.0, -0.0, 2.0**-63, -(2.0**63)], dtype=np.float32))
        for value in (np.nan, np.inf, 2.0**-64, 2.0**64):
            with self.subTest(rejected_input=value), self.assertRaises(ValueError):
                spec.encode(np.array([value], dtype=np.float32))
        inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())

        def rounded(x):
            bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
            return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)

        def decoded(x):
            return (x.astype(np.uint32) << 16).view(np.float32)

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            snapshot = build_snapshot(root)
            for operation, shape, view in (
                ("mul", (33, 35), "dense"),
                ("mul", (65, 513), "dense_large_boundary"),
                ("mul", (1, 15, 50, 163), "dense_large_attention"),
                ("add", (31, 33), "broadcast"),
                ("add", (50, 720), "broadcast"),
                ("mul", (50, 720), "cast_broadcast"),
                ("add", (65, 512), "column_broadcast"),
                ("add", (65, 513), "scalar_broadcast"),
                ("sub", (1, 65), "reverse"),
                ("mul", (3, 5), "square"),
                ("mul", (1, 50, 720), "square"),
                ("relu", (33, 35), "relu"),
                ("relu", (2, 8), "relu_small"),
                ("add", (1, 3, 5, 33), "channel_bias"),
                ("sub", (2, 3, 35), "reverse_rank3"),
                ("mul", (7,), "vector"),
                ("relu", (2, 3, 5, 7), "relu_rank4"),
                ("div", (33, 35), "division"),
                ("div", (1, 65), "division_tail"),
                ("div", (2, 3, 35), "division_rank3"),
                ("div", (2, 3, 35), "division_broadcast"),
                ("rsqrt", (3, 7), "rsqrt"),
                ("rsqrt", (33, 35), "rsqrt_tail"),
                ("rsqrt", (2, 3, 5, 7), "rsqrt_rank4"),
                ("sigmoid", (3, 7), "sigmoid"),
                ("sigmoid", (33, 35), "sigmoid_tail"),
                ("sigmoid", (2, 3, 35), "sigmoid_rank3"),
                ("silu", (3, 7), "silu"),
                ("silu", (1, 50, 720), "silu_large"),
                ("gelu", (33, 35), "gelu_tail"),
                ("mul", (33, 35), "bf16_dense"),
                ("mul", (3, 5), "bf16_square"),
                ("mul", (2, 3, 35), "bf16_broadcast"),
            ):
                if view.startswith("bf16_") != bf16_sources:
                    continue
                if (operation == "silu") != silu:
                    continue
                if (shape == (1, 50, 720) and operation != "silu") != shared:
                    continue
                if (view == "cast_broadcast") != affine_cast:
                    continue
                if view.startswith("dense_large_") != large:
                    continue
                if (shape in {(50, 720), (65, 512), (65, 513)} and not affine_cast and not large) != pressure:
                    continue
                with self.subTest(operation=operation, shape=shape, view=view):
                    raw_bf16 = view.startswith("bf16_")
                    case_name = view
                    view = view.removeprefix("bf16_")
                    source_dtype = "bf16" if raw_bf16 else "f32"
                    m, n = int(np.prod(shape[:-1])), shape[-1]
                    rank = len(shape)
                    unary = operation in {"relu", "rsqrt", "sigmoid", "gelu", "silu"}
                    math_operation = operation in {"div", "rsqrt", "sigmoid", "gelu", "silu"}
                    type_ = TensorType(shape, source_dtype, "raw-bit-pattern")
                    a = SemanticNode("a", "input", (), type_, effect="input")
                    bshape = (
                        (n,)
                        if view in {"broadcast", "division_broadcast", "cast_broadcast"}
                        else (m,) if view == "column_broadcast"
                        else (1,) if view == "scalar_broadcast"
                        else (shape[1],)
                        if view == "channel_bias"
                        else shape
                    )
                    b = SemanticNode(
                        "b",
                        "input",
                        (),
                        TensorType(bshape, source_dtype, "raw-bit-pattern"),
                        effect="input",
                    )
                    identity = pointwise_identity_maps(rank, 1)[0]
                    if view in {"broadcast", "division_broadcast", "cast_broadcast"}:
                        mapping = IndexMap(
                            rank,
                            (tuple(int(i == rank - 1) for i in range(rank)),),
                            (0,),
                        )
                    elif view == "channel_bias":
                        mapping = IndexMap(rank, ((0, 1, 0, 0),), (0,))
                    elif view == "column_broadcast":
                        mapping = IndexMap(2, ((1, 0),), (0,))
                    elif view == "scalar_broadcast":
                        mapping = IndexMap(2, ((0, 0),), (0,))
                    elif view in {"reverse", "reverse_rank3"}:
                        rows = tuple(
                            tuple((-1 if d == rank - 1 else 1) if d == i else 0 for i in range(rank))
                            for d in range(rank)
                        )
                        mapping = IndexMap(rank, rows, (0,) * (rank - 1) + (n - 1,))
                    else:
                        mapping = identity
                    operands = ("a",) if unary else ("a", "a") if view == "square" else ("a", "b")
                    result = SemanticNode(
                        "result",
                        "f32_relu_ordered_select"
                        if operation == "relu"
                        else "source_f32_rsqrt_unqualified"
                        if operation == "rsqrt"
                        else source_dtype + "_" + operation,
                        operands,
                        type_,
                        (
                            (
                                "numeric_semantics",
                                "f32-ordered-lt-positive-zero-select-v1"
                                if operation == "relu"
                                else "source-f32-rsqrt-unqualified-v1"
                                if operation == "rsqrt"
                                else f"ieee754-{source_dtype}-{operation}-rne-v1",
                            ),
                        ),
                        index_maps=pointwise_identity_maps(rank, 1)
                        if unary
                        else (
                            identity,
                            mapping,
                            identity,
                        ),
                    )
                    if operation == "sigmoid":
                        from atlas_mlir_native.sigmoid import SOURCE_ATTRS

                        result = dataclasses.replace(result, op="source_f32_composite_unqualified", attrs=SOURCE_ATTRS)
                    elif operation == "gelu":
                        from atlas_mlir_native.gelu import SOURCE_ATTRS

                        result = dataclasses.replace(result, op="source_f32_composite_unqualified", attrs=SOURCE_ATTRS)
                    nodes = (a, result) if view == "square" or unary else (a, b, result)
                    if operation == "silu":
                        from atlas_mlir_native.sigmoid import SOURCE_ATTRS

                        sigmoid = SemanticNode("sigmoid", "source_f32_composite_unqualified", (a.id,), type_,
                                               SOURCE_ATTRS, index_maps=pointwise_identity_maps(rank, 1))
                        result = dataclasses.replace(result, op="f32_mul", inputs=(a.id, sigmoid.id),
                                         attrs=(("numeric_semantics", "ieee754-f32-mul-rne-v1"),),
                                         index_maps=pointwise_identity_maps(rank, 2))
                        nodes = (a, sigmoid, result)
                    if raw_bf16 and view == "broadcast":
                        nodes = (b, a, result)
                    source = KernelRequest(
                        nodes,
                        (result.id,),
                        ("external",),
                        tuple((node.id, "external") for node in nodes if node.effect == "input"),
                        snapshot.profile.target_identity,
                        "diagnostic",
                    )
                    original_source = folding = None
                    if affine_cast:
                        from merlin.compile.semantic_regions import (
                            extract_semantic_region,
                        )
                        from merlin.semantic_compiler.source_preparation import (
                            prepare_source as prepare_full_source,
                        )
                        from merlin.tests.ir.test_affine_views import (
                            source as broadcast_source,
                        )

                        original_source = dataclasses.replace(
                            broadcast_source(shape=shape), target_identity=source.target_identity,
                            lowering_policy=source.lowering_policy,
                        )
                        original_source = dataclasses.replace(original_source, nodes=(
                            *original_source.nodes[:-1],
                            dataclasses.replace(original_source.nodes[-1], inputs=("x", "cast")),
                        ))
                        prepared, folding = prepare_full_source(original_source)
                        source = extract_semantic_region(prepared, ("out",))
                    precision = prepare_source(source, allow_math=math_operation,
                                               bf16_policy=BF16_POINTWISE_POLICY if raw_bf16 else None)
                    if math_operation:
                        with self.assertRaises(ValueError):
                            prepare_source(source)
                        with self.assertRaises(ValueError):
                            prepare_source(
                                dataclasses.replace(
                                    source,
                                    nodes=(
                                        *source.nodes[:-1],
                                        dataclasses.replace(result, attrs=()),
                                    ),
                                ),
                                allow_math=True,
                            )
                    plan = prepare_plan(precision.request)
                    if view == "square":
                        self.assertEqual(len(precision.request.nodes), 2)
                        self.assertEqual(precision.request.nodes[-1].inputs, ("arg0", "arg0"))
                        self.assertEqual(precision.record()["inputs"], [a.id])
                    if pressure:
                        self.assertTrue(plan.compact_broadcasts)
                        expected_sizes = {"broadcast": (94208, 47104),
                                          "column_broadcast": (98304, 6144),
                                          "scalar_broadcast": (104448, 2048)}[view]
                        self.assertEqual(plan.input_packed_bytes, expected_sizes)
                        self.assertLessEqual(sum(plan.input_packed_bytes) + plan.packed_bytes, 262144)
                    if operation == "relu":
                        for bad in (
                            dataclasses.replace(result, attrs=()),
                            dataclasses.replace(result, index_maps=()),
                        ):
                            with self.assertRaises(ValueError):
                                prepare_source(dataclasses.replace(source, nodes=(a, bad)))
                    abi = plan_abi(precision.request)
                    if large and view == "dense_large_boundary":
                        abi = {"fixed_inputs": {"arg0": 300, "arg1": 360}, "fixed_outputs": [420]}
                    if view == "reverse":
                        abi = {
                            "fixed_inputs": {"arg0": 7, "arg1": 13},
                            "fixed_outputs": [19],
                        }
                    target = root / f"{operation}-{case_name}"
                    compile_pointwise(
                        snapshot,
                        precision.request,
                        fixed_inputs=abi["fixed_inputs"],
                        fixed_outputs=tuple(abi["fixed_outputs"]),
                        target_source=ROOT,
                        destination=target,
                        limits=SearchLimits(),
                    )
                    if affine_cast:
                        (target / "original-source.json").write_text(json.dumps(original_source.record()))
                        (target / "source-preparation.json").write_text(json.dumps(folding))
                    _, files = prepare_c_package(
                        source,
                        target,
                        "pointwise",
                        target_source=ROOT,
                        runtime_linkage="embedded",
                        allow_math=math_operation,
                        bf16_policy=BF16_POINTWISE_POLICY if raw_bf16 else None,
                    )
                    package = target / "c-package"
                    package.mkdir()
                    for name, data in files.items():
                        (package / name).write_bytes(data)
                    subprocess.run(
                        [
                            shutil.which("cc"),
                            "-std=c11",
                            "-O2",
                            "-Wall",
                            "-Wextra",
                            "-Werror",
                            "-c",
                            "pointwise.c",
                            "-o",
                            "pointwise.o",
                        ],
                        cwd=package,
                        check=True,
                        capture_output=True,
                    )
                    llvm = os.environ.get("ATLAS_LLVM_BIN")
                    if rank != 2 and llvm:
                        subprocess.run(
                            [
                                str(Path(llvm) / "clang"),
                                "--target=riscv64-unknown-elf",
                                "-march=rv64gcv",
                                "-mabi=lp64d",
                                "-std=c11",
                                "-O2",
                                "-ffreestanding",
                                "-Wall",
                                "-Wextra",
                                "-Werror",
                                "-c",
                                "pointwise.c",
                                "-o",
                                "pointwise.rv64.o",
                            ],
                            cwd=package,
                            check=True,
                            capture_output=True,
                        )
                        object_bytes = (package / "pointwise.rv64.o").read_bytes()
                        self.assertEqual(int.from_bytes(object_bytes[18:20], "little"), 243)
                        self.assertEqual(int.from_bytes(object_bytes[48:52], "little") & 6, 4)
                    if view == "dense":
                        changed = root / f"rehashed-machine-{case_name}"
                        shutil.copytree(target, changed)
                        physical = json.loads((changed / "physical-program.json").read_text())
                        row = physical["instructions"][0]
                        row["word_u32"] ^= 1
                        row["word_hex"] = f"{row['word_u32']:08x}"
                        (changed / "physical-program.json").write_text(json.dumps(physical))
                        binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in physical["instructions"])
                        (changed / "program.bin").write_bytes(binary)
                        manifest = json.loads((changed / "manifest.json").read_text())
                        manifest["binary_sha256"] = hashlib.sha256(binary).hexdigest()
                        manifest["stage_artifacts"] = {
                            key: hashlib.sha256((changed / key).read_bytes()).hexdigest()
                            for key in manifest["stage_artifacts"]
                        }
                        (changed / "manifest.json").write_text(json.dumps(manifest))
                        with self.assertRaisesRegex(ValueError, "source lowering"):
                            prepare_c_package(source, changed, "pointwise", target_source=ROOT,
                                              bf16_policy=BF16_POINTWISE_POLICY if raw_bf16 else None)
                    # Runtime values are created only after compilation.
                    av = ((np.arange(m * n).reshape(shape) % 37) - 18).astype(np.float32) / 13
                    bv = ((np.arange(np.prod(bshape)).reshape(bshape) % 29) - 14).astype(np.float32) / 11
                    av.reshape(-1)[0] = -0.0
                    if operation == "relu":
                        av.reshape(-1)[-1] = 1.00390625  # BF16 nearest-even midpoint.
                        av.reshape(-1)[n - 1] = 2.5
                    bv.reshape(-1)[0] = -0.0 if operation == "add" else 0.0
                    if affine_cast:
                        # The original BF16 source widens exactly; the boundary
                        # values passed to the selected region have these bits.
                        bv = decoded(rounded(bv))
                    if view == "scalar_broadcast":
                        bv[0] = np.float32(0.5)
                    if view in {"reverse", "reverse_rank3"}:
                        bv.reshape(-1)[-1] = 0.0
                    if raw_bf16:
                        av.reshape(-1)[:3] = [1.0078125, 1.015625, -0.0]
                        bv.reshape(-1)[:3] = [1.5, 1.25, 3.0]
                    if operation == "div":
                        bv = np.exp2((np.arange(np.prod(bshape)).reshape(bshape) % 31 - 15).astype(np.float32))
                        bv.reshape(-1)[-1] = np.float32(3.25)
                    elif operation in {"sigmoid", "silu"}:
                        av.reshape(-1)[:6] = [0.0, -0.0, -32.0, 32.0, 2.0**-63, -(2.0**-63)]
                    elif operation == "gelu":
                        av.reshape(-1)[:6] = [0.0, -0.0, 2.0**-32, -(2.0**-32), 2.0**32, -(2.0**32)]
                    elif operation == "rsqrt":
                        av = np.exp2((np.arange(m * n).reshape(shape) % 61 - 30).astype(np.float32))
                        av.reshape(-1)[-1] = np.float32(1.00390625)
                    lhs = rounded(av)
                    rhs = lhs if view == "square" else rounded(bv)
                    aa = decoded(lhs)
                    bb = decoded(rhs)
                    if view in {"broadcast", "division_broadcast", "cast_broadcast"}:
                        bb = np.broadcast_to(bb, shape)
                    elif view == "column_broadcast":
                        bb = np.broadcast_to(bb[:, None], shape)
                    elif view == "scalar_broadcast":
                        bb = np.broadcast_to(bb, shape)
                    elif view == "channel_bias":
                        bb = np.broadcast_to(bb[None, :, None, None], shape)
                    elif view in {"reverse", "reverse_rank3"}:
                        bb = bb[..., ::-1]
                    if operation == "relu":
                        wanted = np.where(lhs & 0x8000, np.uint16(0), lhs)
                    elif operation == "div":
                        from test_softmax_primitives import _reciprocal

                        reciprocal = np.array(
                            [_reciprocal(int(code)) for code in rhs.reshape(-1)],
                            dtype=np.uint16,
                        ).reshape(rhs.shape)
                        wanted = rounded(
                            (aa.astype(np.float64) * decoded(reciprocal).astype(np.float64)).astype(np.float32)
                        )
                    elif operation == "gelu":
                        from npu_model.configs.numerics import RtlNumerics

                        # Compose pinned model primitives independently of the
                        # compiler's typed DAG and scalar lane ports.
                        values = torch.from_numpy(lhs.copy()).view(torch.bfloat16)
                        cubic = RtlNumerics.unary("cube", values)
                        inner = RtlNumerics.add(values, RtlNumerics.mul(cubic, torch.full_like(values, 0.044677734375)))
                        nonlinear = RtlNumerics.unary("tanh", RtlNumerics.mul(inner, torch.full_like(values, 0.796875)))
                        outer = RtlNumerics.add(torch.ones_like(values), nonlinear)
                        half = RtlNumerics.mul(values, torch.full_like(values, 0.5))
                        wanted = RtlNumerics.mul(half, outer).view(torch.int16).numpy().view(np.uint16)
                    elif operation in {"sigmoid", "silu"}:
                        from npu_model.configs.numerics import RtlNumerics

                        # This expected value uses the model's primitive tables,
                        # not the compiler decomposition or its lane evaluator.
                        values = torch.from_numpy(lhs.copy()).view(torch.bfloat16)
                        negative = RtlNumerics.sub(torch.zeros_like(values), values)
                        exponential = RtlNumerics.unary("exp", negative)
                        denominator = RtlNumerics.add(torch.ones_like(values), exponential)
                        sigmoid = RtlNumerics.unary("rcp", denominator)
                        value = RtlNumerics.mul(values, sigmoid) if operation == "silu" else sigmoid
                        wanted = value.view(torch.int16).numpy().view(np.uint16)
                    elif operation == "rsqrt":
                        from test_softmax_primitives import _reciprocal
                        from test_vpu_sqrt_reference import _sqrt_bits

                        wanted = np.array(
                            [_reciprocal(_sqrt_bits(int(code))) for code in lhs.reshape(-1)],
                            dtype=np.uint16,
                        ).reshape(shape)
                    elif operation == "mul":
                        wanted = rounded((aa.astype(np.float64) * bb.astype(np.float64)).astype(np.float32))
                    else:
                        values = (
                            aa.astype(np.float64)
                            + (bb.astype(np.float64) if operation == "add" else -bb.astype(np.float64))
                        ).astype(np.float32)
                        wanted = (values.view(np.uint32) >> 16).astype(np.uint16)
                    if math_operation:
                        from atlas_mlir_native.reference import (
                            evaluate_gelu,
                            evaluate_pointwise_math,
                            evaluate_sigmoid,
                            evaluate_silu,
                        )

                        inputs = (lhs,) if unary else (lhs, rhs)
                        interpreted = evaluate_graph(
                            precision.request,
                            {
                                node.id: TensorValue(
                                    node.type,
                                    tuple(int(code) for code in value.reshape(-1)),
                                )
                                for node, value in zip(precision.request.nodes[:-1], inputs, strict=True)
                            },
                            primitives={
                                (
                                    precision.request.nodes[-1].op,
                                    "raw-bit-pattern",
                                ): evaluate_silu if operation == "silu" else evaluate_gelu if operation == "gelu" else
                                evaluate_sigmoid if operation == "sigmoid" else evaluate_pointwise_math
                            },
                        )
                        self.assertEqual(
                            interpreted[0].elements,
                            tuple(int(code) for code in wanted.reshape(-1)),
                        )
                    packed = []
                    matrix = wanted.reshape(m, n)
                    for row in range(0, m, 32):
                        for col in range(0, n, 32):
                            tile = np.full(
                                (32, 32),
                                0x3F00 if operation == "sigmoid" else 0x3F80 if operation == "rsqrt" else 0,
                                dtype="<u2",
                            )
                            part = matrix[row : row + 32, col : col + 32]
                            tile[: part.shape[0], : part.shape[1]] = part
                            if pressure or affine_cast:
                                # Independently construct every hardware lane,
                                # including unobserved tails filled by reused
                                # broadcast values rather than new padding.
                                left = np.zeros((32, 32), dtype=np.float32)
                                right = np.zeros_like(left)
                                h, w = part.shape
                                left[:h, :w] = aa[row:row + h, col:col + w]
                                if view in {"broadcast", "cast_broadcast"}:
                                    right[:, :w] = bb[0, col:col + w]
                                elif view == "column_broadcast":
                                    right[:h, :] = bb[row:row + h, 0, None]
                                else:
                                    right.fill(bb[0, 0])
                                tile = (rounded((left.astype(np.float64) * right.astype(np.float64)).astype(np.float32))
                                        if affine_cast else ((left + right).view(np.uint32) >> 16).astype("<u2"))
                            packed.append(pack_bf16_pair_halves(tile.tobytes()))
                    expected = b"".join(packed)
                    execution = json.loads((target / "execution_plan.json").read_text())
                    binary = (target / "program.bin").read_bytes()
                    program = InstantiableProgram(
                        [
                            model_instruction(row, inventory, isa)
                            for row in decode_binary(inventory, binary)["instructions"]
                        ]
                    )
                    program.dram_base = DRAM_BASE
                    initial = bytearray([0xA5]) * (dram_slots() * 2048)
                    fixture_inputs = []
                    for i, (address, codes) in enumerate(
                        zip(
                            execution["input_addresses"],
                            (lhs,) if unary or view == "square" else (lhs, rhs),
                            strict=True,
                        )
                    ):
                        raw = materialize_pointwise_input(plan, i, codes.astype("<u2").tobytes())
                        payload = b"".join(
                            pack_bf16_pair_halves(raw[at : at + 2048]) for at in range(0, len(raw), 2048)
                        )
                        start = address - DRAM_BASE
                        initial[start : start + len(payload)] = payload
                        reads = plan.packed_bytes // len(payload)
                        fixture_inputs.extend(
                            (address + at, payload[at : at + 2048], reads) for at in range(0, len(payload), 2048)
                        )
                    fixture = None
                    if os.environ.get("ATLAS_POINTWISE_FIXTURES"):
                        fixture = Path(os.environ["ATLAS_POINTWISE_FIXTURES"]) / case_name
                        fixture.mkdir(parents=True, exist_ok=False)
                        (fixture / "program.bin").write_bytes(binary)
                        for member in ("manifest.json", "source-request.json", "execution_plan.json",
                                       "physical-program.json", "atlas-virtual.mlir", "atlas-machine.mlir"):
                            shutil.copyfile(target / member, fixture / member)
                        for member in ("pointwise.c", "pointwise.h"):
                            shutil.copyfile(package / member, fixture / member)
                        if affine_cast:
                            for member in ("original-source.json", "source-preparation.json"):
                                shutil.copyfile(target / member, fixture / member)
                        if (package / "pointwise.rv64.o").exists():
                            shutil.copyfile(
                                package / "pointwise.rv64.o",
                                fixture / "pointwise.rv64.o",
                            )
                        (fixture / "expected.bin").write_bytes(expected)
                        properties = [
                            "schema=atlas.rtl_tile_fixture.v2",
                            f"name=pointwise_{operation}_{case_name}",
                            f"program_sha256={hashlib.sha256(binary).hexdigest()}",
                            f"input_count={len(fixture_inputs)}",
                            f"output_count={len(expected) // 2048}",
                            f"output.address={execution['output_addresses'][0]:#x}",
                        ]
                        for i, (address, payload, reads) in enumerate(fixture_inputs):
                            (fixture / f"input-{i}.bin").write_bytes(payload)
                            properties += [
                                f"input.{i}.address={address:#x}",
                                f"input.{i}.read_count={reads}",
                            ]
                        (fixture / "manifest.properties").write_text("\n".join(properties) + "\n")

                    def execute(
                        memory,
                        program=program,
                        target=target,
                        execution=execution,
                        expected=expected,
                        fixture=fixture,
                    ):
                        program.memory_regions = [(0, torch.tensor(list(memory), dtype=torch.uint8))]
                        config = DefaultHardwareConfig()
                        config.arch_state_config = dataclasses.replace(
                            config.arch_state_config,
                            numerics="rtl",
                            randomize_init=True,
                            init_seed=81,
                        )
                        sim = Simulation(
                            config,
                            LoggerConfig(filename=str(target / "model.json")),
                            program,
                            verbose=False,
                        )
                        state = sim.core.arch_state
                        try:
                            state.vmem.fill_(0xCD)
                            sim.run(max_cycles=2000000 if large else 500000)
                            self.assertTrue(state.halted)
                            self.assertEqual(sim.get_stats().runtime_errors, 0)
                            observed = bytes(state.read_dram(DRAM_BASE, len(memory)).tolist())
                            wanted_memory = bytearray(memory)
                            at = execution["output_addresses"][0] - DRAM_BASE
                            wanted_memory[at : at + len(expected)] = expected
                            self.assertEqual(observed, bytes(wanted_memory))
                            if fixture is not None:
                                log = fixture / "model-runs.json"
                                runs = json.loads(log.read_text()) if log.exists() else []
                                runs.append({"cycles": sim.get_stats().cycles, "memory_bytes_checked": len(memory),
                                             "observed_sha256": hashlib.sha256(observed).hexdigest(),
                                             "runtime_errors": sim.get_stats().runtime_errors})
                                log.write_text(json.dumps(runs, indent=2) + "\n")
                            return observed
                        finally:
                            state.close()

                    execute(initial)
                    if pressure or affine_cast or large or raw_bf16 or math_operation or view in {
                        "relu_small",
                        "relu_rank4",
                        "channel_bias",
                        "reverse_rank3",
                        "vector",
                        "square",
                    }:
                        invalid_values = []
                        source_values = (av,) if unary or view == "square" else (av, bv)
                        wanted_source = decoded(wanted).astype("<f4").tobytes()
                        if pressure or affine_cast or large:
                            invalid = (av.copy(), bv.copy())
                            invalid[-1].reshape(-1)[-1] = np.nan
                            invalid_values.append(invalid)
                        if raw_bf16:
                            source_values = (lhs,) if view == "square" else (lhs, rhs)
                            if view == "broadcast":
                                source_values = (rhs, lhs)
                            wanted_source = wanted.astype("<u2").tobytes()
                            for bad in (0x0001, 0x8001, 0x5F01, 0x7F80, 0x7FC1):
                                invalid = tuple(value.copy() for value in source_values)
                                invalid[-1].reshape(-1)[-1] = bad
                                invalid_values.append(invalid)
                        if math_operation:
                            rejected = ((2.0**-33, -(2.0**33), np.nan, np.inf) if operation == "gelu" else
                                        (33.0, -33.0, np.nan, np.inf, 2.0**-64) if operation in {"sigmoid", "silu"} else
                                        (0.0, -1.0))
                            for bad in rejected:
                                invalid = (av.copy(),) if unary else (av.copy(), bv.copy())
                                invalid[-1].reshape(-1)[0] = bad
                                invalid_values.append(invalid)
                        check_source_c_call(
                            self,
                            package,
                            initial,
                            binary,
                            execute,
                            source_values,
                            wanted_source,
                            invalid_values,
                            source_dtype=source_dtype,
                        )


if __name__ == "__main__":
    unittest.main()
