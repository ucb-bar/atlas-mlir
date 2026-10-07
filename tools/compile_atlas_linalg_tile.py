#!/usr/bin/env python3
"""Import an explicit, bounded Linalg MLP slice into Atlas virtual SSA.

The input is a captured Linalg module plus its argument manifest and an
explicit, proposed precision policy. This tool does not inspect sample inputs
or reference outputs. The policy changes f32 source arithmetic to selected
FP8/BF16 operations, so its result needs separate quality evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
from typing import Any

from xdsl.context import Context
from xdsl.dialects import affine, arith, builtin, func, linalg, tensor
from xdsl.ir import BlockArgument, SSAValue
from xdsl.parser import Parser


RTL_REVISION = json.loads((Path(__file__).resolve().parents[1] /
                           "docs/selected-variant-inventory.json")
                          .read_text())["selected_sources"]["rtl_revision"]
POLICY_SCHEMA = "atlas.oot.quantized_linalg_policy.v1"


class UnsupportedLinalg(ValueError):
    """A source operation or policy cannot be represented by this importer."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise UnsupportedLinalg(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fp8_e4m3_normal(value: float) -> int:
    """Proposed finite-normal E4M3 operand policy, separate from MXU execution.

    The policy flushes magnitudes below the smallest normal to signed zero,
    rounds the remaining finite values to nearest with even encoding LSB,
    and clamps above the largest admitted normal. It rejects NaN/Inf.
    """
    require(math.isfinite(value), "FP8 operand is NaN or infinite")
    sign = 0x80 if math.copysign(1.0, value) < 0 else 0
    magnitude = abs(float(value))
    if magnitude < 2.0 ** -6:
        return sign
    candidates = (
        ((1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7),
         (exponent << 3) | mantissa)
        for exponent in range(1, 15) for mantissa in range(8)
    )
    _, code = min(candidates,
                  key=lambda item: (abs(item[0] - magnitude), item[1] & 1,
                                    item[1]))
    return sign | code


def bf16_rne(value: float) -> int:
    require(math.isfinite(value), "BF16 boundary value is NaN or infinite")
    try:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    except OverflowError as error:
        raise UnsupportedLinalg("BF16 boundary value overflows f32") from error
    require((bits & 0x7F800000) != 0x7F800000,
            "BF16 boundary value overflows f32")
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    require((rounded & 0x7F80) != 0x7F80,
            "BF16 boundary value overflows finite range")
    require((rounded & 0x7F80) != 0 or (rounded & 0x7F) == 0,
            "BF16 boundary subnormal is outside the selected addend domain")
    return rounded


def pack_fp8_tile(values: Any) -> bytes:
    require(tuple(values.shape) == (32, 32), "FP8 source tile must be 32x32")
    return bytes(fp8_e4m3_normal(float(value)) for value in values.flat)


def pack_bias_tile(values: Any) -> bytes:
    require(tuple(values.shape) == (32,), "BF16 bias must have 32 columns")
    data = bytearray(2048)
    for row in range(32):
        for col in range(32):
            offset = (col // 16) * 1024 + (row * 16 + col % 16) * 2
            struct.pack_into("<H", data, offset, bf16_rne(float(values[col])))
    return bytes(data)


def shape(value: SSAValue) -> tuple[int, ...]:
    typ = value.type
    require(isinstance(typ, builtin.TensorType), f"expected tensor, got {typ}")
    require(str(typ.element_type) == "f32", f"expected source f32 tensor, got {typ}")
    require(typ.has_static_shape(), f"dynamic tensor is outside this ABI: {typ}")
    return tuple(typ.get_shape())


def maps(op: Any) -> tuple[tuple[int, ...], ...]:
    raw = op.properties.get("indexing_maps")
    require(isinstance(raw, builtin.ArrayAttr), f"{op.name}: missing parsed indexing maps")
    result = []
    for attr in raw.data:
        mapping = attr.data
        require(mapping.num_symbols == 0, f"{op.name}: symbolic index map")
        require(all(type(expr).__name__ == "AffineDimExpr" for expr in mapping.results),
                f"{op.name}: nonprojected index map")
        result.append(tuple(expr.position for expr in mapping.results))
    return tuple(result)


def iterators(op: Any) -> tuple[str, ...]:
    raw = op.properties.get("iterator_types")
    require(isinstance(raw, builtin.ArrayAttr), f"{op.name}: missing iterator types")
    return tuple(item.data.value for item in raw.data)


def region_ops(op: Any) -> tuple[list[SSAValue], list[Any]]:
    require(len(op.regions) == 1 and len(op.regions[0].blocks) == 1,
            f"{op.name}: expected one parsed region block")
    block = op.regions[0].blocks[0]
    return list(block.args), list(block.ops)


def is_zero_constant(value: SSAValue) -> bool:
    owner = getattr(value, "owner", None)
    if getattr(owner, "name", None) != "arith.constant":
        return False
    attr = owner.properties.get("value")
    return isinstance(attr, builtin.FloatAttr) and float(attr.value.data) == 0.0


def check_empty(value: SSAValue, expected_shape: tuple[int, ...]) -> None:
    require(shape(value) == expected_shape, "output tensor shape is not the selected tile")
    require(getattr(getattr(value, "owner", None), "name", None) == "tensor.empty",
            "Linalg output must be tensor.empty before a full overwrite")


def check_matmul(op: Any) -> None:
    require(len(op.operands) == 3 and len(op.results) == 1,
            "linalg.matmul must have two inputs, one init, and one result")
    require(all(shape(value) == (32, 32) for value in (*op.operands, *op.results)),
            "only static 32x32 f32 matmul imports in this tile ABI")
    require(maps(op) == ((0, 2), (2, 1), (0, 1)),
            "linalg.matmul indexing maps differ from the checked contraction")
    args, body = region_ops(op)
    require(len(args) == 3 and [item.name for item in body] ==
            ["arith.mulf", "arith.addf", "linalg.yield"],
            "matmul region is not ordered multiply/add/yield")
    product, accum, output = body
    require(tuple(product.operands) == (args[0], args[1]),
            "matmul product operands differ")
    require(tuple(accum.operands) == (product.results[0], args[2]) and
            tuple(output.operands) == (accum.results[0],),
            "matmul accumulation body differs")
    for arithmetic in (product, accum):
        flags = arithmetic.properties.get("fastmath")
        require(flags is None or not flags.data,
                "fast-math matmul cannot enter the selected numerical policy")
    init = op.operands[2]
    fill = getattr(init, "owner", None)
    require(getattr(fill, "name", None) == "linalg.fill" and
            len(fill.operands) == 2 and is_zero_constant(fill.operands[0]),
            "reset MXU requires an explicit zero-filled accumulator")
    check_empty(fill.operands[1], (32, 32))


def generic_kind(op: Any) -> str:
    require(len(op.results) == 1 and shape(op.results[0]) == (32, 32),
            "generic result is not a 32x32 f32 tile")
    require(iterators(op) == ("parallel", "parallel"),
            "generic requires two parallel iterators")
    check_empty(op.operands[-1], (32, 32))
    args, body = region_ops(op)
    if maps(op) == ((0, 1), (1,), (0, 1)):
        require(len(op.operands) == 3 and shape(op.operands[0]) == (32, 32)
                and shape(op.operands[1]) == (32,),
                "bias add requires a tile and a column vector")
        require(len(args) == 3 and [item.name for item in body] ==
                ["arith.addf", "linalg.yield"],
                "bias region is not addf/yield")
        add, output = body
        require(tuple(add.operands) == (args[0], args[1]) and
                tuple(output.operands) == (add.results[0],),
                "bias region uses unexpected values")
        require(not add.properties.get("fastmath").data,
                "fast-math bias add cannot enter the selected numerical policy")
        return "bias_add"
    if maps(op) == ((0, 1), (0, 1)):
        require(len(op.operands) == 2 and shape(op.operands[0]) == (32, 32),
                "ReLU requires one tile input")
        require(len(args) == 2 and [item.name for item in body] ==
                ["arith.constant", "arith.maximumf", "linalg.yield"],
                "ReLU region is not zero/maximumf/yield")
        zero, maximum, output = body
        require(is_zero_constant(zero.results[0]) and
                tuple(maximum.operands) == (args[0], zero.results[0]) and
                tuple(output.operands) == (maximum.results[0],),
                "ReLU region uses unexpected values")
        require(not maximum.properties.get("fastmath").data,
                "fast-math ReLU cannot enter the selected numerical policy")
        return "relu"
    raise UnsupportedLinalg("generic indexing maps are outside bias/ReLU admission")


def check_policy(policy: dict[str, Any]) -> tuple[int, ...]:
    expected = {
        "schema": POLICY_SCHEMA,
        "selected_rtl_revision": RTL_REVISION,
        "source_dtype": "f32",
        "operand_encoding": "e4m3_finite_normal_rne_flush_subnormal",
        "bias_encoding": "bf16_rne",
        "vpu_add": "selected_f32_sum_bf16_chop",
        "interlayer_pack_scale_code": 127,
        "precision_transform": "explicit_diagnostic_candidate",
    }
    for key, value in expected.items():
        require(policy.get(key) == value, f"policy {key} must be {value!r}")
    units = policy.get("mxu_units")
    require(isinstance(units, list) and units and
            all(isinstance(unit, int) and unit in (0, 1) for unit in units),
            "policy requires an explicit MXU unit per contraction")
    return tuple(units)


class Importer:
    def __init__(self, module: Any, manifest: dict[str, Any],
                 policy: dict[str, Any], input_base: int, output_base: int):
        self.module = module
        self.manifest = manifest
        self.units = check_policy(policy)
        self.input_base = input_base
        self.output_base = output_base
        self.values: dict[SSAValue, str] = {}
        self.transposes: dict[SSAValue, BlockArgument] = {}
        self.fp8_views: dict[SSAValue, str] = {}
        self.external: dict[tuple[int, str], str] = {}
        self.slots: list[dict[str, Any]] = []
        self.ops: list[str] = []
        self.next_value = 0
        self.next_io = 0
        self.matmul_count = 0
        self.function = None

    def name(self) -> str:
        name = f"%t{self.next_value}"
        self.next_value += 1
        return name

    def external_value(self, arg: BlockArgument, role: str) -> str:
        require(arg.owner is self.function.body.blocks[0],
                "external value is not a function argument")
        index = arg.index
        declared = self.manifest.get(str(index))
        require(isinstance(declared, dict), f"argument {index} is absent from manifest")
        actual_shape = shape(arg)
        kind = declared.get("kind")
        require(kind in ("param", "input"), f"argument {index} has unknown kind")
        if role == "bias_bf16":
            require(actual_shape == (32,), "bias must be a 32-element vector")
        else:
            require(actual_shape == (32, 32), "operand must be a 32x32 tensor")
        key = (index, role)
        if key in self.external:
            return self.external[key]
        require(all(previous[0] != index for previous in self.external),
                f"argument {index} has conflicting physical input roles")
        slot = len(self.slots)
        value = self.name()
        next_io = self.next_io + 1
        suffix = "bf16" if role == "bias_bf16" else "fp8"
        self.ops.append(
            f'    %io{next_io}, {value} = "atlas.virtual_input_{suffix}"(%io{self.next_io}) '
            f'{{index = {slot} : i32}} : (!atlas.virtual_state) -> '
            f'(!atlas.virtual_state, !atlas.virtual_{suffix})'
        )
        self.next_io = next_io
        self.external[key] = value
        self.slots.append({
            "slot": slot, "source_argument": index, "source_kind": kind,
            "source_key": declared.get("weight") if kind == "param" else declared.get("name"),
            "source_shape": list(actual_shape), "physical_role": role,
            "dram_address": self.input_base + slot * 2048,
            "payload_bytes": 2048 if role == "bias_bf16" else 1024,
            "reserved_slot_bytes": 2048,
        })
        return value

    def fp8_value(self, value: SSAValue) -> str:
        if isinstance(value, BlockArgument):
            return self.external_value(value, "activation_fp8")
        if value in self.fp8_views:
            return self.fp8_views[value]
        source = self.values.get(value)
        require(source is not None, "matmul activation has no admitted producer")
        packed = self.name()
        self.ops.append(
            f'    {packed} = "atlas.virtual_pack_fp8"({source}) '
            '{scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8'
        )
        self.fp8_views[value] = packed
        return packed

    def compile(self) -> tuple[str, dict[str, Any]]:
        functions = [op for op in self.module.body.block.ops if op.name == "func.func"]
        require(len(functions) == 1 and len(list(self.module.body.block.ops)) == 1,
                "module must contain exactly one captured function")
        fn = functions[0]
        require(len(fn.body.blocks) == 1, "captured function must have one block")
        self.function = fn
        args = list(fn.body.blocks[0].args)
        require(set(self.manifest) == {str(i) for i in range(len(args))},
                "argument manifest does not match function argument indexes")
        require(self.input_base >= 0x80000000 and self.output_base >= 0x80000000 and
                self.input_base % 1024 == 0 and self.output_base % 1024 == 0 and
                self.input_base <= 0xFFFFFFFF and self.output_base <= 0xFFFFFFFF,
                "DRAM bases must be selected 32-bit aligned addresses")
        require(len(fn.function_type.outputs.data) == 1,
                "this importer requires one tensor result")
        self.ops = ['    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state']
        result = None
        for op in fn.body.blocks[0].ops:
            if op.name in ("tensor.empty", "arith.constant"):
                continue
            if op.name == "linalg.transpose":
                require(len(op.operands) == 2 and len(op.results) == 1 and
                        tuple(op.properties["permutation"].get_values()) == (1, 0),
                        "only a parsed 2-D weight transpose is admitted")
                check_empty(op.operands[1], (32, 32))
                source = op.operands[0]
                require(isinstance(source, BlockArgument) and shape(source) == (32, 32),
                        "transpose source must be a 32x32 external weight")
                self.transposes[op.results[0]] = source
                continue
            if op.name == "linalg.fill":
                require(len(op.operands) == 2 and is_zero_constant(op.operands[0]),
                        "only explicit zero fill is admitted")
                check_empty(op.operands[1], (32, 32))
                continue
            if op.name == "linalg.matmul":
                check_matmul(op)
                require(self.matmul_count < len(self.units),
                        "MXU unit policy has fewer entries than contractions")
                weight_arg = self.transposes.get(op.operands[1])
                require(weight_arg is not None,
                        "matmul RHS must be a checked transpose of a weight argument")
                activation = self.fp8_value(op.operands[0])
                weight = self.external_value(weight_arg, "weight_fp8_nk")
                output = self.name()
                unit = self.units[self.matmul_count]
                self.ops.append(
                    f'    {output} = "atlas.virtual_mxu_matmul"({activation}, {weight}) '
                    f'{{unit = {unit} : i32}} : (!atlas.virtual_fp8, !atlas.virtual_fp8) '
                    '-> !atlas.virtual_bf16'
                )
                self.values[op.results[0]] = output
                self.matmul_count += 1
                continue
            if op.name == "linalg.generic":
                kind = generic_kind(op)
                source = self.values.get(op.operands[0])
                require(source is not None, f"{kind} input has no admitted producer")
                output = self.name()
                if kind == "bias_add":
                    bias_arg = op.operands[1]
                    require(isinstance(bias_arg, BlockArgument),
                            "bias vector must be an external function argument")
                    bias = self.external_value(bias_arg, "bias_bf16")
                    self.ops.append(
                        f'    {output} = "atlas.virtual_vpu_binary"({source}, {bias}) '
                        '{kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) '
                        '-> !atlas.virtual_bf16'
                    )
                else:
                    self.ops.append(
                        f'    {output} = "atlas.virtual_vpu_unary"({source}) '
                        '{kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'
                    )
                self.values[op.results[0]] = output
                continue
            if op.name == "func.return":
                require(len(op.operands) == 1 and result is None,
                        "one ordered tensor return is required")
                result = self.values.get(op.operands[0])
                require(result is not None, "returned value has no admitted producer")
                continue
            raise UnsupportedLinalg(f"source operation {op.name} has no Atlas route")
        require(result is not None, "captured function has no return")
        require(self.matmul_count == len(self.units),
                "MXU unit policy count must equal the source contraction count")
        require(len(self.slots) < 64, "FP8 pack scratch requires fewer than 64 input slots")
        input_end = self.input_base + len(self.slots) * 2048
        output_end = self.output_base + 2048
        require(input_end <= 1 << 32 and output_end <= 1 << 32 and
                not (self.input_base < output_end and self.output_base < input_end),
                "input and output DRAM spans overlap or overflow")
        next_io = self.next_io + 1
        self.ops.append(
            f'    %io{next_io} = "atlas.virtual_output_bf16"(%io{self.next_io}, {result}) '
            '{index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) '
            '-> !atlas.virtual_state'
        )
        self.ops.append(f"    return %io{next_io} : !atlas.virtual_state")
        text = "\n".join([
            "module {",
            f"  func.func @atlas_compiled() -> !atlas.virtual_state attributes "
            f"{{atlas.input_dram_base = {self.input_base} : i64, "
            f"atlas.output_dram_base = {self.output_base} : i64}} {{",
            *self.ops,
            "  }",
            "}",
            "",
        ])
        return text, {
            "schema": "atlas.oot.captured_linalg_import.v1",
            "source_function": str(fn.sym_name.data),
            "source_operations_accounted": len(list(fn.body.blocks[0].ops)),
            "source_operation_count_scope": "top_level_function_block",
            "mxu_units": list(self.units),
            "input_slots": self.slots,
            "output": {"index": 0, "dram_address": self.output_base,
                       "payload_bytes": 2048, "physical_layout": "bf16_pair_halves"},
            "numeric_scope": "explicit_precision_transform_diagnostic",
            "unsupported_scope": ["non_32x32_tiles", "nonzero_accumulator", "dynamic_shapes",
                                  "unqualified_numerical_inputs", "multiple_results"],
        }


def parse_module(source: str) -> Any:
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, func.Func, linalg.Linalg, tensor.Tensor,
                    arith.Arith, affine.Affine):
        context.load_dialect(dialect)
    module = Parser(context, source).parse_module()
    module.verify()
    return module


def stage_constants(weights_path: Path, slots: list[dict[str, Any]],
                    output: Path) -> list[dict[str, Any]]:
    try:
        import numpy as np
        from safetensors.numpy import load_file
    except ImportError as error:
        raise UnsupportedLinalg("constant staging requires numpy and safetensors") from error
    tensors = load_file(weights_path)
    constants = output / "constants"
    constants.mkdir()
    recorded = []
    for slot in slots:
        if slot["source_kind"] != "param":
            continue
        key = slot["source_key"]
        require(isinstance(key, str) and key in tensors,
                f"constant {key!r} is absent from the selected weights file")
        values = tensors[key]
        require(values.dtype == np.dtype("float32") and
                list(values.shape) == slot["source_shape"],
                f"constant {key!r} has changed dtype or shape")
        role = slot["physical_role"]
        if role == "bias_bf16":
            data = pack_bias_tile(values)
        elif role == "weight_fp8_nk":
            # Captured Linear stores N x K. The checked Linalg transpose makes
            # it K x N for matmul; Atlas MXU weights consume the original N x K.
            data = pack_fp8_tile(values)
        else:
            raise UnsupportedLinalg(f"parameter role {role!r} has no packer")
        path = constants / f"slot-{slot['slot']:02d}.bin"
        path.write_bytes(data)
        recorded.append({
            "slot": slot["slot"], "source_key": key,
            "payload_file": str(path.relative_to(output)),
            "payload_bytes": len(data), "payload_sha256": sha256(path),
            "dram_address": slot["dram_address"],
        })
    return recorded


def run(command: list[str], *, input_text: str | None = None) -> str:
    result = subprocess.run(command, input=input_text, text=True,
                            capture_output=True)
    if result.returncode:
        raise UnsupportedLinalg(
            f"tool failure ({result.returncode}): {' '.join(command)}: "
            f"{result.stderr.strip()[:2000]}"
        )
    return result.stdout


def compile_program(virtual: str, output: Path, atlas: Path,
                    llvm: Path, linker: Path) -> dict[str, Any]:
    atlas_opt = str(atlas / "atlas-opt")
    atlas_emit = str(atlas / "atlas-emit")
    machine = run([atlas_opt, "--lower-atlas-virtual-to-machine", "-"],
                  input_text=virtual).rstrip() + "\n"
    require("atlas.generated_from_virtual" in machine,
            "virtual lowerer did not mark the generated physical stream")
    (output / "atlas-machine.mlir").write_text(machine)
    run([atlas_opt, "--verify-atlas-generated-schedule",
         "--verify-atlas-machine-stream", "-"], input_text=machine)
    words_text = run([atlas_emit, "-"], input_text=machine)
    words = tuple(int(line, 16) for line in words_text.splitlines())
    require(words, "emitter produced no Atlas instructions")
    (output / "program.words.txt").write_text(words_text)
    word_map = json.loads(run([atlas_emit, "--map-json", "-"],
                              input_text=machine))
    require(word_map.get("word_count") == len(words),
            "machine word map differs from emitted stream")
    (output / "word-map.json").write_text(json.dumps(word_map, indent=2) + "\n")
    physical_program = json.loads(run([atlas_emit, "--program-json", "-"],
                                      input_text=machine))
    require(physical_program.get("schema") == "atlas.physical_program.v1"
            and physical_program.get("selected_rtl_revision") == RTL_REVISION
            and physical_program.get("word_count") == len(words)
            and [row.get("word_u32") for row in
                 physical_program.get("instructions", [])] == list(words),
            "physical functional stream differs from emitted Atlas words")
    (output / "physical-program.json").write_text(
        json.dumps(physical_program, indent=2) + "\n")
    structured = run([atlas_opt, "--convert-atlas-to-llvm-calls", "-"],
                     input_text=machine).rstrip() + "\n"
    require(structured.count("llvm.call @atlas_emit_") == len(words),
            "structured LLVM call count differs from emitted instructions")
    (output / "atlas-llvm-structured.mlir").write_text(structured)
    encoded = run([atlas_opt, "--finalize-atlas-llvm-calls", "-"],
                  input_text=structured).rstrip() + "\n"
    require(encoded.count(".word 0x") == len(words),
            "final LLVM block differs from emitted instructions")
    (output / "atlas-llvm.mlir").write_text(encoded)
    llvm_ir = run([str(llvm / "mlir-translate"), "--mlir-to-llvmir"],
                  input_text=encoded)
    (output / "program.ll").write_text(llvm_ir)
    assembly = run([str(llvm / "llc"), "-mtriple=riscv32-unknown-elf",
                    "-mattr=-c", "-filetype=asm", "-o", "-"],
                   input_text=llvm_ir)
    (output / "program.s").write_text(assembly)
    obj = output / "program.o"
    run([str(llvm / "llc"), "-mtriple=riscv32-unknown-elf",
         "-mattr=-c", "-filetype=obj", "-o", str(obj)],
        input_text=llvm_ir)
    elf = output / "program.elf"
    run([str(linker), "-m", "elf32lriscv", "--no-relax", "-Ttext=0",
         "-e", "atlas_program", "-o", str(elf), str(obj)])
    object_text = output / "program.text.bin"
    run([str(llvm / "llvm-objcopy"), "--dump-section",
         f".text={object_text}", str(obj)])
    linked_text = output / "program.elf.text.bin"
    run([str(llvm / "llvm-objcopy"), "--dump-section",
         f".text={linked_text}", str(elf)])
    expected = b"".join(struct.pack("<I", word) for word in words)
    require(object_text.read_bytes().startswith(expected) and
            linked_text.read_bytes() == object_text.read_bytes(),
            "LLVM object or linked ELF text differs from checked Atlas words")
    return {
        "word_count": len(words),
        "virtual_sha256": sha256(output / "atlas-virtual.mlir"),
        "machine_sha256": sha256(output / "atlas-machine.mlir"),
        "object_sha256": sha256(obj), "elf_sha256": sha256(elf),
        "physical_program_sha256": sha256(output / "physical-program.json"),
        "linked_text_matches_object": True,
        "object_prefix_matches_emitter": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linalg", type=Path, required=True)
    parser.add_argument("--argument-manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True,
                        help="frozen safetensors parameters, not runtime inputs")
    parser.add_argument("--atlas-bin-dir", type=Path, required=True)
    parser.add_argument("--llvm-bin-dir", type=Path, required=True)
    parser.add_argument("--linker", type=Path, required=True)
    parser.add_argument("--input-base", type=lambda x: int(x, 0), default=0x90000000)
    parser.add_argument("--output-base", type=lambda x: int(x, 0), default=0x90010000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.linalg, args.argument_manifest, args.policy, args.weights):
        if not path.is_file():
            parser.error(f"missing compiler input: {path}")
    for path in (args.atlas_bin_dir / "atlas-opt", args.atlas_bin_dir / "atlas-emit",
                 args.llvm_bin_dir / "mlir-translate", args.llvm_bin_dir / "llc",
                 args.llvm_bin_dir / "llvm-objcopy", args.linker):
        if not path.is_file():
            parser.error(f"missing compiler tool: {path}")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be empty to prevent stale success artifacts")
    try:
        policy = json.loads(args.policy.read_text())
        manifest = json.loads(args.argument_manifest.read_text())
        module = parse_module(args.linalg.read_text())
        declared_weights = module.attributes.get("prov.weights_file")
        if declared_weights is not None:
            require(isinstance(declared_weights, builtin.StringAttr),
                    "source weight identity is not a string path")
            declared_path = Path(declared_weights.data)
            if not declared_path.is_absolute():
                declared_path = args.linalg.parent / declared_path
            require(declared_path.resolve() == args.weights.resolve(),
                    "source-declared weights file differs from compiler input")
        virtual, report = Importer(module,
                                   manifest, policy, args.input_base,
                                   args.output_base).compile()
    except (UnsupportedLinalg, ValueError, KeyError) as error:
        parser.error(f"unsupported_semantics: {error}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "atlas-virtual.mlir").write_text(virtual)
    try:
        report["constants"] = stage_constants(args.weights, report["input_slots"],
                                               output)
        report["program"] = compile_program(virtual, output,
                                            args.atlas_bin_dir.resolve(),
                                            args.llvm_bin_dir.resolve(),
                                            args.linker.absolute())
    except UnsupportedLinalg as error:
        parser.error(f"compile_error: {error}")
    report["source_sha256"] = sha256(args.linalg)
    report["argument_manifest_sha256"] = sha256(args.argument_manifest)
    report["policy_sha256"] = sha256(args.policy)
    report["weights_sha256"] = sha256(args.weights)
    report["compiler_source_sha256"] = sha256(Path(__file__))
    report["selected_rtl_revision"] = RTL_REVISION
    report["toolchain"] = {
        "atlas_opt_sha256": sha256(args.atlas_bin_dir / "atlas-opt"),
        "atlas_emit_sha256": sha256(args.atlas_bin_dir / "atlas-emit"),
        "llvm_version": run([str(args.llvm_bin_dir / "mlir-translate"),
                             "--version"]).strip(),
        "linker_version": run([str(args.linker.absolute()),
                               "--version"]).strip(),
    }
    (output / "import-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(output / "import-manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
