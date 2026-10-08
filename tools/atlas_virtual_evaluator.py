"""Atlas virtual IR parsing and bounded reference execution over immutable tiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import zip_longest
from types import MappingProxyType
from typing import Literal

try:
    from xdsl.context import Context
    from xdsl.dialects import arith, builtin, cf, func
    from xdsl.ir import Attribute, Block, Operation, SSAValue
    from xdsl.irdl.dominance import DominanceInfo
    from xdsl.parser import Parser
    from xdsl.utils.exceptions import ParseError, VerifyException
except ModuleNotFoundError as error:
    raise ImportError("Atlas virtual parsing requires tools/requirements-virtual-evaluator.txt") from error


CONTRACT_VERSION = "atlas.virtual-evaluator.v1"
COMPATIBILITY_REVISION = "23fdbf51f89a829bfcd24bbd32fa179616e26d9f"
TileFormat = Literal["bf16", "fp8"]


class VirtualInterfaceError(ValueError):
    """Malformed program or runtime data at the evaluator interface."""


class UnsupportedVirtualMode(VirtualInterfaceError):
    """A parsed form is outside the declared evaluator input contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VirtualInterfaceError(message)


@dataclass(frozen=True)
class Tile:
    """1024 raw encodings, indexed as row * 32 + column; no quantization."""

    format: TileFormat
    bits: tuple[int, ...]

    def __post_init__(self) -> None:
        _require(self.format in ("bf16", "fp8"), "tile format must be bf16 or fp8")
        bits = tuple(self.bits)
        _require(len(bits) == 1024, "logical tiles must contain 32x32 encodings")
        limit = 1 << (16 if self.format == "bf16" else 8)
        _require(all(type(value) is int and 0 <= value < limit for value in bits), f"{self.format} tile requires unsigned raw integer encodings")
        object.__setattr__(self, "bits", bits)

    @property
    def shape(self) -> tuple[int, int]:
        return (32, 32)


@dataclass(frozen=True)
class Scalar:
    """An i1/i32 control represented by unsigned bits, without implicit wrap."""

    width: Literal[1, 32]
    bits: int

    def __post_init__(self) -> None:
        _require(type(self.width) is int and self.width in (1, 32), "scalar width must be 1 or 32")
        _require(type(self.bits) is int and 0 <= self.bits < (1 << self.width), f"i{self.width} control requires unsigned raw integer bits")


@dataclass(frozen=True)
class MemoryRegion:
    """An immutable DRAM byte snapshot, including any caller-supplied guards."""

    address: int
    data: bytes

    def __post_init__(self) -> None:
        _require(type(self.address) is int and 0 <= self.address < (1 << 32), "memory address must be an unsigned 32-bit byte address")
        _require(isinstance(self.data, (bytes, bytearray, memoryview)), "memory data must be a byte buffer")
        data = bytes(self.data)
        _require(bool(data) and self.address + len(data) <= (1 << 32), "memory region must be nonempty and within 32-bit DRAM")
        object.__setattr__(self, "data", data)


def _tile_map(tiles: Mapping[int, Tile]) -> Mapping[int, Tile]:
    result = dict(tiles)
    _require(
        all(type(index) is int and 0 <= index <= 0x7FFFFFFF and isinstance(tile, Tile) for index, tile in result.items()),
        "boundary tiles require nonnegative i32 indices and Tile values",
    )
    return MappingProxyType(result)


def _memory_regions(regions: tuple[MemoryRegion, ...]) -> tuple[MemoryRegion, ...]:
    regions = tuple(regions)
    _require(all(isinstance(region, MemoryRegion) for region in regions), "memory must contain MemoryRegion snapshots")
    ordered = sorted(regions, key=lambda region: region.address)
    for left, right in zip(ordered, ordered[1:]):
        _require(left.address + len(left.data) <= right.address, "initial/result memory regions must not overlap")
    return regions


@dataclass(frozen=True)
class RuntimeInputs:
    tiles: Mapping[int, Tile] = field(default_factory=dict)
    controls: tuple[Scalar, ...] = ()
    memory: tuple[MemoryRegion, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tiles", _tile_map(self.tiles))
        controls = tuple(self.controls)
        _require(all(isinstance(control, Scalar) for control in controls), "controls must contain Scalar values in function argument order")
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "memory", _memory_regions(self.memory))


@dataclass(frozen=True)
class EvaluationResult:
    """Logical outputs and final snapshots of the input memory regions, including guards."""

    outputs: Mapping[int, Tile]
    memory: tuple[MemoryRegion, ...] = ()

    def __post_init__(self) -> None:
        outputs = _tile_map(self.outputs)
        _require(all(tile.format == "bf16" for tile in outputs.values()), "boundary outputs are BF16; FP8 stores are memory effects")
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "memory", _memory_regions(self.memory))


def compare_results(expected: EvaluationResult, actual: EvaluationResult) -> None:
    """Compare visible bits and mapped bytes, independent of event or region order."""
    if expected.outputs.keys() != actual.outputs.keys():
        raise AssertionError(f"output indices: expected {sorted(expected.outputs)}, got {sorted(actual.outputs)}")
    for index in sorted(expected.outputs):
        for element, (wanted, observed) in enumerate(zip(expected.outputs[index].bits, actual.outputs[index].bits)):
            if wanted != observed:
                row, col = divmod(element, 32)
                raise AssertionError(f"output {index} tile[{row},{col}]: expected 0x{wanted:04x}, got 0x{observed:04x}")

    def bytes_at(regions):
        for region in sorted(regions, key=lambda region: region.address):
            yield from ((region.address + offset, value) for offset, value in enumerate(region.data))

    for wanted, observed in zip_longest(bytes_at(expected.memory), bytes_at(actual.memory)):
        if wanted is None or observed is None or wanted[0] != observed[0]:
            address = min(item[0] for item in (wanted, observed) if item is not None)
            side = "actual" if wanted is not None and wanted[0] == address else "expected"
            raise AssertionError(f"memory mapping at 0x{address:08x}: byte missing from {side} result")
        if wanted[1] != observed[1]:
            raise AssertionError(f"memory at 0x{wanted[0]:08x}: expected 0x{wanted[1]:02x}, got 0x{observed[1]:02x}")


def operation_name(op: Operation) -> str:
    return op.op_name.data if isinstance(op, builtin.UnregisteredOp) else op.name


def _type_kind(attr: Attribute) -> tuple[str, int | None]:
    if isinstance(attr, builtin.IntegerType) and attr in (builtin.i1, builtin.i32):
        return f"i{attr.width.data}", None
    if isinstance(attr, builtin.UnregisteredAttr) and attr.is_type.data:
        name = attr.attr_name.data
        prefix = "atlas.virtual_"
        if name.startswith(prefix):
            kind = name[len(prefix):]
            parameter = attr.value.data.strip()
            if kind in ("mxu_weight", "mxu_acc") and parameter in ("0", "1"):
                return kind, int(parameter)
            if not parameter and kind in ("state", "bf16", "fp8", "scale", "dma_load_bf16", "dma_load_fp8", "dma_store"):
                return kind, None
    raise UnsupportedVirtualMode(f"unsupported virtual type: {attr}")


# Operand kinds, result kinds, attributes. MXU unit equality is checked separately.
_SIGNATURES = {
    "start": ((), ("state",), ()),
    "input_bf16": (("state",), ("state", "bf16"), ("index",)),
    "input_fp8": (("state",), ("state", "fp8"), ("index",)),
    "output_bf16": (("state", "bf16"), ("state",), ("index",)),
    "vpu_unary": (("bf16",), ("bf16",), ("kind",)),
    "vpu_binary": (("bf16", "bf16"), ("bf16",), ("kind",)),
    "pack_fp8": (("bf16",), ("fp8",), ("scale_code",)),
    "mxu_matmul": (("fp8", "fp8"), ("bf16",), ("unit",)),
    "mxu_load_weight": (("state", "fp8"), ("state", "mxu_weight"), ("unit",)),
    "mxu_load_acc_fp8": (("state", "fp8"), ("state", "mxu_acc"), ("unit",)),
    "mxu_load_acc_bf16": (("state", "bf16"), ("state", "mxu_acc"), ("unit",)),
    "mxu_reset": (("state", "fp8", "mxu_weight"), ("state", "mxu_acc"), ()),
    "mxu_accumulate": (("state", "fp8", "mxu_weight", "mxu_acc"), ("state", "mxu_acc"), ()),
    "mxu_readout_bf16": (("state", "mxu_acc"), ("state", "bf16"), ()),
    "mxu_readout_fp8": (("state", "mxu_acc", "scale"), ("state", "fp8"), ()),
    "scale_constant": ((), ("scale",), ("code",)),
    "dma_load_fp8": (("state", "i32", "i32"), ("state", "dma_load_fp8"), ()),
    "dma_load_bf16": (("state", "i32", "i32"), ("state", "dma_load_bf16"), ()),
    "dma_await_fp8": (("state", "dma_load_fp8"), ("state", "fp8"), ()),
    "dma_await_bf16": (("state", "dma_load_bf16"), ("state", "bf16"), ()),
    "dma_store_fp8": (("state", "fp8", "i32", "i32"), ("state", "dma_store"), ()),
    "dma_store_bf16": (("state", "bf16", "i32", "i32"), ("state", "dma_store"), ()),
    "dma_wait": (("state", "dma_store"), ("state",), ()),
}

_STANDARD_PROPERTIES = {
    "arith.constant": {"value"},
    "arith.addi": {"overflowFlags"},
    "arith.cmpi": {"predicate"},
    "cf.br": set(),
    "cf.cond_br": {"operandSegmentSizes"},
    "func.return": set(),
}


def _atlas_fields(op: Operation) -> dict[str, Attribute]:
    # MLIR canonical printing moves inherent fields into <{properties}>.
    attributes = {name: value for name, value in op.attributes.items() if name != "op_name__"}
    _require(not attributes.keys() & op.properties.keys(), f"{operation_name(op)}: duplicate attribute/property")
    return attributes | op.properties


def _integer_attribute(op: Operation, name: str) -> int:
    value = _atlas_fields(op).get(name)
    _require(isinstance(value, builtin.IntegerAttr) and value.type == builtin.i32, f"{operation_name(op)} requires {name} : i32")
    assert isinstance(value, builtin.IntegerAttr)
    return value.value.data


def _check_atlas_operation(op: Operation) -> None:
    name = operation_name(op)
    short = name.removeprefix("atlas.virtual_")
    if not name.startswith("atlas.virtual_") or short not in _SIGNATURES:
        raise UnsupportedVirtualMode(f"unsupported virtual operation: {name}")
    _require(not op.regions and not op.successors, f"{name}: regions and successors are unsupported")
    fields = _atlas_fields(op)
    operands, results, attributes = _SIGNATURES[short]
    types = [_type_kind(value.type) for value in (*op.operands, *op.results)]
    _require(
        tuple(kind for kind, _ in types[:len(op.operands)]) == operands and tuple(kind for kind, _ in types[len(op.operands):]) == results,
        f"{name}: incorrect operand/result signature",
    )
    _require(set(fields) == set(attributes), f"{name}: expected attributes {attributes}")
    units = {unit for _, unit in types if unit is not None}
    if "unit" in attributes:
        unit = _integer_attribute(op, "unit")
        _require(unit in (0, 1), f"{name}: unit must be 0 or 1")
        units.add(unit)
    _require(len(units) <= 1, f"{name}: MXU handle units must agree")
    if "index" in attributes:
        _require(_integer_attribute(op, "index") >= 0, f"{name}: boundary index must be nonnegative")
    if "code" in attributes:
        _require(0 <= _integer_attribute(op, "code") <= 255, f"{name}: scale code must be in 0..255")
    if "kind" in attributes:
        kind = fields["kind"]
        allowed = ("mov", "relu") if short == "vpu_unary" else ("add",)
        if not isinstance(kind, builtin.StringAttr) or kind.data not in allowed:
            raise UnsupportedVirtualMode(f"{name}: supported kinds are {allowed}")
    if short == "pack_fp8" and _integer_attribute(op, "scale_code") != 127:
        raise UnsupportedVirtualMode(f"{name}: only scale_code 127 is admitted")


def _check_operation(op: Operation, *, flat: bool) -> None:
    if isinstance(op, builtin.UnregisteredOp):
        _check_atlas_operation(op)
        return
    if isinstance(op, arith.ConstantOp):
        _require(isinstance(op.value, builtin.IntegerAttr) and op.result.type in (builtin.i1, builtin.i32), "arith.constant requires an i1/i32 integer")
    elif isinstance(op, arith.AddiOp):
        _require(op.result.type == builtin.i32 and not op.overflow_flags.data, "arith.addi requires wrapping i32 arithmetic")
    elif isinstance(op, arith.CmpiOp):
        _require(
            op.lhs.type == builtin.i32 and op.rhs.type == builtin.i32 and op.predicate.type == builtin.i64 and 0 <= op.predicate.value.data < 10,
            "arith.cmpi requires i32 operands and one of the ten predicates encoded as i64",
        )
    elif not flat and isinstance(op, (cf.BranchOp, cf.ConditionalBranchOp, func.ReturnOp)):
        pass
    else:
        raise UnsupportedVirtualMode(f"unsupported virtual operation: {operation_name(op)}")
    _require(not op.attributes and set(op.properties) <= _STANDARD_PROPERTIES[op.name], f"{op.name}: unsupported attributes/properties")
    _require(not op.regions, f"{operation_name(op)}: nested regions are unsupported")
    for value in (*op.operands, *op.results):
        _type_kind(value.type)


@dataclass(frozen=True)
class ParsedProgram:
    """Retains xDSL SSA/CFG identities; admission does not establish stream or hardware legality."""

    module: builtin.ModuleOp
    function: func.FuncOp | None
    blocks: tuple[Block, ...]
    input_formats: Mapping[int, TileFormat]
    output_indices: tuple[int, ...]
    control_widths: tuple[int, ...]

    @property
    def operations(self) -> tuple[Operation, ...]:
        return tuple(op for block in self.blocks for op in block.ops)

    def validate_inputs(self, inputs: RuntimeInputs) -> None:
        _require(isinstance(inputs, RuntimeInputs), "expected RuntimeInputs")
        _require(set(inputs.tiles) == set(self.input_formats), f"expected boundary input indices {sorted(self.input_formats)}")
        for index, format in self.input_formats.items():
            _require(inputs.tiles[index].format == format, f"input {index} requires {format}, got {inputs.tiles[index].format}")
        _require(tuple(value.width for value in inputs.controls) == self.control_widths, f"expected scalar controls with widths {self.control_widths}")


def parse_program(source: str, *, function: str | None = None) -> ParsedProgram:
    """Parse flat IR or a selected function; input declarations include all CFG paths."""
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, func.Func, arith.Arith, cf.Cf):
        context.load_dialect(dialect)
    try:
        module = Parser(context, source).parse_module()
        module.verify()
    except (ParseError, VerifyException) as error:
        raise VirtualInterfaceError(f"invalid virtual MLIR: {error}") from error
    top = tuple(module.body.block.ops)
    functions = [op for op in top if isinstance(op, func.FuncOp)]
    selected = None
    controls: tuple[int, ...] = ()
    if functions:
        _require(len(functions) == len(top), "module must not mix functions and flat operations")
        names = [op.sym_name.data for op in functions]
        _require(len(set(names)) == len(names), "duplicate function symbols")
        _require(function is not None or len(functions) == 1, "select a function explicitly in a multi-function module")
        matches = [op for op in functions if function is None or op.sym_name.data == function]
        _require(len(matches) == 1, f"function {function!r} not found")
        selected = matches[0]
        blocks = tuple(selected.body.blocks)
        _require(bool(blocks), "external functions are unsupported")
        _require(tuple(_type_kind(attr) for attr in selected.function_type.outputs) == (("state", None),), "virtual function must return one virtual_state")
        _require(all(arg.type in (builtin.i1, builtin.i32) for arg in blocks[0].args), "function arguments must be i1/i32 controls")
        controls = tuple(arg.type.width.data for arg in blocks[0].args)
        for block in blocks[1:]:
            kinds = tuple(_type_kind(arg.type)[0] for arg in block.args)
            _require(
                bool(kinds) and kinds[0] == "state" and all(kind in ("bf16", "i1", "i32") for kind in kinds[1:]),
                "non-entry block requires state followed by BF16/i1/i32 arguments",
            )
    else:
        _require(function is None, f"function {function!r} not found in flat module")
        blocks = (module.body.block,)
    inputs: dict[int, TileFormat] = {}
    outputs: list[int] = []
    for block in blocks:
        for op in block.ops:
            _check_operation(op, flat=selected is None)
            name = operation_name(op)
            if name in ("atlas.virtual_input_bf16", "atlas.virtual_input_fp8"):
                index = _integer_attribute(op, "index")
                format: TileFormat = "bf16" if name.endswith("bf16") else "fp8"
                _require(index not in inputs or inputs[index] == format, f"input {index} is declared with conflicting tile formats")
                inputs[index] = format
            elif name == "atlas.virtual_output_bf16":
                index = _integer_attribute(op, "index")
                _require(index not in outputs, f"duplicate output index {index}")
                outputs.append(index)
    return ParsedProgram(module, selected, blocks, MappingProxyType(inputs), tuple(outputs), controls)


def evaluate_tile_operation(op: Operation, operands: tuple[Tile, ...]) -> Tile:
    """Evaluate a pure VPU/pack operation, including FP8 results without boundary I/O."""
    _require(isinstance(op, Operation), "expected a parsed virtual operation")
    _check_atlas_operation(op)
    name = operation_name(op)
    if name not in ("atlas.virtual_vpu_unary", "atlas.virtual_vpu_binary", "atlas.virtual_pack_fp8"):
        raise UnsupportedVirtualMode(f"not a supported pure tile operation: {name}")
    _require(isinstance(operands, tuple) and all(isinstance(tile, Tile) for tile in operands), "tile operation operands must be a tuple of Tile values")
    formats = tuple(_type_kind(value.type)[0] for value in op.operands)
    _require(tuple(tile.format for tile in operands) == formats, f"{name}: expected tile operand formats {formats}")
    kind = _atlas_fields(op)["kind"].data if name != "atlas.virtual_pack_fp8" else "pack_fp8"
    if kind == "mov":
        return operands[0]
    try:
        import torch
        from npu_model.configs.numerics import RtlNumerics
    except ModuleNotFoundError as error:
        raise ImportError("VPU execution requires the npu-model Python environment and its source root on PYTHONPATH") from error
    # Reinterpret owned raw bits: floating-point conversion would lose encodings.
    owned = tuple(torch.tensor(tile.bits, dtype=torch.uint16).view(torch.bfloat16).reshape(32, 32) for tile in operands)
    if kind == "add":
        # Host FP32 must preserve subnormals and round before the BF16 bit chop.
        probe_a = torch.tensor((0x0001, 0x0080, 0x3F80, 0xBF80), dtype=torch.uint16).view(torch.bfloat16)
        probe_b = torch.tensor((0x0001, 0x8081, 0x8001, 0x0001), dtype=torch.uint16).view(torch.bfloat16)
        if tuple(RtlNumerics.add(probe_a, probe_b).view(torch.uint16).tolist()) != (0x0002, 0x8001, 0x3F80, 0xBF80):
            raise UnsupportedVirtualMode("BF16 add requires FP32 round-to-nearest-even with gradual underflow")
        result = RtlNumerics.add(*owned)
    elif kind == "pack_fp8":
        result = RtlNumerics.to_fp8(owned[0], _integer_attribute(op, "scale_code"))
    else:
        result = RtlNumerics.unary(kind, owned[0])
    format: TileFormat = "fp8" if kind == "pack_fp8" else "bf16"
    raw = result.contiguous().view(torch.uint8 if format == "fp8" else torch.uint16)
    return Tile(format, tuple(raw.flatten().tolist()))


def _mxu_unit(op: Operation) -> int:
    if "unit" in _atlas_fields(op):
        return _integer_attribute(op, "unit")
    return next(unit for value in op.operands for kind, unit in (_type_kind(value.type),) if kind in ("mxu_weight", "mxu_acc"))


def _mxu_tile(kind: str, operands: tuple[Tile, ...], scale: int = 127, *, unit: int = 0) -> Tile:
    if kind == "matmul":
        # E4M3Mul flushes exponent-zero products and treats raw 7f/ff as +/-480.
        # Keep the addend within the selected MAC's clean BF16 input domain.
        for tile in operands:
            if tile.format == "bf16" and not all((bits & 0x7FFF) == 0 or 0x0080 <= (bits & 0x7FFF) <= 0x7F7F for bits in tile.bits):
                raise UnsupportedVirtualMode(f"MXU{unit} contraction requires finite normal BF16 accumulators or signed zero")
    try:
        import torch
        from npu_model.configs.numerics import RtlNumerics
    except ModuleNotFoundError as error:
        raise ImportError("MXU execution requires the npu-model Python environment and its source root on PYTHONPATH") from error
    tensors = tuple(torch.tensor(tile.bits, dtype=torch.uint8 if tile.format == "fp8" else torch.uint16)
                    .view(torch.float8_e4m3fn if tile.format == "fp8" else torch.bfloat16).reshape(32, 32) for tile in operands)
    if kind == "matmul":
        # Stored weights are W[N,K], while the numerical helper consumes B[K,N].
        matmul = RtlNumerics.systolic_matmul if unit == 0 else RtlNumerics.inner_product_matmul
        result = matmul(tensors[0], tensors[1].T.contiguous(), tensors[2])
    elif kind == "seed_fp8":
        result = RtlNumerics.from_fp8(tensors[0], 127)
    else:
        result = RtlNumerics.acc_to_fp8(tensors[0], scale)
    format: TileFormat = "fp8" if kind == "readout_fp8" else "bf16"
    raw = result.contiguous().view(torch.uint8 if format == "fp8" else torch.uint16)
    return Tile(format, tuple(raw.flatten().tolist()))


def _check_mxu_handles(operations: tuple[Operation, ...]) -> None:
    """Recompute block-local resource identities; no physical slot assignments."""
    remaining: dict[SSAValue, int] = {}
    for op in operations:
        for operand in op.operands:
            if _type_kind(operand.type)[0] == "mxu_weight":
                remaining[operand] = remaining.get(operand, 0) + 1
    weights: list[set[SSAValue]] = [set(), set()]
    accumulators: list[set[SSAValue]] = [set(), set()]
    for op in operations:
        name = operation_name(op).removeprefix("atlas.virtual_")
        if not name.startswith("mxu_"):
            continue
        unit = _mxu_unit(op)
        live_weights = weights[unit] = {handle for handle in weights[unit] if remaining.get(handle, 0)}
        live_accumulators = accumulators[unit]
        if name == "mxu_matmul":
            _require(not live_weights and not live_accumulators, "legacy matmul cannot overlap live explicit MXU handles on its unit")
        elif name == "mxu_load_weight":
            _require(len(live_weights) < 2, "at most two live MXU weights per unit are admitted")
            live_weights.add(op.results[1])
        else:
            if name in ("mxu_accumulate", "mxu_readout_bf16", "mxu_readout_fp8"):
                handle = op.operands[3] if name == "mxu_accumulate" else op.operands[1]
                _require(handle in live_accumulators, f"{name}: expected a current, unconsumed accumulator version")
                live_accumulators.remove(handle)
            if name in ("mxu_reset", "mxu_accumulate"):
                weight = op.operands[2]
                _require(weight in live_weights, f"{name}: expected a live weight handle")
                remaining[weight] -= 1
            if name in ("mxu_load_acc_bf16", "mxu_load_acc_fp8", "mxu_reset", "mxu_accumulate"):
                _require(len(live_accumulators) < 2, "at most two live MXU accumulators per unit are admitted")
                live_accumulators.add(op.results[1])
    _require(not any(accumulators), "every MXU accumulator must be read out before block exit")


def _constant_i32(value: SSAValue, cache: dict[SSAValue, int | None]) -> int | None:
    """Evaluate admitted constant expressions without a Python recursion-depth limit."""
    pending = [(value, False)]
    while pending:
        current, expanded = pending.pop()
        owner = current.owner
        if expanded:
            left, right = cache[owner.lhs], cache[owner.rhs]
            if left is not None and right is not None:
                cache[current] = (left + right) & 0xFFFFFFFF
        elif current not in cache:
            cache[current] = None
            if current.type == builtin.i32 and isinstance(owner, arith.ConstantOp):
                cache[current] = owner.value.value.data & 0xFFFFFFFF
            elif current.type == builtin.i32 and isinstance(owner, arith.AddiOp) and not owner.overflow_flags.data:
                pending.append((current, True))
                pending.extend((operand, False) for operand in owner.operands)
    return cache[value]


def _dma_span(op: Operation, constants: dict[SSAValue, int | None]) -> tuple[int, int, TileFormat, bool]:
    name = operation_name(op)
    store = "dma_store_" in name
    format: TileFormat = "fp8" if name.endswith("fp8") else "bf16"
    address, length = (_constant_i32(value, constants) for value in op.operands[-2:])
    _require(address is not None and length is not None, "DMA address/length require i32 constant/wrapping-add proof")
    assert address is not None and length is not None
    _require(length == (1024 if format == "fp8" else 2048), "DMA length must equal the complete tile size")
    _require(address >= 0x80000000 and address % 32 == 0, "DMA address must be selected DRAM aligned to 32 bytes")
    _require(address + length <= 1 << 32, "DMA span exceeds the 32-bit address space")
    return address, length, format, store


def _overlaps(address: int, length: int, other: int, size: int) -> bool:
    return address < other + size and other < address + length


def _boundary_ranges(program: ParsedProgram) -> tuple[tuple[int, int, str], ...]:
    if program.function is None:
        return ()
    ranges = []
    for kind, indices in (("input", program.input_formats), ("output", program.output_indices), ("control", (0,) if program.control_widths else ())):
        attribute = program.function.attributes.get(f"atlas.{kind}_dram_base")
        if attribute is None:
            continue
        _require(isinstance(attribute, builtin.IntegerAttr), f"{kind} DRAM base must be an integer attribute")
        for index in indices:
            address = attribute.value.data + (index * 2048 if kind != "control" else 0)
            size = 4 * len(program.control_widths) if kind == "control" else (1024 if kind == "input" and program.input_formats[index] == "fp8" else 2048)
            _require(0 <= address and address + size <= 1 << 32, f"{kind} boundary span exceeds 32-bit DRAM")
            ranges.append((address, size, kind))
    return tuple(ranges)


def _check_dma_handles(operations: tuple[Operation, ...], boundaries: tuple[tuple[int, int, str], ...], constants: dict[SSAValue, int | None]) -> None:
    pending: dict[SSAValue, tuple[int, int, TileFormat, bool]] = {}
    for op in operations:
        name = operation_name(op).removeprefix("atlas.virtual_")
        if name.startswith(("dma_load_", "dma_store_")):
            address, length, format, store = span = _dma_span(op, constants)
            _require(len(pending) < 2, "at most two pending DMA transfers are admitted")
            for other, size, _, writing in pending.values():
                if (store or writing) and _overlaps(address, length, other, size):
                    raise UnsupportedVirtualMode("overlapping pending DMA ranges involving a write are unqualified")
            for other, size, kind in boundaries:
                if not _overlaps(address, length, other, size):
                    continue
                _require(kind != "control", "DMA must not overlap the control mailbox")
                if store or kind == "output":
                    raise UnsupportedVirtualMode("explicit DMA aliases an implicit boundary buffer with a write")
            pending[op.results[1]] = span
        elif name.startswith("dma_await_") or name == "dma_wait":
            _require(op.operands[1] in pending, f"{name}: expected a pending, unconsumed DMA handle")
            del pending[op.operands[1]]
        elif name in ("input_bf16", "input_fp8", "output_bf16", "pack_fp8"):
            _require(not pending, f"{name}: pending DMA must complete before implicit I/O or pack")
    _require(not pending, "every pending DMA transfer must complete before block exit")


def _tile_bytes(tile: Tile) -> bytes:
    if tile.format == "fp8":
        return bytes(tile.bits)
    return b"".join(tile.bits[row * 32 + col].to_bytes(2, "little") for half in (0, 16) for row in range(32) for col in range(half, half + 16))


def _tile_from_bytes(data: bytes, format: TileFormat) -> Tile:
    if format == "fp8":
        return Tile("fp8", tuple(data))
    words = tuple(int.from_bytes(data[index:index + 2], "little") for index in range(0, len(data), 2))
    return Tile("bf16", tuple(words[(col // 16) * 512 + row * 16 + col % 16] for row in range(32) for col in range(32)))


class _Memory:
    def __init__(self, regions: tuple[MemoryRegion, ...]):
        self.regions = regions
        self.buffers = [bytearray(region.data) for region in regions]
        self.order = sorted(range(len(regions)), key=lambda index: regions[index].address)

    def chunks(self, address: int, size: int) -> tuple[tuple[int, int, int], ...]:
        chunks = []
        cursor, end = address, address + size
        for index in self.order:
            region = self.regions[index]
            first, last = max(address, region.address), min(end, region.address + len(region.data))
            if first < last:
                _require(first == cursor, f"DMA span has unmapped memory at 0x{cursor:08x}")
                chunks.append((index, first - region.address, last - region.address))
                cursor = last
        _require(cursor == end, f"DMA span has unmapped memory at 0x{cursor:08x}")
        return tuple(chunks)

    def read(self, address: int, size: int) -> bytes:
        return b"".join(self.buffers[index][first:last] for index, first, last in self.chunks(address, size))

    def write(self, address: int, data: bytes) -> None:
        offset = 0
        # Validate the entire span before modifying any region.
        for index, first, last in self.chunks(address, len(data)):
            self.buffers[index][first:last] = data[offset:offset + last - first]
            offset += last - first

    def snapshots(self) -> tuple[MemoryRegion, ...]:
        return tuple(MemoryRegion(region.address, data) for region, data in zip(self.regions, self.buffers))


@dataclass(frozen=True)
class _Transfer:
    address: int
    data: bytes
    format: TileFormat


def _edges(op: Operation) -> tuple[tuple[Block, tuple[SSAValue, ...]], ...]:
    if isinstance(op, cf.BranchOp):
        return ((op.successor, tuple(op.arguments)),)
    if isinstance(op, cf.ConditionalBranchOp):
        return ((op.then_block, tuple(op.then_arguments)), (op.else_block, tuple(op.else_arguments)))
    return ()


def _check_execution(program: ParsedProgram) -> dict[Block, tuple[Operation, ...]]:
    """Validate CFG/SSA and state flow independently of compiler pass decisions."""
    blocks = {block: tuple(block.ops) for block in program.blocks}
    boundaries = _boundary_ranges(program)
    constants: dict[SSAValue, int | None] = {}
    entry = program.blocks[0]
    starts = [op for op in program.operations if operation_name(op) == "atlas.virtual_start"]
    _require(len(starts) == 1 and bool(blocks[entry]) and blocks[entry][0] is starts[0], "virtual_start must be the unique first entry operation")
    supported = {
        "atlas.virtual_start", "atlas.virtual_input_bf16", "atlas.virtual_input_fp8", "atlas.virtual_output_bf16",
        "atlas.virtual_vpu_unary", "atlas.virtual_vpu_binary", "atlas.virtual_pack_fp8",
        "atlas.virtual_scale_constant", "atlas.virtual_mxu_matmul", "atlas.virtual_mxu_load_weight",
        "atlas.virtual_mxu_load_acc_bf16", "atlas.virtual_mxu_load_acc_fp8", "atlas.virtual_mxu_reset",
        "atlas.virtual_mxu_accumulate", "atlas.virtual_mxu_readout_bf16", "atlas.virtual_mxu_readout_fp8",
        "atlas.virtual_dma_load_fp8", "atlas.virtual_dma_load_bf16", "atlas.virtual_dma_await_fp8",
        "atlas.virtual_dma_await_bf16", "atlas.virtual_dma_store_fp8", "atlas.virtual_dma_store_bf16", "atlas.virtual_dma_wait",
        "arith.constant", "arith.addi", "arith.cmpi", "cf.br", "cf.cond_br", "func.return",
    }
    definitions = {}
    for block, operations in blocks.items():
        definitions.update((arg, (block, -1)) for arg in block.args)
        current = None if block is entry else block.args[0]
        if program.function is not None:
            _require(bool(operations) and isinstance(operations[-1], (cf.BranchOp, cf.ConditionalBranchOp, func.ReturnOp)), "function block requires a branch or return terminator")
        for position, op in enumerate(operations):
            name = operation_name(op)
            if name not in supported:
                raise UnsupportedVirtualMode(f"execution is not implemented for {name}")
            if isinstance(op, (cf.BranchOp, cf.ConditionalBranchOp, func.ReturnOp)):
                _require(position == len(operations) - 1, f"{name}: terminator must be last in its block")
            for target, arguments in _edges(op):
                _require(target in blocks and target is not entry, f"{name}: successor must be a non-entry block in the selected function")
                _require(len(arguments) == len(target.args), f"{name}: successor argument arity mismatch")
                _require(all(value.type == arg.type for value, arg in zip(arguments, target.args)), f"{name}: successor argument type mismatch")
            for operand in op.operands:
                if _type_kind(operand.type)[0] == "state":
                    _require(current is not None and operand is current, f"{name}: expected current state token")
            for result in op.results:
                definitions[result] = (block, position)
                if _type_kind(result.type)[0] == "state":
                    current = result

    reachable: set[Block] = set()
    pending = [entry]
    while pending:
        block = pending.pop()
        if block not in reachable:
            reachable.add(block)
            pending.extend(target for op in blocks[block] for target, _ in _edges(op))
    _require(reachable == set(blocks), "virtual CFG contains an unreachable block")
    region = program.function.body if program.function is not None else program.module.body
    dominance = DominanceInfo(region)
    for block, operations in blocks.items():
        for position, op in enumerate(operations):
            for operand in op.operands:
                _require(operand in definitions, f"{operation_name(op)}: operand has no SSA definition in the selected program")
                owner, defined_at = definitions[operand]
                available = defined_at < position if owner is block else dominance.dominates(owner, block)
                _require(available, f"{operation_name(op)}: operand definition does not dominate its use")
                if _type_kind(operand.type)[0] in ("mxu_weight", "mxu_acc", "dma_load_fp8", "dma_load_bf16", "dma_store"):
                    _require(owner is block, "MXU/DMA handles cannot cross block boundaries")
        _check_mxu_handles(operations)
        _check_dma_handles(operations, boundaries, constants)
    return blocks


def evaluate(program: ParsedProgram, inputs: RuntimeInputs, *, max_steps: int = 10000) -> EvaluationResult:
    """Execute admitted virtual operations and completion-visible memory effects."""
    _require(type(max_steps) is int and max_steps > 0, "max_steps must be a positive integer")
    program.validate_inputs(inputs)
    blocks = _check_execution(program)
    for address, size, kind in _boundary_ranges(program):
        if kind == "output" and any(_overlaps(address, size, region.address, len(region.data)) for region in inputs.memory):
            raise UnsupportedVirtualMode("memory snapshots overlapping implicit output buffers require shared boundary-memory semantics")
        if kind == "input":
            index = (address - program.function.attributes["atlas.input_dram_base"].value.data) // 2048
            payload = _tile_bytes(inputs.tiles[index])
            for region in inputs.memory:
                first, last = max(address, region.address), min(address + size, region.address + len(region.data))
                if first < last:
                    actual = region.data[first - region.address:last - region.address]
                    expected = payload[first - address:last - address]
                    _require(actual == expected, f"boundary input {index} and initial memory disagree in span 0x{first:08x}..0x{last:08x}")
    memory = _Memory(inputs.memory)
    values: dict[SSAValue, object] = {}
    outputs: dict[int, Tile] = {}
    state: object | None = None
    steps = 0

    def value(operand: SSAValue) -> object:
        _require(operand in values, "operand has no executed SSA definition")
        return values[operand]

    def scalar(operand: SSAValue) -> Scalar:
        result = value(operand)
        _require(isinstance(result, Scalar), "expected a scalar runtime value")
        assert isinstance(result, Scalar)
        return result

    def tile(operand: SSAValue) -> Tile:
        result = value(operand)
        _require(isinstance(result, Tile), "expected a tile runtime value")
        assert isinstance(result, Tile)
        return result

    block = program.blocks[0]
    bindings: tuple[object, ...] = inputs.controls
    while True:
        values.update(zip(block.args, bindings))
        for op in blocks[block]:
            for result in op.results:
                values.pop(result, None)
        for position, op in enumerate(blocks[block]):
            name = operation_name(op)
            _require(steps < max_steps, f"step budget {max_steps} exhausted at block {program.blocks.index(block)}, operation {position} ({name})")
            steps += 1
            if isinstance(op, arith.ConstantOp):
                width = op.result.type.width.data
                values[op.result] = Scalar(width, op.value.value.data & ((1 << width) - 1))
            elif isinstance(op, arith.AddiOp):
                values[op.result] = Scalar(32, (scalar(op.lhs).bits + scalar(op.rhs).bits) & 0xFFFFFFFF)
            elif isinstance(op, arith.CmpiOp):
                left, right = scalar(op.lhs).bits, scalar(op.rhs).bits
                signed_left = left - (1 << 32) if left & (1 << 31) else left
                signed_right = right - (1 << 32) if right & (1 << 31) else right
                predicates = (
                    left == right, left != right, signed_left < signed_right, signed_left <= signed_right,
                    signed_left > signed_right, signed_left >= signed_right, left < right, left <= right, left > right, left >= right,
                )
                values[op.result] = Scalar(1, int(predicates[op.predicate.value.data]))
            elif isinstance(op, (cf.BranchOp, cf.ConditionalBranchOp)):
                edge = 0 if isinstance(op, cf.BranchOp) or scalar(op.cond).bits else 1
                target, arguments = _edges(op)[edge]
                # Snapshot every source before rebinding a backedge's destinations.
                bindings = tuple(value(argument) for argument in arguments)
                _require(bindings[0] is state, f"{name}: expected current dynamic state token")
                block = target
                break
            elif name == "atlas.virtual_start":
                _require(state is None, "virtual_start cannot execute twice")
                state = values[op.results[0]] = object()
            elif name in ("atlas.virtual_vpu_unary", "atlas.virtual_vpu_binary", "atlas.virtual_pack_fp8"):
                values[op.results[0]] = evaluate_tile_operation(op, tuple(tile(operand) for operand in op.operands))
            elif name == "atlas.virtual_scale_constant":
                values[op.results[0]] = _integer_attribute(op, "code")
            elif name == "atlas.virtual_mxu_matmul":
                operands = tuple(tile(operand) for operand in op.operands) + (Tile("bf16", (0,) * 1024),)
                values[op.results[0]] = _mxu_tile("matmul", operands, unit=_mxu_unit(op))
            else:
                _require(state is not None and value(op.operands[0]) is state, f"{name}: expected current dynamic state token")
                if isinstance(op, func.ReturnOp):
                    return EvaluationResult(outputs, memory.snapshots())
                if name in ("atlas.virtual_input_bf16", "atlas.virtual_input_fp8"):
                    values[op.results[1]] = inputs.tiles[_integer_attribute(op, "index")]
                elif name == "atlas.virtual_output_bf16":
                    outputs[_integer_attribute(op, "index")] = tile(op.operands[1])
                elif name in ("atlas.virtual_mxu_load_weight", "atlas.virtual_mxu_load_acc_bf16", "atlas.virtual_mxu_readout_bf16"):
                    values[op.results[1]] = tile(op.operands[1])
                elif name == "atlas.virtual_mxu_load_acc_fp8":
                    values[op.results[1]] = _mxu_tile("seed_fp8", (tile(op.operands[1]),))
                elif name in ("atlas.virtual_mxu_reset", "atlas.virtual_mxu_accumulate"):
                    accumulator = tile(op.operands[3]) if name.endswith("accumulate") else Tile("bf16", (0,) * 1024)
                    values[op.results[1]] = _mxu_tile("matmul", (tile(op.operands[1]), tile(op.operands[2]), accumulator), unit=_mxu_unit(op))
                elif name == "atlas.virtual_mxu_readout_fp8":
                    scale = value(op.operands[2])
                    _require(type(scale) is int and 0 <= scale <= 255, "MXU readout requires a raw constant scale code")
                    values[op.results[1]] = _mxu_tile("readout_fp8", (tile(op.operands[1]),), scale)
                elif name.startswith(("atlas.virtual_dma_load_", "atlas.virtual_dma_store_")):
                    address, length = (scalar(operand).bits for operand in op.operands[-2:])
                    format: TileFormat = "fp8" if name.endswith("fp8") else "bf16"
                    store = name.startswith("atlas.virtual_dma_store_")
                    memory.chunks(address, length)
                    data = _tile_bytes(tile(op.operands[1])) if store else memory.read(address, length)
                    values[op.results[1]] = _Transfer(address, data, format)
                elif name.startswith("atlas.virtual_dma_await_") or name == "atlas.virtual_dma_wait":
                    transfer = value(op.operands[1])
                    _require(isinstance(transfer, _Transfer), "expected an executed DMA transfer")
                    assert isinstance(transfer, _Transfer)
                    if name == "atlas.virtual_dma_wait":
                        memory.write(transfer.address, transfer.data)
                    else:
                        values[op.results[1]] = _tile_from_bytes(transfer.data, transfer.format)
                    del values[op.operands[1]]
                state = values[op.results[0]] = object()
        else:
            return EvaluationResult(outputs, memory.snapshots())
