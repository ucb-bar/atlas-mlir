"""Atlas virtual IR parsing and immutable runtime data; operation execution is not implemented."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

try:
    from xdsl.context import Context
    from xdsl.dialects import arith, builtin, cf, func
    from xdsl.ir import Attribute, Block, Operation
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


def _integer_attribute(op: Operation, name: str) -> int:
    value = op.attributes.get(name)
    _require(isinstance(value, builtin.IntegerAttr) and value.type == builtin.i32, f"{operation_name(op)} requires {name} : i32")
    assert isinstance(value, builtin.IntegerAttr)
    return value.value.data


def _check_atlas_operation(op: Operation) -> None:
    name = operation_name(op)
    short = name.removeprefix("atlas.virtual_")
    if not name.startswith("atlas.virtual_") or short not in _SIGNATURES:
        raise UnsupportedVirtualMode(f"unsupported virtual operation: {name}")
    _require(not op.regions and not op.successors and not op.properties, f"{name}: regions, successors, and properties are unsupported")
    operands, results, attributes = _SIGNATURES[short]
    types = [_type_kind(value.type) for value in (*op.operands, *op.results)]
    _require(
        tuple(kind for kind, _ in types[:len(op.operands)]) == operands and tuple(kind for kind, _ in types[len(op.operands):]) == results,
        f"{name}: incorrect operand/result signature",
    )
    _require(set(op.attributes) - {"op_name__"} == set(attributes), f"{name}: expected attributes {attributes}")
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
        kind = op.attributes["kind"]
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
