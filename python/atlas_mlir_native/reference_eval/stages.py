"""Compose independently authored numerical adapters over provider buffer edges.

Only selected route/placement and buffer identifiers are consumed. Route contracts
are numerical, not emitted instruction sequences: source_layout preserves values,
source_float_arithmetic rounds each source add to its result dtype, exact-float
routes reproduce source RNE, and already-quantized readout uses MXU1/K32 with a
BF16 readout before the source output cast. These are the contracts declared in
Merlin's source_layout/source_float_arithmetic route providers and atlas-mlir's
already_quantized.py and exact_float_application.py. No stage arithmetic is read
from compiled applications. Unsupported source decompositions fail closed.
"""
from __future__ import annotations

import math
import re


CONTRACTS = {
    "atlas.qlinear.default": ("add", ("readout",), ("operand", 4)),
    "atlas.qconv2d.default": ("add", ("readout",), ("operand", 4)),
    "aten.add.Tensor": ("add", ("operand", 0), ("operand", 1)),
}


def composite_silu_route(target, routes):
    """Recognize the declared exact-cast plus pointwise SiLU composition.

    Temporary fixed route pair. Replace with authenticated numerical stage
    contracts for unary compositions, using the same graph validation as build.
    """
    return (target == "aten.silu.default" and
            routes == ["source_exact_float_device", "source_pointwise_math"])


_ROUTES = {
    "source_layout": ("host", "identity", "value-preserving-layout", "layout"),
    "source_already_quantized": ("accelerator", "readout", "mxu1-k32-bf16-readout-source-cast", "already_quantized"),
    "source_float_arithmetic": ("host", "add", "source-add-rne-at-result-dtype", "float_arithmetic"),
    "source_exact_float_device": ("accelerator", "add", "source-add-rne-at-result-dtype", "exact_float_device"),
}


def _fail(reason):
    raise ValueError("invalid numerical stage graph: " + reason)


def build(rows, binding, target):
    """Validate topology and prove the supported source expression is complete.

    Source SSA operands/results come from authenticated frontend MLIR, not from
    stage names. Symbolic expressions propagate through arbitrary identity stages
    and must equal the independent source adapter's expression at the result.
    This deliberately supports only qlinear or qconv2d with post-cast bias, and source add.
    """
    in_place_group = bool(binding) and "ops" in binding and binding.get("target") is None and _base_target(target) is not None
    if not binding or (target != binding.get("target") and not in_place_group):
        _fail("missing or conflicting source operand/result bindings")
    if "ops" in binding:
        if target not in _GROUPS:
            _fail("missing or conflicting source operand/result bindings")
        return _GROUPS[target](rows, binding, target)
    operands, results = binding["operands"], binding["results"]
    if (not operands or any(not isinstance(x, str) for x in operands)
            or len(set(operands)) != len(operands) or len(results) != 1):
        _fail("unsupported source operand/result bindings")
    if target not in CONTRACTS:
        _fail("unsupported source composition " + str(target))
    expected = CONTRACTS[target]
    if target in {"atlas.qlinear.default", "atlas.qconv2d.default"}:
        if len(operands) != 5 or binding.get("bias_order") != "after_output_cast":
            _fail("source contraction requires declared post-cast bias")
    elif target == "aten.add.Tensor":
        if len(operands) != 2:
            _fail("source add requires two operands")
    producers, ids = {}, set()
    for i, row in enumerate(rows):
        if row["id"] in ids:
            _fail("duplicate stage identifier")
        ids.add(row["id"])
        route = _ROUTES.get(row["implementation"])
        if route is None or route[0] != row["placement"]:
            _fail("undeclared route or placement")
        if "kind" in row and row["kind"] != route[3]:
            _fail("stage kind differs from declared adapter")
        if len(row.get("buffers", [])) != 1:
            _fail("each supported adapter requires one result")
        output = row["buffers"][0]
        if output in producers or output in operands:
            _fail("duplicate producer or overwritten source operand")
        producers[output] = i
    # Detect cycles before enforcing provider order, so diagnostics identify them.
    active, complete = set(), set()
    def visit(i):
        if i in active:
            _fail("cycle")
        if i in complete:
            return
        active.add(i)
        for value in rows[i].get("inputs", []):
            if value in producers:
                visit(producers[value])
            elif value not in operands:
                _fail("missing stage or unbound source operand " + value)
        active.remove(i)
        complete.add(i)
    for i in range(len(rows)):
        visit(i)
    if results[0] not in producers:
        _fail("missing source result")
    needed = set()
    def require(i):
        if i in needed:
            return
        needed.add(i)
        for value in rows[i].get("inputs", []):
            if value in producers:
                require(producers[value])
    require(producers[results[0]])
    if len(needed) != len(rows):
        _fail("incomplete or disconnected stages")
    expressions = {value: ("operand", i) for i, value in enumerate(operands)}
    stages = []
    for i, row in enumerate(rows):
        if any(producers.get(value, -1) >= i for value in row.get("inputs", [])):
            _fail("edge violates declared stage order")
        inputs = [expressions[x] for x in row.get("inputs", [])]
        _, adapter, rounding, _ = _ROUTES[row["implementation"]]
        if adapter == "identity" and len(inputs) == 1:
            expression = inputs[0]
        elif adapter == "readout" and target in {"atlas.qlinear.default", "atlas.qconv2d.default"} and len(inputs) == 4:
            if set(inputs) != {("operand", j) for j in range(4)}:
                _fail("contraction source operand bindings differ")
            expression = ("readout",)
        elif adapter == "add" and len(inputs) == 2:
            expression = ("add", *inputs)
        else:
            _fail("unsupported adapter/source composition")
        expressions[row["buffers"][0]] = expression
        stages.append({**row, "adapter": adapter, "rounding": rounding})
    if expressions[results[0]] != expected:
        _fail("stage expression differs from source operation or rounding order")
    return {"stages": stages, "binding": binding, "target": target}


_VIEWS = {"merlin.dense_buffer_view", "source_layout", "source_layout_device"}


def _f32_shape(type_):
    if not isinstance(type_, str) or not re.fullmatch(r"tensor<(?:[0-9]+x)*f32>", type_):
        _fail("source mean requires static f32 tensors")
    return tuple(int(x) for x in type_[7:-4].split("x") if x)


def _mean(rows, binding, target):
    """A source mean lowered as a seeded sum over the reduced axes, then division by their element count.

    The prepared group must be exactly: a +0.0 seed, one add-only reduction of the external operand, one identity-map
    division of that sum by a single f32 literal, and shape-only views of the quotient. The sum stage runs on its
    selected device reduction policy and the division on the host in f32. The literal must equal the reduced element count; that is
    checked against the source shapes when the node is evaluated.
    """
    if any(len(op["results"]) != 1 for op in binding["ops"]):
        _fail("each lowered operation of a mean must have one result")
    ops = {result: op for op in binding["ops"] for result in op["results"]}
    if len(ops) != len(binding["ops"]):
        _fail("duplicate lowered source result")
    external = [x for op in binding["ops"] for x in op["operands"] if x not in ops]

    splats = binding.get("splats", {})

    def splat_of(value):
        """The [f32 literal, tensor type] a value splats, from this group or an equal shared constant."""
        return splats.get(value)

    names = sorted(op["name"] for op in binding["ops"])
    reduces = [op for op in binding["ops"] if op["name"] == "linalg.reduce"]
    divides = [op for op in binding["ops"] if op["name"] == "linalg.generic"]
    allowed = {"arith.constant", "tensor.splat", "tensor.empty", "linalg.reduce", "linalg.generic",
               "tensor.collapse_shape", "tensor.expand_shape"}
    if set(names) - allowed or len(reduces) != 1 or len(divides) != 1 or len(set(external)) != 1:
        _fail("unsupported source mean lowering")
    reduce, divide = reduces[0], divides[0]
    if len(reduce["operands"]) != 2 or len(divide["operands"]) != 3:
        _fail("unsupported source mean operands")
    seed = splat_of(reduce["operands"][1]) if len(reduce["operands"]) == 2 else None
    if (reduce["operands"][0] != external[0] or seed is None or seed[0] != 0.0 or math.copysign(1.0, seed[0]) < 0
            or reduce.get("body") != [["arith.addf", 0, 1], ["linalg.yield", "body0r0"]]
            or reduce.get("body_types") != ["f32", "f32", "f32"]):
        _fail("source mean sum must add the operand into a +0.0 seed")
    divisor = splat_of(divide["operands"][1]) if len(divide["operands"]) == 3 else None
    if (divide["operands"][0] != reduce["results"][0] or divisor is None or not divide.get("identity_maps")
            or not divide.get("parallel_iterators")
            or ops.get(divide["operands"][2], {}).get("name") != "tensor.empty"
            or divide.get("body") != [["arith.divf", 0, 1], ["linalg.yield", "body0r0"]]
            or divide.get("body_types") != ["f32", "f32", "f32", "f32"]):
        _fail("source mean must divide its sum by one f32 literal")
    # Every source operation must contribute to the single result, in source order.
    needed, active = set(), set()
    def require(value):
        if value not in ops:
            return
        if value in active:
            _fail("cycle in source mean lowering")
        if value in needed:
            return
        active.add(value)
        for operand in ops[value]["operands"]:
            require(operand)
        active.remove(value)
        needed.add(value)
    require(binding["ops"][-1]["results"][0])
    if needed != set(ops):
        _fail("incomplete or disconnected source mean lowering")
    available = set(external)
    for op in binding["ops"]:
        if not set(op["operands"]) <= available:
            _fail("source mean edge violates source order")
        available.update(op["results"])
    produced = {row["buffers"][0]: row for row in rows if len(row.get("buffers", [])) == 1}
    if len(produced) != len(rows):
        _fail("each supported adapter requires one result")
    if len({row["id"] for row in rows}) != len(rows):
        _fail("duplicate stage identifier")
    route_kinds = {("source_reduction", "accelerator"): "reduction",
                   ("source_float_arithmetic", "host"): "float_arithmetic",
                   ("merlin.dense_buffer_view", "host"): "dense_buffer_view",
                   ("source_layout", "host"): "layout", ("source_layout_device", "accelerator"): "layout"}
    for row in rows:
        kind = route_kinds.get((row["implementation"], row["placement"]))
        if kind is None:
            _fail("undeclared route or placement")
        if "kind" in row and row["kind"] != kind:
            _fail("stage kind differs from declared adapter")
    sum_row, quotient_row = produced.get(reduce["results"][0]), produced.get(divide["results"][0])

    def same_inputs(row, source):
        # Providers sort their inputs and may share one equal constant across the graph.
        if row is None or len(row["inputs"]) != len(source):
            return False
        pending = list(source)
        for value in row["inputs"]:
            match = value if value in pending else next((x for x in pending if splat_of(x) is not None
                                                         and splat_of(x) == splat_of(value)), None)
            if match is None:
                return False
            pending.remove(match)
        return True

    if (not same_inputs(sum_row, reduce["operands"]) or not same_inputs(quotient_row, divide["operands"][:2])
            or (sum_row["implementation"], sum_row["placement"]) not in {("source_reduction", "accelerator")}
            or (quotient_row["implementation"], quotient_row["placement"]) != ("source_float_arithmetic", "host")):
        _fail("source mean stages must be one selected reduction and one host division")
    if rows[:2] != [sum_row, quotient_row]:
        _fail("edge violates declared stage order")
    value, stages = divide["results"][0], [{**sum_row, "adapter": "sum"}, {**quotient_row, "adapter": "divide"}]
    for row in rows:
        if row is sum_row or row is quotient_row:
            continue
        if row["implementation"] not in _VIEWS or row["inputs"] != [value] \
                or ops.get(row["buffers"][0], {}).get("name") not in {"tensor.collapse_shape", "tensor.expand_shape"} \
                or ops[row["buffers"][0]]["operands"] != [value]:
            _fail("only shape views may follow a source mean's division")
        value = row["buffers"][0]
        stages.append({**row, "adapter": "identity"})
    if value != binding["ops"][-1]["results"][0]:
        _fail("missing source result")
    input_shape = _f32_shape(reduce["operand_types"][0])
    dimensions = reduce.get("dimensions", [])
    if (not dimensions or any(type(axis) is not int or not 0 <= axis < len(input_shape) for axis in dimensions)
            or dimensions != sorted(set(dimensions))):
        _fail("unsupported source mean dimensions")
    reduced_shape = tuple(n for i, n in enumerate(input_shape) if i not in dimensions)
    for op in (reduce, divide):
        if _f32_shape(op["result_types"][0]) != reduced_shape:
            _fail("source mean arithmetic result shape differs from its reduction")
    result_shape = _f32_shape(binding["ops"][-1]["result_types"][0])
    if math.prod(result_shape) != math.prod(reduced_shape):
        _fail("source mean views change the element count")
    return {"stages": stages, "binding": binding, "target": target, "dimensions": reduce["dimensions"],
            "divisor": divisor[0], "input_shape": input_shape, "result_shape": result_shape}


def _floating_tensor(type_):
    match = re.fullmatch(r"tensor<((?:[0-9]+x)*)(bf16|f32)>", type_)
    if match is None:
        _fail("source promoted multiply requires static floating tensors")
    return tuple(int(x) for x in match[1].split("x") if x), match[2]


def _broadcast_map(map_, shape, output, *, identity=False):
    """Prove the source map is ordinary right-aligned broadcasting, without compiling it."""
    match = re.fullmatch(r"affine_map<\((.*?)\) -> \((.*?)\)>", map_)
    if match is None or len(shape) > len(output):
        return False
    domain = [x.strip() for x in match[1].split(",") if x.strip()]
    indices = [x.strip() for x in match[2].split(",") if x.strip()]
    if domain != [f"d{i}" for i in range(len(output))] or len(indices) != len(shape):
        return False
    for axis, (extent, index) in enumerate(zip(shape, indices), len(output) - len(shape)):
        allowed = {f"d{axis}"} if extent == output[axis] else set()
        if extent == 1 and not identity:
            allowed.add("0")
        if index not in allowed:
            return False
    return True


# Promoted binary arithmetic: the source widens its one BF16 operand exactly, then adds or multiplies in F32.
_PROMOTED = {"aten.mul.Tensor": ("arith.mulf", "multiply"), "aten.add.Tensor": ("arith.addf", "add")}
# torch's in-place form computes in the promoted dtype, then rounds once into its destination's dtype.
_COPY = "aten._to_copy.default"


def _base_target(target):
    """aten.NAME_.OVERLOAD's out-of-place form, or None for an op that is not in-place."""
    parts = target.split(".") if isinstance(target, str) else []
    if len(parts) == 3 and parts[0] == "aten" and parts[1].endswith("_") and not parts[1].startswith("_"):
        return f"aten.{parts[1][:-1]}.{parts[2]}"
    return None


def _promoted_source(ops, target):
    """Prove tensor.empty then one strict generic that widens its BF16 operand once and applies target in F32."""
    arithmetic, _ = _PROMOTED[target]
    if len(ops) != 2 or [op["name"] for op in ops] != ["tensor.empty", "linalg.generic"]:
        _fail("unsupported source promoted arithmetic lowering")
    empty, multiply = ops
    if (empty["operands"] or len(empty["results"]) != 1 or len(multiply["operands"]) != 3
            or len(multiply["results"]) != 1 or multiply["operands"][2:] != empty["results"]
            or empty["results"] == multiply["results"] or multiply.get("operand_segments") != [2, 1]):
        _fail("unsupported source promoted arithmetic bindings")
    operands = multiply["operands"][:2]
    types = [_floating_tensor(x) for x in multiply["operand_types"]]
    result_shape = _f32_shape(multiply["result_types"][0])
    if (len(types) != 3 or sorted(dtype for _, dtype in types[:2]) != ["bf16", "f32"]
            or types[2] != (result_shape, "f32") or empty["result_types"] != multiply["result_types"]
            or len(set(operands + empty["results"] + multiply["results"])) != 4):
        _fail("source promoted arithmetic requires one BF16 and one F32 operand")
    rank = max(len(shape) for shape, _ in types[:2])
    aligned = [(1,) * (rank - len(shape)) + shape for shape, _ in types[:2]]
    broadcast = []
    for left, right in zip(*aligned):
        if left == right or right == 1:
            broadcast.append(left)
        elif left == 1:
            broadcast.append(right)
        else:
            _fail("source promoted arithmetic operands do not broadcast")
    if tuple(broadcast) != result_shape:
        _fail("source promoted arithmetic result differs from source broadcasting")
    promoted = next(i for i, (_, dtype) in enumerate(types[:2]) if dtype == "bf16")
    inputs = ["body0r0" if i == promoted else i for i in range(2)]
    if (multiply.get("body") != [["arith.extf", promoted], [arithmetic, *inputs], ["linalg.yield", "body1r0"]]
            or multiply.get("body_types") != [*(dtype for _, dtype in types), "f32", "f32"]
            or not multiply.get("strict_float")):
        _fail("source promoted arithmetic must widen once then apply its operation in F32")
    maps = multiply.get("indexing_maps", [])
    if (len(maps) != 3 or not multiply.get("parallel_iterators")
            or multiply.get("iterator_count") != len(result_shape)
            or not all(_broadcast_map(map_, shape, result_shape, identity=i == 2)
                       for i, (map_, (shape, _)) in enumerate(zip(maps, types)))):
        _fail("source promoted arithmetic maps differ from source broadcasting")
    return operands, types[:2], promoted, result_shape, multiply["results"][0]


def _stage(row, route, placement, kind):
    if len(row.get("buffers", [])) != 1:
        _fail("each supported adapter requires one result")
    if (row["implementation"], row["placement"]) != (route, placement):
        _fail("undeclared route or placement")
    if "kind" in row and row["kind"] != kind:
        _fail("stage kind differs from declared adapter")
    return row["buffers"][0]


def _source_result(buffer, source):
    # _join_rows authenticates suffixed provider outputs by their frontend result prefix.
    if buffer != source and not buffer.startswith((source + "_", source + ".")):
        _fail("missing source result")


def _promoted_mul(rows, binding, target):
    """Exact BF16 widening followed by a host F32 add or multiply, including source broadcasting."""
    operands, types, promoted, _, source_result = _promoted_source(binding["ops"], target)
    if len(rows) != 2:
        _fail("source promoted arithmetic requires exactly two single-result stages")
    cast, product = rows
    intermediate = _stage(cast, "source_exact_float_device", "accelerator", "exact_float_device")
    result = _stage(product, "source_float_arithmetic", "host", "float_arithmetic")
    if (cast["id"] == product["id"] or len(set(operands + [intermediate, result])) != 4
            or cast.get("inputs") != [operands[promoted]]
            or sorted(product.get("inputs", [])) != sorted([intermediate, operands[1 - promoted]])):
        _fail("selected promoted arithmetic edges differ from source operands")
    _source_result(result, source_result)
    return {"target": target, "binding": binding, "stages": [{**cast, "adapter": "widen"},
              {**product, "adapter": _PROMOTED[target][1]}], "operands": operands, "types": types, "promoted": promoted}


def _promoted_in_place(rows, binding, target):
    """In-place promoted arithmetic on a BF16 destination: widen, F32 arithmetic, then one RNE narrowing into it."""
    base = _base_target(target)
    ops = binding["ops"]
    if base not in _PROMOTED or len(ops) != 4 or [op["target"] for op in ops] != [base, base, _COPY, _COPY]:
        _fail("unsupported source in-place promoted arithmetic lowering")
    operands, types, promoted, result_shape, wide_result = _promoted_source(ops[:2], base)
    if promoted != 0 or types[0] != (result_shape, "bf16"):
        _fail("in-place promoted arithmetic must widen its own unbroadcast BF16 destination")
    empty, narrow = ops[2:]
    destination_type = ops[0]["result_types"][0][:-len("xf32>")] + "xbf16>"
    if ([op["name"] for op in ops[2:]] != ["tensor.empty", "linalg.generic"] or empty["operands"]
            or len(empty["results"]) != 1 or empty["result_types"] != [destination_type]
            or narrow["operands"] != [wide_result, empty["results"][0]] or len(narrow["results"]) != 1
            or narrow["result_types"] != [destination_type] or narrow.get("operand_segments") != [1, 1]
            or narrow.get("body") != [["arith.truncf", 0], ["linalg.yield", "body0r0"]]
            or narrow.get("body_types") != ["f32", "bf16", "bf16"] or not narrow.get("strict_float")
            or not narrow.get("identity_maps") or not narrow.get("parallel_iterators")
            or narrow.get("iterator_count") != len(result_shape)
            or len({wide_result, *operands, *empty["results"], *narrow["results"]}) != 5):
        _fail("source in-place result must round once from F32 into its BF16 destination")
    if len(rows) != 3:
        _fail("source in-place promoted arithmetic requires exactly three single-result stages")
    cast, arithmetic, rounding = rows
    widened = _stage(cast, "source_exact_float_device", "accelerator", "exact_float_device")
    computed = _stage(arithmetic, "source_float_arithmetic", "host", "float_arithmetic")
    rounded = _stage(rounding, "source_exact_float_device", "accelerator", "exact_float_device")
    if (len({cast["id"], arithmetic["id"], rounding["id"]}) != 3
            or len(set(operands + [widened, computed, rounded])) != 5
            or cast.get("inputs") != [operands[0]]
            or sorted(arithmetic.get("inputs", [])) != sorted([widened, operands[1]])
            or rounding.get("inputs") != [computed]):
        _fail("selected in-place promoted arithmetic edges differ from source operands")
    _source_result(computed, wide_result)
    _source_result(rounded, narrow["results"][0])
    return {"target": target, "binding": binding, "stages": [{**cast, "adapter": "widen"},
              {**arithmetic, "adapter": _PROMOTED[base][1]}, {**rounding, "adapter": "narrow"}],
            "operands": operands, "types": types, "promoted": 0, "in_place": True}


_GROUPS = {"aten.mean.dim": _mean, "aten.mul.Tensor": _promoted_mul, "aten.add.Tensor": _promoted_mul,
           "aten.mul_.Tensor": _promoted_in_place, "aten.add_.Tensor": _promoted_in_place}


def _call_promoted_mul(graph, args, kwargs, rules):
    from . import host, placed_device
    from .evaluator import Tensor, _convert
    if len(args) != 2 or kwargs:
        _fail("source promoted arithmetic signature is unsupported")
    for value, (shape, dtype) in zip(args, graph["types"]):
        if not isinstance(value, Tensor) or value.dtype != dtype or value.raw.shape != tuple(shape):
            _fail("source promoted arithmetic operand shape or dtype differs")
    cast, arithmetic = graph["stages"][:2]
    stage_rules = {k: v for k, v in rules.items() if k != "numerical_stages"}
    # The declared bf16-to-f32-exact-v1 cast is a bit-preserving shift, with no rounding.
    placed_device._rtl({**stage_rules, "execution": "device", "selected_routes": [cast["implementation"]]})
    inputs = list(args)
    inputs[graph["promoted"]] = _convert(inputs[graph["promoted"]], "f32")
    base = _base_target(graph["target"]) or graph["target"]
    result = host.call(base, inputs, {}, {**stage_rules, "execution": "host",
                                          "selected_routes": [arithmetic["implementation"]]})
    if graph.get("in_place"):
        # The declared exact-float narrowing is source RNE from F32 into the BF16 destination.
        placed_device._rtl({**stage_rules, "execution": "device", "selected_routes": [graph["stages"][2]["implementation"]]})
        result = _convert(result, "bf16")
    return result


def _call_mean(graph, args, kwargs, rules):
    from . import host, placed_device
    import numpy as np
    from .evaluator import Tensor, _axis
    if kwargs.get("dtype") is not None or len(args) not in (2, 3) or set(kwargs) - {"dtype"}:
        _fail("source mean signature is unsupported")
    value, dims, keepdim = args[0], args[1], args[2] if len(args) > 2 else False
    if not isinstance(value, Tensor) or value.dtype != "f32" or type(keepdim) is not bool:
        _fail("source mean stages require an f32 operand")
    dims = list(range(value.raw.ndim)) if dims is None or dims == [] or dims == () else dims
    axes = [_axis(axis, value.raw.ndim) for axis in (dims if isinstance(dims, (list, tuple)) else [dims])]
    if len(set(axes)) != len(axes):
        _fail("duplicate source mean dimension")
    axes.sort()
    count = int(np.prod([value.raw.shape[axis] for axis in axes]))
    result_shape = tuple(1 if i in axes else n for i, n in enumerate(value.raw.shape)) if keepdim else tuple(
        n for i, n in enumerate(value.raw.shape) if i not in axes)
    if (value.raw.shape != graph["input_shape"] or result_shape != graph["result_shape"]
            or axes != graph["dimensions"] or float(np.float32(count)) != graph["divisor"] or count != graph["divisor"]):
        _fail("source mean divisor or axes differ from its reduced elements")
    sum_stage, divide_stage = graph["stages"][:2]
    stage_rules = {k: v for k, v in rules.items() if k != "numerical_stages"}
    total = placed_device.call("aten.sum.dim_IntList", [value, axes, True], {},
                               {**stage_rules, "execution": "device", "selected_routes": [sum_stage["implementation"]]})
    quotient = host.call("aten.div.Tensor", [total, Tensor(np.array(count, np.float32), "f32")], {},
                         {**stage_rules, "execution": "host", "selected_routes": [divide_stage["implementation"]]})
    return quotient if keepdim else Tensor(np.squeeze(quotient.raw, tuple(axes)), quotient.dtype)


def call(target, args, kwargs, rules):
    from . import device, host, placed_device
    import numpy as np
    from .evaluator import Tensor
    graph = rules["numerical_stages"]
    if "ops" in graph["binding"]:
        # An in-place source node is evaluated through its out-of-place form; revalidate under the source target.
        source = graph.get("target", target)
        if source != target and _base_target(source) != target:
            _fail("missing or conflicting source operand/result bindings")
        graph = build(graph["stages"], graph["binding"], source)
        return (_call_mean if source == "aten.mean.dim" else _call_promoted_mul)(graph, args, kwargs, rules)
    # Revalidate even when invoked directly, without SelectedDecisions.
    graph = build(graph["stages"], graph["binding"], target)
    convolution = target == "atlas.qconv2d.default"
    if target == "atlas.qlinear.default" or convolution:
        names = ("x_codes", "x_exp", "w_codes", "w_exp", "bias", "out_dtype", "weight_format")
        values = device._arguments(args, kwargs, names + (("batch", "out_h", "out_w") if convolution else ()))
        if values[4] is None:
            _fail("requires a source bias")
        dtype = "bf16" if values[5] == "bfloat16" else "f32"
        bias = device._typed(values[4], dtype)
        if bias.shape != (values[2].raw.shape[0],):
            _fail("bias must match source output feature axis")
        source_values = values[:5]
    else:
        source_values = device._arguments(args, kwargs, ("self", "other", "alpha"), (1,))
        if source_values[2] != 1:
            _fail("source add alpha requires a declared stage")
        source_values = source_values[:2]
    env = dict(zip(graph["binding"]["operands"], source_values, strict=True))
    for stage in graph["stages"]:
        inputs = [env[x] for x in stage["inputs"]]
        stage_rules = {k: v for k, v in rules.items() if k != "numerical_stages"}
        stage_rules.update(execution="device" if stage["placement"] == "accelerator" else "host",
                           selected_routes=[stage["implementation"]])
        if stage["adapter"] == "identity":
            value = inputs[0]
            if not isinstance(value, Tensor):
                _fail("layout requires source tensor")
            output = Tensor(value.raw.copy(), value.dtype)
        elif stage["adapter"] == "readout":
            # The source adapter owns argument roles, including provider-sorted
            # external inputs. Output casting is the explicit rounding boundary.
            # qconv2d's readout is qlinear over its unfolded [N*H*W,K] codes.
            output = device.call("atlas.qlinear.default", [*values[:4], None, *values[5:7]], {}, stage_rules)
        elif stage["placement"] == "accelerator":
            output = placed_device.call("aten.add.Tensor", inputs, {}, stage_rules)
        else:
            output = host.call("aten.add.Tensor", inputs, {}, stage_rules)
        env[stage["buffers"][0]] = output
    result = env[graph["binding"]["results"][0]]
    if convolution:
        # Rows are ordered N, H, W; the source result is NCHW after the bias.
        batch, out_h, out_w = values[7:]
        result = Tensor(np.ascontiguousarray(result.raw.reshape(batch, out_h, out_w, -1).transpose(0, 3, 1, 2)),
                        result.dtype)
    return result
