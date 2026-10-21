"""Compile supported source regions through Atlas's existing native backends."""

from __future__ import annotations

from merlin.compile.layout_c import emit_layout_c
from merlin.compile.semantic_application import CompiledCRegion
from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.softmax_source import recognize_row_softmax
from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.xdsl_dialects.lowering.global_plan import (
    BufferRepresentation,
    CycleInterval,
    GlobalPlan,
    RegionAlternative,
    ValueRepresentation,
)

from .application_precision import (
    CONTRACTION_POLICY,
    SCALED_CONTRACTION_POLICY,
    is_source_relu,
    prepare_contraction,
)
from .binding import TARGET_IDENTITY, AtlasNativeBinding
from .gelu import is_source_gelu
from .row_softmax import SOFTMAX_POLICY
from .softmax_precision import lower_semantic_softmax

_LEGACY_LAYOUTS = ("transpose", "reshape", "copy")
_INDEXED_LAYOUTS = ("gather", "insert_slice", "concat")


def _predicate_c(region, name):
    try:
        from merlin.compile.c_predicates import emit_predicate_c

        return emit_predicate_c(region, name)
    except ImportError as error:
        raise NativeCompilationError(
            "tool_unavailable", "selected CPU predicates need Merlin's predicate C emitter"
        ) from error
    except ValueError as error:
        raise NativeCompilationError("unsupported_semantics", str(error)) from error


def _integer_c(region, name):
    try:
        from merlin.compile.c_integer import emit_integer_c
    except ImportError as error:
        raise NativeCompilationError(
            "tool_unavailable", "selected CPU integer operations need Merlin's integer C emitter"
        ) from error
    try:
        return emit_integer_c(region, name)
    except ValueError as error:
        raise NativeCompilationError("unsupported_semantics", str(error)) from error


def _float_cast_c(region, name):
    try:
        from merlin.compile.c_float_casts import emit_float_cast_c
    except ImportError as error:
        raise NativeCompilationError("tool_unavailable", "selected CPU float casts need Merlin's C emitter") from error
    try:
        return emit_float_cast_c(region, name)
    except ValueError as error:
        raise NativeCompilationError("unsupported_semantics", str(error)) from error


def _float_arithmetic_c(region, name):
    try:
        from merlin.compile.c_float_arithmetic import emit_float_arithmetic_c
    except ImportError as error:
        raise NativeCompilationError("tool_unavailable", "selected CPU arithmetic needs Merlin's C emitter") from error
    try:
        return emit_float_arithmetic_c(region, name)
    except ValueError as error:
        raise NativeCompilationError("unsupported_semantics", str(error)) from error


def _unary_math_c(region, name, policy):
    try:
        from merlin.compile.c_unary_math import emit_unary_math_c
    except ImportError as error:
        raise NativeCompilationError("tool_unavailable", "selected CPU math needs Merlin's C emitter") from error
    try:
        return emit_unary_math_c(region, name, policy)
    except ValueError as error:
        raise NativeCompilationError("unsupported_semantics", str(error)) from error


def checked_host_layouts(operations, *, allow_embedding=False, allow_indexed_update=False, allow_identity=False):
    """Validate explicit CPU layout placement without admitting host arithmetic."""
    if (
        not isinstance(operations, list)
        or not operations
        or any(
            not isinstance(op, str)
            or op
            not in (
                *_LEGACY_LAYOUTS,
                *_INDEXED_LAYOUTS,
                *(("embedding_lookup",) if allow_embedding else ()),
                *(("indexed_insert_slice",) if allow_indexed_update else ()),
                *(("identity",) if allow_identity else ()),
            )
            for op in operations
        )
        or len(set(operations)) != len(operations)
    ):
        raise NativeCompilationError("unsupported_semantics", "host layout operations are unknown or repeated")
    return tuple(operations)


class AtlasApplicationAdapter:
    """Structural first-valid policy, with uncalibrated costs kept unknown.

    Source softmax is fused only after its closed-region recognizer succeeds.
    Remaining contractions and byte-preserving layouts stay separate. There is
    no host arithmetic fallback or model-name/shape-list selection.
    """

    @staticmethod
    def application_binding(name, regions, interface):
        from .application_c_context import application_binding

        return application_binding(name, regions, interface)

    def __init__(self, policy, *, original_source_sha256):
        bf16_pointwise = None
        if policy.get("schema") == "atlas.diagnostic_application_precision.v21":
            from .pointwise import BF16_POINTWISE_POLICY

            if policy.get("bf16_pointwise") != BF16_POINTWISE_POLICY:
                raise NativeCompilationError("unsupported_semantics", "selected BF16 pointwise policy differs")
            bf16_pointwise = BF16_POINTWISE_POLICY
            # v21 adds one explicit device arithmetic choice to the v20 fields.
            # Validate every inherited field through the existing v20 path.
            policy = {key: value for key, value in policy.items() if key != "bf16_pointwise"}
            policy["schema"] = "atlas.diagnostic_application_precision.v20"
        self.operand_scale_exponents = None
        self.epilogues = ()
        self.pointwise = False
        self.bf16_pointwise = bf16_pointwise
        self.reductions = False
        self.pointwise_math = False
        self.sigmoid = False
        self.gelu = False
        self.masked_softmax = False
        self.guarded_masked_softmax = False
        self.host_predicates = ()
        self.host_integer_ops = ()
        self.host_float_casts = ()
        self.host_float_arithmetic = ()
        self.host_unary_math = None
        self.host_layouts = _LEGACY_LAYOUTS
        expected = {
            "schema": "atlas.diagnostic_application_precision.v1",
            "source_sha256": original_source_sha256,
            "target_identity": TARGET_IDENTITY,
            "contraction": CONTRACTION_POLICY,
            "row_softmax": SOFTMAX_POLICY,
            "host_support": ["transpose", "reshape", "copy"],
        }
        if policy.get("schema") in {
            "atlas.diagnostic_application_precision.v2",
            "atlas.diagnostic_application_precision.v3",
            "atlas.diagnostic_application_precision.v4",
            "atlas.diagnostic_application_precision.v5",
            "atlas.diagnostic_application_precision.v6",
            "atlas.diagnostic_application_precision.v7",
            "atlas.diagnostic_application_precision.v8",
            "atlas.diagnostic_application_precision.v9",
            "atlas.diagnostic_application_precision.v10",
            "atlas.diagnostic_application_precision.v11",
            "atlas.diagnostic_application_precision.v12",
            "atlas.diagnostic_application_precision.v13",
            "atlas.diagnostic_application_precision.v14",
            "atlas.diagnostic_application_precision.v15",
            "atlas.diagnostic_application_precision.v16",
            "atlas.diagnostic_application_precision.v17",
            "atlas.diagnostic_application_precision.v18",
            "atlas.diagnostic_application_precision.v19",
            "atlas.diagnostic_application_precision.v20",
        }:
            selected = policy.get("contraction")
            if not isinstance(selected, dict) or set(selected) != {
                "policy",
                "operand_scale_exponents",
            }:
                raise NativeCompilationError("unsupported_semantics", "scaled contraction policy fields differ")
            exponents = selected["operand_scale_exponents"]
            if (
                selected["policy"] != SCALED_CONTRACTION_POLICY
                or not isinstance(exponents, list)
                or len(exponents) != 2
                or any(type(value) is not int or not -32 <= value <= 32 for value in exponents)
            ):
                raise NativeCompilationError("unsupported_semantics", "scaled contraction policy is unsupported")
            self.operand_scale_exponents = tuple(exponents)
            expected.update(schema=policy["schema"], contraction=selected)
        if policy.get("schema") in {
            "atlas.diagnostic_application_precision.v3",
            "atlas.diagnostic_application_precision.v4",
            "atlas.diagnostic_application_precision.v5",
            "atlas.diagnostic_application_precision.v6",
            "atlas.diagnostic_application_precision.v7",
            "atlas.diagnostic_application_precision.v8",
            "atlas.diagnostic_application_precision.v9",
            "atlas.diagnostic_application_precision.v10",
            "atlas.diagnostic_application_precision.v11",
            "atlas.diagnostic_application_precision.v12",
            "atlas.diagnostic_application_precision.v13",
            "atlas.diagnostic_application_precision.v14",
            "atlas.diagnostic_application_precision.v15",
            "atlas.diagnostic_application_precision.v16",
            "atlas.diagnostic_application_precision.v17",
            "atlas.diagnostic_application_precision.v18",
            "atlas.diagnostic_application_precision.v19",
            "atlas.diagnostic_application_precision.v20",
        }:
            if policy.get("epilogues") != ["relu"]:
                raise NativeCompilationError(
                    "unsupported_semantics",
                    "selected contraction epilogues are unsupported",
                )
            self.epilogues = ("relu",)
            expected["epilogues"] = ["relu"]
        if policy.get("schema") in {
            "atlas.diagnostic_application_precision.v4",
            "atlas.diagnostic_application_precision.v5",
            "atlas.diagnostic_application_precision.v6",
            "atlas.diagnostic_application_precision.v7",
            "atlas.diagnostic_application_precision.v8",
            "atlas.diagnostic_application_precision.v9",
            "atlas.diagnostic_application_precision.v10",
            "atlas.diagnostic_application_precision.v11",
            "atlas.diagnostic_application_precision.v12",
            "atlas.diagnostic_application_precision.v13",
            "atlas.diagnostic_application_precision.v14",
            "atlas.diagnostic_application_precision.v15",
            "atlas.diagnostic_application_precision.v16",
            "atlas.diagnostic_application_precision.v17",
            "atlas.diagnostic_application_precision.v18",
            "atlas.diagnostic_application_precision.v19",
            "atlas.diagnostic_application_precision.v20",
        }:
            from .pointwise import POINTWISE_POLICY

            if policy.get("pointwise") != POINTWISE_POLICY:
                raise NativeCompilationError("unsupported_semantics", "selected pointwise arithmetic differs")
            expected["pointwise"] = POINTWISE_POLICY
            self.pointwise = True
        if policy.get("schema") in {
            "atlas.diagnostic_application_precision.v5",
            "atlas.diagnostic_application_precision.v6",
            "atlas.diagnostic_application_precision.v7",
            "atlas.diagnostic_application_precision.v8",
            "atlas.diagnostic_application_precision.v9",
            "atlas.diagnostic_application_precision.v10",
            "atlas.diagnostic_application_precision.v11",
            "atlas.diagnostic_application_precision.v12",
            "atlas.diagnostic_application_precision.v13",
            "atlas.diagnostic_application_precision.v14",
            "atlas.diagnostic_application_precision.v15",
            "atlas.diagnostic_application_precision.v16",
            "atlas.diagnostic_application_precision.v17",
            "atlas.diagnostic_application_precision.v18",
            "atlas.diagnostic_application_precision.v19",
            "atlas.diagnostic_application_precision.v20",
        }:
            self.host_layouts = checked_host_layouts(
                policy.get("host_support"),
                allow_identity=policy.get("schema")
                in {
                    "atlas.diagnostic_application_precision.v17",
                    "atlas.diagnostic_application_precision.v18",
                    "atlas.diagnostic_application_precision.v19",
                    "atlas.diagnostic_application_precision.v20",
                },
                allow_embedding=policy.get("schema")
                in {
                    "atlas.diagnostic_application_precision.v8",
                    "atlas.diagnostic_application_precision.v9",
                    "atlas.diagnostic_application_precision.v10",
                    "atlas.diagnostic_application_precision.v11",
                    "atlas.diagnostic_application_precision.v12",
                    "atlas.diagnostic_application_precision.v13",
                    "atlas.diagnostic_application_precision.v14",
                    "atlas.diagnostic_application_precision.v15",
                    "atlas.diagnostic_application_precision.v16",
                    "atlas.diagnostic_application_precision.v17",
                    "atlas.diagnostic_application_precision.v18",
                    "atlas.diagnostic_application_precision.v19",
                    "atlas.diagnostic_application_precision.v20",
                },
                allow_indexed_update=policy.get("schema")
                in {
                    "atlas.diagnostic_application_precision.v11",
                    "atlas.diagnostic_application_precision.v12",
                    "atlas.diagnostic_application_precision.v13",
                    "atlas.diagnostic_application_precision.v14",
                    "atlas.diagnostic_application_precision.v15",
                    "atlas.diagnostic_application_precision.v16",
                    "atlas.diagnostic_application_precision.v17",
                    "atlas.diagnostic_application_precision.v18",
                    "atlas.diagnostic_application_precision.v19",
                    "atlas.diagnostic_application_precision.v20",
                },
            )
            expected["host_support"] = list(self.host_layouts)
        if policy.get("schema") in {
            "atlas.diagnostic_application_precision.v6",
            "atlas.diagnostic_application_precision.v7",
            "atlas.diagnostic_application_precision.v8",
            "atlas.diagnostic_application_precision.v9",
            "atlas.diagnostic_application_precision.v10",
            "atlas.diagnostic_application_precision.v11",
            "atlas.diagnostic_application_precision.v12",
            "atlas.diagnostic_application_precision.v13",
            "atlas.diagnostic_application_precision.v14",
            "atlas.diagnostic_application_precision.v15",
            "atlas.diagnostic_application_precision.v16",
            "atlas.diagnostic_application_precision.v17",
            "atlas.diagnostic_application_precision.v18",
            "atlas.diagnostic_application_precision.v19",
            "atlas.diagnostic_application_precision.v20",
        }:
            from .reduction import REDUCTION_POLICY

            if policy.get("reduction") != REDUCTION_POLICY:
                raise NativeCompilationError("unsupported_semantics", "selected reduction arithmetic differs")
            expected["reduction"] = REDUCTION_POLICY
            self.reductions = True
        if policy.get("schema") in {
            "atlas.diagnostic_application_precision.v7",
            "atlas.diagnostic_application_precision.v8",
            "atlas.diagnostic_application_precision.v9",
            "atlas.diagnostic_application_precision.v10",
            "atlas.diagnostic_application_precision.v11",
            "atlas.diagnostic_application_precision.v12",
            "atlas.diagnostic_application_precision.v13",
            "atlas.diagnostic_application_precision.v14",
            "atlas.diagnostic_application_precision.v15",
            "atlas.diagnostic_application_precision.v16",
            "atlas.diagnostic_application_precision.v17",
            "atlas.diagnostic_application_precision.v18",
            "atlas.diagnostic_application_precision.v19",
            "atlas.diagnostic_application_precision.v20",
        }:
            from .pointwise_math import MATH_POLICY

            if policy.get("pointwise_math") != MATH_POLICY:
                raise NativeCompilationError("unsupported_semantics", "selected pointwise math policy differs")
            expected["pointwise_math"] = MATH_POLICY
            self.pointwise_math = True
        if policy.get("schema") in {
            "atlas.diagnostic_application_precision.v9",
            "atlas.diagnostic_application_precision.v10",
            "atlas.diagnostic_application_precision.v11",
            "atlas.diagnostic_application_precision.v12",
            "atlas.diagnostic_application_precision.v13",
            "atlas.diagnostic_application_precision.v14",
            "atlas.diagnostic_application_precision.v15",
            "atlas.diagnostic_application_precision.v16",
            "atlas.diagnostic_application_precision.v17",
            "atlas.diagnostic_application_precision.v18",
            "atlas.diagnostic_application_precision.v19",
            "atlas.diagnostic_application_precision.v20",
        }:
            from .sigmoid import SIGMOID_POLICY

            if policy.get("sigmoid") != SIGMOID_POLICY:
                raise NativeCompilationError("unsupported_semantics", "selected sigmoid policy differs")
            expected["sigmoid"] = SIGMOID_POLICY
            self.sigmoid = True
        if policy.get("schema") in {
            "atlas.diagnostic_application_precision.v10",
            "atlas.diagnostic_application_precision.v11",
            "atlas.diagnostic_application_precision.v12",
            "atlas.diagnostic_application_precision.v13",
            "atlas.diagnostic_application_precision.v14",
            "atlas.diagnostic_application_precision.v15",
            "atlas.diagnostic_application_precision.v16",
            "atlas.diagnostic_application_precision.v17",
            "atlas.diagnostic_application_precision.v18",
            "atlas.diagnostic_application_precision.v19",
            "atlas.diagnostic_application_precision.v20",
        }:
            from .row_softmax import MASKED_SOFTMAX_POLICY

            if policy.get("masked_softmax") != MASKED_SOFTMAX_POLICY:
                raise NativeCompilationError("unsupported_semantics", "selected masked softmax policy differs")
            expected["masked_softmax"] = MASKED_SOFTMAX_POLICY
            self.masked_softmax = True
        if policy.get("schema") in {
            "atlas.diagnostic_application_precision.v12",
            "atlas.diagnostic_application_precision.v13",
            "atlas.diagnostic_application_precision.v14",
            "atlas.diagnostic_application_precision.v15",
            "atlas.diagnostic_application_precision.v16",
            "atlas.diagnostic_application_precision.v17",
            "atlas.diagnostic_application_precision.v18",
            "atlas.diagnostic_application_precision.v19",
            "atlas.diagnostic_application_precision.v20",
        }:
            predicates = policy.get("host_predicates")
            admitted = {"f32_nonzero", "mask", "select"}
            if policy.get("schema") in {
                "atlas.diagnostic_application_precision.v14",
                "atlas.diagnostic_application_precision.v15",
                "atlas.diagnostic_application_precision.v16",
                "atlas.diagnostic_application_precision.v17",
                "atlas.diagnostic_application_precision.v18",
                "atlas.diagnostic_application_precision.v19",
                "atlas.diagnostic_application_precision.v20",
            }:
                admitted.add("bucketize")
            if (
                not isinstance(predicates, list)
                or not predicates
                or any(not isinstance(op, str) or op not in admitted for op in predicates)
                or len(set(predicates)) != len(predicates)
            ):
                raise NativeCompilationError("unsupported_semantics", "host predicate selection is unsupported")
            self.host_predicates = tuple(predicates)
            expected["host_predicates"] = predicates
        if policy.get("schema") in {
            "atlas.diagnostic_application_precision.v13",
            "atlas.diagnostic_application_precision.v14",
            "atlas.diagnostic_application_precision.v15",
            "atlas.diagnostic_application_precision.v16",
            "atlas.diagnostic_application_precision.v17",
            "atlas.diagnostic_application_precision.v18",
            "atlas.diagnostic_application_precision.v19",
            "atlas.diagnostic_application_precision.v20",
        }:
            try:
                from merlin.compile.c_integer import INTEGER_OPERATIONS
            except ImportError as error:
                raise NativeCompilationError(
                    "tool_unavailable", "selected CPU integer operations need Merlin's integer C emitter"
                ) from error
            operations = policy.get("host_integer_ops")
            if (
                not isinstance(operations, list)
                or not operations
                or any(not isinstance(op, str) or op not in INTEGER_OPERATIONS for op in operations)
                or len(set(operations)) != len(operations)
            ):
                raise NativeCompilationError("unsupported_semantics", "host integer selection is unsupported")
            self.host_integer_ops = tuple(operations)
            expected["host_integer_ops"] = operations
        if policy.get("schema") in {
            "atlas.diagnostic_application_precision.v15",
            "atlas.diagnostic_application_precision.v16",
            "atlas.diagnostic_application_precision.v17",
            "atlas.diagnostic_application_precision.v18",
            "atlas.diagnostic_application_precision.v19",
            "atlas.diagnostic_application_precision.v20",
        }:
            from .row_softmax import GUARDED_SOFTMAX_POLICY

            if policy.get("guarded_masked_softmax") != GUARDED_SOFTMAX_POLICY:
                raise NativeCompilationError("unsupported_semantics", "guarded masked softmax policy differs")
            self.guarded_masked_softmax = True
            expected["guarded_masked_softmax"] = GUARDED_SOFTMAX_POLICY
        if policy.get("schema") in {
            "atlas.diagnostic_application_precision.v16",
            "atlas.diagnostic_application_precision.v17",
            "atlas.diagnostic_application_precision.v18",
            "atlas.diagnostic_application_precision.v19",
            "atlas.diagnostic_application_precision.v20",
        }:
            from .gelu import GELU_POLICY

            if policy.get("gelu") != GELU_POLICY:
                raise NativeCompilationError("unsupported_semantics", "selected GELU policy differs")
            expected["gelu"] = GELU_POLICY
            self.gelu = True
        if policy.get("schema") in {
            "atlas.diagnostic_application_precision.v17",
            "atlas.diagnostic_application_precision.v18",
            "atlas.diagnostic_application_precision.v19",
            "atlas.diagnostic_application_precision.v20",
        }:
            try:
                from merlin.compile.c_float_casts import FLOAT_CAST_OPERATIONS
            except ImportError as error:
                raise NativeCompilationError(
                    "tool_unavailable", "selected CPU float casts need Merlin's C emitter"
                ) from error
            casts = policy.get("host_float_casts")
            if (
                not isinstance(casts, list)
                or not casts
                or any(not isinstance(op, str) or op not in FLOAT_CAST_OPERATIONS for op in casts)
                or len(set(casts)) != len(casts)
            ):
                raise NativeCompilationError("unsupported_semantics", "host float cast selection is unsupported")
            self.host_float_casts = tuple(casts)
            expected["host_float_casts"] = casts
        if policy.get("schema") == "atlas.diagnostic_application_precision.v18" or (
            policy.get("schema")
            in {"atlas.diagnostic_application_precision.v19", "atlas.diagnostic_application_precision.v20"}
            and "source_constant_math" in policy
        ):
            from merlin.semantic_compiler.constant_math_backend import validate_policy

            try:
                expected["source_constant_math"] = validate_policy(policy.get("source_constant_math"))
            except ValueError as error:
                raise NativeCompilationError("unsupported_semantics", str(error)) from error
        if policy.get("schema") in {
            "atlas.diagnostic_application_precision.v19",
            "atlas.diagnostic_application_precision.v20",
        }:
            try:
                from merlin.compile.c_float_arithmetic import (
                    FLOAT_ARITHMETIC_OPERATIONS,
                )
            except ImportError as error:
                raise NativeCompilationError(
                    "tool_unavailable", "selected CPU float arithmetic needs Merlin's C emitter"
                ) from error

            operations = policy.get("host_float_arithmetic")
            if (
                not isinstance(operations, list)
                or not operations
                or any(not isinstance(op, str) or op not in FLOAT_ARITHMETIC_OPERATIONS for op in operations)
                or len(set(operations)) != len(operations)
            ):
                raise NativeCompilationError("unsupported_semantics", "host float arithmetic selection is unsupported")
            self.host_float_arithmetic = tuple(operations)
            expected["host_float_arithmetic"] = operations
        if policy.get("schema") == "atlas.diagnostic_application_precision.v20":
            try:
                from merlin.compile.c_unary_math import validate_policy
            except ImportError as error:
                raise NativeCompilationError(
                    "tool_unavailable", "selected CPU math needs Merlin's C emitter"
                ) from error
            try:
                self.host_unary_math = validate_policy(policy.get("host_unary_math"))
            except ValueError as error:
                raise NativeCompilationError("unsupported_semantics", str(error)) from error
            expected["host_unary_math"] = self.host_unary_math
        if policy != expected:
            raise NativeCompilationError("unsupported_semantics", "selected application precision policy differs")

    def plan(self, source):
        if source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic":
            raise NativeCompilationError(
                "unsupported_semantics",
                "application needs the selected diagnostic source",
            )
        pure = [node for node in source.nodes if node.effect == "pure"]
        indices = {node.id: index for index, node in enumerate(pure)}
        claimed, regions = set(), []

        def add(keys, kind, epilogue=None):
            if claimed.intersection(keys):
                raise NativeCompilationError("compile_error", "source application regions overlap")
            region = extract_semantic_region(source, tuple(keys))
            if kind == "contraction":
                prepare_contraction(
                    region,
                    operand_scale_exponents=self.operand_scale_exponents,
                    epilogue=epilogue,
                )
            elif kind in {"softmax", "masked_softmax", "guarded_softmax"}:
                lower_semantic_softmax(region, allow_masked=kind != "softmax", allow_guarded=kind == "guarded_softmax")
            elif kind in {"pointwise", "pointwise_math"}:
                from .pointwise import prepare_source

                prepare_source(region, allow_math=kind == "pointwise_math", bf16_policy=self.bf16_pointwise)
            elif kind == "reduction":
                from .reduction import prepare_source

                prepare_source(region)
            elif kind == "integer":
                _integer_c(region, "integer")
            elif kind == "float_cast":
                _float_cast_c(region, "float_cast")
            elif kind == "float_arithmetic":
                _float_arithmetic_c(region, "float_arithmetic")
            elif kind == "unary_math":
                _unary_math_c(region, "unary_math", self.host_unary_math)
            elif kind == "predicate":
                _predicate_c(region, "predicate")
            else:
                emit_layout_c(region, "layout")

            def ports(values):
                return tuple(
                    BufferRepresentation(
                        key,
                        ValueRepresentation("host_memory", "dense", region.node(key).type.dtype),
                    )
                    for key in values
                )

            members = tuple(sorted(indices[key] for key in keys))
            regions.append(
                RegionAlternative(
                    f"region_{members[0]}",
                    members,
                    "source_" + kind,
                    "host"
                    if kind in {"layout", "predicate", "integer", "float_cast", "float_arithmetic", "unary_math"}
                    else "accelerator",
                    CycleInterval.unknown("application region costs have not been calibrated"),
                    ports(node.id for node in region.nodes if node.effect in {"input", "constant"}),
                    ports(region.outputs),
                    metadata=tuple(sorted((("kind", kind),) + ((("epilogue", epilogue),) if epilogue else ()))),
                )
            )
            claimed.update(keys)

        if self.guarded_masked_softmax:
            from merlin.semantic_compiler.guarded_softmax_source import (
                recognize_guarded_masked_row_softmax,
            )

            for node in pure:
                if node.op != "select" or node.id in claimed:
                    continue
                try:
                    guarded = recognize_guarded_masked_row_softmax(source, node.id)
                except (KeyError, ValueError):
                    continue
                add(guarded.computations, "guarded_softmax")
        for node in pure:
            if node.op != "f32_div" or node.id in claimed:
                continue
            try:
                softmax = recognize_row_softmax(source, node.id)
            except (KeyError, ValueError):
                continue
            if self.masked_softmax:
                from merlin.semantic_compiler.softmax_source import (
                    recognize_masked_row_softmax,
                )

                try:
                    masked = recognize_masked_row_softmax(source, node.id)
                except (KeyError, ValueError):
                    pass
                else:
                    add(masked.computations, "masked_softmax")
                    continue
            add(softmax.computations, "softmax")
        if self.sigmoid and self.pointwise:
            from merlin.semantic_compiler.silu_source import recognize_silu

            for node in pure:
                if node.op != "f32_mul" or node.id in claimed:
                    continue
                try:
                    pair = recognize_silu(source, node.id)
                except (KeyError, ValueError):
                    continue
                if not claimed.intersection(pair):
                    add(pair, "pointwise_math")
        if "relu" in self.epilogues:
            for node in pure:
                if not is_source_relu(node) or node.id in claimed:
                    continue
                producer = source.node(node.inputs[0])
                consumers = [item.id for item in source.nodes if producer.id in item.inputs]
                if (
                    producer.op == "source_f32_matmul_unqualified"
                    and len(producer.type.shape) == 2
                    and producer.id not in claimed
                    and producer.id not in source.outputs
                    and consumers == [node.id]
                ):
                    add((producer.id, node.id), "contraction", "relu")
        for node in pure:
            if node.id in claimed:
                continue
            if node.op == "source_f32_matmul_unqualified":
                add((node.id,), "contraction")
            elif node.op in self.host_layouts:
                add((node.id,), "layout")
            elif node.op in self.host_integer_ops:
                add((node.id,), "integer")
            elif self.host_unary_math and node.op in self.host_unary_math["operations"]:
                add((node.id,), "unary_math")
            elif node.op in self.host_float_arithmetic:
                add((node.id,), "float_arithmetic")
            elif node.op in self.host_float_casts:
                add((node.id,), "float_cast")
            elif node.op in self.host_predicates:
                add((node.id,), "predicate")
            elif (
                self.pointwise
                and (node.op in {"f32_mul", "f32_add", "f32_sub"} or is_source_relu(node))
                or self.bf16_pointwise
                and node.op == "bf16_mul"
            ):
                add((node.id,), "pointwise")
            elif self.reductions and node.op in {
                "source_f32_reduce_add_unqualified",
                "f32_reduce_add",
            }:
                add((node.id,), "reduction")
            elif self.gelu and node.op == "source_f32_composite_unqualified" and is_source_gelu(node):
                add((node.id,), "pointwise_math")
            elif self.sigmoid and node.op == "source_f32_composite_unqualified":
                from .sigmoid import is_source_sigmoid

                if not is_source_sigmoid(node):
                    raise NativeCompilationError("unsupported_semantics", f"unrecognized source composite: {node.id}")
                add((node.id,), "pointwise_math")
            elif self.pointwise_math and node.op in {
                "f32_div",
                "source_f32_rsqrt_unqualified",
            }:
                add((node.id,), "pointwise_math")
            else:
                raise NativeCompilationError(
                    "unsupported_semantics",
                    f"no compiled application route for {node.id}: {node.op} {node.type.record()}",
                )
        return GlobalPlan(
            tuple(sorted(regions, key=lambda item: item.nodes)),
            (),
            CycleInterval.unknown("structural first-valid; no performance claim"),
        )

    def compile_region(self, snapshot, region, alternative, name, segments, target_source, limits):
        kind = dict(alternative.metadata)["kind"]
        if kind == "contraction":
            from .looped_tiling import compile_looped, prepare_execution_plan

            precision = prepare_contraction(
                region,
                operand_scale_exponents=self.operand_scale_exponents,
                epilogue=dict(alternative.metadata).get("epilogue"),
            )
            abi = AtlasNativeBinding.plan_boundary_abi(precision.request)
            _, _, streaming = prepare_execution_plan(precision.request)
            if streaming is not None and streaming.k_block_tiles is not None:
                from .chunked_tiling import compile_members

                compile_members(
                    snapshot,
                    precision.request,
                    streaming,
                    fixed_inputs=abi["fixed_inputs"],
                    fixed_outputs=tuple(abi["fixed_outputs"]),
                    root=segments,
                    prefix=name + "_",
                    target_source=target_source,
                    limits=limits,
                )
                return self.export_region(region, alternative, name, segments, target_source)
            compile_looped(
                snapshot,
                precision.request,
                fixed_inputs=abi["fixed_inputs"],
                fixed_outputs=tuple(abi["fixed_outputs"]),
                target_source=target_source,
                destination=segments / name,
                limits=limits,
            )
        elif kind in {"softmax", "masked_softmax", "guarded_softmax"}:
            from .row_loop import compile_row_loop

            precision = lower_semantic_softmax(
                region, allow_masked=kind != "softmax", allow_guarded=kind == "guarded_softmax"
            )
            compile_row_loop(
                snapshot,
                precision.request,
                target_source=target_source,
                destination=segments / name,
                limits=limits,
            )
        elif kind in {"pointwise", "pointwise_math", "reduction"}:
            if kind in {"pointwise", "pointwise_math"}:
                from .pointwise import compile_pointwise as compile_region
                from .pointwise import plan_abi, prepare_source
            else:
                from .reduction import compile_reduction as compile_region
                from .reduction import plan_abi, prepare_source

            precision = prepare_source(
                region,
                **(
                    {"allow_math": kind == "pointwise_math", "bf16_policy": self.bf16_pointwise}
                    if kind != "reduction"
                    else {}
                ),
            )
            abi = plan_abi(precision.request)
            compile_region(
                snapshot,
                precision.request,
                fixed_inputs=abi["fixed_inputs"],
                fixed_outputs=tuple(abi["fixed_outputs"]),
                target_source=target_source,
                destination=segments / name,
                limits=limits,
            )
        return self.export_region(region, alternative, name, segments, target_source)

    def export_region(self, region, alternative, name, segments, target_source):
        kind = dict(alternative.metadata)["kind"]
        if kind in {"layout", "predicate", "integer", "float_cast", "float_arithmetic", "unary_math"}:
            if kind == "float_arithmetic":
                emitter = _float_arithmetic_c
            elif kind == "float_cast":
                emitter = _float_cast_c
            elif kind == "integer":
                emitter = _integer_c
            elif kind == "predicate":
                emitter = _predicate_c
            else:
                emitter = emit_layout_c
            record, generated = (
                _unary_math_c(region, name, self.host_unary_math) if kind == "unary_math" else emitter(region, name)
            )
            return CompiledCRegion(
                name,
                name + ".h",
                tuple(record["inputs"]),
                tuple(record["outputs"]),
                {key: value.encode() for key, value in generated.items()},
                (),
                (),
                record,
            )
        if kind in {"pointwise", "pointwise_math", "reduction"}:
            if kind in {"pointwise", "pointwise_math"}:
                from .pointwise_c import prepare_c_package
            else:
                from .reduction_c import prepare_c_package

            record, files = prepare_c_package(
                region,
                segments / name,
                name,
                target_source=target_source,
                **(
                    {"allow_math": kind == "pointwise_math", "bf16_policy": self.bf16_pointwise}
                    if kind != "reduction"
                    else {}
                ),
            )
            return CompiledCRegion(
                name + "_source_dispatch",
                name + ".h",
                tuple(record["input_node_ids"]),
                tuple(record["output_node_ids"]),
                files,
                ("runtime/atlas_host.c", "runtime/atlas_host.h"),
                ("segments/" + name,),
                {
                    "kind": kind,
                    "context_type": name + "_context",
                    "source_context_type": name + "_source_context",
                    "binding_function": name + "_bind",
                    "source_binding_function": name + "_source_bind",
                    **record,
                },
            )
        if kind == "contraction":
            from merlin.semantic_compiler.batched_contraction import (
                BatchedContractionPrecision,
            )

            from .looped_c import prepare_c_package

            precision = prepare_contraction(
                region,
                operand_scale_exponents=self.operand_scale_exponents,
                epilogue=dict(alternative.metadata).get("epilogue"),
            )
            from .looped_tiling import prepare_execution_plan

            _, _, streaming = prepare_execution_plan(precision.request)
            chunked_programs = None
            if streaming is not None and streaming.k_block_tiles is not None:
                from .chunked_c import prepare_members
                from .chunked_tiling import chunk_bodies

                batched = isinstance(precision, BatchedContractionPrecision)
                roots = {role: segments / (name + "_" + role) for role, _ in chunk_bodies(precision.request, streaming)}
                record, files = prepare_members(
                    precision.request,
                    streaming,
                    roots,
                    name,
                    target_source=target_source,
                    runtime_linkage="separate",
                    semantic_source=precision.plan.matrix_source if batched else region,
                    semantic_operand_scale_exponents=self.operand_scale_exponents,
                    semantic_epilogue=dict(alternative.metadata).get("epilogue"),
                )
                if batched:
                    from .batched_c import wrap_batched_c_package

                    record, files = wrap_batched_c_package(precision, name, record, files)
                    inputs = precision.plan.input_ids
                else:
                    inputs = tuple(region.nodes[item.source_value.index].id for item in precision.inputs)
                chunked_programs = tuple("segments/" + path.name for path in roots.values())
            elif isinstance(precision, BatchedContractionPrecision):
                from .batched_c import prepare_batched_c_package

                record, files = prepare_batched_c_package(
                    precision,
                    segments / name,
                    name,
                    target_source=target_source,
                    operand_scale_exponents=self.operand_scale_exponents,
                )
                inputs = precision.plan.input_ids
            else:
                record, files = prepare_c_package(
                    segments / name / "source-request.json",
                    segments / name,
                    name,
                    target_source=target_source,
                    runtime_linkage="separate",
                    semantic_source=region,
                    semantic_operand_scale_exponents=self.operand_scale_exponents,
                    semantic_epilogue=dict(alternative.metadata).get("epilogue"),
                )
                inputs = tuple(region.nodes[item.source_value.index].id for item in precision.inputs)
            programs = chunked_programs or ("segments/" + name,)
            interface = record.get("batch_interface", record["source_region"]["source_interface"])
        elif kind in {"softmax", "masked_softmax", "guarded_softmax"}:
            from .softmax_c import _prepare_c_package

            precision = lower_semantic_softmax(
                region, allow_masked=kind != "softmax", allow_guarded=kind == "guarded_softmax"
            )
            record, files = _prepare_c_package(
                precision.request,
                segments,
                name,
                target_source=target_source,
                runtime_linkage="separate",
                semantic_source=region,
                allow_masked=kind != "softmax",
                allow_guarded=kind == "guarded_softmax",
                row_loop_compilation=segments / name,
            )
            inputs = (
                (precision.source_region.scores, precision.source_region.mask)
                if kind in {"masked_softmax", "guarded_softmax"}
                else (precision.source_region.scores,)
            )
            programs = ("segments/" + name,)
            interface = record["source_interface"]
        else:
            raise NativeCompilationError("compile_error", "unknown selected source region kind")
        description = {
            "kind": kind,
            "context_type": name + "_context",
            "source_context_type": name + "_source_context",
            "binding_function": name + "_bind",
            "source_binding_function": name + "_source_bind",
            "workspace_bytes": record["workspace_bytes"],
            "source_workspace_bytes": interface["workspace_bytes"],
            "precision_plan": precision.record(),
            "native_export": record,
        }
        return CompiledCRegion(
            record.get("batch_dispatch", name + "_source_dispatch"),
            record.get("batch_header", name + ".h"),
            inputs,
            region.outputs,
            files,
            ("runtime/atlas_host.c", "runtime/atlas_host.h"),
            programs,
            description,
        )
