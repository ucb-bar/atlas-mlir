"""Compile supported source regions through Atlas's existing native backends."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

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
    prepare_already_quantized_contraction,
)
from .binding import TARGET_IDENTITY, AtlasNativeBinding
from .c_runtime import RUNTIME_SHARED_FILES
from .gelu import is_source_gelu
from .row_softmax import SOFTMAX_POLICY, SOFTMAX_DOMAIN
from .softmax_precision import lower_semantic_softmax
from .publication_options import publication_options

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


def _host_k1_c(region, name):
    from .host_k1_contraction import emit_c

    try:
        return emit_c(region, name)
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
    no implicit host arithmetic fallback or model-name/shape-list selection.
    """

    # Analysis parameters only; neither attribute is removed from compilation.
    region_loop_parameters = ("static_exp", "tag")
    # Provider capability only. Both caller reuse options default to False.
    supports_materialization_cache = True
    publication_options = staticmethod(publication_options)

    @staticmethod
    def supports_region_materialization(region, alternative):
        # Other kinds retain their complete original compilation path until
        # their own binding contract and byte-equality gates are available.
        return dict(alternative.metadata)["kind"] in {
            "exact_float_device", "layout_device", "activation_quantize_device", "pointwise", "pointwise_math", "reduction",
            "masked_softmax", "already_quantized"}

    def materialize_region(self, snapshot, canonical, alternative, name, segments, target_source, limits):
        if not self.supports_region_materialization(canonical, alternative):
            raise ValueError("Atlas region materialization kind is unsupported")
        kind = dict(alternative.metadata)["kind"]
        if kind in {"pointwise", "pointwise_math", "reduction", "masked_softmax", "already_quantized"}:
            from .selection_materialization import SelectionCapture, supports_captured_tile

            capture = SelectionCapture(snapshot)
            self.compile_region(capture, canonical, alternative, name, segments, target_source, limits)
            if kind == "already_quantized":
                manifest = segments / name / "manifest.json"
                if (not manifest.is_file()
                        or json.loads(manifest.read_bytes()).get("schema") != "atlas.quantized_loop_compilation.v1"):
                    return None
            elif not supports_captured_tile(segments / name / "body", capture):
                # A staged body has checked selections for its substages, not
                # one complete tile proof. Preserve its normal original compile.
                return None
            return canonical, segments / name, capture
        self.compile_region(snapshot, canonical, alternative, name, segments, target_source, limits)
        return canonical, segments / name

    def bind_materialization(self, materialized, region, alternative, name, segments, target_source):
        from .application_materialization import bind_exact_float, bind_layout, bind_quantizer

        if not self.supports_region_materialization(region, alternative):
            raise ValueError("Atlas region materialization binding kind is unsupported")
        kind = dict(alternative.metadata)["kind"]
        canonical, compiled = materialized[:2]
        if kind == "already_quantized":
            from .selection_materialization import bind_counted

            bind_counted(region, canonical, compiled, segments / name, materialized[2], target_source=target_source)
        elif kind in {"pointwise", "pointwise_math", "reduction", "masked_softmax"}:
            from .selection_materialization import bind_pointwise, bind_row_loop

            def prepared(source):
                if kind == "reduction":
                    from .reduction import prepare_source
                    return prepare_source(source, reduction_policy=self.reduction_policy).request
                if kind == "masked_softmax":
                    return lower_semantic_softmax(source, allow_masked=True, allow_guarded=False,
                        score_domain=self.already_quantized_score_domain).request
                from .pointwise import prepare_source
                return prepare_source(source, allow_math=kind == "pointwise_math", bf16_policy=self.bf16_pointwise,
                    exp_policy=self.source_exp_policy,
                    rne_add_policy=self.qualified_rne_add if source.nodes[-1].op in {"f32_add", "bf16_add"} else None).request
            if kind == "masked_softmax":
                bind_row_loop(prepared(region), prepared(canonical), compiled, segments / name, materialized[2], target_source=target_source)
            else:
                bind_pointwise(prepared(region), prepared(canonical), compiled, segments / name,
                               materialized[2], reduction=kind == "reduction")
        else:
            binder = {"exact_float_device": bind_exact_float, "layout_device": bind_layout,
                      "activation_quantize_device": bind_quantizer}[kind]
            binder(region, canonical, compiled, segments / name)
        # The ordinary exporter checks complete original source correspondence,
        # emitted machine/binary replay, bounds, ABI ports and C publication.
        return self.export_region(region, alternative, name, segments, target_source)

    def materialization_identity(self, region, alternative):
        """Approve only provenance erasure; retain every provider policy choice.

        Plan-local IDs and source node names do not affect pure tool behavior.
        Original source identity and publication metadata are regenerated by
        compile_region, export_region and the existing source replay checks.
        """
        from . import binding
        from .application_materialization import provider_sources
        from .binding import _native_pipeline
        from .publication_options import deterministic_receipts_enabled
        from .quantized_operand_reuse import reuse_policy

        names = {node.id: f"node_{i}" for i, node in enumerate(region.nodes)}
        choice = alternative.to_dict()
        choice.pop("id")
        choice.pop("nodes")
        for side in ("inputs", "outputs"):
            for port in choice[side]:
                port["buffer"] = names.get(port["buffer"], port["buffer"])
        for side in ("resident_inputs", "resident_outputs"):
            choice[side] = [names.get(name, name) for name in choice[side]]
        policy = {key: value for key, value in vars(self).items() if key != "original_source_sha256"}
        tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", Path(__file__).resolve().parents[2] / "build/bin"))
        tool_identities = {}
        for tool in (tools / "atlas-opt", tools / "atlas-emit"):
            if tool.is_file():
                stat = tool.stat()
                tool_identities[str(tool.resolve())] = binding._tool_digest(
                    str(tool.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        return {
            "target_choice": choice,
            "source_identity_policy": "provenance",
            "compilation_identity": {
                "schema": "atlas.region_tool_partition.v1",
                "policy": policy,
                "application_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "binding_sha256": hashlib.sha256(Path(binding.__file__).read_bytes()).hexdigest(),
                "materialization_sources": provider_sources(),
                "tools": tool_identities,
                "pipeline": _native_pipeline(),
                "capture_postspill": os.environ.get("ATLAS_CAPTURE_POSTSPILL") == "1",
                "deterministic_receipts": deterministic_receipts_enabled(),
                "quantized_operand_reuse": reuse_policy(),
            },
        }

    @staticmethod
    def application_binding(name, regions, interface):
        if regions and all(row["placement"] == "host" for row in regions):
            return None
        from .application_c_context import application_binding

        return application_binding(name, regions, interface)

    @property
    def source_operation_translators(self):
        from .source_translation import (
            translate_prequantized_contraction, translate_activation_quantizer,
        )
        if not self.already_quantized:
            return {}
        return {**{op: translate_prequantized_contraction for op in
                   ("atlas.qlinear", "atlas.qk_matmul", "atlas.pv_matmul")},
                "atlas.quantize_act": translate_activation_quantizer}

    def __init__(self, policy, *, original_source_sha256):
        self.original_source_sha256 = original_source_sha256
        self.host_preparation_scope = None
        if "host_preparation_scope" in policy:
            from .host_unary_preparation import SCOPE_POLICY
            from merlin.semantic_compiler.model import TensorType

            scope = policy["host_preparation_scope"]
            if (not isinstance(scope, dict) or set(scope) != {"policy", "constant_inputs"}
                    or scope["policy"] != SCOPE_POLICY or not isinstance(scope["constant_inputs"], list)):
                raise NativeCompilationError("unsupported_semantics", "host preparation scope differs")
            identifiers = set()
            for entry in scope["constant_inputs"]:
                if (not isinstance(entry, dict) or set(entry) != {"node_id", "type"}
                        or not isinstance(entry["node_id"], str) or not entry["node_id"]
                        or entry["node_id"] in identifiers):
                    raise NativeCompilationError("unsupported_semantics", "host preparation constant input contract differs")
                try:
                    TensorType.from_record(entry["type"])
                except (TypeError, ValueError, KeyError) as error:
                    raise NativeCompilationError("unsupported_semantics", "host preparation constant input type differs") from error
                identifiers.add(entry["node_id"])
            self.__init__({key: value for key, value in policy.items() if key != "host_preparation_scope"},
                          original_source_sha256=original_source_sha256)
            if not self.host_unary_math:
                raise NativeCompilationError("unsupported_semantics", "host preparation scope requires selected unary math")
            # Payload authentication belongs to the caller's existing immutable
            # parameter binding. Planning checks this explicit source boundary
            # contract, and never infers a constant from a floating input type.
            self.host_preparation_scope = json.loads(json.dumps(scope))
            return
        if "activation_quantizer" in policy:
            route = policy["activation_quantizer"]
            if route not in {"host", "device"}:
                raise NativeCompilationError("unsupported_semantics", "activation quantizer route must be host or device")
            self.__init__({key: value for key, value in policy.items() if key != "activation_quantizer"},
                          original_source_sha256=original_source_sha256)
            self.activation_quantizer_route = route
            return
        self.activation_quantizer_route = "device"
        from .already_quantized import POLICY as ALREADY_QUANTIZED_POLICY

        self.host_bf16_linear_policy = None
        if "host_bf16_linear" in policy:
            from merlin.compile.c_bf16_linear import SOURCE_POLICY as SOURCE_BF16_LINEAR_POLICY

            selected = policy["host_bf16_linear"]
            if selected != SOURCE_BF16_LINEAR_POLICY:
                raise NativeCompilationError("unsupported_semantics", "host BF16 linear policy differs")
            self.__init__({key: value for key, value in policy.items() if key != "host_bf16_linear"},
                          original_source_sha256=original_source_sha256)
            self.host_bf16_linear_policy = selected
            return
        self.already_quantized = False
        self.already_quantized_mask_sentinel = None
        self.already_quantized_score_domain = SOFTMAX_DOMAIN
        self.source_exp_policy = None
        if "already_quantized" in policy:
            if policy["already_quantized"] != ALREADY_QUANTIZED_POLICY:
                raise NativeCompilationError("unsupported_semantics", "already-quantized intake policy differs")
            extra_bf16 = policy.get("bf16_pointwise") if policy.get("schema") != "atlas.diagnostic_application_precision.v21" else None
            selected_exp = policy.get("source_exp")
            if "source_exp" in policy:
                from .source_exp import POLICY as SOURCE_EXP_POLICY

                if selected_exp != SOURCE_EXP_POLICY:
                    raise NativeCompilationError("unsupported_semantics", "already-quantized source exp policy differs")
            selected_mask = policy.get("already_quantized_mask")
            if "already_quantized_mask" in policy and selected_mask not in {
                "bf16-minfinite-zero-bounded32-nonempty-v1", "bf16-minfinite-zero-rowspan80-nonempty-v1"
            }:
                raise NativeCompilationError("unsupported_semantics", "already-quantized finite mask policy differs")
            remove = {"already_quantized", "already_quantized_mask", "source_exp", *(("bf16_pointwise",) if extra_bf16 is not None else ())}
            if extra_bf16 is not None:
                from .pointwise import BF16_POINTWISE_POLICY

                if extra_bf16 != BF16_POINTWISE_POLICY:
                    raise NativeCompilationError("unsupported_semantics", "already-quantized BF16 pointwise policy differs")
            self.__init__({key: value for key, value in policy.items() if key not in remove},
                          original_source_sha256=original_source_sha256)
            if extra_bf16 is not None:
                if self.bf16_pointwise not in (None, extra_bf16):
                    raise NativeCompilationError("unsupported_semantics", "conflicting already-quantized BF16 pointwise policy")
                self.bf16_pointwise = extra_bf16
            if selected_exp is not None:
                self.source_exp_policy = selected_exp
            if selected_mask is not None:
                self.already_quantized_mask_sentinel = 0xff7f0000
                if selected_mask == "bf16-minfinite-zero-rowspan80-nonempty-v1":
                    self.already_quantized_score_domain = "bf16-row-relative-span80-pad32-v1"
            self.already_quantized = True
            return
        if policy.get("schema") == "atlas.diagnostic_normalization_application_precision.v1":
            from .rmsnorm import RMSNORM_POLICY
            from .qualified_rne_add import RNE_ADD_POLICY

            extra = set(policy) - {"schema", "source_sha256", "base_policy"}
            if (not extra or not extra <= {"rmsnorm", "rne_add"}
                    or policy.get("source_sha256") != original_source_sha256
                    or not isinstance(policy.get("base_policy"), dict)):
                raise NativeCompilationError("unsupported_semantics", "normalization precision fields or source differ")
            base = policy["base_policy"]
            if base.get("schema") == policy["schema"] or base.get("source_sha256") != original_source_sha256:
                raise NativeCompilationError("unsupported_semantics", "normalization base policy differs")
            if ("rmsnorm" in extra and policy["rmsnorm"] != RMSNORM_POLICY
                    or "rne_add" in extra and policy["rne_add"] != RNE_ADD_POLICY):
                raise NativeCompilationError("unsupported_semantics", "normalization arithmetic policy differs")
            self.__init__(base, original_source_sha256=original_source_sha256)
            if (("rne_add" in extra and self.rne_add is not None)
                    or ("rmsnorm" in extra and self.rmsnorm_schedules)):
                raise NativeCompilationError("unsupported_semantics", "conflicting nested arithmetic precision policies")
            self.rmsnorm = "rmsnorm" in extra
            self.qualified_rne_add = policy.get("rne_add")
            return
        self.rmsnorm = False
        self.qualified_rne_add = None

        if policy.get("schema") == "atlas.diagnostic_application_precision.v24":
            from .hierarchical_rmsnorm import POLICY as norm_policy

            base = policy.get("base_policy")
            if (set(policy) != {"schema", "source_sha256", "base_policy", "rmsnorm_schedules"}
                    or policy["source_sha256"] != original_source_sha256
                    or not isinstance(base, dict) or base.get("schema") == policy["schema"]
                    or base.get("source_sha256") != original_source_sha256):
                raise NativeCompilationError("unsupported_semantics", "source-bound RMSNorm policy fields or source differ")
            self.__init__(base, original_source_sha256=original_source_sha256)
            if self.rmsnorm:
                raise NativeCompilationError("unsupported_semantics", "conflicting nested normalization precision policies")
            rows = policy["rmsnorm_schedules"]
            if (not isinstance(rows, list) or not rows
                    or any(not isinstance(row, dict) or not isinstance(row.get("source_output"), str) for row in rows)
                    or len({row["source_output"] for row in rows}) != len(rows)):
                raise NativeCompilationError("unsupported_semantics", "RMSNorm schedules need distinct source outputs")
            for row in rows:
                if (set(row) != {"source_output", "source_request_digest", "provenance_sha256", "policy", "output_scale_exponent"}
                        or row["policy"] != norm_policy or type(row["output_scale_exponent"]) is not int
                        or not -32 <= row["output_scale_exponent"] <= 32
                        or any(not isinstance(row.get(key), str) or len(row[key]) != 64
                               or any(c not in "0123456789abcdef" for c in row[key])
                               for key in ("source_request_digest", "provenance_sha256"))):
                    raise NativeCompilationError("unsupported_semantics", "RMSNorm precision selection or hash binding differs")
            self.rmsnorm_schedules = {row["source_output"]: {key: value for key, value in row.items()
                                      if key != "source_output"} for row in rows}
            return
        if policy.get("schema") == "atlas.diagnostic_application_precision.v23":
            from .quantized_softmax_value import POLICY as value_policy
            from .rne_add import RNE_ADD_POLICY

            if (set(policy) != {"schema", "source_sha256", "base_policy", "rne_add",
                               "softmax_value_schedules", "mxu1_contraction_schedules"}
                    or policy["source_sha256"] != original_source_sha256
                    or policy["base_policy"].get("schema") == policy["schema"]
                    or policy["base_policy"].get("source_sha256") != original_source_sha256):
                raise NativeCompilationError("unsupported_semantics", "source-bound attention policy fields or source differ")
            self.__init__(policy["base_policy"], original_source_sha256=original_source_sha256)
            if policy["rne_add"] not in (None, RNE_ADD_POLICY):
                raise NativeCompilationError("unsupported_semantics", "RNE addition precision selection differs")
            if policy["rne_add"] is not None and self.qualified_rne_add is not None:
                raise NativeCompilationError("unsupported_semantics", "conflicting nested addition precision policies")
            self.rne_add = policy["rne_add"]
            for field, attribute in (("softmax_value_schedules", "softmax_value_schedules"),
                                     ("mxu1_contraction_schedules", "mxu1_contraction_schedules")):
                rows = policy[field]
                if (not isinstance(rows, list) or any(not isinstance(row, dict)
                        or not isinstance(row.get("source_output"), str) for row in rows)
                        or len({row["source_output"] for row in rows}) != len(rows)):
                    raise NativeCompilationError("unsupported_semantics", "source-bound precision schedules need distinct outputs")
                selected = {}
                for row in rows:
                    if field == "softmax_value_schedules" and (set(row) != {
                            "source_output", "source_request_digest", "provenance_sha256", "policy", "value_scale_exponent"}
                            or row["policy"] != value_policy or type(row["value_scale_exponent"]) is not int
                            or not -32 <= row["value_scale_exponent"] <= 32):
                        raise NativeCompilationError("unsupported_semantics", "quantized softmax value precision selection differs")
                    if any(not isinstance(row.get(key), str) or len(row[key]) != 64
                           or any(c not in "0123456789abcdef" for c in row[key])
                           for key in ("source_request_digest", "provenance_sha256")):
                        raise NativeCompilationError("unsupported_semantics", "precision source and provenance hashes differ")
                    selected[row["source_output"]] = {key: value for key, value in row.items() if key != "source_output"}
                setattr(self, attribute, selected)
            return
        if policy.get("schema") == "atlas.diagnostic_application_precision.v22":
            if set(policy) != {"schema", "source_sha256", "base_policy", "contraction_schedules"} or policy["source_sha256"] != original_source_sha256:
                raise NativeCompilationError("unsupported_semantics", "column precision application fields or source differ")
            base = policy["base_policy"]
            if base.get("schema") == policy["schema"] or base.get("source_sha256") != original_source_sha256:
                raise NativeCompilationError("unsupported_semantics", "column precision base policy differs")
            self.__init__(base, original_source_sha256=original_source_sha256)
            schedules = policy["contraction_schedules"]
            if not isinstance(schedules, list) or not schedules or any(not isinstance(item, dict) or not isinstance(item.get("source_output"), str) for item in schedules):
                raise NativeCompilationError("unsupported_semantics", "column precision needs source output schedules")
            if len({item["source_output"] for item in schedules}) != len(schedules):
                raise NativeCompilationError("unsupported_semantics", "duplicate contraction precision schedule")
            self.contraction_schedules = {item["source_output"]: {key: value for key, value in item.items() if key != "source_output"} for item in schedules}
            if os.environ.get("ATLAS_KERNEL_LIBRARY"):
                raise NativeCompilationError("unsupported_semantics", "column precision does not admit an executable library provider")
            return
        self.contraction_schedules = None
        self.rne_add = None
        self.softmax_value_schedules = {}
        self.mxu1_contraction_schedules = {}
        self.rmsnorm_schedules = {}
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
        self.reduction_policy = None
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
            from .reduction import REDUCTION_POLICIES

            if policy.get("reduction") not in REDUCTION_POLICIES:
                raise NativeCompilationError("unsupported_semantics", "selected reduction arithmetic differs")
            expected["reduction"] = policy["reduction"]
            self.reduction_policy = policy["reduction"]
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

    def _contraction_exponents(self, region):
        if self.contraction_schedules is None:
            return self.operand_scale_exponents
        if len(region.outputs) != 1 or region.outputs[0] not in self.contraction_schedules:
            raise NativeCompilationError("unsupported_semantics", "missing source-bound contraction precision schedule")
        return self.contraction_schedules[region.outputs[0]]

    def _extra_precision(self, kind, region):
        if kind == "rne_add":
            from . import rne_add as module

            return module, {"policy": self.rne_add}
        if len(region.outputs) != 1:
            raise NativeCompilationError("unsupported_semantics", "source-bound region must have one result")
        schedules = (self.rmsnorm_schedules if kind == "hierarchical_rmsnorm" else
                     self.softmax_value_schedules if kind == "softmax_value" else self.mxu1_contraction_schedules)
        selected = schedules.get(region.outputs[0])
        if selected is None or selected["source_request_digest"] != region.digest():
            raise NativeCompilationError("unsupported_semantics", "selected region source digest differs")
        if kind == "softmax_value":
            from . import quantized_softmax_value as module

            return module, {"value_scale_exponent": selected["value_scale_exponent"],
                            "provenance_sha256": selected["provenance_sha256"]}
        if kind == "hierarchical_rmsnorm":
            from . import hierarchical_rmsnorm as module

            return module, {"policy": selected}
        from . import mxu1_contraction as module

        return module, {"policy": selected}

    def plan(self, source):
        if source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic":
            raise NativeCompilationError(
                "unsupported_semantics",
                "application needs the selected diagnostic source",
            )
        pure = [node for node in source.nodes if node.effect == "pure"]
        from .source_bf16_linear import POLICY as SOURCE_BF16_LINEAR_POLICY, recognize_source_bf16_linear

        source_bf16_linear = {}
        if self.host_bf16_linear_policy == SOURCE_BF16_LINEAR_POLICY:
            for node in pure:
                if node.op != "f32_to_bf16":
                    continue
                try:
                    found = recognize_source_bf16_linear(source, node.id)
                except (KeyError, ValueError):
                    continue
                source_bf16_linear[node.id] = found
        admitted_contractions = {region.matmul for region in source_bf16_linear.values()}
        host_k1 = {
            node.id for node in pure
            if node.op == "source_f32_matmul_unqualified"
            and node.id not in admitted_contractions and self._is_host_k1(source, node)
        }
        fused_norms = {}
        if self.rmsnorm_schedules:
            from merlin.semantic_compiler.rmsnorm_source import recognize_rmsnorm

            for node in pure:
                if node.op != "f32_mul":
                    continue
                try:
                    found = recognize_rmsnorm(source, node.id)
                except (KeyError, ValueError):
                    continue
                fused_norms[node.id] = found
            if set(fused_norms) != set(self.rmsnorm_schedules):
                raise NativeCompilationError("unsupported_semantics", "RMSNorm schedules must cover the exact private source regions")
        fused_values = {}
        if self.softmax_value_schedules:
            from merlin.semantic_compiler.softmax_value_source import recognize_softmax_value

            for node in pure:
                if node.op != "source_f32_matmul_unqualified":
                    continue
                try:
                    found = recognize_softmax_value(source, node.id, masked=True)
                except (KeyError, ValueError):
                    continue
                fused_values[node.id] = found
            if set(fused_values) != set(self.softmax_value_schedules):
                raise NativeCompilationError("unsupported_semantics", "softmax value schedules must cover the exact private source regions")
        fused_nodes = {key for region in (*fused_norms.values(), *fused_values.values(),
                                        *source_bf16_linear.values()) for key in region.computations}
        contractions = {node.id for node in pure if node.op == "source_f32_matmul_unqualified"}
        if (not set(self.mxu1_contraction_schedules) <= contractions - fused_nodes):
            raise NativeCompilationError("unsupported_semantics", "MXU1 schedule does not name an independent source contraction")
        if self.contraction_schedules is not None and set(self.contraction_schedules) != {
            node.id for node in pure if node.op == "source_f32_matmul_unqualified"
            and node.id not in fused_nodes and node.id not in self.mxu1_contraction_schedules
        }:
            raise NativeCompilationError("unsupported_semantics", "contraction precision schedules must cover the exact source contractions")
        host_preparation, host_preparation_receipt = self._host_unary_preparation(
            source, host_k1, excluded=fused_nodes | set(self.mxu1_contraction_schedules)
        )
        if self.host_preparation_scope is not None:
            host_k1 &= host_preparation.keys()
        if self.already_quantized and any(
            node.op == "source_f32_matmul_unqualified"
            and node.id not in admitted_contractions and node.id not in host_k1 for node in pure
        ):
            raise NativeCompilationError(
                "unsupported_semantics",
                "already-quantized intake refuses floating contractions requiring operand recoding; "
                "provide authored code/exponent operands or a separately qualified unchanged-weight lowering",
            )
        indices = {node.id: index for index, node in enumerate(pure)}
        claimed, regions = set(), []

        def add(keys, kind, epilogue=None):
            if claimed.intersection(keys):
                raise NativeCompilationError("compile_error", "source application regions overlap")
            region = extract_semantic_region(source, tuple(keys))
            if kind in {"rne_add", "softmax_value", "mxu1_contraction", "hierarchical_rmsnorm"}:
                module, selected = self._extra_precision(kind, region)
                module.prepare_source(region, **selected)
            elif kind == "vector_quantize":
                from .vector_quantize import prepare_plan

                prepare_plan(region)
            elif kind == "layout_device":
                from .layout_device import prepare_source

                prepare_source(region)
            elif kind == "exact_float_device":
                from .exact_float_application import prepare_plan

                prepare_plan(region)
            elif kind == "activation_quantize_device":
                from .activation_quantize_application import prepare_plan

                prepare_plan(region)
            elif kind == "scalar_broadcast":
                from .scalar_broadcast import prepare_source

                prepare_source(region)
            elif kind == "already_quantized":
                prepare_already_quantized_contraction(region)
            elif kind == "host_bf16_linear":
                from merlin.compile.c_bf16_linear import emit_c

                emit_c(region, "bf16_linear", policy=self.host_bf16_linear_policy)
            elif kind in {"static_quantize", "dynamic_quantize"}:
                from . import dynamic_quantize, static_quantize

                (dynamic_quantize if kind == "dynamic_quantize" else static_quantize).prepare_source(region)
            elif kind == "contraction":
                prepare_contraction(
                    region,
                    operand_scale_exponents=self._contraction_exponents(region),
                    epilogue=epilogue,
                )
            elif kind in {"softmax", "masked_softmax", "guarded_softmax"}:
                lower_semantic_softmax(region, allow_masked=kind != "softmax", allow_guarded=kind == "guarded_softmax",
                                       score_domain=self.already_quantized_score_domain)
            elif kind in {"pointwise", "pointwise_math"}:
                from .pointwise import prepare_source

                prepare_source(region, allow_math=kind == "pointwise_math", bf16_policy=self.bf16_pointwise, exp_policy=self.source_exp_policy, rne_add_policy=self.qualified_rne_add if region.nodes[-1].op in {"f32_add", "bf16_add"} else None)
            elif kind == "rmsnorm":
                from .rmsnorm import prepare_source

                prepare_source(region)
            elif kind == "reduction":
                from .reduction import prepare_source

                prepare_source(region, reduction_policy=self.reduction_policy)
            elif kind == "integer":
                _integer_c(region, "integer")
            elif kind == "float_cast":
                _float_cast_c(region, "float_cast")
            elif kind == "float_arithmetic":
                _float_arithmetic_c(region, "float_arithmetic")
            elif kind == "host_k1_contraction":
                _host_k1_c(region, "host_k1_contraction")
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
                    if kind in {"layout", "predicate", "integer", "float_cast", "float_arithmetic", "unary_math", "static_quantize", "dynamic_quantize", "host_bf16_linear", "host_k1_contraction"}
                    else "accelerator",
                    CycleInterval.unknown("application region costs have not been calibrated"),
                    ports(node.id for node in region.nodes if node.effect in {"input", "constant"}),
                    ports(region.outputs),
                    metadata=tuple(sorted(
                        (("kind", kind),) + ((("epilogue", epilogue),) if epilogue else ())
                        + ((("application_source_digest", source.digest()),)
                           if kind == "host_bf16_linear" else ())
                        + ((("host_preparation", host_preparation_receipt),)
                           if all(key in host_preparation for key in keys) else ())
                    )),
                )
            )
            claimed.update(keys)

        for region in source_bf16_linear.values():
            add(region.computations, "host_bf16_linear")
        for node in pure:
            if node.id in host_preparation and node.id not in claimed:
                add((node.id,), host_preparation[node.id])
        if self.rmsnorm:
            from merlin.semantic_compiler.qualified_rmsnorm_source import recognize_rmsnorm

            for node in pure:
                if node.op != "f32_mul" or node.id in claimed or node.id in fused_nodes:
                    continue
                try:
                    norm = recognize_rmsnorm(source, node.id)
                except (KeyError, ValueError):
                    continue
                add(norm.computations, "rmsnorm")
        for region in fused_norms.values():
            add(region.computations, "hierarchical_rmsnorm")
        for region in fused_values.values():
            add(region.computations, "softmax_value")
        for node in pure:
            if node.id in self.mxu1_contraction_schedules:
                add((node.id,), "mxu1_contraction")
        if self.already_quantized:
            for node in pure:
                if node.op == "atlas.qconv2d":
                    raise NativeCompilationError("unsupported_semantics", "already-quantized qconv2d intake is unsupported")
                if node.op == "atlas.quantize_act.codes":
                    peers = [item for item in pure if item.op == "atlas.quantize_act.exponent"
                             and item.inputs == node.inputs and item.attrs == node.attrs]
                    if len(peers) != 1:
                        raise NativeCompilationError("unsupported_semantics", "static activation quantizer exponent projection differs")
                    keys = (node.id, peers[0].id)
                    aligned = self._is_vector_quantizer(source, keys)
                    kind = ("activation_quantize_device" if self.activation_quantizer_route == "device" and aligned
                            else "static_quantize" if dict(node.attrs).get("mode") == "static" else "dynamic_quantize")
                    add(keys, kind)
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
                    and producer.id not in host_k1
                    and len(producer.type.shape) == 2
                    and producer.id not in claimed
                    and producer.id not in source.outputs
                    and consumers == [node.id]
                ):
                    add((producer.id, node.id), "contraction", "relu")
        for node in pure:
            if node.id in claimed:
                continue
            if self.already_quantized and node.op in {"atlas.qlinear", "atlas.qk_matmul", "atlas.pv_matmul"}:
                add((node.id,), "already_quantized")
            elif self.host_bf16_linear_policy and node.op == "source_bf16_matmul_unqualified":
                add((node.id,), "host_bf16_linear")
            elif self.source_exp_policy and node.op == "source_f32_exp_unqualified":
                add((node.id,), "pointwise_math")
            elif node.id in host_k1:
                add((node.id,), "host_k1_contraction")
            elif node.op == "source_f32_matmul_unqualified":
                add((node.id,), "contraction")
            elif node.op == "gather" and self._is_scalar_broadcast(source, node):
                add((node.id,), "scalar_broadcast")
            elif self.already_quantized and node.op in self.host_layouts and self._supports_device_source(source, node, "layout_device"):
                add((node.id,), "layout_device")
            elif node.op in self.host_layouts:
                add((node.id,), "layout")
            elif node.op in self.host_integer_ops:
                add((node.id,), "integer")
            elif (self.host_preparation_scope is None and self.host_unary_math
                  and node.op in self.host_unary_math["operations"]):
                add((node.id,), "unary_math")
            elif self.already_quantized and node.op in (*self.host_float_arithmetic, *self.host_float_casts) and self._supports_device_source(source, node, "exact_float_device"):
                add((node.id,), "exact_float_device")
            elif self.host_preparation_scope is None and node.op in self.host_float_arithmetic:
                add((node.id,), "float_arithmetic")
            elif node.op in self.host_float_casts:
                add((node.id,), "float_cast")
            elif node.op in self.host_predicates:
                add((node.id,), "predicate")
            elif self.rne_add and node.op == "f32_add":
                add((node.id,), "rne_add")
            elif (
                self.pointwise
                and (node.op in {"f32_mul", "f32_add", "f32_sub"} or is_source_relu(node))
                or self.bf16_pointwise
                and node.op == "bf16_mul"
                or self.qualified_rne_add and node.op in {"f32_add", "bf16_add"}
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

    @staticmethod
    def _is_vector_quantizer(source, keys):
        from math import prod
        from .vector_quantize import prepare_plan
        from .activation_quantize_application import prepare_plan as schedule

        try:
            region = extract_semantic_region(source, keys)
            scheduled = schedule(region)
            plans = [prepare_plan(body) for body in scheduled.bodies]
            # Each body stages repeated authored exponents in one FP8 slot.
            # Larger exponent rows keep the explicit host route selectable.
            if any(prod(plan.exponents.type.shape) > 256 for plan in plans):
                return False
        except ValueError:
            return False
        return all(plan.width == 2 and not any(n % 32 for n in
                   (plan.count, plan.group_elements, plan.head_elements)) for plan in plans)

    def _is_host_k1(self, source, node):
        # An explicit device precision schedule retains ownership of its source.
        if ("f32_mul" not in self.host_float_arithmetic or self.contraction_schedules is not None
                or node.id in self.mxu1_contraction_schedules or node.id in self.softmax_value_schedules):
            return False
        from .host_k1_contraction import prepare_source

        try:
            prepare_source(extract_semantic_region(source, (node.id,)))
        except ValueError:
            return False
        return True

    def _host_unary_preparation(self, source, host_k1, *, excluded):
        if not self.host_unary_math:
            return {}, ""
        from .host_unary_preparation import POLICY, select

        roots = tuple(node.id for node in source.nodes if node.effect == "pure"
                      and node.op in self.host_unary_math["operations"])
        def host_kind(node):
            if node.id in host_k1:
                kind, emitter = "host_k1_contraction", _host_k1_c
            elif node.op in self.host_layouts:
                kind, emitter = "layout", emit_layout_c
            elif node.op in self.host_integer_ops:
                kind, emitter = "integer", _integer_c
            elif node.op in self.host_float_arithmetic:
                kind, emitter = "float_arithmetic", _float_arithmetic_c
            elif node.op in self.host_float_casts:
                kind, emitter = "float_cast", _float_cast_c
            elif node.op in self.host_unary_math["operations"]:
                kind = "unary_math"
                emitter = lambda region, name: _unary_math_c(region, name, self.host_unary_math)
            else:
                return None
            try:
                emitter(extract_semantic_region(source, (node.id,)), "host_preparation")
            except (ValueError, NativeCompilationError):
                return None
            return kind

        boundaries = None
        scope_receipt = {}
        if self.host_preparation_scope is not None:
            boundaries = {node.id for node in source.nodes if node.effect == "input"
                          and node.type.dtype.startswith(("i", "u"))
                          and node.type.dtype[1:].isdigit()}
            for entry in self.host_preparation_scope["constant_inputs"]:
                try:
                    node = source.node(entry["node_id"])
                except KeyError as error:
                    raise NativeCompilationError("unsupported_semantics", "host preparation constant input is absent") from error
                if node.effect != "input" or node.type.record() != entry["type"]:
                    raise NativeCompilationError("unsupported_semantics", "host preparation constant input source type differs")
                boundaries.add(node.id)
            scope_receipt = {
                "scope": self.host_preparation_scope,
                "constant_input_authentication": "immutable parameter binding authenticates payload bytes",
            }
        if not roots:
            return {}, ""
        selected = select(source, roots, host_kind, excluded=excluded, input_boundaries=boundaries)
        if boundaries is not None:
            used = {key for node in source.nodes if node.id in selected for key in node.inputs}
            scope_receipt["input_boundaries"] = [
                {"node_id": node.id, "type": node.type.record()}
                for node in source.nodes if node.id in boundaries and node.id in used
            ]
        receipt = json.dumps({
            "policy": POLICY, "source_request_digest": source.digest(),
            "selected_unary_nodes": [key for key in roots if key in selected],
            "source_nodes": [node.id for node in source.nodes if node.id in selected],
            "placement": "host", "arithmetic_rewrite": "none",
            "forward_boundary": "stop at operands outside preparation or supported constant-derived values",
            "qualification": "existing emitter arithmetic and selected unary library/effect requirements remain in force",
            **scope_receipt,
        }, sort_keys=True, separators=(",", ":"))
        return selected, receipt

    @staticmethod
    def _supports_device_source(source, node, kind):
        from . import layout_device, exact_float_application

        prepare = layout_device.prepare_source if kind == "layout_device" else exact_float_application.prepare_plan
        try:
            prepare(extract_semantic_region(source, (node.id,)))
        except ValueError:
            return False
        return True

    @staticmethod
    def _is_scalar_broadcast(source, node):
        from .scalar_broadcast import prepare_source

        try:
            prepare_source(extract_semantic_region(source, (node.id,)))
        except ValueError:
            return False
        return True

    def compile_region(self, snapshot, region, alternative, name, segments, target_source, limits):
        kind = dict(alternative.metadata)["kind"]
        if kind in {"rne_add", "softmax_value", "mxu1_contraction", "hierarchical_rmsnorm"}:
            module, selected = self._extra_precision(kind, region)
            module.compile_source(snapshot, region, **selected, target_source=target_source,
                                  destination=segments / name, limits=limits)
        elif kind in {"vector_quantize", "layout_device", "exact_float_device"}:
            from . import vector_quantize, layout_device, exact_float_application

            module = {"vector_quantize": vector_quantize, "layout_device": layout_device,
                      "exact_float_device": exact_float_application}[kind]
            module.compile_source(snapshot, region, target_source=target_source,
                                  destination=segments / name, limits=limits)
        elif kind == "activation_quantize_device":
            from .activation_quantize_application import compile_source

            compile_source(snapshot, region, target_source=target_source,
                           destination=segments / name, limits=limits)
        elif kind == "scalar_broadcast":
            from .scalar_broadcast import compile_source

            compile_source(snapshot, region, target_source=target_source,
                           destination=segments / name, limits=limits)
        elif kind == "already_quantized":
            from .quantized_loop import compile_source

            compile_source(snapshot, region, target_source=target_source,
                           destination=segments / name, limits=limits)
        elif kind == "contraction":
            from .looped_tiling import compile_looped, prepare_execution_plan

            precision = prepare_contraction(
                region,
                operand_scale_exponents=self._contraction_exponents(region),
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
                region, allow_masked=kind != "softmax", allow_guarded=kind == "guarded_softmax",
                score_domain=self.already_quantized_score_domain,
            )
            compile_row_loop(
                snapshot,
                precision.request,
                target_source=target_source,
                destination=segments / name,
                limits=limits,
            )
        elif kind == "rmsnorm":
            from .rmsnorm import prepare_source
            from .row_loop import compile_row_loop

            precision = prepare_source(region)
            compile_row_loop(snapshot, precision.request, target_source=target_source,
                             destination=segments / name, limits=limits)
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
                    {"allow_math": kind == "pointwise_math", "bf16_policy": self.bf16_pointwise, "exp_policy": self.source_exp_policy, "rne_add_policy": self.qualified_rne_add if region.nodes[-1].op in {"f32_add", "bf16_add"} else None}
                    if kind != "reduction"
                    else {"reduction_policy": self.reduction_policy}
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
        if kind == "host_bf16_linear":
            from merlin.compile.c_bf16_linear import emit_c

            record, files = emit_c(region, name, policy=self.host_bf16_linear_policy)
            if any(node.op == "source_f32_matmul_unqualified" for node in region.nodes):
                from .source_bf16_linear import recognize_source_bf16_linear

                admitted = recognize_source_bf16_linear(region, region.outputs[0])
                record["source_to_replacement_receipt"] = admitted.receipt(
                    region, original_source_sha256=self.original_source_sha256,
                    application_source_digest=dict(alternative.metadata)["application_source_digest"])
            return CompiledCRegion(name, name + ".h", tuple(record["inputs"]), region.outputs,
                                   {key: value.encode() if isinstance(value, str) else value
                                    for key, value in files.items()}, (), (), record)
        if kind == "rmsnorm":
            from .rmsnorm import prepare_source
            from .rmsnorm_c import prepare_c_package

            precision = prepare_source(region)
            record, files = prepare_c_package(precision, segments / name, name,
                                             target_source=target_source, runtime_linkage="separate")
            return CompiledCRegion(name + "_source_dispatch", name + ".h", precision.input_ids,
                                   region.outputs, files, RUNTIME_SHARED_FILES,
                                   ("segments/" + name,), record)
        if kind in {"rne_add", "softmax_value", "mxu1_contraction", "hierarchical_rmsnorm"}:
            module, selected = self._extra_precision(kind, region)
            if kind == "softmax_value":
                from . import quantized_softmax_value_c as module
            elif kind == "hierarchical_rmsnorm":
                from . import hierarchical_rmsnorm_c as module
            record, files = module.prepare_c_package(region, segments / name, name,
                **selected, target_source=target_source, runtime_linkage="separate")
            programs = ("segments/" + name,)
            if kind == "hierarchical_rmsnorm":
                paths = record.get("native_program_paths")
                if (not isinstance(paths, list) or len(paths) != 4
                        or any(not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts
                               or Path(path).as_posix() != path or path in {"", "."} for path in paths)
                        or len(set(paths)) != len(paths)):
                    raise NativeCompilationError("compile_error", "RMSNorm export needs four relative native program paths")
                programs = tuple("segments/" + name + "/" + path for path in paths)
            return CompiledCRegion(record["dispatch_function"], name + ".h",
                tuple(record["input_node_ids"]), tuple(record["output_node_ids"]), files,
                RUNTIME_SHARED_FILES, programs, {"kind": kind, **record})
        if kind in {"vector_quantize", "exact_float_device"}:
            from . import vector_quantize, exact_float_application

            module = vector_quantize if kind == "vector_quantize" else exact_float_application
            return module.prepare_c_package(region, segments / name, name, target_source=target_source)
        if kind == "layout_device":
            from .layout_device import prepare_c_package

            record, files = prepare_c_package(region, segments / name, name, target_source=target_source)
            paths = record.get("native_program_paths")
            programs = (("segments/" + name,) if paths is None else
                        tuple("segments/" + name + "/" + path for path in paths))
            return CompiledCRegion(name + "_source_dispatch", name + ".h",
                tuple(record["input_node_ids"]), region.outputs, files, RUNTIME_SHARED_FILES,
                programs, record)
        if kind == "activation_quantize_device":
            from .activation_quantize_application import prepare_c_package

            return prepare_c_package(region, segments / name, name, target_source=target_source)
        if kind == "scalar_broadcast":
            from .scalar_broadcast import prepare_c_package

            record, files = prepare_c_package(region, segments / name, name,
                                              target_source=target_source)
            return CompiledCRegion(name + "_source_dispatch", name + ".h",
                tuple(record["input_node_ids"]), region.outputs, files, RUNTIME_SHARED_FILES,
                ("segments/" + name,), record)
        if kind == "already_quantized":
            from .quantized_loop_c import prepare_c_package

            record, files = prepare_c_package(region, segments / name, name, target_source=target_source)
            paths = record.get("native_program_paths")
            programs = (("segments/" + name,) if paths is None else
                        tuple("segments/" + name + "/" + path for path in paths))
            return CompiledCRegion(name + "_source_dispatch", name + ".h",
                tuple(record["input_node_ids"]), region.outputs, files, RUNTIME_SHARED_FILES,
                programs, record)
        if kind in {"static_quantize", "dynamic_quantize"}:
            from . import dynamic_quantize, static_quantize

            record, files = (dynamic_quantize if kind == "dynamic_quantize" else static_quantize).emit_c(region, name)
            return CompiledCRegion(name, name + ".h", tuple(record["inputs"]), region.outputs,
                                   files, (), (), record)
        if kind in {"layout", "predicate", "integer", "float_cast", "float_arithmetic", "unary_math", "host_k1_contraction"}:
            if kind == "host_k1_contraction":
                emitter = _host_k1_c
            elif kind == "float_arithmetic":
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
            placement_receipt = dict(alternative.metadata).get("host_preparation")
            if placement_receipt is not None:
                record["host_preparation"] = json.loads(placement_receipt)
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
                    {"allow_math": kind == "pointwise_math", "bf16_policy": self.bf16_pointwise, "exp_policy": self.source_exp_policy, "rne_add_policy": self.qualified_rne_add if region.nodes[-1].op in {"f32_add", "bf16_add"} else None}
                    if kind != "reduction"
                    else {"reduction_policy": self.reduction_policy}
                ),
            )
            return CompiledCRegion(
                name + "_source_dispatch",
                name + ".h",
                tuple(record["input_node_ids"]),
                tuple(record["output_node_ids"]),
                files,
                RUNTIME_SHARED_FILES,
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
                operand_scale_exponents=self._contraction_exponents(region),
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
                    semantic_operand_scale_exponents=self._contraction_exponents(region),
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
                    operand_scale_exponents=self._contraction_exponents(region),
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
                    semantic_operand_scale_exponents=self._contraction_exponents(region),
                    semantic_epilogue=dict(alternative.metadata).get("epilogue"),
                )
                inputs = tuple(region.nodes[item.source_value.index].id for item in precision.inputs)
            programs = chunked_programs or ("segments/" + name,)
            interface = record.get("batch_interface", record["source_region"]["source_interface"])
        elif kind in {"softmax", "masked_softmax", "guarded_softmax"}:
            from .softmax_c import _prepare_c_package

            precision = lower_semantic_softmax(
                region, allow_masked=kind != "softmax", allow_guarded=kind == "guarded_softmax",
                score_domain=self.already_quantized_score_domain,
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
                finite_sentinel_bits=self.already_quantized_mask_sentinel if kind == "masked_softmax" else None,
                score_domain=self.already_quantized_score_domain,
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
            RUNTIME_SHARED_FILES,
            programs,
            description,
        )
