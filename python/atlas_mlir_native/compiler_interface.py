"""Limited implementation claims for checked quantized and CNN applications.

This is deliberately not an exhaustive description of AtlasApplicationAdapter.
Only a small set of routes used by that application are declared. Source recognizers,
selected numerical policies and native publication checks remain authoritative.
The source digest pins Python emitters and runtime source, not installed native
compiler binaries, build options, external toolchains or hardware facts.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path

from merlin.compiler.backend.interface import (
    CompilerDescription, ImplementationFamily, SourceDomain, TargetPipeline,
)
from merlin.compiler.backend.protocol import LoweringRecipe, SourcePattern

_PROVIDER = "atlas"
_ABI = "atlas.compiled-c-region.v1"
_ARTIFACTS = (("c-source", "wrapper"), ("binary", "device-program"))
# A kind names a route, never a model, node identity, or shape whitelist.
_FAMILIES = (
    ("activation-quantize", "activation_quantize_device",
     ("atlas.quantize_act.codes", "atlas.quantize_act.exponent"),
     "closed paired code/exponent source quantizer; unchanged authored reduction groups and exact selected encoder semantics"),
    ("host-static-quantize", "static_quantize",
     ("atlas.quantize_act.codes", "atlas.quantize_act.exponent"),
     "closed paired static code/exponent quantizer; original checked host encoder and unchanged authored signed exponent"),
    ("host-static-quantize-f32", "static_quantize",
     ("atlas.quantize_act.codes", "atlas.quantize_act.exponent"),
     "closed paired static quantizer of an f32 source; the authored binary32 to BF16 RNE cast, then the checked host "
     "encoder and unchanged authored signed exponent"),
    ("ordered-quantized-contraction", "already_quantized",
     ("atlas.qlinear", "atlas.qk_matmul", "atlas.pv_matmul"),
     "unchanged cap448 FP8 codes and signed i32 exponents; ordered MXU1 K32 reset/continuation and checked BF16 restoration"),
    ("ordered-quantized-contraction-f32", "already_quantized", ("atlas.qlinear", "atlas.qk_matmul", "atlas.pv_matmul"),
     "unchanged cap448 FP8 codes and signed i32 exponents; ordered MXU1 K32 reset/continuation, checked BF16 "
     "restoration, and exact BF16-to-f32 widening of the restored result"),
    ("device-layout", "layout_device", ("reshape", "copy", "gather", "transpose", "concat"),
     "one closed dense raw-byte layout; existing affine bounds, device width/address and publication checks"),
    ("host-layout", "layout", ("reshape", "copy", "gather", "transpose", "concat", "insert_slice", "identity"),
     "one closed dense byte-preserving layout admitted by the existing host emitter"),
    ("layout", "layout_device", ("reshape", "copy", "gather", "transpose", "concat", "insert_slice", "identity"),
     "one closed dense raw-byte layout on the device where its existing checks admit it beside a quantized contraction, "
     "otherwise the checked host layout"),
    ("host-bias-add", "float_arithmetic", ("bf16_add",),
     "explicitly selected host BF16 bias addition; original BF16 RNE and disjoint dense buffers"),
    ("host-ordered-select", "host_ordered_relu", ("relu",),
     "BF16 ordered x < +0 select; all BF16 bit patterns; preserve signed zero and NaN payloads"),
    ("host-window-max", "host_pointwise", ("bf16_window_max",),
     "ordered BF16 NCHW window maximum; non-NaN inputs, first-equal tie, checked strided/dilated/padded maps"),
    ("host-batch-norm", "host_pointwise", ("bf16_batch_norm_inference",),
     "checked BF16 inference normalization with exact declared binary32 channel steps and fused element update"),
    ("host-sigmoid", "host_pointwise", ("bf16_sigmoid",),
     "checked source expf/add/divide/BF16-RNE order; pinned host supplier and exact receipt qualification obligation"),
    ("host-gelu", "host_pointwise", ("bf16_gelu",),
     "checked source binary32 tanh or erf GELU order; pinned correctly rounded host supplier; finite BF16 inputs"),
    ("host-layer-norm", "host_pointwise", ("bf16_layer_norm",),
     "checked pinned torch 2.11.0+cpu AVX2/AVX512 layer norm kernel (chunked Welford moments); finite inputs and results"),
    ("host-f32-silu", "host_pointwise", ("f32_silu",),
     "checked source f32 SiLU in the pinned torch 2.11.0+cpu vector arithmetic, x/(1+expf(-x)); finite inputs"),
    ("host-f32-rsqrt", "host_pointwise", ("source_f32_rsqrt_unqualified",),
     "checked source f32 rsqrt operation as two binary32 roundings, 1/sqrt(x); finite nonnegative inputs"),
    ("host-f32-sum", "host_pointwise", ("source_f32_reduce_add_unqualified",),
     "checked +0-seeded sum of a source f32 mean over the last axis in the declared order torch-2.11-cpu-f32-sum-v1; "
     "not torch eager for a noncontiguous operand; finite inputs and results; one row of 32,768 or more elements and "
     "reductions lowered from other source operations stay refused"),
    ("host-f32-softmax", "host_pointwise", ("source_f32_softmax_unqualified",),
     "checked source f32 softmax over the last axis in the declared order host-softmax-f32-left-fold-div-rne-v1 "
     "(left-fold maximum and sum, pinned correctly rounded expf, one division); not torch eager's vector order"),
    ("host-bf16-contraction", "host_pointwise", ("bf16_affine_contraction",),
     "checked BF16 contraction over one affine loop domain (unfolded convolution); binary32 products (exact unless underflow), ascending binary32 sums"),
    ("host-sign-flip", "host_pointwise", ("bf16_neg",),
     "exact BF16 sign-bit flip for every bit pattern, including signed zero and NaN payloads"),
    ("exact-bf16-add", "exact_float_device", ("bf16_add",),
     "source BF16 RNE addition excluding prepared contraction bias; checked exact-device vector/pair/scalar routes "
     "where the closed region is admitted, otherwise the checked host adder"),
    ("host-f32-add", "float_arithmetic", ("f32_add",),
     "source f32 add on the checked host emitter; binary32 RNE in source operand order"),
    ("host-f32-sub", "float_arithmetic", ("f32_sub",),
     "source f32 sub on the checked host emitter; binary32 RNE in source operand order"),
    ("host-f32-mul", "float_arithmetic", ("f32_mul",),
     "source f32 mul on the checked host emitter; binary32 RNE in source operand order"),
    ("host-f32-div", "float_arithmetic", ("f32_div",),
     "source f32 div on the checked host emitter; binary32 RNE in source operand order"),
    ("exact-bf16-to-f32", "exact_float_device", ("bf16_to_f32",),
     "source exact widening on the exact device conversion route where the closed region is admitted, otherwise the "
     "checked host cast"),
    ("exact-f32-to-bf16", "exact_float_device", ("f32_to_bf16",),
     "source BF16 RNE narrowing on the exact device conversion route where the closed region is admitted, otherwise the "
     "checked host cast"),
    ("device-bf16-mul", "pointwise", ("bf16_mul",),
     "source BF16 RNE multiplication on the device VPU; operands admitted only in the bounded normal-or-zero code domain "
     "whose products stay normal and finite; runtime domain scan refuses others"),
)
# The single declaration of each family's second existing route, used by the planner only when the first refuses
# its closed region. An alternate is a host route of the same source operation under the same numerical contract;
# a family with a precision change never has one (describe_numerical_families refuses it).
_ALTERNATE_ROUTES = (("exact-bf16-add", "float_arithmetic", ("bf16_add",)),
                     ("layout", "layout", ("reshape", "copy", "gather", "transpose", "concat", "insert_slice", "identity")),
                     ("activation-quantize", "static_quantize", ("atlas.quantize_act.codes", "atlas.quantize_act.exponent")),
                     ("activation-quantize", "dynamic_quantize", ("atlas.quantize_act.codes", "atlas.quantize_act.exponent")),
                     ("exact-bf16-to-f32", "float_cast", ("bf16_to_f32",)),
                     ("exact-f32-to-bf16", "float_cast", ("f32_to_bf16",)))


# These share a kind with another family, so only a generated claim can name them.
_CLAIM_ONLY = {"host-static-quantize-f32", "layout"}


def _routes(selected_family):
    """Declared routes; an alternate counts only for the family a generated cover explicitly selected."""
    return tuple((family, kind, operations) for family, kind, operations, _ in _FAMILIES
                 if family not in _CLAIM_ONLY or selected_family == "atlas." + family) + tuple(
        row for row in _ALTERNATE_ROUTES if selected_family == "atlas." + row[0])


def alternate_kinds(family_id):
    """Declared alternate route kinds of one family identity; empty when it has none."""
    return tuple(kind for family, kind, _ in _ALTERNATE_ROUTES if family_id == "atlas." + family)


def describe_compiler():
    patterns = tuple(SourcePattern("atlas." + family, operations, "atlas.application", 1)
                     for family, _, operations, _ in _FAMILIES)
    return CompilerDescription(
        _PROVIDER,
        SourceDomain("merlin.semantic", tuple(sorted({op for _, _, ops, _ in _FAMILIES for op in ops})),
                     ("i8", "i32", "i64", "bf16", "f32", "fp8_e4m3"),
                     (("scope", "limited quantized/CNN adapter families; no source model matching; other kinds remain undeclared"),
                      ("admission", "original closed-region recognizers and explicitly selected numerical policy"),
                      ("implementation-pin", "Python emitters and runtime source only; native tool binaries and build configuration unpinned"))),
        patterns,
        tuple(ImplementationFamily("atlas." + family, ("atlas." + family,), "atlas.application",
                                   (("admission", premise),))
              for family, _, _, premise in _FAMILIES),
        (TargetPipeline("atlas.application", "merlin.semantic", "atlas.program", _ARTIFACTS, _ABI,
                        ("source-admission", "selected-region-lowering", "atlas-emission", "checked-c-export")),),
    )


def describe_numerical_families():
    """Query numerical extensions of this description's implementation claims."""
    from .numerical_contracts import describe_numerical_families as query

    rows = query()
    changed = {row.family for row in rows if row.precision_change is not None}
    if any("atlas." + family in changed for family, _, _ in _ALTERNATE_ROUTES):
        raise ValueError("a precision-changing Atlas family cannot declare an alternate route")
    return rows


def _source_roots():
    """Explicit shallow roots: no repository-wide or hardware traversal."""
    from merlin.compiler.lower import c as host_c
    from merlin.runtime import host_math_supplier

    atlas = Path(__file__).resolve().parents[2]
    roots = [("atlas-python", Path(__file__).parent, (".py", ".json")),
             ("merlin-host-c", Path(host_c.__file__).parent, (".py",)),
             ("merlin-host-ops", Path(host_c.__file__).parent / "host_ops", (".py",)),
             ("merlin-host-math", host_math_supplier.ROOT, (".py", ".c", ".h", ".json")),
             ("merlin-host-math-core", host_math_supplier.ROOT / "third_party/core-math", (".c", ".h")),
             ("merlin-host-math-pow", host_math_supplier.ROOT / "third_party/core-math/pow", (".c", ".h"))]
    runtime = atlas / "runtime"
    if runtime.is_dir():
        roots.append(("atlas-runtime", runtime, (".c", ".h")))
    return tuple(roots)


def implementation_sha256():
    """Hash sorted, named direct Python/runtime sources, without executing them.

    No recursion or symlink following is needed. Changes to Merlin's actual host
    emitters are included alongside Atlas's wrappers. This is a source pin, not
    a claim to identify the separately built atlas-opt/atlas-emit binaries.
    """
    digest = hashlib.sha256()
    for label, root, suffixes in sorted(_source_roots()):
        for path in sorted(root.iterdir()):
            if (not path.is_symlink() and path.is_file()
                    and path.suffix in suffixes):
                digest.update(f"{label}/{path.name}".encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _guards(kind, numerics=None, family_id=None):
    if kind == "already_quantized" and family_id is None and numerics is not None:
        # The contraction family follows its declared output stage.
        family_id = "atlas.ordered-quantized-contraction" + ("-f32" if "exact-widen-f32" in numerics.stages else "")
    matches = [family for family, selected, _ in _routes(family_id)
               if selected == kind and (family_id is None or "atlas." + family == family_id)]
    if len(matches) != 1:
        raise ValueError(f"Atlas has no declared implementation family for {kind}")
    family = matches[0]
    numerical_guard = ()
    if kind == "already_quantized":
        from .fp8_contract import atlas_numerics, OPS
        from merlin.compiler.quantization.contraction.scaled_fp8 import require_same_numerics
        numerics = numerics or atlas_numerics(OPS[0])
        dtype = "f32" if "exact-widen-f32" in numerics.stages else "bf16"
        require_same_numerics(atlas_numerics(OPS[0], output_dtype=dtype), numerics)
        numerical_guard = (("numerical.contraction", json.dumps(numerics.record(), sort_keys=True, separators=(",", ":"))),)
    return numerical_guard + (("family", "atlas." + family), ("selected-kind", kind),
            ("source", "existing structural region admission; declared operation subset only"),
            ("numerics", "explicit selected policy and runtime domain checks"))


def _artifacts(kind):
    return _ARTIFACTS[:1] if kind in {"layout", "float_arithmetic", "host_ordered_relu", "host_pointwise", "static_quantize",
                                      "dynamic_quantize"} else _ARTIFACTS


def validate_recipe(recipe):
    """Reject stale source pins and undeclared routes in an Atlas recipe."""
    describe_compiler().validate_recipe(recipe)
    kind = dict(recipe.guards).get("selected-kind")
    numerics = None
    if kind == "already_quantized":
        from merlin.compiler.quantization.contraction.scaled_fp8 import ContractionNumerics
        raw = dict(recipe.guards).get("numerical.contraction")
        if not isinstance(raw, str):
            raise ValueError("Atlas contraction recipe needs its numerical contract")
        numerics = ContractionNumerics.from_record(json.loads(raw))
    try:
        declared_guards = _guards(kind, numerics, dict(recipe.guards).get("family"))
    except ValueError as error:
        raise ValueError("Atlas lowering recipe guards differ from the declared route") from error
    if recipe.guards != declared_guards or recipe.artifacts != _artifacts(kind):
        raise ValueError("Atlas lowering recipe guards differ from the declared route")
    if recipe.implementation_sha256 != implementation_sha256():
        raise ValueError("Atlas lowering implementation sources changed; rediscover proposals")


def lowering_recipe(alternative, *, numerics=None, family_id=None):
    kind = dict(alternative.metadata).get("kind")
    guards = _guards(kind, numerics, family_id or dict(alternative.metadata).get("numerical_family"))
    recipe = LoweringRecipe(implementation_sha256(), "atlas.application", guards, _artifacts(kind), _ABI)
    validate_recipe(recipe)
    return recipe


def record_compiler_recipe(export_region):
    """Attach a recipe to exported regions within this limited declaration.

    Existing adapter routes outside the declared kind/operation subset continue
    without a compiler-interface claim. The wrapped export still performs all
    original source, numerical and publication checks before any claim is made.
    """
    @wraps(export_region)
    def described_export(self, region, alternative, *args, **kwargs):
        product = export_region(self, region, alternative, *args, **kwargs)
        recipe = region_compiler_recipe(region, alternative)
        if recipe is None:
            return product
        return replace(product, description={**product.description, "compiler_recipe": recipe})
    return described_export


def region_compiler_recipe(region, alternative):
    """The recipe record an exported region carries, or None outside the declared kind and operation subset."""
    kind = dict(alternative.metadata).get("kind")
    pure = [node.op for node in region.nodes if node.effect == "pure"]
    selected_family = dict(alternative.metadata).get("numerical_family")
    if kind == "already_quantized" and selected_family is None and region.outputs:
        # Without a generated claim, the contraction family follows the region's declared output type.
        output = next((node for node in region.nodes if node.id == region.outputs[0]), None)
        if output is not None:
            selected_family = "atlas.ordered-quantized-contraction" + ("-f32" if output.type.dtype == "f32" else "")
    matches = [("atlas." + family, ops) for family, selected, ops in _routes(selected_family)
               if selected == kind and pure and set(pure) <= set(ops)
               and (selected_family is None or "atlas." + family == selected_family)]
    types = set(describe_compiler().source_domain.types)
    if len(matches) != 1 or any(node.type.dtype not in types for node in region.nodes):
        if selected_family is not None:
            raise ValueError("selected numerical family differs from exported route or source roots")
        return None
    numerics = None
    if kind == "already_quantized":
        from .source_translation import lower_already_quantized_contraction
        numerics = lower_already_quantized_contraction(region).numerics
    recipe = lowering_recipe(alternative, numerics=numerics, family_id=matches[0][0])
    return json.loads(json.dumps(asdict(recipe)))
