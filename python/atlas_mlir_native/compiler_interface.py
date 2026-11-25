"""Limited implementation claims for checked quantized and CNN applications.

This is deliberately not an exhaustive description of AtlasApplicationAdapter.
Only a small set of routes used by that application are declared; under a
generated family cover an exported region outside them refuses. Source recognizers,
selected numerical policies and native publication checks remain authoritative.
Atlas declares its device families here; host CPU families are declared by
Merlin's host target, and the adapter names its own route for each of them.
The source digest pins Atlas's Python and runtime sources and the host target's
digest, not installed native compiler binaries, build options, external
toolchains or hardware facts.
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
from merlin.compiler.lower.c import host_target

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
    ("layout", "layout_device", ("reshape", "copy", "gather", "transpose", "concat", "insert_slice", "identity"),
     "one closed dense raw-byte layout on the device where its existing checks admit it beside a quantized contraction, "
     "otherwise the checked host layout"),
    ("exact-bf16-add", "exact_float_device", ("bf16_add",),
     "source BF16 RNE addition excluding prepared contraction bias; checked exact-device vector/pair/scalar routes "
     "where the closed region is admitted, otherwise the checked host adder"),
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
# The adapter route that runs each host target family; the family itself is Merlin's declaration.
_HOST_ROUTES = {
    "host.layout": "layout", "host.bias-add": "float_arithmetic", "host.ordered-select": "host_ordered_relu",
    **{host_target.family_id(name): "host_pointwise" for name in (
        "window-max", "batch-norm", "sigmoid", "gelu", "layer-norm", "f32-silu", "f32-rsqrt", "f32-sum",
        "f32-softmax", "bf16-contraction", "sign-flip")},
    **{host_target.family_id(f"f32-{name}"): "float_arithmetic" for name in ("add", "sub", "mul", "div")},
}
# Family identity to its first route kind; generated policy admission reads this, not a copy.
FAMILY_KINDS = {**{f"{_PROVIDER}.{family}": kind for family, kind, _, _ in _FAMILIES}, **_HOST_ROUTES}
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
    """Declared routes by family identity; an alternate counts only for the family a generated cover selected."""
    host = host_target.family_operations()
    return tuple(("atlas." + family, kind, operations) for family, kind, operations, _ in _FAMILIES
                 if family not in _CLAIM_ONLY or selected_family == "atlas." + family) + tuple(
        ("atlas." + family, kind, operations) for family, kind, operations in _ALTERNATE_ROUTES
        if selected_family == "atlas." + family) + tuple(
        (family, kind, host[family]) for family, kind in _HOST_ROUTES.items())


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
                     (("scope", "limited quantized/CNN adapter families; no source model matching; other kinds refuse at export"),
                      ("admission", "original closed-region recognizers and explicitly selected numerical policy"),
                      ("implementation-pin", "Python emitters and runtime source only; native tool binaries and build configuration unpinned"))),
        patterns,
        tuple(ImplementationFamily("atlas." + family, ("atlas." + family,), "atlas.application",
                                   (("admission", premise),))
              for family, _, _, premise in _FAMILIES),
        (TargetPipeline("atlas.application", "merlin.semantic", "atlas.program", _ARTIFACTS, _ABI,
                        ("source-admission", "selected-region-lowering", "atlas-emission", "checked-c-export")),),
    )


def describe_targets():
    """The descriptions an Atlas application selects families from: this device and Merlin's host target."""
    return describe_compiler(), host_target.describe_compiler()


def describe_target_families():
    """Numerical families of describe_targets(), in the same order."""
    return describe_numerical_families() + host_target.describe_numerical_families()


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
    atlas = Path(__file__).resolve().parents[2]
    roots = [("atlas-python", Path(__file__).parent, (".py", ".json"))]
    runtime = atlas / "runtime"
    if runtime.is_dir():
        roots.append(("atlas-runtime", runtime, (".c", ".h")))
    return tuple(roots)


def implementation_sha256():
    """Hash sorted, named direct Python/runtime sources, without executing them.

    No recursion or symlink following is needed. The host target's own source
    digest is included because Atlas routes call its emitters. This is a source
    pin, not a claim to identify the separately built atlas-opt/atlas-emit binaries.
    """
    digest = hashlib.sha256()
    for label, root, suffixes in sorted(_source_roots()):
        for path in sorted(root.iterdir()):
            if (not path.is_symlink() and path.is_file()
                    and path.suffix in suffixes):
                digest.update(f"{label}/{path.name}".encode() + b"\0" + path.read_bytes() + b"\0")
    digest.update(b"host-target\0" + host_target.implementation_sha256().encode() + b"\0")
    return digest.hexdigest()


def _guards(kind, numerics=None, family_id=None):
    if kind == "already_quantized" and family_id is None and numerics is not None:
        # The contraction family follows its declared output stage.
        family_id = "atlas.ordered-quantized-contraction" + ("-f32" if "exact-widen-f32" in numerics.stages else "")
    matches = [family for family, selected, _ in _routes(family_id)
               if selected == kind and (family_id is None or family == family_id)]
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
    return numerical_guard + (("family", family), ("selected-kind", kind),
            ("source", "existing structural region admission; declared operation subset only"),
            ("numerics", "explicit selected policy and runtime domain checks"))


def _artifacts(kind):
    return _ARTIFACTS[:1] if kind in {"layout", "float_arithmetic", "host_ordered_relu", "host_pointwise", "static_quantize",
                                      "dynamic_quantize"} else _ARTIFACTS


def validate_recipe(recipe):
    """Reject stale source pins and undeclared routes in an Atlas or host family recipe."""
    host = recipe.pipeline == host_target.PIPELINE
    if host:
        host_target.validate_recipe(recipe)
    else:
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
    if (recipe.guards != declared_guards or recipe.artifacts != _artifacts(kind)
            or host != (dict(declared_guards)["family"] in _HOST_ROUTES)):
        raise ValueError("Atlas lowering recipe guards differ from the declared route")
    if not host and recipe.implementation_sha256 != implementation_sha256():
        raise ValueError("Atlas lowering implementation sources changed; rediscover proposals")


def lowering_recipe(alternative, *, numerics=None, family_id=None):
    kind = dict(alternative.metadata).get("kind")
    guards = _guards(kind, numerics, family_id or dict(alternative.metadata).get("numerical_family"))
    recipe = (host_target.lowering_recipe(guards) if dict(guards)["family"] in _HOST_ROUTES
              else LoweringRecipe(implementation_sha256(), "atlas.application", guards, _artifacts(kind), _ABI))
    validate_recipe(recipe)
    return recipe


# Temporary: the route kinds an adapter built from a legacy hand-written policy may still export without a claim.
# The general replacement is the generated family cover. This set may only shrink, and it goes when legacy policy
# decoding is removed; test_compiler_interface pins it.
LEGACY_UNCLAIMED_KINDS = frozenset({
    "activation_quantize_device",
    "already_quantized",
    "contraction",
    "dynamic_quantize",
    "exact_float_device",
    "float_arithmetic",
    "float_cast",
    "guarded_softmax",
    "hierarchical_rmsnorm",
    "host_bf16_linear",
    "host_k1_contraction",
    "host_ordered_relu",
    "host_pointwise",
    "host_softmax",
    "integer",
    "layout",
    "layout_device",
    "masked_softmax",
    "mxu1_contraction",
    "pointwise",
    "pointwise_math",
    "predicate",
    "reduction",
    "rmsnorm",
    "rne_add",
    "scalar_broadcast",
    "softmax",
    "softmax_value",
    "static_quantize",
    "unary_math",
})


def record_compiler_recipe(export_region):
    """Attach a recipe to every exported region; a generated cover refuses a route this declaration does not claim.

    Temporary: an adapter built from a legacy hand-written policy still exports
    such a route of a kind in LEGACY_UNCLAIMED_KINDS without a claim. The wrapped
    export still performs all original source, numerical and publication checks
    before any claim is made.
    """
    @wraps(export_region)
    def described_export(self, region, alternative, *args, **kwargs):
        product = export_region(self, region, alternative, *args, **kwargs)
        recipe = region_compiler_recipe(region, alternative)
        kind = dict(alternative.metadata).get("kind")
        if recipe is None and getattr(self, "_generated_family_claims", None) is None and kind in LEGACY_UNCLAIMED_KINDS:
            return product
        if recipe is None:
            from merlin.compiler.semantic.target_binding import NativeCompilationError

            operations = sorted({node.op for node in region.nodes if node.effect == "pure"})
            raise NativeCompilationError(
                "unsupported_semantics",
                f"Atlas declares no implementation family for the {kind} route "
                f"over {', '.join(operations) or 'no operation'}")
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
    matches = [(family, ops) for family, selected, ops in _routes(selected_family)
               if selected == kind and pure and set(pure) <= set(ops)
               and (selected_family is None or family == selected_family)]
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
