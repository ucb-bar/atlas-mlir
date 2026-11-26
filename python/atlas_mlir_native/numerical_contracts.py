"""Numerical declarations for the limited Atlas compiler interface.

These records describe existing implementations. Merlin still requires explicit
precision authorization, and the original closed-region recognizers, runtime
checks and publication checks remain mandatory. This is not a replacement for
export authentication or policy_derivation, nor an exhaustive adapter catalogue.
"""
from __future__ import annotations

import json

from merlin.compiler.quantization.policy_generation import NumericalFamily

# The exporter-to-MXU mapping is a separately selected numerical change. Raw
# code/exponent metadata does not authorize this accumulation implementation.
MXU1_PRECISION_AUTHORIZATION = "mxu1-model47-e4m3-cap448-anchor-bf16-seeded-ordered-v1"
_ALREADY_QUANTIZED = "atlas-already-quantized-e4m3-signed-exponents-mxu1-model47-v1"
_STORAGE_POLICIES = ("raw-bit-pattern", "signed-i8", "i32-wrap-k-ascending")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fields(**values):
    return tuple((key, _json(value)) for key, value in sorted(values.items()))


def _constraints(*entries):
    return tuple((path, _json(value)) for path, value in entries)


def _family(name, implementation_policy, *, constraints=(), policy_fields=(),
            precision_change=None, source_policies=_STORAGE_POLICIES, excluded_constraints=(), **contract):
    return NumericalFamily(
        "atlas." + name, "atlas." + name, source_policies, implementation_policy,
        constraints=constraints, policy_fields=policy_fields,
        precision_change=precision_change,
        numerical_contract=_fields(
            schema="atlas.numerical-family.v1",
            evidence_status="implementation_claim_pending_admission",
            source_equivalence="conditional_on_declared_source_and_runtime_domains",
            admission="existing closed-region recognizer, resource/ABI and publication checks",
            **contract,
        ),
        excluded_constraints=excluded_constraints,
    )


def describe_numerical_families() -> tuple[NumericalFamily, ...]:
    """Query typed contracts bound to the current CompilerDescription identities.

    Host and device layout families intentionally overlap. Callers must select
    family identities explicitly or retain Merlin's ambiguity refusal. Selection
    neither enables precision changes nor bypasses device layout legality.

    The contraction declaration initially covers BF16 output only. The existing
    f32 output implementation, host dynamic quantizer routes and
    broader adapter families need additional declarations before generation can
    claim their coverage. Multi-result quantizers still require paired regional
    admission; a per-node match alone cannot establish their closure.
    """
    from .compiler_interface import describe_compiler

    cap448 = {
        "storage_dtype": "i8", "byte_interpretation": "uint8 raw bits; no signed-i8 arithmetic",
        "exponent_bits": 4, "mantissa_bits": 3, "bias": 7, "normal_exponent_range": [1, 15],
        "exponent_15_finite": True, "finite_max": 448, "min_normal": 0.015625,
        "reserved_codes": [127, 255], "admitted_codes": [[0, 0], [8, 126], [136, 254]],
        "zero": "positive zero only", "subnormal_codes": "refused at contraction input",
        "negative_zero": "refused at contraction input",
    }
    rows = (
        _family(
            "ordered-quantized-contraction", _ALREADY_QUANTIZED,
            precision_change=MXU1_PRECISION_AUTHORIZATION,
            constraints=_constraints(
                (("operand_count",), 4),
                (("result", "dtype"), "bf16"),
                (("operands", "0", "dtype"), "i8"),
                (("operands", "1", "dtype"), "i32"),
                (("operands", "2", "dtype"), "i8"),
                (("operands", "3", "dtype"), "i32"),
                *(((("attributes", key), value) for key, value in {
                    "semantic_kind": "prequantized_contraction", "code_dtype": "uint8",
                    "exponent_dtype": "int32", "weight_format": "atlas_e4m3",
                    "dequantization": "e4m3(code)*2^exponent", "rhs_transposed": "true",
                    "output_rescale": "2^(lhs_exp+rhs_exp)",
                    "bias_order": "after_output_cast", "output_dtype": "bfloat16",
                }.items())),
            ),
            policy_fields=_fields(already_quantized=_ALREADY_QUANTIZED),
            operand_roles=["lhs_codes", "lhs_exponents", "rhs_codes", "rhs_exponents"],
            encoding=cap448,
            exponents={
                "storage_dtype": "i32", "signed": True, "operand_bounds": [-127, 127],
                "paired_sum_bounds": [-133, 127], "scale": "2^(lhs_exp+rhs_exp)",
                "granularity": ["scalar", "per-row", "per-head"],
                "logical_shape": "broadcast to corresponding NT operand; invariant over K",
                "layout": "preserve logical code/exponent correspondence and operand order",
                "restoration": "finite nonzero BF16 power-of-two scale; finite normal result or admitted zero",
            },
            accumulation={
                "unit": "mxu1", "model": 47, "exporter_arithmetic": "ipt",
                "policy": MXU1_PRECISION_AUTHORIZATION, "integer_anchor": True,
                "alignment": "truncating", "integer_overflow": "wrap-i32",
                "short_sum_normalization": True, "k_order": "ascending stored logical K",
                "instruction_k": 32, "first_instruction": "reset",
                "continuation": "previous BF16 readout seeds next instruction",
                "tail_padding": "positive zero", "readout": "BF16 RNE after every instruction",
            },
            boundaries={
                "sequence": ["ordered MXU1 BF16 readouts", "restore exponent sum at final readout",
                          "declared output cast", "separate source bias BF16 RNE when present"],
                "output_dtype": "bf16", "f32_extension": "undeclared here; existing route uses exact BF16 widening",
                "bias": "must be split before admission; no MXU seed bias or primitive VPU add substitution",
                "weights": "unchanged prequantized codes and exponents; no quantizer or encoder",
                "activation_quantization": "separate authored quantizer family",
            },
            runtime={
                "checks": ["all code bytes", "each signed exponent", "paired exponent sums before device IO",
                        "each instruction readout finite-normal or admitted zero", "restored result domain",
                        "device completion and status"],
                "exceptions": "NaN, infinity, reserved/subnormal FP8 and out-of-domain readouts refuse",
                "publication": "caller output unchanged on failure; publish workspace only after success",
            },
            authorization={
                "required": MXU1_PRECISION_AUTHORIZATION,
                "source_equivalence": "selected exporter ipt to MXU1 claim; no framework float dot-product proof",
                "evidence": "existing reviewed single-product proof and sampled long-K evidence; no new qualification",
            },
        ),
        _family(
            "ordered-quantized-contraction-f32", _ALREADY_QUANTIZED,
            precision_change=MXU1_PRECISION_AUTHORIZATION,
            constraints=_constraints(
                (("operand_count",), 4),
                (("result", "dtype"), "f32"),
                (("operands", "0", "dtype"), "i8"),
                (("operands", "1", "dtype"), "i32"),
                (("operands", "2", "dtype"), "i8"),
                (("operands", "3", "dtype"), "i32"),
                *(((("attributes", key), value) for key, value in {
                    "semantic_kind": "prequantized_contraction", "code_dtype": "uint8",
                    "exponent_dtype": "int32", "weight_format": "atlas_e4m3",
                    "dequantization": "e4m3(code)*2^exponent", "rhs_transposed": "true",
                    "output_rescale": "2^(lhs_exp+rhs_exp)",
                    "bias_order": "after_output_cast", "output_dtype": "float32",
                }.items())),
            ),
            policy_fields=_fields(already_quantized=_ALREADY_QUANTIZED),
            operand_roles=["lhs_codes", "lhs_exponents", "rhs_codes", "rhs_exponents"],
            encoding=cap448,
            exponents={
                "storage_dtype": "i32", "signed": True, "operand_bounds": [-127, 127],
                "paired_sum_bounds": [-133, 127], "scale": "2^(lhs_exp+rhs_exp)",
                "granularity": ["scalar", "per-row", "per-head"],
                "logical_shape": "broadcast to corresponding NT operand; invariant over K",
                "layout": "preserve logical code/exponent correspondence and operand order",
                "restoration": "finite nonzero BF16 power-of-two scale; finite normal result or admitted zero",
            },
            accumulation={
                "unit": "mxu1", "model": 47, "exporter_arithmetic": "ipt",
                "policy": MXU1_PRECISION_AUTHORIZATION, "integer_anchor": True,
                "alignment": "truncating", "integer_overflow": "wrap-i32",
                "short_sum_normalization": True, "k_order": "ascending stored logical K",
                "instruction_k": 32, "first_instruction": "reset",
                "continuation": "previous BF16 readout seeds next instruction",
                "tail_padding": "positive zero", "readout": "BF16 RNE after every instruction",
            },
            boundaries={
                "sequence": ["ordered MXU1 BF16 readouts", "restore exponent sum at final readout",
                          "declared output cast", "separate source bias BF16 RNE when present"],
                "output_dtype": "f32", "f32_extension": "exact BF16-to-f32 widening of the restored BF16 result",
                "bias": "must be split before admission; no MXU seed bias or primitive VPU add substitution",
                "weights": "unchanged prequantized codes and exponents; no quantizer or encoder",
                "activation_quantization": "separate authored quantizer family",
            },
            runtime={
                "checks": ["all code bytes", "each signed exponent", "paired exponent sums before device IO",
                        "each instruction readout finite-normal or admitted zero", "restored result domain",
                        "device completion and status"],
                "exceptions": "NaN, infinity, reserved/subnormal FP8 and out-of-domain readouts refuse",
                "publication": "caller output unchanged on failure; publish workspace only after success",
            },
            authorization={
                "required": MXU1_PRECISION_AUTHORIZATION,
                "source_equivalence": "selected exporter ipt to MXU1 claim; no framework float dot-product proof",
                "evidence": "existing reviewed single-product proof and sampled long-K evidence; no new qualification",
            },
        ),
        _family(
            "activation-quantize", "atlas.authored-activation-quantizer-exact.v1",
            constraints=_constraints(
                (("operand_count",), 1),
                (("operands", "0", "dtype"), "bf16"),
                (("attributes", "code_dtype"), "uint8"),
                (("attributes", "exponent_dtype"), "int32"),
                (("attributes", "weight_format"), "atlas_e4m3"),
                (("attributes", "dequantization"), "e4m3(code)*2^exponent"),
            ),
            encoding=cap448,
            source_contract={
                "paired_outputs": ["atlas.quantize_act.codes", "atlas.quantize_act.exponent"],
                "pairing": "same input and authored attributes; both projections required",
                "modes": {"static": "static_activation_quantizer", "dynamic": "dynamic_activation_quantizer"},
                "rules": ["ref", "nc"], "reductions": ["tensor", "token", "head", "block", "head_block"],
                "groups": "authored rows/reduction axes; dynamic amax groups never split",
            },
            encoder={
                "source_dtype": "bf16", "rounding": "RNE significand then exponent-range flush and saturation",
                "subnormal": "flush to positive zero", "negative_zero": "positive zero",
                "nan": "positive zero", "infinity": "signed finite saturation",
                "exponent": "signed i32 output; clamp packing shift to [-127,127] without rewriting authored exponent",
                "static": "emit unchanged authored static_exp, a signed-i32 scalar",
                "dynamic": "finite BF16 magnitude amax per authored group; ref/nc exponent rule remains unchanged",
                "weights": "no weight encoding or conversion",
            },
            runtime={
                "checks": ["paired-output and shape closure", "selected device body legality",
                        "instruction/memory budget", "completion and status"],
                "source_change": "none beyond the explicitly authored quantizer",
                "publication": "existing checked paired-output dispatch; no unconditional atomicity claim",
                "route": "device body where admitted; otherwise the checked host encoder of the quantizer's mode",
            },
        ),
        _family(
            "device-layout", "atlas.raw-byte-device-layout.v1",
            transformation="exact raw-byte permutation or copy; no numerical conversion",
            runtime={"checks": ["closed dense affine bounds", "device widths and addresses",
                                 "buffer layout and completion/status", "publication"]},
            layout="preserve logical element bytes and explicit index-map operand order",
        ),
        _family(
            "host-static-quantize", "atlas.authored-static-activation-quantizer-exact.v1",
            constraints=_constraints(
                (("operand_count",), 1), (("operands", "0", "dtype"), "bf16"),
                (("attributes", "mode"), "static"),
                (("attributes", "semantic_kind"), "static_activation_quantizer"),
                (("attributes", "code_dtype"), "uint8"), (("attributes", "exponent_dtype"), "int32"),
                (("attributes", "weight_format"), "atlas_e4m3"),
                (("attributes", "dequantization"), "e4m3(code)*2^exponent"),
            ),
            encoding=cap448,
            source_contract={"paired_outputs": ["atlas.quantize_act.codes", "atlas.quantize_act.exponent"],
                             "pairing": "same input and authored attributes; both projections required",
                             "mode": "static", "rules": ["ref", "nc"],
                             "static_exp": "unchanged authored signed-i32 scalar"},
            encoder={"source_dtype": "bf16", "rounding": "RNE significand then exponent-range flush and saturation",
                     "packing_shift": "clamp authored exponent to [-127,127] without rewriting exponent output",
                     "subnormal": "positive zero", "negative_zero": "positive zero", "nan": "positive zero",
                     "infinity": "signed finite saturation", "weights": "no encoding or conversion"},
            runtime={"checks": ["original paired static source recognizer", "fixed exponent and code/exponent shapes",
                                "dense disjoint output/input buffers"],
                     "publication": "existing checked host static encoder; no device fallback or arithmetic change"},
        ),
        _family(
            "host-static-quantize-f32", "atlas.authored-static-activation-quantizer-f32-exact.v1",
            constraints=_constraints(
                (("operand_count",), 1), (("operands", "0", "dtype"), "f32"),
                (("attributes", "mode"), "static"),
                (("attributes", "semantic_kind"), "static_activation_quantizer"),
                (("attributes", "code_dtype"), "uint8"), (("attributes", "exponent_dtype"), "int32"),
                (("attributes", "weight_format"), "atlas_e4m3"),
                (("attributes", "dequantization"), "e4m3(code)*2^exponent"),
            ),
            encoding=cap448,
            source_contract={"paired_outputs": ["atlas.quantize_act.codes", "atlas.quantize_act.exponent"],
                             "pairing": "same input and authored attributes; both projections required",
                             "mode": "static", "rules": ["ref", "nc"],
                             "static_exp": "unchanged authored signed-i32 scalar"},
            encoder={"source_dtype": "f32", "source_cast": "the authored quantizer's binary32 to BF16 RNE cast",
                     "rounding": "RNE significand then exponent-range flush and saturation",
                     "packing_shift": "clamp authored exponent to [-127,127] without rewriting exponent output",
                     "subnormal": "positive zero", "negative_zero": "positive zero", "nan": "positive zero",
                     "infinity": "signed finite saturation", "weights": "no encoding or conversion"},
            runtime={"checks": ["original paired static source recognizer", "fixed exponent and code/exponent shapes",
                                "dense disjoint output/input buffers"],
                     "publication": "existing checked host static encoder; no device fallback or arithmetic change"},
        ),
        _family(
            "layout", "atlas.raw-byte-layout.v1",
            transformation="exact raw-byte permutation or copy; no numerical conversion",
            runtime={"checks": ["closed dense affine bounds", "device widths and addresses where the device route runs",
                                 "host emitter overlap and storage checks where the host route runs",
                                 "buffer layout and completion/status", "publication"],
                     "route": "device where its existing checks admit the closed layout beside a quantized contraction; "
                              "contraction bias broadcasts, scalar broadcasts, and every other layout run "
                              "otherwise the checked host layout"},
            layout="preserve logical element bytes and explicit index-map operand order",
        ),
        _family(
            "exact-bf16-add", "ieee754-bf16-add-rne-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 2), (("result", "dtype"), "bf16"),
                (("operands", "0", "dtype"), "bf16"), (("operands", "1", "dtype"), "bf16"),
                (("attributes", "numeric_semantics"), "ieee754-bf16-add-rne-v1"),
            ),
            excluded_constraints=_constraints(
                (("operands", "0", "producer", "attributes", "semantic_kind"), "prequantized_contraction"),
                (("operands", "1", "producer", "operation"), "gather"),
            ),
            arithmetic={"operation": "source BF16 addition", "rounding": "BF16 RNE with guard/sticky/parity correction",
                        "operand_order": "unchanged",
                        "implementation": "existing exact-device vector, pair-row and exceptional scalar routes where the "
                                          "closed region is admitted; otherwise the checked host BF16 RNE adder",
                        "primitive_vpu_substitution": "forbidden; FP32 RNE then BF16 chop is a different operation"},
            runtime={"checks": ["closed same-shaped raw BF16 operands and pointwise maps", "original domain/status guards",
                                "bounded slicing, selected device resources and completion"],
                     "publication": "atomic logical output after every device status succeeds",
                     "bias": "prepared post-cast contraction bias excluded and kept in the host family"},
        ),
        _family(
            "exact-bf16-to-f32", "bf16-to-f32-exact-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 1), (("result", "dtype"), "f32"), (("operands", "0", "dtype"), "bf16"),
                (("attributes", "numeric_semantics"), "bf16-to-f32-exact-v1"),
            ),
            arithmetic={"operation": "exact widening", "implementation": "existing exact device conversion where the closed "
                                                   "region is admitted; otherwise the checked host cast"},
            runtime={"checks": ["closed same-shaped raw operand and pointwise maps", "device status and completion"],
                     "publication": "atomic logical output after every device status succeeds"},
        ),
        _family(
            "exact-f32-to-bf16", "f32-to-bf16-rne-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 1), (("result", "dtype"), "bf16"), (("operands", "0", "dtype"), "f32"),
                (("attributes", "numeric_semantics"), "f32-to-bf16-rne-v1"),
            ),
            arithmetic={"operation": "BF16 RNE narrowing", "implementation": "existing exact device conversion where the closed "
                                                   "region is admitted; otherwise the checked host cast"},
            runtime={"checks": ["closed same-shaped raw operand and pointwise maps", "device status and completion"],
                     "publication": "atomic logical output after every device status succeeds"},
        ),
        _family(
            "device-bf16-mul", "bf16-bounded-pointwise-rne-v1", source_policies=("raw-bit-pattern",),
            constraints=_constraints(
                (("operand_count",), 2), (("result", "dtype"), "bf16"),
                (("operands", "0", "dtype"), "bf16"), (("operands", "1", "dtype"), "bf16"),
                (("attributes", "numeric_semantics"), "ieee754-bf16-mul-rne-v1"),
            ),
            arithmetic={"operation": "source BF16 multiplication", "rounding": "BF16 RNE of the exact product",
                        "operand_order": "unchanged",
                        "domain": "zero or normal BF16 operands whose exponents keep every product normal and finite",
                        "implementation": "existing device pointwise route (VPU mul_bf16_zero)"},
            runtime={"checks": ["closed same-shaped raw BF16 operands and pointwise maps",
                                "bounded input code domain scanned before device stores",
                                "selected device resources and completion"],
                     "publication": "atomic logical output after every device status succeeds"},
        ),
    )
    description = describe_compiler()
    families = {family.id: family for family in description.families}
    patterns = {pattern.id: pattern for pattern in description.patterns}
    for row in rows:
        family, pattern = families.get(row.family), patterns.get(row.pattern)
        if (family is None or pattern is None or row.pattern not in family.patterns
                or family.pipeline != pattern.pipeline):
            raise ValueError("Atlas numerical declaration differs from compiler family/pattern identity")
    return rows
