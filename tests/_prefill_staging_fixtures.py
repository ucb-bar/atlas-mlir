"""Atlas-local synthetic staging fixtures; no dependency on Merlin's test package.

Adapted from Merlin's staging test helpers. Only fixtures used by Atlas's
arithmetic checks live here; all compiler and reference arithmetic comes from
production.
"""

import json
from dataclasses import replace
from hashlib import sha256

from merlin.compiler.backend.protocol import PlatformDeployment
from merlin.compiler.build.staging import band_rules, output_columns, split_rule, stage
from merlin.compiler.build.regions import CompiledCRegion
from merlin.compiler.partition.semantic_regions import extract_semantic_region, outline_semantic_regions
from merlin.compiler.semantic.codegen import CodegenOptions, codegen_options
from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType
from merlin.compiler.partition.global_plan import (
    BufferRepresentation, CycleInterval, GlobalPlan, RegionAlternative, ValueRepresentation,
)

BASE, WINDOW = 0x10000, 4096
K, N = 64, 128
HEADS, ROWS, COLUMNS = 4, 32, 16


def plan_for(source, groups):
    pure = [node for node in source.nodes if node.effect == "pure"]
    rep = ValueRepresentation("memory", "dense", pure[0].type.dtype)
    alternatives = []
    for index, indices in enumerate(groups):
        region = extract_semantic_region(source, tuple(pure[i].id for i in indices))
        inputs = tuple(node.id for node in region.nodes if node.effect in {"input", "constant"})
        alternatives.append(
            RegionAlternative(
                f"region_{index}",
                tuple(indices),
                f"kernel_{index}",
                "device",
                CycleInterval.unknown("uncalibrated"),
                tuple(BufferRepresentation(key, replace(rep, dtype=source.node(key).type.dtype)) for key in inputs),
                tuple(
                    BufferRepresentation(key, replace(rep, dtype=source.node(key).type.dtype)) for key in region.outputs
                ),
            )
        )
    return GlobalPlan(tuple(alternatives), (), CycleInterval.unknown("uncalibrated"))


def onchip():
    return codegen_options(CodegenOptions(io_memory_base=BASE, io_memory_bytes=WINDOW, io_memory_capacity_base=BASE,
                                          io_memory_capacity_bytes=WINDOW))


class Provider(PlatformDeployment):
    """Declares 32-column refill groups for the unsplit weight, a 64-byte frame, and one launch per device step,
    deployed on chip."""

    def deployment_required(self):
        return True

    def application_binding(self, name, regions, interface):
        # Each step package renders an ordinary single-call application.
        from merlin.compiler.build.regions import ApplicationCBinding

        return ApplicationCBinding("", "", {"invoke": name + "_run", "completion": "synchronous"})

    def __init__(self, groups=True):
        self.groups = groups

    def deployment_requirements(self, regions, native_root):
        (region,) = regions
        host = region["placement"] == "host"
        width = region["description"].get("width")
        images = [dict(path=program + "/program.bin", sha256=sha256(b"\x73\0\0\0").hexdigest(), bytes=4,
                       imem_offset_bytes=0, capacity_bytes=1024, upload_convention="counted_ensure_loaded")
                  for program in region["native_programs"]]
        record = dict(
            schema="merlin.provider_deployment.v1", window=dict(address=BASE, bytes=WINDOW), images=images,
            window_section=".device_window", image_section=".device_images",
            reservations=[] if host else [dict(name="frame", address=BASE, bytes=64, role="input",
                                               lifetime=dict(scope="region_call", context_index=0),
                                               promotion_receipts=[])],
            regions=[dict(context_index=0, placement=region["placement"],
                          launches_per_invocation=0 if host else 1)],
            allocation_alignment=64, host_context_reservation_bytes=64, max_polls=1,
            host_map=dict(imem_address=0, imem_bytes=1024, csr_address=0),
            runtime=dict(host_type="h", io_type="io", start="s", wait="w", image_completion="c", reload_policy="r"),
            host_allocation=dict(address=0, bytes=1024, reserved_bytes=64, measurement="declared_reservations",
                                 declared_reservations=[dict(name="context", bytes=64)]),
            raw_scratch_bytes=1, conversion_scratch_bytes=1)
        if self.groups and width is None and not host:
            record["refill_groups"] = [dict(context_index=0, operand="w", region_operand="w", groups=[
                dict(first_launch=g, launch_count=1, source_offset=g * 32 * K, bytes=32 * K) for g in range(N // 32)])]
        return record


def package(region, width=None, programs=()):
    inputs = tuple(node.id for node in region.nodes if node.effect in {"input", "constant"})
    files = {"call.c": b"int call;\n", "call.h": b"#pragma once\n"}
    return CompiledCRegion("call", "call.h", inputs, region.outputs, files, (), programs,
                           {"width": width} if width else {})


def _width(region, request):
    """Extent of the first output along the axis its slice request narrowed."""
    full, part = region.node(region.outputs[0]).type.shape, request.node(region.outputs[0]).type.shape
    return next(b for a, b in zip(full, part) if a != b)


def _identity(loops, rank):
    from merlin.compiler.semantic.model import IndexMap

    return IndexMap(loops, tuple(tuple(int(i == j) for j in range(loops)) for i in range(rank)), (0,) * rank)


def contraction(m, k, n):
    """A prequantized contraction on the device, then an index-mapped host copy of its result."""
    def tensor(shape, dtype):
        return TensorType(shape, dtype, "raw-bit-pattern")

    nodes = (
        SemanticNode("x", "input", (), tensor((m, k), "i8"), effect="input"),
        SemanticNode("xe", "input", (), tensor((), "i32"), effect="input"),
        SemanticNode("w", "input", (), tensor((n, k), "i8"), effect="input"),
        SemanticNode("we", "input", (), tensor((n, 1), "i32"), effect="input"),
        SemanticNode("y", "synthetic.qlinear", ("x", "xe", "w", "we"), tensor((m, n), "bf16"),
                     (("rhs_transposed", "true"), ("semantic_kind", "prequantized_contraction"))),
        SemanticNode("z", "synthetic.copy", ("y",), tensor((m, n), "bf16"), index_maps=(_identity(2, 2),) * 2),
    )
    source = KernelRequest(nodes, ("z",), ("external",), tuple((key, "external") for key in ("x", "xe", "w", "we")),
                           "synthetic")
    plan = plan_for(source, ((0,), (1,)))
    plan = replace(plan, selected=(plan.selected[0], replace(plan.selected[1], placement="host")))
    region = outline_semantic_regions(source, entry="application", plan=plan).regions[0][1]
    assert split_rule(region)[0] == "output_columns" and [rule.name for rule in band_rules(region)] == ["rows"]
    return source, plan


def f32(shape):
    return TensorType(shape, "f32", "raw-bit-pattern")


def publish(segments, request, label):
    """Publish a one-word native program for a device step package."""
    (segments / label).mkdir(parents=True, exist_ok=True)
    (segments / label / "program.bin").write_bytes(b"\x73\0\0\0")
    (segments / label / "manifest.json").write_text(json.dumps(dict(
        engine="merlin_native", request_digest=request.digest(), binary_sha256=sha256(b"\x73\0\0\0").hexdigest(),
        target_identity=request.target_identity)))
    return (f"segments/{label}",)


def quantizer(reduce="head", placement="accelerator"):
    """A dynamic activation quantizer [1, H, R, C] -> codes and one exponent per head (or per token)."""
    shape = (1, HEADS, ROWS, COLUMNS)
    exponents = (1, HEADS, 1, 1) if reduce == "head" else (1, HEADS, ROWS, 1) if reduce == "token" else ()
    attrs = (("mode", "dynamic"), ("reduce", reduce), ("rows", ROWS), ("semantic_kind", "dynamic_activation_quantizer"))
    nodes = (SemanticNode("x", "input", (), f32(shape), effect="input"),
             SemanticNode("codes", "atlas.quantize_act.codes", ("x",), TensorType(shape, "i8", "raw-bit-pattern"), attrs),
             SemanticNode("exponent", "atlas.quantize_act.exponent", ("x",), TensorType(exponents, "i32", "raw-bit-pattern"),
                          attrs))
    source = KernelRequest(nodes, ("codes", "exponent"), ("external",) * 2, (("x", "external"),), "synthetic")
    plan = plan_for(source, ((0, 1),))
    plan = replace(plan, selected=tuple(replace(row, placement=placement) for row in plan.selected))
    return source, plan, outline_semantic_regions(source, entry="application", plan=plan)


def program(region, label, root):
    return publish(root / "segments", region, label)


def staged(source, plan, provider, compiled_widths, root):
    """Compile synthetic slices under an explicit test-local output directory."""
    outlined = outline_semantic_regions(source, entry="application", plan=plan)
    compiled = [package(region, programs=() if i else program(region, "region_000", root))
                for i, (_, region) in enumerate(outlined.regions)]

    def compile_slice(index, key, region, request, label, root):
        compiled_widths.append(_width(region, request))
        return package(request, _width(region, request), program(request, label, root))

    with onchip():
        return outlined, stage(source, provider, outlined, plan, compiled, capacity=WINDOW, native_root=root,
                               compile_slice=compile_slice)
