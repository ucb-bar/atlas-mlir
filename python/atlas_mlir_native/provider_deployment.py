"""Serialize existing window reservations; placement remains the caller's job.

Image offsets are relative to the caller's IMEM map. This record does not
authenticate that map or turn lifetime/promotion receipts into allocations.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import prod
from pathlib import Path

from merlin.semantic_compiler.target_binding import NativeCompilationError


WINDOW_ADDRESS = 0x200C0000
WINDOW_BYTES = 786432
IMEM_BYTES = 131072
HOST_CONTEXT_BYTES = 4096
ALLOCATION_ALIGNMENT = 64


def deployment_requirements(regions, native_root):
    from .binding import validate_io_window
    from .options import current_codegen_options

    options = current_codegen_options()
    if (options.io_memory_base != WINDOW_ADDRESS or options.io_memory_bytes != WINDOW_BYTES
            or options.io_memory_capacity_base != WINDOW_ADDRESS
            or options.io_memory_capacity_bytes != WINDOW_BYTES):
        return None
    base, size = validate_io_window()
    root = Path(native_root).resolve()
    reservations, images, seen, calls, refill_groups = [], [], set(), [], []

    def refuse(message):
        raise NativeCompilationError("unsupported_semantics", "deployment: " + message)

    def reserve(name, address, count, role, lifetime=None, promotions=()):
        if (type(address) is not int or type(count) is not int or count <= 0
                or not base <= address < address + count <= base + size):
            refuse("provider reservation outside selected window: " + name)
        reservations.append(dict(name=name, address=address, bytes=count, role=role,
                                 lifetime=lifetime, promotion_receipts=list(promotions)))

    for index, region in enumerate(regions):
        call = dict(context_index=region.get("context_index", index), placement=region["placement"])
        calls.append(call)
        if region["placement"] == "host":
            if region.get("native_programs"):
                refuse("host region unexpectedly contains native programs")
            call["launches_per_invocation"] = 0
            continue
        if region["placement"] != "accelerator":
            refuse("unsupported provider region placement: " + str(region["placement"]))
        call_lifetime = dict(scope="region_call", context_index=region["context_index"])
        programs = region.get("native_programs", [])
        if not programs:
            refuse("device region has no published native program")
        first_reservation = len(reservations)
        first_image = len(images)
        tile_buffers, standalone_tiles = [], True
        for relative in programs:
            if not isinstance(relative, str):
                refuse("provider native program path is not relative")
            path = Path(relative)
            if (path.is_absolute() or ".." in path.parts
                    or path.as_posix() != relative or relative in {"", "."}):
                refuse("provider native program path is not relative")
            folder = (root / path).resolve()
            if not folder.is_relative_to(root):
                refuse("provider native program escapes publication root")
            if relative in seen:
                continue
            seen.add(relative)
            plan = json.loads((folder / "execution_plan.json").read_bytes())
            schema = plan.get("schema")
            standalone_tiles &= (schema == "atlas.native_tile_execution_plan.v8"
                                 and plan.get("scope") == "diagnostic_selected_standalone_core")
            if schema in {"atlas.quantized_loop_execution.v1", "atlas.pointwise_execution_plan.v1",
                          "atlas.layout_device.v1", "atlas.exact_float.v1"}:
                counted = schema == "atlas.quantized_loop_execution.v1"
                if counted and region["description"].get("execution_route") != "checked_counted_image":
                    refuse("quantized deployment requires checked_counted_image route")
                if plan.get("resident_bindings"):
                    refuse("on-chip deployment does not support resident bindings")
                input_sizes = plan["input_payload_bytes"] if schema in {
                    "atlas.layout_device.v1", "atlas.exact_float.v1"} else plan["input_buffer_bytes"]
                buffers = [
                    ("input", plan["input_addresses"], input_sizes),
                    ("output", plan["output_addresses"], [plan["output_bytes"]]),
                ]
                if counted:
                    buffers.insert(1, ("exponent", plan["exponent_addresses"], plan["exponent_buffer_bytes"]))
                for role, addresses, counts in buffers:
                    if len(addresses) != len(counts):
                        refuse("provider execution buffer lengths differ")
                    for i, (address, count) in enumerate(zip(addresses, counts, strict=True)):
                        reserve(f"{relative}/{role}_{i}", address, count, role, call_lifetime)
                if counted:
                    reserve(relative + "/status", plan["status_address"], plan["status_bytes"], "status", call_lifetime)
            elif (schema == "atlas.native_tile_execution_plan.v8"
                  and plan.get("scope") == "diagnostic_selected_standalone_core"):
                for role, addresses, counts in (
                    ("input", plan["input_addresses"], plan["input_payload_bytes"]),
                    ("output", plan["output_addresses"], [plan["output_bytes"]]),
                ):
                    if len(addresses) != len(counts):
                        refuse("provider execution buffer lengths differ")
                    for i, (address, count) in enumerate(zip(addresses, counts, strict=True)):
                        if type(address) is not int or type(count) is not int or count <= 0:
                            refuse("invalid standalone provider buffer")
                        if any(start < address + count and address < end for start, end, name, _ in tile_buffers
                               if name.rsplit("/", 1)[0] == relative):
                            refuse("provider reservations overlap within an image")
                        tile_buffers.append((address, address + count, f"{relative}/{role}_{i}", role))
            elif schema in {"atlas.resident_graph_execution_plan.v1", "atlas.native_tile_execution_plan.v8"}:
                # Keep reservations even when a receipt promotes individual
                # accesses. An address can later hold an unpromoted public value.
                lives = {row["name"]: row for row in plan.get("source_liveness_buffers", [])}
                promotion = plan.get("private_staging")
                if promotion is not None and promotion.get("schema") not in {
                        "atlas.private_staging.v1", "atlas.private_staging.v2"}:
                    refuse("unsupported provider promotion schema: " + str(promotion.get("schema")))
                promoted = set(promotion["slot_addresses"]) if promotion else set()
                if promotion and promotion["schema"] == "atlas.private_staging.v2":
                    cache = promotion["input_cache"]
                    if cache.get("schema") != "atlas.readonly_staging.v1":
                        refuse("unsupported provider readonly promotion schema: " + str(cache.get("schema")))
                    promoted.update(cache["cached_addresses"])
                inputs = {row["node"] for row in plan["inputs"]}
                outputs = {row["node"] for row in plan["outputs"]}
                for row in plan["buffers"]:
                    node, address = row["node"], row["address"]
                    receipts = [promotion] if address in promoted else []
                    reserve(relative + "/" + node, address, row["bytes"],
                            "input" if node in inputs else "output" if node in outputs else "private",
                            lives.get(node), receipts)
            else:
                refuse("unsupported provider execution schema: " + str(schema))
            binary = (folder / "program.bin").read_bytes()
            physical = json.loads((folder / "physical-program.json").read_bytes())
            if physical.get("schema") != "atlas.physical_program.v1":
                refuse("unsupported provider physical schema: " + str(physical.get("schema")))
            if (not binary or len(binary) > IMEM_BYTES or len(binary) % 4
                    or binary != b"".join(row["word_u32"].to_bytes(4, "little")
                                          for row in physical["instructions"])):
                refuse("provider image differs from physical program or exceeds IMEM capacity")
            images.append(dict(path=relative + "/program.bin", sha256=sha256(binary).hexdigest(),
                               bytes=len(binary), imem_offset_bytes=0, capacity_bytes=IMEM_BYTES,
                               upload_convention="counted_ensure_loaded"))
        merged = []
        for address, end, name, role in sorted(tile_buffers):
            if merged and address < merged[-1][1]:
                start, prior_end, prior_name, prior_role = merged.pop()
                address, end, name, role = start, max(end, prior_end), prior_name, role if role == prior_role else "private"
            merged.append((address, end, name, role))
        for address, end, name, role in merged:
            reserve(name, address, end - address, role, call_lifetime)
        description = region["description"]
        precision = description.get("precision_plan", {})
        if (standalone_tiles and description.get("kind") == "already_quantized"
                and precision.get("schema") == "atlas.already_quantized_contraction.v1"
                and "operand_permutation" not in precision
                and prod(precision.get("batch_shape", [])) == 1):
            m, n, k = precision["shape"]
            if all(type(value) is int and value > 0 for value in (m, n, k)) and m <= 32:
                rhs = precision["operand_ids"][1]
                local_inputs = description["input_node_ids"]
                if len(local_inputs) != len(region["inputs"]) or rhs not in local_inputs:
                    refuse("refill operand differs from region inputs")
                refill_groups.append(dict(context_index=region["context_index"],
                    operand=region["inputs"][local_inputs.index(rhs)], region_operand=rhs, groups=[
                        dict(first_launch=group*((k+31)//32), launch_count=(k+31)//32,
                             source_offset=first_n*k, bytes=min(32, n-first_n)*k)
                        for group, first_n in enumerate(range(0, n, 32))]))
        if len(programs) > 1 and region["description"].get("kind") == "exact_float_device":
            # Reconstruct the source-bound schedule, rather than treating an
            # arbitrary image list or a claimed lifetime as sequential proof.
            from .exact_float_application import prepare_plan, _verify_publication
            from merlin.semantic_compiler.model import KernelRequest
            from merlin.compile.launch_schedule import make_launch_schedule

            parent = (root / programs[0]).parent.parent
            if not parent.is_relative_to(root):
                refuse("exact float publication escapes deployment root")
            source = KernelRequest.from_record(json.loads((parent / "source-request.json").read_bytes()))
            schedule = prepare_plan(source)
            manifest = json.loads((parent / "float-manifest.json").read_bytes())
            _verify_publication(parent, manifest, schedule)
            expected = [(parent / row["path"]).relative_to(root).as_posix()
                        for row in schedule.record()["bodies"]]
            runs = []
            for launch in schedule.launches:
                image = launch["body_index"]
                if runs and runs[-1]["image"] == image:
                    runs[-1]["count"] += 1
                else:
                    runs.append(dict(image=image, count=1))
            proof = make_launch_schedule([row["sha256"] for row in images[first_image:]],
                                         [dict(repeat=1, runs=runs)])
            if (programs != expected or region["description"].get("float_schedule") != schedule.record()
                    or region["description"].get("device_launch_schedule") != proof):
                refuse("exact float sequential schedule differs from source publication")
            sources = reservations[first_reservation:]
            # A sequential proof permits reuse across images, never overlap
            # between two buffers in the same image.
            for relative in programs:
                rows = [row for row in sources if row["name"].startswith(relative + "/")]
                for i, left in enumerate(rows):
                    for right in rows[i + 1:]:
                        if (left["address"] < right["address"] + right["bytes"]
                                and right["address"] < left["address"] + left["bytes"]):
                            refuse("provider reservations overlap within an image")
            start = min(row["address"] for row in sources)
            end = max(row["address"] + row["bytes"] for row in sources)
            reservations[first_reservation:] = [dict(
                name=parent.relative_to(root).as_posix() + "/sequential_frame",
                address=start, bytes=end - start, role="sequential_device_frame",
                lifetime=call_lifetime,
                promotion_receipts=[receipt for row in sources for receipt in row["promotion_receipts"]],
                source_reservations=sources, sequential_images=proof)]
        description = region["description"]
        schedule = description.get("device_launch_schedule")
        count = description.get("launches_per_invocation", description.get("launches_per_call"))
        if schedule is not None:
            from merlin.compile.launch_schedule import validate_launch_schedule

            image_hashes = {row["path"]: row["sha256"] for row in images}
            try:
                checked = validate_launch_schedule(schedule,
                    image_sha256s=[image_hashes[path + "/program.bin"] for path in programs],
                    placement=region["placement"])
            except ValueError as error:
                refuse(str(error))
            if count is not None and (type(count) is not int or count != checked["total_launches"]):
                refuse("provider launch count differs from device launch schedule")
            count = checked["total_launches"]
            call["device_launch_schedule"] = checked
        elif count is None:
            count = description.get("streaming", description.get("native_export", {}).get("streaming", {})).get(
                "launch_count", 1 if len(programs) == 1 else None)
        if type(count) is not int or count <= 0:
            refuse("provider region requires an explicit positive launch count")
        call["launches_per_invocation"] = count
    return dict(schema="merlin.provider_deployment.v1", window=dict(address=base, bytes=size),
                reservations=reservations, images=images,
                regions=calls,
                allocation_alignment=ALLOCATION_ALIGNMENT,
                host_context_reservation_bytes=HOST_CONTEXT_BYTES,
                host_map=dict(imem_address=0x20000, imem_bytes=IMEM_BYTES, csr_address=0x40000),
                imem_address_convention="host_map.imem_address + image.imem_offset_bytes",
                max_polls=10000000,
                transport_timeout_seconds=60, transport_clock_hz_bound=1_000_000_000,
                runtime=dict(host_type="atlas_host", io_type="atlas_tile_memory_io",
                             start="atlas_host_start", wait="atlas_host_wait",
                             image_completion="ensure_loaded_before_start",
                             reload_policy="generated counted ensure_loaded"),
                host_allocation=dict(address=0x08000000, bytes=65536, measurement="declared_reservations",
                    declared_reservations=[
                        dict(name="code", bytes=36864, includes="embedded instruction images and constants"),
                        dict(name="data", bytes=8192), dict(name="context", bytes=HOST_CONTEXT_BYTES),
                        dict(name="stack", bytes=12288), dict(name="heap", bytes=4096)],
                    reserved_bytes=65536),
                raw_scratch_bytes=max(1, *(row["description"].get("workspace_bytes", 0) for row in regions)),
                conversion_scratch_bytes=max(1, *(row["description"].get("source_workspace_bytes", 0)
                                                  for row in regions)),
                **({"refill_groups": refill_groups} if refill_groups else {}))
