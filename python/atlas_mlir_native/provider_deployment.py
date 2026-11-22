"""Serialize existing window reservations; placement remains the caller's job.

Image offsets are relative to the caller's IMEM map. This record does not
authenticate that map or turn lifetime/promotion receipts into allocations.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import prod
from pathlib import Path

from merlin.compiler.semantic.target_binding import NativeCompilationError
from .soc import SOC


WINDOW_SECTION = ".atlas_window"
# A board DDR program loads the immutable parameters, which end the window data, read-only in their own load.
PARAMETER_SECTION = ".atlas_parameters"
WINDOW_ADDRESS = SOC.memory_region("atlas_data").base
WINDOW_BYTES = SOC.memory_region("atlas_data").capacity_bytes
IMEM_BYTES = SOC.memory_region("atlas_imem").capacity_bytes
HOST_CONTEXT_BYTES = 4096
ALLOCATION_ALIGNMENT = 64


def deployment_requirements(regions, native_root, *, soc=None):
    from .binding import validate_io_window
    from .c_runtime import IMAGE_SECTION
    from .options import current_codegen_options

    soc = SOC if soc is None else soc
    window = soc.memory_region("atlas_data")
    imem = soc.memory_region("atlas_imem")
    host = soc.memory_region("host_scratch")
    ddr = soc.memory_region("ddr")
    options = current_codegen_options()
    geometry = (options.io_memory_base, options.io_memory_bytes,
                options.io_memory_capacity_base, options.io_memory_capacity_bytes)
    # The on-chip window, or the end of the board's DDR after a host that starts at the DDR base. A window that
    # starts at the base is the measuring compile, whose host follows the placed data.
    board = geometry[:2] == geometry[2:] and ddr.base <= geometry[0] and geometry[0] + geometry[1] == ddr.end
    if not board and geometry != (window.base, window.capacity_bytes) * 2:
        return None
    base, size = validate_io_window()
    root = Path(native_root).resolve()
    reservations, images, seen, calls, refill_groups = [], [], set(), [], []

    def refuse(message, status="unsupported_semantics"):
        raise NativeCompilationError(status, "deployment: " + message)

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
            if schema == "atlas.activation_quantize_vector_plan.v1":
                # The private reduction/exponent slots live in VMEM. The entire
                # external frame consists of these three padded DMA buffers;
                # workspace_bytes must not hide an unreserved private tail.
                fields = ("input_address", "input_bytes", "code_address", "code_bytes",
                          "exponent_address", "exponent_bytes", "output_bytes", "workspace_bytes")
                if (any(type(plan.get(key)) is not int or plan[key] <= 0 for key in fields)
                        or any(plan[key] % 32 for key in fields)
                        or plan.get("engine") != "vector"
                        or type(plan.get("launch_count")) is not int or plan["launch_count"] != 1
                        or plan.get("resident_bindings")
                        or plan.get("private_staging")
                        or any(type(plan.get(key)) is not list
                               or len(plan[key]) != 1 or type(plan[key][0]) is not int
                               for key in ("input_addresses", "input_payload_bytes", "output_addresses"))
                        or plan["input_addresses"] != [plan["input_address"]]
                        or plan["input_payload_bytes"] != [plan["input_bytes"]]
                        or plan["output_addresses"] != [plan["code_address"]]
                        or plan["code_address"] != plan["input_address"] + plan["input_bytes"]
                        or plan["exponent_address"] != plan["code_address"] + plan["code_bytes"]
                        or plan["output_bytes"] != plan["code_bytes"] + plan["exponent_bytes"]
                        or plan["workspace_bytes"] != plan["input_bytes"] + plan["output_bytes"]):
                    refuse("activation quantizer buffer contract differs")
                for role in ("input", "code", "exponent"):
                    reserve(f"{relative}/{role}", plan[role + "_address"], plan[role + "_bytes"],
                            role, call_lifetime)
            elif schema in {"atlas.row_loop_execution_plan.v1", "atlas.row_loop_execution_plan.v2"}:
                # Readonly row frames, then one writable output-plus-scratch frame per row group.
                fields = ("groups", "readonly_base", "readonly_stride_bytes", "output_base",
                          "output_stride_bytes", "scratch_address", "scratch_bytes", "scratch_stride_bytes")
                if (any(type(plan.get(key)) is not int or plan[key] < 0 for key in fields)
                        or plan["groups"] <= 0 or plan["readonly_stride_bytes"] <= 0
                        or plan.get("resident_bindings") or plan.get("private_staging")
                        or plan["output_base"] != plan["readonly_base"] + plan["groups"] * plan["readonly_stride_bytes"]
                        or plan["scratch_stride_bytes"] != plan["output_stride_bytes"]
                        or not 0 <= plan["scratch_bytes"] < plan["output_stride_bytes"]
                        or plan["scratch_address"] != plan["output_base"] + plan["output_stride_bytes"]
                        - plan["scratch_bytes"]):
                    refuse("row loop frame contract differs")
                reserve(relative + "/readonly", plan["readonly_base"],
                        plan["groups"] * plan["readonly_stride_bytes"], "input", call_lifetime)
                for group in range(plan["groups"]):
                    offset = group * plan["output_stride_bytes"]
                    reserve(f"{relative}/output_{group}", plan["output_base"] + offset,
                            plan["output_stride_bytes"] - plan["scratch_bytes"], "output", call_lifetime)
                    if plan["scratch_bytes"]:
                        reserve(f"{relative}/scratch_{group}", plan["scratch_address"] + offset,
                                plan["scratch_bytes"], "private", call_lifetime)
            elif schema in {"atlas.quantized_loop_execution.v1", "atlas.pointwise_execution_plan.v1",
                          "atlas.layout_device.v1", "atlas.exact_float.v1", "atlas.reduction_execution_plan.v1"}:
                counted = schema == "atlas.quantized_loop_execution.v1"
                route = region["description"].get("execution_route")
                if counted and route != "checked_counted_image":
                    # Row blocks and bounded K32 tiles are chosen only when one counted image does not fit.
                    refuse("quantized deployment requires checked_counted_image route", "resource_limit"
                           if route in {"checked_counted_row_blocks", "bounded_k32_tile_dispatch"}
                           else "unsupported_semantics")
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
            if (not binary or len(binary) > imem.capacity_bytes or len(binary) % 4
                    or binary != b"".join(row["word_u32"].to_bytes(4, "little")
                                          for row in physical["instructions"])):
                refuse("provider image differs from physical program or exceeds IMEM capacity")
            images.append(dict(path=relative + "/program.bin", sha256=sha256(binary).hexdigest(),
                               bytes=len(binary), imem_offset_bytes=0, capacity_bytes=imem.capacity_bytes,
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
        if len(programs) > 1 and region["description"].get("kind") in {
                "exact_float_device", "activation_quantize_device"}:
            # Reconstruct the source-bound schedule, rather than treating an
            # arbitrary image list or a claimed lifetime as sequential proof.
            from merlin.compiler.semantic.model import KernelRequest
            from merlin.compiler.schedule.launch_schedule import make_launch_schedule

            parent = (root / programs[0]).parent.parent
            if not parent.is_relative_to(root):
                refuse("exact float publication escapes deployment root")
            source = KernelRequest.from_record(json.loads((parent / "source-request.json").read_bytes()))
            quantizer = region["description"]["kind"] == "activation_quantize_device"
            label = "activation quantizer" if quantizer else "exact float"
            stem = "quantizer" if quantizer else "float"
            saved_schedule = json.loads((parent / (stem + "-schedule.json")).read_bytes())
            if quantizer:
                from .activation_quantize_application import prepare_plan, _verify_publication
                from types import SimpleNamespace

                schedule = prepare_plan(source, limits=SimpleNamespace(
                    functional_instruction_limit=saved_schedule["functional_instruction_limit"]),
                    _pipeline=saved_schedule.get("lowering_pipeline", "mandatory"),
                    _static_sizing="static_instruction_envelope" in saved_schedule,
                    _static_capacity=saved_schedule["static_whole_body_max_elements"],
                    _device_kv_quantizer="dynamic_group_admission" in saved_schedule)
            else:
                from .exact_float_application import prepare_plan, _verify_publication

                schedule = prepare_plan(source,
                    private_output=saved_schedule.get("private_output", False),
                    _chunk_elements=saved_schedule.get("chunk_elements"))
            manifest = json.loads((parent / (stem + "-manifest.json")).read_bytes())
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
            if (programs != expected or region["description"].get(stem + "_schedule") != schedule.record()
                    or region["description"].get("device_launch_schedule") != proof):
                refuse(label + " sequential schedule differs from source publication")
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
            from merlin.compiler.schedule.launch_schedule import validate_launch_schedule

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
    if not board:
        placed = dict(address=host.base, bytes=host.capacity_bytes, usable_bytes=host.capacity_bytes - HOST_CONTEXT_BYTES)
    elif base == ddr.base:
        placed = dict(address=base, bytes=size, usable_bytes=size, placement="after_window_data")
    else:
        placed = dict(address=ddr.base, bytes=base - ddr.base, usable_bytes=base - ddr.base)
    return dict(schema="merlin.provider_deployment.v1", window=dict(address=base, bytes=size),
                window_section=WINDOW_SECTION, image_section=IMAGE_SECTION,
                **({"parameter_section": PARAMETER_SECTION} if board else {}),
                platform_owned_requirements=["CHIP", "PLATFORM", "RVV_TYPE", "PENGUIN26_EXIT_TOHOST",
                    "PENGUIN26_UART_BAUDRATE", "CMAKE_TOOLCHAIN_FILE", "CMAKE_PROJECT_INCLUDE",
                    "PLATFORM_ROOT", "PLATFORM_MAP"],
                reservations=reservations, images=images,
                regions=calls,
                # Platform compile convention: the deploy's clock, else the declared host clock; neither is measured.
                sys_clk_freq_hz=options.host_clock_hz or next(
                    clock.frequency_hz for clock in soc.clocks if clock.id == "host"),
                allocation_alignment=ALLOCATION_ALIGNMENT,
                # A board host is measured after linking, so its context is bounded by the declared DDR it links into.
                host_context_reservation_bytes=ddr.capacity_bytes if board else HOST_CONTEXT_BYTES,
                host_map=dict(imem_address=imem.base, imem_bytes=imem.capacity_bytes,
                              csr_address=soc.resource("atlas.csr").address),
                imem_address_convention="host_map.imem_address + image.imem_offset_bytes",
                max_polls=10000000,
                transport_timeout_seconds=60, transport_clock_hz_bound=1_000_000_000,
                runtime=dict(host_type="atlas_host", io_type="atlas_tile_memory_io",
                             start="atlas_host_start", wait="atlas_host_wait",
                             image_completion="ensure_loaded_before_start",
                             reload_policy="generated counted ensure_loaded"),
                # Measured objects fill the host memory left after the declared stack and heap. The object
                # stage refuses unless the linked code's stack bound fits the stack and links no allocator.
                # usable_bytes is the frozen aggregate that saved declared-reservation deployments (Kit A)
                # regenerate with; new deployments place into all 65,536 backed bytes (Merlin place()).
                host_allocation=dict(**placed,
                    measurement="measured_sections",
                    declared_reservations=[dict(name="stack", bytes=8192), dict(name="heap", bytes=0)],
                    reserved_bytes=8192),
                raw_scratch_bytes=max(1, *(row["description"].get("workspace_bytes", 0) for row in regions)),
                conversion_scratch_bytes=max(1, *(row["description"].get("source_workspace_bytes", 0)
                                                  for row in regions)),
                **({"refill_groups": refill_groups} if refill_groups else {}))
