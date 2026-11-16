"""Serialize existing window reservations; placement remains the caller's job.

Image offsets are relative to the caller's IMEM map. This record does not
authenticate that map or turn lifetime/promotion receipts into allocations.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from merlin.semantic_compiler.target_binding import NativeCompilationError


def deployment_requirements(regions, native_root):
    from .binding import validate_io_window

    base, size = validate_io_window()
    root = Path(native_root).resolve()
    reservations, images, seen = [], [], set()

    def refuse(message):
        raise NativeCompilationError("unsupported_semantics", "deployment: " + message)

    def reserve(name, address, count, role, lifetime=None, promotions=()):
        if (type(address) is not int or type(count) is not int or count <= 0
                or not base <= address < address + count <= base + size):
            refuse("provider reservation outside selected window: " + name)
        reservations.append(dict(name=name, address=address, bytes=count, role=role,
                                 lifetime=lifetime, promotion_receipts=list(promotions)))

    for region in regions:
        if region["placement"] == "host":
            if region.get("native_programs"):
                refuse("host region unexpectedly contains native programs")
            continue
        if region["placement"] != "accelerator":
            refuse("unsupported provider region placement: " + str(region["placement"]))
        programs = region.get("native_programs", [])
        if not programs:
            refuse("device region has no published native program")
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
            if schema == "atlas.quantized_loop_execution.v1":
                if region["description"].get("execution_route") != "checked_counted_image":
                    refuse("quantized deployment requires checked_counted_image route")
                if plan.get("resident_bindings"):
                    refuse("on-chip deployment does not support resident bindings")
                for role, addresses, counts in (
                    ("input", plan["input_addresses"], plan["input_buffer_bytes"]),
                    ("exponent", plan["exponent_addresses"], plan["exponent_buffer_bytes"]),
                    ("output", plan["output_addresses"], [plan["output_bytes"]]),
                ):
                    if len(addresses) != len(counts):
                        refuse("provider execution buffer lengths differ")
                    for i, (address, count) in enumerate(zip(addresses, counts, strict=True)):
                        reserve(f"{relative}/{role}_{i}", address, count, role)
                reserve(relative + "/status", plan["status_address"], plan["status_bytes"], "status")
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
            if (not binary or len(binary) > 131072 or len(binary) % 4
                    or binary != b"".join(row["word_u32"].to_bytes(4, "little")
                                          for row in physical["instructions"])):
                refuse("provider image differs from physical program or exceeds IMEM capacity")
            images.append(dict(path=relative + "/program.bin", sha256=sha256(binary).hexdigest(),
                               bytes=len(binary), imem_offset_bytes=0, capacity_bytes=131072,
                               upload_convention="counted_ensure_loaded"))
    return dict(schema="atlas.provider_deployment.v1", window=dict(address=base, bytes=size),
                reservations=reservations, images=images,
                raw_scratch_bytes=max(1, *(row["description"].get("workspace_bytes", 0) for row in regions)),
                conversion_scratch_bytes=max(1, *(row["description"].get("source_workspace_bytes", 0)
                                                  for row in regions)))
