"""Checked preload preparation for the selected diagnostic MXU0 program.

The standalone-core driver still owns execution. This function consumes only
the compiled package and runtime inputs; it never runs a semantic reference.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.semantic_compiler.model import KernelRequest

from .movement import _canonical, _sha
from .mxu0 import (
    Mxu0Contract, admits_bf16_anchor_pair, admits_bf16_exp2_pair,
    admits_bf16_raw_pair,
    admits_fp8_tile, mxu0_profile,
)
from .mxu0_spatial_tiling import lower_tiled_contraction


def prepare_mxu0_launch(
    artifact: Path, runtime_inputs: dict[str, bytes]
) -> tuple[tuple[int, bytes], ...]:
    """Return checked physical preloads for one invocation of this package.

    The caller must separately use a qualified runner and check completion.
    Constants are reloaded each invocation, even though allocation keeps their
    storage intact, so each invocation has an explicit initialization event.
    """
    manifest = json.loads((artifact / "manifest.json").read_text())
    plan_bytes = (artifact / "execution_plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    request = KernelRequest.from_record(
        json.loads((artifact / "request.json").read_text())
    )
    program = (artifact / "program.bin").read_bytes()
    contract = Mxu0Contract.load()
    source_path = artifact / "source_request.json"
    tiling = None
    if source_path.exists():
        source = KernelRequest.from_record(json.loads(source_path.read_text()))
        tiling = lower_tiled_contraction(source, contract)
        tiling_bytes = _canonical(tiling.record()) + b"\n"
        if (
            tiling.lowered.record() != request.record()
            or (artifact / "tiling.json").read_bytes() != tiling_bytes
            or manifest.get("lowered_request_digest") != request.digest()
            or manifest.get("tiling_plan_sha256") != _sha(tiling_bytes)
            or plan.get("source_tiling")
            != {
                "sha256": _sha(tiling_bytes),
                "host_activity": tiling.host_activity,
            }
        ):
            raise ValueError(
                "ordered-K source, lowering, or physical plan identity changed"
            )
        runtime_inputs = tiling.panelize_inputs(runtime_inputs)
    elif (
        (artifact / "tiling.json").exists()
        or "lowered_request_digest" in manifest
        or "tiling_plan_sha256" in manifest
        or "source_tiling" in plan
    ):
        raise ValueError("unexpected ordered-K source or tiling artifact")
    if (
        manifest.get("schema") != "atlas.native_tensor_compilation.v3"
        or manifest.get("status") != "diagnostic_program"
        or manifest.get("engine") != "merlin_native"
        or manifest.get("target_identity") != contract.identity
        or manifest.get("profile_digest") != mxu0_profile(contract).digest()
        or manifest.get("source_revision") != contract.record["rtl_revision"]
        or manifest.get("software_spec_sha256")
        != contract.record["software_spec_sha256"]
        or manifest.get("request_digest")
        != (tiling.source.digest() if tiling is not None else request.digest())
        or request.target_identity != contract.identity
        or plan.get("schema") != "atlas.native_tensor_diagnostic_plan.v3"
        or plan.get("timing_qualification")
        != "diagnostic_only_no_architectural_mxu_wait"
        or manifest.get("execution_plan_sha256") != _sha(plan_bytes)
        or manifest.get("binary_sha256") != _sha(program)
        or plan.get("program")
        != {"sha256": _sha(program), "word_count": len(program) // 4}
        or len(program) % 4
    ):
        raise ValueError("MXU0 package, target, or execution plan identity changed")
    tile_bytes = contract.record["geometry"]["matrix_register_bytes"]
    conversion = contract.record["vfp8_conversion"]
    base = contract.movement.record["dram_program_base"]
    window = contract.movement.record["dram_window_bytes"]
    input_nodes = {node.id: node for node in request.nodes if node.op == "input"}
    fixed_inputs = manifest.get("fixed_inputs")
    fixed_outputs = manifest.get("fixed_outputs")
    if (
        not isinstance(fixed_inputs, dict)
        or not isinstance(fixed_outputs, list)
        or len(fixed_outputs) != len(request.outputs)
        or any(type(value) is not int for value in fixed_outputs)
    ):
        raise ValueError("MXU0 fixed physical ABI is malformed")
    if not isinstance(runtime_inputs, dict) or set(runtime_inputs) != {
        node.id for node in request.nodes if node.op == "input"
    }:
        raise ValueError("MXU0 runtime inputs differ from the compiled signature")
    rows = plan.get("inputs")
    if not isinstance(rows, list) or {
        row.get("source") for row in rows if isinstance(row, dict)
    } != set(runtime_inputs):
        raise ValueError("MXU0 input plan differs from the compiled signature")
    if len(rows) != len(runtime_inputs):
        raise ValueError("MXU0 input plan has a duplicate or missing source")

    preloads: list[tuple[int, bytes]] = []
    for row in rows:
        source = row["source"]
        payload = runtime_inputs[source]
        slot = fixed_inputs.get(source)
        node = input_nodes[source]
        count = 2 if node.type.dtype == "bf16" else 1
        domain_ok = (
            (
                (
                    admits_bf16_raw_pair(contract, node.type.numerical_policy, payload)
                    or admits_bf16_anchor_pair(contract, node.type.numerical_policy, payload)
                    or admits_bf16_exp2_pair(contract, node.type.numerical_policy, payload)
                )
                if count == 2
                else admits_fp8_tile(contract, node.type.numerical_policy, payload)
            )
            if type(payload) is bytes
            else False
        )
        if (
            type(payload) is not bytes
            or len(payload) != count * tile_bytes
            or not domain_ok
        ):
            if count == 2:
                domain = (
                    "BF16 anchor domain"
                    if node.type.numerical_policy == conversion["input_policy"]
                    else "BF16 exp2 bounded domain"
                    if node.type.numerical_policy == contract.record["vpu_exp2"]["operand_policy"]
                    else "BF16 raw domain"
                )
                raise ValueError(
                    f"MXU0 runtime input {source} is outside the declared {domain} or pair size"
                )
            raise ValueError(
                f"MXU0 runtime input {source} is outside the declared FP8 domain or tile size"
            )
        if type(slot) is not int or row != {
            "source": source,
            "byte_address": base + slot * tile_bytes,
            "byte_length": count * tile_bytes,
            "preserve": True,
            "domain": node.type.numerical_policy,
        }:
            raise ValueError(
                "MXU0 runtime input address or domain differs from the checked ABI"
            )
        preloads.append((row["byte_address"], payload))

    constants = {binding.node_id: binding for binding in request.constants}
    constant_rows = plan.get("constants")
    if (
        not isinstance(constant_rows, list)
        or len(constant_rows) != len(constants)
        or {row.get("source") for row in constant_rows if isinstance(row, dict)}
        != set(constants)
    ):
        raise ValueError("MXU0 constant plan differs from declared compiler inputs")
    package_path = artifact / "constants.bin"
    if constants:
        if not package_path.is_file():
            raise ValueError("MXU0 constant package is missing")
        package = package_path.read_bytes()
        if manifest.get("constants_sha256") != _sha(package) or manifest.get(
            "constant_count"
        ) != len(constants):
            raise ValueError("MXU0 constant package identity changed")
    else:
        if (
            package_path.exists()
            or "constants_sha256" in manifest
            or "constant_count" in manifest
        ):
            raise ValueError("MXU0 package has an undeclared constant payload")
        package = b""
    used_offsets: set[int] = set()
    for row in constant_rows:
        binding = constants[row["source"]]
        offset, address = row.get("package_offset"), row.get("byte_address")
        if (
            type(offset) is not int
            or type(address) is not int
            or offset < 0
            or address < base
            or address + tile_bytes > base + window
            or (address - base) % tile_bytes
            or offset in used_offsets
        ):
            raise ValueError("MXU0 constant package has an invalid physical placement")
        used_offsets.add(offset)
        payload = package[offset : offset + tile_bytes]
        if (
            len(payload) != tile_bytes
            or binding.encoding != "fp8_e4m3"
            or binding.storage != "external_fp8"
            or payload.hex() != binding.data_hex
            or not admits_fp8_tile(
                contract,
                next(
                    node.type.numerical_policy
                    for node in request.nodes
                    if node.id == binding.node_id
                ),
                payload,
            )
            or row
            != {
                "source": binding.node_id,
                "byte_address": address,
                "byte_length": tile_bytes,
                "package_offset": offset,
                "encoding": binding.encoding,
                "sha256": _sha(payload),
                "preload": "before_each_invocation",
                "preserve": True,
            }
        ):
            raise ValueError(
                "MXU0 constant package differs from the checked compiler bytes"
            )
        preloads.append((address, payload))
    if used_offsets != set(range(0, len(package), tile_bytes)):
        raise ValueError("MXU0 constant package has gaps or trailing bytes")
    layout = contract.record["bf16_result_layout"]
    geometry = contract.record["geometry"]
    register_count = layout["register_count"]
    expected_outputs = []
    for index, (source, slot) in enumerate(zip(request.outputs, fixed_outputs)):
        node = request.node(source)
        is_fp8 = node.type.dtype == "fp8_e4m3"
        column_policies = {
            contract.record[f"vpu_col_{kind}"]["result_policy"]: contract.record[f"vpu_col_{kind}"]
            for kind in ("min", "max", "sum")
        }
        column_mode = column_policies.get(node.type.numerical_policy)
        expected_shape = (
            tuple(column_mode["output_shape"])
            if column_mode is not None
            else (geometry["array_rows"], geometry["array_cols"])
        )
        if node.type.shape != expected_shape:
            raise ValueError("MXU0 output shape differs from its selected policy")
        count = 1 if is_fp8 else register_count
        physical = (
            {
                "kind": (
                    conversion["pack_row_order"]
                    if node.type.numerical_policy
                    in conversion["pack_result_policies"].values()
                    else "row_major_fp8"
                ),
                "register_count": 1,
                "rows_per_register": geometry["array_rows"],
                "columns_per_register": geometry["array_cols"],
                "element_bytes": 1,
            }
            if is_fp8
            else {
                "kind": (
                    column_mode["physical_layout"]
                    if column_mode is not None else layout["kind"]
                ),
                "register_count": register_count,
                "rows_per_register": geometry["array_rows"],
                "columns_per_register": geometry["array_cols"] // register_count,
                "element_bytes": layout["element_bytes"],
            }
        )
        expected_outputs.append(
            {
                "source": source,
                "position": index,
                "byte_address": base + slot * tile_bytes,
                "byte_length": count * tile_bytes,
                "logical_shape": list(node.type.shape),
                "physical_layout": physical,
                "domain": node.type.numerical_policy,
            }
        )
    if plan.get("outputs") != expected_outputs:
        raise ValueError("MXU0 output plan differs from the checked physical ABI")
    if any(
        row["byte_address"] < base
        or row["byte_address"] + row["byte_length"] > base + window
        or (row["byte_address"] - base)
        % (row["physical_layout"]["register_count"] * tile_bytes)
        for row in expected_outputs
    ):
        raise ValueError(
            "MXU0 output exceeds or misaligns the selected external window"
        )
    if any(
        left["byte_address"] < right["byte_address"] + right["byte_length"]
        and right["byte_address"] < left["byte_address"] + left["byte_length"]
        for index, left in enumerate(expected_outputs)
        for right in expected_outputs[index + 1 :]
    ):
        raise ValueError("MXU0 output placements overlap")
    for index, (left, payload) in enumerate(preloads):
        if left < base or left + len(payload) > base + window:
            raise ValueError("MXU0 preload exceeds the selected external window")
        if any(
            left < right + len(other) and right < left + len(payload)
            for right, other in preloads[index + 1 :]
        ):
            raise ValueError("MXU0 input and constant preloads overlap")
        if any(
            left < output["byte_address"] + output["byte_length"]
            and output["byte_address"] < left + len(payload)
            for output in expected_outputs
        ):
            raise ValueError("MXU0 output overlaps a preserved input or constant")
    return tuple(preloads)
