"""Construct execution-plan obligations and publish the admitted device program.

Compiler-side admission/replay retains its diagnostic scope. Independent reference
and evaluator authority remain separate; no hardware execution is established.
"""

from __future__ import annotations

from pathlib import Path

from merlin.compiler.semantic.search import (
    SearchResult,
)
from merlin.compiler.semantic.snapshot import NativeSnapshot

from .native import _MaterializedCandidate, _sha256
from .planning import TilePlan
from .policy import (
    BF16_ADD_ATTRS,
    BF16_EXP_ATTRS,
    BF16_MUL_ATTRS,
    BF16_RECIP_ATTRS,
    BF16_ROW_MAX_ATTRS,
    BF16_ROW_REDUCTION_ATTRS,
    BF16_ROW_SUM_ATTRS,
    BF16_SUB_ATTRS,
    BF16_UNARY_ATTRS,
    DRAM_BASE,
    MXU0_INPUT_DOMAIN,
    MXU0_POLICY,
    PACK_POLICY,
    TARGET_IDENTITY,
    TILE_BYTES,
)


def publish_result(
    selected: SearchResult,
    materialized: _MaterializedCandidate,
    plan: TilePlan,
    *,
    snapshot: NativeSnapshot,
    destination: Path,
    binding_source: Path,
    publish_program,
    source_pins: dict[str, str],
) -> dict[str, object]:
    request = plan.request
    fixed_inputs = plan.fixed_inputs
    bf16_tile = plan.bf16_tile
    constant_payloads = plan.constant_payloads
    copy_transpose = plan.copy_transpose
    input_slots = plan.input_slots
    inventory_path = plan.inventory_path
    mxu = plan.mxu
    mxu_binary = plan.mxu_binary
    mxu_dag = plan.mxu_dag
    mxu_inputs = plan.mxu_inputs
    mxu_k_chain = plan.mxu_k_chain
    mxu_relu = plan.mxu_relu
    mxu_unit = plan.mxu_unit
    output_slots = plan.output_slots
    pack = plan.pack
    pack_domain = plan.pack_domain
    pack_scale_code = plan.pack_scale_code
    pack_scale_codes = plan.pack_scale_codes
    pipeline = plan.pipeline
    relu_pack = plan.relu_pack
    scaled_code = plan.scaled_code
    simple_mxu_mode = plan.simple_mxu_mode
    transpose = plan.transpose
    transpose_inputs = plan.transpose_inputs
    has_relu = any(
        selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
        in {"atlas.vpu_relu_bf16_tile", "atlas.vpu_relu_bf16_from_mxu0_tile"}
        for value in selected.graph.values
    )
    has_exp = any(node.op == "exp_bf16" for node in request.nodes)
    has_recip = any(node.op == "recip_bf16" for node in request.nodes)
    has_add = any(node.op == "add_bf16" for node in request.nodes)
    has_sub = any(node.op == "sub_bf16" for node in request.nodes)
    has_mul = any(node.op == "mul_bf16" for node in request.nodes)
    has_row_max = any(node.op == "row_max_bf16" for node in request.nodes)
    has_row_sum = any(node.op == "row_sum_bf16" for node in request.nodes)
    selected_registers = materialized.selected_registers
    virtual = materialized.virtual
    machine = materialized.machine
    program = materialized.program
    words = materialized.words
    tool_hashes = materialized.tool_hashes
    input_address = materialized.input_address
    output_address = materialized.output_address

    destination.mkdir(parents=True, exist_ok=False)
    constant_rows = []
    if constant_payloads:
        (destination / "constants").mkdir()
    for index, node_id in enumerate(sorted(constant_payloads)):
        payload = constant_payloads[node_id]
        path = f"constants/constant-{index:03d}.bin"
        (destination / path).write_bytes(payload)
        constant_rows.append(
            {
                "node_id": node_id,
                "address": DRAM_BASE + fixed_inputs[node_id] * TILE_BYTES,
                "bytes": len(payload),
                "encoding": "bf16-le-raw",
                "physical_layout": "bf16_pair_halves",
                "path": path,
                "sha256": _sha256(payload),
            }
        )
    semantic_nodes = {node.id: node for node in request.nodes}

    def output_numerical_policy(output_id: str) -> str:
        node = semantic_nodes[output_id]
        while node.op == "identity":
            node = semantic_nodes[node.inputs[0]]
        if node.op == "relu":
            return "bf16-sign-clamp-v1"
        if node.op in BF16_UNARY_ATTRS:
            return BF16_UNARY_ATTRS[node.op][1][1]
        if node.op == "add_bf16":
            return BF16_ADD_ATTRS[1][1]
        if node.op == "sub_bf16":
            return BF16_SUB_ATTRS[1][1]
        if node.op == "mul_bf16":
            return BF16_MUL_ATTRS[1][1]
        if node.op in BF16_ROW_REDUCTION_ATTRS:
            return BF16_ROW_REDUCTION_ATTRS[node.op][1][1]
        return node.type.numerical_policy

    output_policies = [output_numerical_policy(output) for output in request.outputs]
    boundary_domain_obligations = []
    boundary_requirements: set[tuple[str, str]] = set()

    def require_boundary_domain(source_id: str, domain: str) -> None:
        key = (source_id, domain)
        if key in boundary_requirements:
            return
        boundary_requirements.add(key)
        boundary_domain_obligations.append(
            {
                "input": source_id,
                "required_domain": domain,
                "status": "caller_premise_unchecked",
            }
        )

    for node in request.nodes:
        own_domain = dict(node.attrs).get("input_domain")
        if node.op == "input" and own_domain is not None:
            require_boundary_domain(node.id, own_domain)
        elif own_domain is not None:
            for source_id in node.inputs:
                if semantic_nodes[source_id].op == "input":
                    require_boundary_domain(source_id, own_domain)
    producer_domain_obligations = [
        {
            "producer": source_id,
            "consumer": node.id,
            "required_domain": dict(node.attrs)["input_domain"],
            "status": "caller_premise_unchecked",
        }
        for node in request.nodes
        if "input_domain" in dict(node.attrs)
        for source_id in node.inputs
        if semantic_nodes[source_id].op != "input"
    ]
    if mxu_dag:
        producer_domain_obligations.extend(
            {
                "producer": source_id,
                "consumer": node.id,
                "required_domain": MXU0_INPUT_DOMAIN,
                "status": "caller_premise_unchecked",
            }
            for node in request.nodes
            if node.op == "matmul"
            for source_id in node.inputs
            if semantic_nodes[transpose_inputs.get(source_id, source_id)].op
            == "pack_fp8"
        )

    plan = {
        "schema": "atlas.native_tile_execution_plan.v8",
        "scope": "diagnostic_selected_standalone_core",
        "input_address": input_address,
        "output_address": output_address,
        "input_addresses": [DRAM_BASE + slot * TILE_BYTES for slot in input_slots],
        "output_addresses": [DRAM_BASE + slot * TILE_BYTES for slot in output_slots],
        "input_bytes": None
        if (mxu_binary or mxu_dag)
        else TILE_BYTES
        if (bf16_tile or pack)
        else 1024,
        "weight_address": (
            DRAM_BASE + fixed_inputs[next(iter(mxu_inputs.values()))[1]] * TILE_BYTES
            if mxu_dag
            and len(mxu_inputs) == 1
            and next(iter(mxu_inputs.values()))[1] in fixed_inputs
            else DRAM_BASE + input_slots[1] * TILE_BYTES
            if (mxu or mxu_k_chain or mxu_binary)
            else None
        ),
        "weight_addresses": (
            [
                DRAM_BASE + input_slots[index] * TILE_BYTES
                for index in range(1, len(input_slots), 2)
            ]
            if mxu_k_chain
            else None
        ),
        "weight_bytes": 1024 if (mxu or mxu_k_chain or mxu_binary or mxu_dag) else None,
        "output_bytes": TILE_BYTES
        if (bf16_tile or mxu or mxu_k_chain or mxu_binary or mxu_dag)
        else 1024,
        "shape": [32, 32],
        "dtype": "bf16"
        if (bf16_tile or mxu or mxu_k_chain or mxu_binary or mxu_dag)
        else "fp8_e4m3",
        "output_physical_layouts": [
            "bf16_pair_halves" if semantic_nodes[output].type.dtype == "bf16" else None
            for output in request.outputs
        ],
        "operation": (
            "mxu0_bf16_dag"
            if mxu_dag
            else "mxu0_bf16_binary"
            if mxu_binary
            else f"mxu0_k{len(input_slots) // 2}"
            if mxu_k_chain
            else "mxu0_relu"
            if mxu_relu
            else "bf16_relu_pack"
            if relu_pack
            else "bf16_mixed"
            if sum(
                (
                    has_add,
                    has_sub,
                    has_mul,
                    has_row_max,
                    has_row_sum,
                    has_exp,
                    has_recip,
                )
            )
            > 1
            else "bf16_add_relu"
            if has_add and has_relu
            else "bf16_add"
            if has_add
            else "bf16_sub_relu"
            if has_sub and has_relu
            else "bf16_sub"
            if has_sub
            else "bf16_mul_relu"
            if has_mul and has_relu
            else "bf16_mul"
            if has_mul
            else "bf16_row_max_relu"
            if has_row_max and has_relu
            else "bf16_row_max"
            if has_row_max
            else "bf16_row_sum_relu"
            if has_row_sum and has_relu
            else "bf16_row_sum"
            if has_row_sum
            else "bf16_exp"
            if has_exp
            else "bf16_recip"
            if has_recip
            else "bf16_relu"
            if has_relu
            else "bf16_movement"
            if bf16_tile
            else "bf16_to_fp8_scaled_pack"
            if pack and scaled_code is not None
            else "bf16_to_fp8_unit_pack"
            if pack
            else "copy_transpose"
            if copy_transpose
            else "transpose"
            if transpose
            else f"mxu{mxu_unit}_reset"
        ),
        "numerical_policy": (
            "per-operation-declared"
            if (mxu_binary or mxu_dag)
            else MXU0_POLICY
            if mxu_k_chain
            else request.nodes[2].type.numerical_policy
            if mxu
            else request.nodes[1].type.numerical_policy
            if pack and scaled_code is not None
            else PACK_POLICY
            if pack
            else output_policies[0]
            if len(set(output_policies)) == 1
            else "per-output"
        ),
        "output_numerical_policies": output_policies,
        "input_domain": (
            "per-operation-declared"
            if (mxu_binary or mxu_dag)
            else MXU0_INPUT_DOMAIN
            if mxu_k_chain
            else simple_mxu_mode[1]
            if mxu
            else pack_domain
            if pack and scaled_code is not None
            else "bf16-seven-powers-and-zero-v1"
            if pack
            else "per-operation-declared"
            if sum(
                (
                    has_add,
                    has_sub,
                    has_mul,
                    has_row_max,
                    has_row_sum,
                    has_exp,
                    has_recip,
                )
            )
            > 1
            else BF16_ADD_ATTRS[0][1]
            if has_add
            else BF16_SUB_ATTRS[0][1]
            if has_sub
            else BF16_MUL_ATTRS[0][1]
            if has_mul
            else BF16_ROW_MAX_ATTRS[0][1]
            if has_row_max
            else BF16_ROW_SUM_ATTRS[0][1]
            if has_row_sum
            else BF16_EXP_ATTRS[0][1]
            if has_exp
            else BF16_RECIP_ATTRS[0][1]
            if has_recip
            else "finite-normal-or-zero-bf16"
            if has_relu
            else "all-bit-patterns"
        ),
        "runtime_domain_guard": False,
        "boundary_domain_obligations": boundary_domain_obligations,
        "producer_domain_obligations": producer_domain_obligations,
    }
    if mxu_binary or mxu_dag:
        plan["input_payload_bytes"] = [
            TILE_BYTES if node.type.dtype == "bf16" else 1024
            for node in request.nodes
            if node.op in {"input", "constant"}
        ]
    if constant_rows:
        plan["constant_inputs"] = constant_rows
    if mxu_dag:
        plan["weight_addresses"] = [
            DRAM_BASE + fixed_inputs[weight] * TILE_BYTES
            if weight in fixed_inputs
            else None
            for _, weight in mxu_inputs.values()
        ]
        plan["weight_source_nodes"] = [weight for _, weight in mxu_inputs.values()]
        plan["transpose_source_nodes"] = [
            {"result": result, "source": source}
            for result, source in transpose_inputs.items()
        ]
        plan["interlayer_pack_scale_codes"] = [
            {"producer": node.id, "scale_code": pack_scale_codes[node.id]}
            for node in request.nodes
            if node.id in pack_scale_codes
        ]
    if pack:
        plan["pack_scale_code"] = pack_scale_code
    assert selected.graph is not None and selected.allocation is not None
    assert selected.rules is not None and selected.candidate is not None
    selection = {
        "schema": "merlin.native_selected_candidate.v1",
        "engine": "merlin_native",
        "request_digest": request.digest(),
        "target_identity": TARGET_IDENTITY,
        "candidate_digest": selected.candidate.digest(),
        "check_fingerprint": selected.check_fingerprint,
        "source_outputs": list(request.outputs),
        "graph_outputs": list(selected.graph.outputs),
        "order": list(selected.allocation.order),
        "issue_times": [list(item) for item in selected.allocation.issue_times],
        "schedule_scope": "native_candidate_virtual",
        "physical_assignment_scope": "mandatory_qualification",
        "candidate_attempts": selected.candidate_attempts,
        "ordering_attempts": selected.ordering_attempts,
        "rejected_materialization": selected.rejected_materialization,
        "values": [
            {
                "id": value.id,
                "symbol": value.symbol,
                "descriptor": selected.rules.symbols[value.symbol]
                .get("descriptor", {})
                .get("name"),
                "source_node": value.source_node,
                "kind": value.kind,
                "storage": value.storage,
                "extent": value.extent,
                "children": list(value.children),
                "validity": [condition.record() for condition in value.validity],
                "input_read_offsets": list(value.input_read_offsets),
                "completion_offset": value.completion_offset,
                "in_place_inputs": list(value.in_place_inputs),
                "preserve_input": value.preserve_input,
                "address": selected.allocation.addresses[value.id],
            }
            for value in selected.graph.values
        ],
    }
    manifest: dict[str, object] = {
        "schema": "atlas.native_tile_compilation.v6",
        "lowering_pipeline": pipeline,
        "semantic_qualification_pipeline": "mandatory",
        "artifact_kind": "program",
        "engine": "merlin_native",
        "qualification_scope": "diagnostic_only",
        "compile_mode": "diagnostic",
        "request_digest": request.digest(),
        "target_identity": TARGET_IDENTITY,
        "native_check_fingerprint": selected.check_fingerprint,
        "native_snapshot_sha256": _sha256(
            (snapshot.root / "manifest.json").read_bytes()
        ),
        "binding_implementation_sources": source_pins,
        "binding_source_sha256": _sha256(binding_source.read_bytes()),
        "target_inventory_sha256": _sha256(inventory_path.read_bytes()),
        "atlas_opt_sha256": tool_hashes[0],
        "atlas_emit_sha256": tool_hashes[1],
        "candidate_attempts": selected.candidate_attempts,
        "ordering_attempts": selected.ordering_attempts,
        "rejected_materialization": selected.rejected_materialization,
        "selected_registers": selected_registers,
        "selected_registers_scope": "mandatory_qualification",
    }
    return publish_program(
        destination,
        words=words,
        virtual=virtual,
        machine=machine,
        program=program,
        plan=plan,
        selection=selection,
        metadata=manifest,
        constant_rows=constant_rows,
        qualification_machine=materialized.qualification_machine,
        qualification_program=materialized.qualification_program,
    )
