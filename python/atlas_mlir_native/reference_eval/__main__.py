"""Inspect an authenticated export or evaluate an authenticated source capture."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from atlas_mlir_native.export_loader import load_export
from . import EvaluationError, evaluate, inspect_ops
from .source_io import capture_inputs, capture_sources, checked_path, digest, export_nodes, safetensor_constants


class _Audited(Exception):
    """An audit ends the capture branch without reference outputs."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--capture", action="store_true", help="evaluate a source capture instead of inspecting an export")
    parser.add_argument("--receipt-sha256", help="trusted capture receipt digest, required with --capture")
    parser.add_argument("--out", type=Path, help="fresh output directory for receipt and successful nonzero references")
    parser.add_argument("--target-rules", type=Path, help="declared source order, host-math supplier, and selected device rules JSON")
    parser.add_argument("--evaluate-export", action="store_true", help="evaluate authenticated export fixture zero independently")
    parser.add_argument("--compare-reference", type=Path, help="compare authenticated existing outputs in source producer order")
    parser.add_argument("--compare-receipt-sha256", help="trusted comparison reference.json digest")
    parser.add_argument("--selected-application", type=Path, help="application directory; reads authenticated placement decisions only")
    parser.add_argument("--application-manifest-sha256", help="trusted selected application manifest digest")
    parser.add_argument("--audit", action="store_true",
                        help="with --capture: report the rule for every source node and every refusal at once (coverage.py)")
    parser.add_argument("--inputs", type=Path, help="with --capture: .npz of user inputs input_000, ... to evaluate instead of the capture's")
    args = parser.parse_args()
    rules = json.loads(checked_path(args.target_rules).read_text()) if args.target_rules else {}
    record = {"compiler_decision_metadata_inputs": bool(args.selected_application),
              "compiler_arithmetic_or_emitted_program_inputs": False,
              "ordinary_framework_kernels_called": False}
    if args.target_rules:
        record["target_rules_sha256"] = digest(args.target_rules)
    implementation_paths = [Path(__file__).parent / name for name in
                            ("__init__.py", "evaluator.py", "source_io.py", "__main__.py", "host.py", "device.py", "export_io.py", "target_rules.py", "compare.py", "decisions.py", "placed_device.py", "stages.py", "coverage.py")]
    implementation_hashes = {p.name: digest(p) for p in implementation_paths}
    outputs = None
    comparison = None
    decisions = None
    supplier_hashes = {}
    config = rules.get("host_math", {})
    if config:
        binding = checked_path(config["binding_path"])
        supplier_paths = [binding, checked_path(config.get("policy_path", binding.parent / "host-unary-math-policy.json")),
                          checked_path(Path(config["build_dir"]) / "build-receipt.json")]
        supplier_hashes = {str(path): digest(path) for path in supplier_paths}
        record["host_math_identity"] = supplier_hashes
    try:
        if args.inputs and not args.capture:
            raise ValueError("--inputs requires --capture")
        if args.audit and not args.capture:
            raise ValueError("--audit requires --capture")
        if args.capture:
            if not args.receipt_sha256:
                raise ValueError("--capture requires --receipt-sha256")
            root, nodes, authentication = capture_sources(args.source, receipt_sha256=args.receipt_sha256)
            record.update(mode="source_capture", authentication=authentication,
                          source_trace_sha256=authentication["artifacts"]["frontend-trace.json"])
            if args.selected_application:
                from .decisions import SelectedDecisions
                decisions = SelectedDecisions(args.selected_application, args.application_manifest_sha256,
                                              root, args.receipt_sha256)
                record["compiler_decision_inputs"] = decisions.record
            constants = safetensor_constants(root / "weights.safetensors")
            inputs = capture_inputs(root, nodes, args.inputs)
            if args.inputs:
                # Supplied inputs replace only the capture's; every other source artifact stays authenticated above.
                record["inputs"] = {f"input_{i:03d}": sha256(v.raw.tobytes()).hexdigest() for i, v in enumerate(inputs.values())}
            if args.compare_reference:
                if not args.compare_receipt_sha256:
                    raise ValueError("comparison requires --compare-receipt-sha256")
                from .compare import OutputComparison
                comparison = OutputComparison(nodes, args.compare_reference, args.compare_receipt_sha256,
                                              source_trace_sha256=record["source_trace_sha256"])
            if args.audit:
                from .coverage import audit
                record["coverage"] = audit(nodes, constants, inputs, rules, decisions=decisions)
                record["status"] = "COVERED" if not record["coverage"]["refused"] else "GAPS"
                raise _Audited
            result = evaluate(nodes, constants, inputs, rules, observer=comparison, decisions=decisions)
            outputs = {f"output_{i}": v.raw for i, v in enumerate(result.outputs)}
            if not outputs or not any(np.count_nonzero(a) for a in outputs.values()):
                raise ValueError("empty or all-zero references refuse coverage")
            record.update(status="EXPECTED", outputs={key: {"shape": list(a.shape), "dtype": str(a.dtype),
                "nonzero": int(np.count_nonzero(a))} for key, a in outputs.items()})
        else:
            bundle = load_export(checked_path(args.source))
            record.update(mode="export_operator_inventory", metadata=bundle.metadata,
                          inventory=bundle.inventory)
            if bundle.graph_path is None:
                raise ValueError("export has no authenticated source graph")
            if args.evaluate_export:
                from .export_io import export_sources
                nodes, constants, inputs = export_sources(bundle)
                result = evaluate(nodes, constants, inputs, rules)
                values = result.outputs if isinstance(result.outputs, (list, tuple)) else [result.outputs]
                outputs = {f"output_{i}": v.raw for i, v in enumerate(values)}
                if not outputs or not any(np.count_nonzero(a) for a in outputs.values()):
                    raise ValueError("empty or all-zero references refuse coverage")
                record.update(mode="source_export", status="EXPECTED", outputs={key: {
                    "shape": list(a.shape), "dtype": str(a.dtype), "nonzero": int(np.count_nonzero(a))}
                    for key, a in outputs.items()})
            else:
                record.update(inspect_ops(export_nodes(bundle.graph_path), rules))
            record["qualification"] = ("independent source execution; selected-model arithmetic only, no silicon claim"
                                       if args.evaluate_export else
                                       "operator inventory only; argument domains and arithmetic are not qualified")
    except _Audited:
        pass
    except EvaluationError as error:
        record.update(status="REFUSED", node_id=error.node_id, target=error.target,
                      reason=error.reason, completed_node_ids=error.completed_node_ids)
    except (ValueError, KeyError, OSError) as error:
        record.update(status="REFUSED", reason=str(error))
    if any(digest(path) != expected for path, expected in supplier_hashes.items()):
        record.update(status="REFUSED", reason="host-math supplier identity changed during evaluation")
        outputs = None
    if any(digest(p) != implementation_hashes[p.name] for p in implementation_paths):
        record.update(status="REFUSED", reason="evaluator implementation changed during evaluation")
        outputs = None
    if decisions is not None:
        try:
            decisions.verify()
        except (ValueError, OSError) as error:
            record.update(status="REFUSED", reason=str(error))
            outputs = None
    if comparison is not None:
        record["comparison"] = comparison.record
    if args.out:
        out = checked_path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        if outputs is not None and record.get("status") == "EXPECTED":
            np.savez(out / "expected.npz", **outputs)
            record["expected_sha256"] = digest(out / "expected.npz")
        record["implementation_sha256"] = implementation_hashes
        (out / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "coverage"} if args.audit else record,
                     indent=2))
    return 0 if record.get("status") in ("EXPECTED", "COVERED") else 1


if __name__ == "__main__":
    sys.exit(main())
