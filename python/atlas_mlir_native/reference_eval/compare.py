"""Authenticated output comparison, never an evaluator input binding."""
import json
import numpy as np
from .source_io import checked_path, digest
from .evaluator import Tensor


class OutputComparison:
    """Compare output producers in source execution order, stopping on mismatch.

    Output ordinals retain source output-list positions. The first observed
    mismatch is not necessarily the lowest ordinal (logits often appear first).
    """
    def __init__(self, nodes, reference_dir, receipt_sha256, *, source_trace_sha256=None):
        root = checked_path(reference_dir)
        receipt = root / "reference.json"
        if digest(receipt) != receipt_sha256:
            raise ValueError("comparison reference receipt differs from trusted digest")
        metadata = json.loads(receipt.read_text())
        if source_trace_sha256 is not None and metadata.get("source_trace_sha256") != source_trace_sha256:
            raise ValueError("comparison source trace differs from authenticated capture")
        path = root / "expected.npz"
        if digest(path) != metadata["expected_sha256"]:
            raise ValueError("comparison expected archive differs from authenticated receipt")
        if metadata.get("status") != "EXPECTED":
            raise ValueError("comparison reference is not EXPECTED")
        refs = nodes[-1]["args"][0]
        if not isinstance(refs, list):
            raise ValueError("comparison requires a flat source output list")
        self.bindings = {}
        self.expected = {}
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {f"output_{i}" for i in range(len(refs))}:
                raise ValueError("comparison output inventory differs from source")
            for i, ref in enumerate(refs):
                name = f"output_{i}"
                self.expected[name] = archive[name].copy()
                self.bindings.setdefault(ref["node_id"], []).append((name, ref.get("value_id")))
        self.record = dict(reference_sha256=receipt_sha256, expected_sha256=metadata["expected_sha256"],
                           order="source-producer-order", compared=[], mismatch=None)

    def __call__(self, node, value):
        for name, value_id in self.bindings.get(node["id"], []):
            if isinstance(value, (list, tuple)):
                index = next(i for i, result in enumerate(node["results"]) if result["id"] == value_id)
                actual = value[index]
            else:
                actual = value
            expected = self.expected[name]
            if not isinstance(actual, Tensor) or actual.raw.dtype != expected.dtype or actual.raw.shape != expected.shape:
                raise ValueError("comparison output dtype/shape differs: " + name)
            different = np.flatnonzero(actual.raw.reshape(-1) != expected.reshape(-1))
            if different.size:
                offset = int(different[0])
                width = actual.raw.itemsize * 2
                mismatch = dict(tensor=name, node_id=node["id"], logical_dtype=actual.dtype,
                                index=list(np.unravel_index(offset, expected.shape)),
                                actual_bits=f"0x{int(actual.raw.flat[offset]):0{width}x}",
                                expected_bits=f"0x{int(expected.flat[offset]):0{width}x}")
                mismatch["index"] = [int(i) for i in mismatch["index"]]
                self.record["mismatch"] = mismatch
                raise ValueError("reference mismatch: " + json.dumps(mismatch))
            self.record["compared"].append(name)
