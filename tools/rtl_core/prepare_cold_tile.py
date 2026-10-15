"""Fresh arithmetic for one authenticated cold tile; canonical raw trace replay."""
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import stat
import numpy as np

ORACLE = "7976cb9496e31fe9fcfa2bcbbaa2a84c9015c5c471b6c5fef71f5ee94aacc8b0"
VALIDATOR = "7287428fafe48529dcf801333d58f12d8a2cc2707bdddbe2a58f5ee8109e5034"
REPLAY = "1a47464fbaa8e6235722b24d952ae2067791b0dbcf5572ec971f4b08d7a2d942"
_canonical = None


def preflight(support, path):
    path = support.safe_path(path)
    if path.suffix.lower() in {".npz", ".jar", ".pth"}:
        with zipfile.ZipFile(path) as archive:
            names = []
            for member in archive.infolist():
                name = member.filename
                if (not name or name.startswith(("/", "\\")) or "\\" in name or ":" in name
                        or any(part in {"", ".", ".."} or any(bad in part.lower() for bad in ("hammer", "vlsi"))
                               for part in name.rstrip("/").split("/"))
                        or stat.S_ISLNK(member.external_attr >> 16)):
                    raise ValueError("Unadmitted archive member path")
                names.append(name)
            if len(names) != len(set(names)):
                raise ValueError("Duplicate archive members")
    return path


def validate_descriptor(value):
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("source_kind") != "cold_operator_tile"
            or type(value.get("tile_index")) is not int or value["tile_index"] < 0
            or type(value.get("scale_codes")) is not dict or set(value["scale_codes"]) != {"A", "B"}
            or any(type(v) is not int or not 1 <= v <= 254 for v in value["scale_codes"].values())
            or type(value.get("output_rescale_exponent")) is not int
            or not -126 <= value["output_rescale_exponent"] <= 127
            or type(value.get("factor_bf16")) is not int):
        raise ValueError("Invalid explicit cold-tile descriptor types")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def selected_bytes(arithmetic, descriptor, original, source, encoded, target, initial):
    """Check GLOBAL scales and independently rebuild the selected full-K tile."""
    index = descriptor["tile_index"]
    if type(index) is not int or not 0 <= index < len(original["tiles"]):
        raise ValueError("Selected tile index outside parent")
    tile = original["tiles"][index]
    if tile != descriptor["tile"]:
        raise ValueError("Selected tile descriptor differs from parent")
    codes, exponent = descriptor["scale_codes"], descriptor["output_rescale_exponent"]
    if (set(codes) != {"A", "B"} or any(type(v) is not int or not 1 <= v <= 254 for v in codes.values())
            or type(exponent) is not int or exponent != sum(v - 127 for v in codes.values())
            or not -126 <= exponent <= 127 or exponent != original["output_rescale_exponent"]
            or descriptor["factor_bf16"] != (exponent + 127) << 7
            or original["recipe"] != "atlas_mxu0_fp8_tensor_pow2_bf16acc_v1"
            or original["output_rescale_execution"] != "VLI_ALL_and_VMUL_BF16"
            or any(original["conversion"][name]["scale_code"] != codes[name] for name in codes)):
        raise ValueError("Global scale/factor binding differs")
    r0, r1 = tile["row_range"]
    c0, c1 = tile["column_range"]
    k = original["shape"][1]
    if tile["k_range"] != [0, k] or r1-r0 != 32 or c1-c0 != 32:
        raise ValueError("Selected tile must retain full K and 32x32 output")
    a, b = encoded["A"][r0:r1], encoded["B"][:, c0:c1]
    for name, words, actual in (("A", source["A_bf16"][r0:r1], a),
                                ("B", source["B_bf16"][:, c0:c1], b)):
        if words.dtype != np.uint16 or words.shape != actual.shape or actual.dtype != np.uint8:
            raise ValueError("Selected source/encoded operand layout differs")
        table = {int(v): arithmetic.pack_bf16(int(v), codes[name]) for v in np.unique(words)}
        packed = np.array([table[int(v)] for v in words.flat], dtype=np.uint8).reshape(words.shape)
        if not np.array_equal(packed, actual):
            raise ValueError("Independent source-to-global-encoded packing differs")
    payload = b"".join(a[:, inner:inner+32].tobytes() for inner in range(0, k, 32))
    payload += b"".join(b[inner:inner+32].tobytes() for inner in range(0, k, 32))
    expected = arithmetic.output_halves([[arithmetic.rescale_bf16(
        arithmetic.mxu0_dot(a[row], b[:, col]), exponent) for col in range(32)] for row in range(32)])
    if initial != payload + b"\xa5"*256 + b"\xcd"*2048 + b"\x5a"*256:
        raise ValueError("Selected encoded DRAM payload/guards differ")
    if arithmetic.output_halves(target[r0:r1, c0:c1].view(np.uint32) >> 16) != expected:
        raise ValueError("Fresh selected arithmetic differs from parent aggregate actual")
    return expected, {"dram_base": 1 << 32, "input_bytes": len(payload),
                      "output_offset": len(payload)+256, "output_bytes": 2048,
                      "total_dram_bytes": len(initial)}


def prepare(support, source, receipt_sha, oracle_path, out, controls, *, reuse_model_arithmetic=False, model_root=None):
    global _canonical
    if controls or reuse_model_arithmetic or model_root is None:
        raise ValueError("Cold tile requires fresh selected arithmetic, explicit model root, no controls/reuse")
    source, out = support.safe_path(source), support.safe_path(out)
    records = [support.identity(preflight(support, source / "descriptor.json"), receipt_sha)]
    descriptor = json.loads((source / "descriptor.json").read_text())
    validate_descriptor(descriptor)
    def bind(path, sha=None):
        record = support.identity(preflight(support, path), sha); records.append(record); return Path(record["path"])
    root = support.safe_path(descriptor["producer_root"])
    if set(descriptor["producer_sha256"]) != {"run_model_contraction.py", "run_model_tile.py", "atlas_tile_arithmetic.py", "validate_model_contraction.py"}:
        raise ValueError("Incomplete producer dependency tuple")
    for name, sha in descriptor["producer_sha256"].items():
        if name not in {"run_model_contraction.py", "run_model_tile.py", "atlas_tile_arithmetic.py", "validate_model_contraction.py"}:
            raise ValueError("Unexpected producer dependency")
        bind(root / "tools" / name, sha)
    bind(root / "tools/atlas_tile_arithmetic.py", ORACLE)
    bind(root / "tools/validate_model_contraction.py", VALIDATOR)
    bind(oracle_path, ORACLE)
    sys.path.insert(0, str(root / "tools"))
    arithmetic = load(root / "tools/atlas_tile_arithmetic.py", "atlas_tile_arithmetic")
    load(root / "tools/run_model_tile.py", "run_model_tile")
    validator = load(root / "tools/validate_model_contraction.py", "cold_tile_validator")
    parent = bind(descriptor["parent_receipt"], descriptor["parent_sha256"])
    original = json.loads(parent.read_text())
    for path in (support.safe_path(descriptor["native_bundle"]) / "operands.npz",
                 *(parent.parent / name for name in ("source.npz", "result.npz", "encoded.npz"))):
        preflight(support, path)
    native, arrays, target, admitted, _ = validator.admit_contraction(
        receipt_path=parent, receipt_sha256=descriptor["parent_sha256"],
        native_bundle=descriptor["native_bundle"], native_manifest_sha256=descriptor["native_manifest_sha256"],
        producer_root=root, model_root=model_root)
    records.extend(admitted)
    if descriptor["producer_sha256"] != {**original["code_before"], "validate_model_contraction.py": VALIDATOR}:
        raise ValueError("Descriptor producer tuple differs from admitted parent")
    encoded_path = bind(parent.parent / "encoded.npz", original["artifacts"]["encoded.npz"])
    with zipfile.ZipFile(encoded_path) as archive:
        members = archive.infolist()
        if sorted(v.filename for v in members) != ["A.npy", "B.npy"] or sum(v.file_size for v in members) > 32*1024**2:
            raise ValueError("Unexpected bounded encoded archive members")
    with np.load(encoded_path, allow_pickle=False) as archive:
        encoded = {name: archive[name].copy() for name in ("A", "B")}
    m, k, n = original["shape"]
    if encoded["A"].shape != (m, k) or encoded["B"].shape != (k, n):
        raise ValueError("Global encoded operand shape differs")
    if type(descriptor["tile_index"]) is not int or not 0 <= descriptor["tile_index"] < len(original["tiles"]):
        raise ValueError("Selected tile index outside parent")
    child = parent.parent / f"tile-{descriptor['tile_index']:04d}"
    tile = descriptor["tile"]
    initial_path = bind(child / "initial-dram.bin", tile["initial_dram_sha256"])
    expected, memory = selected_bytes(arithmetic, descriptor, original, arrays, encoded, target, initial_path.read_bytes())
    if any(original["memory"][key] != value for key, value in {
            "dram_base": memory["dram_base"], "input_bytes_per_tile": memory["input_bytes"],
            "output_offset": memory["output_offset"], "guard_bytes_each": 256,
            "dram_working_set_bytes": memory["total_dram_bytes"]}.items()):
        raise ValueError("Parent cold-tile memory contract differs")
    if expected != bind(child / "expected-bf16-halves.bin", tile["expected_sha256"]).read_bytes() or expected != bind(child / "actual-bf16-halves-0.bin", tile["actual_sha256"]).read_bytes():
        raise ValueError("Fresh selected arithmetic differs from retained expected/actual physical bytes")
    program = bind(parent.parent / "program.bin", original["fixed_image_sha256"])
    image = program.read_bytes()
    if len(image) != 4*original["instruction_words"] or image[-4:] != b"\x73\0\0\0":
        raise ValueError("Malformed literal raw instruction image")
    for path, name in ((program, "program.bin"), (initial_path, "initial-dram.bin")):
        support.copy_source(path, out / name, support.identity(path)["sha256"]); bind(out / name)
    canonical = bind(descriptor["canonical_replay"], REPLAY)
    support.copy_source(canonical, out / "canonical_replay.py", REPLAY); bind(out / "canonical_replay.py", REPLAY)
    _canonical = load(out / "canonical_replay.py", "canonical_cold_tile_replay")
    validator.recheck(records)
    return {"schema_version": 1, "memory": memory, "initial": str(out / "initial-dram.bin"),
            "requested_dma": {"load_bytes": memory["input_bytes"], "store_bytes": 2048},
            "cases": [{"id": "positive", "program": str(out / "program.bin"), "expected": expected.hex(), "actual_oracle": expected.hex()}],
            "arithmetic_evidence": {"mode": "fresh_selected_cold_tile_fraction_rebuild", "fresh_arithmetic_rebuild": True,
                "whole_operator_arithmetic_rebuilt": False, "parent": str(parent), "tile_index": descriptor["tile_index"],
                "scale_codes": descriptor["scale_codes"], "output_rescale_exponent": descriptor["output_rescale_exponent"]}}, records


def replay(data, case, trace_path, actual_path, program_path):
    return _canonical.replay(data, case, trace_path, actual_path, program_path)
