"""Read-only admission of a pinned contraction's actual physical model outputs.

This trusted support edge does not execute a model or establish native fidelity.
The caller owns the native operation/site and must pin this loader's source.
"""

import json
from pathlib import Path
import subprocess
import sys
import zipfile

from atlas_tile_arithmetic import output_halves
from run_model_tile import (MODEL_PIN, RECIPES, safe_path, sha256,
                            validate_archive_members, validate_arrays)


def physical_path(value):
    path = Path(value)
    if ".." in path.parts or any(bad in part.lower() for part in path.parts
                                 for bad in ("hammer", "vlsi")):
        raise ValueError("Prohibited or parent-traversing path")
    return safe_path(path)


def bind(path, records, expected=None):
    path = physical_path(path)
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Bound file hash differs: {path}")
    records.append({"path": str(path), "sha256": actual})
    return path


def recheck(records):
    for record in records:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Admission file changed during use")


def load_arrays(path, *, result=False):
    import numpy as np
    path = physical_path(path)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if result:
            required = {"target_C.npy", "actual_bits.npy", "expected_bits.npy"}
            names = [member.filename for member in members]
            if (len(names) != len(set(names)) or not required <= set(names) or
                    set(names) - required - {"raw_accumulator_bits.npy"} or
                    sum(member.file_size for member in members) > 32 * 1024 * 1024):
                raise ValueError("Invalid bounded result archive members")
        else:
            validate_archive_members(members)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def admit_result_arrays(arrays, shape):
    import numpy as np
    required = {"target_C", "actual_bits", "expected_bits"}
    if not required <= arrays.keys() or arrays.keys() - required - {"raw_accumulator_bits"}:
        raise ValueError("Unexpected result schema")
    target = arrays["target_C"]
    if target.shape != shape or target.dtype != np.float32 or not np.isfinite(target).all():
        raise ValueError("Invalid target float32 view")
    for name in arrays.keys() - {"target_C"}:
        words = arrays[name]
        if (words.shape != shape or words.dtype != np.uint16 or
                np.any((words & 0x7F80) == 0x7F80)):
            raise ValueError("Invalid finite BF16 result words")
    if (not np.array_equal(arrays["actual_bits"], arrays["expected_bits"]) or
            not np.array_equal(target.view(np.uint32), arrays["actual_bits"].astype(np.uint32) << 16)):
        raise ValueError("Actual/oracle words or exact target view disagree")
    return target


def admit_tile_bytes(actual_bits, tile, payload):
    r0, r1 = tile["row_range"]
    c0, c1 = tile["column_range"]
    if output_halves(actual_bits[r0:r1, c0:c1]) != payload:
        raise ValueError("Aggregate words disagree with actual physical tile bytes")


def admit_bookends(receipt, source_hashes):
    before = receipt["model_binding_before"]
    after = receipt["source_bookends"]
    code = receipt["code_before"]
    if (receipt.get("status") != "passed_model_only" or "error" in receipt or
            receipt.get("recipe") not in RECIPES.values() or
            receipt.get("native_fidelity_claim") is not False or
            receipt.get("raw_image_execution_claim") is not False or
            receipt.get("invocations") != "serialized_fresh_core_instances_one_python_process" or
            receipt["source_inputs"] != source_hashes or
            after.get("unchanged") is not True or
            after["input_before"] != source_hashes or after["input_after"] != source_hashes or
            before != after["model_after"] or before["revision"] != MODEL_PIN or
            before["tracked_source_diff_exit"] != 0 or
            not before["imported_source_sha256"] or
            receipt["model"]["revision"] != MODEL_PIN or
            receipt["model"]["numerics"] != "rtl" or
            receipt["model"]["backend"] != "curve" or
            receipt["model"]["curve_sha256"] != before["curve_sha256"] or
            receipt["model"]["imported_source_sha256"] != before["imported_source_sha256"] or
            code != receipt["code_after"] or set(code) != {
                "run_model_contraction.py", "run_model_tile.py", "atlas_tile_arithmetic.py"}):
        raise ValueError("Missing or changed producer source/model/input bookends")


def model_source_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "npu_model":
        raise ValueError("Invalid relative model source path")
    return physical_path(root / path)


def admit_contraction(*, receipt_path, receipt_sha256, native_bundle,
                      native_manifest_sha256, producer_root, model_root):
    """Return unchanged source arrays and authenticated POST-VPU physical output.

    Raw accumulator words remain an oracle diagnostic and are never returned as
    the replacement. All files read here are rechecked before the return; callers
    must recheck the returned records after using the admitted data.
    """
    import numpy as np
    records = []
    producer_root, model_root = physical_path(producer_root), physical_path(model_root)
    for name in ("atlas_tile_arithmetic", "run_model_tile"):
        if physical_path(sys.modules[name].__file__) != producer_root / "tools" / (name + ".py"):
            raise ValueError("Imported provider helper outside selected producer root")
    if physical_path(__file__) != producer_root / "tools/validate_model_contraction.py":
        raise ValueError("Validator loaded outside selected producer root")
    receipt_path = bind(receipt_path, records, receipt_sha256)
    native_bundle = physical_path(native_bundle)
    manifest_path = bind(native_bundle / "manifest.json", records, native_manifest_sha256)
    native = json.loads(manifest_path.read_text())
    source_path = bind(native_bundle / "operands.npz", records, native["npz_sha256"])
    source = load_arrays(source_path)
    a, b, c = validate_arrays(source, native["source_dtype"], matrix=True)
    shape = (a.shape[0], b.shape[1])
    receipt = json.loads(receipt_path.read_text())
    admit_bookends(receipt, {"input": native["npz_sha256"], "manifest": native_manifest_sha256})
    if receipt["shape"] != [a.shape[0], a.shape[1], b.shape[1]]:
        raise ValueError("Producer/source contraction shape differs")
    for name, digest in receipt["code_before"].items():
        bind(producer_root / "tools" / name, records, digest)
    for name in ("validate_model_contraction.py",):
        bind(producer_root / "tools" / name, records)
    before = receipt["model_binding_before"]
    if (receipt["model"]["module_file"] != str(model_root / "npu_model/__init__.py") or
            subprocess.check_output(["git", "-C", str(model_root), "rev-parse", "HEAD"], text=True).strip() != MODEL_PIN):
        raise ValueError("Selected model root/pin differs")
    if subprocess.run(["git", "-C", str(model_root), "diff", "--quiet", "HEAD", "--", "npu_model",
                       ":(exclude,icase)**/*hammer*", ":(exclude,icase)**/*vlsi*"]).returncode:
        raise ValueError("Selected model has tracked source changes or failed diff")
    for name, digest in before["imported_source_sha256"].items():
        bind(model_source_path(model_root, name), records, digest)
    bind(model_root / "npu_model/configs/memory_curves/ee290sim_vcs_probe.json", records, before["curve_sha256"])
    artifact_names = {"encoded.npz", "program.S", "program.bin", "program.hex", "result.npz", "source-manifest.json", "source.npz"}
    if set(receipt["artifacts"]) != artifact_names:
        raise ValueError("Unexpected producer artifact path/schema")
    directory = receipt_path.parent
    for name, digest in receipt["artifacts"].items():
        bind(directory / name, records, digest)
    if json.loads((directory / "source-manifest.json").read_text()) != native:
        raise ValueError("Producer retained a different native manifest")
    retained = load_arrays(directory / "source.npz")
    if set(retained) != set(source) or any(
            retained[key].dtype != source[key].dtype or retained[key].shape != source[key].shape or
            retained[key].tobytes() != source[key].tobytes() for key in source):
        raise ValueError("Producer retained different source operand bytes")
    result = load_arrays(directory / "result.npz", result=True)
    target = admit_result_arrays(result, shape)
    coverage = np.zeros(shape, dtype=np.bool_)
    if len(receipt["tiles"]) != shape[0] * shape[1] // 1024:
        raise ValueError("Wrong number of physical output tiles")
    cycles = 0
    for index, tile in enumerate(receipt["tiles"]):
        r0, r1 = tile["row_range"]
        c0, c1 = tile["column_range"]
        if (any(type(v) is not int for v in (r0, r1, c0, c1)) or
                not 0 <= r0 < r1 <= shape[0] or not 0 <= c0 < c1 <= shape[1] or
                r1 - r0 != 32 or c1 - c0 != 32 or r0 % 32 or c0 % 32 or
                tile["k_range"] != [0, a.shape[1]] or coverage[r0:r1, c0:c1].any()):
            raise ValueError("Tiles overlap, escape bounds or omit the full reduction")
        execution = tile["execution"]
        if (execution["halt_reason"] != "ecall" or
                any(execution[key] is not True for key in (
                    "all_output_bytes_exact", "drained_before_halt_observation", "input_and_guards_preserved")) or
                execution["observed"]["load_bytes"] != 64 * a.shape[1] or
                execution["observed"]["store_bytes"] != 2048 or execution["cycles"] <= 0):
            raise ValueError("Tile lacks exact guarded/drained completion and expected traffic")
        child = directory / f"tile-{index:04d}"
        child_receipt = bind(child / "receipt.json", records)
        if json.loads(child_receipt.read_text()) != tile:
            raise ValueError("Aggregate/per-tile receipts differ")
        for name, key in (("actual-bf16-halves-0.bin", "actual_sha256"),
                          ("expected-bf16-halves.bin", "expected_sha256"),
                          ("initial-dram.bin", "initial_dram_sha256"),
                          ("cycles-0.jsonl", "cycles_trace_sha256"), ("perfetto-0.json", "perfetto_sha256")):
            bind(child / name, records, tile[key])
        if tile["actual_sha256"] != tile["expected_sha256"]:
            raise ValueError("Actual/oracle physical payload hashes differ")
        admit_tile_bytes(result["actual_bits"], tile, (child / "actual-bf16-halves-0.bin").read_bytes())
        coverage[r0:r1, c0:c1] = True
        cycles += execution["cycles"]
    if not coverage.all() or cycles != receipt["total_model_cycles"]:
        raise ValueError("Incomplete tile coverage or inconsistent model-cycle sum")
    recheck(records)
    return native, source, target.copy(), records, {"recipe": receipt["recipe"], "shape": receipt["shape"],
        "tiles": len(receipt["tiles"]), "total_model_cycles": cycles,
        "raw_accumulator_role": "Independent pre-rescale oracle only; never substituted"}
