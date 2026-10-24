"""Bounded physical-output array admission for contraction capsules."""

import zipfile
from atlas_path_guard import checked as physical_path
from run_model_tile import validate_archive_members


def load_arrays(path, *, result=False):
    import numpy as np

    path = physical_path(path)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if result:
            required = {"target_C.npy", "actual_bits.npy", "expected_bits.npy"}
            names = [member.filename for member in members]
            if (len(names) != len(set(names)) or not required <= set(names)
                or set(names) - required - {"raw_accumulator_bits.npy"}
                or sum(member.file_size for member in members) > 32 * 1024 * 1024):
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
        if words.shape != shape or words.dtype != np.uint16 or np.any((words & 0x7F80) == 0x7F80):
            raise ValueError("Invalid finite BF16 result words")
    if not np.array_equal(arrays["actual_bits"], arrays["expected_bits"]) or not np.array_equal(target.view(np.uint32),
        arrays["actual_bits"].astype(np.uint32) << 16):
        raise ValueError("Actual/oracle words or exact target view disagree")
    return target
