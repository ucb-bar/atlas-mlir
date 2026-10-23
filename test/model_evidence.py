"""Shared opt-in rule for tests requiring external model evidence."""
import os
from pathlib import Path
import sys
import unittest

EVIDENCE_VARIABLES = ("ATLAS_TEST_OP_TIMING", "ATLAS_TEST_PAIR_TIMING")


def isolate_model_imports(test_case, root):
    """Select one source checkout for a test class without leaking it to discovery."""
    search_path = list(sys.path)
    modules = {name: value for name, value in sys.modules.items()
               if name == "npu_model" or name.startswith("npu_model.")}

    def clear_model_modules():
        for name in list(sys.modules):
            if name == "npu_model" or name.startswith("npu_model."):
                del sys.modules[name]

    def restore_imports():
        sys.path[:] = search_path
        clear_model_modules()
        sys.modules.update(modules)

    test_case.addClassCleanup(restore_imports)
    clear_model_modules()
    sys.path.insert(0, root)


def model_evidence_available():
    """Skip only an unconfigured environment; incomplete or missing pins fail."""
    if not any(name in os.environ for name in EVIDENCE_VARIABLES):
        return False
    for name in EVIDENCE_VARIABLES:
        value = os.environ.get(name)
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"{name} must name an existing evidence file: {value!r}")
    return True


def requires_model_evidence(test):
    return unittest.skipUnless(model_evidence_available(),
                               "set ATLAS_TEST_OP_TIMING and ATLAS_TEST_PAIR_TIMING for pinned evidence")(test)
