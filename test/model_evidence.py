"""Shared opt-in rule for tests requiring a qualified external model bundle."""
import os
from pathlib import Path
import sys
import unittest


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
    """Skip only an unconfigured environment; incomplete bundles fail."""
    if "ATLAS_TEST_MODEL_FACTS" not in os.environ:
        return False
    value = os.environ["ATLAS_TEST_MODEL_FACTS"]
    if not value:
        raise FileNotFoundError("ATLAS_TEST_MODEL_FACTS must name a qualified bundle directory")
    directory = Path(value)
    for name in ("facts.lock.json", "model-timing.json", "model-overlap.json"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"ATLAS_TEST_MODEL_FACTS requires {name}: {value!r}")
    return True


def requires_model_evidence(test):
    return unittest.skipUnless(model_evidence_available(),
                               "set ATLAS_TEST_MODEL_FACTS for a qualified evidence bundle")(test)
