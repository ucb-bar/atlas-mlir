"""Shared fixtures for selected model sources and qualified evidence bundles."""
import json
import os
from pathlib import Path
import sys
import tempfile
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


class NativeModelFixture:
    @classmethod
    def setUpClass(cls):
        isolate_model_imports(cls, os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        from atlas_model_timing import configuration
        self.cfg, _ = configuration()


def semantic_sources(model):
    """Read only reviewed relative source entries, checking paths before access."""
    from atlas_model_source import read_source
    manifest = json.loads((Path(__file__).resolve().parents[1] / "tools/model-pins.json").read_text())
    sources = {}
    for relative in manifest["files"]:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise AssertionError("guard manifest source must be relative")
        sources[relative] = read_source(model / "npu_model" / relative)
    return sources


def copy_model_sources(test_case, sources, subdirectory=""):
    """Give every test its own sparse source tree and register its cleanup."""
    directory = tempfile.TemporaryDirectory()
    test_case.addCleanup(directory.cleanup)
    root = Path(directory.name)
    model = root / subdirectory
    for relative, source in sources.items():
        path = model / "npu_model" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root, model


def model_evidence_available():
    """Skip only an unconfigured environment; incomplete bundles fail."""
    if "ATLAS_TEST_MODEL_FACTS" not in os.environ:
        return False
    value = os.environ["ATLAS_TEST_MODEL_FACTS"]
    if not value:
        raise FileNotFoundError("ATLAS_TEST_MODEL_FACTS must name a qualified bundle directory")
    directory = Path(value)
    if not (directory / "model-timing.json").is_file():
        raise FileNotFoundError(f"ATLAS_TEST_MODEL_FACTS requires model-timing.json: {value!r}")
    return True


def requires_model_evidence(test):
    return unittest.skipUnless(model_evidence_available(),
                               "set ATLAS_TEST_MODEL_FACTS for a qualified evidence bundle")(test)
