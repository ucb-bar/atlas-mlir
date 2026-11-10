"""Opt-in all-class publication gates over an authenticated saved application.

Set ATLAS_MATERIALIZATION_CORPUS and MERLIN_NATIVE_SNAPSHOT to run the complete
real-tool gate. Native publications and generated C are compared byte for byte.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import os

import pytest

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.pointwise import BF16_POINTWISE_POLICY
from atlas_mlir_native.reduction import REDUCTION_POLICY
from atlas_mlir_native.source_exp import POLICY as EXP_POLICY
from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.materialization import canonical_request_record, canonical_request, materialization_session
from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.snapshot import open_native_snapshot

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("exact_float_device", "layout_device", "activation_quantize_device", "pointwise",
         "pointwise_math", "reduction", "masked_softmax", "already_quantized")


def _members(root):
    result = {}
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            assert not path.is_symlink()
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


@pytest.mark.skipif(not (os.environ.get("ATLAS_MATERIALIZATION_CORPUS") and os.environ.get("MERLIN_NATIVE_SNAPSHOT")),
                    reason="select saved application and exact native snapshot for all-class gate")
@pytest.mark.parametrize("kind", KINDS)
def test_every_saved_source_class_retains_every_published_byte(tmp_path, kind):
    root = Path(os.environ["ATLAS_MATERIALIZATION_CORPUS"])
    manifest = json.loads((root / "manifest.json").read_bytes())
    whole = KernelRequest.from_record(json.loads((root / "source-request.json").read_bytes()))
    pure = [node for node in whole.nodes if node.effect == "pure"]
    choices = {row["id"]: row for row in json.loads((root / "selected-plan.json").read_bytes())["selected"]}
    classes = {}
    for row in manifest["regions"]:
        if row["description"].get("kind") != kind:
            continue
        source = extract_semantic_region(whole, tuple(pure[index].id for index in choices[row["id"]]["nodes"]))
        assert source.digest() == row["source_request_digest"]
        identity = json.dumps(canonical_request_record(source, source_identity_policy="provenance"), sort_keys=True)
        classes.setdefault(identity, source)
    assert classes, "selected corpus has no class for this supported target kind"
    snapshot = open_native_snapshot(Path(os.environ["MERLIN_NATIVE_SNAPSHOT"]))
    adapter = AtlasApplicationAdapter.__new__(AtlasApplicationAdapter)
    adapter.bf16_pointwise = BF16_POINTWISE_POLICY
    adapter.source_exp_policy = EXP_POLICY
    adapter.qualified_rne_add = None
    adapter.reduction_policy = REDUCTION_POLICY
    adapter.already_quantized_score_domain = "bf16-row-relative-span80-pad32-v1"
    adapter.already_quantized_mask_sentinel = None
    alternative = SimpleNamespace(metadata=(("kind", kind),))
    with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
        for index, original in enumerate(classes.values()):
            directory = tmp_path / str(index)
            with materialization_session(enabled=False):
                fresh = adapter.compile_region(snapshot, original, alternative, "region", directory / "fresh", ROOT, SearchLimits())
            canonical = canonical_request(original, source_identity_policy="provenance")
            with materialization_session(enabled=True):
                materialized = adapter.materialize_region(snapshot, canonical, alternative, "canonical", directory / "private", ROOT, SearchLimits())
                assert materialized is not None, "selected corpus includes a conservatively declined construction"
                rebound = adapter.bind_materialization(materialized, original, alternative, "region", directory / "cached", ROOT)
            assert fresh == rebound
            assert _members(directory / "fresh") == _members(directory / "cached")
