"""Compile exported runtime bundles in isolation and check application sharing."""

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from test_looped_mnk import ROOT


class RuntimePackagingTest(unittest.TestCase):
    def compile_and_run(self, root, files, sources):
        for name, payload in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("a C compiler is required")
        subprocess.run(
            [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", *sources, "-o", "check"],
            cwd=root, check=True, capture_output=True,
        )
        subprocess.run([str(root / "check")], cwd=root, check=True, capture_output=True)

    @unittest.skipUnless(os.environ.get("MERLIN_NATIVE_ROOT"), "select Merlin")
    def test_embedded_and_separate_runtime_packages_compile_and_launch(self):
        from atlas_mlir_native.c_runtime import runtime_headers, runtime_members
        from merlin.compile.c_sources import c_source_pair

        headers = runtime_headers(ROOT / "runtime")
        runtime = (ROOT / "runtime/atlas_host.c").read_text()
        for linkage in ("embedded", "separate"):
            with self.subTest(linkage=linkage), tempfile.TemporaryDirectory() as directory:
                members, files, _ = runtime_members(headers, runtime, linkage)
                pair = c_source_pair(
                    "region", public_headers=headers, private_headers={},
                    sources={**members, "region.c": "int region_layout(void) { return ATLAS_CSR_DBG1; }"},
                    declarations="int region_layout(void);",
                )
                public = pair["region.h"]
                self.assertLess(public.index('"atlas_host_layout.h"'), public.index('"atlas_host.h"'))
                self.assertIsNone(re.search(r'^\s*#include\s*"', public, re.M))
                files.update({name: text.encode() for name, text in pair.items()})
                files["main.c"] = b'''#include "region.h"
#include <assert.h>
static uint32_t marker, command;
static uint32_t read32(void *context, uintptr_t address) {
  (void)context;
  return address == ATLAS_CSR_BASE + ATLAS_CSR_DBG0 ? marker : 5;
}
static void write32(void *context, uintptr_t address, uint32_t value) {
  (void)context;
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_DBG0) marker=value;
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_DBG1) command=value;
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_EXEC_CONTROL) marker=command;
}
int main(void) {
  atlas_host host;
  const uint32_t words[]={0x13,0x73};
  assert(region_layout()==ATLAS_CSR_DBG1);
  assert(atlas_host_init(&host,(atlas_host_map){ATLAS_IMEM_BASE,ATLAS_CSR_BASE,128},
      (atlas_host_mmio){read32,write32,0},(atlas_host_memory){1,0,0})==ATLAS_HOST_OK);
  assert(atlas_host_ensure_loaded(&host,words,2)==ATLAS_HOST_OK);
  assert(atlas_host_ensure_loaded(&host,words,2)==ATLAS_HOST_OK);
  assert(atlas_host_load_raw(&host,words,2)==ATLAS_HOST_OK);
  assert(atlas_host_start(&host,0,0)==ATLAS_HOST_OK);
  assert(atlas_host_wait(&host,1)==ATLAS_HOST_OK);
  assert(host.image_loads==2 && host.image_reuses==1 && host.imem_words_written==9);
  return 0;
}
'''
                sources = ["region.c", "main.c"] + (["runtime/atlas_host.c"] if linkage == "separate" else [])
                self.compile_and_run(Path(directory), files, sources)

    def test_runtime_headers_reject_missing_or_linked_dependencies(self):
        from atlas_mlir_native.c_runtime import runtime_headers

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            shutil.copyfile(ROOT / "runtime/atlas_host.h", runtime / "atlas_host.h")
            with self.assertRaisesRegex(ValueError, "absent or unsafe"):
                runtime_headers(runtime)
            (runtime / "atlas_host_layout.h").symlink_to(ROOT / "runtime/atlas_host_layout.h")
            with self.assertRaisesRegex(ValueError, "absent or unsafe"):
                runtime_headers(runtime)
            (runtime / "atlas_host_layout.h").unlink()
            shutil.copyfile(ROOT / "runtime/atlas_host_layout.h", runtime / "atlas_host_layout.h")
            self.assertEqual(runtime_headers(runtime), runtime_headers(ROOT / "runtime"))

    def test_diagnostic_runtime_bundles_close_over_project_headers(self):
        from tools.atlas_ordered_mn_host_data import RUNTIME_FILES as ordered
        from tools.atlas_source_mn_host_data import RUNTIME_FILES as source

        for bundle in (ordered, source):
            with self.subTest(bundle=bundle):
                for name in bundle:
                    text = (ROOT / "runtime" / name).read_text()
                    for dependency in re.findall(r'^\s*#include\s*"([^"]+)"', text, re.M):
                        self.assertIn(dependency, bundle, f"{name} requires {dependency}")

    @unittest.skipUnless(
        os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
        "select Merlin and Atlas tools",
    )
    def test_application_regions_share_the_complete_runtime_header_bundle(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from test_application_c_package import ApplicationLayoutPolicyTest
        from test_looped_mnk import build_snapshot
        from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.tests.runtime.test_batched_c import source_graph

        policy = ApplicationLayoutPolicyTest().policy()
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
        contraction = replace(source_graph(()), target_identity=policy["target_identity"])
        tensor = TensorType((2, 4), "f32", "raw-bit-pattern")
        pointwise = KernelRequest(
            (SemanticNode("a", "input", (), tensor, effect="input"),
             SemanticNode("b", "input", (), tensor, effect="input"),
             SemanticNode("y", "f32_add", ("a", "b"), tensor,
                          (("numeric_semantics", "ieee754-f32-add-rne-v1"),),
                          index_maps=pointwise_identity_maps(2, 2))),
            ("y",), ("external",), (("a", "external"), ("b", "external")),
            policy["target_identity"], "diagnostic",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = build_snapshot(root)
            segments = root / "segments"
            segments.mkdir()
            exports = []
            for name, request in (("contract", contraction), ("add", pointwise)):
                (selected,) = adapter.plan(request).selected
                compiled = adapter.compile_region(snapshot, request, selected, name, segments, ROOT, SearchLimits())
                self.assertEqual(compiled, adapter.export_region(request, selected, name, segments, ROOT))
                runtime_files = {path for path in compiled.files if path.startswith("runtime/")}
                self.assertEqual(set(compiled.shared_files), runtime_files)
                self.assertIn("runtime/atlas_host_layout.h", compiled.shared_files)
                exports.append(compiled)
            for path in exports[0].shared_files:
                self.assertEqual(exports[0].files[path], exports[1].files[path])
                self.assertEqual(exports[0].files[path], (ROOT / path).read_bytes())


if __name__ == "__main__":
    unittest.main()
