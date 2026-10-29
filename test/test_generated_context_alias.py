"""Reject overlapping generated contexts before changing caller-owned storage."""

import os
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import test_application_c_package
from test_looped_mnk import ROOT, build_snapshot


@unittest.skipUnless(
    os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR") and shutil.which("cc"),
    "select Merlin, Atlas build tools and a C compiler",
)
class GeneratedContextAliasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from atlas_mlir_native.application import AtlasApplicationAdapter

        from merlin.compile.semantic_application import compile_semantic_application
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.tests.runtime.test_batched_c import source_graph

        cls.temporary = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        cls.root = Path(cls.temporary.name)
        snapshot = build_snapshot(cls.root)
        cls.cases = []
        policy = test_application_c_package.ApplicationLayoutPolicyTest().policy()
        for k in (3, 4097):
            source = source_graph(())
            shapes = ((1, k), (k, 2), (1, 2), (1, 2))
            source = replace(
                source, target_identity=policy["target_identity"],
                nodes=tuple(replace(node, type=replace(node.type, shape=shape))
                            for node, shape in zip(source.nodes, shapes, strict=True)),
                constants=(replace(source.constants[0], data_hex="00000000" * 2),),
            )
            adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
            artifact = cls.root / str(k)
            manifest = compile_semantic_application(
                source, adapter=adapter, snapshot=snapshot, target_source=ROOT, destination=artifact,
                limits=SearchLimits(), name="alias", capacity=1 << 20,
            )
            region = manifest["regions"][0]["description"]["context_type"].removesuffix("_context")
            cls.cases.append((k, artifact / "c-package/combined", region))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_probe(self, package, body, arguments=((),)):
        source = package / "alias-probe.c"
        source.write_text('''#include "alias_combined.h"
#include <assert.h>
#include <string.h>
static uint32_t read32(void *p, uintptr_t a) { (void)p; (void)a; return 5; }
static void write32(void *p, uintptr_t a, uint32_t v) { (void)p; (void)a; (void)v; }
static int put(void *p, uintptr_t a, const uint8_t *d, size_t n) {
  (void)p; (void)a; (void)d; (void)n; return 0;
}
static int get(void *p, uintptr_t a, uint8_t *d, size_t n) {
  (void)p; (void)a; (void)d; (void)n; return 0;
}
''' + body)
        executable = package / "alias-probe"
        result = subprocess.run(
            [shutil.which("cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
             str(source), str(package / "alias_combined.c"), "-o", str(executable)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for argument in arguments:
            with self.subTest(arguments=argument):
                result = subprocess.run([str(executable), *argument], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_chunk_binding_rejects_context_host_overlap_without_mutation(self):
        for k, package, region in self.cases:
            if k == 4097:
                with self.subTest(k=k):
                    self.run_probe(package, self.context_host_probe(region, False))

    def test_application_binding_rejects_context_host_overlap_without_mutation(self):
        for k, package, _ in self.cases:
            with self.subTest(k=k):
                self.run_probe(package, self.context_host_probe("alias", True))

    @staticmethod
    def context_host_probe(name, application):
        workspace = "" if application else "workspace,sizeof(workspace),"
        return (r'''
int main(void) {
  static union { atlas_host host; NAME_context context;
                 unsigned char bytes[sizeof(atlas_host)+sizeof(NAME_context)]; } shared;
  static unsigned char before[sizeof(shared)];
WORKSPACE
  atlas_host_map map={0x20000u,0x40000u,131072u};
  atlas_host_mmio mmio={read32,write32,0};
  atlas_host_memory memory={.coherent=1};
  atlas_tile_memory_io io={put,get,0};
  for (size_t offset=0;offset<=_Alignof(NAME_context);offset+=_Alignof(NAME_context)) {
    memset(&shared,0,sizeof(shared));
    assert(atlas_host_init(&shared.host,map,mmio,memory)==ATLAS_HOST_OK);
    memcpy(before,&shared,sizeof(shared));
    NAME_context *context=(NAME_context *)(void *)(shared.bytes+offset);
    assert(NAME_bind(context,&shared.host,io,ARGUMENTS100)==INVALID);
    assert(memcmp(before,&shared,sizeof(shared))==0);
  }
  static NAME_context disjoint;
  assert(NAME_bind(&disjoint,&shared.host,io,ARGUMENTS100)==0);
  return 0;
}
'''.replace("WORKSPACE", "" if application else "  static unsigned char workspace[NAME_WORKSPACE_BYTES];")
                .replace("ARGUMENTS", workspace).replace("INVALID", "-1" if application else "NAME_INVALID")
                .replace("NAME_WORKSPACE_BYTES", name.upper() + "_WORKSPACE_BYTES")
                .replace("NAME_INVALID", name.upper() + "_INVALID").replace("NAME", name))

    def test_source_binding_rejects_context_overlap_without_mutation(self):
        for k, package, region in self.cases:
            with self.subTest(k=k):
                self.run_probe(package, r'''
int main(int argc,char **argv) {
  assert(argc==2);
  static union { NAME_context region; NAME_source_context source;
                 unsigned char bytes[sizeof(NAME_context)+sizeof(NAME_source_context)]; } shared;
  static union { atlas_host host; NAME_source_context source;
                 unsigned char bytes[sizeof(atlas_host)+sizeof(NAME_source_context)]; } host;
  static _Alignas(NAME_source_context) unsigned char workspace[UPPER_WORKSPACE_BYTES+sizeof(NAME_source_context)];
  static unsigned char scratch[UPPER_SOURCE_WORKSPACE_BYTES];
  unsigned char before_region[sizeof(shared)],before_host[sizeof(host)],before_workspace[sizeof(workspace)];
  atlas_host_map map={0x20000u,0x40000u,131072u};
  atlas_host_mmio mmio={read32,write32,0};
  atlas_host_memory memory={.coherent=1};
  atlas_tile_memory_io io={put,get,0};
  for (size_t offset=0;offset<=_Alignof(NAME_source_context);offset+=_Alignof(NAME_source_context)) {
    memset(&shared,0xa5,sizeof(shared)); memset(&host,0xa5,sizeof(host)); memset(workspace,0xa5,sizeof(workspace));
    assert(atlas_host_init(&host.host,map,mmio,memory)==ATLAS_HOST_OK);
    assert(NAME_bind(&shared.region,&host.host,io,workspace,UPPER_WORKSPACE_BYTES,100)==UPPER_OK);
    memcpy(before_region,&shared,sizeof(shared)); memcpy(before_host,&host,sizeof(host));
    memcpy(before_workspace,workspace,sizeof(workspace));
    unsigned char *targets[]={shared.bytes,host.bytes,workspace};
    NAME_source_context *context=(NAME_source_context *)(void *)(targets[argv[1][0]-'0']+offset);
    assert(NAME_source_bind(context,&shared.region,scratch,sizeof(scratch))==UPPER_INVALID);
    assert(memcmp(before_region,&shared,sizeof(shared))==0);
    assert(memcmp(before_host,&host,sizeof(host))==0);
    assert(memcmp(before_workspace,workspace,sizeof(workspace))==0);
    NAME_source_context disjoint;
    assert(NAME_source_bind(&disjoint,&shared.region,scratch,sizeof(scratch))==UPPER_OK);
  }
  return 0;
}
'''.replace("UPPER", region.upper()).replace("NAME", region), arguments=(("0",), ("1",), ("2",)))
