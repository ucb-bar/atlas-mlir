"""Shared host dispatch C: one guarded copy per source, the region ABI checks, and image placement."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_looped_mnk import ROOT

from atlas_mlir_native.c_runtime import IMAGE_SECTION, program_array, runtime_headers
from atlas_mlir_native.shared_dispatch_c import CHECKS, REGION_TYPES, shape


def compile_c(root, source, *flags):
    compiler = shutil.which("cc")
    assert compiler, "a C compiler is required"
    headers = runtime_headers(ROOT / "runtime")
    for name, text in headers.items():
        (root / name).write_text(text)
    (root / "check.c").write_text(source)
    subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", *flags, "check.c", "-o", "check"],
                   cwd=root, check=True, capture_output=True)
    return subprocess.run([str(root / "check")], cwd=root, capture_output=True, text=True)


class SharedDispatchTest(unittest.TestCase):
    def test_images_enter_their_section_only_when_selected(self):
        source = "#include <stdint.h>\n" + program_array("a_words", b"\x73\0\0\0") + program_array("b_words", b"\x73\0\0\0")
        source += "const uint32_t *image(int b) { return b ? b_words : a_words; }\n"
        compiler = shutil.which("cc")
        assert compiler, "a C compiler is required"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.c").write_text(source)
            for flags, expected in (((), False), (("-DATLAS_IMAGE_SECTIONS",), True)):
                with self.subTest(flags=flags):
                    assembly = subprocess.run([compiler, "-std=c11", "-Wall", "-Werror", *flags, "-S", "-o", "-", "image.c"],
                                              cwd=root, check=True, capture_output=True, text=True).stdout
                    self.assertEqual(IMAGE_SECTION in assembly, expected)
                    self.assertEqual(assembly.count(".long\t115"), 1)

    def test_region_dispatch_checks_have_no_port_ceiling(self):
        count = 40
        main = f'''#include "atlas_host_layout.h"
#include "atlas_host.h"
{REGION_TYPES}{CHECKS}
typedef struct {{ void *region; }} source_context;
{shape("wide", context="source_context", region="atlas_region_context", workspace_bytes=64, inputs=[8] * count, outputs=[8])}
static unsigned char memory[1024];
static atlas_host host;
static atlas_region_context region;
static source_context ctx;
int main(void) {{
  const void *inputs[{count}]; void *outputs[1]={{memory+512}};
  for (unsigned i=0;i<{count}u;++i) inputs[i]=memory+128+8*i;
  if (atlas_dispatch_conflict(&wide_shape,&ctx,&region,&host,memory,inputs,outputs)) return 1;
  outputs[0]=memory+128+8*({count}-1);
  return !atlas_dispatch_conflict(&wide_shape,&ctx,&region,&host,memory,inputs,outputs) ? 2 : 0;
}}
'''
        with tempfile.TemporaryDirectory() as directory:
            result = compile_c(Path(directory), main)
            self.assertEqual(result.returncode, 0, result.stdout)

    def test_region_dispatch_checks_follow_the_region_abi(self):
        # Trapping UBSan needs no runtime library: a null member access or pointer overflow aborts the check.
        main = f'''#include "atlas_host_layout.h"
#include "atlas_host.h"
{REGION_TYPES}{REGION_TYPES}{CHECKS}{CHECKS}
typedef struct {{ void *region; }} source_context;
{shape("probe", context="source_context", region="atlas_region_context", workspace_bytes=64, inputs=[16, 16], outputs=[16])}
#include <stdio.h>
#define HOST offsetof(atlas_region_context,host)
#define WORKSPACE offsetof(atlas_region_context,workspace)
static unsigned char memory[1024];
static atlas_host host;
static atlas_region_context region;
static source_context ctx;
static int run(const void *a,const void *b,void *out) {{
  const void *inputs[2]={{a,b}}; void *outputs[1]={{out}};
  return atlas_dispatch_conflict(&probe_shape,&ctx,&region,&host,memory,inputs,outputs);
}}
int main(void) {{
  unsigned char *x=memory+128,*y=memory+256,*z=memory+512;
  region.host=&host; region.workspace=memory;
  /* Zero passes: inputs may alias each other; outputs, states, and invalid spans may not. */
  int cases[]={{
    run(x,y,z), run(x,x,z), !run(x,y,x+8), !run(x,y,memory+32), !run(memory,y,z), !run(0,y,z),
    !run(x,y,(void *)&host),
    atlas_source_bind_conflict(&ctx,sizeof(ctx),&region,sizeof(region),HOST,WORKSPACE,64u),
    !atlas_source_bind_conflict(&ctx,sizeof(ctx),&ctx,sizeof(ctx),HOST,WORKSPACE,64u),
    /* A NULL or wrapping region is refused before any of its fields is read. */
    !atlas_source_bind_conflict(&ctx,sizeof(ctx),0,sizeof(region),HOST,WORKSPACE,64u),
    !atlas_source_bind_conflict(&ctx,sizeof(ctx),(void *)(UINTPTR_MAX-4u),sizeof(region),HOST,WORKSPACE,64u),
  }};
  for (unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);++i) if (cases[i]) {{ printf("case %u\\n",i); return 1; }}
  return 0;
}}
'''
        with tempfile.TemporaryDirectory() as directory:
            result = compile_c(Path(directory), main, "-fsanitize=undefined", "-fsanitize-undefined-trap-on-error")
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
