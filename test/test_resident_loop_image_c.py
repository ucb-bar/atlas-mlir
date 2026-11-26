"""Generated shared-template C relocation, forced reload, and original-model execution."""
import ctypes
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.options import codegen_options
from atlas_mlir_native.quantized_resident import pack_resident_input
import test_resident_loop_image as _fixture
from test_resident_loop_image import ROOT, moved_options, fixture_values, expected_tile


class ResidentLoopImageCTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get('ATLAS_RESIDENT_IMAGE_RECEIPTS')
        if directory:
            with patch.dict(os.environ, ATLAS_RESIDENT_IMAGE_RECEIPTS=str(Path(directory) / 'runtime-c-abi')):
                _fixture.ResidentLoopImageTest.setUpClass.__func__(cls)
        else:
            _fixture.ResidentLoopImageTest.setUpClass.__func__(cls)
    tearDownClass = classmethod(_fixture.ResidentLoopImageTest.tearDownClass.__func__)

    def test_generated_template_reloads_and_executes_current_trip(self):
        with codegen_options(self.selected):
            emitted = self.image.emit_c('shared_image')
        work = self.root / 'generated-c'
        work.mkdir()
        (work / 'shared_image.h').write_text(emitted.header)
        (work / 'shared_image.c').write_text(emitted.source)
        for name in ('atlas_host.h', 'atlas_host_layout.h', 'atlas_host.c'):
            shutil.copyfile(ROOT / 'runtime' / name, work / name)
        (work / 'harness.c').write_text(r'''#include "shared_image.h"
#include <assert.h>
#include <string.h>
static atlas_host host;
static uint32_t words[SHARED_IMAGE_WORD_COUNT],imem[SHARED_IMAGE_WORD_COUNT];
static uint32_t read32(void *p,uintptr_t a) { (void)p;(void)a;return 5u; }
static void write32(void *p,uintptr_t a,uint32_t v) {
  (void)p; if (a>=0x20000u && a<0x20000u+sizeof(imem)) imem[(a-0x20000u)/4u]=v;
}
void setup(void) {
  assert(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,sizeof(words)},
    (atlas_host_mmio){read32,write32,0},(atlas_host_memory){1,0,0})==ATLAS_HOST_OK);
}
int bind(const uint32_t *addresses,uint32_t *out) {
  int r=shared_image_bind(&host,words,SHARED_IMAGE_WORD_COUNT,addresses,SHARED_IMAGE_ADDRESS_COUNT);
  if (!r) { assert(!host.resident_words); memcpy(out,imem,sizeof(imem)); }
  return r;
}
void interleave(void) {
  static const uint32_t other[]={0x13u,0x73u};
  assert(atlas_host_ensure_loaded(&host,other,2u)==ATLAS_HOST_OK);
}
void invalid(const uint32_t *addresses) {
  uint32_t before[SHARED_IMAGE_WORD_COUNT];memcpy(before,words,sizeof(words));
  uint64_t loads=host.image_loads;
  assert(shared_image_bind(0,words,SHARED_IMAGE_WORD_COUNT,addresses,SHARED_IMAGE_ADDRESS_COUNT)==ATLAS_HOST_INVALID);
  assert(shared_image_bind(&host,words,0,addresses,SHARED_IMAGE_ADDRESS_COUNT)==ATLAS_HOST_INVALID);
  assert(shared_image_bind(&host,words,SHARED_IMAGE_WORD_COUNT,addresses,0)==ATLAS_HOST_INVALID);
  assert(shared_image_bind(&host,words,SHARED_IMAGE_WORD_COUNT,words,SHARED_IMAGE_ADDRESS_COUNT)==ATLAS_HOST_INVALID);
  assert(shared_image_bind(&host,(uint32_t*)((char*)words+1),SHARED_IMAGE_WORD_COUNT,addresses,SHARED_IMAGE_ADDRESS_COUNT)==ATLAS_HOST_INVALID);
  host.active=1;
  assert(shared_image_bind(&host,words,SHARED_IMAGE_WORD_COUNT,addresses,SHARED_IMAGE_ADDRESS_COUNT)==ATLAS_HOST_BUSY);
  host.active=0;
  assert(host.image_loads==loads && !memcmp(before,words,sizeof(words)));
}
uint64_t loads(void) { return host.image_loads; }
uint64_t reuses(void) { return host.image_reuses; }
''')
        compiler = shutil.which('cc')
        self.assertIsNotNone(compiler, 'C compiler required for executed generated ABI test')
        subprocess.run([compiler, '-std=c11', '-Wall', '-Wextra', '-Werror', '-fPIC',
                        '-shared', 'shared_image.c', 'atlas_host.c', 'harness.c', '-o', 'image.so'],
                       cwd=work, check=True, capture_output=True)
        lib = ctypes.CDLL(str(work / 'image.so'))
        ptr = ctypes.POINTER(ctypes.c_uint32)
        lib.bind.argtypes = [ptr, ptr]
        lib.invalid.argtypes = [ptr]
        lib.loads.restype = lib.reuses.restype = ctypes.c_uint64
        lib.setup()
        bound = []
        addresses = []
        for trip in range(2):
            selected = moved_options(self.source, self.selected, trip)
            if trip:
                changed = deepcopy(selected.resident_bindings)
                changed[self.source.digest()]['sw']['address'] += 2048
                selected = replace(selected, resident_bindings=changed)
            with codegen_options(selected):
                image = self.image.bind(self.source)
                table = self.image.verify_binding(self.source, image)
                broken = deepcopy(image.receipt)
                broken['resident_addresses'][0] += 2048
                broken.pop('binding_sha256')
                broken['binding_sha256'] = sha256(json.dumps(broken, sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                with self.assertRaises(ValueError):
                    self.image.binding_addresses(replace(image, receipt=broken))
                with self.assertRaises(ValueError):
                    self.image.verify_binding(self.source, replace(image, binary=b'bad'))
            bound.append(image)
            addresses.append((ctypes.c_uint32 * len(table))(*table))
        output = (ctypes.c_uint32 * (len(self.template) // 4))()
        lib.invalid(addresses[0])
        for bad in ((0, 0), (0xffffffff, 0xffffffff), (addresses[0][0], addresses[0][0])):
            self.assertEqual(lib.bind((ctypes.c_uint32 * 2)(*bad), output), 1)
        # A/B/A/A reuse the exact same mutable words array. A static image is
        # interleaved before the return to A; every generated binding must load.
        generated = []
        for index in (0, 1, 0, 0):
            if len(generated) == 2:
                lib.interleave()
            self.assertEqual(lib.bind(addresses[index], output), 0)
            generated.append(bytes(output))
            self.assertEqual(generated[-1], bound[index].binary)
        self.assertEqual(lib.loads(), 5)
        self.assertEqual(lib.reuses(), 0)
        self.assertEqual(emitted.source.count('static const uint32_t shared_image_template['), 1)
        self.assertTrue(emitted.receipt['forced_reload'])

        # Execute the bytes captured from generated C MMIO, never the Python
        # binding's binary, in the original instruction interpreter.
        import torch
        torch.set_num_threads(1)
        model = Path(os.environ.get('ATLAS_NPU_MODEL_ROOT', os.environ['NPU_MODEL_ROOT']))
        sys.path[:0] = [str(model), str(ROOT), str(ROOT / 'tools'),
                       str(ROOT.parent / 'merlin/examples/atlas/whole-model/unchanged_fx')]
        import functional_runtime
        from functional_runtime import FunctionalModelRuntime, PhysicalFunctionalInterpreter
        selected = self.selected
        class CapacityInterpreter(PhysicalFunctionalInterpreter):
            def __init__(self, config, **kwargs):
                super().__init__(replace(config, dram_base=selected.io_memory_capacity_base,
                    dram_size=selected.io_memory_capacity_bytes), **kwargs)
        values = [fixture_values(i) for i in range(2)]
        immutable = [(row['address'], pack_resident_input(self.source, row['node_id'],
                      data[row['node_id']].tobytes()))
                     for image, data in zip(bound, values, strict=True)
                     for row in image.execution_plan['resident_bindings']]
        plans = {sha256(image.binary).hexdigest(): image.execution_plan for image in bound}
        runtime = FunctionalModelRuntime(ROOT, model, list(dict.fromkeys(generated)),
            self.root / 'generated-c-model', memory_bytes=self.selected.io_memory_bytes,
            memory_base=DRAM_BASE, image_plans=plans, trace_payloads=False,
            immutable_loads=immutable, capacity_base=self.selected.io_memory_capacity_base,
            capacity_bytes=self.selected.io_memory_capacity_bytes)
        self.assertEqual(lib.bind(addresses[1], output), 0)
        generated.append(bytes(output))
        for launch, (binary, index) in enumerate(zip(generated, (0, 1, 0, 0, 1), strict=True)):
            plan = bound[index].execution_plan
            data = values[index]
            if launch == 4:
                data = {key: value.copy() for key, value in data.items()}
                data['w'][-1, -1] = 0x40
                self.assertNotEqual(expected_tile(data), expected_tile(values[index]))
                payload = pack_resident_input(self.source, 'w', data['w'].tobytes())
                at = plan['input_addresses'][1] - DRAM_BASE
                runtime.memory[at:at+len(payload)] = payload
            for key, address in (('a', plan['input_addresses'][0]), ('sx', plan['exponent_addresses'][0])):
                payload = pack_resident_input(self.source, key, data[key].tobytes())
                staging = ctypes.create_string_buffer(payload)
                self.assertEqual(runtime.put(None, address, ctypes.addressof(staging), len(payload)), 0)
            for word in range(len(binary) // 4):
                runtime.write(None, 0x20000 + word*4, int.from_bytes(binary[word*4:word*4+4], 'little'))
            with patch.object(functional_runtime, 'PhysicalFunctionalInterpreter', CapacityInterpreter):
                runtime.write(None, 0x40018, 1)
            self.assertEqual(runtime.read(None, 0x40008), 0)
            self.assertEqual(runtime.read(None, 0x40008), 5, runtime.errors)
            at = plan['output_addresses'][0] - DRAM_BASE
            self.assertEqual(bytes(runtime.memory[at:at+plan['output_bytes']]), expected_tile(data))
        record = runtime.finish()
        self.assertFalse(record['errors'])
        self.assertEqual(len(record['launches']), 5)
        (work / 'c-abi-receipt.json').write_text(json.dumps(dict(
            abi=emitted.receipt, exact=True, generated_c_launches=5, executed_last_weight_mutation=True,
            image_loads=lib.loads(), image_reuses=lib.reuses()), indent=2)+'\n')
