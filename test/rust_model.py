"""Run a compiler-emitted program in the explicitly selected Rust model."""
import json
import os
import subprocess


class Application:
    """Execute exported application C with synchronous Rust-backed MMIO.

    Device cycles exclude host transfers, image loading and callback overhead.
    Only the ordinary reset-entry runtime is modeled here.
    """

    def __init__(self, package, name, destination, *, base, size):
        import ctypes

        self.ctypes = ctypes
        destination.mkdir(parents=True)
        self.destination, self.base = destination, base
        self.memory = bytearray([0xa5]) * size
        self.words, self.stats, self.errors = {}, [], []
        self.retired = 0
        self.upload_bytes = self.download_bytes = 0
        shim = destination / 'shim.c'
        shim.write_text(f'''#include "combined/{name}_combined.h"
static atlas_host host;
static {name}_context application;
int initialize(atlas_host_read32 read,atlas_host_write32 write,
               atlas_tile_write put,atlas_tile_read get) {{
  int status=atlas_host_init(&host,(atlas_host_map){{0x20000u,0x40000u,131072u}},
    (atlas_host_mmio){{read,write,0}},(atlas_host_memory){{1,0,0}});
  return status?status:{name}_bind(&application,&host,(atlas_tile_memory_io){{put,get,0}},100u);
}}
int invoke(const void *const *inputs,void *const *outputs) {{
  return {name}_run(&application,inputs,outputs);
}}
size_t word_count(void) {{return host.resident_word_count;}}
uint64_t launches(void) {{return host.launches;}}
''')
        library = destination / 'application.so'
        result = subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                                 '-shared', '-fPIC', '-I', str(package),
                                 str(package / f'combined/{name}_combined.c'), str(shim),
                                 '-o', str(library)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        self.library = ctypes.CDLL(str(library))
        read = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t)
        write = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
        transfer = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                                   ctypes.c_void_p, ctypes.c_size_t)
        self.callbacks = read(self._read), write(self._write), transfer(self._put), transfer(self._get)
        self.library.initialize.argtypes = [read, write, transfer, transfer]
        self.library.word_count.restype = ctypes.c_size_t
        self.library.launches.restype = ctypes.c_uint64
        self.library.invoke.argtypes = [ctypes.POINTER(ctypes.c_void_p)] * 2
        assert self.library.initialize(*self.callbacks) == 0

    def _read(self, _, address):
        if address == 0x40004:
            return self.retired
        if address == 0x40008:
            return 5  # Halted at ECALL.
        self.errors.append(f'unexpected CSR read {address:#x}')
        return 0

    def _write(self, _, address, value):
        try:
            if 0x20000 <= address < 0x40000 and address % 4 == 0:
                self.words[(address - 0x20000) // 4] = value
            else:
                assert address == 0x40018 and value == 1
                physical = {'instructions': [{'word_u32': self.words[i]}
                            for i in range(self.library.word_count())]}
                actual, stats = execute(physical, self.memory, self.base,
                                        self.destination / f'launch-{len(self.stats)}')
                self.memory[:] = actual
                self.stats.append(stats)
                # Distinct retirement observation acknowledges synchronous completion.
                self.retired = (self.retired + 1) % (1 << 32)
        except Exception as error:
            self.errors.append(str(error))

    def _range(self, address, size):
        offset = address - self.base
        assert 0 <= offset <= offset + size <= len(self.memory)
        return offset

    def _put(self, _, address, pointer, size):
        try:
            offset = self._range(address, size)
            self.memory[offset:offset+size] = self.ctypes.string_at(pointer, size)
            self.upload_bytes += size
            return 0
        except Exception as error:
            self.errors.append(str(error))
            return -1

    def _get(self, _, address, pointer, size):
        try:
            offset = self._range(address, size)
            self.ctypes.memmove(pointer, bytes(self.memory[offset:offset+size]), size)
            self.download_bytes += size
            return 0
        except Exception as error:
            self.errors.append(str(error))
            return -1

    def invoke(self, inputs, outputs):
        array = self.ctypes.c_void_p * len(inputs)
        result = self.library.invoke(array(*(self.ctypes.addressof(value) for value in inputs)),
                                    (self.ctypes.c_void_p * len(outputs))(
                                        *(self.ctypes.addressof(value) for value in outputs)))
        assert not self.errors, self.errors
        assert result == 0, result
        assert self.library.launches() == len(self.stats)


def execute(physical, memory, base, destination):
    destination.mkdir(parents=True)
    (destination / 'input.bin').write_bytes(memory)
    bundle = dict(schema_version=1,
                  words=[row['word_u32'] for row in physical['instructions']],
                  config=dict(numerics='rtl', dram_base=base, dram_size=len(memory)),
                  memory_regions=[dict(address=base, file='input.bin')])
    program = destination / 'program.json'
    program.write_text(json.dumps(bundle)+'\n')
    state = destination / 'state'
    result = subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(program),
                             '--compute', 'cpu', '--max-cycles', '100000000',
                             '--dump-state', str(state)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    stats = json.loads(result.stdout)
    assert stats['completed'], stats
    (destination / 'stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    first = base - base % 4096
    pages = []
    for address in range(first, base + len(memory), 4096):
        path = state / f'dram_{address:016x}.bin'
        pages.append(path.read_bytes() if path.exists() else bytes(4096))
    actual = b''.join(pages)[base-first:base-first+len(memory)]
    return actual, stats
