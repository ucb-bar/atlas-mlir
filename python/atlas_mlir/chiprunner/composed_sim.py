"""Offline composed transport using emitted NPU words and a Shuttle ELF in Spike.

No tensor operations are implemented here. NPU words are independently decoded,
re-encoded, and executed by the pinned cycle model. Spike executes host code.
A linked step ELF runs unchanged in Spike with Atlas devices (spike_atlas.cc): its IMEM
and CSR words reach the model checker's functional runtime, which executes every launch
in the pinned model (numerics="rtl") over the shared VMEM. A step application without
an ELF runs as the model checker runs one: its generated driver natively at the published
window address, with the same runtime.
"""
from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import select
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

if __package__:
    from .kit_manifest import HOST_WINDOW, IMEM_WINDOW, PAGE, safe_path
else:
    from kit_manifest import HOST_WINDOW, IMEM_WINDOW, PAGE, safe_path

MODEL_COMMIT = 'b97b2f5a97e432141c4ef8504bec396651d77a8f'
VMEM_BASE = 0x20000000
VMEM_BYTES = 1536 * 1024
HOST_BASE, HOST_BYTES = HOST_WINDOW[0], HOST_WINDOW[1] - HOST_WINDOW[0]


def _hash(path):
    return hashlib.sha256(safe_path(path).read_bytes()).hexdigest()


def _tool(name, default):
    value = os.environ.get(name) or shutil.which(default)
    if not value:
        raise ValueError(f'{name} or {default} on PATH is required')
    # Executable symlinks are common in Python environments. Validate the
    # final target before opening or executing it.
    path = safe_path(os.path.realpath(value))
    if not path.is_file():
        raise ValueError(f'executable is absent: {name}')
    return os.path.abspath(value)


def _model_root():
    value = os.environ.get('ATLAS_NPU_MODEL_ROOT') or os.environ.get('NPU_MODEL_ROOT')
    if not value:
        raise ValueError('ATLAS_NPU_MODEL_ROOT is required')
    root = safe_path(value)
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], timeout=10, text=True).strip()
    if head != MODEL_COMMIT:
        raise ValueError('npu_model is not at the required pinned revision')
    # Restrict Git's inspection to the actual model source and configuration.
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain',
        '--untracked-files=all', '--', 'npu_model', 'pyproject.toml'], timeout=10, text=True)
    if dirty:
        raise ValueError('pinned npu_model source has local changes')
    return root


def _application_runtime():
    """The model checker's functional runtime (generated host callbacks over the pinned model)."""
    if not os.environ.get('COMPOSED_RUNTIME_PATH'):
        raise ValueError('step regions require COMPOSED_RUNTIME_PATH')
    runtime = safe_path(os.environ['COMPOSED_RUNTIME_PATH'])
    return runtime, {name: _hash(runtime / name) for name in ('functional_runtime.py', 'model_runtime.py')}


def runtime_identity():
    """Identity bound into the resumable runner journal before any execution."""
    root = _model_root()
    atlas = safe_path(os.environ['ATLAS_ROOT'])
    python = _tool('COMPOSED_MODEL_PYTHON', 'python3.14')
    spike = _tool('SPIKE', 'spike')
    decoder = atlas / 'tools' / 'check_program_decode.py'
    binder = atlas / 'tools' / 'model_instructions.py'
    inventory = atlas / 'docs' / 'selected-variant-inventory.json'
    identity = dict(npu_model_commit=MODEL_COMMIT, numerics='rtl',
        model_python_sha256=_hash(os.path.realpath(python)), spike_sha256=_hash(os.path.realpath(spike)),
        decoder_sha256=_hash(decoder), binder_sha256=_hash(binder),
        inventory_sha256=_hash(inventory))
    if os.environ.get('COMPOSED_RUNTIME_PATH'):
        identity['step_runtime'] = _application_runtime()[1]
        identity['spike_devices_sha256'] = _hash(Path(__file__).with_name('spike_atlas.cc'))
    return identity


class SimTransport:
    """Byte transport with a single outstanding, bounded simulator launch."""

    def __init__(self, root, work, manifest):
        self.root = safe_path(root)
        self.work = safe_path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.identity = runtime_identity()
        self.pages = {}
        self.pending = None
        self.counter = 0
        self.closed = False

    def _member(self, name):
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError('simulator artifact must be a relative bundle member')
        return safe_path(self.root / name)

    def write(self, address, data):
        if self.closed or not isinstance(address, int) or address < 0:
            raise ValueError('invalid simulator write')
        data = bytes(data)
        cursor = 0
        while cursor < len(data):
            page, offset = divmod(address + cursor, PAGE)
            count = min(PAGE - offset, len(data) - cursor)
            self.pages.setdefault(page, bytearray(PAGE))[offset:offset + count] = data[cursor:cursor + count]
            cursor += count

    def read(self, address, size):
        if self.closed or not isinstance(address, int) or address < 0 or not isinstance(size, int) or size < 0:
            raise ValueError('invalid simulator read')
        result = bytearray(size)
        cursor = 0
        while cursor < size:
            page, offset = divmod(address + cursor, PAGE)
            count = min(PAGE - offset, size - cursor)
            if page in self.pages:
                result[cursor:cursor + count] = self.pages[page][offset:offset + count]
            cursor += count
        return bytes(result)

    def prepare(self, region):
        if region['kind'] == 'elf':
            # Load the step ELF as the bench loads it (step_loads), with each load's zero fill as before.
            raw = self._member(region['elf']).read_bytes()
            for address, data, count, flags in _elf_loads(raw)[1]:
                self.write(address + len(data), bytes(count-len(data)))
            for address, data, count, flags in step_loads(raw, runner_ranges(region))[0]:
                self.write(address, data)
        if region['kind'] == 'host':
            entry, loads = _elf_loads(self._member(region['elf']).read_bytes())
            if entry != region['entry']:
                raise ValueError('host entry differs from emitted ELF')
            for address, data, count, flags in loads:
                if address < HOST_BASE or address+count > HOST_BASE+HOST_BYTES:
                    raise ValueError('host ELF does not fit Shuttle scratchpad')
                self.write(address, data + bytes(count-len(data)))
            status = region['status']
            self.write(status['address'], status.get('initial_value', 0).to_bytes(8, 'little'))

    def launch(self, region):
        if self.closed or self.pending is not None:
            raise ValueError('simulator already has an outstanding launch')
        self.counter += 1
        work = safe_path(self.work / ('launch-%d-%s' % (self.counter, region['id'])))
        work.mkdir(exist_ok=False)
        self.pending = (region, work)

    def wait(self, region, timeout):
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('simulator timeout must be finite and positive')
        if self.pending is None or self.pending[0] != region:
            raise ValueError('simulator wait does not match launch')
        _, work = self.pending
        try:
            if region['kind'] == 'device':
                return self._device(region, work, timeout)
            if region['kind'] == 'host':
                return self._host(region, work, timeout)
            if region['kind'] in ('elf', 'application'):
                return self._step(region, work, timeout)
            raise ValueError('unknown simulator region kind')
        finally:
            self.pending = None

    def _device(self, region, work, timeout):
        root = _model_root()
        atlas = safe_path(os.environ['ATLAS_ROOT'])
        request = dict(program=str(self._member(region['program'])),
            format=region.get('format', 'physical'),
            inventory=str(self._member(region['inventory'])) if region.get('inventory') else str(atlas/'docs'/'selected-variant-inventory.json'),
            model_root=str(root), atlas_root=str(atlas),
            max_cycles=region.get('max_cycles', 1_000_000),
            pages=[dict(address=page * PAGE, data=bytes(data).hex()) for page, data in sorted(self.pages.items())],
            outputs=region['outputs'])
        request_path = work / 'request.json'
        request_path.write_text(json.dumps(request))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        script = safe_path(Path(__file__).absolute())
        with (work/'model.log').open('wb') as log:
            subprocess.run([_tool('COMPOSED_MODEL_PYTHON', 'python3.14'), str(script), '--device-worker', str(request_path)],
                cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
        result = json.loads((work/'result.json').read_text())
        for row in result.pop('outputs'):
            self.write(row['address'], bytes.fromhex(row['data']))
        # Verify the pin again so changes during a run cannot silently qualify.
        _model_root()
        return result

    def _step(self, region, work, timeout):
        runtime, _ = _application_runtime()
        request = dict(kind=region['kind'], window=region['window'], launches=region['launches'], outputs=region['outputs'],
            deployment=str(self._member(region['deployment'])),
            images=[{key: str(self._member(row[key])) for key in ('program', 'plan')} for row in region['images']],
            atlas_root=str(safe_path(os.environ['ATLAS_ROOT'])), model_root=str(_model_root()), runtime=str(runtime),
            cache=str(self.work / 'libraries'), timeout=timeout,
            pages=[dict(address=page * PAGE, data=bytes(data).hex()) for page, data in sorted(self.pages.items())])
        if region['kind'] == 'elf':
            request.update(elf=str(self._member(region['elf'])), entry=region['entry'], spike=_tool('SPIKE', 'spike'),
                           record_tag=self.manifest['runner']['record_tag'])
        else:
            request.update(sources=[str(self._member(name)) for name in region['sources']])
        request_path = work / 'request.json'
        request_path.write_text(json.dumps(request))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (work/'model.log').open('wb') as log:
            done = subprocess.run([_tool('COMPOSED_MODEL_PYTHON', 'python3.14'), str(safe_path(Path(__file__).absolute())),
                                   '--step-worker', str(request_path)],
                                  cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout + 60)
        if done.returncode:
            lines = (work/'model.log').read_text(errors='replace').strip().splitlines()
            raise ValueError('step %s failed: %s' % (region['id'], lines[-1] if lines else 'no worker output'))
        result = json.loads((work/'result.json').read_text())
        for row in result.pop('outputs'):
            self.write(row['address'], bytes.fromhex(row['data']))
        _model_root()
        return result

    def _host(self, region, work, timeout):
        original = self._member(region['elf']).read_bytes()
        entry, loads = _elf_loads(original)
        if region['entry'] != entry:
            raise ValueError('host entry differs from emitted ELF')
        for address, data, memory_bytes, flags in loads:
            if address < HOST_BASE or address + memory_bytes > HOST_BASE + HOST_BYTES:
                raise ValueError('host ELF does not fit Shuttle scratchpad')
        status = region['status']
        address, done = status['address'], status['done_value']
        if address % 8 or not 0 <= done < 1 << 64:
            raise ValueError('Spike completion requires an aligned 64-bit status word')
        if int.from_bytes(self.read(address, 8), 'little') == done:
            raise ValueError('completion status already equals done before host execution')
        # Appended PT_LOAD payloads provide the actual predecessor bytes. Code
        # loads keep the producer's ELF bytes and memory extents unchanged.
        loads.extend((page * PAGE, bytes(data), PAGE, 6) for page, data in sorted(self.pages.items()))
        # --disable-dtb omits Spike's reset ROM. Supply its reset-to-entry
        # jump explicitly, without an external device-tree compiler.
        reset_base = 0x1000
        high = (entry + 0x800) >> 12
        low = entry - (high << 12)
        reset = struct.pack('<II', (high << 12) | (5 << 7) | 0x37,
                            ((low & 0xfff) << 20) | (5 << 15) | 0x67)
        loads.append((reset_base, reset, len(reset), 5))
        payload = work/'payload.elf'
        payload.write_bytes(_payload_elf(original[:64], loads))
        spans = [(HOST_BASE, HOST_BYTES), (reset_base, PAGE)]
        # A declared step window is host memory too, for example a workspace the host code fills.
        if 'window' in region:
            spans.append((region['window']['address'], region['window']['bytes']))
        spans.extend((page * PAGE, PAGE) for page in self.pages)
        spans.extend((a, (n + PAGE - 1)//PAGE*PAGE) for a, n in [(address//PAGE*PAGE, 8)])
        maps = _merge_spans(spans)
        commands = work/'commands.txt'
        commands.write_text(f'until mem {address:x} {done:x}\ndump\nquit\n')
        argv = [_tool('SPIKE', 'spike'), '--isa=rv64gcv', '--priv=m', '--disable-dtb',
            '-m'+','.join(f'0x{a:x}:0x{n:x}' for a, n in maps), f'--pc=0x{entry:x}',
            '-d', '--debug-cmd='+str(commands), str(payload)]
        with (work/'spike.log').open('wb') as log:
            subprocess.run(argv, cwd=work, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
        # Spike's dump command writes each mapped physical memory region.
        dumps = {}
        for base, count in maps:
            path = work/f'mem.0x{base:x}.bin'
            raw = path.read_bytes()
            if len(raw) != count:
                raise ValueError('Spike memory dump is truncated')
            dumps[base] = raw
        def observed(a, n):
            for base, raw in dumps.items():
                if base <= a and a+n <= base+len(raw):
                    return raw[a-base:a-base+n]
            raise ValueError('Spike output is outside mapped memory')
        if int.from_bytes(observed(address, 8), 'little') != done:
            raise ValueError('Spike did not reach declared completion status')
        for row in region['outputs']:
            self.write(row['address'], observed(row['address'], row['bytes']))
        self.write(address, observed(address, 8))
        return dict(completed=True, backend='spike', status=done)

    def close(self):
        self.pending = None
        self.closed = True


def _merge_spans(spans):
    result = []
    for start, size in sorted(spans):
        end = start + size
        if result and start <= result[-1][0]+result[-1][1]:
            old_start, old_size = result[-1]
            result[-1] = (old_start, max(old_start+old_size, end)-old_start)
        else:
            result.append((start, size))
    return result


def _elf_loads(raw):
    if len(raw) < 64 or raw[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', raw, 18)[0] != 243:
        raise ValueError('host image must be ELF64 little-endian RISC-V')
    entry, phoff = struct.unpack_from('<QQ', raw, 24)
    phsize, phcount = struct.unpack_from('<HH', raw, 54)
    if phsize != 56 or phoff+phsize*phcount > len(raw):
        raise ValueError('invalid host ELF program headers')
    loads = []
    for i in range(phcount):
        kind, flags, offset, virtual, physical, count, memory, align = struct.unpack_from('<IIQQQQQQ', raw, phoff+i*phsize)
        if kind != 1 or memory == 0:  # an empty load segment occupies no memory
            continue
        if count > memory or offset+count > len(raw) or physical != virtual:
            raise ValueError('invalid host ELF load segment')
        loads.append((physical, raw[offset:offset+count], memory, flags))
    if not loads:
        raise ValueError('host ELF has no load segments')
    return entry, loads


# Window spans a step ELF clears at startup, before it binds Atlas: (address, bytes) pairs of 64-bit words, then (0, 0).
CLEARED_TABLE = b'atlas_window_zero'


def _image(loads, address, size):
    """The file-backed image bytes of one load at [address, address + size)."""
    for start, data, memory, flags in loads:
        if start <= address and address + size <= start + len(data):
            return data[address - start:address - start + size]
    raise ValueError('range is not file-backed by one ELF load')


def cleared_spans(raw):
    """The spans of an ELF's image that its startup clears, each holding only zero bytes in the image.

    Clearing them on chip leaves the memory that loading them gives, so a loader need not send them. An ELF
    without the table clears nothing.
    """
    _, loads = _elf_loads(raw)
    shoff, = struct.unpack_from('<Q', raw, 40)
    shsize, count = struct.unpack_from('<HH', raw, 58)
    if not shoff or shsize != 64 or shoff + count * shsize > len(raw):
        return []
    sections = [struct.unpack_from('<IIQQQQIIQQ', raw, shoff + i * shsize) for i in range(count)]
    found = []
    for section in sections:
        if section[1] != 2 or section[6] >= count:  # SHT_SYMTAB and its string table
            continue
        strings = sections[section[6]][4]
        for at in range(section[4], section[4] + section[5] - 23, 24):
            name, _, _, _, value, size = struct.unpack_from('<IBBHQQ', raw, at)
            if raw[strings + name:].split(b'\0', 1)[0] == CLEARED_TABLE:
                found.append((value, size))
    if not found:
        return []
    if len(found) != 1 or found[0][1] < 16 or found[0][1] % 16:
        raise ValueError('invalid cleared-span table')
    words = struct.unpack('<%dQ' % (found[0][1] // 8), _image(loads, *found[0]))
    spans = list(zip(words[0::2], words[1::2]))
    if spans.pop() != (0, 0) or any(not size for _, size in spans):
        raise ValueError('cleared-span table must end with its only empty entry')
    for (address, size), following in zip(spans, spans[1:] + [None]):
        if any(_image(loads, address, size)) or following and following[0] < address + size:
            raise ValueError('cleared spans must be ordered, disjoint, and zero in the image')
    return spans


def runner_ranges(region):
    """The ranges the composed runner writes after loading a step and before starting it."""
    return [(row['address'], row['bytes']) for role in ('inputs', 'weights', 'outputs') for row in region.get(role, [])]


def step_loads(raw, written=()):
    """The PT_LOAD pieces a loader sends for a step ELF, as (address, data, memory bytes, flags), and the bytes left out.

    It leaves out the spans the ELF clears at startup and each range in `written` (what the runner writes before the
    start) whose image bytes are all zero. The memory at the start equals what loading every PT_LOAD gives.
    """
    _, loads = _elf_loads(raw)
    cleared = cleared_spans(raw)
    skip = list(cleared)
    for address, size in written:
        if any(a < address + size and address < a + n for a, n in cleared):
            raise ValueError('a runner range overlaps a span the ELF clears at startup')
        try:
            if not any(_image(loads, address, size)):
                skip.append((address, size))
        except ValueError:
            continue  # not wholly inside one file-backed load: load it as packaged
    pieces, left_out = [], 0
    for start, data, memory, flags in loads:
        cursor, end = start, start + len(data)
        for a, n in sorted(skip):
            low, high = max(a, cursor), min(a + n, end)
            if high <= low:
                continue
            if low > cursor:
                pieces.append((cursor, data[cursor - start:low - start], low - cursor, flags))
            left_out += high - low
            cursor = high
        if cursor < end:
            pieces.append((cursor, data[cursor - start:], end - cursor, flags))
    return pieces, left_out


def _payload_elf(header, loads):
    header = bytearray(header)
    struct.pack_into('<Q', header, 32, 64)
    struct.pack_into('<Q', header, 40, 0)
    struct.pack_into('<HHHHHH', header, 52, 64, 56, len(loads), 0, 0, 0)
    payload = bytearray(header) + bytearray(56*len(loads))
    for i, (address, data, memory, flags) in enumerate(loads):
        offset = len(payload)
        struct.pack_into('<IIQQQQQQ', payload, 64+i*56, 1, flags, offset, address, address, len(data), memory, 1)
        payload.extend(data)
    names = b'\0.shstrtab\0'
    name_offset = len(payload)
    payload.extend(names)
    section_offset = len(payload)
    payload.extend(bytes(64))
    payload.extend(struct.pack('<IIQQQQIIQQ', 1, 3, 0, 0, name_offset, len(names), 0, 0, 1, 0))
    struct.pack_into('<Q', payload, 40, section_offset)
    struct.pack_into('<HHH', payload, 58, 64, 2, 1)
    return payload


def _device_worker(path):
    request_path = safe_path(path)
    request = json.loads(request_path.read_text())
    sys.path.insert(0, str(safe_path(request['model_root'])))
    sys.path.insert(0, str(safe_path(request['atlas_root'])))
    import torch
    torch.set_num_threads(1)
    from npu_model.configs import isa_definition as isa
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.hardware.core import Core
    from npu_model.logging import Logger, LoggerConfig
    from npu_model.software.program import InstantiableProgram
    from tools.check_program_decode import decode_binary
    from tools.model_instructions import model_instruction
    from dataclasses import replace
    if request['format'] != 'physical':
        raise ValueError('device simulator accepts physical emitted program words')
    binary = safe_path(request['program']).read_bytes()
    inventory = json.loads(safe_path(request['inventory']).read_text())
    decoded = decode_binary(inventory, binary)
    instructions = [model_instruction(row, inventory, isa) for row in decoded['instructions']]
    if not instructions:
        raise ValueError('device program is empty')
    config = DefaultHardwareConfig()
    config.arch_state_config = replace(config.arch_state_config, numerics='rtl',
        dram_base=VMEM_BASE, dram_size=0x10_8000_0000-VMEM_BASE)
    logger = Logger(LoggerConfig(filename=str(request_path.parent/'trace.json')))
    core = Core(config, logger)
    try:
        core.load_program(InstantiableProgram(instructions))
        state = core.arch_state
        class BusMemory:
            # The board bus maps these addresses to the same six VMEM banks.
            # Only byte routing is adapted; all instruction arithmetic and
            # timing checks execute in the unmodified pinned model.
            def __init__(self, vmem, dram):
                self.vmem, self.dram = vmem, dram
            def __getitem__(self, key):
                if VMEM_BASE <= key.start <= key.stop <= VMEM_BASE+VMEM_BYTES:
                    return self.vmem[key.start-VMEM_BASE:key.stop-VMEM_BASE]
                return self.dram[key]
            def __setitem__(self, key, value):
                if VMEM_BASE <= key.start <= key.stop <= VMEM_BASE+VMEM_BYTES:
                    self.vmem[key.start-VMEM_BASE:key.stop-VMEM_BASE] = value
                else:
                    self.dram[key] = value
        state.dram = BusMemory(state.vmem, state.dram)
        for row in request['pages']:
            address, data = row['address'], bytes.fromhex(row['data'])
            tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            if VMEM_BASE <= address and address+len(data) <= VMEM_BASE+VMEM_BYTES:
                state.vmem[address-VMEM_BASE:address-VMEM_BASE+len(data)] = tensor
            elif state.cfg.dram_base <= address and address+len(data) <= state.cfg.dram_base+state.cfg.dram_size:
                state.write_dram(address, tensor)
        limit = request['max_cycles']
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError('max_cycles must be positive')
        cycles = 0
        while not core.is_finished() and cycles < limit:
            core.tick()
            cycles += 1
        if not core.is_finished():
            raise TimeoutError('device cycle limit reached')
        core.stop()
        if not state.halted or state.halt_reason not in ('ecall', 'ebreak'):
            raise ValueError('device program did not signal a legal completion trap')
        outputs = []
        for row in request['outputs']:
            address, count = row['address'], row['bytes']
            if VMEM_BASE <= address and address+count <= VMEM_BASE+VMEM_BYTES:
                data = state.vmem[address-VMEM_BASE:address-VMEM_BASE+count].numpy().tobytes()
            else:
                data = state.read_dram(address, count).numpy().tobytes()
            outputs.append(dict(address=address, data=data.hex()))
        result = dict(completed=True, backend='npu_model', numerics='rtl', cycles=cycles,
            instructions=core.total_completed, halt_reason=state.halt_reason, outputs=outputs)
        (request_path.parent/'result.json').write_text(json.dumps(result))
    finally:
        core.close()
        logger.close()


# The model checker's shim: bind the generated deployment to the model's callbacks, then run it.
_SHIM = r'''#include "deployment_adapter.h"
static atlas_host model_host;
int model_bind(atlas_host_read32 read32,atlas_host_write32 write32,
               atlas_tile_write write,atlas_tile_read read) {
  atlas_host_map map={MERLIN_DEPLOYMENT_IMEM_BASE,MERLIN_DEPLOYMENT_CSR_BASE,MERLIN_DEPLOYMENT_IMEM_BYTES};
  atlas_host_mmio mmio={read32,write32,0};
  atlas_host_memory memory={1,0,0};
  atlas_tile_memory_io io={write,read,0};
  int status=atlas_host_init(&model_host,map,mmio,memory);
  return status ? status : merlin_deployment_bind(&model_host,io,MERLIN_DEPLOYMENT_MAX_POLLS);
}
int model_run(void) { return merlin_deployment_run(); }
'''


def _build(compiler, arguments, sources, cache, suffix):
    """Build once per source set; the key binds the compiler, flags, and every source byte."""
    compiler = shutil.which(compiler)
    if compiler is None:
        raise ValueError('a host compiler is required for simulated step regions')
    key = hashlib.sha256(json.dumps([_hash(os.path.realpath(compiler)), arguments]).encode()
                         + b''.join(safe_path(path).read_bytes() for path in sources)).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / (key + suffix)
    if not library.is_file():
        temporary = cache / (key + '.tmp' + suffix)
        subprocess.run([compiler, *arguments, *map(str, sources), '-o', str(temporary)], check=True, capture_output=True, text=True)
        temporary.replace(library)
    return library, key


def _step_runtime(request, work, base, size):
    from functional_runtime import FunctionalModelRuntime
    images = [safe_path(row['program']).read_bytes() for row in request['images']]
    plans = {hashlib.sha256(raw).hexdigest(): json.loads(safe_path(row['plan']).read_text())
             for raw, row in zip(images, request['images'])}
    return FunctionalModelRuntime(safe_path(request['atlas_root']), safe_path(request['model_root']), images,
        work / 'execution', memory_base=base, memory_bytes=size,
        image_plans=plans, max_instructions=128000000, trace_payloads=False, compact_dram_prefix=True)


def _application_sources(request, shim):
    """The generated step source and the shim, plus the Atlas host runtime when no device region embeds it."""
    sources = [safe_path(request['sources'][0]), shim]
    if all(call['placement'] == 'host' for call in json.loads(safe_path(request['deployment']).read_text())['calls']):
        # As the step link does for a host-only deployment: the shim binds atlas_host_init.
        sources.append(safe_path(request['atlas_root']) / 'runtime' / 'atlas_host.c')
    return sources


def _run_application(request, work, base, size):
    import ctypes as ct
    libc = ct.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_int, ct.c_int, ct.c_long]
    libc.mmap.restype = ct.c_void_p
    # MAP_FIXED_NOREPLACE: the generated driver uses the published window addresses.
    if libc.mmap(base, size, 3, 0x02 | 0x20 | 0x100000, -1, 0) != base:
        raise OSError(ct.get_errno(), 'published window address is unavailable')
    for row in request['pages']:
        data = bytes.fromhex(row['data'])
        start, end = max(row['address'], base), min(row['address'] + len(data), base + size)
        if start < end:
            ct.memmove(start, data[start - row['address']:end - row['address']], end - start)
    shim = work / 'shim.c'
    shim.write_text(_SHIM)
    sources = _application_sources(request, shim)
    library, key = _build('gcc', ['-O2', '-std=c11', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                                  '-I', str(sources[0].parent)], sources, safe_path(request['cache']), '.so')
    library = ct.CDLL(str(library))
    runtime = _step_runtime(request, work, base, size)
    library.model_bind.argtypes = [type(callback) for callback in runtime.callbacks]
    library.model_bind.restype = library.model_run.restype = ct.c_int
    bind_status = library.model_bind(*runtime.callbacks)
    status = library.model_run() if bind_status == 0 else bind_status
    execution = runtime.finish()
    if status:
        raise ValueError(f'generated step application failed: bind={bind_status}, run={status}')
    read = lambda address, count: ct.string_at(address, count)
    return execution, read, dict(library_sources_sha256=key)


def _run_elf(request, work, base, size):
    """Run the unchanged step ELF in Spike; Atlas IMEM and CSR words reach the functional runtime."""
    vmem_path = work / 'vmem.bin'
    with vmem_path.open('wb') as stream:
        stream.truncate(VMEM_BYTES)
    original = safe_path(request['elf']).read_bytes()
    entry, _ = _elf_loads(original)
    if entry != request['entry']:
        raise ValueError('step entry differs from emitted ELF')
    loads = []
    with vmem_path.open('r+b') as stream:
        for row in request['pages']:
            address, data = row['address'], bytes.fromhex(row['data'])
            if VMEM_BASE <= address and address + len(data) <= VMEM_BASE + VMEM_BYTES:
                stream.seek(address - VMEM_BASE); stream.write(data)
            elif HOST_BASE <= address and address + len(data) <= HOST_BASE + HOST_BYTES:
                loads.append((address, data, len(data), 7))
            else:
                raise ValueError('staged page is outside VMEM and the host scratchpad')
    payload = work / 'payload.elf'
    payload.write_bytes(_payload_elf(original[:64], loads))
    plugin, _ = _build('g++', ['-std=c++17', '-O2', '-shared', '-fPIC', '-Wall', '-I',
                               os.environ.get('SPIKE_INCLUDE') or str(Path(os.path.realpath(request['spike'])).parents[1] / 'include')],
                       [Path(__file__).with_name('spike_atlas.cc')], safe_path(request['cache']), '.so')
    runtime = _step_runtime(request, work, base, size)
    descriptor = os.open(vmem_path, os.O_RDWR)
    shared = mmap.mmap(descriptor, size, offset=base - VMEM_BASE)
    os.close(descriptor)
    runtime.memory = shared  # the runtime's window is the Spike device's VMEM
    requests, plugin_requests = os.pipe()
    plugin_responses, responses = os.pipe()
    uart = work / 'uart.log'
    # Plugin devices are placed only from a generated device tree, so this path needs dtc on PATH.
    if shutil.which('dtc') is None:
        raise ValueError('dtc on PATH is required to attach the simulated Atlas devices')
    argv = [request['spike'], '--isa=rv64gcv_zicntr', '--priv=m', f'-m0x{HOST_BASE:x}:0x{HOST_BYTES:x}',
            f'--pc=0x{entry:x}', f'--extlib={plugin}', f'--device=atlas_vmem,0x{VMEM_BASE:x},0x{VMEM_BYTES:x},{vmem_path}',
            f'--device=atlas_mmio,0x{IMEM_WINDOW[0]:x},{plugin_requests},{plugin_responses}',
            f'--device=atlas_uart,0x10020000,{uart},{request["record_tag"]}', str(payload)]
    if any(',' in str(path) for path in (vmem_path, uart)):
        raise ValueError('simulator paths must not contain commas')
    deadline = time.monotonic() + request['timeout']
    with (work / 'spike.log').open('wb') as log:
        process = subprocess.Popen(argv, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                                   pass_fds=(plugin_requests, plugin_responses))
    os.close(plugin_requests); os.close(plugin_responses)
    try:
        pending = b''
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([requests], [], [], remaining)[0]:
                raise TimeoutError('step ELF did not complete before its deadline')
            chunk = os.read(requests, 4096)
            if not chunk:
                break
            pending += chunk
            while len(pending) >= 13:
                operation, address, word = pending[:1], *struct.unpack_from('<QI', pending, 1)
                pending = pending[13:]
                if operation == b'R':
                    os.write(responses, struct.pack('<I', runtime.read(None, address) & 0xffffffff))
                elif operation == b'W':
                    runtime.write(None, address, word)
                else:
                    raise ValueError('invalid device request')
        code = process.wait(timeout=max(1, deadline - time.monotonic()))
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
        os.close(requests); os.close(responses)
    execution = runtime.finish()
    text = uart.read_text(errors='replace') if uart.is_file() else ''
    records = text.splitlines()
    tag = request['record_tag']
    if code != 0 or not records or records[-1] != tag + ' COMPLETE':
        raise ValueError(f'step ELF did not report completion (exit {code}): ' + (records[-1] if records else 'no UART output'))
    # The bench's record rule: status 0, one hash per declared range, and each equals the dumped bytes.
    if __package__:
        from .run_kit import step_hashes
    else:
        from run_kit import step_hashes
    rows = json.loads(safe_path(request['deployment']).read_text())['hash_ranges']
    reported = step_hashes(rows, text, tag)
    read = lambda address, count: bytes(shared[address - base:address - base + count])
    for row in rows:
        if reported[row['id']] != hashlib.sha256(read(row['address'], row['bytes'])).hexdigest():
            raise ValueError('UART hash differs from dumped bytes: ' + row['id'])
    return execution, read, dict(plugin_sha256=_hash(plugin), uart_records=len(records), hash_ranges=len(reported))


def _step_worker(path):
    request_path = safe_path(path)
    request = json.loads(request_path.read_text())
    for name in ('runtime', 'atlas_root', 'model_root'):
        sys.path.insert(0, str(safe_path(request[name])))
    import torch
    torch.set_num_threads(1)
    base, size = request['window']['address'], request['window']['bytes']
    if base % PAGE or size % PAGE or not VMEM_BASE <= base < base + size <= VMEM_BASE + VMEM_BYTES:
        raise ValueError('step window must be page aligned inside VMEM')
    run = _run_elf if request['kind'] == 'elf' else _run_application
    execution, read, extra = run(request, request_path.parent, base, size)
    if len(execution['launches']) != request['launches']:
        raise ValueError(f"step made {len(execution['launches'])} launches, expected {request['launches']}")
    outputs = [dict(address=row['address'], data=read(row['address'], row['bytes']).hex()) for row in request['outputs']]
    result = dict(completed=True, backend=execution['backend'], numerics=execution['numerics'],
                  launches=len(execution['launches']), instructions=execution['total_instructions'], outputs=outputs, **extra)
    (request_path.parent / 'result.json').write_text(json.dumps(result))


if __name__ == '__main__':
    workers = {'--device-worker': _device_worker, '--step-worker': _step_worker}
    if len(sys.argv) != 3 or sys.argv[1] not in workers:
        raise SystemExit('internal worker requires a request file')
    workers[sys.argv[1]](sys.argv[2])
