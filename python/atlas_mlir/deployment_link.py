"""Prepare measured platform objects, then link only authenticated EXACT inputs."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import struct
import subprocess

from .deploy import Refusal, check_hashes, digest, safe_path


RECORD_TAG = 'ATLAS'  # UART record prefix; kits declare it as runner.record_tag
# The harness STATUS when a cleared window span reads back nonzero; Atlas is then never bound or run.
WINDOW_NOT_CLEARED = -100

_HARNESS = '#define RECORD_TAG "%s"\n#define WINDOW_NOT_CLEARED (%d)\n' % (RECORD_TAG, WINDOW_NOT_CLEARED) + r'''#include "deployment_adapter.h"
#include "harness_sha256.h"
#include <string.h>
#include <sys/types.h>
static atlas_host host;
/* Zero spans of the window image, cleared before Atlas is bound, so a loader need not send them. Interim: the generic
   form is a sparse ELF whose zero spans are NOBITS, which every loader then skips without reading this table. */
const uint64_t atlas_window_zero[][2]={ATLAS_WINDOW_ZERO{0,0}};
extern ssize_t _write(int,const void *,size_t);
static void emit(const char *s) { _write(1,s,strlen(s)); }
static void number(uint64_t n) {
  char s[21]; unsigned i=sizeof(s)-1; s[i]=0;
  do { s[--i]=(char)('0'+n%10); n/=10; } while(n); emit(s+i);
}
static uint64_t cycles(void *unused) {
  (void)unused; uint64_t n; __asm__ volatile("rdcycle %0":"=r"(n)); return n;
}
static void fence(void) { __asm__ volatile("fence rw,rw":::"memory"); }
static int sync_buffer(void *unused,const atlas_host_buffer *buffer,int before) {
  (void)unused; (void)buffer; (void)before; fence(); return 0;
}
/* Clear every span, then read every cleared byte back (aligned words, bytes at the edges), since no loader verifies
   these bytes. */
static int clear_window(const uint64_t (*zero)[2]) {
  for(const uint64_t (*span)[2]=zero;span[0][1];span++) memset((void *)(uintptr_t)span[0][0],0,(size_t)span[0][1]);
  fence();
  for(;zero[0][1];zero++) {
    uintptr_t at=(uintptr_t)zero[0][0], end=at+(uintptr_t)zero[0][1];
    for(;at<end&&at%8;at++) if(*(const volatile unsigned char *)at) return WINDOW_NOT_CLEARED;
    for(;end-at>=8;at+=8) if(*(const volatile uint64_t *)at) return WINDOW_NOT_CLEARED;
    for(;at<end;at++) if(*(const volatile unsigned char *)at) return WINDOW_NOT_CLEARED;
  }
  return 0;
}
int main(void) {
  (void)merlin_output_sha256;
  const uint64_t (*zero)[2]=atlas_window_zero; __asm__ volatile("":"+r"(zero));
  int result=clear_window(zero);
  atlas_host_map map={MERLIN_DEPLOYMENT_IMEM_BASE,MERLIN_DEPLOYMENT_CSR_BASE,MERLIN_DEPLOYMENT_IMEM_BYTES};
  atlas_host_mmio mmio={atlas_host_direct_read32,atlas_host_direct_write32,0};
  atlas_host_memory memory={0,sync_buffer,0};
  atlas_tile_memory_io io={atlas_host_direct_memory_write,atlas_host_direct_memory_read,0};
  if(!result) result=atlas_host_init(&host,map,mmio,memory);
  if(!result) result=atlas_host_set_clock(&host,cycles,0);
  if(!result) result=merlin_deployment_bind(&host,io,MERLIN_DEPLOYMENT_MAX_POLLS);
  uint64_t begin=cycles(0);
  if(!result) result=merlin_deployment_run();
  uint64_t elapsed=cycles(0)-begin;
  emit(RECORD_TAG " STATUS ");
  if(result<0) { emit("-"); number((uint64_t)-(int64_t)result); }
  else number((uint64_t)result);
  emit(" CYCLES "); number(elapsed); emit("\n");
  if(result) return 1; fence();
  for(size_t i=0;i<merlin_deployment_hash_count;i++) {
    merlin_sha256 hash; unsigned char value[32]; char hex[65];
    merlin_sha256_init(&hash);
    merlin_sha256_update(&hash,(const void *)merlin_deployment_hash_ranges[i].address,merlin_deployment_hash_ranges[i].bytes);
    merlin_sha256_final(&hash,value);
    for(unsigned j=0;j<32;j++) { hex[2*j]="0123456789abcdef"[value[j]>>4]; hex[2*j+1]="0123456789abcdef"[value[j]&15]; }
    hex[64]=0; emit(RECORD_TAG " HASH "); emit(merlin_deployment_hash_ranges[i].name);
    emit(" "); emit(merlin_deployment_hash_ranges[i].role); emit(" "); emit(hex); emit("\n");
  }
  emit(RECORD_TAG " COMPLETE\n"); return 0;
}
'''


def _run(command, log, *, env=None):
    with safe_path(log).open('w') as stream:
        result = subprocess.run(list(map(str, command)), stdout=stream,
                                stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise Refusal('platform build failed; see ' + str(log))


# GNU ar and ranlib record member timestamps, uids, and gids unless told not to; the objects receipt hashes
# every platform archive. Set after project() and before the platform's subdirectories, so every library uses it.
_DETERMINISTIC_ARCHIVES = '''foreach(lang C CXX)
  set(CMAKE_${lang}_ARCHIVE_CREATE "<CMAKE_AR> qcD <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_APPEND "<CMAKE_AR> qD <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_FINISH "<CMAKE_RANLIB> -D <TARGET>")
endforeach()
set(CMAKE_ASM_CREATE_STATIC_LIBRARY "<CMAKE_AR> crD <TARGET> <LINK_FLAGS> <OBJECTS>" "<CMAKE_RANLIB> -D <TARGET>")
'''


def _configure_check(log):
    if 'Manually-specified variables were not used by the project' in safe_path(log).read_text():
        raise Refusal('CMake reports unused platform variables; see ' + str(log))


_FP_NAMES = ([f'ft{i}' for i in range(8)] + ['fs0', 'fs1'] + [f'fa{i}' for i in range(8)]
             + [f'fs{i}' for i in range(2, 12)] + [f'ft{i}' for i in range(8, 12)])


def _executable_code(path):
    """(address, bytes) of every executable section of a 64-bit little-endian ELF file."""
    raw = Path(path).read_bytes()
    if raw[:4] != b'\x7fELF' or raw[4] != 2 or raw[5] != 1:
        raise Refusal('host ELF is not a 64-bit little-endian ELF file')
    shoff, = struct.unpack_from('<Q', raw, 40)
    shsize, shcount = struct.unpack_from('<2H', raw, 58)
    if shcount and (shsize != 64 or shoff + shsize * shcount > len(raw)):
        raise Refusal('host ELF has an invalid section header table')
    rows = [struct.unpack_from('<IIQQQQ', raw, shoff + i * shsize) for i in range(shcount)]
    return [(address, raw[offset:offset + size]) for _, kind, flags, address, offset, size in rows
            if flags & 4 and kind != 8]  # SHF_EXECINSTR with file contents (not SHT_NOBITS)


# (mask, match, name, reason) of 32-bit instructions refused wherever they appear.
_REFUSED = ((0x707f, 0x100f, 'fence.i', 'the chip core does not discard instructions fetched before it '
             '(ucb-bar/shuttle issue 19)'),
            (0xfc00707f, 0x40001057, 'vfmv.f.s', 'the chip core can drop its FP-register write and stall the '
             'next reader forever (ucb-bar/shuttle PR 22); store the element with vse and load it with flw'))


def _instruction_effects(halves, address):
    """(length, refused, FP loads, FP writes) of the instruction at address, or None past the code.

    Lengths follow the core: 32 bits when the low two bits are 11, else 16. Compressed FP loads are c.fld and
    c.fldsp. refused is the matching _REFUSED row or None. fence.i is decoded by opcode and funct3 alone, as
    the core does, and vfmv.f.s by opcode, funct3, and funct6, ignoring its vs1 and vm fields.
    """
    low = halves.get(address)
    if low is None or low & 3 == 3 and address + 2 not in halves:
        return None
    if low & 3 != 3:
        rd = {0x2000: (low >> 2 & 7) + 8, 0x2002: low >> 7 & 31}.get(low & 0xe003)
        return 2, None, {rd} - {None}, {rd} - {None}
    value = low | halves[address + 2] << 16
    opcode, funct3, rd = value & 0x7f, value >> 12 & 7, value >> 7 & 31
    load = opcode == 0x07 and 1 <= funct3 <= 4
    writes = (load or opcode in (0x43, 0x47, 0x4b, 0x4f)  # FP loads and fused multiply-adds
              or opcode == 0x53 and value >> 27 not in (0x14, 0x18, 0x1c))  # OP-FP except compare, to-int, fmv.x
    refused = next((row for row in _REFUSED if value & row[0] == row[1]), None)
    return 4, refused, {rd} if load else set(), {rd} if writes else set()


def _core_errata_check(disassembly, code):
    """Refuse host code that the chip's Shuttle core, which keeps two known bugs, can execute wrongly.

    ucb-bar/shuttle issue 19: fence.i does not discard instructions fetched before it, so code the core
    writes may run stale. Host code is loaded only by the debugger, so any fence.i is refused.
    PR 22: a vector result bound for a scalar FP register is written only when the divider happens to hold a
    finished result; otherwise it is dropped and the next reader of that register stalls forever. Saturn
    returns such results only through vfmv.f.s, so any vfmv.f.s is refused (store with vse, then load).
    Issue 20: a two-wide packet may pair an FP-register write with the next instruction, an FP load to the
    same register, without a stall, and the older value can survive. Any such adjacent pair is refused.
    Instructions are decoded from the executable section bytes, so data directives and instructions that
    straddle them count. Decoding runs on from each section start, symbol, and objdump line address.
    Not covered: a jump into the middle of an instruction already decoded from an earlier start, since
    decoding does not restart at every halfword.
    """
    labels, seeds = {}, set()
    for line in disassembly.splitlines():
        match = re.match(r'^([0-9a-f]+) <([^>]+)>:|^\s*([0-9a-f]+):', line)
        if match and match[1]:
            labels[int(match[1], 16)] = match[2]
        elif match:
            seeds.add(int(match[3], 16))
    halves = {}
    for start, data in code:
        count = len(data) // 2
        halves.update(zip(range(start, start + 2 * count, 2), struct.unpack(f'<{count}H', data[:2 * count])))

    def where(address):
        function = labels.get(max((label for label in labels if label <= address), default=None), '?')
        return f'0x{address:x} in {function}'

    decoded = set()
    for address in sorted((seeds | set(labels) | {start for start, _ in code}) & halves.keys()):
        effects = _instruction_effects(halves, address)
        while effects and address not in decoded:
            decoded.add(address)
            length, refused, _, writes = effects
            following = _instruction_effects(halves, address + length)
            if refused:
                raise Refusal(f'host code executes {refused[2]} at {where(address)}; {refused[3]}')
            if following and writes & following[2]:
                raise Refusal(f'host code writes {_FP_NAMES[min(writes & following[2])]} at {where(address)} '
                              'and loads it in the next instruction; the chip core can keep the older value '
                              '(ucb-bar/shuttle issue 20)')
            address, effects = address + length, following


def _startup_check(path):
    """Require reachable platform UART setup, a trap vector safe for Shuttle, and no code its errata break."""
    objdump = shutil.which('riscv64-unknown-elf-objdump')
    if objdump is None:
        raise Refusal('RISC-V objdump required for startup qualification')
    # objdump names its input in the header; name the file alone so the digest is the code's, not the folder's.
    elf = Path(path).resolve()
    result = subprocess.run([objdump, '-d', elf.name], cwd=elf.parent, capture_output=True, text=True)
    if result.returncode:
        raise Refusal('ELF disassembly failed')
    _core_errata_check(result.stdout, _executable_code(path))
    functions = {}
    current = None
    for line in result.stdout.splitlines():
        match = re.match(r'^[0-9a-f]+ <([^>]+)>:', line)
        if match:
            current = match[1]
            functions.setdefault(current, [])
        elif current is not None:
            functions[current].append(line)
    edges = {}
    order = list(functions)
    for name, lines in functions.items():
        edges[name] = set()
        if lines and name != order[-1]:
            instructions = [line for line in lines if re.match(r'\s*[0-9a-f]+:', line)]
            if instructions and not re.search(r'\s(?:j|jr|ret|mret)\s*(?:[^a-z]|$)', instructions[-1]):
                edges[name].add(order[order.index(name) + 1])
        for line in lines:
            # Direct calls and tail jumps, including relaxed compressed forms.
            if re.search(r'\s(?:jal|jalr|j|call|tail|beq|bne|blt|bge|beqz|bnez)\s', line):
                edges[name].update(target.split('+')[0] for target in re.findall(r'<([^>]+)>', line))
    reached, pending = set(), ['_start']
    while pending:
        name = pending.pop()
        if name not in reached:
            reached.add(name)
            pending.extend(edges.get(name, ()))
    if not {'__wrap_main', 'uart_init', 'main'} <= reached:
        raise Refusal('ELF startup does not reach platform UART initialization and main')
    wrapper = '\n'.join(functions.get('__wrap_main', []))
    if not (0 <= wrapper.find('<uart_init>') < wrapper.find('<main>')):
        raise Refusal('platform wrapper must initialize UART before main')
    trap = functions.get('trap_vector')
    if not trap:
        raise Refusal('ELF trap vector missing')
    if any(re.search(r'\bcsr\w*\b.*(?:0x401|\b1025\b)', line) for line in result.stdout.splitlines()):
        raise Refusal('ELF trap vector accesses unsupported CSR 0x401')
    return dict(uart_init_reachable=True, wrapped_main_reachable=True,
                trap_vector_no_lbr_csr=True, core_errata_clear=True,
                disassembly_sha256=__import__('hashlib').sha256(result.stdout.encode()).hexdigest())


# Shorter zero runs stay loaded: a cost choice (a table row and a clear against the bytes saved), not a correctness rule.
_ZERO_SPAN_BYTES = 64


def _bound_rows(deployment):
    """The window rows the binder writes: inputs, state, outputs, and parameters."""
    return [*deployment['inputs'], *deployment.get('state_inputs', []), *deployment['outputs'],
            *([deployment['parameters']] if deployment.get('parameters') else [])]


def window_zero_spans(deployment, window):
    """(address, bytes) zero runs of the bound window image that the harness clears before it binds Atlas.

    A run lies outside every row the binder writes (inputs, state, parameters, and outputs), which a runner may also
    write before the start, so clearing it never replaces a runner's bytes.
    """
    base = deployment['window']['address']
    bound = sorted((row['address'] - base, row['address'] - base + row['bytes']) for row in _bound_rows(deployment))
    spans = []
    for run in re.finditer(rb'\0{%d,}' % _ZERO_SPAN_BYTES, window):
        cursor, end = run.span()
        for low, high in [*bound, (end, end)]:
            if high <= cursor:
                continue
            if min(low, end) - cursor >= _ZERO_SPAN_BYTES:
                spans.append((base + cursor, min(low, end) - cursor))
            cursor = max(cursor, high)
            if cursor >= end:
                break
    return spans


def _quoted(path):
    value = str(safe_path(path))
    if ']==]' in value or '\n' in value or '\r' in value:
        raise Refusal('unsupported platform build path')
    return '[==[' + value + ']==]'


def _object_paths(build):
    """Read CMake's explicit command inventory without traversing dependencies."""
    commands = json.loads(safe_path(build / 'compile_commands.json').read_text())
    objects, sources = set(), set()
    for row in commands:
        args = row.get('arguments') or shlex.split(row['command'])
        if '-o' not in args:
            raise Refusal('platform compile command lacks object output')
        path = Path(args[args.index('-o') + 1])
        path = safe_path(Path(row['directory']) / path)
        if not path.is_relative_to(build):
            raise Refusal('platform object outside managed build')
        if path.is_file():
            objects.add(path)
            sources.add(safe_path(row['file']))
    if not objects:
        raise Refusal('platform build produced no measured objects')
    return sorted(objects), sorted(sources)


def objects(context):
    """Compile application, checker, startup and drivers without an ELF link."""
    from atlas_mlir_native.elf_build.sections import measure
    from atlas_mlir_native.c_runtime import IMAGE_SECTION
    from atlas_mlir_native.provider_deployment import PARAMETER_SECTION, WINDOW_SECTION
    from .deployment_gate import prepare_binding

    root_value = os.environ.get('ATLAS_BAREMETAL_IDE')
    if not root_value:
        raise Refusal('ATLAS_BAREMETAL_IDE must select the platform checkout')
    platform = safe_path(root_value)
    cmake = shutil.which('cmake')
    if cmake is None or shutil.which('riscv64-unknown-elf-gcc') is None:
        raise Refusal('CMake and RISC-V GCC must be available on PATH')
    application = safe_path(context.get('application') or context['out'] / 'compiled' / 'application')
    prepared = context.get('binding') or prepare_binding(context)
    deployment = prepared['platform_deployment']
    if deployment.get('paging'):
        raise Refusal('paged onchip ELF preparation is unqualified')
    interface = json.loads(safe_path(application / 'handoff' / 'abi.json').read_text())
    c_interface = json.loads(safe_path(application / 'application-c.json').read_text())
    # Host math is linked only from an archive selected by its declared hash;
    # its members are bounded by the final ELF record, not by object sizing.
    candidates = {digest(path): safe_path(path) for path in context.get('host_archives', ())}
    host_archives = []
    for contract in c_interface.get('host_linkage', []):
        if contract.get('archive_sha256') not in candidates:
            raise Refusal('declared host math archive is not supplied: ' + str(contract.get('archive_sha256')))
        host_archives.append(candidates[contract['archive_sha256']])
    root = safe_path(context['out'] / 'objects')
    root.mkdir(exist_ok=False)
    generated = root / 'generated'
    generated.mkdir()
    hashes = {}
    # Generated C is already composed by the application compiler.
    for member in (interface['source'], interface['header'], 'deployment_adapter.h'):
        if Path(member).name != member:
            raise Refusal('handoff source must be a direct member')
        source = safe_path(application / 'handoff' / member)
        (generated / member).write_bytes(source.read_bytes())
        hashes[str(source)] = digest(source)
    for member, data in prepared['generated'].items():
        path = Path(member)
        if len(path.parts) != 2 or path.parts[0] != 'handoff' or path.name in (
                interface['source'], interface['header'], 'deployment_adapter.h'):
            raise Refusal('unexpected platform generated member')
        (generated / path.name).write_bytes(data)
    for required in ('chip.ld', 'window.S', 'window.bin', 'harness_sha256.h'):
        if not (generated / required).is_file():
            raise Refusal('platform binding omitted ' + required)
    zero = window_zero_spans(deployment, (generated / 'window.bin').read_bytes())
    (generated / 'harness.c').write_text(_HARNESS.replace('ATLAS_WINDOW_ZERO', ''.join('{%#x,%d},' % span for span in zero)))
    sources = [generated / interface['source'], generated / 'harness.c', generated / 'window.S']
    if all(call['placement'] == 'host' for call in deployment['calls']):
        # No device region embeds the Atlas host runtime that the harness binds; link it once.
        runtime = Path(__file__).resolve().parents[2] / 'runtime'
        for name in ('atlas_host.c', 'atlas_host.h', 'atlas_host_layout.h'):
            (generated / name).write_bytes(safe_path(runtime / name).read_bytes())
        sources.append(generated / 'atlas_host.c')
    final = root / 'target-final.cmake'
    final.write_text('''get_directory_property(deployment_link_options LINK_OPTIONS)
list(REMOVE_ITEM deployment_link_options -T "${LINKER_SCRIPT}")
set_directory_properties(PROPERTIES LINK_OPTIONS "${deployment_link_options}")
add_library(atlas_deployment_objects OBJECT
''' + '\n'.join(_quoted(p) for p in sources) + ''')
target_include_directories(atlas_deployment_objects PRIVATE ''' + _quoted(generated) + ''')
target_compile_options(atlas_deployment_objects PRIVATE -Os -ffunction-sections -fdata-sections -fstack-usage -fcallgraph-info=su
  -fno-fast-math -ffp-contract=off -frounding-math -DATLAS_IMAGE_SECTIONS)
target_compile_options(atlas_deployment_objects PRIVATE "$<$<COMPILE_LANGUAGE:ASM>:-Wa,-I,'''
                     + str(generated).replace('\\', '/').replace('"', '\\"') + '''>")
target_link_libraries(atlas_deployment_objects PRIVATE glossy)
add_custom_target(atlas_deployment_sizing DEPENDS atlas_deployment_objects glossy)
add_executable(atlas_deployment $<TARGET_OBJECTS:atlas_deployment_objects>)
target_link_libraries(atlas_deployment PRIVATE -Wl,--whole-archive glossy -Wl,--no-whole-archive '''
                     + ''.join(_quoted(path) + ' ' for path in host_archives) + '''m)
target_link_options(atlas_deployment PRIVATE "-T''' + str(generated / 'chip.ld').replace('\\', '/').replace('"', '\\"')
                     + '''" -Wl,--gc-sections "-Wl,-Map,'''
                     + str(root / 'chip.map').replace('\\', '/').replace('"', '\\"') + '''")
''')
    entry = root / 'target.cmake'
    entry.write_text(_DETERMINISTIC_ARCHIVES + 'cmake_language(DEFER CALL include ' + _quoted(final) + ')\n')
    settings = context['settings']
    flags = f"-DSYS_CLK_FREQ={settings['clock_hz']} -DMTIME_FREQ={settings['mtime_hz']} -DGLOSSY_NO_LBR_CSR"
    build = root / 'build'
    env = {**os.environ, 'OMP_NUM_THREADS': '1'}
    # A board DDR program ends the way the declared board host learns it ended; the JTAG bench reads the UART.
    from atlas_mlir_native import soc
    tohost = context['transport'] != 'jtag' and soc.BOARD_PROGRAM['completion'] == 'tohost'

    configure = [cmake, '-S', platform, '-B', build, '-G', 'Unix Makefiles',
                 '-DCMAKE_TOOLCHAIN_FILE=' + str(platform / 'riscv-gcc.cmake'),
                 '-DCMAKE_PROJECT_INCLUDE=' + str(entry), '-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY',
                 '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DCHIP=penguin26', '-DPLATFORM=CHIP',
                 '-DRVV_TYPE=0', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_FLAGS=' + flags,
                 '-DCMAKE_ASM_FLAGS=' + flags, '-DPENGUIN26_EXIT_TOHOST=' + ('ON' if tohost else 'OFF'),
                 f"-DPENGUIN26_UART_BAUDRATE={settings['receiver_baud']}"]
    _run(configure, root / 'configure.log', env=env)
    _configure_check(root / 'configure.log')
    _run([cmake, '--build', build, '--target', 'atlas_deployment_sizing', '-j2'], root / 'objects.log', env=env)
    paths, runtime_sources = _object_paths(build)
    measurements = [measure(path) for path in paths]
    link_command = build / 'CMakeFiles' / 'atlas_deployment.dir' / 'link.txt'
    archives = []
    for argument in shlex.split(link_command.read_text()):
        if argument.endswith('.a') and safe_path(build / argument) not in host_archives:
            archive = safe_path(build / argument)
            if not archive.is_relative_to(build):
                raise Refusal('platform archive outside managed build')
            archives.append(archive)
    for path in [*paths, *runtime_sources, *archives, *host_archives, entry, final, link_command,
                 build / 'compile_commands.json', build / 'CMakeCache.txt',
                 platform / 'riscv-gcc.cmake',
                 platform / 'platform' / 'penguin26' / 'chip_config.h',
                 *sorted(generated.iterdir())]:
        hashes[str(path)] = digest(path)
    sizes = {name: sum(section['size'] for row in measurements for section in row['sections']
                       if section['name'] == name) for name in (WINDOW_SECTION, PARAMETER_SECTION, IMAGE_SECTION)}
    window_bytes, image_bytes = sizes[WINDOW_SECTION] + sizes[PARAMETER_SECTION], sizes[IMAGE_SECTION]
    if window_bytes != len((generated / 'window.bin').read_bytes()):
        raise Refusal('window object differs from single bound backing')
    if image_bytes > deployment.get('window_image_reserve_bytes', 0):
        raise Refusal('instruction image objects exceed the placed window image reservation')
    if -(-window_bytes // 16) * 16 + image_bytes > deployment['window']['bytes']:
        raise Refusal('bound window plus instruction images exceed the deployment window')
    linked = _sizing_link(build, link_command, root, env)
    readelf = shutil.which('riscv64-unknown-elf-readelf')
    if readelf is None:
        raise Refusal('RISC-V readelf must be available on PATH to read generated stack usage')
    usage = generated_stack([path for path in paths if Path(path).name.startswith(tuple(
        p.name for p in generated.iterdir() if p.suffix == '.c'))], readelf)
    _, symbols = _elf_symbols(safe_path(root / 'sizing.elf').read_bytes())
    usage['heap_allocator_linked'] = any(name in symbols for name in _ALLOCATOR)
    declared = {row['name']: row['bytes'] for row in deployment['host_allocation']['declared_reservations']}
    if usage['generated_stack_bytes'] > declared['stack']:
        raise Refusal(f"generated code may use {usage['generated_stack_bytes']} stack bytes; {declared['stack']} declared")
    if usage['heap_allocator_linked'] and not declared['heap']:
        raise Refusal('linked code includes an allocator but no heap is declared')
    record = dict(measurements=measurements, hashes=hashes, host_usage=usage,
                  host_linked=dict(linked, bytes=linked['__stack_end'] - deployment['host_allocation']['address']),
                  window_allocated_bytes=window_bytes, window_image_bytes=image_bytes, settings=settings,
                  commands=[list(map(str, configure))], linked=False)
    receipt = root / 'sections.json'
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    hashes[str(receipt)] = digest(receipt)
    context['objects'] = {**record, 'root': root, 'generated': generated,
                          'build': build, 'receipt': receipt}
    return context['objects']


HOST_ALIGNMENT = 4096


def measure_host(context, arguments):
    """Measure the host that a board DDR deployment places first, and return the compile that binds it.

    The measuring compile starts the device window at the declared DDR base, with the host after its data. Its sizing
    link gives the host's code, data, heap, and stack; the bound compile starts the window at the next 4 KiB boundary
    after that host and gives it the rest of the DDR (a resident arena, when selected, takes the end of it), and the
    bound objects must still fit that host.
    """
    from atlas_mlir_native.soc import SOC
    ddr = SOC.memory_region('ddr')
    out = context['out']
    record = objects(context)
    host = record['host_linked']['bytes']
    reserved = -(-host // HOST_ALIGNMENT) * HOST_ALIGNMENT
    if not 0 < reserved < ddr.capacity_bytes:
        raise Refusal(f'measured host exceeds the declared DDR: {host} bytes')
    measured = out / 'measure'
    measured.mkdir()
    for name in ('compiled', 'objects'):
        (out / name).rename(measured / name)
    del context['objects']
    (measured / 'host.json').write_text(json.dumps(dict(host_linked=record['host_linked'], reserved_bytes=reserved,
                                                         window_base=ddr.base + reserved), indent=2) + '\n')
    arguments = [str(value) for value in arguments]
    if '--io-memory-bytes' in arguments:
        raise Refusal('the measuring compile takes the whole declared DDR')
    return [*arguments, '--io-memory-bytes', str(ddr.capacity_bytes - reserved)]


def _elf_symbols(raw):
    """Section headers and every symbol's (value, size, section index)."""
    shoff, = struct.unpack_from('<Q', raw, 40)
    shsize, shcount = struct.unpack_from('<2H', raw, 58)
    sections = [struct.unpack_from('<IIQQQQIIQQ', raw, shoff + i * shsize) for i in range(shcount)]
    symbols = {}
    for sec in sections:
        if sec[1] != 2:
            continue
        if sec[9] != 24 or sec[6] >= len(sections):
            raise Refusal('invalid ELF symbols')
        names = sections[sec[6]]
        strings = raw[names[4]:names[4] + names[5]]
        for offset in range(sec[4], sec[4] + sec[5], 24):
            name, _, _, index, value, size = struct.unpack_from('<IBBHQQ', raw, offset)
            name = strings[name:].split(b'\0')[0].decode()
            # Image symbols are checked one by one below, so each must be unique.
            if name in symbols and name.startswith('atlas_image_'):
                raise Refusal('duplicate instruction image symbol ' + name)
            symbols[name] = (value, size, index)
    return sections, symbols


_ALLOCATOR = ('malloc', '_malloc_r', 'calloc', 'realloc', 'sbrk', '_sbrk', '_sbrk_r')
# Relocations that only call or jump; any other relocation against a function takes its address.
_CALL_RELOCATIONS = ('R_RISCV_CALL', 'R_RISCV_CALL_PLT', 'R_RISCV_JAL', 'R_RISCV_BRANCH', 'R_RISCV_RVC_JUMP',
                     'R_RISCV_RVC_BRANCH', 'R_RISCV_RELAX', 'R_RISCV_ALIGN')


# Standard C functions whose specification takes no callable argument and calls no user code.
_NO_CALLBACK = ('memcpy', 'memmove', 'memset', 'memcmp', 'strlen')
_CI_LINE = re.compile(r'graph: \{ title: "[^"]*"|\}|node: \{ title: "__indirect_call" label: "Indirect Call Placeholder" shape : ellipse \}'
                      r'|node: \{ title: "(?P<node>[^"]+)" label: "(?P<label>[^"]*)"(?P<shape> shape : ellipse)? \}'
                      r'|edge: \{ sourcename: "(?P<source>[^"]+)" targetname: "(?P<target>[^"]+)"(?: label: "[^"]*")? \}')


def _section_headers(raw):
    """(name, type, flags, size) of every section header of a 64-bit little-endian ELF object."""
    if raw[:4] != b'\x7fELF' or raw[4] != 2 or raw[5] != 1:
        raise Refusal('generated object is not a 64-bit little-endian ELF file')
    shoff, = struct.unpack_from('<Q', raw, 40)
    shsize, shcount, shstrndx = struct.unpack_from('<3H', raw, 58)
    if shsize != 64 or shoff + shsize * shcount > len(raw) or shstrndx >= shcount:
        raise Refusal('generated object has an invalid section header table')
    rows = [struct.unpack_from('<IIQQQQIIQQ', raw, shoff + i * shsize) for i in range(shcount)]
    names = rows[shstrndx]
    strings = raw[names[4]:names[4] + names[5]]
    return [(strings[row[0]:].split(b'\0')[0].decode(errors='replace'), row[1], row[2], row[5]) for row in rows]


def generated_stack(objects, readelf):
    """Sum of generated C frames along the worst path of a call model built from GCC's own
    -fstack-usage/-fcallgraph-info=su data.

    Refused: a missing .ci or .su file or any line not in the exact format GCC writes; a function
    symbol (global, weak, or local) not at the address of a function with static usage; dynamic usage
    (alloca, variable-length arrays); a call or taken address that resolves to code without a parsed
    frame; any executable byte not inside a function symbol with a parsed frame (section headers are
    read from the ELF itself; a symbol or relocation row that cannot be read is refused); any cycle in the
    model; and a symbol defined in two generated objects. Call targets and
    taken addresses are resolved through object and global symbol addresses, so aliases and weak
    definitions reach their functions. Model edges: each direct call; from an indirect call, every
    generated function whose address the generated objects take and every library symbol they take;
    and from any call into library code, every taken generated function, except calls to memcpy,
    memmove, memset, memcmp, and strlen. Assumed, not checked: no arithmetic on function pointers;
    library and platform code reaches generated code only through those taken addresses, apart from
    startup's direct call to main; and inline asm in generated C neither writes sp nor transfers
    control (the generated asm today is fence and rdcycle). The sum covers generated frames only; library and platform frames
    (newlib, libgcc helpers, _write, startup, trap handler, drivers) are measured, not bounded here.
    """
    frames, raw_edges, external, located, references, defined_globals = {}, [], set(), {}, [], {}
    for obj in objects:
        stem = str(obj).rsplit('.', 1)[0]
        graph, usage_file = Path(stem + '.ci'), Path(stem + '.su')
        if not graph.is_file() or not usage_file.is_file():
            raise Refusal('generated object lacks its GCC call graph or stack usage: ' + Path(obj).name)
        # A static function's title (FILE:NAME) is made unique per object.
        key = lambda title: f'{obj}|{title}' if ':' in title else title
        nodes, su = {}, []
        for line in safe_path(graph).read_text().splitlines():
            match = _CI_LINE.fullmatch(line)
            if match is None:
                raise Refusal('malformed GCC call graph line in ' + graph.name)
            if match.group('node'):
                title, label = key(match.group('node')), match.group('label')
                usage = re.fullmatch(r'([^\\]+)\\n[^\\]+(?:\\n(\d+) bytes \(([a-z,]+)\))?', label)
                if usage is None or bool(usage.group(2)) == bool(match.group('shape')):
                    raise Refusal('malformed GCC call graph node in ' + graph.name)
                if not usage.group(2):
                    continue  # a callee without usage; resolved by symbol below
                if usage.group(3) != 'static':
                    raise Refusal(f'generated function {usage.group(1)} has {usage.group(3)} stack usage')
                if title in frames:
                    raise Refusal('generated function is defined twice: ' + title)
                frames[title] = int(usage.group(2))
                nodes[title.rsplit(':', 1)[-1]] = title
                su.append((usage.group(1), int(usage.group(2))))
            elif match.group('source'):
                raw_edges.append((obj, key(match.group('source')), match.group('target')))
        usage_rows = []
        for line in safe_path(usage_file).read_text().splitlines():
            fields = line.split('\t')
            if len(fields) != 3 or not fields[1].isdigit():
                raise Refusal('malformed GCC stack usage line in ' + usage_file.name)
            if fields[2] != 'static':
                raise Refusal(f'generated function {fields[0].rsplit(":", 1)[-1]} has {fields[2]} stack usage')
            usage_rows.append((fields[0].rsplit(':', 1)[-1], int(fields[1])))
        if sorted(usage_rows) != sorted(su):
            raise Refusal('GCC call graph and stack usage file disagree for ' + Path(obj).name)
        # Every function symbol must sit at the address of a function with parsed static usage.
        symbols, at, extents = {}, {}, {}
        for line in subprocess.run([readelf, '-sW', str(obj)], check=True, capture_output=True, text=True).stdout.splitlines():
            fields = line.split()
            if not fields or not fields[0].rstrip(':').isdigit():
                continue
            if len(fields) > 8:
                raise Refusal(f'a symbol name in {Path(obj).name} contains whitespace and cannot be read reliably')
            if len(fields) < 8:
                continue  # the unnamed null symbol
            value, kind, bind, index, name = int(fields[1], 16), fields[3], fields[4], fields[6], fields[7]
            symbols[name] = (kind, index, value)
            if kind == 'FUNC' and index != 'UND':
                extents.setdefault(index, []).append((value, int(fields[2], 0), name))
            if index != 'UND' and bind in ('GLOBAL', 'WEAK'):
                if name in defined_globals:
                    raise Refusal('generated symbol is defined in two objects: ' + name)
                defined_globals[name] = (obj, kind, index, value)
            if kind == 'FUNC' and index != 'UND' and name in nodes:
                at[(index, value)] = nodes[name]
        for name, (kind, index, value) in symbols.items():
            if kind == 'FUNC' and index != 'UND' and (index, value) not in at:
                raise Refusal(f'generated function {name} has no parsed static stack usage')
            if kind == 'FUNC' and index != 'UND':
                located[(obj, name)] = at[(index, value)]
        # Every executable byte must lie inside a function symbol with a parsed frame, so code with no
        # frame (untyped assembly labels, local labels) can never be reached unaccounted.
        # Section headers come from the ELF itself, so a section name of any spelling is read.
        for position, header in enumerate(_section_headers(safe_path(obj).read_bytes())):
            name, kind, flags, size = header
            if not flags & 0x4 or kind == 8 or not size:  # SHF_EXECINSTR; SHT_NOBITS holds no bytes
                continue
            index, covered = str(position), 0
            for start, length, symbol in sorted(extents.get(index, [])):
                if start > covered or (index, start) not in at:
                    raise Refusal(f'executable bytes in {name} of {Path(obj).name} lie outside every '
                                  'generated function with a parsed frame')
                covered = max(covered, start + length)
            if covered < size:
                raise Refusal(f'executable bytes in {name} of {Path(obj).name} lie outside every '
                              'generated function with a parsed frame')
        sections = {name: index for name, (kind, index, value) in symbols.items() if kind == 'SECTION'}
        for line in subprocess.run([readelf, '-rW', str(obj)], check=True, capture_output=True, text=True).stdout.splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[2].startswith('R_RISCV_') and len(fields) not in (4, 7):
                raise Refusal(f'a relocation in {Path(obj).name} names a symbol that cannot be read reliably')
            if len(fields) < 5 or not fields[2].startswith('R_RISCV_') or fields[2] in _CALL_RELOCATIONS \
                    or fields[2].startswith('R_RISCV_PCREL_LO12'):
                continue
            addend = re.search(r'([+-]) ([0-9a-f]+)$', line)
            references.append((obj, fields[4], (int(addend.group(2), 16) * (-1 if addend.group(1) == '-' else 1)) if addend else 0,
                               symbols, sections, at))
    # Resolve every call target by symbol address, so a call to an alias reaches its function; a defined
    # target without a parsed frame is refused, and an undefined one is library code.
    edges = {}
    for obj, source, target in raw_edges:
        if target == '__indirect_call':
            resolved = target
        elif ':' in target:
            # A file-local callee that GCC folded into an identical function (IPA ICF) has no node of its own;
            # its symbol sits at the folded function's address, so it resolves by symbol like a global.
            resolved = f'{obj}|{target}'
            if resolved not in frames:
                if (obj, target.rsplit(':', 1)[-1]) not in located:
                    raise Refusal('generated call target has no parsed frame: ' + target.rsplit(':', 1)[-1])
                resolved = located[(obj, target.rsplit(':', 1)[-1])]
        elif (obj, target) in located:
            resolved = located[(obj, target)]
        elif target in defined_globals:
            if defined_globals[target][1] != 'FUNC' or (defined_globals[target][0], target) not in located:
                raise Refusal('generated call target has no parsed frame: ' + target)
            resolved = located[(defined_globals[target][0], target)]
        else:
            resolved = target
            external.add(target)
        edges.setdefault(source, set()).add(resolved)
    # Resolve every non-call reference: a code address must be a function with static usage; an address
    # of an undefined (library) symbol is a possible indirect target in library code.
    taken, library_taken = set(), set()
    for obj, name, addend, symbols, sections, at in references:
        if name in sections:
            if name.startswith('.text'):
                if (sections[name], addend) not in at:
                    raise Refusal(f'a code address other than a function start is taken in {Path(obj).name}')
                taken.add(at[(sections[name], addend)])
            continue
        kind, index, value = symbols.get(name, ('NOTYPE', 'UND', 0))
        if index == 'UND':
            if name in defined_globals:
                target, kind = defined_globals[name][0], defined_globals[name][1]
                if kind == 'FUNC':
                    taken.add(located[(target, name)])
            else:
                library_taken.add(name)
        elif kind == 'FUNC':
            taken.add(located[(obj, name)])
    external -= set(frames)
    # Library code may call back any generated function whose address is taken, unless the call is to a
    # standard C function that takes no callable argument. Indirect calls may also enter library code.
    calls_library = {name for name in frames if (edges.get(name, set()) - set(frames) - {'__indirect_call'} - set(_NO_CALLBACK))
                     or ('__indirect_call' in edges.get(name, ()) and library_taken - set(_NO_CALLBACK))}
    graph = {name: {t for t in edges.get(name, ()) if t in frames}
             | (taken if '__indirect_call' in edges.get(name, ()) or name in calls_library else set()) for name in frames}
    unresolved = sorted(name for name in frames if '__indirect_call' in edges.get(name, ()) and not (taken or library_taken))
    if unresolved:
        raise Refusal('generated indirect calls have no resolvable target: ' + ', '.join(unresolved))
    bound, active = {}, []

    def deepest(name):
        if name in active:
            raise Refusal('generated call cycle through ' + ' -> '.join(t.rsplit(':', 1)[-1] for t in active[active.index(name):] + [name]))
        if name not in bound:
            active.append(name)
            bound[name] = frames[name] + max((deepest(t) for t in graph[name]), default=0)
            active.pop()
        return bound[name]

    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * len(frames) + 100))
    worst = max(frames, key=deepest, default=None)
    path = [worst] if worst else []
    while path and graph[path[-1]]:
        path.append(max(sorted(graph[path[-1]]), key=deepest))  # ties: the first callee by name
    show = lambda titles: [title.rsplit('|', 1)[-1].rsplit('/', 1)[-1] for title in titles]
    return dict(generated_stack_bytes=bound.get(worst, 0), generated_worst_path=show(path),
                indirect_targets=sorted(show(taken)) + sorted(library_taken), library_callees=sorted(external | library_taken),
                library_stack='generated frames only; library and platform frames are measured, not bounded')


def measure_onchip(context, arguments):
    """Link an on-chip plan's host once, and return the compile that stages it when it exceeds the declared host memory.

    Returns None for a plan that is staged already, whose steps each size their own host, and for a host that fits,
    whose measured objects then serve the objects stage.
    """
    if staged(context):
        return None
    out = context['out']
    record = objects(context)
    deployment = json.loads(safe_path(out / 'compiled/application/deployment.json').read_text())
    host, usable = record['host_linked']['bytes'], deployment['host_allocation']['usable_bytes']
    if host <= usable:
        return None
    arguments = [str(value) for value in arguments]
    if '--measured-host-bytes' in arguments:
        raise Refusal('the on-chip host is measured once')
    measured = out / 'measure'
    measured.mkdir()
    for name in ('compiled', 'objects'):
        (out / name).rename(measured / name)
    del context['objects']
    (measured / 'host.json').write_text(json.dumps(dict(host_linked=record['host_linked'], usable_bytes=usable),
                                                   indent=2) + '\n')
    return [*arguments, '--measured-host-bytes', str(host)]


def _sizing_link(build, link_command, root, env):
    """Link the measured objects once to measure exact placement; the ELF is never shipped.

    Uses the deployment link command unchanged except for its outputs and
    --noinhibit-exec, so an over-budget layout still reports its extent.
    """
    arguments = shlex.split(link_command.read_text())
    if arguments.count('-o') != 1 or sum(a.startswith('-Wl,-Map,') for a in arguments) != 1:
        raise Refusal('platform link command lacks one output and one map')
    elf = root / 'sizing.elf'
    arguments[arguments.index('-o') + 1] = str(elf)
    arguments = ['-Wl,-Map,' + str(root / 'sizing.map') if a.startswith('-Wl,-Map,') else a for a in arguments]
    with safe_path(root / 'sizing.log').open('w') as stream:
        subprocess.run([*arguments, '-Wl,--noinhibit-exec'], cwd=build, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if not elf.is_file():
        raise Refusal('sizing link failed; see ' + str(root / 'sizing.log'))
    _, symbols = _elf_symbols(elf.read_bytes())
    try:
        return {name: symbols[name][0] for name in ('__bss_end', '__heap_start', '__stack_end')}
    except KeyError as error:
        raise Refusal('sizing link lacks host reservation symbols') from error


def _elf_record(path, deployment, settings):
    """Validate real load spans and the linked clock/stack symbols."""
    from atlas_mlir_native.elf_build.sections import measure
    measurement = measure(path)
    if measurement['elf_type'] != 2:
        raise Refusal('final platform output is not executable')
    raw = safe_path(path).read_bytes()
    entry, phoff, shoff = struct.unpack_from('<QQQ', raw, 24)
    phsize, phcount, shsize, shcount, _ = struct.unpack_from('<5H', raw, 54)
    if phsize != 56 or phoff + phsize * phcount > len(raw):
        raise Refusal('invalid ELF load table')
    host = deployment['host_allocation']
    window = deployment['window']
    bounds = [(host['address'], host['address'] + host['usable_bytes']),
              (window['address'], window['address'] + window['bytes'])]
    resident = deployment.get('resident')
    if resident:
        bounds.append((resident['address'], resident['address'] + resident['bytes']))
    if entry != bounds[0][0]:
        raise Refusal('ELF entry differs from bound host origin')
    loads, offsets = [], []
    for i in range(phcount):
        kind, flags, offset, address, physical, size, memory, _ = struct.unpack_from('<II6Q', raw, phoff + i * phsize)
        if kind != 1:
            continue
        if (physical != address or size > memory or offset + size > len(raw)
                or not any(start <= address and address + memory <= end for start, end in bounds)):
            raise Refusal('ELF load exceeds bound physical memory')
        if any(address < row['address'] + row['memory_bytes'] and row['address'] < address + memory for row in loads):
            raise Refusal('ELF load spans overlap')
        loads.append(dict(address=address, file_bytes=size, memory_bytes=memory, flags=flags))
        offsets.append(offset)
    # Read-only parameters end the window data in a third load, with the instruction images after them.
    parameters = read_only_parameters(deployment)
    starts = {start for start, _ in bounds} | ({parameters['address']} if parameters else set())
    read_only = [row['address'] for row in (parameters, resident) if row]
    if len(loads) != len(starts) or {row['address'] for row in loads} != starts or any(next(
            row['flags'] for row in loads if row['address'] == address) != 4 for address in read_only):
        raise Refusal('ELF must retain exactly one host and one window backing, '
                      'and read-only parameters and resident arena if declared')
    if resident:
        # Every resident load is linked byte-equal to the packed asset the compiler published for it.
        index = next(i for i, row in enumerate(loads) if row['address'] == resident['address'])
        if loads[index]['file_bytes'] != resident['bytes']:
            raise Refusal('resident arena load differs from the deployment arena')
        for row in resident['loads']:
            start = offsets[index] + row['address'] - resident['address']
            if hashlib.sha256(raw[start:start + row['bytes']]).hexdigest() != row['sha256']:
                raise Refusal('linked resident load differs from its packed asset: ' + row['name'])
    for section in measurement['sections']:
        if section['size'] and not any(
                row['address'] <= section['address']
                and section['address'] + section['size'] <= row['address'] + row['memory_bytes']
                for row in loads):
            raise Refusal('allocated ELF section lies outside load backing')
    sections, symbols = _elf_symbols(raw)
    try:
        stack = symbols['__sp'][0]
        program_end = max((row['address'] + row['size'] for row in measurement['sections']
                           if row['name'] not in ('.heap', '.stack')
                           and bounds[0][0] <= row['address'] < bounds[0][1]), default=bounds[0][0])
        if not (symbols['__heap_end'][0] <= symbols['__stack_start'][0] < stack <= bounds[0][1]
                and bounds[0][0] <= program_end <= symbols['__heap_start'][0] <= symbols['__heap_end'][0]
                and symbols['__stack_end'][0] == stack and stack % 16 == 0
                and symbols['_start'][0] == entry):
            raise Refusal('ELF heap/stack lie outside backed host memory')
        for row in loads:
            if (row['address'] == bounds[0][0]
                    and row['address'] + row['memory_bytes'] > symbols['__heap_start'][0]):
                raise Refusal('ELF load zero fill includes symbolic heap/stack reservations')
        if any(row['name'] in ('.heap', '.stack') and row['size']
               for row in measurement['sections']):
            raise Refusal('ELF heap/stack must be symbolic reservations')
        value, size, index = symbols['sys_clk_freq']
        section = sections[index]
        if not 0 < size <= 8 or section[1] == 8 or not section[3] <= value < section[3] + section[5]:
            raise Refusal('ELF platform clock lacks initialized storage')
        offset = section[4] + value - section[3]
        clock = int.from_bytes(raw[offset:offset + size], 'little')
    except (KeyError, IndexError) as error:
        raise Refusal('ELF lacks platform clock or stack symbols') from error
    if clock != settings['clock_hz']:
        raise Refusal('linked platform clock differs from bound settings')
    # Every instruction image lies in the window load after the bound window
    # bytes, and its linked words hash to the compiled program it names.
    images = {}
    image_floor = window['address'] + deployment['window_required_bytes']
    for name, (value, size, _) in symbols.items():
        if not re.fullmatch(r'atlas_image_[0-9a-f]{64}', name):
            continue
        index = next((i for i, row in enumerate(loads) if row['address'] <= value
                      and value + size <= row['address'] + row['file_bytes']), None)
        if index is None or loads[index]['address'] != (parameters or window)['address'] or value < image_floor:
            raise Refusal('instruction image is not loaded with the window after bound bytes')
        start = offsets[index] + value - loads[index]['address']
        if hashlib.sha256(raw[start:start + size]).hexdigest() != name[12:]:
            raise Refusal('linked instruction image differs from its compiled program')
        images[name[12:]] = dict(address=value, bytes=size)
    if set(images) != {row['sha256'] for row in deployment['provider']['images']
                       if row.get('upload_convention') != 'resident_template_bind'}:
        raise Refusal('linked instruction images differ from the deployment image set')
    from .chiprunner.composed_sim import cleared_spans
    try:
        zero = cleared_spans(raw)
    except ValueError as error:
        raise Refusal('ELF window zero table is invalid: ' + str(error)) from error
    rows = _bound_rows(deployment)
    if any(not bounds[1][0] <= address < address + size <= bounds[1][1]
           or any(row['address'] < address + size and address < row['address'] + row['bytes'] for row in rows)
           for address, size in zero):
        raise Refusal('ELF window zero spans must lie in the window outside every bound row')
    return dict(sha256=measurement['sha256'], entry=entry, loads=loads,
                allocated_sections=measurement['sections'], clock_hz=clock,
                receiver_baud=settings['receiver_baud'],
                startup_qualification=_startup_check(path),
                host_reservations={name: symbols[name][0] for name in (
                    '__heap_start', '__heap_end', '__stack_start', '__stack_end', '__sp')},
                host_bytes=next(row['memory_bytes'] for row in loads if row['address'] == bounds[0][0]),
                **({'resident_loads': len(resident['loads'])} if resident else {}),
                instruction_images=dict(sorted(images.items())),
                window_zero=[dict(address=address, bytes=size) for address, size in zero])


def read_only_parameters(deployment):
    """The bound parameter row when the provider loads it read-only after the writable window data, else None."""
    return deployment['parameters'] if deployment['provider'].get('parameter_section') else None


def link(context):
    """Link the measured objects after checking the gate at the actual boundary."""
    gate = context.get('gate')
    if gate is None:
        raise Refusal('bound EXACT gate is required before ELF linking')
    gate.verify()
    objects = context.get('objects')
    if not objects:
        raise Refusal('measured platform objects are required before ELF linking')
    check_hashes(objects['hashes'])
    if any(gate.hashes.get(path) != expected for path, expected in objects['hashes'].items()):
        raise Refusal('EXACT gate does not authenticate measured platform objects')
    bundle = safe_path(context['bundle'])
    for name in ('chip.ld', 'window.S', 'window.bin', 'harness_sha256.h'):
        if digest(bundle / 'handoff' / name) != digest(objects['generated'] / name):
            raise Refusal('bound platform payload drift before ELF linking: ' + name)
    linker = safe_path(bundle / 'handoff/chip.ld').read_text()
    if '.heap (NOLOAD)' in linker or '.stack (NOLOAD)' in linker:
        raise Refusal('generated linker reserves heap/stack as load extents; symbolic reservations required')
    cmake = shutil.which('cmake')
    if cmake is None:
        raise Refusal('CMake must be available on PATH')
    gate.verify()
    _run([cmake, '--build', objects['build'], '--target', 'atlas_deployment', '-j2'],
         objects['root'] / 'link.log', env={**os.environ, 'OMP_NUM_THREADS': '1'})
    check_hashes(objects['hashes'])
    gate.verify()
    elf = safe_path(objects['build'] / 'atlas_deployment.elf')
    deployment = json.loads(safe_path(bundle / 'application' / 'deployment.json').read_text())
    record = _elf_record(elf, deployment, context['settings'])
    expected = window_zero_spans(deployment, safe_path(bundle / 'handoff/window.bin').read_bytes())
    if [(row['address'], row['bytes']) for row in record['window_zero']] != expected:
        raise Refusal('ELF window zero table differs from the bound window image')
    if record['host_reservations']['__stack_end'] != objects['host_linked']['__stack_end']:
        raise Refusal('final link placement differs from the measured sizing link')
    record['gate_receipt_sha256'] = digest(gate.receipt)
    record['object_receipt_sha256'] = digest(objects['receipt'])
    (context['out'] / 'link.json').write_text(json.dumps(record, indent=2) + '\n')
    context['elf'], context['elf_record'] = elf, record
    gate.verify()
    return elf


def host_fit(objects, deployment, window):
    """The sizing link's placement of the measured objects, heap, and stack must fit the deployment's own host window."""
    host = deployment['host_allocation']
    if host.get('measurement') != 'measured_sections':
        raise Refusal('on-chip binding requires measured host sections')
    # The sizing link places the measured objects, heap, and stack exactly.
    required = objects.get('host_linked', {}).get('bytes', host['usable_bytes'] + 1)
    if required > host['usable_bytes']:
        raise Refusal(f"linked host sections plus stack/heap exceed backed host capacity "
                      f"({required} > {host['usable_bytes']} bytes)")
    if objects.get('window_allocated_bytes') != len(window):
        raise Refusal('measured window bytes differ from bound invocation')
    return dict(host_linked_bytes=required, usable_bytes=host['usable_bytes'])


def staged(context):
    """The compiled application is a staged plan of step packages (bench host holds every tensor)."""
    path = context['out'] / 'compiled' / 'application' / 'deployment.json'
    return path.is_file() and json.loads(safe_path(path).read_text()).get('mode') == 'bench_staged'


def step_binding(step, settings):
    """Bind one step package for its ELF: the packaging clock and zero bytes in every input range.

    The composed runner writes each step's inputs and weights before every start, so the
    window image carries zeros there; outputs keep the binder's poison.
    """
    from merlin.compiler.build.handoff import bind_deployment_payloads
    deployment = json.loads(safe_path(step / 'deployment.json').read_text())
    if deployment.get('schema') != 'merlin.onchip_deployment.v1' or deployment.get('paging'):
        raise Refusal('only complete nonpaged onchip step deployments are qualified')
    if deployment.get('parameters'):
        raise Refusal('a step deployment declares no parameter blob; staged weights are step inputs')
    payloads = {row['path']: bytes(row['bytes']) for row in deployment['inputs']}
    platform = dict(deployment, provider=dict(deployment['provider'], sys_clk_freq_hz=settings['clock_hz']))
    generated = bind_deployment_payloads(platform, payloads)
    return dict(deployment=deployment, platform_deployment=platform, payloads=payloads, generated=generated)


MAX_STEP_WORKERS = 32


def step_workers():
    """Per-step workers from the cores this process may use, capped. Each item writes only its own directory."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    return max(1, min(MAX_STEP_WORKERS, cores))


def _parallel(function, items):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=step_workers()) as pool:
        return dict(zip(items, pool.map(function, items)))


def step_objects(context):
    """Measured objects for every step package, each checked against its own host window.

    A package that does not fit is recorded as blocked with its reason; it keeps its
    generated step application and gets no ELF.
    """
    application = context['out'] / 'compiled' / 'application'
    plan = json.loads(safe_path(application / 'deployment.json').read_text())
    packages = list(dict.fromkeys(step['package'] for step in plan['steps']))
    (context['out'] / 'steps').mkdir(exist_ok=False)

    def prepare(package):
        work = context['out'] / 'steps' / package
        work.mkdir()
        step = dict(out=work, transport=context['transport'], settings=context['settings'],
                    application=application / 'steps' / package, host_archives=context.get('host_archives', ()),
                    binding=step_binding(application / 'steps' / package, context['settings']))
        measured = objects(step)
        try:
            fit = host_fit(measured, step['binding']['deployment'], step['binding']['generated']['handoff/window.bin'])
        except Refusal as error:
            (work / 'refusal.txt').write_text(str(error) + '\n')
            return dict(blocked=str(error))
        # The bound handoff and deployment that the link checks its payloads against.
        bound = work / 'binding'
        for name, data in step['binding']['generated'].items():
            (bound / name).parent.mkdir(parents=True, exist_ok=True)
            (bound / name).write_bytes(data)
        (bound / 'application').mkdir()
        (bound / 'application' / 'deployment.json').write_bytes(safe_path(step['application'] / 'deployment.json').read_bytes())
        return dict(objects=measured, bundle=bound, fit=fit)
    steps = _parallel(prepare, packages)
    context['steps'] = steps
    context['blocked'] = {package: row['blocked'] for package, row in steps.items() if 'blocked' in row}
    hashes = {}
    for row in steps.values():
        hashes.update(row.get('objects', {}).get('hashes', {}))
    context['objects'] = dict(hashes=hashes, measurements=[], staged=True)
    return context['objects']


def link_steps(context):
    """Link every unblocked step package after checking the composed gate at each boundary."""
    gate = context.get('gate')
    if gate is None:
        raise Refusal('bound EXACT gate is required before ELF linking')
    gate.verify()
    linked = [package for package, row in context['steps'].items() if 'objects' in row]
    record = json.loads(safe_path(gate.receipt).read_text())
    roots = {str(gate.receipt), record.get('bundle_inventory')}

    def one(package):
        row = context['steps'][package]
        # The same gate, re-verified at this step's boundary over its receipt and this step's objects only.
        view = type(gate)(receipt=gate.receipt, hashes={path: gate.hashes[path] for path in gate.hashes
                                                        if path in row['objects']['hashes'] or path in roots})
        step = dict(out=context['out'] / 'steps' / package, settings=context['settings'], gate=view,
                    objects=row['objects'], bundle=row['bundle'])
        elf = link(step)
        return dict(elf=elf, record=step['elf_record'])
    for package, result in _parallel(one, linked).items():
        context['steps'][package].update(result)
    gate.verify()
    return {package: context['steps'][package]['elf'] for package in linked}
