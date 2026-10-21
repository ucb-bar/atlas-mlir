"""Measure the native C runtime through model-backed, virtual MMIO callbacks.

This is evaluator code, not a platform driver. Python callback overhead remains
in the measurements; bus latency and cache maintenance are not modeled.
"""

import ctypes
import hashlib
import json
import platform
import shutil
import struct
import subprocess

from tools.atlas_boot_pack import IMEM_SIZE


class HostRuntimeProfile:
    def __init__(self, work, record, files, binary, target_source):
        if hashlib.sha256(binary).hexdigest() != record["program_sha256"]:
            raise ValueError("host profile package contains a different program")
        for path, digest in record["files"].items():
            if hashlib.sha256(files[path]).hexdigest() != digest:
                raise ValueError("host profile package member changed")
        source = [key for key in files if key.endswith(".c")]
        if len(source) != 1 or "/" in source[0]:
            raise ValueError("host profiling needs one exported C/H pair")
        name = source[0][:-2]
        runtime_header = (target_source / "runtime/atlas_host.h").read_bytes()
        if runtime_header not in files[name + ".h"]:
            raise ValueError("exported C/H and selected runtime headers differ")
        work.mkdir(exist_ok=False)
        for path in (name + ".c", name + ".h"):
            (work / path).write_bytes(files[path])
        adapter = work / "profile.c"
        adapter.write_text(f"""#define _POSIX_C_SOURCE 200809L
#include "{name}.h"
#include <time.h>
static atlas_host host;
static uint64_t residency_ticks;
static uint64_t ticks(void *context) {{
  (void)context; struct timespec now;
  clock_gettime(CLOCK_MONOTONIC,&now);
  return (uint64_t)now.tv_sec*UINT64_C(1000000000)+(uint64_t)now.tv_nsec;
}}
int profile_init(atlas_host_read32 read,atlas_host_write32 write) {{
  int status=atlas_host_init(&host,(atlas_host_map){{0,{IMEM_SIZE}u,{IMEM_SIZE}u}},
    (atlas_host_mmio){{read,write,0}},(atlas_host_memory){{1,0,0}});
  if (status) return status;
  return atlas_host_set_clock(&host,ticks,0);
}}
int profile_begin(void) {{
  uint64_t begin=ticks(0);
  int status=atlas_host_ensure_loaded(&host,{name}_program,{name.upper()}_PROGRAM_WORDS);
  residency_ticks+=ticks(0)-begin;
  if (status) return status;
  status=atlas_host_start(&host,{name}_device_buffers,{name.upper()}_DEVICE_BUFFER_COUNT);
  if (status) return status;
  return atlas_host_poll(&host)==ATLAS_HOST_BUSY ? 0 : -1;
}}
int profile_end(void) {{ return atlas_host_wait(&host,8); }}
uint64_t profile_counter(unsigned index) {{
  uint64_t values[]={{host.timings.load_ticks,host.timings.start_ticks,host.timings.wait_ticks,
    residency_ticks,host.image_loads,host.image_reuses,host.imem_words_written,host.launches}};
  return index<8 ? values[index] : 0;
}}
""")
        cc = shutil.which("cc")
        if cc is None:
            raise ValueError("a native C compiler is required for host profiling")
        command = [
            cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            "-I" + str(target_source / "runtime"),
            str(work / source[0]),
            str(adapter),
            str(target_source / "runtime/atlas_host.c"),
            "-o",
            str(work / "host.so"),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        (work / "build.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise ValueError("native host runtime build failed; see build.log")
        (work / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        self.library = ctypes.CDLL(str(work / "host.so"))
        read = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t)
        write = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32
        )
        self.callbacks = read(self._read), write(self._write)
        self.library.profile_init.argtypes = [read, write]
        self.library.profile_counter.argtypes = [ctypes.c_uint]
        self.library.profile_counter.restype = ctypes.c_uint64
        self.binary, self.words, self.errors = binary, {}, []
        self.active, self.pending = None, None
        self.last_status, self.last_retired = 1, 0
        self.before = None
        if self.library.profile_init(*self.callbacks):
            raise ValueError("native host runtime initialization failed")
        self.provenance = {
            "platform": platform.machine(),
            "clock": "CLOCK_MONOTONIC",
            "unit": "nanoseconds",
            "scope": (
                "native C plus Python MMIO callbacks; coherent memory; "
                "no bus/cache latency; model execution excluded"
            ),
            "model_state": "independently initialized per invocation; host image residency retained",
            "runtime_files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    target_source / "runtime/atlas_host.c",
                    target_source / "runtime/atlas_host.h",
                ]
            },
        }

    def _read(self, _, address):
        try:
            if address == IMEM_SIZE + 4:
                return (
                    self.active.read_csrf(0xC01)
                    if self.active is not None
                    else self.last_retired
                )
            if address == IMEM_SIZE + 8:
                return (
                    self.active.read_csrf(0xC02)
                    if self.active is not None
                    else self.last_status
                )
            raise ValueError("unexpected virtual CSR read")
        except Exception as error:
            self.errors.append(str(error))
            return 0

    def _write(self, _, address, word):
        if 0 <= address < len(self.binary) and address % 4 == 0 and self.active is None:
            self.words[address // 4] = word
        elif (
            address == IMEM_SIZE + 24
            and word == 1
            and self.active is None
            and self.pending is not None
        ):
            self.active, self.pending = self.pending, None
        else:
            self.errors.append("unexpected virtual MMIO write or overlapping launch")

    def _counters(self):
        return tuple(self.library.profile_counter(i) for i in range(8))

    def begin(self, state):
        if self.active is not None or self.before is not None:
            raise ValueError("host profile invocation already active")
        self.before, self.pending = self._counters(), state
        if self.library.profile_begin() or self.errors or self.active is not state:
            raise ValueError("host runtime rejected load/start: " + repr(self.errors))
        loaded = struct.pack(
            f"<{len(self.binary) // 4}I",
            *(self.words.get(i, -1) for i in range(len(self.binary) // 4)),
        )
        if loaded != self.binary:
            raise ValueError("host runtime loaded different instruction words")

    def end(self, state):
        if self.active is not state or not state.halted or self.before is None:
            raise ValueError("cannot report host completion before the model halts")
        if self.library.profile_end() or self.errors:
            raise ValueError(
                "host runtime rejected model completion: " + repr(self.errors)
            )
        after = self._counters()
        names = (
            "load_ns",
            "start_ns",
            "wait_ns",
            "load_or_reuse_ns",
            "image_loads",
            "image_reuses",
            "words_written",
            "launches",
        )
        result = dict(
            zip(
                names,
                (a - b for a, b in zip(after, self.before, strict=True)),
                strict=True,
            )
        )
        if result["launches"] != 1:
            raise ValueError("expected one host launch per model invocation")
        self.last_status, self.last_retired = (
            state.read_csrf(0xC02),
            state.read_csrf(0xC01),
        )
        self.active, self.before = None, None
        return result
