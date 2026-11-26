"""Execute the generated fixed-address application, then a one-byte negative.

Public buffers use the published CPU addresses. Tile I/O callbacks copy between
that CPU window and the instruction model's separate device memory, exactly as
the generated counted runtime requests; this checker adds no execution plan.

Deploy's EXACT gate runs this file by path in the qualified interpreter. It loads its siblings here and
the declared Atlas decoder from their files (sources.py) and imports nothing of the compiler it checks:
python check_model.py --bundle B --out O --receipt R --atlas-root ROOT --qualification DIR.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes as ct
from functools import wraps
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

PIN = "b97b2f5a97e432141c4ef8504bec396651d77a8f"
# Fills for the separate-window sentinel runs: a 255-byte cycle and its complement differ at every byte.
SENTINELS = (bytes(range(1, 256)), bytes(value ^ 0xFF for value in range(1, 256)))


def digest(data):
    return sha256(data).hexdigest()


def model_checkout_identity():
    """Measure the checkout supplying the imported functional model."""
    import npu_model

    package = Path(npu_model.__file__).resolve().parent
    root = Path(subprocess.check_output(
        ["git", "-C", str(package), "rev-parse", "--show-toplevel"], text=True).strip())
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=normal"], text=True)
    identity = dict(root=str(root), package=str(package), revision=revision,
                    clean=not status, status_porcelain=status, expected_revision=PIN)
    if revision != PIN or status:
        raise AssertionError("functional model checkout is not clean at the required revision: " +
                             json.dumps(identity, sort_keys=True))
    return identity


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_library(bundle, output):
    handoff = bundle / "application/handoff"
    source = output / "model-shim.c"
    source.write_text('''#include "deployment_adapter.h"
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
uint64_t model_counter(unsigned which) {
  switch(which) {
  case 0:return model_host.image_loads;
  case 1:return model_host.image_reuses;
  case 2:return model_host.imem_words_written;
  case 3:return model_host.launches;
  default:return 0;
  }
}
''')
    library = output / "application.so"
    compiler = shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("GCC is required to execute the generated application")
    command = [compiler, "-O2", "-std=c11", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
               "-I", str(handoff), str(handoff / "capsule_combined.c"), str(source), "-lm", "-o", str(library)]
    completed = subprocess.run(command, capture_output=True, text=True)
    (output / "host-build.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError("generated C library build failed; see " + str(output / "host-build.log"))
    return library, dict(command=command, compiler_version=subprocess.check_output(
        [compiler, "--version"], text=True).splitlines()[0], shim_sha256=digest(source.read_bytes()),
        generated_source_sha256=digest((handoff / "capsule_combined.c").read_bytes()),
        generated_header_sha256=digest((handoff / "capsule_combined.h").read_bytes()),
        adapter_header_sha256=digest((handoff / "deployment_adapter.h").read_bytes()),
        library_sha256=digest(library.read_bytes()))


class FixedWindow:
    def __init__(self, base, size):
        self.base, self.size = base, size
        self.libc = ct.CDLL(None, use_errno=True)
        self.libc.mmap.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_int, ct.c_int, ct.c_long]
        self.libc.mmap.restype = ct.c_void_p
        self.libc.munmap.argtypes = [ct.c_void_p, ct.c_size_t]
        # Linux MAP_FIXED_NOREPLACE refuses occupied mappings without clobbering.
        result = self.libc.mmap(base, size, 3, 0x02 | 0x20 | 0x100000, -1, 0)
        if result != base:
            if result != ct.c_void_p(-1).value:
                self.libc.munmap(result, size)
            raise OSError(ct.get_errno(), "mmap MAP_FIXED_NOREPLACE did not return published window")

    def load(self, payload):
        if len(payload) > self.size:
            raise ValueError("window payload exceeds deployment")
        ct.memset(self.base, 0, self.size)
        ct.memmove(self.base, payload, len(payload))

    def close(self):
        if self.libc.munmap(self.base, self.size):
            raise OSError(ct.get_errno(), "munmap failed")


class DMAAudit:
    def __init__(self, base, size, load_size=None):
        """Every DMA stays in [base, base + size); loads alone may reach load_size (a resident arena after it)."""
        self.base, self.size = base, size
        self.limits = dict(load=size if load_size is None else load_size, store=size)
        self.commands, self.bytes, self.spans = Counter(), Counter(), {}
        self.sequence, self.originals = sha256(), {}

    def __enter__(self):
        from npu_model.configs.isa_definition import DMA_LOAD, DMA_STORE
        for instruction_type, operation, register in ((DMA_LOAD, "load", "rs1"), (DMA_STORE, "store", "rd")):
            original = instruction_type.exec
            self.originals[instruction_type] = original

            def wrap(original, operation, register):
                @wraps(original)
                def execute(instruction, state):
                    address = state.dma_dram_address(state.read_xrf(getattr(instruction, register)))
                    length = state.read_xrf(instruction.rs2)
                    if not self.base <= address < address + length <= self.base + self.limits[operation]:
                        raise AssertionError(f"DMA {operation} leaves published window: {address:#x}+{length}")
                    original(instruction, state)
                    self.commands[operation] += 1
                    self.bytes[operation] += length
                    low, high = self.spans.get(operation, (address, address + length))
                    self.spans[operation] = min(low, address), max(high, address + length)
                    self.sequence.update(json.dumps((operation, address, length), separators=(",", ":")).encode() + b"\n")
                return execute

            instruction_type.exec = wrap(original, operation, register)
        return self

    def __exit__(self, *unused):
        for instruction_type, original in self.originals.items():
            instruction_type.exec = original

    def record(self):
        return dict(scope="every executed DMA_LOAD and DMA_STORE", commands=dict(self.commands),
                    bytes=dict(self.bytes), spans={key: dict(first=low, end=high)
                    for key, (low, high) in self.spans.items()},
                    ordered_events_sha256=self.sequence.hexdigest(), bounds_checked_before_execution=True)


def memory_windows(deployment):
    """Return the host map and the device memory, the compiled provider window.

    On chip the two are one window. Otherwise the device window lies inside the
    host map, on the provider's allocation alignment, and apart from every
    published host buffer, so DMA confined to it cannot reach one.
    """
    host, device = deployment["window"], deployment["provider"]["window"]
    if deployment.get("schema") == "merlin.onchip_deployment.v1" and device != host:
        raise AssertionError("on-chip deployment requires the provider window to equal the host map")
    alignment = deployment["provider"].get("allocation_alignment")
    if (any(type(window.get(key)) is not int or window[key] <= 0 for window in (host, device)
            for key in ("address", "bytes"))
            or type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1)):
        raise AssertionError("memory windows and allocation alignment must be positive integers")
    if device["address"] % alignment or device["bytes"] % alignment:
        raise AssertionError("device window is not on the provider allocation alignment")
    host_end, device_end = host["address"] + host["bytes"], device["address"] + device["bytes"]
    if not host["address"] <= device["address"] < device_end <= host_end:
        raise AssertionError("device window extends past the host map")
    rows = [*deployment["inputs"], *deployment["outputs"], *deployment["hash_ranges"],
            *deployment.get("allocations", []), *([deployment["parameters"]] if deployment.get("parameters") else [])]
    for row in rows:
        start, end = row["address"], row["address"] + row["bytes"]
        if not host["address"] <= start <= end <= host_end:
            raise AssertionError(f"published buffer lies outside the host map: {row.get('name')}")
        if device != host and start < device_end and device["address"] < end:
            raise AssertionError(f"published host buffer overlaps the device window: {row.get('name')}")
    for row in deployment["provider"].get("reservations", []):
        if not device["address"] <= row["address"] <= row["address"] + row["bytes"] <= device_end:
            raise AssertionError(f"provider reservation lies outside the device window: {row.get('name')}")
    return host, device


def resident_arena(deployment, application):
    """The resident arena bytes, which must start where the window ends, or b"" for a nonresident deployment.

    The bytes come from the bundle and must equal its deployment record; that the linked ELF carries the same bytes
    is checked by the link (_elf_record) and the loader manifest, not here.
    """
    row = deployment.get("resident")
    if not row:
        return b""
    data = (application / row["path"]).read_bytes()
    if (any(row["address"] != window["address"] + window["bytes"]
            for window in (deployment["window"], deployment["provider"]["window"]))
            or len(data) != row["bytes"] or digest(data) != row["sha256"]):
        raise AssertionError("resident arena differs from the deployment or does not follow the window")
    return data


def check_resident_counters(label, published, execution, counters):
    """The host's counters, the IMEM words the model received, and the deployment's replay must agree."""
    words = published["imem_words"]["resident"]
    if (execution["imem_residency"]["imem_words_written"] != words or counters["imem_words_written"] != words
            or counters["launches"] != published["launches"] or counters["image_loads"] != published["image_loads"]):
        raise AssertionError(f"{label}: resident IMEM counters differ from the deployment record")


def run_case(label, payload, library, deployment, runtime_options, images, plans, expected, output, fill=b"\0",
             arena=b"", *, atlas_root):
    from functional_runtime import FunctionalModelRuntime, resident_schedule
    host, device = memory_windows(deployment)
    base, size = host["address"], host["bytes"]
    # Separate windows are one physical memory on the chip. The model sees device
    # memory only through tile I/O, so the host map's copy of it must stay zero, or
    # in a sentinel run must keep the fill it starts with.
    device_range = slice(device["address"] - base, device["address"] - base + device["bytes"])
    if device != host and any(payload[device_range]):
        raise AssertionError("payload writes the device window through the host map")
    output.mkdir(parents=True, exist_ok=True)
    ct.memset(base, 0, size)
    ct.memmove(base, payload, len(payload))
    staging = (fill * (device["bytes"] // len(fill) + 1))[:device["bytes"]]
    if device != host and any(staging):
        ct.memmove(device["address"], staging, device["bytes"])
    immutable = [*deployment["inputs"]]
    if deployment.get("parameters"):
        immutable.append(deployment["parameters"])
    if arena:
        # The resident arena follows the window; the device reads it in place, and it must stay unchanged.
        ct.memmove(deployment["resident"]["address"], arena, len(arena))
        immutable.append(deployment["resident"])
    before = [(row, ct.string_at(row["address"], row["bytes"])) for row in immutable]
    residency = resident_schedule(deployment)
    runtime = FunctionalModelRuntime(atlas_root, Path(os.environ["ATLAS_NPU_MODEL_ROOT"]),
        images, output / "execution", memory_base=device["address"], memory_bytes=device["bytes"], image_plans=plans,
        max_instructions=128000000, trace_payloads=False, compact_dram_prefix=True, **runtime_options,
        # The device reads the resident arena in place, so the model's device memory starts with it loaded.
        **(dict(immutable_loads=[(deployment["resident"]["address"], arena)], capacity_base=device["address"],
                capacity_bytes=device["bytes"] + len(arena)) if arena else {}),
        **(dict(resident_images=residency) if residency else {}))
    library.model_bind.argtypes = [type(callback) for callback in runtime.callbacks]
    library.model_bind.restype = library.model_run.restype = ct.c_int
    library.model_counter.argtypes, library.model_counter.restype = [ct.c_uint], ct.c_uint64
    started = time.monotonic()
    # Device DMA may also read the resident arena after the window, and never store into it.
    with DMAAudit(device["address"], device["bytes"], device["bytes"] + len(arena)) as dma:
        bind_status = library.model_bind(*runtime.callbacks)
        status = library.model_run() if bind_status == 0 else bind_status
        execution = runtime.finish()
    if status or not execution["launches"] or not dma.commands["load"] or not dma.commands["store"]:
        raise AssertionError(f"{label}: generated application failed: bind={bind_status}, run={status}")
    counters = dict(zip(("image_loads", "image_reuses", "imem_words_written", "launches"),
                        (library.model_counter(i) for i in range(4)), strict=True))
    if residency:
        check_resident_counters(label, deployment["provider"]["imem_residency"], execution, counters)
    if arena:
        start = deployment["resident"]["address"] - runtime.base
        if bytes(runtime.memory[start:start + len(arena)]) != arena:
            raise AssertionError(f"{label}: the device changed its resident arena")
    if device != host and ct.string_at(device["address"], device["bytes"]) != staging:
        raise AssertionError(f"{label}: generated host code wrote the device window through the host map")
    for row, original in before:
        if ct.string_at(row["address"], row["bytes"]) != original:
            raise AssertionError("generated application changed immutable public buffer")
    outputs, mismatch = [], None
    for row in deployment["hash_ranges"]:
        actual = ct.string_at(row["address"], row["bytes"])
        reference = expected[row["index"]]
        if len(reference) != len(actual):
            raise AssertionError("reference and published output sizes differ")
        member = f"output_{row['index']:03d}.bin"
        (output / member).write_bytes(actual)
        differences = [i for i, (a, b) in enumerate(zip(actual, reference, strict=True)) if a != b]
        if differences and mismatch is None:
            i = differences[0]
            word = i // 4
            mismatch = dict(output_index=row["index"], name=row["id"], byte_offset=i,
                address=row["address"] + i, actual_byte=actual[i], expected_byte=reference[i],
                word_index=word, actual_word_hex=actual[word*4:word*4+4].hex(),
                expected_word_hex=reference[word*4:word*4+4].hex())
        outputs.append(dict(index=row["index"], name=row["id"], role=row["role"], address=row["address"],
            bytes=len(actual), sha256=digest(actual), expected_sha256=digest(reference),
            mismatched_bytes=len(differences), exact=not differences, path=member))
    record = dict(schema="atlas.model_case.v1", case=label, result="EXACT" if mismatch is None else "MISMATCH",
        bind_status=bind_status, run_status=status, outputs=outputs, first_mismatch=mismatch,
        elapsed_seconds=time.monotonic()-started, launch_count=len(execution["launches"]),
        total_instructions=execution["total_instructions"], dma=dma.record(),
        host_counters=counters,
        runtime_path="execution/runtime.json", runtime_sha256=digest((output / "execution/runtime.json").read_bytes()),
        public_buffers_hashed_at_published_addresses=True, immutable_buffers_unchanged=True)
    write_json(output / "result.json", record)
    return record


def bind_sources(atlas_root, modules):
    """Load MODULES (siblings here) and ATLAS_ROOT/tools' decoder from the files the gate hashed (sources.py)."""
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("exactness_sources", here / "sources.py")
    sources = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sources)
    sources.bind(here, atlas_root, modules)


def qualified_execution_options(native_qualification, memory_qualification, native_cache):
    """Authenticate both optional execution implementations before device I/O."""
    from npu_model.configs import numerics, rtl_math
    from npu_model.functional import state as original_state
    from qualified_native_dot import load_qualified_math
    from merlin.runtime.uniform_sparse_memory import UniformSparseByteMemory, verify_qualification

    native_qualification, memory_qualification = Path(native_qualification), Path(memory_qualification)
    proxy = load_qualified_math(numerics.NUMERICS["rtl"], rtl_math.product,
        {"numerics": numerics.__file__, "rtl_math": rtl_math.__file__}, native_qualification, native_cache)
    memory_receipt = json.loads(memory_qualification.read_text())
    memory_proof = verify_qualification(memory_receipt, reference_source=original_state.__file__,
        expected_reference_sha256="211081f8f356bccf397069de24f9fc328420b7589642c50f211565c0cfd78ca0")
    metadata = {"backend": "qualified_native",
        "arithmetic": proxy.execution_metadata(), "memory": memory_proof,
        "native_qualification_file_sha256": sha256(native_qualification.read_bytes()).hexdigest(),
        "memory_qualification_file_sha256": sha256(memory_qualification.read_bytes()).hexdigest(),
        "independent_reference_engine_used_for_execution": False}
    return dict(math_proxy=proxy, memory_factory=UniformSparseByteMemory, execution_metadata=metadata)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--atlas-root", type=Path, required=True,
                        help="Atlas support files: tools/ (instruction decoder) and docs/ (variant inventory)")
    parser.add_argument("--qualification", type=Path, required=True,
                        help="directory with qualification.json and memory-qualification.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--flip-offset", type=int, help="byte offset in input_0; default first nonzero byte")
    args = parser.parse_args()
    bundle, output, atlas_root = args.bundle.resolve(), args.out.resolve(), args.atlas_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_raw = (bundle / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    deployment = manifest["deployment"]
    application = bundle / "application"
    if json.loads((application / "deployment.json").read_text()) != deployment:
        raise AssertionError("bundle deployment differs from compiler publication")
    memory_windows(deployment)
    payload = (bundle / "window.bin").read_bytes()
    if digest(payload) != manifest["window_sha256"]:
        raise AssertionError("bundle window changed")
    reference = json.loads((bundle / "reference/reference.json").read_text())
    for name, expected_hash in reference["evaluator_members"].items():
        if digest((bundle / "reference" / name).read_bytes()) != expected_hash:
            raise AssertionError("frozen reference member changed: " + name)
    expected = {int(key.removeprefix("output_")): value.tobytes() for key, value in
                dict(np.load(bundle / "reference/expected.npz", allow_pickle=False)).items()}
    for row in manifest["expected_hashes"]:
        if digest(expected[next(item["index"] for item in deployment["outputs"] if item["id"] == row["name"])]) != row["sha256"]:
            raise AssertionError("frozen expected hash differs from bundle")
    images, plans = [], {}
    for row in deployment["provider"]["images"]:
        raw = (application / row["path"]).read_bytes()
        if digest(raw) != row["sha256"] or len(raw) != row["bytes"]:
            raise AssertionError("instruction image changed")
        images.append(raw)
        plans[digest(raw)] = json.loads((application / Path(row["path"]).parent / "execution_plan.json").read_text())
    library_path, build = build_library(bundle, output)
    write_json(output / "host-build.json", build)
    if args.prepare_only:
        print("PREPARED", library_path, flush=True)
        return
    bind_sources(atlas_root, ("model_runtime", "qualified_native_dot"))
    model_identity = model_checkout_identity()
    options = qualified_execution_options(args.qualification / "qualification.json",
        args.qualification / "memory-qualification.json", output / "native-cache")
    # functional_runtime imports the rest of npu_model. Binding it last keeps the import order, and so the order of
    # the model source hashes each execution record lists, what it was before the checker moved.
    bind_sources(atlas_root, ("functional_runtime",))
    arena = resident_arena(deployment, application)
    window = FixedWindow(deployment["window"]["address"], deployment["window"]["bytes"] + len(arena))
    try:
        library = ct.CDLL(str(library_path))
        print("RUN positive", flush=True)
        positive = run_case("positive", payload, library, deployment, options, images, plans, expected, output / "positive",
                            arena=arena, atlas_root=atlas_root)
        if positive["result"] != "EXACT":
            raise AssertionError("positive execution differs from frozen expected output")
        row = next(row for row in deployment["inputs"] if row["index"] == 0)
        start = row["address"] - deployment["window"]["address"]
        input_before = payload[start:start + row["bytes"]]
        offset = args.flip_offset if args.flip_offset is not None else next(i for i, value in enumerate(input_before) if value & 0x7f)
        if not 0 <= offset < row["bytes"]:
            raise ValueError("flip offset outside input_0")
        negative_payload = bytearray(payload)
        absolute_offset = start + offset
        old = negative_payload[absolute_offset]
        negative_payload[absolute_offset] ^= 0x80
        changed = dict(input_index=0, input_name=row["id"], input_byte_offset=offset,
            window_byte_offset=absolute_offset, address=row["address"] + offset, xor_mask=128,
            before_byte=old, after_byte=negative_payload[absolute_offset],
            input_before_sha256=digest(input_before),
            input_after_sha256=digest(negative_payload[start:start + row["bytes"]]),
            window_before_sha256=digest(payload), window_after_sha256=digest(negative_payload),
            changed_byte_count=sum(a != b for a, b in zip(payload, negative_payload, strict=True)))
        write_json(output / "negative-byte-flip.json", changed)
        print("RUN negative", json.dumps(changed, sort_keys=True), flush=True)
        negative = run_case("negative", bytes(negative_payload), library, deployment, options, images, plans,
                            expected, output / "negative", arena=arena, atlas_root=atlas_root)
        if negative["result"] != "MISMATCH" or negative["first_mismatch"] is None:
            raise AssertionError("negative byte flip did not influence a declared output")
        # Separate windows only: rerun the positive from each fill, so a host store of any value into
        # the host map's copy of the device window changes the final bytes of at least one run.
        sentinels = []
        if deployment["window"] != deployment["provider"]["window"]:
            for index, fill in enumerate(SENTINELS):
                label = f"sentinel-{index}"
                print("RUN", label, flush=True)
                record = run_case(label, payload, library, deployment, options, images, plans, expected,
                                  output / label, fill, arena=arena, atlas_root=atlas_root)
                if record["result"] != "EXACT":
                    raise AssertionError(f"{label}: execution differs from frozen expected output")
                sentinels.append(dict(fill_sha256=digest(fill), fill_period=len(fill), **record))
        if digest(library_path.read_bytes()) != build["library_sha256"]:
            raise AssertionError("application library changed between runs")
        for row, raw in zip(deployment["provider"]["images"], images, strict=True):
            if (application / row["path"]).read_bytes() != raw:
                raise AssertionError("instruction program changed during execution")
        if model_checkout_identity() != model_identity:
            raise AssertionError("functional model checkout changed during execution")
        receipt = dict(schema="atlas.model_check.v1", status="PASS", model_revision=model_identity["revision"],
            model_checkout=model_identity,
            backend="qualified_native", scope=manifest["scope"], positive=positive, negative=negative,
            byte_flip=changed, same_application_library=True, same_instruction_images=True,
            independent_expected_unchanged=True, instruction_images=[dict(path=row["path"], sha256=row["sha256"],
                bytes=row["bytes"]) for row in deployment["provider"]["images"]],
            bundle_manifest_sha256=digest(manifest_raw), build=build,
            checker_sha256=digest(Path(__file__).read_bytes()),
            mapping=dict(address=window.base, bytes=window.size, flags="MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE"),
            **(dict(resident={key: deployment["resident"][key] for key in ("address", "bytes", "sha256")})
               if arena else {}),
            **(dict(sentinel_runs=sentinels) if sentinels else {}))
        write_json(args.receipt, receipt)
        print("PASS positive EXACT; negative executed output MISMATCH", flush=True)
    finally:
        window.close()


if __name__ == "__main__":
    main()
