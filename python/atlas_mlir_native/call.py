"""Caller-configured raw-BF16 C ABI adaptation for one Atlas native region.

Platform records are original caller claims, not hardware execution evidence.
The adapter preserves BF16 bits and leaves image/loader qualification separate.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path, PurePosixPath

from merlin.compiler.artifact import Artifact, Compilation
from merlin.compiler.backend.cabi import CRegionABI, TensorPort
from merlin.compiler.backend.stateful import CContext, CRegionABI2, EmbeddedPayload, render_u32_payload
from merlin.compiler.build import process
from merlin.compiler.semantic.target_binding import verify_native_publication

from .policy import BF16_DRAM_PAIR_LAYOUT, DRAM_BASE, TILE_BYTES


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _member(name):
    if (type(name) is not str or not name or "\\" in name or ":" in name or
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or
            PurePosixPath(name).as_posix() != name):
        raise ValueError("BF16 platform members need plain relative paths")
    return name


def _identifier(name):
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    return (type(name) is str and bool(name) and name[0] in alphabet and
            all(c in alphabet + "0123456789" for c in name))


def _interval(row):
    if type(row) is not dict or set(row) != {"address", "bytes", "alignment"}:
        raise ValueError("device pre-reservations require exact address/bytes/alignment")
    a, n, align = row["address"], row["bytes"], row["alignment"]
    if (type(a) is not int or type(n) is not int or type(align) is not int or
            a <= 0 or n <= 0 or a + n > 1 << 32 or align < 2 or align & (align - 1) or a % align):
        raise ValueError("device pre-reservation has invalid extent, address or alignment")
    return a, a + n


class BF16Call:
    """One selected host compiler, caller platform and context resource ceiling.

    ``contract`` selects a JSON artifact in the saved platform Compilation.
    Its SHA is the original CContext contract identity. The two device intervals
    are pre-reserved outside available topology capacity, never hidden scratch.
    A platform initializer has the declared signature
    int(atlas_bf16_call *, const atlas_bf16_program *, void *, size_t).
    """
    def __init__(self, *, context, compiler, flags, runtime, headers, platform, contract,
                 platform_header, platform_initialize, link_inputs):
        if type(context) is not CContext or type(platform) is not Compilation:
            raise ValueError("BF16 call needs an original typed context and saved caller platform Compilation")
        if (context.declaration, context.ctype) != ("call/region.h", "atlas_bf16_call"):
            raise ValueError("BF16 call requires the explicit call/region.h atlas_bf16_call context declaration")
        if (not _identifier(platform_initialize) or context.initialize == platform_initialize or
                type(link_inputs) is not tuple or not link_inputs or len(set(link_inputs)) != len(link_inputs)):
            raise ValueError("BF16 call requires a distinct selected platform initializer and unique link inputs")
        compiler = Path(compiler)
        if not compiler.is_absolute() or not compiler.is_file() or not os.access(compiler, os.X_OK):
            raise ValueError("BF16 host compiler must be an absolute executable")
        if (type(flags) is not tuple or not flags or any(type(f) is not str or not f.startswith("-") or
                "\x00" in f or f in {"-c", "-S", "-E", "-r", "-shared"} or f.startswith("-o") for f in flags)):
            raise ValueError("BF16 host flags must be explicit option tokens without stage/output overrides")
        self.context, self.platform = context, platform
        self.compiler, self.flags = compiler.resolve(), flags
        if not Path(runtime).is_absolute() or not Path(headers).is_absolute():
            raise ValueError("BF16 runtime/header directories must be explicitly absolute")
        self.runtime, self.headers = Path(runtime).resolve(), Path(headers).resolve()
        self.contract, self.platform_header = _member(contract), _member(platform_header)
        self.platform_initialize = platform_initialize
        self.link_inputs = tuple(_member(name) for name in link_inputs)
        self._owners = (context, platform)
        platform.verify()
        products = {a.path.relative_to(platform.directory).as_posix(): a for a in platform.artifacts}
        if (self.contract not in products or products[self.contract].kind != "json" or
                self.platform_header not in products or products[self.platform_header].kind != "c-header" or
                any(name not in products or products[name].kind not in {"c-source", "object", "archive"}
                    for name in self.link_inputs)):
            raise ValueError("BF16 platform contract/header/link inputs must be actual admitted platform artifacts")
        if products[self.contract].sha256 != context.contract_sha256:
            raise ValueError("BF16 context is not bound to the original platform contract artifact")
        self._document = self._read_contract(products[self.contract].path)
        names = ("atlas_bf16_call.c", "atlas_bf16_call.h", "atlas_host.c", "atlas_host.h", "atlas_model_tile.h")
        paths = (self.compiler, platform.manifest, *(self.runtime / name for name in names),
                 self.headers / "finite_float_boundary.h", Path(__file__))
        if any(not path.is_file() for path in paths):
            raise ValueError("BF16 selected compiler/runtime/header sources are unavailable")
        self._files = tuple((path, _hash(path)) for path in paths)
        self._settings = self._current()
        self.identity = hashlib.sha256(_json({"configuration": self._settings,
            "sources": [(str(p), sha) for p, sha in self._files]}).encode()).hexdigest()
        self.revalidate()

    def _current(self):
        return _json({"context": self.context.record(), "platform": self.platform.to_dict(),
                      "compiler": str(self.compiler), "flags": self.flags, "runtime": str(self.runtime),
                      "headers": str(self.headers), "contract": self.contract, "header": self.platform_header,
                      "initialize": self.platform_initialize, "link_inputs": self.link_inputs})

    @staticmethod
    def _read_contract(path):
        def pairs(rows):
            result = {}
            for key, value in rows:
                if key in result:
                    raise ValueError("duplicate BF16 platform contract field")
                result[key] = value
            return result
        doc = json.loads(path.read_text(), object_pairs_hook=pairs)
        fields = {"schema", "map", "input", "output", "pre_reserved_outside_available_capacity", "control"}
        if (type(doc) is not dict or set(doc) != fields or doc["schema"] != "atlas.bf16-call-platform.v1" or
                doc["pre_reserved_outside_available_capacity"] is not True or doc["control"] != "single_context"):
            raise ValueError("BF16 platform needs explicit pre-reservations and "
                             "single-context exclusive-control obligation")
        mapping = doc["map"]
        if type(mapping) is not dict or set(mapping) != {"imem_base", "csr_base", "imem_capacity_bytes"}:
            raise ValueError("BF16 platform needs an exact selected host map and program storage limit")
        if any(type(value) is not int or value < 0 or value % 4 for value in mapping.values()):
            raise ValueError("BF16 platform map needs aligned nonnegative integer values")
        if (mapping["imem_capacity_bytes"] < 4 or
                mapping["imem_base"] + mapping["imem_capacity_bytes"] > 1 << 32 or
                mapping["csr_base"] + 0x1c > 1 << 32):
            raise ValueError("BF16 platform map/program extent is outside the selected host ABI")
        spans = [_interval(doc[name]) for name in ("input", "output")]
        spans.extend(((mapping["imem_base"], mapping["imem_base"] + mapping["imem_capacity_bytes"]),
                      (mapping["csr_base"], mapping["csr_base"] + 0x1c)))
        ordered = sorted(spans)
        if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("BF16 device pre-reservations overlap program storage or control registers")
        return doc

    def revalidate(self):
        if (self.context is not self._owners[0] or self.platform is not self._owners[1] or
                self._current() != self._settings or any(_hash(p) != sha for p, sha in self._files)):
            raise ValueError("original BF16 context/platform/compiler/runtime selection changed")
        loaded = Compilation.load(self.platform.manifest)
        if loaded.to_dict() != self.platform.to_dict():
            raise ValueError("BF16 platform differs from its original saved Compilation")
        if self._read_contract(self.platform.directory / self.contract) != self._document:
            raise ValueError("BF16 platform reservations or map changed")

    def check_slots(self, fixed_inputs, fixed_outputs):
        self.revalidate()
        if (set(dict(fixed_inputs)) != {"arg0"} or len(fixed_outputs) != 1 or
                self._document["input"]["address"] != DRAM_BASE + dict(fixed_inputs)["arg0"] * TILE_BYTES or
                self._document["output"]["address"] != DRAM_BASE + fixed_outputs[0] * TILE_BYTES or
                self._document["input"]["bytes"] != TILE_BYTES or self._document["output"]["bytes"] != TILE_BYTES):
            raise ValueError("BF16 pre-reservations differ from selected profile slot ABI")
        if self.context.workspace_bytes < TILE_BYTES:
            raise ValueError("BF16 caller workspace ceiling is smaller than the admitted physical tile")

    def check_plan(self, plan, binary_bytes):
        self.revalidate()
        doc = self._document
        if (plan.get("schema") != "atlas.native_tile_execution_plan.v8" or plan.get("shape") != [32, 32] or
                plan.get("dtype") != "bf16" or plan.get("input_bytes") != TILE_BYTES or
                plan.get("output_bytes") != TILE_BYTES or plan.get("output_physical_layouts") != ["bf16_pair_halves"] or
                plan.get("input_addresses") != [doc["input"]["address"]] or
                plan.get("output_addresses") != [doc["output"]["address"]] or
                doc["input"]["bytes"] != TILE_BYTES or doc["output"]["bytes"] != TILE_BYTES or
                not binary_bytes or len(binary_bytes) % 4 or len(binary_bytes) > doc["map"]["imem_capacity_bytes"]):
            raise ValueError("BF16 native plan/program differs from original layout, "
                             "slot reservations or program limit")
        if (plan.get("boundary_domain_obligations") != [{"input": "arg0",
                "required_domain": "finite-normal-or-zero-bf16", "status": "caller_premise_unchecked"}]):
            raise ValueError("BF16 wrapper cannot discharge this native source-domain contract")
        if self.context.workspace_bytes < TILE_BYTES:
            raise ValueError("BF16 caller workspace ceiling is smaller than the admitted physical tile")

    def build(self, compilation, *, resource_kind):
        """Compile actual adapter/runtime/platform objects; no image is linked."""
        self.revalidate()
        compilation.verify()
        native = json.loads((compilation.directory / "manifest.json").read_text())
        verify_native_publication(compilation.directory, native, engine="merlin_native",
                                  request_digest=compilation.metadata["request_digest"],
                                  target_identity=compilation.metadata["target_identity"])
        plan = json.loads((compilation.directory / "execution_plan.json").read_text())
        binary = (compilation.directory / "program.bin").read_bytes()
        self.check_plan(plan, binary)
        root = compilation.directory / "call"
        root.mkdir()
        sdk, platform = root / "sdk", root / "platform"
        sdk.mkdir()
        platform.mkdir()
        selected_files = dict(self._files)
        for name in ("atlas_bf16_call.c", "atlas_bf16_call.h", "atlas_host.c", "atlas_host.h", "atlas_model_tile.h"):
            shutil.copyfile(self.runtime / name, sdk / name)
            if _hash(sdk / name) != selected_files[self.runtime / name]:
                raise ValueError("BF16 runtime source changed while copying")
        shutil.copyfile(self.headers / "finite_float_boundary.h", sdk / "finite_float_boundary.h")
        if _hash(sdk / "finite_float_boundary.h") != selected_files[self.headers / "finite_float_boundary.h"]:
            raise ValueError("BF16 generic numerical header changed while copying")
        for artifact in self.platform.artifacts:
            destination = platform / artifact.path.relative_to(self.platform.directory)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact.path, destination)
            if _hash(destination) != artifact.sha256:
                raise ValueError("BF16 platform changed while copying")
        (root / "program.h").write_text(
            render_u32_payload("atlas_region_words", binary), encoding="utf-8", newline="\n")
        header, wrapper = self._sources(plan, len(binary) // 4)
        (root / "region.h").write_text(header, encoding="utf-8", newline="\n")
        (root / "region.c").write_text(wrapper, encoding="utf-8", newline="\n")
        sources = [root / "region.c", sdk / "atlas_bf16_call.c", sdk / "atlas_host.c"]
        links = []
        for name in self.link_inputs:
            path = platform / name
            if path.suffix == ".c":
                sources.append(path)
            else:
                links.append(path)
        inputs = tuple(path for path in root.rglob("*") if path.is_file())
        pinned_inputs = tuple(Artifact.from_file(path, kind="data", role="compile-input") for path in inputs)
        built = []
        def check():
            self.revalidate()
            compilation.verify()
            for artifact in (*pinned_inputs, *built):
                artifact.verify()
        for index, source in enumerate(sources):
            check()
            output = root / f"part-{index}.o"
            declaration = (["-include", platform / self.platform_header] if source.is_relative_to(platform) else [])
            process.run([self.compiler, *self.flags, "-I" + str(root), "-I" + str(sdk), "-I" + str(platform),
                         *declaration, "-c", source, "-o", output], environment=dict(os.environ), timeout_limit_s=60,
                        error=RuntimeError, inputs=inputs, outputs=(output,), stage="bf16_call_compile")
            built.append(Artifact.from_file(output, kind="object", role="call-product"))
            links.append(output)
            check()
        record = {"schema": "atlas.bf16-call.v1", "configuration": json.loads(self._settings),
                  "configuration_sha256": self.identity,
                  "selected_sources": [{"path": str(p), "sha256": sha} for p, sha in self._files],
                  "actual_objects": [{"path": a.path.relative_to(root).as_posix(), "sha256": a.sha256}
                                     for a in built],
                  "scope": "compiled adapter and resource assertions; platform execution/hardware unqualified",
                  "source_closure": "selected direct inputs; not loaded bytecode or complete toolchain attestation"}
        (root / "configuration.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
        # Retain all actual sources, compiler invocations and object products.
        artifacts = tuple(Artifact.from_file(path, kind="object" if path.suffix == ".o" else
                          "c-header" if path.suffix == ".h" else "c-source" if path.suffix == ".c" else "data",
                          role="call-product") for path in sorted(root.rglob("*")) if path.is_file())
        ports = (TensorPort((32, 32), "bf16"),)
        call = CRegionABI("atlas_region_call", ports, ports, 2,
                         tuple(path.relative_to(compilation.directory).as_posix() for path in links),
                         payloads=("program.bin",), resource_kind=resource_kind)
        abi = CRegionABI2(call, self.context, (EmbeddedPayload("program.bin", "call/program.h", "atlas_region_words"),))
        result = replace(compilation, artifacts=(*compilation.artifacts, *artifacts), metadata={**compilation.metadata,
            "c_call": abi.record(), "call_configuration_sha256": self.identity,
            "call_scope": "compiled_raw_bf16_adapter; platform_init_execution_and_device_qualification_unobserved",
            "device_pre_reservations": self._document,
            "physical_layout_contract": BF16_DRAM_PAIR_LAYOUT,
            "runtime_domain_guard": "compiled_before_io; execution_unobserved",
            "platform_obligations": ["IO", "completion", "cache", "DMA_reachability", "exclusive_device_control"]})
        CRegionABI2.from_compilation(result)
        check()
        return result

    def _sources(self, plan, words):
        c = self.context
        d = self._document
        alignment = max(d["input"]["alignment"], d["output"]["alignment"])
        if any(d[name]["address"] % alignment for name in ("input", "output")):
            raise ValueError("BF16 physical slots do not meet the selected common device alignment")
        header = f"""#ifndef ATLAS_REGION_CALL_H
#define ATLAS_REGION_CALL_H
#include "sdk/atlas_bf16_call.h"
int {c.initialize}(atlas_bf16_call *, void *, size_t);
int atlas_region_call(void *, const void *const *, void *const *);
atlas_host_result atlas_region_drain(atlas_bf16_call *, size_t);
#endif
"""
        source = f"""#include "region.h"
#include "program.h"
#include "platform/{self.platform_header}"
#include <stdatomic.h>
typedef struct {{ atomic_flag lock; atlas_bf16_call *owner; }} region_control;
_Static_assert(sizeof(atlas_bf16_call)+sizeof(region_control)<={c.state_bytes}ULL, "selected context state ceiling");
_Static_assert(_Alignof(atlas_bf16_call)<={c.state_alignment}ULL, "selected context alignment");
_Static_assert({TILE_BYTES}ULL<={c.workspace_bytes}ULL, "selected workspace ceiling");
_Static_assert(sizeof(atlas_region_words)<={d['map']['imem_capacity_bytes']}ULL, "selected program storage");
static const atlas_bf16_program program = {{atlas_region_words,{words}u,32u,32u,16u,{TILE_BYTES}u,
 {d['input']['address']}u,{d['output']['address']}u,{alignment}u}};
static region_control control = {{ATOMIC_FLAG_INIT,0}};
int {c.initialize}(atlas_bf16_call *context, void *workspace, size_t bytes) {{
 if(!context || !workspace || bytes<{TILE_BYTES}u || (uintptr_t)context%{c.state_alignment}u ||
    (uintptr_t)workspace%{c.workspace_alignment}u) return ATLAS_MODEL_TILE_INVALID;
 if(atomic_flag_test_and_set(&control.lock)) return ATLAS_MODEL_TILE_DEVICE_ERROR;
 int result=ATLAS_MODEL_TILE_INVALID;
 if(!control.owner) {{
  result={self.platform_initialize}(context,&program,workspace,bytes);
  if(!result && (!context->initialized || context->program!=&program || context->workspace!=workspace ||
      context->workspace_bytes!=bytes || context->host.map.imem_base!={d['map']['imem_base']}u ||
      context->host.map.csr_base!={d['map']['csr_base']}u ||
      context->host.map.imem_capacity_bytes!={d['map']['imem_capacity_bytes']}u ||
      (context->host.memory.coherent!=0 && context->host.memory.coherent!=1))) result=ATLAS_MODEL_TILE_INVALID;
  if(!result) control.owner=context;
 }}
 atomic_flag_clear(&control.lock); return result;
}}
int atlas_region_call(void *context, const void *const *inputs, void *const *outputs) {{
 if(atomic_flag_test_and_set(&control.lock)) return ATLAS_MODEL_TILE_DEVICE_ERROR;
 int result=context==control.owner && control.owner ?
   atlas_bf16_call_run(context,inputs,outputs) : ATLAS_MODEL_TILE_INVALID;
 atomic_flag_clear(&control.lock); return result;
}}
atlas_host_result atlas_region_drain(atlas_bf16_call *context, size_t polls) {{
 if(atomic_flag_test_and_set(&control.lock)) return ATLAS_HOST_BUSY;
 atlas_host_result result=context==control.owner && control.owner ?
   atlas_bf16_call_drain(context,polls) : ATLAS_HOST_INVALID;
 atomic_flag_clear(&control.lock); return result;
}}
"""
        return header, source
