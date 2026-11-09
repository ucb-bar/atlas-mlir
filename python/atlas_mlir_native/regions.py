"""Configured source offers and selected native binary lowering for Atlas.

The first family is an exact source BF16 ReLU region. Discovery performs no native
compilation and grants no input-domain, numerical or hardware qualification.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from merlin.compiler.backend.sourceproposal import (
    LoweringRecipe,
    SourceProposal,
    proposal_boundary,
    source_numerical_identity,
)
from merlin.compiler.invocation import request_identity
from merlin.compiler.ir.region import preview_region
from merlin.compiler.request import CompileOptions
from merlin.frontends.linalg_bf16_relu import recognize_bf16_sign_clamp_relu
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.snapshot import NativeSnapshot, open_native_snapshot
from merlin.transforms.certificates import ir_text
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval, RegionAlternative

from .admission import _bf16_tile_request
from .binding import AtlasNativeBinding, implementation_sources, revalidate_implementation
from .planning import admit_request
from .policy import NATIVE_PIPELINES


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _directory(value):
    if not isinstance(value, Path) or not value.is_absolute() or not value.is_dir():
        raise ValueError("Atlas region configuration requires explicit absolute existing directories")
    return value.resolve()


def _kernel(module, entry, target):
    from xdsl.dialects import func, tensor
    from xdsl.dialects.linalg.ops import GenericOp

    functions = tuple(module.body.block.ops)
    if len(functions) != 1 or type(functions[0]) is not func.FuncOp:
        raise ValueError("Atlas region family requires one complete entry without unrelated definitions")
    function = functions[0]
    if function.sym_name.data != entry or len(function.body.blocks) != 1:
        raise ValueError("Atlas region family requires the exact single-block source entry")
    if (module.properties or any(not key.startswith("prov.") for key in module.attributes)
            or any(not key.startswith("prov.") and key != "llvm.emit_c_interface" for key in function.attributes)
            or set(function.properties) - {"sym_name", "function_type", "sym_visibility", "arg_attrs", "res_attrs"}):
        raise ValueError("Atlas region family cannot drop inherited module/function policies")
    for name in ("arg_attrs", "res_attrs"):
        attributes = getattr(function, name)
        if attributes is not None and any(dictionary.data for dictionary in attributes):
            raise ValueError("Atlas region family cannot transfer source signature semantic attributes")
    operations = tuple(function.body.block.ops)
    if (len(operations) != 3 or type(operations[0]) is not tensor.EmptyOp or
            type(operations[1]) is not GenericOp or type(operations[2]) is not func.ReturnOp or
            len(function.body.block.args) != 1):
        raise ValueError("Atlas region family requires only empty, exact BF16 ReLU and return")
    empty, relu, returned = operations
    if returned.properties or any(not key.startswith("prov.") for key in returned.attributes):
        raise ValueError("Atlas region family cannot drop inherited result/return policies")
    input_value, destination = recognize_bf16_sign_clamp_relu(relu)
    if (input_value is not function.body.block.args[0] or destination is not empty.results[0] or
            tuple(returned.operands) != tuple(relu.results) or empty.operands or empty.properties or
            any(not key.startswith("prov.") for key in empty.attributes)):
        raise ValueError("Atlas region BF16 source has a different input, initializer or complete result boundary")
    translated = translate_linalg_text(ir_text(module), entry=entry, target_identity=target,
                                       lowering_policy="diagnostic").request
    if (not _bf16_tile_request(translated) or len(translated.nodes) != 2 or
            tuple(node.op for node in translated.nodes) != ("input", "relu") or
            translated.outputs != (translated.nodes[-1].id,)):
        raise ValueError("Atlas region source is outside the implemented BF16 tile/domain contract")
    return translated


class AtlasRegions:
    """An explicit native compiler configuration consumed by SystemCompiler.

    Tool and pipeline selection is per invocation; environment defaults are used
    only by the untouched ordinary backend route. Fixed ABI values retain the
    native binding's slot units. Optional caller-bound C adaptation is separate
    from image construction, platform execution and device qualification.
    """
    name = "atlas"

    def __init__(self, *, snapshot, target_source, tool_directory, pipeline, fixed_inputs, fixed_outputs, limits,
                 call=None):
        if type(snapshot) is not NativeSnapshot or type(limits) is not SearchLimits:
            raise ValueError("Atlas regions require an original native snapshot and explicit search limits")
        if type(pipeline) is not str or pipeline not in NATIVE_PIPELINES:
            raise ValueError("Atlas regions require an explicit implemented native pipeline")
        if (type(fixed_inputs) is not tuple or not fixed_inputs or
                any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not str or not row[0] or
                    type(row[1]) is not int or row[1] < 0 for row in fixed_inputs) or
                len(dict(fixed_inputs)) != len(fixed_inputs) or type(fixed_outputs) is not tuple or
                not fixed_outputs or any(type(n) is not int or n < 0 for n in fixed_outputs)):
            raise ValueError("Atlas regions require immutable named input slots and ordered output slots")
        from .call import BF16Call

        if call is not None and type(call) is not BF16Call:
            raise ValueError("Atlas callable route requires an explicit typed BF16Call")
        self.call = call
        self.snapshot, self.limits = snapshot, limits
        self.target_source, self.tool_directory = _directory(target_source), _directory(tool_directory)
        self.pipeline, self.fixed_inputs, self.fixed_outputs = pipeline, fixed_inputs, fixed_outputs
        # Discovery custody observes one stable callable object, rather than
        # Python's fresh bound-method wrapper on each attribute lookup.
        self._proposal_hook = self._propose
        self.propose = self._proposal_hook
        self._owners = (snapshot, limits, call)
        self._sources = implementation_sources()
        self._profile = AtlasNativeBinding.profile().digest()
        self._settings = self._current()
        paths = (snapshot.root / "manifest.json", snapshot.root / "profile.json", snapshot.bridge,
                 self.target_source / "docs/selected-variant-inventory.json",
                 self.tool_directory / "atlas-opt", self.tool_directory / "atlas-emit")
        if any(not p.is_file() for p in paths) or any(not os.access(p, os.X_OK) for p in paths[-2:]):
            raise ValueError("Atlas regions require actual selected snapshot, inventory and executable native tools")
        self._files = tuple((path, _sha(path)) for path in paths)
        self._identity = hashlib.sha256(_json({"configuration": self._settings, "profile": self._profile,
            "sources": self._sources, "files": [(str(p), digest) for p, digest in self._files]}).encode()).hexdigest()
        self._revalidate()

    def _current(self):
        return _json({"snapshot": str(self.snapshot.root), "snapshot_manifest": self.snapshot.manifest,
                      "target_source": str(self.target_source), "tool_directory": str(self.tool_directory),
                      "pipeline": self.pipeline, "inputs": self.fixed_inputs, "outputs": self.fixed_outputs,
                      "limits": self.limits.record(), "call": None if self.call is None else self.call.identity})

    def _revalidate(self):
        if (self.snapshot is not self._owners[0] or self.limits is not self._owners[1] or
                self.call is not self._owners[2] or self.propose is not self._proposal_hook or
                self._current() != self._settings):
            raise ValueError("Atlas region configuration changed after explicit selection")
        if self.call is not None:
            self.call.revalidate()
            self.call.check_slots(self.fixed_inputs, self.fixed_outputs)
        revalidate_implementation(self._sources)
        if any(_sha(path) != digest for path, digest in self._files):
            raise ValueError("Atlas region snapshot, target inventory or selected native tool bytes changed")
        opened = open_native_snapshot(self.snapshot.root)
        if (opened.profile.digest() != self._profile or self.snapshot.profile.digest() != self._profile or
                opened.manifest != self.snapshot.manifest or AtlasNativeBinding.profile().digest() != self._profile):
            raise ValueError("Atlas region profile differs from its original selected native snapshot")

    def _recipe(self):
        return LoweringRecipe(self._identity, "atlas.native/" + self.pipeline,
            (("configuration_sha256", self._identity), ("input_domain", "caller_premise_unchecked"),
             ("physical_abi", _json({"inputs": self.fixed_inputs, "outputs": self.fixed_outputs})),
             ("boundary_workspace", "conditional_zero; native_allocation_and_stream_checks_required"
              if self.call is None else "host_in_c.context; fixed_device_buffers_pre_reserved")),
            (("device-binary", "device-program"), ("json", "execution-plan"), ("json", "native-manifest")) +
            ((("object", "call-product"),) if self.call is not None else ()), "logical_tensor_v1")

    def _admit(self, kernel):
        return admit_request(kernel, fixed_inputs=dict(self.fixed_inputs), fixed_outputs=self.fixed_outputs,
                             target_source=self.target_source, pipeline=self.pipeline)

    def _propose(self, source, *, resource, memory_domain, preparation=None):
        self._revalidate()
        source.revalidate()
        if preparation is not None:
            preparation.revalidate_ir(source)
        metadata = (("artifact_admission", "compiled_regions_only; loader_and_physical_handoff_pending"
                     if self.call is None else "compiled_call; platform_execution_and_image_admission_pending"),
                    ("boundary_workspace", "conditional_zero; selected_native_checks_required"
                     if self.call is None else "host_in_c.context; fixed_device_buffers_pre_reserved"),
                    ("configuration_sha256", self._identity), ("input_domain", "caller_premise_unchecked"))
        if self.call is not None:
            metadata = (*metadata, self.call.context.selection_metadata(),
                        ("device_storage", "original_platform_pre_reserved_outside_available_capacity"),
                        ("call_configuration_sha256", self.call.identity))
        result = []
        # Region construction, complete live ports and independent SSA replay
        # are shared core mechanisms. The actual family/domain/physical compiler
        # contract remains this backend's responsibility.
        from xdsl.dialects import tensor
        from xdsl.dialects.linalg.ops import GenericOp

        for index, operation in enumerate(source.operations[:-1]):
            if type(operation) is not tensor.EmptyOp or type(source.operations[index + 1]) is not GenericOp:
                continue
            nodes = (index, index + 1)
            try:
                preview = preview_region(source.module, source.entry,
                    tuple(source.operations[n] for n in nodes), symbol="atlas_region")
                kernel = _kernel(preview.module, "atlas_region", self.snapshot.profile.target_identity)
                self._admit(kernel)
            except ValueError:
                continue
            if len(result) >= 64:
                raise ValueError("Atlas source region discovery exceeded its bounded candidate count")
            inputs, outputs = proposal_boundary(source, nodes)
            suffix = "whole" if nodes == tuple(range(len(source.operations))) else f"region/{index:06d}-{index+2:06d}"
            alternative = RegionAlternative(
                "atlas.bf16-relu/" + suffix, nodes, "atlas.bf16_relu.native.v1", resource,
                CycleInterval.unknown("no comparable native/host-adapter/transfer/launch/full-region cost"),
                inputs=inputs, outputs=outputs, metadata=tuple(sorted(metadata)))
            result.append(SourceProposal(alternative, self.name, resource, memory_domain, "logical_tensor_v1",
                source.stamp.source_sha256, tuple(source.operation_ordinals[n] for n in nodes),
                source_numerical_identity(source, nodes), lowering=self._recipe()))
        source.revalidate()
        self._revalidate()
        return tuple(result)

    def validate(self, request):
        self._revalidate()
        if request.source.kind != "mlir" or asdict(request.options) != asdict(CompileOptions()):
            raise ValueError("configured Atlas regions require MLIR and no additional implicit compile options")
        module = parse_mlir_text(request.source.text)
        functions = tuple(module.body.block.ops)
        if len(functions) != 1 or not hasattr(functions[0], "sym_name"):
            raise ValueError("Atlas selected source has no unique typed function entry")
        kernel = _kernel(module, functions[0].sym_name.data, self.snapshot.profile.target_identity)
        self._admit(kernel)

    def compile(self, request):
        self.validate(request)
        raise ValueError("configured Atlas region lowering requires an original selected lowering ticket")

    def validate_selected(self, request, ticket):
        self.validate(request)
        ticket.revalidate(request)
        proposal = ticket.admitted.proposal
        if (proposal.provider != self.name or proposal.lowering != self._recipe() or
                proposal.alternative.implementation != "atlas.bf16_relu.native.v1"):
            raise ValueError("Atlas selected offer differs from its original source-region configuration/recipe")
        original = ticket.partition.selection.source
        preview = preview_region(original.module, original.entry,
            tuple(original.operations[n] for n in proposal.alternative.nodes), symbol="atlas_region")
        before = _kernel(preview.module, "atlas_region", self.snapshot.profile.target_identity)
        after = _kernel(parse_mlir_text(request.source.text), ticket.entry, self.snapshot.profile.target_identity)
        if replace(before, source_identity="") != replace(after, source_identity=""):
            raise ValueError("Atlas outlined kernel changes original BF16 numerical meaning or complete ports")
        self._revalidate()
        ticket.revalidate(request)

    def compile_selected(self, request, ticket):
        from .backend import AtlasBackend

        self.validate_selected(request, ticket)
        original = request_identity(request)
        options = CompileOptions(parameters={
            "atlas.snapshot": str(self.snapshot.root), "atlas.target-source": str(self.target_source),
            "atlas.entry": ticket.entry, "atlas.fixed-inputs": dict(self.fixed_inputs),
            "atlas.fixed-outputs": list(self.fixed_outputs), "atlas.search-limits": self.limits.record(),
        })
        result = AtlasBackend.compile(replace(request, options=options), tool_directory=self.tool_directory,
                                      pipeline=self.pipeline)
        native = json.loads(next(a.path for a in result.artifacts if a.role == "native-manifest").read_text())
        if native.get("lowering_pipeline", native.get("native_pipeline")) != self.pipeline:
            raise ValueError("Atlas native result changed the explicitly selected lowering pipeline")
        if self.call is not None:
            resource = ticket.partition.selection.topology.resource(ticket.admitted.proposal.resource)
            result = self.call.build(result, resource_kind=resource.kind)
        self.validate_selected(request, ticket)
        if request_identity(request) != original:
            raise ValueError("Atlas selected lowering changed the original request")
        # Logical ports remain source-bound; physical layout/load/domain and
        # independent semantics/execution qualification remain explicit gaps.
        result = replace(result, metadata={**result.metadata,
            "region_configuration_sha256": self._identity,
            "region_scope": ("native_device_binary; no_loader_or_callable_abi" if self.call is None else
                             "native_device_binary_and_compiled_bf16_call; platform_execution_unobserved"),
            "numerical_verification": "not_established", "execution_observed": False})
        return replace(result, metadata={**result.metadata,
                                         "selected_lowering": ticket.acknowledgment(request, result)})
