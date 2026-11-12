"""Public Merlin adapter for diagnostic Atlas command-program compilation.

The output is a device program and its explicit execution plan, not a host
image. Numerical preparation and hardware qualification remain separate.
"""
from __future__ import annotations

import json
from pathlib import Path

from merlin.compiler.artifact import Artifact, Compilation
from merlin.compiler.request import CompileRequest
from merlin.compiler.semantic.linalg_bridge import translate_linalg_text
from merlin.compiler.semantic.model import KernelRequest
from merlin.compiler.semantic.search import SearchLimits
from merlin.compiler.semantic.snapshot import open_native_snapshot
from merlin.compiler.semantic.target_binding import verify_native_publication

from .binding import AtlasNativeBinding


class AtlasBackend:
    name = "atlas"
    _required = {"atlas.snapshot", "atlas.target-source", "atlas.fixed-inputs", "atlas.fixed-outputs"}
    _optional = {"atlas.entry", "atlas.search-limits"}

    @staticmethod
    def configure(*, snapshot, target_source, tool_directory, pipeline, fixed_inputs, fixed_outputs, limits, call=None):
        """Bind explicit native configuration for source proposals and selected lowering."""
        from .regions import AtlasRegions

        return AtlasRegions(snapshot=snapshot, target_source=target_source, tool_directory=tool_directory,
                            pipeline=pipeline, fixed_inputs=fixed_inputs, fixed_outputs=fixed_outputs,
                            limits=limits, call=call)

    @classmethod
    def validate(cls, request: CompileRequest) -> None:
        options = request.options
        if any(value is not None for value in (options.system, options.platform, options.harts)):
            raise ValueError("Atlas command programs do not implement host-image deployment options")
        parameters = options.parameters
        if not cls._required <= parameters.keys() <= cls._required | cls._optional:
            raise ValueError("Atlas needs snapshot, target-source, fixed-inputs and fixed-outputs; "
                             "unknown options refuse")
        for key in ("atlas.snapshot", "atlas.target-source"):
            if not isinstance(parameters[key], str) or not Path(parameters[key]).is_dir():
                raise ValueError(f"{key} must name an existing directory")
        inputs, outputs = parameters["atlas.fixed-inputs"], parameters["atlas.fixed-outputs"]
        if (not isinstance(inputs, dict) or not inputs or
            any(not isinstance(key, str) or not key or type(value) is not int or value < 0
                for key, value in inputs.items()) or not isinstance(outputs, list) or not outputs or
            any(type(value) is not int or value < 0 for value in outputs)):
            raise ValueError("Atlas ABI needs named nonnegative input addresses and ordered output addresses")
        if request.source.kind == "mlir":
            if not isinstance(parameters.get("atlas.entry"), str) or not parameters["atlas.entry"]:
                raise ValueError("Atlas MLIR input needs atlas.entry")
        elif request.source.kind == "merlin.kernel-request":
            if "atlas.entry" in parameters:
                raise ValueError("a typed kernel request already declares its complete boundary")
        else:
            raise ValueError("Atlas accepts admitted MLIR or merlin.kernel-request source")
        if "atlas.search-limits" in parameters:
            SearchLimits.from_record(parameters["atlas.search-limits"])

    @classmethod
    def compile(cls, request: CompileRequest, *, tool_directory=None, pipeline=None) -> Compilation:
        cls.validate(request)
        parameters = request.options.parameters
        snapshot = open_native_snapshot(Path(parameters["atlas.snapshot"]))
        if snapshot.profile.digest() != AtlasNativeBinding.profile().digest():
            raise ValueError("native snapshot differs from selected Atlas profile")
        if request.source.kind == "mlir":
            kernel = translate_linalg_text(
                request.source.text, entry=parameters["atlas.entry"],
                target_identity=snapshot.profile.target_identity, lowering_policy="diagnostic",
            ).request
        else:
            kernel = KernelRequest.from_record(json.loads(request.source.text))
        if kernel.target_identity != snapshot.profile.target_identity or kernel.lowering_policy != "diagnostic":
            raise ValueError("Atlas request must select this target and explicit diagnostic semantics")
        limits = (SearchLimits.from_record(parameters["atlas.search-limits"])
                  if "atlas.search-limits" in parameters else SearchLimits())
        manifest = AtlasNativeBinding.compile(
            snapshot, kernel, fixed_inputs=parameters["atlas.fixed-inputs"],
            fixed_outputs=tuple(parameters["atlas.fixed-outputs"]),
            target_source=Path(parameters["atlas.target-source"]),
            destination=request.output, limits=limits,
            **({"tool_directory": tool_directory} if tool_directory is not None else {}),
            **({"pipeline": pipeline} if pipeline is not None else {}),
        )
        verify_native_publication(request.output, manifest, engine="merlin_native", request_digest=kernel.digest(),
                                  target_identity=kernel.target_identity)
        kinds = {"program.bin": ("device-binary", "device-program"),
                 "execution_plan.json": ("json", "execution-plan"),
                 "manifest.json": ("json", "native-manifest")}
        for relative in manifest.get("stage_artifacts", {}):
            kinds.setdefault(relative, ("mlir" if relative.endswith(".mlir") else "data", "intermediate"))
        products = tuple(Artifact.from_file(request.output / name, kind=kind, role=role)
                         for name, (kind, role) in sorted(kinds.items()))
        return Compilation(request.output, cls.name, request.source.sha256, products,
                           metadata={"qualification_scope": manifest.get("qualification_scope"),
                                     "target_identity": kernel.target_identity,
                                     "request_digest": kernel.digest(),
                                     "native_pipeline": manifest.get(
                                         "native_pipeline", manifest.get("lowering_pipeline")),
                                     "execution_observed": False})
