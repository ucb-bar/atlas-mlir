"""Share exact-float/quantizer C bodies after fresh source and native replay.

The public native paths continue to identify the original stored template.
Regenerated temporary metadata proves the current source boundary separately.
"""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from merlin.compiler.build.materialization import canonical_request
from merlin.compiler.semantic.model import KernelRequest

from .application_materialization import bind_exact_float, bind_quantizer, bind_layout
from .options import current_codegen_options


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def bind_replayed_loop_region(adapter, representative, region, alternative, nodes, segments, target_source, *, state):
    """Return current ports/metadata sharing the checked original body, or decline.

    ``state`` is the invocation-private receipt captured at original export. An
    arbitrary package description cannot establish the options used at export.
    Replay runs the existing verifier/emitter, but never selects or compiles a
    source. The generated files must agree byte for byte with the template.
    """
    kind = dict(getattr(alternative, "metadata", ())).get("kind")
    strategies = {
        "exact_float_device": (bind_exact_float, "float", "float-plan.json"),
        "activation_quantize_device": (bind_quantizer, "quantizer", "quantizer-plan.json"),
        "layout_device": (bind_layout, "layout", "execution_plan.json"),
    }
    if (kind not in strategies or representative.description.get("kind") != kind
            or representative.description.get("resident_bindings") or not isinstance(state, dict)
            or representative.files is not state.get("files")):
        return None
    binder, stem, leaf_plan = strategies[kind]
    options = asdict(current_codegen_options())
    options.pop("resident_bindings", None)
    if options != state.get("options"):
        return None
    function = representative.description.get("source_binding_function", "")
    suffix = "_source_bind"
    if not isinstance(function, str) or not function.endswith(suffix):
        return None
    name = function[:-len(suffix)]
    if not name.isascii() or not name.isidentifier():
        return None
    segments = Path(segments)
    root = segments / name
    if (segments.is_symlink() or root.is_symlink() or root.resolve() != state.get("root")
            or not root.resolve().is_relative_to(segments.resolve())):
        raise ValueError("replayed loop template path differs from original export")
    manifest_member = stem+"-manifest.json"
    schedule_member = stem+"-schedule.json"
    if kind == "layout_device":
        prefix = (root / "layout-prefix-manifest.json").exists()
        manifest_member = "layout-prefix-manifest.json" if prefix else "manifest.json"
        schedule_member = "layout-prefix-schedule.json" if prefix else "execution_plan.json"
    if any((root / member).is_symlink() or not (root / member).is_file()
           for member in ("source-request.json", manifest_member)):
        raise ValueError("replayed loop template metadata absent or linked")
    original = KernelRequest.from_record(json.loads((root / "source-request.json").read_bytes()))
    manifest_hash = sha256((root / manifest_member).read_bytes()).hexdigest()
    if (manifest_hash != state.get("native_manifest_sha256")
            or manifest_hash != representative.description.get("native_manifest_sha256")
            or original.digest() != state.get("source_digest")):
        raise ValueError("replayed loop template publication identity differs")
    bindings = current_codegen_options().resident_bindings or {}
    if bindings.get(original.digest()) or bindings.get(region.digest()):
        return None
    if (canonical_request(original, source_identity_policy="provenance").record()
            != canonical_request(region, source_identity_policy="provenance").record()):
        return None
    correspondence = dict(zip((n.id for n in original.nodes), (n.id for n in region.nodes), strict=True))
    if dict(nodes) != correspondence:
        return None
    try:
        with TemporaryDirectory(prefix="atlas-replayed-loop-binding-") as temporary:
            temporary = Path(temporary)
            binder(region, original, root, temporary / name)
            # Export performs fresh source/machine/binary replay. Restore the
            # original private receipt after the temporary export captures its
            # own state; otherwise a second trip would point at deleted files.
            try:
                package = adapter.export_region(region, alternative, name, temporary, target_source)
            finally:
                adapter._loop_body_packages[representative.symbol] = state
            if (package.description.get("resident_bindings")
                    or any(getattr(package, field) != getattr(representative, field)
                           for field in ("files", "symbol", "header", "shared_files", "programs"))
                    or package.inputs != tuple(correspondence[key] for key in representative.inputs)
                    or package.outputs != tuple(correspondence[key] for key in representative.outputs)):
                return None
            current_root = temporary / name
            manifest = json.loads((current_root / manifest_member).read_bytes())
            metadata = {member: sha256((current_root / member).read_bytes()).hexdigest()
                        for member in ("source-request.json", schedule_member, manifest_member)}
            for row in manifest.get("segments", []):
                for member in ("manifest.json", "source-request.json", leaf_plan, "execution_plan.json"):
                    relative = row["path"] + "/" + member
                    metadata[relative] = sha256((current_root / relative).read_bytes()).hexdigest()
            proof = {
                "schema": ("atlas.exact_float_loop_source_binding.v1" if stem == "float"
                           else "atlas.activation_quantize_loop_source_binding.v1" if stem == "quantizer"
                           else "atlas.layout_loop_source_binding.v1"),
                "source_request_digest": region.digest(),
                "template_source_request_digest": original.digest(),
                "template_native_manifest_sha256": manifest_hash,
                "regenerated_metadata_sha256": sha256(_encoded(metadata)).hexdigest(),
                "codegen_options_sha256": sha256(_encoded(options)).hexdigest(),
                "native_paths": "original stored template; current source metadata verified separately",
            }
            return replace(package, description={**package.description,
                "native_manifest_sha256": manifest_hash, "source_binding_proof": proof})
    except ValueError:
        # A canonical match is a candidate, not permission to share changed
        # machine plans, geometry, options or generated C.
        return None
