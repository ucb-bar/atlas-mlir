"""Receipt-bound captured buffers keep explicit state and independent authority."""

import json
import struct
import zipfile
from hashlib import sha256
from types import SimpleNamespace

import pytest
from atlas_mlir_native.binding import TARGET_IDENTITY, AtlasNativeBinding
from atlas_mlir_native.generated_export_policy import derive_generated_export_policy
from merlin.compiler.frontend.inputs import stage_compile_inputs
from merlin.compiler.semantic.target_binding import NativeCompilationError
from merlin.development.targetgen.execution.native.native_application import _source
from test_export_contract import exported
from test_generated_policy import bundle


def _write(path, value):
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    path.write_bytes(raw)
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def _refresh_capture(capture, quantization):
    """Reissue every affected producer join after constructing a synthetic capture."""
    source_path = capture / "model.mlir"
    source = source_path.read_text()
    prefix, marker, tail = source.partition('prov.quantization_manifest_sha256 = "')
    assert marker
    _, quote, suffix = tail.partition('"')
    digest = sha256(
        json.dumps(quantization, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_path.write_text(prefix + marker + digest + quote + suffix)
    _write(capture / "quantization-manifest.json", quantization)
    trace = json.loads((capture / "frontend-trace.json").read_text())
    trace["mlir"]["sha256"] = sha256(source_path.read_bytes()).hexdigest()
    trace_entry = _write(capture / "frontend-trace.json", trace)
    metadata = json.loads((capture / "meta.json").read_text())
    metadata["frontend_trace"]["sha256"] = trace_entry["sha256"]
    metadata["input_abi"] = metadata["input_abi"][:2]
    _write(capture / "meta.json", metadata)
    receipt = json.loads((capture / "capture_receipt.json").read_text())
    receipt["materialized_abi"]["inputs"] = 2
    names = {*receipt["artifacts"], "input_order.json"}
    receipt["artifacts"] = {
        name: {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
        for name in sorted(names)
        for raw in [(capture / name).read_bytes()]
    }
    _write(capture / "capture_receipt.json", receipt)


def _buffer_capture(tmp_path, *, offline=True, bad_record=None, mutation=None):
    producer = exported()
    if offline:
        producer["sites"]["projection"]["weight"]["mode"] = "static (offline)"
    _, _, _, capture = bundle(tmp_path, producer_policy=producer)
    # High-bit U8 codes and signed exponents make any accidental conversion
    # observable when the native state boundary returns the initial raw bytes.
    codes = bytes((0, 8, 56, 184, 126, 254)) * 16
    exponents = struct.pack("<iii", -1, 0, 1)
    header = json.dumps(
        {
            "captured.codes": {
                "dtype": "U8",
                "shape": [3, 32],
                "data_offsets": [0, len(codes)],
            },
            "captured.exponents": {
                "dtype": "I32",
                "shape": [3, 1],
                "data_offsets": [len(codes), len(codes) + len(exponents)],
            },
        }
    ).encode()
    header += b" " * (-len(header) % 8)
    _write(
        capture / "weights.safetensors",
        struct.pack("<Q", len(header)) + header + codes + exponents,
    )
    _write(
        capture / "weights.safetensors.manifest.json",
        {
            "0": {"kind": "input", "name": "lhs"},
            "1": {"kind": "input", "name": "lhs_exponent"},
            "2": {
                "kind": "buffer",
                "weight": "captured.codes",
                "dtype": "uint8",
                "shape": [3, 32],
            },
            "3": {
                "kind": "buffer",
                "weight": "captured.exponents",
                "dtype": "int32",
                "shape": [3, 1],
            },
        },
    )
    _write(capture / "input_order.json", {"lhs": 0, "lhs_exponent": 1})
    quantization = json.loads((capture / "quantization-manifest.json").read_text())
    quantization["atlas"]["buffers"] = {
        "captured.codes": {
            "source_name": "codes",
            "dtype": "uint8",
            "shape": [3, 32],
            "bytes": len(codes),
            "sha256": sha256(codes).hexdigest(),
        },
        "captured.exponents": {
            "source_name": "exponents",
            "dtype": "int32",
            "shape": [3, 1],
            "bytes": len(exponents),
            "sha256": sha256(exponents).hexdigest(),
        },
    }
    if bad_record is not None:
        quantization["atlas"]["buffers"]["captured.codes"].update(bad_record)
    if mutation is not None:
        outputs = [{"user_output": {"arg": {"as_tensor": {"name": "result"}}}}]
        if mutation:
            outputs.append(
                {
                    "buffer_mutation": {
                        "arg": {"name": "updated_codes"},
                        "buffer_name": "codes",
                    }
                }
            )
        saved = capture.parent / "fx" / "model.pt2"
        with zipfile.ZipFile(saved, "w") as archive:
            archive.writestr(
                "bundle/models/program.json",
                json.dumps(
                    {
                        "schema_version": {"major": 8, "minor": 17},
                        "graph_module": {
                            "signature": {
                                "input_specs": [
                                    {
                                        "buffer": {
                                            "arg": {"name": "b_codes"},
                                            "buffer_name": "codes",
                                            "persistent": True,
                                        }
                                    },
                                    {
                                        "buffer": {
                                            "arg": {"name": "b_exponents"},
                                            "buffer_name": "exponents",
                                            "persistent": True,
                                        }
                                    },
                                    {
                                        "user_input": {
                                            "arg": {"as_tensor": {"name": "lhs"}}
                                        }
                                    },
                                    {
                                        "user_input": {
                                            "arg": {
                                                "as_tensor": {"name": "lhs_exponent"}
                                            }
                                        }
                                    },
                                ],
                                "output_specs": outputs,
                            }
                        },
                    }
                ),
            )
        saved_entry = {
            "bytes": saved.stat().st_size,
            "sha256": sha256(saved.read_bytes()).hexdigest(),
        }
        manifest_path = saved.parent / "fx_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["model.pt2"] = saved_entry
        _write(manifest_path, manifest)
        quantization["atlas"]["saved_native_program"]["sha256"] = saved_entry["sha256"]
    _refresh_capture(capture, quantization)
    return capture.parent / "export", capture, (codes, exponents)


def _native_source(tmp_path, capture, policy):
    document = tmp_path / "precision-policy.json"
    _write(document, policy)
    state_file = tmp_path / "state.json"
    _write(state_file, policy["application_state"])
    staged = tmp_path / "staged"
    stage_compile_inputs(capture, staged, precision_policy=document)
    args = SimpleNamespace(
        compile_inputs=staged,
        mode="diagnostic",
        linalg_entry=None,
        application_parameters="external",
        application_state=state_file,
    )
    return _source(args, AtlasNativeBinding(), TARGET_IDENTITY)


def test_explicit_offline_buffers_remain_state_with_unchanged_native_initializers(
    tmp_path,
):
    export, capture, payloads = _buffer_capture(tmp_path / "bundle")
    policy = derive_generated_export_policy(export, capture)
    declaration = policy["application_state"]
    assert [
        (port["argument_index"], port["mode"], port["initialization"])
        for port in declaration["ports"]
    ] == [(2, "read_only", "captured"), (3, "read_only", "captured")]
    assert {
        port["authority"] for port in policy["readonly_state_evidence"]["ports"]
    } == {"authenticated_explicit_offline_weight_role"}
    staged, source, adapter, state, _, parameters = _native_source(
        tmp_path, capture, policy
    )
    assert [row["kind"] for row in staged.argument_bindings] == [
        "input",
        "input",
        "buffer",
        "buffer",
    ]
    assert source.constants == () and parameters is None
    assert source.node("arg2").effect == source.node("arg3").effect == "input"
    assert tuple(port.initial for port in state.ports) == payloads
    assert tuple(port.type.dtype for port in state.ports) == ("i8", "i32")
    assert [
        dict(region.metadata)["kind"] for region in adapter.plan(source).selected
    ] == ["already_quantized"]


@pytest.mark.parametrize(
    ("options", "reason"),
    (
        ({"offline": False}, "no independent read-only declaration"),
        (
            {"bad_record": {"sha256": "0" * 64}},
            "buffer bytes, shape or storage dtype differ",
        ),
        (
            {"bad_record": {"source_name": "unbound_target"}},
            "no independent read-only declaration",
        ),
    ),
)
def test_reissued_receipts_cannot_invent_buffer_readonly_authority(
    tmp_path, options, reason
):
    export, capture, _ = _buffer_capture(tmp_path / "bundle", **options)
    with pytest.raises(NativeCompilationError, match=reason):
        derive_generated_export_policy(export, capture)


def test_saved_mutation_signature_overrides_explicit_offline_weight_role(tmp_path):
    export, capture, _ = _buffer_capture(tmp_path / "readonly", mutation=False)
    policy = derive_generated_export_policy(export, capture)
    assert {
        port["authority"] for port in policy["readonly_state_evidence"]["ports"]
    } == {"authenticated_saved_program_signature"}
    export, capture, _ = _buffer_capture(tmp_path / "mutated", mutation=True)
    with pytest.raises(
        NativeCompilationError,
        match="saved program does not declare a read-only buffer",
    ):
        derive_generated_export_policy(export, capture)


@pytest.mark.parametrize("supplied", (False, True))
def test_existing_prepare_forwards_captured_state_for_default_and_supplied_policy(
    tmp_path, monkeypatch, supplied
):
    from atlas_mlir import driver, export_intake
    from atlas_mlir_native import export_loader

    export, capture, _ = _buffer_capture(tmp_path / "bundle")
    fixture = SimpleNamespace(inputs=(), outputs=())
    authenticated = SimpleNamespace(
        export_dir=export,
        root=export.parent,
        fixtures=(fixture,),
        metadata={},
        inventory={},
        verify_unchanged=lambda: None,
    )
    monkeypatch.setattr(export_loader, "load_export", lambda path: authenticated)
    monkeypatch.setattr(driver, "capture", lambda *args, **kwargs: capture)
    supplied_path = None
    if supplied:
        supplied_path = tmp_path / "supplied.json"
        _write(supplied_path, derive_generated_export_policy(export, capture))
    out = tmp_path / "intake"
    out.mkdir()
    arguments = export_intake.prepare(export.parent, supplied_path, out)
    state_file = arguments[arguments.index("--application-state") + 1]
    policy_file = arguments[arguments.index("--precision-policy") + 1]
    policy = json.loads(policy_file.read_text())
    assert json.loads(state_file.read_text()) == policy["application_state"]
    assert [
        port["argument_index"] for port in policy["application_state"]["ports"]
    ] == [2, 3]
    assert arguments[arguments.index("--application-parameters") + 1] == "external"
