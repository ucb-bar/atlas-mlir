#!/usr/bin/env python3
"""Prepare exact ResNet C54 RTL launch inputs and optionally link its native driver.

This command never launches the linked driver or an instruction model. The caller
supplies all source, model-evidence, toolchain and generated-RTL providers.
"""
from atlas_tooling import write_json, run_command


import argparse
import json
from pathlib import Path
import struct
import subprocess

from atlas_path_guard import checked
from atlas_tooling import Admission, sha
from atlas_path_guard import lexical_absolute_path, map_metadata_paths


def relocation_mapper(specifications):
    """Build an explicit prefix substitution with no filesystem access."""
    prefixes = {}
    for specification in specifications:
        old, separator, new = specification.partition("=")
        if not separator:
            raise ValueError("relocation requires OLD_PREFIX=NEW_PREFIX")
        old, new = lexical_absolute_path(old), lexical_absolute_path(new)
        if old in prefixes:
            raise ValueError("duplicate relocation source prefix")
        prefixes[old] = new

    def relocate(path):
        for old in sorted(prefixes, key=lambda p: len(p.parts), reverse=True):
            if path.is_relative_to(old):
                return prefixes[old] / path.relative_to(old)
        return path

    return relocate if prefixes else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("host", "model-partition", "reference", "templates", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--model-producer-tools", type=Path,
        help="Source snapshot bound by the model receipts; defaults to this tools directory")
    parser.add_argument("--rtl-object-dir", type=Path)
    parser.add_argument("--verilator-include", type=Path)
    parser.add_argument("--cc", type=Path)
    parser.add_argument("--cxx", type=Path)
    parser.add_argument("--relocate", action="append", default=[], metavar="OLD_PREFIX=NEW_PREFIX")
    args = parser.parse_args()
    producer_tools = checked(args.model_producer_tools or Path(__file__).absolute().parent)
    mapper = relocation_mapper(args.relocate)
    mapped_paths = {}
    for value in vars(args).values():
        if isinstance(value, Path):
            checked(value)
    build_requested = any((args.rtl_object_dir, args.verilator_include, args.cc, args.cxx))
    if build_requested and not all((args.rtl_object_dir, args.verilator_include, args.cc, args.cxx)):
        parser.error("linking requires --rtl-object-dir, --verilator-include, --cc and --cxx")
    if args.out.exists():
        raise ValueError("fresh output required")
    ad = Admission()
    ad.read(Path(__file__).absolute())
    ad.read(Path(checked.__code__.co_filename).absolute())
    ad.read(Path(Admission.read.__code__.co_filename).absolute())
    ad.read(Path(map_metadata_paths.__code__.co_filename).absolute())
    host = json.loads(ad.read(args.host / "build.json"))
    if (host["returncode"] or not host["source_bookends_equal"] or host["kernel_count"] != 54
        or host["captured_successor_A_embedded"]):
        raise ValueError("complete C54 host build required")
    for field in ("inputs_sha256", "generated_inputs_sha256"):
        for path, digest in host[field].items():
            ad.read(Path(path), digest)
    ad.read(args.host / "host.elf", host["elf_sha256"])
    model = map_metadata_paths(json.loads(ad.read(args.model_partition)), mapper, mapped_paths=mapped_paths)
    reference = json.loads(ad.read(args.reference / "receipt.json"))
    templates = json.loads(ad.read(args.templates))["kernels"]
    calls = model.get("invocations", [])
    if (model.get("status") != "passed_model_partition" or len(calls) != 54 or len(reference["kernels"]) != 54
        or len(templates) != 54 or model["candidate_logits"] != reference["final_logits"]):
        raise ValueError("complete matching 54-kernel model/reference required")
    args.out.mkdir(parents=True)
    plan, cases = ["atlas_resnet_graph_v1"], []
    for index, (call, ref, template) in enumerate(zip(calls, reference["kernels"], templates, strict=True)):
        if (call["site"] != ref["site"] or call["site"] != template["site"] or call["activation"] != ref["activation"]
            or call["target_output"] != ref["target_output"] or not call["exact_target_gate"] or call["returncode"]):
            raise ValueError("ordered model/reference binding differs")
        receipt_path = checked(Path(call["receipt"]["path"]))
        provider = receipt_path.parent
        receipt = map_metadata_paths(json.loads(ad.read(receipt_path, call["receipt"]["sha256"])), mapper,
            mapped_paths=mapped_paths)
        if (receipt["status"] != "passed_model_only" or not receipt["runs"]
            or not all(r["all_output_bytes_exact"] and r["input_and_guards_preserved"]
            and r["drained_before_halt_observation"] for r in receipt["runs"])):
            raise ValueError("model child did not pass exact/guard/drain gates")
        bookends = receipt["source_bookends"]
        if (not bookends["unchanged"] or bookends["input_before"] != bookends["input_after"]
            or bookends["model_after"] != receipt["model_binding_before"] or not receipt["code_before"]
            or receipt["code_before"] != receipt["code_after"]):
            raise ValueError("model child source/input bookends differ")
        helpers = {"run_workload_contraction.py", "atlas_vector_arithmetic.py", "atlas_program.py", "run_model_tile.py",
            "atlas_tile_arithmetic.py"}
        recorded_helpers = helpers - {"atlas_program.py"} | {"run_model_program_contraction.py"}
        if set(receipt["code_before"]) not in (helpers, recorded_helpers):
            raise ValueError("incomplete model producer source closure")
        for name, digest in receipt["code_before"].items():
            ad.read(producer_tools / name, digest)
        artifacts = receipt.get("artifacts", receipt.get("checksums", {}))
        payload = {name: ad.read(provider / name, artifacts[name]) for name in ("program.bin", "initial-dram.bin",
            "expected-bf16-tiles.bin")}
        actual = ad.read(provider / "actual-bf16-halves-0.bin", artifacts["actual-bf16-halves-0.bin"])
        ad.read(provider / "result.npz", artifacts["result.npz"])
        a = ad.read(Path(ref["encoded_a_tiled"]["path"]), ref["encoded_a_tiled"]["sha256"])
        expected = ad.read(Path(ref["expected_physical_output"]["path"]), ref["expected_physical_output"]["sha256"])
        weight = ad.read(Path(template["weight"]["fp8_tiled"]["path"]), template["weight"]["fp8_tiled"]["sha256"])
        initial, memory = payload["initial-dram.bin"], receipt["memory"]
        if (actual != expected or expected != payload["expected-bf16-tiles.bin"] or initial[: len(a)] != a
            or initial[len(a) : len(a) + len(weight)] != weight):
            raise ValueError("model/reference physical bytes differ")
        raw = bytearray(ad.read(Path(template["program"]["path"]), template["program"]["sha256"]))
        exponent = ref["a_scale_exponent"] + template["b_scale_exponent"]
        if not -126 <= exponent <= 127:
            raise ValueError("rescale exponent outside qualified table")
        struct.pack_into("<I", raw, template["rescale_patch_word_index"] * 4, ((exponent + 127) << 23) | 0x25F)
        if bytes(raw) != payload["program.bin"]:
            raise ValueError("regenerated template differs from actual model program")
        final = initial[: memory["output_offset"]] + expected + initial[memory["output_offset"] + len(expected) :]
        counts, guard = receipt["requested_dma"], b"\x69" * 32
        source = ad.read(provider / "program.S", artifacts["program.S"])
        if b"0(x0)" in source:
            raise ValueError("reserved VMEM guard unavailable")
        image = f"dma {counts['load_bytes']//32} {counts['store_bytes']//32}\n"
        if index == 0:
            image += "vmem 0 " + guard.hex() + "\n"
        image += (
            f"dram {memory['dram_base']} {initial.hex()}\n" f"expect_dram {memory['dram_base']} {
            final.hex()}\n" f"expect_vmem 0 {guard.hex()}\n")
        image_path, program_path = args.out / f"image-{index}.txt", args.out / f"program-{index}.bin"
        image_path.write_text(image)
        program_path.write_bytes(payload["program.bin"])
        plan.append(str(image_path) + " " + str(program_path))
        cases.append({"index": index, "site": call["site"], "model_receipt": call["receipt"], "memory": memory,
            "model_cycles": call["cycles"], "image_sha256": sha(image.encode()), "program_sha256": sha(bytes(raw)),
            "a_scale_exponent": ref["a_scale_exponent"], "requested_dma": counts})
    plan_path = args.out / "graph.plan"
    plan_path.write_text("\n".join(plan) + "\n")
    commands, results = [], []
    if build_requested:
        for tool in (args.cc, args.cxx):
            ad.read(tool)
        objects = []
        sources = [args.host / name for name in ("runtime/atlas_host.c", "runtime/atlas_resnet_graph_math.c",
            "runtime/atlas_resnet_graph_device.c", "assets.c", "graph/graph.c", "graph/constants.S", "images.S")]
        for index, source in enumerate(sources):
            ad.read(source)
            obj = args.out / f"source-{index}.o"
            objects.append(str(obj))
            command = [str(args.cc), "-O2", "-ffp-contract=off", "-fno-fast-math", "-Wall", "-Wextra", "-Werror",
                "-I" + str(args.host / "runtime"), "-I" + str(args.host)]
            if source.suffix != ".S":
                command.append("-std=c11")
            commands.append(command + ["-c", str(source), "-o", str(obj)])
        rtl = args.rtl_object_dir
        libraries = [rtl / name for name in ("verilated.o", "verilated_vcd_c.o", "verilated_threads.o", "VAtlasCore__ALL.a")]
        for path in libraries:
            ad.read(path)
        driver = Path(__file__).absolute().parent / "rtl_core/resnet_graph_runner.cpp"
        ad.read(driver)
        commands.append([str(args.cxx), "-std=c++17", "-O2", "-DVM_TRACE=1", "-DVM_TRACE_VCD=1", "-DVM_TRACE_FST=0",
            "-I" + str(rtl), "-I" + str(args.verilator_include), "-I" + str(args.verilator_include / "vltstd"),
            "-I" + str(args.host / "runtime"), "-I" + str(args.host),
            str(driver)] + [str(p) for p in libraries] + objects + ["-pthread", "-latomic", "-o",
            str(args.out / "resnet_graph_runner")])
        with (args.out / "build.log").open("w") as log:
            for command in commands:
                result = run_command(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                results.append(result.returncode)
                if result.returncode:
                    break
    generated = {str(p): sha(p.read_bytes()) for p in [plan_path]
        + [args.out / f"{stem}-{i}.{extension}" for i in range(54) for stem,
        extension in (("image", "txt"), ("program", "bin"))]}
    preserved = ad.recheck()
    passed = preserved and (not build_requested or len(results) == len(commands) and not any(results))
    receipt = {"status": "PASS" if passed else "FAIL", "scope": "C54 launch-input preparation and optional native link only",
        "model_or_rtl_executed": False, "metadata_relocations": args.relocate, "inputs_sha256": ad.inputs,
        "generated_inputs_sha256": generated, "source_bookends_equal": preserved, "cases": cases, "commands": commands,
        "returncodes": results}
    if build_requested and passed:
        receipt["binary_sha256"] = sha((args.out / "resnet_graph_runner").read_bytes())
    write_json(args.out / "preparation.json", receipt)
    print(json.dumps({"status": receipt["status"], "cases": len(cases), "linked": build_requested}))
    raise SystemExit(not passed)


if __name__ == "__main__":
    main()
