"""Authenticate native reference controls without treating them as device runs."""
import hashlib
import json
import re

from bundle import relative_name
from result_lines import verify


def require(value, message):
    if not value:
        raise ValueError("Smol native preparation: "+message)


def validate(chain, data, preparation, phases, sources, compiled_layout):
    manifest = chain.read("segments/native-input-SHA256SUMS")
    require(hashlib.sha256(manifest).hexdigest() == preparation.get("native_input_checksums_sha256"),
            "input manifest identity")
    previous = {}
    for line in manifest.decode("ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "input checksum syntax")
        checksum, member = match.groups()
        relative_name(member)
        require(member not in previous and member != "SHA256SUMS", "duplicate input checksum")
        previous[member] = checksum
    require(previous and len(previous) <= 20000, "bounded input inventory")
    replacements = {"segments.json": ("segments/native-input-segments.json", "native_input_segments_sha256"),
                    "segments/receipt.json": ("segments/native-input-receipt.json", "native_input_receipt_sha256")}
    require(set(replacements) <= set(previous), "original chain and pending receipt required")
    for member, checksum in previous.items():
        if member in replacements:
            snapshot, key = replacements[member]
            require(chain.sums.get(snapshot) == checksum == preparation.get(key), "replaced metadata snapshot")
        else:
            require(chain.sums.get(member) == checksum, "control input differs from current bundle")
    old = json.loads(chain.read("segments/native-input-segments.json"))
    require({k:v for k,v in old.items() if k != "capture_receipt_sha256"}
            == {k:v for k,v in chain.data.items() if k != "capture_receipt_sha256"}, "input chain differs")
    replay = json.loads(chain.read(data["source_reference"]))
    build = json.loads(chain.read("host/build.json"))
    units = [argv[argv.index("-c")+1] for argv in build["commands"] if "-c" in argv]
    mains = [path for path in units if path.endswith("/main.c")]
    require(len(mains) == 1, "one actual guest generated main")
    work = mains[0].rsplit("/", 1)[0]
    production = {work+"/"+relative:checksum for relative,checksum in build["compilation_inputs"].items()}
    for name in ("main.c", "assets.S"):
        path = work+"/"+name
        require(path in units and path in build["source_sha256"], "actual guest main/assets compilation")
        production[path] = build["source_sha256"][path]
    predecessors = [dict(index=i, sha256=row["input"]["sha256"], dtype=row["input"]["dtype"],
                         bytes=row["input"]["bytes"]) for i,row in enumerate(replay["calls"])]
    modes = ("full", "phased", "corrupt")
    for key in ("controls", "control_compiles", "control_uart"):
        require(set(preparation.get(key, {})) == set(modes), "complete control inventory: "+key)
    for mode in modes:
        record = json.loads(chain.read(preparation["controls"][mode]))
        compiled = json.loads(chain.read(preparation["control_compiles"][mode]))
        require(record.get("status") == "PASS_NATIVE_DUAL_DRIVER_"+mode.upper()
                and record.get("guest_execution") is False and record.get("device_execution") is False
                and record.get("synthetic_counter_is_timing") is False
                and record.get("compiled_phase_layout") == list(compiled_layout)
                and compiled.get("guest_execution") is False and compiled.get("device_execution") is False
                and compiled.get("source_bookends") is True
                and compiled.get("bundle_manifest_sha256") == preparation["native_input_checksums_sha256"],
                "exact-driver scope/input/ABI binding")
        require(all(compiled["source_sha256"].get(path) == checksum for path,checksum in production.items()),
                "complete actual guest production/main/assets source binding")
        for path, checksum in compiled["source_sha256"].items():
            if path.rsplit(".", 1)[-1] in ("c", "h", "py"):
                require(sources.get(path) == checksum, "compile source snapshot coverage")
        uart = chain.read(preparation["control_uart"][mode])
        if mode != "corrupt":
            require(record.get("starts") == 2715 and record.get("predecessors") == predecessors,
                    "all1421 original predecessors")
        if mode == "full":
            require(record.get("source_bookends") is True and record.get("full_canonical_boundaries_exact") == 200
                    and record.get("full_cycle_observations_restored") is True
                    and record.get("actual_buffers") == [dict(name=r["name"], sha256=r["sha256"], exact=True)
                        for r in data["reference_buffers"]], "full traversal and all47 references")
            verify(uart, "smol-full", 2715, progress_launches=record["progress"])
            require(len(record["progress"]) == 212, "full progress inventory")
        elif mode == "phased":
            require(record.get("source_bookends_verified") is True and record.get("boundaries") == 200
                    and record.get("scratch_reset_events") == record.get("arena_reset_events") == 200
                    and record.get("scratch_fill_byte") == 165, "scratch-independent phase traversal")
            observations = record["observations"]
            require(len(observations) == 200, "all phase observations")
            transcripts = uart.split(b"ATLAS_KERNEL_BEGIN ")
            require(not transcripts[0] and len(transcripts) == 201, "all200 phase UART begins")
            for phase, observed, transcript in zip(phases, observations, transcripts[1:]):
                count = phase["launch_end"]-phase["launch_start"]
                values = observed.get("native_synthetic_protocol_deltas")
                require(observed.get("index") == phase["index"] and observed.get("measured_model_cycles") is False
                        and isinstance(values, list) and len(values) == count
                        and all(type(v) is int and 0 < v < 1 << 32 for v in values), "own-range observations")
                verify(b"ATLAS_KERNEL_BEGIN "+transcript, "smol-segment", count, progress_launches=[count])
        else:
            mailbox = record.get("negative_mailbox")
            lines = uart.decode("ascii").splitlines()
            require(record.get("source_bookends_verified") is True
                    and record.get("starts") == record.get("boundaries") == 0
                    and isinstance(mailbox, list) and len(mailbox) == 16 and mailbox[0] == 0x534D4F4C46554C4C
                    and mailbox[1] == 3 and mailbox[8] == 11 and record.get("negative_tohost") == 23
                    and len(lines) == 2 and lines[0].startswith("ATLAS_KERNEL_BEGIN name=smol-segment ")
                    and lines[1].startswith("ATLAS_KERNEL_FAIL name=smol-segment ")
                    and " check=smol_v_tokens " in lines[1]
                    and " launches_completed=0 " in lines[1], "actual live-byte UART failure")
