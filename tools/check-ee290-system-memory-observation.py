#!/usr/bin/env python3
"""Recompute finite full-system competition and behavioral SRAM port observations."""
import argparse
from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import stat

def bootstrap(path):
    path = Path(path)
    if not path.is_absolute(): path = Path.cwd() / path
    denied = lambda p: any("hammer" in x.lower() or "vlsi" in x.lower() for x in p.parts)
    if denied(path): raise ValueError("restricted dependency path")
    pending, current, links = deque(path.parts[1:]), Path(path.anchor), set()
    while pending:
        part = pending.popleft()
        if part == ".": continue
        if part == "..": current = current.parent; continue
        candidate = current / part
        if denied(candidate): raise ValueError("restricted dependency path")
        if stat.S_ISLNK(candidate.lstat().st_mode):
            if candidate in links or len(links) >= 64: raise ValueError("recursive dependency symlink")
            links.add(candidate)
            target = Path(os.readlink(candidate))
            if denied(target): raise ValueError("restricted dependency symlink target")
            if target.is_absolute():
                current = Path(target.anchor); pending.extendleft(reversed(target.parts[1:]))
            else: pending.extendleft(reversed(target.parts))
        else: current = candidate
    return current



def load(name, filename):
    path = bootstrap(Path(__file__).parent / filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

SRAM = load("system_memory_sram", "check-atlascore-sram-observation.py")
OBS, CHECK = SRAM.OBS, SRAM.CHECK
VERSION = "atlas.ee290.system_memory.fixed356.v2"


def require(condition, message):
    if not condition: raise ValueError(message)


def fixed_projection():
    signals = dict(OBS.SIGNALS)
    for name in ("io_tl_a_valid", "io_lsuScalarRead_valid", "io_lsuScalarWrite_valid", "io_dmaRead_valid", "io_dmaWrite_valid"):
        signals["competitor_vmem_" + name] = ("vmem." + name, 1)
    for engine in ("mxu0", "mxu1", "vpu"):
        for direction in ("Read", "Write"):
            for port in (0, 1):
                name = f"io_{engine}{direction}Req{port}_valid"
                signals["competitor_mreg_" + name] = ("mreg." + name, 1)
    for direction in ("Read", "Write"):
        name = f"io_xlu{direction}Req_valid"
        signals["competitor_mreg_" + name] = ("mreg." + name, 1)
    for bank in range(6):
        for key, suffix, width in (("en", "en", 1), ("mode", "wmode", 1), ("clk", "clk", 1), ("addr", "addr", 13), ("read", "rdata", 256), ("write", "wdata", 256), ("mask", "wmask", 32)):
            signals[f"v{bank}_{key}"] = (f"vmem.banks_{bank}.RW0_{suffix}", width)
    for bank in range(32):
        for key, suffix, width in (("ren", "R0_en", 1), ("wen", "W0_en", 1), ("rclk", "R0_clk", 1), ("wclk", "W0_clk", 1), ("raddr", "R0_addr", 6), ("waddr", "W0_addr", 6), ("read", "R0_data", 256), ("write", "W0_data", 256)):
            signals[f"m{bank}_{key}"] = (f"mreg.banks_{bank}.{suffix}", width)
    require(len(signals) == 356, "fixed signal projection size changed")
    return signals


SIGNALS = fixed_projection()
COMPETITORS = tuple(key for key in SIGNALS if key.startswith("competitor_"))
CLOCKS = tuple(key for key in SIGNALS if key not in OBS.SIGNALS and key.endswith(("_clk", "_rclk", "_wclk")))


def clocked_samples(stream, scope, max_edges):
    """Pair pre-edge samples with actual clock transitions at that timestamp.

    The existing decoder validates the VCD syntax and fixed declarations. This
    small parallel scan retains only clock state and pending rising timestamps;
    it never changes the decoder's pre-timestamp event convention.
    """
    wanted = {scope + "." + SIGNALS[key][0]: key for key in ("clock", *CLOCKS)}
    transitions = deque()
    hierarchy, declarations, header_tokens, clock_codes = [], {}, [], set()
    header, stamp, values, changes, change_counts, edges = True, None, {}, {}, {}, 0

    def flush():
        nonlocal edges
        global_code = declarations["clock"]
        if values.get(global_code) == 0 and changes.get(global_code, values.get(global_code)) == 1:
            edges += 1
            require(edges <= max_edges, "VCD edge limit exceeded")
            rises = {key: values.get(code) == 0 and changes.get(code) == 1 and change_counts.get(code) == 1
                     for key, code in declarations.items() if key != "clock"}
            transitions.append((stamp, rises))
        values.update(changes)
        changes.clear()
        change_counts.clear()

    def scanned_lines():
        nonlocal header, stamp
        for line in stream:
            tokens = line.split()
            if header:
                header_tokens.extend(tokens)
                while "$end" in header_tokens:
                    end = header_tokens.index("$end")
                    entry = header_tokens[:end]
                    del header_tokens[:end + 1]
                    if entry and entry[0] == "$scope" and len(entry) == 3:
                        hierarchy.append(entry[2])
                    elif entry == ["$upscope"] and hierarchy:
                        hierarchy.pop()
                    elif entry and entry[0] == "$var" and len(entry) >= 5:
                        key = wanted.get(".".join(hierarchy + [entry[4]]))
                        if key is not None:
                            declarations[key] = entry[3]
                            clock_codes.add(entry[3])
                    elif entry and entry[0] == "$enddefinitions":
                        header = False
            else:
                index = 0
                while index < len(tokens):
                    token = tokens[index]
                    index += 1
                    if token.startswith("#"):
                        if stamp is not None: flush()
                        stamp = int(token[1:])
                    elif not token.startswith("$"):
                        if stamp is None: stamp = 0
                        if token[0].lower() == "b":
                            if index >= len(tokens): break  # Decoder rejects truncation.
                            bits, code = token[1:].lower(), tokens[index]
                            index += 1
                        else:
                            bits, code = token[0].lower(), token[1:]
                        if code in clock_codes:
                            changes[code] = None if "x" in bits or "z" in bits else int(bits, 2)
                            change_counts[code] = change_counts.get(code, 0) + 1
            yield line
        if not header and stamp is not None: flush()

    private = SRAM.decoder("system_memory_private_projection")
    private.SIGNALS = SIGNALS.copy()
    for sample in private.vcd_edges(scanned_lines(), scope, max_edges):
        require(transitions and transitions[0][0] == sample["time"], "clock/sample timestamp mismatch")
        _, sample["sram_clock_rises"] = transitions.popleft()
        yield sample


def windows_from_boundary(boundary):
    starts = [item["edge"] for item in boundary["endpoints"] if item["kind"] == "halt_deasserted"]
    panels = boundary["panels"]
    require(len(starts) == len(panels) == 3, "three unique halt-deassertion/panel windows required")
    windows = []
    for index, (start, panel) in enumerate(zip(starts, panels)):
        end = panel["halt_edge"]
        require(type(start) is int and type(end) is int and start < panel["commands"][0]["edge"] <= panel["marker_edge"] < end, "invalid execution window endpoints")
        require(not windows or windows[-1]["end_edge"] < start, "overlapping execution windows")
        windows.append({"panel": index, "start_edge": start, "end_edge": end, "sampled_edges": end-start+1, "scope": "observed_halt_deassertion_through_ecall_halt_inclusive"})
    return windows


def sampled_value(values, key):
    result = values.get(key)
    require(type(result) is int and 0 <= result < 1 << SIGNALS[key][1], "unknown/out-of-width active sampled value: " + key)
    return result


def validate_window_samples(samples, boundary):
    """Validate all finite execution edges, including idle gaps/completion.

    Reuse SRAM's active command data/address/mask/response validator. Add idle
    enable checks and pre-arbitration request quiescence over the whole envelope.
    """
    windows = windows_from_boundary(boundary)
    counts = [0] * len(windows)
    previous = [None] * len(windows)
    selected = []
    active_index = 0
    for sample in samples:
        edge = sample["edge"]
        while active_index < len(windows) and edge > windows[active_index]["end_edge"]: active_index += 1
        if active_index == len(windows): break
        window = windows[active_index]
        if edge < window["start_edge"]: continue
        require(previous[active_index] is None and edge == window["start_edge"] or previous[active_index] is not None and edge == previous[active_index]+1, "execution window sample gap")
        previous[active_index] = edge
        counts[active_index] += 1
        values = sample["values"]
        require(sampled_value(values, "reset") == 0, "reset within admitted execution window")
        for key in COMPETITORS:
            require(sampled_value(values, key) == 0, "competing request within admitted execution window: " + key)
        # Decoder samples the settled values immediately before the global
        # rising timestamp. Direct SRAM clocks must still be known low then.
        for key in CLOCKS:
            require(sampled_value(values, key) == 0, "unknown or phase-mismatched sampled SRAM clock: " + key)
            require(sample.get("sram_clock_rises", {}).get(key) is True,
                    "missing or mismatched SRAM rising clock at global edge: " + key)
        for bank in range(6):
            expected = int((sampled_value(values, "vr") == 1 and sampled_value(values, "vr_bank") == bank) or (sampled_value(values, "vw") == 1 and sampled_value(values, "vw_bank") == bank))
            require(sampled_value(values, f"v{bank}_en") == expected, "physical VMEM enable differs from admitted request")
        for bank in range(32):
            ren = int(sampled_value(values, "mr") == 1 and sampled_value(values, "mr_id") % 32 == bank)
            wen = int(sampled_value(values, "mw") == 1 and sampled_value(values, "mw_id") % 32 == bank)
            require(sampled_value(values, f"m{bank}_ren") == ren and sampled_value(values, f"m{bank}_wen") == wen, "physical MREG enable differs from admitted request")
        selected.append(sample)
    require(counts == [window["sampled_edges"] for window in windows], "missing execution window edge coverage")
    physical = SRAM.validate_physical(selected, boundary)
    require(len(physical) == 6, "six complete finite SRAM command windows required")
    for window in windows:
        window.update({"competing_request_signals": len(COMPETITORS), "competing_request_edges": 0, "physical_ports_match": True, "sampled_sram_clocks_known_low": True, "sampled_sram_clocks_rise_at_global_edge": True})
    return {"windows": windows, "physical_observations": physical}


def analyze_trace(trace_path, scope, max_edges=3000000):
    """Public API: fixed projection, independently recomputed finite events."""
    trace = CHECK.allowed(trace_path)
    require(isinstance(scope, str) and scope and type(max_edges) is int and max_edges > 0, "explicit scope and positive edge bound required")
    with trace.open() as stream:
        boundary = OBS.validate_edges(OBS.vcd_edges(stream, scope, max_edges), 3)
    with trace.open() as stream:
        result = validate_window_samples(clocked_samples(stream, scope, max_edges), boundary)
    return {"schema": "atlas.ee290_system_memory_events.v0", "target_config": "EE290SimConfig", "validator": VERSION, "state": "finite_system_memory_events_passed", "scheduling_qualified": False, "operand_domain_qualified": False, "boundary_observations": boundary, **result,
            "assumptions": ["Fixed authored projection and pre-timestamp rising-edge sampling; no delta ordering or asynchronous/off-edge clock glitch certification.", "Finite observed halt-deassertion through ECALL/halt envelopes only; no universal accelerator entry quiescence or other program/domain qualification.", "All five VMEM and fourteen MREG competitor request valids are known zero over admitted envelopes, including gaps and completion; no proof for uncaptured request paths.", "Behavioral SRAM wrapper enables/addresses/data/masks/responses are observed; no physical macro/PVT timing qualification.", "Host CSR status polling is outside the VMEM competing-request restriction; instruction/debug-register immutability and caller initialization require separate selected host fixture reasoning.", "Capture production, simulator/source applicability, numerical results and resolver-to-program links require the separate aggregate checker."]}


def select(checker, path, expected):
    require(CHECK.hash_string(expected), "explicit selected capture digest required")
    identity = checker.identity(path)
    data = CHECK.allowed(identity["path"]).read_bytes()
    require(identity["sha256"] == expected == CHECK.sha256(data), "selected capture digest mismatch")
    return identity, CHECK.strict_json(data)


def run(capture_report, expected_sha256, output):
    checker = CHECK.Checker()
    helper_names = (Path(__file__).name, "check-atlascore-sram-observation.py", "observe-ee290-vls.py", "check-ee290-provenance.py")
    helpers = {name: checker.identity(Path(__file__).parent / name) for name in helper_names}
    capture_id, capture = select(checker, capture_report, expected_sha256)
    require(capture.get("schema") == "atlas.ee290_vcs_observation_capture.v0" and capture.get("target_config") == "EE290SimConfig" and capture.get("scheduling_qualified") is False, "capture schema/target/qualification mismatch")
    plan_id = checker.verify(capture["inputs"]["plan"], Path(capture_id["path"]).parent)
    plan = CHECK.strict_json(CHECK.allowed(plan_id["path"]).read_bytes())
    require(plan.get("schema") == "atlas.ee290_vcs_observation_capture_plan.v0" and plan.get("target_config") == "EE290SimConfig", "capture plan schema/target mismatch")
    boundary_plan_id = checker.verify(plan["boundary_plan"], Path(plan_id["path"]).parent)
    boundary_plan = CHECK.strict_json(CHECK.allowed(boundary_plan_id["path"]).read_bytes())
    scope = boundary_plan["atlas_scope"]
    require(plan["capture_signals"] == {scope + "." + name: width for name, width in SIGNALS.values()}, "capture projection differs from fixed356 mapping")
    trace_id = checker.verify(capture["trace"], Path(capture_id["path"]).parent)
    saved_id = checker.verify(capture["boundary_observation"], Path(capture_id["path"]).parent)
    saved = CHECK.strict_json(CHECK.allowed(saved_id["path"]).read_bytes())
    require(saved.get("schema") == "atlas.ee290_vls_boundary_observation.v0", "saved boundary schema mismatch")
    require(saved["inputs"]["trace"] == trace_id and saved["inputs"]["plan"] == boundary_plan_id, "capture/boundary plan/trace mismatch")
    result = analyze_trace(trace_id["path"], scope)
    require(result["boundary_observations"] == saved["observations"], "saved boundary observations differ from recomputation")
    destination = OBS._fresh_output(output)
    destination.mkdir(parents=True)
    snapshots = {}
    for name, identity in helpers.items():
        path = destination / (name.removesuffix(".py") + ".executed.py")
        path.write_bytes(CHECK.allowed(identity["path"]).read_bytes())
        snapshots[name] = checker.identity(path)
        require((snapshots[name]["sha256"], snapshots[name]["bytes"]) == (identity["sha256"], identity["bytes"]), "validator snapshot changed")
    packet = {"schema": "atlas.ee290_system_memory_observation.v0", "target_config": "EE290SimConfig", "state": "finite_system_memory_observation_passed", "scheduling_qualified": False, "capture_execution_link_verified": False, "selection": {"capture": capture_id, "plan": plan_id, "boundary_plan": boundary_plan_id, "trace": trace_id, "saved_boundary": saved_id}, "scope": scope, "events": result, "validators": helpers, "validator_snapshots": snapshots}
    checker.recheck()
    (destination / "report.json").write_text(json.dumps(packet, indent=2) + "\n")
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-report", required=True, type=Path)
    parser.add_argument("--expected-capture-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try: run(args.capture_report, args.expected_capture_sha256, args.output)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, "system memory observation rejected: " + str(error) + "\n")


if __name__ == "__main__": main()
