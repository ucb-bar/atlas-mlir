"""Mutation checks for finite competition and behavioral SRAM observations."""
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("system_memory", Path(__file__).parent.parent / "tools/check-ee290-system-memory-observation.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def fixture():
    panels, endpoints, samples = [], [], []
    for panel in range(3):
        shift = panel * 100
        commands = [{"edge": shift+3, "op": 1, "line": 32, "mreg": 4}, {"edge": shift+38, "op": 2, "line": 96, "mreg": 4}]
        panels.append({"commands": commands, "marker_edge": shift+73, "halt_edge": shift+74})
        endpoints.append({"kind": "halt_deasserted", "edge": shift+1})
        for edge in range(shift+1, shift+75):
            values = {key: 0 for key in M.SIGNALS}
            for command in commands:
                age = edge-command["edge"]
                store = command["op"] == 2
                if 1 <= age <= 32:
                    if store:
                        values.update(mr=1, mr_id=4, mr_row=age-1, m4_ren=1, m4_raddr=age-1)
                    else:
                        values.update(vr=1, vr_bank=0, vr_addr=32+age-1, v0_en=1, v0_addr=32+age-1)
                if 2 <= age <= 33:
                    if store: values.update(mresp=1, mresp_data=5, m4_read=5)
                    else: values.update(vresp=1, vresp_data=5, v0_read=5)
                if 3 <= age <= 34:
                    if store:
                        values.update(vw=1, vw_bank=0, vw_addr=96+age-3, vw_data=5, v0_en=1, v0_mode=1, v0_addr=96+age-3, v0_write=5, v0_mask=0xffffffff)
                    else:
                        values.update(mw=1, mw_id=4, mw_row=age-3, mw_data=5, m4_wen=1, m4_waddr=age-3, m4_write=5)
            samples.append({"edge": edge, "time": 4*edge, "values": values,
                            "sram_clock_rises": {key: True for key in M.CLOCKS}})
    return samples, {"panels": panels, "endpoints": endpoints}


def full_vcd(mutation=None, alias_clocks=False, mutation_edge=5):
    """Three complete panels, including a halted host interval before entry."""
    original, _ = fixture()
    samples = {sample["edge"]+1: dict(sample["values"]) for sample in original}
    for edge in range(1, 276):
        if edge not in samples:
            samples[edge] = {key: 0 for key in M.SIGNALS}
            samples[edge]["halt"] = 1
            samples[edge]["marker"] = int(edge in range(76, 101) or edge in range(176, 201))
    for panel in range(3):
        shift = panel*100+1
        for command_edge, op, line in ((shift+3, 1, 32), (shift+38, 2, 96)):
            samples[command_edge].update(fire=1, launch=1, cmd=1, op=op, line=line, mreg=4)
            for age in range(1, 35):
                samples[command_edge+age]["store_busy" if op == 2 else "load_busy"] = 1
        samples[shift+73].update(csr=1, fire=1, csr_addr=0xc10, csr_op=1, csr_data=1)
        samples[shift+74].update(marker=1, halt=1, ecall=1)
    codes = {key: f"symbol{index}" for index, key in enumerate(M.SIGNALS)}
    if alias_clocks:
        for key in M.CLOCKS: codes[key] = codes["clock"]
    lines = ["$timescale 1ns $end", "$scope module TOP $end"]
    lines += [f"$var wire {width} {codes[key]} {name} $end"
              for key, (name, width) in M.SIGNALS.items()]
    lines += ["$upscope $end", "$enddefinitions $end", "#0"]
    for edge in range(1, 276):
        if edge != 1: lines.append(f"#{4*edge-2}")
        emitted = set()
        for key, value in samples[edge].items():
            if codes[key] not in emitted:
                lines.append(f"b{value:b} {codes[key]}")
                emitted.add(codes[key])
        lines += [f"#{4*edge}", "1"+codes["clock"]]
        emitted = {codes["clock"]}
        for key in M.CLOCKS:
            if codes[key] in emitted: continue
            emitted.add(codes[key])
            if mutation == "frozen" or (key == "m4_rclk" and edge == mutation_edge and mutation in ("missing", "late")):
                continue
            if key == "m4_rclk" and edge == mutation_edge and mutation == "unknown":
                lines.append("x"+codes[key])
            else:
                lines.append("1"+codes[key])
                if key == "m4_rclk" and edge == mutation_edge and mutation == "multiple":
                    lines += ["0"+codes[key], "1"+codes[key]]
        if mutation == "late" and edge == mutation_edge:
            lines += [f"#{4*edge+1}", "1"+codes["m4_rclk"]]
    return "\n".join(lines)+"\n"


class MemoryMutationTests(unittest.TestCase):
    def setUp(self): self.samples, self.boundary = fixture()

    def reject(self, key, value, edge=4):
        next(x for x in self.samples if x["edge"] == edge)["values"][key] = value
        with self.assertRaises((ValueError, M.OBS.ObservationError)):
            M.validate_window_samples(self.samples, self.boundary)

    def test_valid_adjacent_command_release_and_terminal_windows(self):
        result = M.validate_window_samples(self.samples, self.boundary)
        self.assertEqual(len(result["physical_observations"]), 6)
        self.assertEqual([x["sampled_edges"] for x in result["windows"]], [74]*3)

    def test_each_captured_competitor_rejected_in_entry_and_terminal(self):
        for key in M.COMPETITORS:
            for edge in (1, 74):
                self.samples, self.boundary = fixture()
                with self.subTest(key=key, edge=edge): self.reject(key, 1, edge)

    def test_unknown_competitor_rejected(self): self.reject(M.COMPETITORS[0], None)
    def test_unknown_active_data_rejected(self): self.reject("v0_read", None, 5)
    def test_mask_mismatch_rejected(self): self.reject("v0_mask", 0, 41)
    def test_address_mismatch_rejected(self): self.reject("v0_addr", 999)
    def test_response_data_mismatch_rejected(self): self.reject("v0_read", 7, 5)
    def test_destination_data_mismatch_rejected(self): self.reject("m4_write", 7, 6)
    def test_extra_physical_bank_enable_rejected(self): self.reject("v1_en", 1)
    def test_missing_physical_enable_rejected(self): self.reject("m4_wen", 0, 6)
    def test_idle_entry_enable_rejected(self): self.reject("m0_ren", 1, 1)
    def test_idle_terminal_enable_rejected(self): self.reject("v0_en", 1, 74)
    def test_unknown_or_shifted_sram_clock_rejected(self): self.reject("m4_rclk", None)
    def test_missing_rising_clock_evidence_rejected(self):
        del self.samples[0]["sram_clock_rises"]
        with self.assertRaisesRegex(ValueError, "rising clock"):
            M.validate_window_samples(self.samples, self.boundary)
    def test_complete_vcd_clock_transitions(self):
        for mutation in (None, "frozen", "missing", "late", "unknown", "multiple"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix="system-memory-vcd-") as directory:
                trace = Path(directory)/"trace.vcd"
                trace.write_text(full_vcd(mutation))
                if mutation is None:
                    result = M.analyze_trace(trace, "TOP")
                    self.assertEqual(len(result["physical_observations"]), 6)
                    self.assertTrue(all(window["sampled_sram_clocks_rise_at_global_edge"] for window in result["windows"]))
                else:
                    with self.assertRaisesRegex(ValueError, "rising clock"):
                        M.analyze_trace(trace, "TOP")
    def test_complete_vcd_aliased_direct_clocks(self):
        with tempfile.TemporaryDirectory(prefix="system-memory-vcd-") as directory:
            trace = Path(directory)/"trace.vcd"
            trace.write_text(full_vcd(alias_clocks=True))
            M.analyze_trace(trace, "TOP")
    def test_complete_vcd_missing_clocks_at_each_entry_and_terminal(self):
        for edge in (2, 75, 102, 175, 202, 275):
            with self.subTest(edge=edge), tempfile.TemporaryDirectory(prefix="system-memory-vcd-") as directory:
                trace = Path(directory)/"trace.vcd"
                trace.write_text(full_vcd("missing", mutation_edge=edge))
                with self.assertRaisesRegex(ValueError, "rising clock"):
                    M.analyze_trace(trace, "TOP")
    def test_missing_window_sample_rejected(self):
        del self.samples[20]
        with self.assertRaisesRegex(ValueError, "sample gap"):
            M.validate_window_samples(self.samples, self.boundary)
    def test_truncated_completion_rejected(self):
        self.samples.pop()
        with self.assertRaisesRegex(ValueError, "coverage"):
            M.validate_window_samples(self.samples, self.boundary)
    def test_inactive_unknown_memory_data_allowed(self):
        self.samples[0]["values"]["v5_read"] = None
        M.validate_window_samples(self.samples, self.boundary)
    def test_scope_is_fixed_and_signal_counts_are_explicit(self):
        self.assertEqual(len(M.SIGNALS), 356)
        self.assertEqual(len(M.COMPETITORS), 19)
        self.assertEqual(len(M.CLOCKS), 70)
    def test_missing_or_unknown_enable_rejected(self): self.reject("v5_en", None, 74)
    def test_no_window_start_from_copied_command_pass_flag(self):
        self.boundary["endpoints"].pop()
        with self.assertRaisesRegex(ValueError, "unique"):
            M.windows_from_boundary(self.boundary)
    def test_missing_or_wrong_width_captured_signal_declaration_rejected(self):
        for mode in ("missing", "width"):
            private = M.SRAM.decoder("mutation_private_projection_" + mode)
            private.SIGNALS = M.SIGNALS.copy()
            lines = ["$timescale 1ns $end", "$scope module TOP $end"]
            for index, (key, (name, width)) in enumerate(M.SIGNALS.items()):
                if key == M.COMPETITORS[0]:
                    if mode == "missing": continue
                    width += 1
                lines.append(f"$var wire {width} symbol{index} {name} $end")
            lines += ["$upscope $end", "$enddefinitions $end", "#0"]
            with self.subTest(mode=mode), self.assertRaises(private.ObservationError):
                list(private.vcd_edges(io.StringIO("\n".join(lines) + "\n"), "TOP"))

    def test_restricted_path_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory(prefix="system-memory-path-") as directory:
            path = Path(directory)/"trace"
            path.symlink_to('/tmp/' + 'ham' + 'mer' + '/absent')
            with self.assertRaises(ValueError): M.CHECK.allowed(path)


if __name__ == "__main__": unittest.main()
