#!/usr/bin/env python3
"""Synthetic receipt/trace rejection checks; these are not system timing evidence."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parent.parent
SPEC = importlib.util.spec_from_file_location("integrated", ROOT / "tools/check-ee290-vcs-observation.py")
I = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(I)
FIXTURE_SPEC = importlib.util.spec_from_file_location("capture_fixture", ROOT / "test/test_ee290_vcs_observation_capture.py")
F = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(F)


def identity(path):
    return I.CHECK.Checker().identity(path)


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")
    return identity(path)


class IntegratedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = F.CaptureTests()
        cls.fixture.setUp()
        base = cls.fixture.base
        witness_id, phase_id = cls.fixture.fixture()
        witness = json.loads(Path(witness_id["path"]).read_text())
        words = [0x10000000 + index for index in range(10)]
        words_path = base / "words.txt"
        words_path.write_text("".join(f"{value:08x}\n" for value in words))
        words_id = identity(words_path)
        include_path = Path(witness["generated_include"]["path"])
        include_path.write_text(f"#define ATLAS_PROGRAM_WORDS {len(words)}U\nstatic const uint32_t atlas_program[] = {{\n"
                               + "".join(f"  0x{value:08x}U,\n" for value in words) + "};\n")
        witness["generated_include"] = identity(include_path)
        host_path = Path(witness["host_binary"]["path"])
        host_path.write_bytes(b"\x7fELF\x02\x01synthetic-header" + b"".join(struct.pack("<I", value) for value in words))
        witness["host_binary"] = identity(host_path)
        host_source = Path(witness["provenance"]["host_source"]["path"])
        host_source.write_text('#include "atlas_program.inc"\n/* Synthetic fixture; not an executable host program. */\n')
        witness["provenance"]["host_source"] = identity(host_source)
        executed_source = base / "ee290-vls-host.executed.c"
        executed_source.write_bytes(host_source.read_bytes())
        program = witness["provenance"]["program"]
        emitter = cls.fixture.member("emitter", "synthetic compiler")
        host_cc = cls.fixture.member("host-cc", "synthetic host compiler")
        witness["provenance"].update(atlas_emit=emitter, host_cc=host_cc)
        witness["scheduling_qualified"] = False
        witness["program_word_count"] = len(words)
        witness["runtime_environment"] = {key: os.environ.get(key, "") for key in ("VCS_HOME", "VCS_64", "LD_LIBRARY_PATH")}
        compilation_log = cls.fixture.member("host-build.log", "synthetic compile completed")
        emit_record = {"argv": [emitter["path"], program["path"]], "returncode": 0, "timed_out": False, "log": words_id}
        build_record = {"argv": [host_cc["path"], "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror", "-fno-common",
                                 "-fno-builtin-printf", "-march=rv64imafd", "-mabi=lp64d", "-mcmodel=medany", "-specs=htif_nano.specs",
                                 "-static", "-T", "htif.ld", str(executed_source), "-o", str(host_path)],
                        "returncode": 0, "timed_out": False, "log": compilation_log}
        witness["commands"] = [emit_record, build_record, witness["commands"][-1]]
        witness_id = write_json(Path(witness_id["path"]), witness)
        phase = json.loads(Path(phase_id["path"]).read_text())
        phase["inputs_before"] = [{"identity": source, "role": "source"} for source in witness["provenance"]["simulator_sources"]]
        phase_id = write_json(Path(phase_id["path"]), phase)
        converter = cls.fixture.converter
        converter.write_text(converter.read_text() + "print('Done writing vcd value changes.\\nDone')\n")
        output = base / "capture"
        F.CAP.prepare(witness_id["path"], witness_id["sha256"], phase_id["path"], phase_id["sha256"],
                      "TestDriver.atlas", output, 30, converter)
        plan_id = identity(output / "plan.json")
        receipt = F.CAP.execute(plan_id["path"], plan_id["sha256"])
        if receipt["state"] != "numerical_and_boundary_observation_passed":
            raise AssertionError(receipt)
        cls.plan_id, cls.capture_id = plan_id, identity(output / "report.json")
        cls.phase = phase
        cls.numerical_pair = {"runs": [{"arm": "baseline", "receipt": witness_id}]}
        cls.compiler = {"artifacts": {"baseline.mlir": program, "baseline.words": words_id}}
        cls.original_read_member = I.read_member
        cls.plan = json.loads(Path(plan_id["path"]).read_text())
        cls.capture = receipt
        cls.witness = witness

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDown()

    def verify(self):
        return I.verify_arm(I.CHECK.Checker(), "baseline", self.plan_id, self.capture_id,
                            self.phase, self.numerical_pair, self.compiler)

    def test_complete_synthetic_chain_recomputes_events(self):
        result = self.verify()
        self.assertEqual(result["visible_signal_declarations"], 356)
        self.assertEqual(len(result["boundary_observations"]["panels"]), 3)
        self.assertEqual(result["program"]["word_values"], [0x10000000 + index for index in range(10)])
        self.assertTrue(result["program"]["program_words_embedded_once"])
        self.assertTrue(result["validated_links"]["native_capture_and_conversion_commands"])

    def substituted_json(self, selected_path, mutate):
        original = type(self).original_read_member
        def selected(checker, member, base):
            member_id, value = original(checker, member, base)
            if member_id["path"] == selected_path:
                value = copy.deepcopy(value)
                mutate(value)
            return member_id, value
        return patch.object(I, "read_member", side_effect=selected)

    def test_saved_boundary_facts_cannot_replace_trace_recomputation(self):
        path = self.capture["boundary_observation"]["path"]
        def mutate(value):
            value["observations"]["panels"][0]["commands"][1]["release_edge"] += 1
        with self.substituted_json(path, mutate):
            with self.assertRaisesRegex(ValueError, "saved boundary observations"):
                self.verify()

    def test_success_flags_do_not_hide_bad_conversion_or_simulation(self):
        for index, key, value in [(2, "returncode", 1), (1, "timed_out", True), (1, "descendants_after_parent_exit", True)]:
            with self.subTest(index=index, field=key):
                with self.substituted_json(self.capture_id["path"], lambda data: data["commands"][index].update({key: value})):
                    with self.assertRaises(ValueError):
                        self.verify()

    def test_convert_trace_or_program_argument_substitution_rejects(self):
        for index, position in [(2, -1), (1, -1)]:
            with self.subTest(command=index):
                def mutate(data):
                    data["commands"][index]["argv"][position] = "/tmp/unselected-artifact"
                with self.substituted_json(self.capture_id["path"], mutate):
                    with self.assertRaisesRegex(ValueError, "invocation mismatch"):
                        self.verify()

    def test_trace_identity_change_rejects_before_recomputation(self):
        with self.substituted_json(self.capture_id["path"], lambda data: data["trace"].update(sha256="0" * 64)):
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                self.verify()

    def test_program_crosslink_and_qualification_promotion_reject(self):
        compiler = copy.deepcopy(self.compiler)
        compiler["artifacts"]["baseline.words"] = self.fixture.member("substitute-words.txt", "00000001\n")
        with self.assertRaisesRegex(ValueError, "words"):
            I.verify_arm(I.CHECK.Checker(), "baseline", self.plan_id, self.capture_id, self.phase, self.numerical_pair, compiler)
        with self.substituted_json(self.capture_id["path"], lambda data: data.update(scheduling_qualified=True)):
            with self.assertRaisesRegex(ValueError, "promotion"):
                self.verify()

    def test_runtime_and_signal_mapping_substitution_reject(self):
        for path, mutate in [
            (self.capture_id["path"], lambda data: data["runtime_environment"].update(VCS_64="changed")),
            (self.plan_id["path"], lambda data: data["capture_signals"].pop("TestDriver.atlas.vmem.banks_0.RW0_en")),
        ]:
            with self.subTest(path=path), self.substituted_json(path, mutate):
                with self.assertRaisesRegex(ValueError, "environment|projection"):
                    self.verify()

    def test_historical_snapshot_does_not_read_mutable_original_producer(self):
        def mutate(data):
            data["producer"]["path"] = "/nonexistent/historical-producer.py"
        with self.substituted_json(self.plan_id["path"], mutate):
            self.assertTrue(self.verify()["validated_links"]["recomputed_boundary_events"])
        with self.substituted_json(self.plan_id["path"], lambda data: data["producer"].update(sha256="0" * 64)):
            with self.assertRaisesRegex(ValueError, "producer snapshot"):
                self.verify()

    def test_duplicate_compared_module_basename_cannot_choose_last_match(self):
        compared = {"path": "/reviewed/LSU.sv", "sha256": "1" * 64, "bytes": 100}
        matching = dict(compared, path="/captured/LSU.sv")
        wrong = dict(compared, path="/other/LSU.sv", sha256="2" * 64)
        I.bind_compared_sources([compared], {"baseline": {"source_inventory": [matching]}})
        # The matching member is last: a dict comprehension would hide the
        # earlier conflicting module and incorrectly accept the inventory.
        with self.assertRaisesRegex(ValueError, "ambiguous.*LSU.sv"):
            I.bind_compared_sources([compared], {"baseline": {"source_inventory": [wrong, matching]}})

    def test_independent_selection_and_restricted_paths_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "selection digest"):
            I.select(I.CHECK.Checker(), self.plan_id["path"], "0" * 64)
        with patch.object(Path, "lstat", side_effect=AssertionError("unexpected access")):
            with self.assertRaises(ValueError):
                I.bootstrap('/tmp/forbidden-' + 'ham' + 'mer' + '/absent')

    def test_all_capture_command_fields_are_bound_to_prepared_invocations(self):
        for index in range(3):
            for field, value in (("cwd", "/tmp/unselected-run"), ("timeout_seconds", 9999)):
                with self.subTest(command=index, field=field):
                    with self.substituted_json(self.capture_id["path"],
                                              lambda data: data["commands"][index].update({field: value})):
                        with self.assertRaisesRegex(ValueError, "invocation mismatch"):
                            self.verify()
        with self.substituted_json(self.capture_id["path"], lambda data: data["commands"].reverse()):
            with self.assertRaisesRegex(ValueError, "invocation mismatch"):
                self.verify()

    def test_log_success_flags_require_supported_output_and_no_hidden_errors(self):
        for index, text in ((0, "-file -type -add -depth -fid\nucli% puts ATLAS_UCLI_HELP_COMPLETED\n"),
                            (1, "ATLAS_UCLI_CAPTURE_SETUP_OK\n" + F.NUMERICAL + "Error-UCLI: hidden signal\n"),
                            (2, "conversion claimed successful\n")):
            log_id = self.fixture.member(f"untrusted-success-{index}.log", text)
            with self.subTest(command=index):
                with self.substituted_json(self.capture_id["path"],
                                          lambda data: data["commands"][index].update(log=log_id)):
                    with self.assertRaisesRegex(ValueError, "help probe|error/license|conversion completion"):
                        self.verify()
        with self.substituted_json(self.capture_id["path"], lambda data: data["observations"].pop()):
            with self.assertRaisesRegex(ValueError, "observations differ from log"):
                self.verify()

    def test_finite_boundary_report_cannot_promote_unchecked_execution_or_physical_scope(self):
        for key in ("scheduling_qualified", "physical_sram_arbitration_verified", "capture_execution_link_verified"):
            with self.subTest(field=key):
                with self.substituted_json(self.capture["boundary_observation"]["path"],
                                          lambda data: data.update({key: True})):
                    with self.assertRaisesRegex(ValueError, "qualification mismatch"):
                        self.verify()

    def test_host_compilation_and_emission_invocations_cannot_change(self):
        base = Path(self.capture_id["path"]).parent
        for index, position in ((0, 0), (0, -1), (1, 1), (1, -1)):
            witness = copy.deepcopy(self.witness)
            witness["commands"][index]["argv"][position] = "/tmp/unselected-input"
            with self.subTest(command=index, position=position):
                with self.assertRaisesRegex(ValueError, "invocation mismatch"):
                    I.host_program(I.CHECK.Checker(), witness, self.plan, self.compiler, "baseline", base)
        for index in range(3):
            witness = copy.deepcopy(self.witness)
            witness["commands"][index]["returncode"] = 1
            with self.subTest(failed_command=index), self.assertRaises(ValueError):
                I.host_program(I.CHECK.Checker(), witness, self.plan, self.compiler, "baseline", base)

    def test_host_word_array_include_and_original_result_are_rechecked(self):
        base = Path(self.capture_id["path"]).parent
        words = b"".join(struct.pack("<I", value) for value in self.verify()["program"]["word_values"])
        cases = [("host_binary", b"\x7fELF\x02\x01absent"),
                 ("host_binary", b"\x7fELF\x02\x01" + words + words),
                 ("host_binary", b"\x7fELF\x02\x02" + words),
                 ("generated_include", b"#define ATLAS_PROGRAM_WORDS 0U\n"),
                 ("host_source", b"int main(void) { return 0; }\n")]
        for index, (key, blob) in enumerate(cases):
            plan = copy.deepcopy(self.plan)
            path = self.fixture.base / f"host-substitution-{index}"
            path.write_bytes(blob)
            plan["snapshots"][key] = identity(path)
            with self.subTest(member=key, index=index), self.assertRaisesRegex(ValueError, "ELF|include"):
                I.host_program(I.CHECK.Checker(), self.witness, plan, self.compiler, "baseline", base)
        witness = copy.deepcopy(self.witness)
        witness["commands"][-1]["log"] = self.fixture.member("false-original-result.log", "EE290_VLS_PASSED panels=3\n")
        with self.assertRaisesRegex(ValueError, "original numerical result"):
            I.host_program(I.CHECK.Checker(), witness, self.plan, self.compiler, "baseline", base)

    def test_build_source_and_simulator_links_cannot_be_replaced(self):
        phase = copy.deepcopy(self.phase)
        phase["inputs_before"] = phase["inputs_before"][1:]
        with self.assertRaisesRegex(ValueError, "not a captured build input"):
            I.verify_arm(I.CHECK.Checker(), "baseline", self.plan_id, self.capture_id,
                         phase, self.numerical_pair, self.compiler)
        phase = copy.deepcopy(self.phase)
        phase["outputs"][0]["files"][0]["identity"] = self.fixture.member("unselected-simulator", "other executable")
        with self.assertRaisesRegex(ValueError, "compiled/captured simulator"):
            I.verify_arm(I.CHECK.Checker(), "baseline", self.plan_id, self.capture_id,
                         phase, self.numerical_pair, self.compiler)
        numerical = copy.deepcopy(self.numerical_pair)
        numerical["runs"].append(copy.deepcopy(numerical["runs"][0]))
        with self.assertRaisesRegex(ValueError, "missing/ambiguous"):
            I.verify_arm(I.CHECK.Checker(), "baseline", self.plan_id, self.capture_id,
                         self.phase, numerical, self.compiler)

    def resolver_fixture(self):
        arm = self.verify()
        words = arm["program"]["word_values"]
        panel = arm["boundary_observations"]["panels"][0]
        instructions = [{"word_index": index, "word_u32": word, "mnemonic": "addi",
                         "logical_issue_cycle": 0, "operands": {}, "footprint": {}}
                        for index, word in enumerate(words)]
        for index, command in enumerate(panel["commands"]):
            store = command["op"] == 2
            edge = command["edge"]
            instructions[index].update(mnemonic="vstore" if store else "vload", logical_issue_cycle=edge,
                                       operands={"rd": command["mreg"]}, footprint={
                "accesses": [{"resource": "vmem", "first": command["line"], "write": store,
                              "age": 3 if store else 1, "count": 32, "step": 1, "anywhere": False, "at_completion": False},
                             {"resource": "mreg", "first": command["mreg"] * 32, "write": not store,
                              "age": 1 if store else 3, "count": 32, "step": 1, "anywhere": False, "at_completion": False}],
                "holds": [{"unit": unit, "to": command["release_edge"] - edge - 1}
                          for unit in ("VLOAD path", "VSTORE path")]})
        instructions[2].update(mnemonic="csrrw", logical_issue_cycle=panel["marker_edge"])
        instructions[3].update(mnemonic="ecall", logical_issue_cycle=panel["halt_edge"], event_kind="terminal_acceptance")
        evidence = {"evidence_sha256": "1" * 64, "manifest_sha256": "2" * 64, "hardware_ir_sha256": "3" * 64}
        export = {"schema": "atlas.resolved_rtl_timing.v0", "target_config": "EE290SimConfig",
                  "qualification": "conditional", "scheduling_qualified": False,
                  "resolver": {"id": "atlas.vls.conservative.v1", "version": 1}, "evidence": evidence.copy(),
                  "program": {"words": words, "word_count": len(words), "words_sha256": hashlib.sha256(
                      b"".join(struct.pack("<I", word) for word in words)).hexdigest()}, "instructions": instructions,
                  "applicability": {"scope": "finite synthetic fixture", "supported_operations": ["vload", "vstore"],
                                    "operand_domain": {}, "environment_assumptions": ["sram_read_response_one_cycle_after_request"],
                                    "unsupported": ["general scheduling qualification"]}}
        return arm, evidence, export

    def test_fresh_resolver_command_binds_words_evidence_and_recomputed_events(self):
        arm, evidence, export = self.resolver_fixture()
        emitter_path = self.fixture.base / "synthetic-resolver"
        emitter_path.write_text("#!/usr/bin/env python3\nimport json,sys\n"
                                + f"assert sys.argv[1:] == ['--rtl-timing-json', {arm['program']['program']['path']!r}]\n"
                                + f"print({json.dumps(export)!r})\n")
        emitter_path.chmod(0o755)
        with tempfile.TemporaryDirectory(dir=self.fixture.base, prefix="resolver-output-") as directory:
            result = I.export_binding(I.CHECK.Checker(), identity(emitter_path), arm, evidence, Path(directory))
            self.assertTrue(all(result["event_binding"].values()))
            self.assertEqual(json.loads(Path(result["command"]["stdout"]["path"]).read_text()), export)
            self.assertFalse(result["command"]["timed_out"])
        mutations = [lambda data: data["evidence"].update(manifest_sha256="0" * 64),
                     lambda data: data["program"]["words"].__setitem__(0, 0),
                     lambda data: data["instructions"][0]["operands"].update(rd=5),
                     lambda data: data["instructions"][1].update(logical_issue_cycle=39),
                     lambda data: data.update(scheduling_qualified=True)]
        for index, mutate in enumerate(mutations):
            changed = copy.deepcopy(export)
            mutate(changed)
            completed = subprocess.CompletedProcess([], 0, json.dumps(changed).encode(), b"")
            with self.subTest(mutation=index), tempfile.TemporaryDirectory(dir=self.fixture.base) as directory:
                with patch.object(I.subprocess, "run", return_value=completed), self.assertRaises(ValueError):
                    I.export_binding(I.CHECK.Checker(), identity(emitter_path), arm, evidence, Path(directory))
                self.assertFalse((Path(directory) / "resolved.json").exists())

    def test_fresh_resolver_failure_cannot_publish_a_bound_export(self):
        arm, evidence, export = self.resolver_fixture()
        emitter = identity(self.fixture.base / "emitter")
        failed = subprocess.CompletedProcess([], 1, json.dumps(export).encode(), b"failed")
        for kwargs in ({"return_value": failed}, {"side_effect": subprocess.TimeoutExpired([emitter["path"]], 60)}):
            with tempfile.TemporaryDirectory(dir=self.fixture.base) as directory:
                with patch.object(I.subprocess, "run", **kwargs), self.assertRaises((ValueError, subprocess.TimeoutExpired)):
                    I.export_binding(I.CHECK.Checker(), emitter, arm, evidence, Path(directory))
                self.assertFalse((Path(directory) / "resolved.json").exists())

    def source_bridge_fixture(self):
        arm = self.verify()
        compiler_id = write_json(self.fixture.base / "bridge-compiler.json", self.compiler)
        source_id = write_json(self.fixture.base / "bridge-source.json", {"synthetic": "source receipt"})
        compared = arm["source_inventory"][0]
        derivation_id = write_json(self.fixture.base / "bridge-derivation.json", {
            "module_comparisons": [{"file_comparisons": [{"previous": compared}]}]})
        ram_id = write_json(self.fixture.base / "bridge-ram.json", {"synthetic": "RAM receipt"})
        conditional_id = write_json(self.fixture.base / "bridge-conditional.json", {
            "schema": "atlas.conditional_vls_hw_check.v0", "target_config": "EE290SimConfig",
            "state": "conditional_checks_passed", "rtl_rules_enabled": False, "resolver_bindings": [],
            "conditional_replay_expectations_met": True, "compiler_comparison_matches": True,
            "scope": {"opaque_sram_implementation_qualified": False, "integrated_target_execution_qualified": False},
            "inputs": {"manifest": arm["manifest"], "hardware_ir": arm["hardware_ir"]}})
        packet_id = write_json(self.fixture.base / "bridge-packet.json", {"synthetic": "selected packet identity"})
        packet = {"schema": "atlas.ee290_bounded_applicability.v0", "target_config": "EE290SimConfig",
                  "state": "finite_observations_validated", "qualification": "conditional",
                  "scheduling_qualified": False, "system_qualified": False, "operand_domain_qualified": False,
                  "source_to_simulation_complete": False, "validators": {}, "validator_snapshots": {},
                  "selection": {"manifest": arm["manifest"], "compiler-witness": compiler_id,
                                "source-correspondence": source_id, "verilog-derivation": derivation_id,
                                "implicit-ram-correspondence": ram_id, "conditional-report": conditional_id},
                  "source_correspondence": {"recomputed": "source"},
                  "verilog_content_correspondence": {"recomputed": "Verilog"}}
        return packet_id, packet, {"baseline": arm, "scheduled": copy.deepcopy(arm)}, compiler_id

    def bridge(self, fixture):
        # Source/HW recomputers have their own tests. These mocks isolate the
        # aggregate's required calls and links, never bypassing member hashes.
        with patch.object(I.BOUND, "source_binding", return_value={"recomputed": "source"}) as source:
            with patch.object(I.BOUND, "verilog_content_binding", return_value={"recomputed": "Verilog"}) as verilog:
                result = I.source_bridge(I.CHECK.Checker(), *fixture)
                self.assertEqual(source.call_count, 1)
                self.assertEqual(verilog.call_count, 1)
                return result

    def test_source_bridge_recomputes_both_correspondences_and_binds_both_arms(self):
        fixture = self.source_bridge_fixture()
        result = self.bridge(fixture)
        packet = fixture[1]
        self.assertEqual(result["matched_integrated_module_inputs"], 1)
        self.assertEqual(result["evidence"]["evidence_sha256"], packet["selection"]["conditional-report"]["sha256"])
        self.assertFalse(result["source_to_simulation_complete"])
        for key in ("source_correspondence", "verilog_content_correspondence"):
            changed = copy.deepcopy(fixture)
            changed[1][key] = {"saved": "success"}
            with self.subTest(saved=key), self.assertRaisesRegex(ValueError, "differs from recomputation"):
                self.bridge(changed)
        changed = copy.deepcopy(fixture)
        changed[2]["scheduled"]["source_inventory"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "integrated compared Atlas module"):
            self.bridge(changed)

    def test_source_bridge_rejects_inventory_and_compiler_or_manifest_substitution(self):
        fixture = self.source_bridge_fixture()
        alternative = write_json(self.fixture.base / "unselected-bridge.json", {"unselected": True})
        for mutation, message in [
            (lambda data: data[1]["validator_snapshots"].update(unselected={}), "snapshot inventory"),
            (lambda data: data[2]["baseline"].update(manifest=alternative), "component manifest"),
            (lambda data: data.__setitem__(3, alternative), "compiler witness"),
            (lambda data: data[1]["selection"].pop("implicit-ram-correspondence"), "selections required"),
            (lambda data: data[1].update(source_to_simulation_complete=True), "qualification mismatch"),
        ]:
            changed = list(copy.deepcopy(fixture))
            mutation(changed)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.bridge(changed)

    def test_source_bridge_rejects_unfinished_or_promoted_conditional_evidence(self):
        fixture = self.source_bridge_fixture()
        conditional_id = fixture[1]["selection"]["conditional-report"]
        mutations = [lambda data: data.update(state="running"),
                     lambda data: data.update(target_config="OtherConfig"),
                     lambda data: data.update(rtl_rules_enabled=True),
                     lambda data: data.update(resolver_bindings=[{"unselected": True}]),
                     lambda data: data.update(conditional_replay_expectations_met=False),
                     lambda data: data.update(compiler_comparison_matches=False),
                     lambda data: data["scope"].update(opaque_sram_implementation_qualified=True),
                     lambda data: data["scope"].update(integrated_target_execution_qualified=True),
                     lambda data: data["inputs"]["manifest"].update(sha256="0" * 64)]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index), self.substituted_json(conditional_id["path"], mutate):
                with self.assertRaises(ValueError):
                    self.bridge(fixture)

    def test_source_bridge_rejects_unfinished_or_promoted_bounded_packet(self):
        fixture = self.source_bridge_fixture()
        for field, value in (("state", "prepared_not_executed"), ("qualification", "qualified")):
            changed = copy.deepcopy(fixture)
            changed[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.bridge(changed)

    def run_fixture(self, output):
        arm = self.verify()
        compiler_id = write_json(self.fixture.base / "run-compiler.json", {
            "schema": "atlas.compiler_vls_witness.v0", "qualification": "conditional", **self.compiler})
        numerical_id = write_json(self.fixture.base / "run-numerical.json", {
            "target_config": "EE290SimConfig", "scheduling_qualified": False,
            "state": "both_execution_witnesses_passed", "compiler_receipt": compiler_id,
            "simulator": arm["simulator"], "simulator_build": {"receipt": arm["build_phase"]}})
        launcher = self.fixture.member("launcher.executed.py", "synthetic historical launcher")
        pair_id = write_json(self.fixture.base / "run-pair.json", {
            "schema": "atlas.ee290_vcs_observation_pair.v0", "target_config": "EE290SimConfig",
            "scheduling_qualified": False, "state": "both_numerical_and_boundary_observations_passed",
            "launcher": launcher, "selected_pair": numerical_id, "selected_build_phase": arm["build_phase"],
            "simulator": arm["simulator"], "capture_tool": arm["capture_producer_snapshot"],
            "arms": [{"arm": name, "plan": self.plan_id, "receipt": self.capture_id}
                     for name in ("baseline", "scheduled")]})
        args = SimpleNamespace(output=output, bounded_applicability=None, expected_bounded_applicability_sha256=None,
                               atlas_emit=None, expected_atlas_emit_sha256=None, include_system_memory=False)
        selections = {"pair": pair_id, "baseline-plan": self.plan_id, "scheduled-plan": self.plan_id,
                      "baseline-capture": self.capture_id, "scheduled-capture": self.capture_id}
        for name, selected in selections.items():
            setattr(args, name.replace("-", "_"), Path(selected["path"]))
            setattr(args, "expected_" + name.replace("-", "_") + "_sha256", selected["sha256"])
        return args, arm

    def integrated_run(self, args, arm):
        # This tests aggregate selection and optional orchestration; verify_arm
        # and captured-phase validation are exercised independently.
        with patch.object(I.BOUND, "phase", return_value=self.phase):
            with patch.object(I, "verify_arm", side_effect=lambda *unused: copy.deepcopy(arm)):
                return I.run(args)

    def test_run_publishes_finite_coverage_and_rejects_independent_selection_substitution(self):
        with tempfile.TemporaryDirectory(dir=self.fixture.base) as directory:
            args, arm = self.run_fixture(Path(directory) / "output")
            result = self.integrated_run(args, arm)
            self.assertEqual(result["state"], "finite_integrated_observations_validated")
            self.assertEqual(result["coverage"], {"capture_record_links": True, "boundary_events": True,
                                                "source_hw_content_bridge": False, "fresh_resolver_event_binding": False,
                                                "system_memory_windows": False})
            self.assertFalse(result["scheduling_qualified"])
            self.assertFalse(result["system_qualified"])
            self.assertFalse(result["source_to_simulation_complete"])
            self.assertEqual(json.loads((args.output / "report.json").read_text())["state"], result["state"])
        with tempfile.TemporaryDirectory(dir=self.fixture.base) as directory:
            args, arm = self.run_fixture(Path(directory) / "output")
            alternative = write_json(self.fixture.base / "run-other-plan.json", {"different": "plan"})
            args.scheduled_plan = Path(alternative["path"])
            args.expected_scheduled_plan_sha256 = alternative["sha256"]
            with self.assertRaisesRegex(ValueError, "independently selected"):
                self.integrated_run(args, arm)
            self.assertFalse((args.output / "report.json").exists())

    def test_run_source_export_and_memory_options_bind_every_arm(self):
        with tempfile.TemporaryDirectory(dir=self.fixture.base) as directory:
            args, arm = self.run_fixture(Path(directory) / "output")
            packet_id, packet, _, _ = self.source_bridge_fixture()
            packet_id = write_json(Path(packet_id["path"]), packet)
            args.bounded_applicability = Path(packet_id["path"])
            args.expected_bounded_applicability_sha256 = packet_id["sha256"]
            args.atlas_emit = self.fixture.base / "emitter"
            args.expected_atlas_emit_sha256 = identity(args.atlas_emit)["sha256"]
            args.include_system_memory = True
            source = {"evidence": {"evidence_sha256": "1" * 64}}
            memory = SimpleNamespace(__file__=ROOT / "tools/check-ee290-system-memory-observation.py",
                                     analyze_trace=lambda *unused: {
                                         "schema": "atlas.ee290_system_memory_events.v0", "target_config": "EE290SimConfig",
                                         "scheduling_qualified": False, "boundary_observations": arm["boundary_observations"]})
            with patch.object(I, "source_bridge", return_value=source) as bridge:
                with patch.object(I, "export_binding", return_value={"synthetic": "fresh event binding"}) as export:
                    with patch.object(I, "load", return_value=memory) as load:
                        result = self.integrated_run(args, arm)
            self.assertEqual(bridge.call_count, 1)
            self.assertEqual(export.call_count, 2)
            self.assertEqual(load.call_count, 1)
            self.assertTrue(all(result["coverage"].values()))
            for call in export.call_args_list:
                self.assertEqual(call.args[3], source["evidence"])
                self.assertEqual(call.args[2]["program"]["word_values"], arm["program"]["word_values"])
            for name in ("baseline", "scheduled"):
                self.assertIn("resolver_export", result["arms"][name])
                self.assertIn("system_memory_observations", result["arms"][name])

    def test_run_rejects_unpaired_options_and_export_without_source(self):
        emitter = identity(self.fixture.base / "emitter")
        cases = [{"bounded_applicability": self.fixture.base / "missing-packet"},
                 {"expected_bounded_applicability_sha256": "1" * 64},
                 {"atlas_emit": Path(emitter["path"])},
                 {"expected_atlas_emit_sha256": emitter["sha256"]},
                 {"atlas_emit": Path(emitter["path"]), "expected_atlas_emit_sha256": emitter["sha256"]}]
        for fields in cases:
            with self.subTest(fields=fields), tempfile.TemporaryDirectory(dir=self.fixture.base) as directory:
                args, arm = self.run_fixture(Path(directory) / "output")
                for key, value in fields.items():
                    setattr(args, key, value)
                with patch.object(I, "export_binding", side_effect=AssertionError("unexpected resolver execution")):
                    with self.assertRaisesRegex(ValueError, "must be paired|requires selected bounded"):
                        self.integrated_run(args, arm)
                self.assertFalse((args.output / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
