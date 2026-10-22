"""CPU-only Smol contract tests; no QEMU or model execution claim."""

import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools" / "host_iss"))
import smol_contract as smol


class Image:
    def __init__(self):
        self.symbols = {}
        self.instructions_by_name = {
            "atlas_workload_main": [(0x100, 4, 0x10500073), (0x108, 4, 0x10500073)],
            "atlas_block_trap_report": [(0x200, 4, 0x10500073)]}

    def instructions(self, name):
        return self.instructions_by_name[name]


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.image = Image()
        self.memory = {}
        self.payloads = {}
        self.image.symbols = {
            "smol_workload_mailbox": dict(address=0x1000, size=120),
            "tohost": dict(address=0x2000, size=8),
            "smol_selected_state": dict(address=0x3000, size=32)}
        mailbox = bytearray(120)
        struct.pack_into("<6I", mailbox, 0, 0x534d4f4c, 1, 0, 7, 6, 0)
        struct.pack_into("<7I", mailbox, 24, *range(1, 8))
        struct.pack_into("<7I", mailbox, 52, *range(100, 107))
        self.memory[0x1000] = bytes(mailbox)
        self.memory[0x2000] = struct.pack("<Q", 1)
        self.memory[0x3000] = struct.pack("<8I", 1, 7, 6, 0, 1, 1, 0, 0)
        for index, (name, actual, size) in enumerate(smol.FINAL):
            address = 0x10000 + index * 0x100000
            self.image.symbols[actual] = dict(address=address, size=size)
            self.payloads["data/" + name + ".bin"] = bytes([index + 1]) * size
            self.memory[address] = self.payloads["data/" + name + ".bin"]
        self.remote = SimpleNamespace(read=lambda address, size: self.memory[address][:size])
        self.bundle = SimpleNamespace(read=lambda name: self.payloads[name])
        self.device = SimpleNamespace(completed=[dict(index=i, sequence=i + 1, exact=True, drained=True,
                                                      reset_count=1) for i in range(7)])
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)

    def verify(self):
        with patch.object(smol, "validate_bundle"):
            return smol.verify_completion(self.remote, self.image, self.device, self.bundle, self.out)

    def word(self, offset, value):
        raw = bytearray(self.memory[0x1000])
        struct.pack_into("<I", raw, offset, value)
        self.memory[0x1000] = bytes(raw)

    def test_exact_completion_saves_real_buffers(self):
        result = self.verify()
        self.assertTrue(result["all_outputs_exact"] and result["all_caches_exact"])
        self.assertEqual(result["tohost"], 1)
        for _, actual, _ in smol.FINAL:
            address = self.image.symbols[actual]["address"]
            self.assertEqual((self.out / (actual + ".bin")).read_bytes(), self.memory[address])

    def test_each_mailbox_status_field_is_required(self):
        original = self.memory[0x1000]
        for offset in (0, 4, 8, 12, 16, 20, 80, 84, 88, 92, 96, 104, 112):
            with self.subTest(offset=offset):
                self.memory[0x1000] = original
                self.word(offset, int.from_bytes(original[offset:offset+4], "little") ^ 1)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_each_sequence_and_cycle_is_required(self):
        original = self.memory[0x1000]
        for offset in range(24, 80, 4):
            with self.subTest(offset=offset):
                self.memory[0x1000] = original
                self.word(offset, 0)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_truncated_tohost_is_rejected(self):
        self.memory[0x2000] = b"\1"
        with self.assertRaisesRegex(ValueError, "tohost"):
            self.verify()

    def test_tohost_failure_is_rejected(self):
        self.memory[0x2000] = struct.pack("<Q", 511)
        with self.assertRaisesRegex(ValueError, "tohost"):
            self.verify()

    def test_every_block_state_field_is_required(self):
        original = self.memory[0x3000]
        for offset in range(0, 32, 4):
            with self.subTest(offset=offset):
                raw = bytearray(original)
                raw[offset] ^= 1
                self.memory[0x3000] = bytes(raw)
                with self.assertRaisesRegex(ValueError, "block completion"):
                    self.verify()

    def test_live_output_and_both_caches_are_checked(self):
        for _, actual, _ in smol.FINAL:
            with self.subTest(actual=actual):
                address = self.image.symbols[actual]["address"]
                original = self.memory[address]
                self.memory[address] = original[:-1] + bytes([original[-1] ^ 1])
                with self.assertRaisesRegex(ValueError, "live output/cache"):
                    self.verify()
                self.memory[address] = original

    def test_truncated_live_output_is_rejected(self):
        address = self.image.symbols["output"]["address"]
        self.memory[address] = self.memory[address][:-1]
        with self.assertRaises(ValueError):
            self.verify()

    def test_device_ledger_cannot_skip_or_fake_execution(self):
        original = copy.deepcopy(self.device.completed)
        mutations = [dict(index=1), dict(sequence=2), dict(exact=False), dict(drained=False), dict(reset_count=2)]
        for changes in mutations:
            with self.subTest(changes=changes):
                self.device.completed = copy.deepcopy(original)
                self.device.completed[0].update(changes)
                with self.assertRaisesRegex(ValueError, "ledger"):
                    self.verify()
        self.device.completed = original[:-1]
        with self.assertRaisesRegex(ValueError, "launch count"):
            self.verify()

    def test_fixed_live_symbol_extent_is_required(self):
        self.image.symbols["output"]["size"] -= 2
        with self.assertRaisesRegex(ValueError, "symbol extent"):
            self.verify()


class SyntheticAdmission:
    """Complete fixed-shape synthetic inputs with independent ELF byte views."""
    def __init__(self):
        self.image = Image()
        self.image.raw = b"synthetic ELF"
        self.image.entry = 0x80000000
        self.image.symbols = {}
        self.embedded = {}
        self.image.bytes_at = lambda address, size: self.embedded[address][:size]
        self.payloads = {"host/host.elf": self.image.raw}
        self.sums = {"host/host.elf": smol.digest(self.image.raw)}
        self.build = dict(status="passed_RV64GC_LP64D_ELF_build_only", ELF_executed=False,
                          source_bookends_unchanged=True, elf=dict(sha256=self.sums["host/host.elf"]))
        self.kernel_count = 7
        self.kernels = []
        self.value = dict(schema="atlas.smol_selected_layer1.epoch3_launch_data.v1",
                          ELF=dict(sha256=self.sums["host/host.elf"]), symbols={}, assets={},
                          ordered_projections=[], constant_parameters=dict(input_epsilon=1e-5,
                          post_epsilon=1e-5, first_table_position=-1, table_rows=8193))
        self.provenance = dict(workload="smol", contract="host/contract.json")
        self.next_address = 0x80010000
        for name, size in (("smol_workload_mailbox", 120), ("smol_selected_state", 32), ("tohost", 8)):
            self.add_symbol(name, size)
            self.value["symbols"][name] = self.image.symbols[name].copy()
        extents = dict(input=339840, pad=177, attention_boundaries=177, input_norm_weight=1920,
                       post_norm_weight=1920, cosine=1048704, sine=1048704, expected_output=339840,
                       expected_key_cache=113280, expected_value_cache=113280)
        for name, size in extents.items():
            raw = bytes(size)
            self.payloads["data/" + name + ".bin"] = raw
            symbol = "smol_asset_" + name
            address = self.add_symbol(symbol, 0, raw)
            self.value["assets"][name] = dict(file=dict(bytes=size, sha256=smol.digest(raw)),
                ELF_constant_symbol=symbol, ELF_constant_address=address)
        for name, actual, size in smol.FINAL:
            address = self.add_symbol(actual, size)
            self.value["assets"][name].update(actual_C_buffer_symbol=actual,
                                              actual_C_buffer=dict(address=address, size=size))
        for index, (role, wi, wo) in enumerate(zip(smol.ROLES, smol.INPUT_WIDTHS, smol.OUTPUT_WIDTHS)):
            source = b"ecall\n"
            original = struct.pack("<I", 0x73)
            weights = bytes(wi * wo)
            offset = 192 * wi + len(weights) + 256
            expected = bytes(192 * wo * 2)
            initial = bytes(offset + len(expected) + 256)
            paths = dict(output_offset=offset, dram_base=0x100000000,
                         requested_dma=dict(load_bytes=192 * wi + len(weights), store_bytes=len(expected)))
            projection = dict(index=index, role=role, physical_rows=192, native_rows=177,
                              input_width=wi, output_width=wo, output_offset=offset,
                              dram_bytes=len(initial), requested_dma=paths["requested_dma"].copy())
            for key, record, raw in (("source", "program_S", source), ("original", "original_program_bin", original),
                                     ("initial", "initial_dram", initial),
                                     ("expected", "expected_physical_output", expected)):
                name = str(index) + "/" + key
                self.payloads[name] = raw
                paths[key] = name
                projection[record] = dict(bytes=len(raw), sha256=smol.digest(raw))
            ps, ws = "program_" + str(index), "weights_" + str(index)
            self.add_symbol(ps, 0, original)
            address = self.add_symbol(ws, 0, weights)
            projection.update(ELF_original_program_symbol=ps, ELF_weight_symbol=ws,
                              ELF_weight_address=address, weights=dict(sha256=smol.digest(weights)))
            self.kernels.append(paths)
            self.value["ordered_projections"].append(projection)
        self.refresh()

    def add_symbol(self, name, size, raw=None):
        address = self.next_address
        self.next_address += max(size, len(raw) if raw is not None else 0) + 64
        self.image.symbols[name] = dict(address=address, size=size, section=2)
        if raw is not None:
            self.embedded[address] = raw
        return address

    def refresh(self):
        raw = json.dumps(self.value).encode()
        self.payloads["host/contract.json"] = raw
        self.provenance["contract_sha256"] = smol.digest(raw)

    def read(self, name):
        return self.payloads[name]

    def kernel_paths(self, index):
        return self.kernels[index]


class AdmissionTests(unittest.TestCase):
    def bundle(self, value=None):
        value = value or dict(schema="atlas.smol_selected_layer1.epoch3_launch_data.v1")
        raw = json.dumps(value).encode()
        return SimpleNamespace(provenance=dict(workload="smol", contract="host/contract.json",
            contract_sha256=smol.digest(raw)), kernel_count=7, read=lambda name: raw)

    def test_complete_synthetic_admission(self):
        bundle = SyntheticAdmission()
        smol.validate_bundle(bundle, bundle.image)

    def test_descriptor_and_embedded_payload_mutations_are_rejected(self):
        bundle = SyntheticAdmission()
        original = copy.deepcopy(bundle.kernels[0])
        for changes in (dict(output_offset=1), dict(dram_base=0),
                        dict(requested_dma=dict(load_bytes=0, store_bytes=1))):
            with self.subTest(changes=changes):
                bundle.kernels[0] = dict(original, **changes)
                with self.assertRaises(ValueError):
                    smol.validate_bundle(bundle, bundle.image)
        bundle.kernels[0] = original
        for symbol in ("smol_asset_expected_output", "program_0", "weights_0"):
            with self.subTest(symbol=symbol):
                address = bundle.image.symbols[symbol]["address"]
                raw = bundle.embedded[address]
                bundle.embedded[address] = bytes([raw[0] ^ 1]) + raw[1:]
                with self.assertRaises(ValueError):
                    smol.validate_bundle(bundle, bundle.image)
                bundle.embedded[address] = raw

    def test_source_payload_geometry_and_role_mutations_are_rejected(self):
        bundle = SyntheticAdmission()
        path = bundle.kernels[0]["initial"]
        original = bundle.payloads[path]
        bundle.payloads[path] = bytes([1]) + original[1:]
        with self.assertRaisesRegex(ValueError, "projection payload"):
            smol.validate_bundle(bundle, bundle.image)
        bundle.payloads[path] = original
        for key, value in (("role", "mlp.down_proj"), ("native_rows", 192), ("output_width", 320)):
            with self.subTest(key=key):
                projection = bundle.value["ordered_projections"][0]
                before = projection[key]
                projection[key] = value
                bundle.refresh()
                with self.assertRaises(ValueError):
                    smol.validate_bundle(bundle, bundle.image)
                projection[key] = before
        bundle.refresh()

    def test_source_contract_is_digest_bound(self):
        bundle = self.bundle()
        self.assertIn("schema", smol.contract(bundle))
        bundle.provenance["contract_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            smol.contract(bundle)

    def test_wrong_schema_workload_or_count_is_rejected(self):
        wrong_schema = self.bundle(dict(schema="other"))
        wrong_count = self.bundle()
        wrong_count.kernel_count = 6
        for bundle in (wrong_schema, wrong_count):
            with self.assertRaises(ValueError):
                smol.contract(bundle)
        bundle = self.bundle()
        bundle.provenance["workload"] = "tiny"
        with self.assertRaises(ValueError):
            smol.contract(bundle)

    def test_build_only_admission_requires_original_success_and_bookends(self):
        for build in ({}, dict(status="passed_RV64GC_LP64D_ELF_build_only", ELF_executed=True,
                               source_bookends_unchanged=True),
                      dict(status="passed_RV64GC_LP64D_ELF_build_only", ELF_executed=False,
                           source_bookends_unchanged=False)):
            bundle = self.bundle()
            bundle.build = build
            with self.assertRaisesRegex(ValueError, "build status"):
                smol.validate_bundle(bundle, Image())

    def test_stop_sites_include_success_and_trap(self):
        self.assertEqual(smol.stop_sites(Image()), {0x100: 4, 0x108: 4, 0x200: 4})
        for name in ("atlas_workload_main", "atlas_block_trap_report"):
            image = Image()
            image.instructions_by_name[name] = [(0x300, 4, 0x13)]
            with self.assertRaisesRegex(ValueError, "WFI"):
                smol.stop_sites(image)


if __name__ == "__main__":
    unittest.main()
