"""Focused full Tiny model geometry and final-state admission controls."""
import copy
from pathlib import Path
import sys
import struct
import unittest
from unittest import mock

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/host_iss")]
import tiny_full_contract as tiny


def fixture():
    geometry = tiny.plan(2)
    kernels = []
    for index, item in enumerate(geometry):
        m, k, n = item["physical_mkn"]
        load = (m*k*(n//32)+k*n if k*n <= 262144 else
                m*k+k*n*(m//32) if m*k <= 262144 else (m//32)*(n//32)*(2*k//32*1024))
        kernels.append({**item, "index": index, "dram_base": 0x100000000,
            "output_offset": m*k+k*n+256, "total_dram_bytes": m*k+k*n+512+2*m*n,
            "requested_dma": {"load_bytes": load, "store_bytes": 2*m*n},
            "A_scale": -7, "B_scale": -8, "rescale": -15})
    references = []
    for name, (shape, dtype) in tiny.reference_geometry(2).items():
        size = 2 if dtype == "uint16" else 4
        for extent in shape:
            size *= extent
        references.append({"actual_symbol": name, "expected_symbol": name + "_expected",
                           "shape": shape, "dtype": dtype, "bytes": size})
    parameters = {name: {"shape": shape, "dtype": dtype, "symbol": f"tiny_full_param_{index:03d}"}
                  for index, (name, (shape, dtype)) in enumerate(sorted(tiny.parameter_geometry().items()))}
    data = {"schema": "atlas.tiny_full_contract.v1", "cycle_cap": tiny.CYCLE_CAP,
        "decode_count": 2, "kernel_count": 840, "projection_calls": 465, "kernels": kernels,
        "host_parameters": parameters, "reference_buffers": references, "greedy_tokens": [17, 18, 19],
        "mailbox_symbol": "tiny_full_mailbox", "panel_symbol": "tiny_full_panels",
        "state_symbol": "tiny_full_state_global", "context_symbol": "tiny_full_projection_state"}
    return data, {"kernel_count": 840, "kernels": copy.deepcopy(kernels)}


class TinyFullContractTest(unittest.TestCase):
    def test_actual_reference_must_be_writable_loaded_and_disjoint(self):
        image = mock.Mock()
        image.symbols = {"actual": {"address": 0x1800, "size": 4},
                         "expected": {"address": 0x1100, "size": 0}}
        image.segments = [{"address": 0x1000, "memory_bytes": 0x1000, "flags": 6}]
        items = [{"actual_symbol": "actual", "expected_symbol": "expected", "bytes": 4}]
        self.assertEqual(tiny.reference_locations(image, items), [(0x1800, 4)])
        for address, flags in ((0x3000, 6), (0x1800, 4), (0x1100, 6), (0x1102, 6)):
            image.symbols["actual"]["address"] = address
            image.segments[0]["flags"] = flags
            with self.subTest(address=address, flags=flags), self.assertRaises(ValueError):
                tiny.reference_locations(image, items)
        image.symbols["actual"]["address"] = 0x1800
        image.segments[0]["flags"] = 6
        with self.assertRaises(ValueError):
            tiny.reference_locations(image, items, [(0x1802, 8)])
        image.bytes_at.assert_not_called()

    def test_raw_bf16_nonfinite_reference_refused(self):
        self.assertTrue(tiny.finite_bf16(struct.pack("<4H", 0, 0x8000, 0x7f7f, 0xff7f)))
        for word in (0x7f80, 0xff80, 0x7fc0, 0xffc1):
            with self.subTest(word=word):
                self.assertFalse(tiny.finite_bf16(struct.pack("<H", word)))
        self.assertFalse(tiny.finite_bf16(b"\x00"))

    def test_complete_22_layer_two_decode_geometry(self):
        data, provenance = fixture()
        plan = tiny.validate_plan(data, provenance)
        self.assertEqual(len(plan), 840)
        self.assertEqual(len(data["host_parameters"]), 49)
        self.assertEqual(len(data["reference_buffers"]), 137)
        self.assertEqual(plan[0]["physical_mkn"], [128, 2048, 2048])
        self.assertEqual(plan[263]["site"], "model.layers.21.mlp.down_proj")
        self.assertEqual(plan[264]["site"], "lm_head")
        self.assertEqual(plan[279]["columns"], [30720, 32000])
        self.assertEqual(plan[280]["physical_mkn"], [32, 2048, 2048])
        self.assertEqual(plan[-1]["step"], 2)

    def test_missing_or_wrong_layer_role_geometry_dma_scale_rejected(self):
        mutations = (
            lambda d: d["kernels"].pop(),
            lambda d: d["kernels"][12].update(site="model.layers.0.self_attn.q_proj"),
            lambda d: d["kernels"][279].update(columns=[30720, 31968]),
            lambda d: d["kernels"][280].update(physical_mkn=[128, 2048, 2048]),
            lambda d: d["kernels"][0]["requested_dma"].update(load_bytes=32),
            lambda d: d["kernels"][0].update(rescale=-14),
            lambda d: d.update(projection_calls=464),
            lambda d: d.update(decode_count=True),
        )
        for mutate in mutations:
            data, provenance = fixture()
            mutate(data)
            provenance["kernels"] = copy.deepcopy(data["kernels"])
            with self.subTest(mutation=mutate), self.assertRaises((ValueError, KeyError)):
                tiny.validate_plan(data, provenance)

    def test_all_steps_all22_cache_logits_tokens_parameters_required(self):
        mutations = (
            lambda d: d["reference_buffers"].pop(),
            lambda d: d["reference_buffers"][0].update(shape=[1, 4, 127, 64]),
            lambda d: d["reference_buffers"][0].update(dtype="float32"),
            lambda d: d["reference_buffers"][0].update(actual_symbol="tiny_full_cache_00_22_keys"),
            lambda d: d["host_parameters"].pop("model.layers.21.post_attention_layernorm.weight"),
            lambda d: d["host_parameters"]["model.embed_tokens.weight"].update(shape=[31999, 2048]),
            lambda d: d.update(greedy_tokens=[17, 18, 32000]),
        )
        for mutate in mutations:
            data, provenance = fixture()
            mutate(data)
            with self.subTest(mutation=mutate), self.assertRaises((ValueError, KeyError)):
                tiny.validate_plan(data, provenance)

    def test_trap_and_unknown_stop_fail_before_read_or_output(self):
        data, _ = fixture()
        remote = mock.Mock()
        for pc in (0x200, 0x300):
            remote.reg.return_value = pc
            with mock.patch.object(tiny, "contract", return_value=data), \
                    mock.patch.object(tiny, "stop_sites", return_value={0x100: 4, 0x200: 2}):
                with self.assertRaises(ValueError):
                    tiny.verify_completion(remote, mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock())
            remote.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
