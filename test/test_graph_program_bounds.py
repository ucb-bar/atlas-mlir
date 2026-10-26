"""Unconditional conservative program extents; no assembler/model execution needed."""

from pathlib import Path
import json
import re
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools/chipyard"))
import atlas_program
from atlas_host_build import GRAPH_PROGRAM_WORDS, validate_graph_program, wrap
from atlas_host_layout import ATLAS_IMEM_BYTES
import resnet_segments
import build_resnet_graph_host

# The canonical npu-model/npu_model/util/converter.py expands only LI (at
# most LUI+ADDI). Its other admitted instructions occupy exactly one word.
# Keep the emitted vocabulary explicit so new pseudo-ops cannot silently pass.
ARITY = {"li": 2, "dma.config": 2, "vli.all": 2, "delay": 1, "add": 3,
         "dma.load": 4, "dma.wait": 1, "addi": 3, "beq": 3, "nop": 0,
         "jal": 2, "vload": 2, "vtrpose.xlu": 2, "vstore": 2,
         "vmatpush.weight.mxu0": 2, "vmatmul.mxu0": 3,
         "vmatmul.acc.mxu0": 3, "vmatpop.bf16.acc.mxu0": 2,
         "vmul.bf16": 3, "dma.store": 4, "ecall": 0}


def raw_word_upper_bound(source):
    words = 0
    for original in source.splitlines():
        line = original.split("#", 1)[0].strip()
        if not line:
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*:", line):
            continue
        op, *args = [token for token in re.split(r"[\s,]+", line) if token]
        if op not in ARITY or len(args) != ARITY[op]:
            raise ValueError("unknown instruction or malformed arity: " + line)
        if op == "li":
            if not re.fullmatch(r"x(?:[0-9]|[12][0-9]|3[01])", args[0]):
                raise ValueError("invalid LI register")
            value = int(args[1], 0)
            if not -(1 << 31) <= value < 1 << 32:
                raise ValueError("LI immediate outside 32-bit domain")
            words += 2
        else:
            words += 1
    return words


def program(words):
    return struct.pack("<I", 0x13) * (words - 1) + struct.pack("<I", 0x73)


class ProgramBoundTest(unittest.TestCase):
    def test_all_57_dispatch_shapes_fit_hardware_and_graph_exception_is_explicit(self):
        self.assertEqual(len(atlas_program._TRAFFIC_REUSE_MODULES), 57)
        graph_exceptions = set()
        for m, k, n in atlas_program._TRAFFIC_REUSE_MODULES:
            with self.subTest(shape=(m, k, n)):
                source = atlas_program.assembly(m, k, n, m*k+k*n+256, 0, traffic_reuse=True)
                bound = raw_word_upper_bound(source)
                self.assertLessEqual(bound + 5, ATLAS_IMEM_BYTES // 4)
                if bound > GRAPH_PROGRAM_WORDS:
                    graph_exceptions.add((m, k, n))
        self.assertEqual(graph_exceptions, {(12544, 160, 64)})

    def test_counter_comments_labels_and_li_domain(self):
        self.assertEqual(raw_word_upper_bound("# comment\nstart: # label\nli x31, 0xffffffff\nnop\necall"), 4)
        for source in (".word 0", "unknown x1", "li", "li x1", "li x1, 1, 2",
                       "li x32, 1", "li m0, 1", "li x1, invalid", "li x1, 0x100000000",
                       "li x1, -2147483649", "nop x1", "add x1, x2", "bad label:"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                raw_word_upper_bound(source)
        for immediate in ("-2147483648", "2147483647", "4294967295"):
            self.assertEqual(raw_word_upper_bound("li x0, " + immediate), 2)

    def test_python_cap_matches_runtime_enum(self):
        header = (ROOT / "runtime/atlas_resnet_graph_device.h").read_text()
        cap, = re.findall(r"\bATLAS_GRAPH_PROGRAM_WORDS\s*=\s*(\d+)", header)
        self.assertEqual(GRAPH_PROGRAM_WORDS, int(cap))

    def test_actual_encoded_boundary_and_descriptor_lies(self):
        raw = program(GRAPH_PROGRAM_WORDS)
        self.assertEqual(validate_graph_program(raw), GRAPH_PROGRAM_WORDS)
        self.assertEqual(validate_graph_program(raw, GRAPH_PROGRAM_WORDS), GRAPH_PROGRAM_WORDS)
        self.assertEqual(len(wrap(raw)), len(raw) + 20)
        for declared in (0, GRAPH_PROGRAM_WORDS - 1, GRAPH_PROGRAM_WORDS + 1, None, True, float(GRAPH_PROGRAM_WORDS)):
            with self.subTest(declared=declared), self.assertRaises(ValueError):
                validate_graph_program(raw, declared)
        for declared in (GRAPH_PROGRAM_WORDS, GRAPH_PROGRAM_WORDS + 1):
            with self.subTest(oversize_declared=declared), self.assertRaises(ValueError):
                validate_graph_program(program(GRAPH_PROGRAM_WORDS + 1), declared)

    def test_structural_admission_is_preserved(self):
        for raw in (b"", b"x", struct.pack("<I", 0x13), struct.pack("<2I", 0x73, 0x73),
                    struct.pack("<2I", 0x00100073, 0x73), struct.pack("<2I", 0xc1001073, 0x73)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_graph_program(raw)

    def test_full_host_rejects_bad_raw_or_descriptor_before_linking(self):
        with tempfile.TemporaryDirectory(prefix="atlas-full-program-bounds-") as directory:
            root = Path(directory)
            names = ("generated", "replay-gate", "reference", "templates", "rescale-table",
                     "atlas-root", "clang", "lld", "out")
            paths = {name: root / name for name in names}
            graphpath = paths["generated"] / "graph.json"
            receiptpath = paths["reference"] / "receipt.json"
            gate = {"status": "PASS", "source_bookends_equal": True,
                    "all_54_device_inputs_and_scales_exact": True, "final_1000_logit_words_exact": True,
                    "inputs_sha256": {str(graphpath): "graph", str(receiptpath): "reference"}}
            reference = {"nodes": [{"payload": {"path": str(root / "entry"), "sha256": "entry"}}],
                         "final_logits": {"sha256": "logits"}}
            for count, declared in ((8193, 8193), (8193, 8192), (4, 3), (4, None)):
                template = {"base_rescale_exponent": 0, "program_words": declared,
                            "program": {"path": str(root / "program"), "sha256": "program"}}
                graph = {"input_sha256": {str(paths["templates"]): "templates"},
                         "generated_sha256": {}, "params_sha256": {},
                         "geometry_admission": "all operators/shapes/arity/topology/54 contractions/groups 1 proven before generation",
                         "kernels": [{"template": template, "kernel_index": i} for i in range(54)]}
                payloads = {paths["replay-gate"]: json.dumps(gate).encode(),
                            graphpath: json.dumps(graph).encode(), receiptpath: json.dumps(reference).encode(),
                            paths["templates"]: json.dumps({"kernels": [template] * 54,
                                                          "rescale_table_sha256": "scale"}).encode(),
                            root / "entry": bytes(3 * 224 * 224 * 4),
                            paths["reference"] / "target-logits.f32.bin": bytes(4000),
                            paths["rescale-table"]: struct.pack("<254I", *[((e + 127) << 23) | 0x25F
                                                                         for e in range(-126, 128)]),
                            root / "program": program(count)}
                argv = ["build_resnet_graph_host.py", *(arg for name in names
                         for arg in ("--" + name, str(paths[name])))]
                with self.subTest(count=count, declared=declared), \
                     mock.patch.object(sys, "argv", argv), \
                     mock.patch.object(build_resnet_graph_host, "Admission") as admission, \
                     mock.patch.object(build_resnet_graph_host.subprocess, "run") as linker:
                    admission.return_value.read.side_effect = lambda path, *args: payloads.get(Path(path), b"")
                    with self.assertRaises(ValueError):
                        build_resnet_graph_host.main()
                    linker.assert_not_called()
                    self.assertFalse(paths["out"].exists())

    def test_segment_rejects_bad_raw_or_descriptor_before_creating_outputs(self):
        with tempfile.TemporaryDirectory(prefix="atlas-program-bounds-") as directory:
            root = Path(directory)
            args = SimpleNamespace(host=root / "host", capture=root / "capture", out=root / "out", segments=[0])
            for count, declared in ((8193, 8193), (8193, 8192), (4, 3), (4, None)):
                descriptor = "" if declared is None else f".program_words={declared},"
                assets = ("graph_kernels[54]={\n{" + descriptor + "}\n};").encode()
                def bound_file(path, digests):
                    return assets if path.name == "assets.c" else program(count) if path.suffix == ".bin" else b""
                with self.subTest(count=count, declared=declared), \
                     mock.patch.object(resnet_segments, "plan", return_value={"segments": [{}]}), \
                     mock.patch.object(resnet_segments, "capture_manifest", return_value=({}, "capture", set())), \
                     mock.patch.object(resnet_segments, "bound_host_file", side_effect=bound_file), \
                     mock.patch.object(resnet_segments.subprocess, "run") as linker:
                    with self.assertRaises(ValueError):
                        resnet_segments.build(args)
                    linker.assert_not_called()
                    self.assertFalse(args.out.exists())


if __name__ == "__main__":
    unittest.main()
