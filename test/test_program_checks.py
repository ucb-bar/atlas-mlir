"""M6 mandatory static checks, including direct recipe entry points."""

from importlib import import_module
from hashlib import sha256
from pathlib import Path
import re
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_program
from atlas_epilogue import Epilogue, Step, TileInput
from atlas_program_checks import check_matrix_registers, check_scalar_scratch
from atlas_resnet_input import DirectInput, relocate_assembly


class OddSource(Epilogue):
    def apply(self, output_reg):
        return ["vexp.bf16 m2, m3"]


class ProgramChecks(unittest.TestCase):
    def test_no_hook_raw_and_relocated_sources_are_unchanged(self):
        # Captured after the direct-input baseline fix, before per-recipe remap.
        raw, relocated = sha256(), sha256()
        for m, k, n in sorted(atlas_program._TRAFFIC_REUSE_MODULES):
            for exponent in (0, -3):
                source = atlas_program.assembly(m, k, n, m * k + k * n + 256, exponent, traffic_reuse=True)
                prefix = f"{m},{k},{n},{exponent}\n".encode()
                raw.update(prefix + source.encode())
                relocated.update(prefix + relocate_assembly(source, (1 << 32) + 0x400000).encode())
        self.assertEqual(raw.hexdigest(), "60694b044d95243bada903635fdf6ed738286dce7850131c68b048de6a1d158f")
        self.assertEqual(relocated.hexdigest(), "5e329d4546f7e9f1d5dfa01de823c37fe141e599f8127ea76b91b9563e608e1a")

    def test_all_recipes_select_free_scratch_and_reject_other_explicit_pairs(self):
        for (m, k, n), name in atlas_program._TRAFFIC_REUSE_MODULES.items():
            module = import_module(name)
            pair = module.TINY_SCRATCH if name.rsplit("_", 1)[-1] == "tiny" or name.endswith("tiny_breadth") else module.SCALAR_SCRATCH
            expected = (27, 28) if name.endswith("resident") else (10, 11)
            self.assertEqual(pair, expected)
            offset = m * k + k * n + 256
            operand = TileInput(offset + 2 * m * n + 256, "channel")
            default = Epilogue(m, n, offset, (Step("add", operand),))
            selected = Epilogue(m, n, offset, (Step("add", operand),), scratch_registers=pair)
            naked = module.assembly(m, k, n, offset, 0)
            check_scalar_scratch(naked, pair)
            with self.subTest(shape=(m, k, n), module=name):
                source = module.assembly(m, k, n, offset, 0, epilogue=default)
                self.assertEqual(source, module.assembly(m, k, n, offset, 0, epilogue=selected))
                self.assertEqual(source, atlas_program.assembly(m, k, n, offset, 0, traffic_reuse=True, epilogue=default))
                self.assertEqual(default.scratch_registers, (30, 31))
                for register in (30, 31):
                    self.assertEqual(len(re.findall(rf"\bx{register}\b", source)), len(re.findall(rf"\bx{register}\b", naked)))
                for supplied in ((22, 23), (11, 10), (28, 27)):
                    rejected = Epilogue(m, n, offset, scratch_registers=supplied)
                    with self.assertRaisesRegex(ValueError, "scalar scratch must be default"):
                        module.assembly(m, k, n, offset, 0, epilogue=rejected)
                    with self.assertRaisesRegex(ValueError, "scalar scratch must be default"):
                        atlas_program.assembly(m, k, n, offset, 0, traffic_reuse=True, epilogue=rejected)

    def test_resident_and_stream_loop_variants_leave_selected_pair_free(self):
        for name, shapes, pair in (("resident", ((64, 32, 32), (128, 64, 96)), (27, 28)),
                                   ("stream", ((32, 32, 32), (96, 64, 96), (128, 3072, 768)), (10, 11))):
            module = import_module("atlas_program_reuse_" + name)
            for m, k, n in shapes:
                offset = m * k + k * n + 256
                naked = module.assembly(m, k, n, offset, 0)
                check_scalar_scratch(naked, pair)
                hook = Epilogue(m, n, offset, (Step("gelu_tanh"),), scratch_registers=pair)
                module.assembly(m, k, n, offset, 0, epilogue=hook)

    def test_odd_binary_sources_and_destinations(self):
        for op in ("vadd", "vsub", "vmul", "vminimum", "vmaximum"):
            for operands in ("m1, m2, m4", "m0, m3, m4", "m0, m2, m5"):
                with self.subTest(op=op, operands=operands), self.assertRaisesRegex(ValueError, "even"):
                    check_matrix_registers(f"{op}.bf16 {operands}\n")

    def test_odd_unary_sources_and_destinations(self):
        for op in ("vexp.bf16", "vrecip.bf16", "vrelu.bf16", "vmov", "vredsum.row.bf16"):
            for operands in ("m1, m2", "m0, m3"):
                with self.subTest(op=op, operands=operands), self.assertRaisesRegex(ValueError, "even"):
                    check_matrix_registers(f"{op} {operands}\n")

    def test_unary_arity(self):
        for op in ("vexp.bf16", "vrecip.bf16", "vmov", "vredmax.row.bf16"):
            for operands in ("m2", "m2, m0, m32"):
                with self.subTest(op=op, operands=operands), self.assertRaisesRegex(ValueError, "two registers"):
                    check_matrix_registers(f"{op} {operands}")

    def test_pair_immediates_mxu_and_mixed_formats(self):
        for source in ("vli.all m1, 0", "vli.row m3, 0", "vmatpush.acc.bf16.mxu0 acc0, m3",
                       "vmatpop.bf16.acc.mxu1 m3, acc0", "vpack.bf16.fp8 m1, e0, m3",
                       "vunpack.fp8.bf16 m3, e0, m1"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "even"):
                check_matrix_registers(source)

    def test_binary_bank_aliases(self):
        for operands in ("m2, m0, m32", "m2, m34, m2", "m2, m30, m62"):
            with self.subTest(operands=operands), self.assertRaisesRegex(ValueError, "physical bank"):
                check_matrix_registers(f"vmul.bf16 {operands}")
        check_matrix_registers("vmul.bf16 m2, m0, m0\nvadd.bf16 m2, m0, m34\n")

    def test_single_register_formats_accept_odd_registers(self):
        check_matrix_registers("\n".join(("vload m1, 0(x1)", "vstore m3, 0(x9)",
            "vtrpose.xlu m1, m1", "vli.col m3, 0", "vli.one m1, 0",
            "vmatpush.weight.mxu0 w0, m1", "vmatpush.acc.fp8.mxu1 acc0, m3",
            "vmatpop.fp8.acc.mxu1 m3, acc0, e0", "vmatmul.mxu0 acc0, m1, w0",
            "vpack.bf16.fp8 m1, e0, m2", "vunpack.fp8.bf16 m2, e0, m1")))

    def test_dma_setup_carveout_does_not_hide_later_scratch_uses(self):
        prefix = "li x1, 0\nli x30, 1\ndma.config x30, ch0\ndma.config x30, ch1\n"
        check_scalar_scratch(prefix + "loop:\nadd x12, x12, x8\n", (30, 31))
        for suffix in ("loop:\nli x30, 1\ndma.config x30, ch0\n", "loop:\nadd x12, x31, x8\n"):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "scalar scratch"):
                check_scalar_scratch(prefix + suffix, (30, 31))
        with self.assertRaisesRegex(ValueError, "scalar scratch"):
            check_scalar_scratch("li x30, 1\nadd x12, x12, x8\n", (30, 31))

    def test_direct_recipes_and_dispatcher_reject_odd_hook(self):
        specs = [(atlas_program, (32, 32, 32))]
        seen = set()
        for shape, name in atlas_program._TRAFFIC_REUSE_MODULES.items():
            if name not in seen:
                seen.add(name)
                specs.append((import_module(name), shape))
        for module, (m, k, n) in specs:
            offset = m * k + k * n + 256
            hook = OddSource(m, n, offset)
            with self.subTest(module=module.__name__), self.assertRaisesRegex(ValueError, "even"):
                module.assembly(m, k, n, offset, 0, epilogue=hook)

    def test_recipe_scalar_scratch_collision(self):
        m, k, n = 32, 32, 32
        offset = m * k + k * n + 256
        hook = Epilogue(m, n, offset, scratch_registers=(14, 15))
        with self.assertRaisesRegex(ValueError, "scalar scratch"):
            atlas_program.assembly(m, k, n, offset, 0, epilogue=hook)
        for name, shape in (("resident", (128, 768, 768)), ("blocked", (64, 32, 736)),
                            ("stream", (128, 3072, 768)), ("smol_connector", (64, 12288, 960))):
            m, k, n = shape
            offset = m * k + k * n + 256
            hook = Epilogue(m, n, offset, scratch_registers=(19, 20))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "scalar scratch"):
                import_module("atlas_program_reuse_" + name).assembly(m, k, n, offset, 0, epilogue=hook)

    def test_direct_input_call_local_temporaries_allow_hook_scratch(self):
        for m, k, n, reuse, free_pair in ((32, 32, 32, False, (22, 23)), (64, 32, 736, True, (10, 11))):
            offset = m * k + k * n + 256
            direct = DirectInput(0, k, 1, m, zero_offset=m * k)
            with self.subTest(shape=(m, k, n)):
                atlas_program.assembly(m, k, n, offset, 0, traffic_reuse=reuse,
                    direct_input=direct, epilogue=Epilogue(m, n, offset))
            atlas_program.assembly(m, k, n, offset, 0, traffic_reuse=reuse,
                direct_input=direct, epilogue=Epilogue(m, n, offset, scratch_registers=free_pair))

    def test_tiny_silu_and_strided_hooks(self):
        for name, shape in (("tiny", (128, 2048, 2048)), ("tiny_breadth", (128, 5632, 1024))):
            m, k, n = shape
            offset = m * k + k * n + 256
            for steps, stride, column in (((Step("silu"),), None, 0), ((), n * 64, 1)):
                hook = Epilogue(m, n, offset, steps, output_dtype="fp8",
                                output_row_stride=stride, output_column_tile=column)
                with self.subTest(name=name, steps=steps, stride=stride):
                    import_module("atlas_program_reuse_" + name).assembly(m, k, n, offset, 0, epilogue=hook)
                    atlas_program.assembly(m, k, n, offset, 0, traffic_reuse=True, epilogue=hook)

    def test_all_legacy_shapes_and_c_hooks(self):
        shapes = [(32, 32, 32), (64, 32, 64), (64, 64, 64), *atlas_program._TRAFFIC_REUSE_MODULES]
        for m, k, n in shapes:
            offset = m * k + k * n + 256
            for reuse in ([False, True] if (m, k, n) in atlas_program._TRAFFIC_REUSE_MODULES else [False]):
                for exponent in (0, -3):
                    with self.subTest(shape=(m, k, n), reuse=reuse, exponent=exponent):
                        atlas_program.assembly(m, k, n, offset, exponent, traffic_reuse=reuse)
            for dtype in ("bf16", "fp8"):
                with self.subTest(shape=(m, k, n), dtype=dtype):
                    atlas_program.assembly(m, k, n, offset, 0,
                        traffic_reuse=(m, k, n) in atlas_program._TRAFFIC_REUSE_MODULES,
                        epilogue=Epilogue(m, n, offset, output_dtype=dtype))


if __name__ == "__main__":
    unittest.main(verbosity=2)
