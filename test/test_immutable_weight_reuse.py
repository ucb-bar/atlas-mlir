"""Automatic weight reuse preserves state, aliasing and bounded allocation."""
import unittest

from test_virtual_ssa import run
from test_virtual_scheduling import (
    F, S, T, const, dma_load, dma_store, function, inp, mxu_chain, outp, start,
)

PASS = "--reuse-atlas-immutable-weights"
ABI = "atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2416967680 : i64"


def matmul(dst, activation, weight):
    return (f'%{dst} = "atlas.virtual_mxu_matmul"(%{activation}, %{weight}) '
            f'{{unit = 0 : i32}} : ({F}, {F}) -> {T}')


def repeated(*, typed=False, explicit=False, alias=False, partial=False,
             activation_use=False, separate_constants=False):
    lines = [start(), inp("io0", "a", "x", 1, "fp8")]
    state = "a"
    if explicit:
        lines += [const("address", -1879048192), const("size", 1024)]
    for i in range(2):
        after = f"load{i}"
        if explicit:
            address, size = "address", "size"
            if separate_constants:
                address, size = f"address{i}", f"size{i}"
                lines += [const(address, -1879048192), const(size, 1024)]
            lines.append(dma_load(state, after, f"w{i}", address, size, "fp8"))
        else:
            lines.append(inp(state, after, f"w{i}", 0, "fp8"))
        state = after
        if typed:
            lines += mxu_chain(state, 0, f"w{i}", "x", f"y{i}", f"m{i}")
            state = f"m{i}s3"
        else:
            lines.append(matmul(f"y{i}", f"w{i}" if activation_use else "x", f"w{i}"))
        if alias and i == 0:
            # A write anywhere to the weight range prevents the immutable
            # classification, including a partially overlapping BF16 store.
            lines += [const("write", -1879048224 if partial else -1879048192),
                      const("write_size", 2048),
                      dma_store(state, "write_state", "y0", "write", "write_size")]
            state = "write_state"
        lines.append(outp(state, f"output{i}", f"y{i}", i))
        state = f"output{i}"
    return function("repeat", [*lines, f"return %{state} : {S}"], attrs=ABI)


class ImmutableWeightReuseTest(unittest.TestCase):
    def checked(self, source, *options):
        result = run("atlas-opt", source, *options)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_boundary_legacy_and_typed_weights_and_state_chain(self):
        for typed in (False, True):
            source = repeated(typed=typed)
            reused = self.checked(source, PASS, "--verify-atlas-virtual-stream")
            self.assertEqual(reused.count('"atlas.virtual_input_fp8"'), 2)
            self.assertEqual(reused.count('"atlas.virtual_mxu_load_weight"'), 2 if typed else 0)
            self.assertEqual(self.checked(reused, PASS), reused)
            for pipeline in (
                ("--lower-atlas-virtual-to-machine",),
                ("--lower-atlas-virtual-boundary-to-dma", "--schedule-atlas-virtual",
                 "--lower-atlas-virtual-to-machine"),
            ):
                flags = (*pipeline, "--insert-atlas-dma-waits", "--schedule-atlas-stream",
                         "--verify-atlas-timing", "--verify-atlas-generated-schedule")
                # Every ordinary pipeline invokes the same transformation.
                self.assertEqual(self.checked(source, *flags), self.checked(reused, *flags))

    def test_explicit_dma_removes_only_newly_dead_constants(self):
        source = repeated(typed=True, explicit=True, separate_constants=True)
        reused = self.checked(source, PASS, "--verify-atlas-virtual-stream")
        self.assertEqual(reused.count('"atlas.virtual_dma_load_fp8"'), 1)
        # Shared unused constants that predate reuse are retained. Only the
        # removed load's two newly dead constants disappear.
        self.assertEqual(reused.count("arith.constant"), 4)
        self.checked(source, "--schedule-atlas-virtual", "--lower-atlas-virtual-to-machine",
                     "--insert-atlas-dma-waits", "--schedule-atlas-stream",
                     "--verify-atlas-timing", "--verify-atlas-generated-schedule")

    def test_full_and_partial_writes_prevent_boundary_and_dma_reuse(self):
        for explicit in (False, True):
            for partial in (False, True):
                source = repeated(explicit=explicit, alias=True, partial=partial)
                self.assertEqual(self.checked(source, PASS), self.checked(source))

    def test_activation_or_mixed_uses_are_not_weight_candidates(self):
        source = repeated(activation_use=True)
        self.assertEqual(self.checked(source, PASS), self.checked(source))

    def test_distinct_weights_and_cfg_are_exact_noops(self):
        source = repeated().replace("index = 0 : i32", "index = 2 : i32", 1)
        self.assertEqual(self.checked(source, PASS), self.checked(source))
        # Both reads would qualify in a single block, but the function has
        # a CFG edge. Outputs belong to the return block by virtual contract.
        cfg = function("cfg_weights", [
            start(), inp("io0", "s1", "x", 1, "fp8"),
            inp("s1", "s2", "w0", 0, "fp8"), matmul("y0", "x", "w0"),
            inp("s2", "s3", "w1", 0, "fp8"), matmul("y1", "x", "w1"),
            f"cf.br ^done(%s3, %y0, %y1 : {S}, {T}, {T})",
            f"^done(%state: {S}, %a: {T}, %b: {T}):",
            outp("state", "o0", "a", 0), outp("o0", "o1", "b", 1),
            f"return %o1 : {S}",
        ], attrs=ABI)
        self.assertEqual(self.checked(cfg, PASS), self.checked(cfg))

    def test_original_boundary_overlap_is_not_hidden(self):
        source = repeated().replace("atlas.output_dram_base = 2416967680",
                                    "atlas.output_dram_base = 2415919104")
        self.assertEqual(self.checked(source, PASS), self.checked(source))
        for pipeline in ("--lower-atlas-virtual-boundary-to-dma", "--lower-atlas-virtual-to-machine"):
            result = run("atlas-opt", source, pipeline)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("overlap", result.stderr)

    def test_boundary_and_explicit_inputs_do_not_hide_slot_checks(self):
        # Boundary b and explicit w0/w1 address the same bytes and all have
        # exclusively weight uses. Only w1 can be replaced by w0.
        lines = [start(), inp("io0", "a", "x", 1, "fp8"),
                 inp("a", "bstate", "b", 0, "fp8"), matmul("by", "x", "b"),
                 outp("bstate", "bo", "by", 0),
                 const("address", -1879048192), const("size", 1024)]
        state = "bo"
        for i in range(2):
            lines += [dma_load(state, f"d{i}", f"w{i}", "address", "size", "fp8"),
                      matmul(f"y{i}", "x", f"w{i}"), outp(f"d{i}", f"o{i}", f"y{i}", i+1)]
            state = f"o{i}"
        source = function("mixed", [*lines, f"return %{state} : {S}"], attrs=ABI)
        reused = self.checked(source, PASS)
        self.assertEqual(reused.count('"atlas.virtual_input_fp8"'), 2)
        self.assertEqual(reused.count('"atlas.virtual_dma_load_fp8"'), 1)

    def test_writes_before_or_after_loads_prevent_immutable_classification(self):
        source = repeated()
        constants = [const("write", -1879048192), const("write_size", 2048)]
        before = source.replace(
            inp("a", "load0", "w0", 0, "fp8"),
            "\n".join([*constants,
                       f'%seed = "atlas.virtual_vpu_splat"() {{bits = 0 : i32}} : () -> {T}',
                       dma_store("a", "written", "seed", "write", "write_size"),
                       inp("written", "load0", "w0", 0, "fp8")]))
        after = source.replace(
            f"return %output1 : {S}",
            "\n".join([*constants, dma_store("output1", "written", "y1", "write", "write_size"),
                       f"return %written : {S}"]))
        for changed in (before, after):
            self.assertEqual(self.checked(changed, PASS), self.checked(changed))

    def test_allocation_failure_falls_back_without_changing_ir(self):
        lines = [start(), inp("io0", "s1", "x", 1, "fp8"),
                 inp("s1", "s2", "first", 0, "fp8"), matmul("early", "x", "first"),
                 outp("s2", "early_out", "early", 0)]
        state = "early_out"
        # Original: x plus 31 live weights fit exactly. Caching `first`
        # across this region would need 33 FP8 registers; reuse must fit without spilling.
        for i in range(31):
            lines.append(inp(state, f"s{i+3}", f"w{i}", i+2, "fp8"))
            state = f"s{i+3}"
        for i in range(31):
            lines += [matmul(f"y{i}", "x", f"w{i}"), outp(state, f"o{i}", f"y{i}", i+1)]
            state = f"o{i}"
        lines += [inp(state, "last", "again", 0, "fp8"), matmul("final", "x", "again"),
                  outp("last", "done", "final", 32), f"return %done : {S}"]
        source = function("pressure", lines, attrs=ABI)
        self.assertEqual(self.checked(source, PASS), self.checked(source))
        self.checked(source, "--lower-atlas-virtual-to-machine", "--insert-atlas-dma-waits",
                     "--insert-atlas-delays", "--verify-atlas-generated-schedule")


if __name__ == "__main__":
    unittest.main()
