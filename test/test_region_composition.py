"""Shared stage contracts and constant rematerialization."""
import dataclasses
import hashlib
from pathlib import Path
import tempfile
import unittest


def source_graph(m=2, k=65, widths=(3, 5)):
    from atlas_mlir_native.binding import TARGET_IDENTITY
    from merlin.compiler.semantic.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType

    def type_(shape):
        return TensorType(shape, "f32", "raw-bit-pattern")

    maps = tuple(IndexMap(3, rows, (0, 0)) for rows in (
        ((1, 0, 0), (0, 0, 1)), ((0, 0, 1), (0, 1, 0)), ((1, 0, 0), (0, 1, 0))))
    nodes = [SemanticNode("a", "input", (), type_((m, k)), effect="input")]
    constants, outputs = [], []
    for i, n in enumerate(widths):
        w, z, y = f"w{i}", f"z{i}", f"y{i}"
        nodes.extend((SemanticNode(w, "input", (), type_((k, n)), effect="input"),
                      SemanticNode(z, "constant", (), type_((m, n)), effect="constant"),
                      SemanticNode(y, "source_f32_matmul_unqualified", ("a", w, z), type_((m, n)),
                                   (("numeric_semantics", "source-f32-mul-add-unqualified-v1"), ("reduction_order", "unresolved")),
                                   index_maps=(*maps, maps[-1]))))
        constants.append(ConstantBinding(z, "f32-le-raw", "00000000" * (m * n)))
        outputs.append(y)
    return KernelRequest(tuple(nodes), tuple(outputs), ("external",) * len(outputs),
                         tuple((node.id, "external") for node in nodes if node.effect == "input"),
                         TARGET_IDENTITY, lowering_policy="diagnostic", constants=tuple(constants))



class RegionCompositionContractTest(unittest.TestCase):
    def test_rematerialization_retains_typed_values_at_each_consumer(self):
        from atlas_mlir_native.region_composition import rematerialize_scalar_constants
        from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

        source = """module {
          func.func @controls() -> i32 {
            %base = arith.constant -1879048192 : i32
            %step = arith.constant 2048 : i32
            %first = arith.addi %base, %step : i32
            %second = arith.addi %first, %step : i32
            return %second : i32
          }
        }"""
        result = parse_mlir_text(rematerialize_scalar_constants(source))
        constants = [op for op in result.walk() if op.name == "arith.constant"]
        self.assertEqual(sorted(op.properties["value"].value.data for op in constants), [-1879048192, 2048, 2048])
        for constant in constants:
            consumers = {use.operation for use in constant.results[0].uses}
            self.assertEqual(len(consumers), 1)
            consumer = next(iter(consumers))
            following = constant.next_op
            while following is not consumer:
                self.assertEqual(following.name, "arith.constant")
                following = following.next_op


    def test_dependency_and_incomplete_stage_contracts_are_rejected(self):
        from atlas_mlir_native.binding import DRAM_BASE
        from atlas_mlir_native.region_composition import DeviceBuffer, VirtualStage, _contract

        buffers = tuple(DeviceBuffer(name, DRAM_BASE + index * 2048, 2048, "bf16_tiles")
                        for index, name in enumerate(("input", "partial", "result")))
        stages = (VirtualStage("first", "", ("input",), ("partial",)),
                  VirtualStage("second", "", ("partial",), ("result",)))
        initialized, outputs = ("input",), ("partial", "result")
        _contract(stages, buffers, initialized, outputs, 1 << 20)
        for stage_rows, initial, storage, window, message in (
            (stages, (), buffers, 1 << 20, "before its producer"),
            (stages, (*initialized, "partial"), buffers, 1 << 20, "immutable"),
            (stages, initialized, buffers, 2048, "selected I/O"),
            (stages, initialized,
             (*buffers[:-1], dataclasses.replace(buffers[-1], address=buffers[-1].address + 2048)),
             1 << 20, "contiguous"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _contract(stage_rows, storage, initial, outputs, window)

    def test_contract_follows_the_selected_io_base(self):
        from atlas_mlir_native.soc import SOC
        from atlas_mlir_native.region_composition import DeviceBuffer, VirtualStage, _contract

        names, sizes = ("input", "result"), {"input": 4096, "result": 2048}
        base = SOC.memory_region("atlas_data").base
        addresses = {"input": base, "result": base + 4096}
        buffers = tuple(DeviceBuffer(name, addresses[name], sizes[name], "bf16_tiles") for name in names)
        stages = (VirtualStage("only", "", ("input",), ("result",)),)
        _contract(stages, buffers, ("input",), ("result",), 768 * 1024, base)
        with self.assertRaisesRegex(ValueError, "selected I/O"):
            _contract(stages, buffers, ("input",), ("result",), 1 << 20)


if __name__ == "__main__":
    unittest.main()
