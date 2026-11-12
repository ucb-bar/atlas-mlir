"""Shared stage contracts and source-bound adjacent publication."""
import dataclasses
import hashlib
from pathlib import Path
import tempfile
import unittest


def source_graph(m=2, k=65, widths=(3, 5)):
    from atlas_mlir_native.binding import TARGET_IDENTITY
    from merlin.semantic_compiler.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType

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
    def test_adjacent_publication_hashes_nested_files_and_member_manifests(self):
        from atlas_mlir_native.region_composition import publish_adjacent_group

        region = source_graph()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'members/0').mkdir(parents=True)
            (root / 'members/0/manifest.json').write_text('{}')
            (root / 'members/0/body.bin').write_bytes(b'member')
            (root / 'program.bin').write_bytes(b'image')
            # A composition manifest may already exist at this location.
            (root / 'manifest.json').write_text('{}')
            result = publish_adjacent_group(dict(region=region), root, kind='adjacent_pointwise',
                                            execution={}, enlargement={})
            self.assertEqual(set(result['stage_artifacts']), {
                'members/0/manifest.json', 'members/0/body.bin', 'program.bin',
                'execution_plan.json', 'source-request.json', 'enlargement.json'})
            for name, digest in result['stage_artifacts'].items():
                self.assertEqual(digest, hashlib.sha256((root / name).read_bytes()).hexdigest())


    def test_rematerialization_retains_typed_values_at_each_consumer(self):
        from atlas_mlir_native.region_composition import rematerialize_scalar_constants
        from merlin.frontends.linalg_mlir import parse_mlir_text

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


if __name__ == "__main__":
    unittest.main()
