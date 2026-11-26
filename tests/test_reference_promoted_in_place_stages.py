"""Source-proven promoted add, and in-place promoted arithmetic rounded once into a BF16 destination."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir_native.reference_eval import decisions, stages
from atlas_mlir_native.reference_eval.evaluator import Tensor

NODE = 'g:quantized:root:addi'


def _source(arithmetic='arith.addf', aten='aten.add.Tensor', in_place=True):
    origin = f'prov.origin_node_ids = ["{NODE}"]'
    first = '{' + origin + f', prov.aten = "{aten}"' + '}'
    copy_ = '{' + origin + ', prov.aten = "aten._to_copy.default"}'
    narrow = f'''
        %dest = "tensor.empty"() {copy_} : () -> tensor<2x3xbf16>
        %rounded = "linalg.generic"(%wide_result, %dest) <{{
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
          operandSegmentSizes = array<i32: 1, 1>}}> ({{
        ^bb2(%v: f32, %o: bf16):
          %t = "arith.truncf"(%v) : (f32) -> bf16
          "linalg.yield"(%t) : (bf16) -> ()
        }}) {copy_} : (tensor<2x3xf32>, tensor<2x3xbf16>) -> tensor<2x3xbf16>''' if in_place else ''
    result, rtype = ('%rounded', 'tensor<2x3xbf16>') if in_place else ('%wide_result', 'tensor<2x3xf32>')
    return f'''"builtin.module"() ({{
      "func.func"() <{{sym_name = "forward", function_type = (tensor<2x3xbf16>, tensor<3xf32>) -> {rtype}}}> ({{
      ^bb0(%self: tensor<2x3xbf16>, %other: tensor<3xf32>):
        %empty = "tensor.empty"() {first} : () -> tensor<2x3xf32>
        %wide_result = "linalg.generic"(%self, %other, %empty) <{{
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>,
            affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
          operandSegmentSizes = array<i32: 2, 1>}}> ({{
        ^bb1(%a: bf16, %b: f32, %out: f32):
          %wide = "arith.extf"(%a) : (bf16) -> f32
          %r = "{arithmetic}"(%wide, %b) <{{fastmath = #arith.fastmath<none>}}> : (f32, f32) -> f32
          "linalg.yield"(%r) : (f32) -> ()
        }}) {first} : (tensor<2x3xbf16>, tensor<3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>{narrow}
        "func.return"({result}) : ({rtype}) -> ()
      }}) : () -> ()
    }}) : () -> ()'''


def _binding(**kw):
    return decisions._source_provenance(_source(**kw), bindings=True)[NODE]


def _rows(in_place=True):
    rows = [
        {'id': 'any-widen', 'implementation': 'source_exact_float_device', 'placement': 'accelerator',
         'kind': 'exact_float_device', 'inputs': ['arg0'], 'buffers': ['wide']},
        {'id': 'any-add', 'implementation': 'source_float_arithmetic', 'placement': 'host',
         'kind': 'float_arithmetic', 'inputs': ['arg1', 'wide'], 'buffers': ['op1r0_s1']},
    ]
    if in_place:
        rows.append({'id': 'any-narrow', 'implementation': 'source_exact_float_device', 'placement': 'accelerator',
                     'kind': 'exact_float_device', 'inputs': ['op1r0_s1'], 'buffers': ['op3r0_s0']})
    return rows


def _rne(f32_bits):
    raw = f32_bits.astype(np.uint32)
    return ((raw + np.uint32(0x7fff) + ((raw >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


LOW = Tensor(np.array([[0x3f80, 0x3f81, 0xc000], [0x0000, 0x8000, 0x4049]], np.uint16), 'bf16')
HIGH = Tensor(np.array([2.0 ** -9 + 2.0 ** -17, 1.5e-3, -0.0], np.float32).view(np.uint32), 'f32')


class PromotedInPlaceTests(unittest.TestCase):
    def _expected_f32(self, op):
        wide = (LOW.raw.astype(np.uint32) << 16).view(np.float32)
        return op(wide, HIGH.raw.view(np.float32), dtype=np.float32).view(np.uint32)

    def test_in_place_add_widens_adds_in_f32_and_rounds_once_into_bf16(self):
        for arithmetic, aten, op, adapter in (('arith.addf', 'aten.add.Tensor', np.add, 'add'),
                                              ('arith.mulf', 'aten.mul.Tensor', np.multiply, 'multiply')):
            with self.subTest(aten=aten):
                target = aten.replace('.Tensor', '_.Tensor')
                graph = stages.build(_rows(), _binding(arithmetic=arithmetic, aten=aten), target)
                self.assertEqual([s['adapter'] for s in graph['stages']], ['widen', adapter, 'narrow'])
                with patch('atlas_mlir_native.reference_eval.placed_device._rtl') as admission:
                    # The evaluator calls an in-place node through its out-of-place form.
                    actual = stages.call(aten, [LOW, HIGH], {}, {'numerical_stages': graph})
                self.assertEqual(actual.dtype, 'bf16')
                np.testing.assert_array_equal(actual.raw, _rne(self._expected_f32(op)))
                self.assertEqual([c.args[0]['selected_routes'] for c in admission.call_args_list],
                                 [['source_exact_float_device'], ['source_exact_float_device']])

    def test_out_of_place_promoted_add_keeps_f32_bits(self):
        graph = stages.build(_rows(False), _binding(in_place=False), 'aten.add.Tensor')
        with patch('atlas_mlir_native.reference_eval.placed_device._rtl'):
            actual = stages.call('aten.add.Tensor', [LOW, HIGH], {}, {'numerical_stages': graph})
        self.assertEqual(actual.dtype, 'f32')
        np.testing.assert_array_equal(actual.raw, self._expected_f32(np.add))

    def test_policy_accepts_the_in_place_routes_and_records_edges(self):
        policy = object.__new__(decisions.SelectedDecisions)
        policy._bindings, policy._by_node = {'addi': _binding()}, {'addi': _rows()}
        policy.record = {'applied': {}}
        rules = policy.node_rules({'id': 'addi', 'target': 'aten.add_.Tensor'}, {})
        self.assertEqual(rules['selected_routes'], ['source_exact_float_device', 'source_float_arithmetic'])
        self.assertEqual(rules['numerical_stages']['stages'][2]['inputs'], ['op1r0_s1'])
        self.assertEqual(policy.record['applied']['addi']['numerical_stages'], rules['numerical_stages'])

    def test_invalid_in_place_graphs_fail_closed(self):
        for mutation in ('truncate_body', 'narrow_source', 'narrow_dtype', 'copy_target', 'missing_narrow',
                         'extra_op', 'body_op', 'destination_broadcast', 'other_promoted', 'route', 'placement',
                         'edge', 'narrow_edge', 'result', 'duplicate_id', 'two_rows', 'wrong_target'):
            with self.subTest(mutation=mutation):
                binding, rows, target = _binding(), _rows(), 'aten.add_.Tensor'
                widen, narrow = binding['ops'][1], binding['ops'][3]
                if mutation == 'truncate_body': narrow['body'][0][0] = 'arith.extf'
                elif mutation == 'narrow_source': narrow['operands'][0] = 'arg1'
                elif mutation == 'narrow_dtype': narrow['result_types'] = ['tensor<2x3xf32>']
                elif mutation == 'copy_target': narrow['target'] = 'aten.add.Tensor'
                elif mutation == 'missing_narrow': del binding['ops'][2:]
                elif mutation == 'extra_op': binding['ops'].append(copy.deepcopy(binding['ops'][2]))
                elif mutation == 'body_op': widen['body'][1][0] = 'arith.subf'
                elif mutation == 'destination_broadcast':
                    widen['operand_types'][0] = 'tensor<3xbf16>'
                    widen['indexing_maps'][0] = 'affine_map<(d0, d1) -> (d1)>'
                elif mutation == 'other_promoted':
                    widen['operand_types'][:2] = ['tensor<2x3xf32>', 'tensor<3xbf16>']
                    widen['body'] = [['arith.extf', 1], ['arith.addf', 0, 'body0r0'], ['linalg.yield', 'body1r0']]
                    widen['body_types'][:2] = ['f32', 'bf16']
                elif mutation == 'route': rows[2]['implementation'] = 'source_float_arithmetic'
                elif mutation == 'placement': rows[2]['placement'] = 'host'
                elif mutation == 'edge': rows[1]['inputs'] = ['arg0', 'wide']
                elif mutation == 'narrow_edge': rows[2]['inputs'] = ['wide']
                elif mutation == 'result': rows[2]['buffers'] = ['unrelated']
                elif mutation == 'duplicate_id': rows[2]['id'] = rows[0]['id']
                elif mutation == 'two_rows': rows.pop()
                elif mutation == 'wrong_target': target = 'aten.sub_.Tensor'
                with self.assertRaisesRegex(ValueError, '^invalid numerical stage graph:'):
                    stages.build(rows, binding, target)

    def test_runtime_rejects_wrong_dtype_and_unrelated_evaluated_target(self):
        graph = stages.build(_rows(), _binding(), 'aten.add_.Tensor')
        for target, args in (('aten.add.Tensor', [HIGH, LOW]), ('aten.mul.Tensor', [LOW, HIGH]),
                             ('aten.add.Tensor', [Tensor(np.ones(3, np.uint16), 'bf16'), HIGH])):
            with self.subTest(target=target), self.assertRaises(ValueError):
                with patch('atlas_mlir_native.reference_eval.placed_device._rtl'):
                    stages.call(target, args, {}, {'numerical_stages': graph})


if __name__ == '__main__':
    unittest.main()
