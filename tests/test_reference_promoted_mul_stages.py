"""Source-proven BF16 widening and host F32 multiplication across selected routes."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir_native.reference_eval import decisions, stages
from atlas_mlir_native.reference_eval.evaluator import Tensor


def _source(promoted=0):
    dtype = ['bf16', 'f32'] if promoted == 0 else ['f32', 'bf16']
    types = ['tensor<3xbf16>', 'tensor<2x3xf32>'] if promoted == 0 else ['tensor<2x3xf32>', 'tensor<3xbf16>']
    maps = ['d1', 'd0, d1'] if promoted == 0 else ['d0, d1', 'd1']
    factors = '%wide, %b' if promoted == 0 else '%a, %wide'
    provenance = '{prov.origin_node_ids = ["g:quantized:root:mul"], prov.aten = "aten.mul.Tensor"}'
    return f'''"builtin.module"() ({{
      "func.func"() <{{sym_name = "forward", function_type = ({types[0]}, {types[1]}) -> tensor<2x3xf32>}}> ({{
      ^bb0(%left: {types[0]}, %right: {types[1]}):
        %empty = "tensor.empty"() {provenance} : () -> tensor<2x3xf32>
        %product = "linalg.generic"(%left, %right, %empty) <{{
          indexing_maps = [affine_map<(d0, d1) -> ({maps[0]})>, affine_map<(d0, d1) -> ({maps[1]})>,
            affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
          operandSegmentSizes = array<i32: 2, 1>}}> ({{
        ^bb1(%a: {dtype[0]}, %b: {dtype[1]}, %out: f32):
          %wide = "arith.extf"({'%a' if promoted == 0 else '%b'}) : (bf16) -> f32
          %result = "arith.mulf"({factors}) <{{fastmath = #arith.fastmath<none>}}> : (f32, f32) -> f32
          "linalg.yield"(%result) : (f32) -> ()
        }}) {provenance} : ({types[0]}, {types[1]}, tensor<2x3xf32>) -> tensor<2x3xf32>
        "func.return"(%product) : (tensor<2x3xf32>) -> ()
      }}) : () -> ()
    }}) : () -> ()'''


def _binding(promoted=0):
    return decisions._source_provenance(_source(promoted), bindings=True)['g:quantized:root:mul']


def _rows(promoted=0):
    return [
        {'id': 'any-cast', 'implementation': 'source_exact_float_device', 'placement': 'accelerator',
         'kind': 'exact_float_device', 'inputs': [f'arg{promoted}'], 'buffers': ['temporary']},
        {'id': 'any-product', 'implementation': 'source_float_arithmetic', 'placement': 'host',
         'kind': 'float_arithmetic', 'inputs': [f'arg{1-promoted}', 'temporary'], 'buffers': ['op1r0_s1']},
    ]


class PromotedMultiplyTests(unittest.TestCase):
    def test_source_cast_then_host_product_preserves_f32_bits_and_broadcasting(self):
        low = Tensor(np.array([0x3f81, 0xbf80, 0x0080], np.uint16), 'bf16')
        high = Tensor(np.array([[1.000000119, 0.0, 0.5], [3.1, -0.0, -0.5]], np.float32), 'f32')
        wide = (low.raw.astype(np.uint32) << 16).view(np.float32)
        for promoted in (0, 1):
            with self.subTest(promoted=promoted):
                binding, rows = _binding(promoted), _rows(promoted)
                graph = stages.build(rows, binding, 'aten.mul.Tensor')
                args = [low, high] if promoted == 0 else [high, low]
                with patch('atlas_mlir_native.reference_eval.placed_device._rtl') as admission:
                    actual = stages.call('aten.mul.Tensor', args, {}, {'numerical_stages': graph})
                expected = np.multiply(wide, high.raw.view(np.float32)) if promoted == 0 else np.multiply(
                    high.raw.view(np.float32), wide)
                np.testing.assert_array_equal(actual.raw, expected.view(np.uint32))
                self.assertEqual(admission.call_args.args[0]['selected_routes'], ['source_exact_float_device'])
                self.assertEqual([s['adapter'] for s in graph['stages']], ['widen', 'multiply'])

    def test_views_of_the_bf16_operand_may_carry_it_to_the_widening(self):
        low = Tensor(np.array([0x3f81, 0xbf80, 0x0080], np.uint16), 'bf16')
        high = Tensor(np.array([[1.000000119, 0.0, 0.5], [3.1, -0.0, -0.5]], np.float32), 'f32')
        views = [{'id': 'any-view', 'implementation': 'source_layout', 'placement': 'host', 'kind': 'layout',
                  'inputs': ['arg0'], 'buffers': ['broadcast']},
                 {'id': 'any-alias', 'implementation': 'merlin.dense_buffer_view', 'placement': 'host',
                  'inputs': ['broadcast'], 'buffers': ['aliased']}]
        rows = _rows()
        rows[0]['inputs'] = ['aliased']
        graph = stages.build([*views, *rows], _binding(), 'aten.mul.Tensor')
        self.assertEqual([s['adapter'] for s in graph['stages']], ['widen', 'multiply', 'identity', 'identity'])
        with patch('atlas_mlir_native.reference_eval.placed_device._rtl'):
            actual = stages.call('aten.mul.Tensor', [low, high], {}, {'numerical_stages': graph})
        wide = (low.raw.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(actual.raw, np.multiply(wide, high.raw.view(np.float32)).view(np.uint32))
        for mutation in ('other_operand', 'unused', 'skips_view'):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy([*views, *rows])
                if mutation == 'other_operand': changed[0]['inputs'] = ['arg1']
                elif mutation == 'unused': changed[2]['inputs'] = ['broadcast']
                elif mutation == 'skips_view': changed[2]['inputs'] = ['arg0']
                with self.assertRaisesRegex(ValueError, '^invalid numerical stage graph:'):
                    stages.build(changed, _binding(), 'aten.mul.Tensor')

    def test_policy_accepts_the_proven_mixed_routes_and_records_edges(self):
        policy = object.__new__(decisions.SelectedDecisions)
        policy._bindings, policy._by_node = {'mul': _binding()}, {'mul': _rows()}
        policy.record = {'applied': {}}
        rules = policy.node_rules({'id': 'mul', 'target': 'aten.mul.Tensor'}, {})
        self.assertEqual(rules['selected_routes'], ['source_exact_float_device', 'source_float_arithmetic'])
        self.assertEqual(rules['numerical_stages']['stages'][1]['inputs'], ['arg1', 'temporary'])
        self.assertEqual(policy.record['applied']['mul']['numerical_stages'], rules['numerical_stages'])

    def test_invalid_source_body_maps_and_routes_fail_closed(self):
        for mutation in ('body', 'yield', 'fastmath', 'map', 'broadcast_shape', 'iterator', 'segments', 'orphan',
                         'route', 'placement', 'kind', 'edge', 'result', 'order', 'duplicate_id', 'overwrite'):
            with self.subTest(mutation=mutation):
                binding, rows = _binding(), _rows()
                source = binding['ops'][1]
                if mutation == 'body': source['body'][1][0] = 'arith.addf'
                elif mutation == 'yield': source['body'][2][1] = 'body0r0'
                elif mutation == 'fastmath': source['strict_float'] = False
                elif mutation == 'map': source['indexing_maps'][0] = 'affine_map<(d0, d1) -> (d0)>'
                elif mutation == 'broadcast_shape':
                    source['operand_types'][1] = 'tensor<3xf32>'
                    source['indexing_maps'][1] = 'affine_map<(d0, d1) -> (d1)>'
                elif mutation == 'iterator': source['parallel_iterators'] = False
                elif mutation == 'segments': source['operand_segments'] = [1, 2]
                elif mutation == 'orphan': binding['ops'].append(copy.deepcopy(binding['ops'][0]))
                elif mutation == 'route': rows[0]['implementation'] = 'source_pointwise'
                elif mutation == 'placement': rows[1]['placement'] = 'accelerator'
                elif mutation == 'kind': rows[0]['kind'] = 'layout'
                elif mutation == 'edge': rows[1]['inputs'] = ['arg0', 'temporary']
                elif mutation == 'result': rows[1]['buffers'] = ['unrelated-result']
                elif mutation == 'order': rows.reverse()
                elif mutation == 'duplicate_id': rows[1]['id'] = rows[0]['id']
                elif mutation == 'overwrite': rows[0]['buffers'] = ['arg0']
                with self.assertRaisesRegex(ValueError, '^invalid numerical stage graph:'):
                    stages.build(rows, binding, 'aten.mul.Tensor')

    def test_wrong_runtime_dtype_shape_or_signature_is_rejected(self):
        graph = stages.build(_rows(), _binding(), 'aten.mul.Tensor')
        low, high = Tensor(np.ones(3, np.uint16), 'bf16'), Tensor(np.ones((2, 3), np.float32), 'f32')
        for args, kwargs in (([high, low], {}), ([Tensor(np.ones(6, np.uint16), 'bf16'), high], {}),
                             ([low, high], {'alpha': 2}), ([low], {})):
            with self.subTest(args=len(args), kwargs=kwargs), self.assertRaises(ValueError):
                stages.call('aten.mul.Tensor', args, kwargs, {'numerical_stages': graph})

    def test_runtime_revalidates_source_graph(self):
        graph = stages.build(_rows(), _binding(), 'aten.mul.Tensor')
        graph['binding']['ops'][1]['body'][1][0] = 'arith.addf'
        with self.assertRaisesRegex(ValueError, '^invalid numerical stage graph:'):
            stages.call('aten.mul.Tensor', [], {}, {'numerical_stages': graph})


if __name__ == '__main__':
    unittest.main()
