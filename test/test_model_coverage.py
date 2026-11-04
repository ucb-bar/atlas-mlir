"""The hand-written SmolVLA and TinyLlama device plans equal the device ops torch.export derives from the models.

The fixtures are atlas-kernels coverage/fixture/{smolvla,tinyllama}.json: aggregated op rows that
coverage/export_ops.py writes from the captured checkpoints and inputs. Regenerate them with that tool, never by hand.
"""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import kernel_library  # noqa: E402
from smol_attention_assets import event_plan  # noqa: E402
from smol_full_plan import calls  # noqa: E402
sys.path.insert(0, str(ROOT / 'tools/tiny_full'))
from capture_boundaries import validate_inventory  # noqa: E402

# Use the fixtures in the pinned submodule, including after generator path moves.
FIXTURE = kernel_library.LIBRARY / 'coverage' / 'fixture'


PROJECTION = ('aten.linear.default', 'aten.conv2d.default', 'aten.conv2d.padding')
ATTENTION = ('aten.softmax.int', 'aten.scaled_dot_product_attention.default')


def module_name(row, layer):
    return row['module'].replace('layers.*', f'layers.{layer}') if layer is not None else row['module']


def load(model):
    """Fixture plus its ordered projection calls and attention cores, rebuilt from the rows' call indices."""
    inventory = json.loads((FIXTURE / f'{model}.json').read_text())
    if inventory.get('schema') != 'atlas.coverage.inventory.v2' or inventory.get('model') != model:
        raise ValueError('unexpected coverage fixture')
    calls, attention = {}, []
    for row in inventory['ops']:
        for index, layer, step in row.get('calls', []):
            entry = {'module': module_name(row, layer), 'phase': row['phase'], 'step': step}
            if row['op'] in PROJECTION:
                calls[index] = dict(entry, input=row['inputs'][0], output=row['outputs'][0])
            else:
                kind = 'sdpa' if row['op'] == ATTENTION[1] else 'eager'
                shapes = dict(q=row['inputs'][0], k=row['inputs'][1]) if kind == 'sdpa' else dict(scores=row['inputs'][0])
                attention.append(dict(entry, kind=kind, calls_before=index, **shapes))
    if sorted(calls) != list(range(len(calls))):
        raise ValueError('fixture projection call indices are not contiguous')
    inventory['calls'] = [calls[index] for index in range(len(calls))]
    inventory['attention'] = sorted(attention, key=lambda row: row['calls_before'])
    return inventory


def smol_plan_geometry():
    """smol_full_plan.calls() in the export's terms, with validate_reference's input and output shapes."""
    result = []
    for call in calls():
        rows, k, n = call['rows'], call['input_width'], call['output_width']
        shapes = ([1, k], [1, n]) if call['site'] == 0 else ([1, rows, k], [1, rows, n])
        if call['site'] == 7 and call['role'] == 0:
            shapes = [1, 3, 512, 512], [1, 768, 32, 32]
        dtype = 'bfloat16' if call['input_dtype'] == 0 else 'float32'
        result.append((call['module'], shapes[0], shapes[1], dtype, call.get('step')))
    return result


def attention_geometry(inventory):
    """(query rows, keys, query heads, kv heads, calls before) per attention core in export order."""
    result = []
    for row in inventory['attention']:
        if row['kind'] == 'sdpa':
            q, k = row['q'][0], row['k'][0]
            result.append((q[2], k[2], q[1], k[1], row['calls_before']))
        else:
            _, heads, rows, keys = row['scores'][0]
            value = inventory['calls'][row['calls_before'] - 1]
            result.append((rows, keys, heads, value['output'][0][-1] // 64, row['calls_before']))
    return result


def tiny_capture_view(inventory):
    """The capture fields validate_inventory reads, filled from the export.

    Logical fields (step, site order, module, rows, K, output width, KV cache shapes, logits width) come from the
    export. Physical panels are the plan's column split and M32 decode padding, the part under test elsewhere.
    """
    data, steps = {'calls': []}, {}
    for row in inventory['calls']:
        step, (shape, _), width = row['step'], row['input'], row['output'][0][-1]
        site = steps.get(step, 0)
        steps[step] = site + 1
        rows, k = math.prod(shape[:-1]), shape[-1]
        m, panels, column = (128 if rows == 128 else 32), [], 0
        while column < width:
            n = min(1024 if k == 5632 else 2048, width - column)
            panels.append({'columns': [column, column + n], 'logical_mkn': [rows, k, n], 'physical_mkn': [m, k, n],
                           'expected_bf16': {'shape': [m, n], 'dtype': 'uint16', 'bytes': m * n * 2}})
            column += n
        data['calls'].append({'step': step, 'site_index': site, 'site': row['module'],
                              'A_bf16': {'shape': [rows, k], 'dtype': 'uint16', 'bytes': rows * k * 2},
                              'panels': panels})
    data['target_trace'] = [{'step': step, 'cache': [], 'logits_payload': None} for step in sorted(steps)]
    for row in inventory['ops']:
        if row['module'] == 'model.layers.*.self_attn' and row['source'].endswith(':270') and row['op'] == 'aten.cat.default':
            if row['count'] != 2 * len(row['layers']) * len(row['steps']):
                raise ValueError('KV cache appends are not one key and one value per layer')
            for step in row['steps']:
                data['target_trace'][step]['cache'] += [
                    {'layer': layer, 'field': field, 'shape': row['outputs'][0][0], 'dtype': 'uint16',
                     'bytes': math.prod(row['outputs'][0][0]) * 2} for layer in row['layers'] for field in ('keys', 'values')]
    for row in inventory['calls']:
        if row['module'] == 'lm_head':
            width = row['output'][0][-1]
            data['target_trace'][row['step']]['logits_payload'] = {'shape': [width], 'dtype': 'float32', 'bytes': 4 * width}
    return data


class SmolCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = load('smolvla')

    def test_capture_crosscheck_passed(self):
        check = self.inventory['capture_crosscheck']
        self.assertEqual((check['status'], check['exported_calls'], check['captured_calls']), ('PASS', 1421, 1421))

    def test_plan_calls_equal_export(self):
        export = [(row['module'], row['input'][0], row['output'][0], row['input'][1],
                   row['step'] if row['phase'] == 'denoise' else None) for row in self.inventory['calls']]
        self.assertEqual(smol_plan_geometry(), export)

    def test_plan_attention_equals_export(self):
        planned = [event['before_call'] for event in event_plan() if event['kind'] == 'attention']
        derived = attention_geometry(self.inventory)
        self.assertEqual([row[4] for row in derived], planned)
        expected = [(1024, 1024, 12, 12)] * 24 + [(177, 177, 15, 5)] * 16 + [
            (50, 227 if i % 2 == 0 else 177, 15, 5) for i in range(160)]
        self.assertEqual([row[:4] for row in derived], expected)

    def test_mutated_export_is_detected(self):
        export = [(row['module'], row['input'][0], row['output'][0], row['input'][1],
                   row['step'] if row['phase'] == 'denoise' else None) for row in self.inventory['calls']]
        for index, field, value in ((5, 0, 'renamed'), (300, 1, [1, 177, 961]), (1420, 3, 'bfloat16')):
            changed = copy.deepcopy(export)
            changed[index] = changed[index][:field] + (value,) + changed[index][field + 1:]
            self.assertNotEqual(smol_plan_geometry(), changed)
        self.assertNotEqual(smol_plan_geometry(), export[:-1])


class TinyCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = load('tinyllama')

    def test_capture_crosscheck_passed(self):
        check = self.inventory['capture_crosscheck']
        self.assertEqual((check['status'], check['exported_calls'], check['captured_calls']), ('PASS', 465, 465))

    def test_validate_inventory_admits_export(self):
        validate_inventory(tiny_capture_view(self.inventory))

    def test_attention_geometry(self):
        derived = attention_geometry(self.inventory)
        expected = [(rows, keys, 32, 4) for rows, keys in [(128, 128)] * 22 + [(1, 129)] * 22 + [(1, 130)] * 22]
        self.assertEqual([row[:4] for row in derived], expected)

    def test_mutated_export_is_rejected(self):
        def mutate(change):
            inventory = copy.deepcopy(self.inventory)
            change(inventory)
            with self.assertRaises(ValueError):
                validate_inventory(tiny_capture_view(inventory))

        mutate(lambda d: d['calls'][3].update(module='model.layers.0.self_attn.o_projection'))
        mutate(lambda d: d['calls'][6]['input'][0].__setitem__(-1, 2048))
        mutate(lambda d: d['calls'][200]['output'][0].__setitem__(-1, 2560))
        mutate(lambda d: d['calls'].pop(154))
        mutate(lambda d: next(row for row in d['ops'] if row['source'].endswith(':270'))['outputs'][0][0].__setitem__(2, 160))


if __name__ == '__main__':
    unittest.main()
