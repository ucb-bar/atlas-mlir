"""Fixed semantic phase partitions and native reference live inventory."""
from itertools import groupby
from smol_full_plan import calls, panels, reference_geometry


def segments():
    sequence, physical = calls(), panels()
    if [x['index'] for x in sequence] != list(range(len(sequence))):
        raise ValueError('call indices are not in complete source order')
    def key(call):
        site, layer = call['site'], call['layer']
        if site == 7: return ('vision', call['image'], min(layer, 11))
        if site in (0, 5): return ('prefix', 0, layer if site == 5 else 0)
        return ('expert', call['step'], layer if site == 6 else 15 if site == 4 else 0)
    result = []
    for identity, grouped in groupby(sequence, key):
        group = list(grouped)
        start, end = group[0]['index'], group[-1]['index'] + 1
        covered = [p['index'] for p in physical if start <= p['call'] < end]
        if covered != list(range(covered[0], covered[-1] + 1)):
            raise ValueError('noncontiguous physical phase')
        result.append(dict(index=len(result), kind=identity[0], ordinal=identity[1], layer=identity[2],
                           call_start=start, call_end=end, launch_start=covered[0], launch_end=covered[-1]+1))
    if len(result) != 200 or result[0]['launch_start'] != 0 or result[-1]['launch_end'] != len(physical):
        raise ValueError('incomplete fixed phase partition')
    for left, right in zip(result, result[1:]):
        if left['launch_end'] != right['launch_start'] or left['call_end'] != right['call_start']:
            raise ValueError('overlap or gap between phases')
    return result


EXTRA_LIVE = {'smol_v_tokens':1572864, 'smol_v_output':122880, 'smol_full_image_output':245760,
              'smol_b_output':339840, 'smol_g_velocity':6400, 'smol_full_cycles':10860,
              'smol_full_graph_state':16, 'smol_full_vision_states':32, 'smol_full_phase_state':40}


def live_regions(data, image):
    expected = reference_geometry()
    records = data['reference_buffers']
    if len(records) != 47 or {x['name'] for x in records} != set(expected):
        raise ValueError('exact complete reference inventory required')
    result = dict(EXTRA_LIVE)
    for row in records:
        shape, dtype = expected[row['name']]
        count = 1
        for dim in shape: count *= dim
        size = count * {'float32':4, 'uint16':2, 'bool':1}[dtype]
        if row['bytes'] != size: raise ValueError('reference inventory size')
        result[row['actual_symbol']] = size
    for name, size in result.items():
        if name not in image.symbols or image.symbols[name]['size'] != size:
            raise ValueError('live symbol extent differs: '+name)
    return result


def scratch_regions(image, live):
    result = {}
    exact = {'smol_full_input_copy','smol_full_activation_copy','smol_full_image'}
    for name, row in image.symbols.items():
        if name not in live and (name.startswith(('smol_v_','smol_b_','smol_g_')) or name in exact):
            if row['size']: result[name] = row['size']
    return result


def check_observations(values, start, end):
    """Validate only genuinely executed current-range deltas before zeroing.

    Native values are synthetic protocol observations, never measured timing.
    Real segment admission must additionally match each value to Device receipts.
    Prior and future slots are canonical zero; no fake prior nonzero seeds exist.
    """
    count = len(panels())
    if (len(values) != count or type(start) is not int or type(end) is not int
        or not 0 <= start < end <= count
        or any(type(x) is not int or not 0 <= x < 1 << 32 for x in values)):
        raise ValueError('invalid segment observation array/range')
    if any(values[:start]) or any(values[end:]) or not all(values[start:end]):
        raise ValueError('observations must cover only the actual current range')
    return values[start:end]
