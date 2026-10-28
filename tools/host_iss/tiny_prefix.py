"""Opt-in continuous full-ELF Tiny layer-zero boundary, with no guest restoration."""

import json
from pathlib import Path
import struct
import time

from atlas_tooling import sha as digest, write_json
from bundle import verify_kernel_receipt
from segment_boundary import SegmentChain
from tiny_full_contract import CYCLE_CAP, contract, require, EPILOGUE


def live_mapping(image, data):
    """Use this full ELF's compiled ABI, state pointers, and reference symbols."""
    symbol = image.symbols['tiny_full_compiled_live_layout']
    values = struct.unpack('<8Q', image.bytes_at(symbol['address'], 64))
    version, workspace_bytes, h0, h1, hidden_bytes, bank_bytes, phase_bytes, state_bytes = values
    require(version == 1 and hidden_bytes == 524288 and bank_bytes == 1802240
        and phase_bytes == 32 and state_bytes == 48 and h1 == h0 + hidden_bytes
        and h1 + hidden_bytes <= workspace_bytes <= image.symbols['workspace']['size'], 'full compiled live layout')
    state = struct.unpack('<5QII', image.bytes_at(image.symbols['tiny_full_state_global']['address'], 48))
    result = {f'{field}_{bank}': (state[2 * index + bank], bank_bytes)
              for index, field in enumerate(('keys', 'values')) for bank in range(2)}
    result.update({f'hidden_{bank}': (image.symbols['workspace']['address'] + offset, hidden_bytes)
                   for bank, offset in enumerate((h0, h1))})
    for item in data['reference_buffers']:
        name = item['actual_symbol']
        require(image.symbols[name]['size'] == item['bytes'], 'full reference extent')
        result[name] = image.symbols[name]['address'], item['bytes']
    for address, size in result.values():
        require(any(s['flags'] & 2 and s['address'] <= address and address + size <= s['address'] + s['memory_bytes']
                    for s in image.segments), 'full live mapping outside writable load extent')
    spans = list(result.values())
    require(all(a + n <= b or b + m <= a for i, (a, n) in enumerate(spans)
                for b, m in spans[:i]), 'full semantic live regions overlap')
    return result, values


def verify_prefix_uart(raw, build, launches):
    records = []
    for line in raw.decode('ascii').splitlines():
        if not line.startswith('ATLAS_KERNEL_'):
            continue
        fields = line.split()
        values = dict(field.split('=', 1) for field in fields[1:])
        require(len(values) == len(fields) - 1 and values.get('name') == build['result_name']
            and values.get('launches_expected') == str(build['result_progress_launches'][-1]), 'prefix UART identity')
        records.append((fields[0], values))
    require(len(records) == 2 and records[0][0] == 'ATLAS_KERNEL_BEGIN'
        and records[0][1]['launches_completed'] == '0' and records[1][0] == 'ATLAS_KERNEL_PROGRESS'
        and records[1][1]['launches_completed'] == str(launches)
        and records[1][1].get('boundary') == 'layer', 'prefix UART begin/progress coverage')
    return dict(begin=True, progress_records=1, launches_completed=launches, final='PREFIX_BOUNDARY')


class PrefixBoundary:
    def __init__(self, root, bundle, image):
        self.chain = SegmentChain(root, 'tiny-full')
        self.bundle, self.image, self.data = bundle, image, contract(bundle)
        require(bundle.workload == 'tiny-full' and not bundle.build.get('segment_mode'), 'full ELF prefix only')
        self.launches = self.chain.boundaries[1]['launch_cursor']
        require(self.launches == 12 and self.chain.kernel_count == bundle.kernel_count == 840, 'layer-zero launch partition')
        captured = json.loads(self.chain.read('contract.json'))
        require(captured['decode_count'] == self.data['decode_count'] == 2, 'same captured decode extent')
        for a, b in zip(captured['kernels'][:self.launches], self.data['kernels'][:self.launches]):
            for key in ('initial_sha256', 'physical_mkn', 'output_offset', 'A_scale', 'logical_rows', 'columns'):
                require(a[key] == b[key], 'captured/full kernel geometry: ' + key)
            for key in ('activation', 'weights', 'expected'):
                require(self.chain.sums[a[key]] == bundle.sums[b[key]], 'captured/full kernel payload: ' + key)
        require(captured['reference_buffers'] == self.data['reference_buffers'], 'same full reference inventory')
        require(captured['host_parameters'] == self.data['host_parameters'], 'same captured/full host parameters')
        for name in ('tiny_full_model.c', 'tiny_full_model.h'):
            require({h for p, h in bundle.build['source_sha256'].items() if p.rsplit('/', 1)[-1] == name}
                == {self.chain.sums['host/' + name]}, 'same captured/full arithmetic source')
        self.mapping, values = live_mapping(image, self.data)
        prep = json.loads(self.chain.read('segments/receipt.json'))
        layout = prep['compiled_live_layout']
        require(values == (layout['version'], layout['workspace_bytes'], *layout['hidden_offset'],
            layout['hidden_bytes'], layout['cache_bank_bytes'], layout['phase_bytes'], layout['state_bytes']),
            'captured/full compiled layout equality')
        require(set(self.mapping) | {'phase', 'control'} == set(self.chain.layout), 'all captured live regions mapped')
        for name, (_, size) in self.mapping.items():
            require(self.chain.layout[name]['bytes'] == size, 'captured/full live extent: ' + name)
        self.pc = image.symbols['tiny_full_model_layer']['address']
        self.length = next(iter(image.instructions('tiny_full_model_layer')))[1]
        self.admission = dict(chain_sha256sums_sha256=self.chain.checksums_sha256,
            boundary_index=1, launches=self.launches, stop_symbol='tiny_full_model_layer',
            stop_pc=self.pc, stop_layer_argument=1, compiled_live_layout=values,
            mapping={name: dict(address=a, bytes=n) for name, (a, n) in self.mapping.items()},
            phase_mapping='layer-entry a2 stack pointer',
            control_mapping='actual phase, state, projection context, and mailbox scalar fields')
        self.started = time.monotonic()

    def attach(self, remote, physical, out):
        self.physical, self.out = physical, out
        require(remote.read(self.pc, self.length) == self.image.bytes_at(self.pc, self.length), 'layer entry instruction')
        remote.breakpoint(self.pc, self.length)

    def admit_device(self, device):
        profile = dict(revision=device.binding['revision'], backend=device.memory_backend['backend'],
            curve_sha256=device.binding['curve_sha256'])
        require(profile == self.chain.model_profile and device.binding['tracked_source_diff_exit'] == 0,
                'prefix requires pinned clean genuine model and curve')

    def at_boundary(self, remote, device):
        layer = remote.reg(13)
        require(layer in (0, 1) and len(device.completed) == (0 if layer == 0 else self.launches),
                'layer entry launch cursor')
        require(remote.reg(11) == self.image.symbols['tiny_full_state_global']['address']
            and remote.reg(15) == self.image.symbols['tiny_full_projection_state']['address']
            and remote.reg(16) == self.image.symbols['workspace']['address'], 'full layer-entry pointer ABI')
        self.phase_address = remote.reg(12)
        require(self.image.symbols['__stack_bottom']['address'] <= self.phase_address
            and self.phase_address + 32 <= self.image.symbols['__stack_top']['address'], 'full stack phase extent')
        return layer == 1

    def heartbeat(self, device):
        write_json(self.out / 'progress.json', dict(utc=time.time(), elapsed_seconds=time.monotonic()-self.started,
            launches_completed=len(device.completed), current_model_cycles=device.core.cycle_count,
            completed_cycles=sum(k['cycles'] for k in device.completed),
            core_cycle_csr=device.core.arch_state.read_csrf(0xC00)))

    def verify(self, remote, device):
        require(remote.reg(32) == self.pc and remote.reg(13) == 1 and len(device.completed) == self.launches,
                'real host epilogue completed before layer one entry')
        state_raw = self.physical.read(self.image.symbols['tiny_full_state_global']['address'], 48)
        context_raw = self.physical.read(self.image.symbols['tiny_full_projection_state']['address'], 72)
        mailbox_raw = self.physical.read(self.image.symbols['tiny_full_mailbox']['address'], 128)
        host_raw = self.physical.read(self.image.symbols['tiny_full_host']['address'], 112)
        phase_raw = self.physical.read(self.phase_address, 32)
        state, context = struct.unpack('<5QII', state_raw), struct.unpack('<9Q', context_raw)
        mailbox, phase = struct.unpack('<16Q', mailbox_raw), struct.unpack('<Q6I', phase_raw)
        initial_state = struct.unpack('<5QII', self.image.bytes_at(self.image.symbols['tiny_full_state_global']['address'], 48))
        require(state == initial_state and phase == (0, 0, 0, 1, 128, 1, 0), 'prefix KV state and phase')
        require(mailbox[0] == 0x54494E5946554C4C and context[2] == mailbox[2] == self.launches and context[7] == 7 and mailbox[3] == 11
            and mailbox[1] == 1 and mailbox[4:7] == (0, 0, 0), 'guest projection and completion counters agree')
        require(struct.unpack_from('<QQ', host_raw, 72) == (0, 0)
            and struct.unpack_from('<6I', host_raw, 88)[1:3] == (0, 0)
            and struct.unpack_from('<I', host_raw, 108)[0] == self.launches, 'guest ownership drained')
        control = struct.pack('<8Q', mailbox[6], phase[3], state[4], state[5], state[6], phase[4], context[7], context[2])
        (self.out / 'raw').mkdir()
        for name, raw in (('state', state_raw), ('projection-context', context_raw), ('mailbox', mailbox_raw), ('host', host_raw)):
            (self.out / 'raw' / (name + '.bin')).write_bytes(raw)
        results = []
        for name, member in self.chain.boundaries[1]['regions'].items():
            raw = phase_raw if name == 'phase' else control if name == 'control' else self.physical.read(*self.mapping[name])
            (self.out / 'raw' / (name + '.bin')).write_bytes(raw)
            expected = self.chain.read(member)
            exact = raw == expected
            results.append(dict(name=name, bytes=len(raw), sha256=digest(raw), expected_sha256=digest(expected), exact=exact))
        write_json(self.out / 'boundary-comparison.json', results)
        require(all(item['exact'] for item in results), 'all145 live prefix regions must match captured boundary exactly')
        for index, summary in enumerate(device.completed):
            item = self.data['kernels'][index]
            verify_kernel_receipt(summary, dict(item, program=self.bundle.read(item['original']),
                expected_sha256=self.bundle.sums[item['expected']]), index, CYCLE_CAP)
        require(device.core.arch_state.read_csrf(0xC10) == device.core.arch_state.read_csrf(0xC11)
            == self.launches, 'guest invocation and device publication markers agree')
        cycles = sum(k['cycles'] for k in device.completed)
        require(cycles == device.core.cycle_count == device.core.arch_state.read_csrf(0xC00), 'measured cycle counters agree')
        self.heartbeat(device)
        return dict(prefix=True, continuous=True, segmented=False, restored=False, expected_starts=self.launches,
            layer_zero_host_epilogue_complete=True, layer_one_not_started=True, all_live_regions_exact=True,
            live_regions=results, measured_kernel_cycles=cycles, core_cycle_count=device.core.cycle_count,
            device_cycle_csr=device.core.arch_state.read_csrf(0xC00), guest_completed_launches=mailbox[2],
            tohost=int.from_bytes(self.physical.read(self.image.symbols['tohost']['address'], 8), 'little'))

    def verify_uart(self, raw, build):
        return verify_prefix_uart(raw, build, self.launches)
