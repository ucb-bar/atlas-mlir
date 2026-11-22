"""Boundary checks use synthetic objects; they make no numerical claim."""
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from atlas_mlir import deployment_link as d
from atlas_mlir.deploy import Refusal, digest


class LinkBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.generated = self.root / 'generated'
        self.generated.mkdir()
        self.bundle = self.root / 'bundle'
        (self.bundle / 'handoff').mkdir(parents=True)
        (self.bundle / 'application').mkdir()
        (self.bundle / 'application' / 'deployment.json').write_text('{"window": {"address": 0}, "inputs": [], "outputs": []}')
        self.obj = self.root / 'code.o'
        self.obj.write_bytes(b'synthetic object')
        self.receipt = self.root / 'sections.json'
        self.receipt.write_text('{}')
        self.gate_receipt = self.root / 'check.json'
        self.gate_receipt.write_text('{}')
        for name in ('chip.ld', 'window.S', 'window.bin', 'harness_sha256.h'):
            (self.generated / name).write_bytes(name.encode())
            (self.bundle / 'handoff' / name).write_bytes(name.encode())
        self.hashes = {str(path): digest(path) for path in (self.obj, self.receipt)}
        self.gate = SimpleNamespace(hashes=dict(self.hashes), receipt=self.gate_receipt, verify=Mock())
        self.context = dict(out=self.root, bundle=self.bundle, settings={}, gate=self.gate,
                            objects=dict(hashes=self.hashes, generated=self.generated,
                                         root=self.root, build=self.root / 'build', receipt=self.receipt))

    def test_missing_or_failing_gate_never_invokes_build(self):
        with patch.object(d, '_run') as run:
            self.context.pop('gate')
            with self.assertRaisesRegex(Refusal, 'gate'):
                d.link(self.context)
            self.context['gate'] = self.gate
            self.gate.verify.side_effect = Refusal('executed mutation missing')
            with self.assertRaisesRegex(Refusal, 'mutation'):
                d.link(self.context)
            run.assert_not_called()

    def test_changed_object_never_invokes_link(self):
        self.obj.write_bytes(b'changed')
        with patch.object(d, '_run') as run:
            with self.assertRaisesRegex(Refusal, 'hash drift'):
                d.link(self.context)
            run.assert_not_called()

    def test_unbound_object_never_invokes_link(self):
        self.gate.hashes = {}
        with patch.object(d, '_run') as run:
            with self.assertRaisesRegex(Refusal, 'authenticate measured'):
                d.link(self.context)
            run.assert_not_called()

    def test_bound_payload_drift_never_invokes_link(self):
        (self.bundle / 'handoff' / 'window.bin').write_bytes(b'changed')
        with patch.object(d, '_run') as run:
            with self.assertRaisesRegex(Refusal, 'payload drift'):
                d.link(self.context)
            run.assert_not_called()

    def test_storage_reservations_refuse_before_link(self):
        script = '.stack (NOLOAD) : { . += 12288; } >HOST :host'
        for root in (self.generated, self.bundle / 'handoff'):
            (root / 'chip.ld').write_text(script)
        with patch.object(d, '_run') as run:
            with self.assertRaisesRegex(Refusal, 'symbolic reservations required'):
                d.link(self.context)
            run.assert_not_called()

    def test_gate_rechecked_immediately_before_and_after_link(self):
        events = []
        self.gate.verify.side_effect = lambda: events.append('verify')
        record = {'sha256': 'synthetic', 'host_reservations': {'__stack_end': 8192}, 'window_zero': []}
        self.context['objects']['host_linked'] = {'__stack_end': 8192}
        with patch.object(d.shutil, 'which', return_value='/usr/bin/cmake'), \
             patch.object(d, '_run', side_effect=lambda *args, **kwargs: events.append('link')) as run, \
             patch.object(d, '_elf_record', return_value=record):
            elf = d.link(self.context)
        self.assertEqual(events, ['verify', 'verify', 'link', 'verify', 'verify'])
        self.assertEqual(run.call_args.args[0][-3:], ['--target', 'atlas_deployment', '-j2'])
        self.assertEqual(elf, self.root / 'build' / 'atlas_deployment.elf')
        self.assertEqual(self.context['elf'], elf)
        self.assertTrue((self.root / 'link.json').is_file())
        (self.root / 'link.json').unlink()
        record['window_zero'] = [{'address': 0, 'bytes': 64}]
        with patch.object(d.shutil, 'which', return_value='/usr/bin/cmake'), patch.object(d, '_run'), \
             patch.object(d, '_elf_record', return_value=record):
            with self.assertRaisesRegex(Refusal, 'zero table differs from the bound window image'):
                d.link(self.context)
        record['window_zero'] = []
        record['host_reservations']['__stack_end'] = 8208
        with patch.object(d.shutil, 'which', return_value='/usr/bin/cmake'), patch.object(d, '_run'), \
             patch.object(d, '_elf_record', return_value=record):
            with self.assertRaisesRegex(Refusal, 'differs from the measured sizing link'):
                d.link(self.context)

    def test_object_drift_during_link_blocks_publication(self):
        with patch.object(d.shutil, 'which', return_value='/usr/bin/cmake'), \
             patch.object(d, '_run', side_effect=lambda *args, **kwargs: self.obj.write_bytes(b'changed')), \
             patch.object(d, '_elf_record') as inspect:
            with self.assertRaisesRegex(Refusal, 'hash drift'):
                d.link(self.context)
            inspect.assert_not_called()
        self.assertNotIn('elf', self.context)

    def test_compile_inventory_refuses_unmanaged_object(self):
        build = self.root / 'build'
        build.mkdir()
        (build / 'compile_commands.json').write_text(json.dumps([
            dict(directory=str(build), file=str(self.root / 'source.c'),
                 arguments=['cc', '-c', 'source.c', '-o', str(self.obj)])]))
        with self.assertRaisesRegex(Refusal, 'outside managed'):
            d._object_paths(build)

    def test_compile_inventory_retains_only_built_object_sources(self):
        build = self.root / 'build'
        build.mkdir()
        obj = build / 'present.o'
        obj.write_bytes(b'x')
        source = self.root / 'source.c'
        source.write_text('')
        (build / 'compile_commands.json').write_text(json.dumps([
            dict(directory=str(build), file=str(source), command='cc -c source.c -o present.o'),
            dict(directory=str(build), file=str(self.root / 'unused.c'), command='cc -c unused.c -o absent.o')]))
        self.assertEqual(d._object_paths(build), ([obj], [source]))

    def test_board_programs_exit_the_declared_way_and_jtag_programs_do_not(self):
        from atlas_mlir_native import soc
        generated = {'handoff/' + name: b'' for name in ('chip.ld', 'window.S', 'window.bin', 'harness_sha256.h')}
        deployment = dict(calls=[dict(placement='device')], window=dict(address=0), inputs=[], outputs=[])
        prepared = dict(platform_deployment=deployment, generated=generated)
        cases = (('spi', 'tohost', 'ON'), ('jtag', 'tohost', 'OFF'), ('spi', 'return', 'OFF'))
        for number, (transport, completion, expected) in enumerate(cases):
            out = self.root / str(number)
            handoff = out / 'compiled' / 'application' / 'handoff'
            handoff.mkdir(parents=True)
            for name in ('app.c', 'app.h', 'deployment_adapter.h'):
                (handoff / name).write_text('')
            (handoff / 'abi.json').write_text(json.dumps(dict(source='app.c', header='app.h')))
            (handoff.parent / 'application-c.json').write_text('{}')
            context = dict(out=out, transport=transport,
                           settings=dict(clock_hz=500000000, mtime_hz=500000, receiver_baud=115200))
            with patch.dict(d.os.environ, ATLAS_BAREMETAL_IDE=str(self.root)), \
                 patch.object(d.shutil, 'which', return_value='/usr/bin/cmake'), \
                 patch.dict(soc.BOARD_PROGRAM, completion=completion), \
                 patch('atlas_mlir.deployment_gate.prepare_binding', return_value=prepared), \
                 patch.object(d, '_run', side_effect=Refusal('configure reached')) as run:
                with self.assertRaisesRegex(Refusal, 'configure reached'):
                    d.objects(context)
            configure = run.call_args.args[0]
            self.assertIn('-DPENGUIN26_EXIT_TOHOST=' + expected, configure)
            self.assertIn('-DPENGUIN26_UART_BAUDRATE=115200', configure)

    def test_declared_board_program_completion_is_validated(self):
        from atlas_mlir_native import soc
        self.assertEqual(soc.BOARD_PROGRAM['completion'], 'tohost')
        self.assertIn('Puffin', soc.BOARD_PROGRAM['evidence'])
        document = json.loads(soc.DECLARATION.read_text())
        for change in (dict(completion='halt'), dict(start=''), dict(extra='x')):
            with self.assertRaisesRegex(ValueError, 'board_program'):
                soc.board_program({**document, 'board_program': {**document['board_program'], **change}})
        with self.assertRaisesRegex(ValueError, 'board_program'):
            soc.board_program({key: value for key, value in document.items() if key != 'board_program'})


class SizingLinkTests(unittest.TestCase):
    def test_sizing_link_reports_placement_or_refuses(self):
        import shutil
        compiler = shutil.which('cc')
        self.assertIsNotNone(compiler, 'a C compiler is required')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'layout.S').write_text('.globl _start\n_start: .byte 0\n')
            symbols = ' '.join(f'-Wl,--defsym={name}={value}' for name, value in
                               (('__bss_end', 4096), ('__heap_start', 4096), ('__stack_end', 8192)))
            command = root / 'link.txt'
            command.write_text(f'{compiler} -nostdlib -no-pie {symbols} layout.S -o out.elf -Wl,-Map,out.map')
            placed = d._sizing_link(root, command, root, None)
            self.assertEqual(placed, {'__bss_end': 4096, '__heap_start': 4096, '__stack_end': 8192})
            self.assertTrue((root / 'sizing.map').is_file())
            command.write_text('true -o out.elf -Wl,-Map,out.map')
            (root / 'sizing.elf').unlink()
            with self.assertRaisesRegex(Refusal, 'sizing link failed'):
                d._sizing_link(root, command, root, None)
            command.write_text('true out.elf')
            with self.assertRaisesRegex(Refusal, 'one output and one map'):
                d._sizing_link(root, command, root, None)


class GeneratedStackTests(unittest.TestCase):
    """The stack bound covers generated C only, from GCC's -fstack-usage/-fcallgraph-info=su data."""

    GENERATED = """typedef int (*step)(int);
__attribute__((noinline)) int leaf(int x) { volatile char b[40]; b[0] = (char)x; return b[0]; }
__attribute__((noinline)) int callback(int x) { volatile char b[24]; b[0] = (char)x; return b[0] + 1; }
__attribute__((noinline)) int other(int x) { volatile char b[8]; b[0] = (char)x; return b[0] + 2; }
static step const table[] = {callback, other};
int run(int x) { volatile char b[64]; b[0] = (char)x; return table[x & 1](b[0]) + leaf(x); }
"""

    def objects(self, source):
        import shutil
        compiler, readelf = shutil.which('riscv64-unknown-elf-gcc'), shutil.which('riscv64-unknown-elf-readelf')
        self.assertTrue(compiler and readelf, 'RISC-V GCC and readelf are required')
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / 'generated.c').write_text(source)
        subprocess.run([compiler, '-Os', '-ffunction-sections', '-fdata-sections', '-fstack-usage', '-fcallgraph-info=su',
                        '-c', 'generated.c', '-o', 'generated.c.obj'], cwd=root, check=True, capture_output=True)
        return [root / 'generated.c.obj'], readelf

    def test_bound_sums_su_frames_through_resolved_indirect_calls(self):
        objects, readelf = self.objects(self.GENERATED)
        usage = d.generated_stack(objects, readelf)
        su = {line.split()[0].rsplit(':', 1)[-1]: int(line.split()[1])
              for line in (objects[0].parent / 'generated.c.su').read_text().splitlines()}
        self.assertEqual(usage['indirect_targets'], ['callback', 'other'])
        self.assertEqual(usage['generated_stack_bytes'], su['run'] + max(su['callback'], su['other'], su['leaf']))
        self.assertEqual(usage['generated_worst_path'][0], 'run')

    def test_equally_deep_callees_give_one_worst_path_under_any_hash_seed(self):
        callee = '__attribute__((noinline)) int %s(int x) { volatile char b[40]; b[0] = (char)x; return b[0] + %d; }\n'
        names = ('tie_d', 'tie_b', 'tie_a', 'tie_c')
        source = ''.join(callee % (name, i) for i, name in enumerate(names))
        source += 'int run(int x) { return ' + ' + '.join(f'{name}(x)' for name in names) + '; }\n'
        objects, readelf = self.objects(source)
        su = {line.split()[0].rsplit(':', 1)[-1]: int(line.split()[1])
              for line in (objects[0].parent / 'generated.c.su').read_text().splitlines()}
        self.assertEqual(len({su[name] for name in names}), 1, 'the callees must tie')
        script = ('import sys; from atlas_mlir import deployment_link as d; '
                  'print(d.generated_stack([sys.argv[1]], sys.argv[2])["generated_worst_path"])')
        paths = {subprocess.run([sys.executable, '-c', script, str(objects[0]), readelf], check=True, text=True,
                                capture_output=True, env={**os.environ, 'PYTHONHASHSEED': str(seed)}).stdout
                 for seed in range(6)}
        self.assertEqual(paths, {str(['run', 'tie_a']) + '\n'})

    def test_static_functions_of_one_name_in_two_objects_stay_distinct(self):
        source = 'static __attribute__((noinline)) int helper(int x) { volatile char b[%d]; b[0] = (char)x; return b[0]; }\n' \
                 'int entry_%d(int x) { return helper(x) + 1; }\n'
        first, readelf = self.objects(source % (16, 1))
        second, _ = self.objects(source % (96, 2))
        usage = d.generated_stack(first + second, readelf)
        self.assertEqual(usage['generated_worst_path'][0], 'entry_2')

    def test_identical_static_helpers_folded_by_gcc_resolve_by_symbol(self):
        # GCC's identical code folding (on at -Os and -O2) keeps one body for two identical static helpers; the
        # other's call graph node has no frame, and its local symbol sits at the kept body's address.
        source = ('static __attribute__((noinline)) int round_a(int x) { volatile char b[24]; b[0] = (char)x; return b[0] + 3; }\n'
                  'static __attribute__((noinline)) int round_b(int x) { volatile char b[24]; b[0] = (char)x; return b[0] + 3; }\n'
                  'int entry(int x) { volatile char b[32]; b[0] = (char)x; return round_a(b[0]) * round_b(x); }\n')
        objects, readelf = self.objects(source)
        su = {line.split()[0].rsplit(':', 1)[-1]: int(line.split()[1])
              for line in (objects[0].parent / 'generated.c.su').read_text().splitlines()}
        self.assertEqual(len({'round_a', 'round_b'} & su.keys()), 1, su)  # folded: one of the two has a frame
        usage = d.generated_stack(objects, readelf)
        kept, = {'round_a', 'round_b'} & su.keys()
        self.assertEqual(usage['generated_worst_path'], ['entry', 'generated.c:' + kept])
        self.assertEqual(usage['generated_stack_bytes'], su['entry'] + su[kept])

    def test_recursion_dynamic_frames_and_unresolved_indirect_calls_refuse(self):
        cases = {
            'call cycle through depth': 'int depth(int x) { volatile char b[16]; b[0] = (char)x; '
                                        'return x ? depth(x - 1) + depth(x - 2) + b[0] : 0; }\n',
            'call cycle through spin': 'typedef int (*step)(int); int spin(int x); int stop(int x);\n'
                                       'static step const table[] = {spin, stop};\n'
                                       '__attribute__((noinline)) int stop(int x) { return x; }\n'
                                       'int spin(int x) { volatile char b[16]; b[0] = (char)x; return table[x & 1](b[0]) + 1; }\n',
            'dynamic': 'int grow(int n) { volatile char *p = __builtin_alloca(n); p[0] = 1; return p[0]; }\n',
            'no resolvable target': 'int call(int (*f)(int), int x) { volatile char b[8]; b[0] = (char)x; '
                                    'return f(b[0]) + 1; }\n',
        }
        for expected, source in cases.items():
            with self.subTest(expected=expected):
                objects, readelf = self.objects(source)
                with self.assertRaisesRegex(Refusal, expected):
                    d.generated_stack(objects, readelf)

    def test_resident_regions_load_their_image_by_direct_call_so_the_bound_accepts_them(self):
        """The resident runtime's call shape: a region binds and loads its image, then dispatches with the status.

        The image load reaches the host transport through an indirect call. Passed as a callback, the region's
        load is address-taken, so that indirect call may reach it and the bound refuses a cycle; called directly,
        the bound accepts it.
        """
        from atlas_mlir_native.quantized_loop_c import _resident_dispatch_helpers
        helpers = _resident_dispatch_helpers()
        self.assertIn('void *const *outputs,atlas_host_result loaded) {', helpers)
        self.assertNotIn('(*load)', helpers)
        common = ('typedef int (*transport)(int);\n'
                  '__attribute__((noinline)) int host_write(int x) { volatile char b[16]; b[0] = (char)x; return b[0]; }\n'
                  '__attribute__((noinline)) int host_read(int x) { volatile char b[8]; b[0] = (char)x; return b[0]; }\n'
                  'static transport const host_io[] = {host_write, host_read};\n'
                  '__attribute__((noinline)) int host_load(int x) { volatile char b[16]; b[0] = (char)x; '
                  'return host_io[x & 1](b[0]); }\n'
                  '__attribute__((noinline)) int region_resident_image_bind(int x) { return host_load(x) + 1; }\n'
                  '__attribute__((noinline)) int region_resident_load(int x) { return region_resident_image_bind(x); }\n')
        direct = common + ('__attribute__((noipa)) int dispatch(int x, int loaded) { volatile char b[32]; '
                           'b[0] = (char)x; return b[0] + loaded; }\n'
                           'int region(int x) { return dispatch(x, region_resident_load(x)); }\n')
        callback = common + ('__attribute__((noipa)) int dispatch(int x, int (*load)(int)) { volatile char b[32]; '
                             'b[0] = (char)x; return b[0] + load(x); }\n'
                             'int region(int x) { return dispatch(x, region_resident_load); }\n')
        objects, readelf = self.objects(direct)
        usage = d.generated_stack(objects, readelf)
        self.assertEqual(usage['generated_worst_path'][:2], ['region', 'region_resident_load'])
        objects, readelf = self.objects(callback)
        with self.assertRaisesRegex(Refusal, 'call cycle through .*region_resident_load'):
            d.generated_stack(objects, readelf)

    def compile_files(self, *sources, directory=None):
        import shutil
        compiler, readelf = shutil.which('riscv64-unknown-elf-gcc'), shutil.which('riscv64-unknown-elf-readelf')
        self.assertTrue(compiler and readelf, 'RISC-V GCC and readelf are required')
        if directory is None:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            directory = Path(temporary.name)
        objects = []
        for source in sources:
            target = directory / source.name
            target.write_text(source.read_text())
            subprocess.run([compiler, '-Os', '-ffunction-sections', '-fdata-sections', '-fstack-usage', '-fcallgraph-info=su',
                            '-c', str(target), '-o', str(target) + '.obj'], check=True, capture_output=True)
            objects.append(Path(str(target) + '.obj'))
        return objects, readelf

    def test_review_inputs_through_aliases_library_callbacks_and_pointers(self):
        cases = Path(__file__).parent / 'stack_counterexamples' / 'generated'
        refused = {('alias.c',): 'call cycle through f -> f',  # recursion through an alias of f
                   ('library_callback.c',): 'call cycle through run -> compare -> run',  # run -> qsort -> compare -> run
                   ('static_library_callback.c',): 'call cycle through cmp -> cmp',  # cmp -> qsort -> cmp
                   ('cold.c',): 'call cycle through f -> f',
                   ('cross_a.c', 'cross_b.c'): 'call cycle through f -> f',
                   ('weak_a.c', 'weak_b.c'): 'call cycle through f -> f',  # a weak function taken in another object
                   ('direct_alias.c',): 'call cycle through f -> f',  # f calls its own alias h directly
                   ('cross_alias_a.c', 'cross_alias_b.c'): 'call cycle through f -> g -> f',
                   # untyped assembly code (global or local label) with its address taken has no parsed frame
                   ('asm_label.c',): 'lie outside every generated function with a parsed frame',
                   ('asm_local.c',): 'lie outside every generated function with a parsed frame',
                   # the same code in a section named "extra code": refused, never skipped as an unreadable row
                   ('asm_label_spaced_section.c',): 'extra code of .* lie outside|cannot be read reliably',
                   ('asm_local_spaced_section.c',): 'extra code of .* lie outside|cannot be read reliably'}
        for names, expected in refused.items():
            with self.subTest(names=names):
                objects, readelf = self.compile_files(*(cases / name for name in names))
                with self.assertRaisesRegex(Refusal, expected):
                    d.generated_stack(objects, readelf)
        # A direct call through an alias in another object resolves to the aliased function and counts its frame.
        objects, readelf = self.compile_files(cases / 'acyclic_alias_a.c', cases / 'acyclic_alias_b.c')
        usage = d.generated_stack(objects, readelf)
        su = {}
        for obj in objects:
            for line in Path(str(obj).rsplit('.', 1)[0] + '.su').read_text().splitlines():
                su[line.split()[0].rsplit(':', 1)[-1]] = int(line.split()[1])
        self.assertEqual(usage['generated_stack_bytes'], su['g'] + su['f'])
        self.assertEqual(usage['generated_worst_path'], ['g', 'f'])
        self.assertNotIn('h', usage['library_callees'])
        # An indirect call that may reach a library function lists it as a library callee and indirect target.
        objects, readelf = self.compile_files(cases / 'libptr.c')
        usage = d.generated_stack(objects, readelf)
        self.assertIn('memset', usage['library_callees'])
        self.assertIn('memset', usage['indirect_targets'])

    def test_section_headers_are_read_from_the_elf_whatever_their_names(self):
        cases = Path(__file__).parent / 'stack_counterexamples' / 'generated'
        objects, _ = self.compile_files(cases / 'asm_label_spaced_section.c')
        headers = {name: (flags, size) for name, kind, flags, size in d._section_headers(objects[0].read_bytes())}
        self.assertIn('extra code', headers)
        self.assertTrue(headers['extra code'][0] & 0x4)
        self.assertEqual(headers['extra code'][1], 4)

    def test_inline_asm_that_moves_sp_is_outside_the_rule(self):
        # Documents a stated limit, not a guarantee: GCC's su data does not see inline asm, so these
        # inputs loop on addi sp,sp,-16 yet get a finite bound by design. The rule assumes generated
        # inline asm neither writes sp nor transfers control; the generated asm is fence and rdcycle.
        cases = Path(__file__).parent / 'stack_counterexamples' / 'generated'
        for name in ('inline_asm_sp.c', 'naked.c'):
            with self.subTest(name=name):
                objects, readelf = self.compile_files(cases / name)
                usage = d.generated_stack(objects, readelf)
                self.assertEqual(usage['generated_worst_path'], ['run'])
                self.assertLess(usage['generated_stack_bytes'], 64)

    def test_unparsable_su_data_is_refused_not_treated_as_external(self):
        # A source path with a double quote breaks the call graph's quoting; it must refuse, not bound alloca at 0 B.
        with tempfile.TemporaryDirectory() as directory:
            quoted = Path(directory) / 'quoted"path'
            quoted.mkdir()
            source = quoted / 'generated.c'
            source.write_text('int grow(int n) { volatile char *p = __builtin_alloca(n); p[0] = 1; return p[0]; }\n')
            objects, readelf = self.compile_files(source, directory=quoted)
            with self.assertRaisesRegex(Refusal, 'malformed GCC call graph|dynamic stack usage'):
                d.generated_stack(objects, readelf)

    def test_code_without_su_data_gets_no_static_bound(self):
        # The removed binary analysis bounded these hand-written programs at 16 B while each grows the
        # stack without limit (review counterexamples). Code without GCC su data, such as newlib and the
        # platform startup, is never bounded statically now: its stack is measured on the shipped probes.
        import shutil
        compiler, readelf = shutil.which('riscv64-unknown-elf-gcc'), shutil.which('riscv64-unknown-elf-readelf')
        self.assertTrue(compiler and readelf, 'RISC-V GCC and readelf are required')
        sources = sorted((Path(__file__).parent / 'stack_counterexamples').glob('*.S'))
        self.assertEqual(len(sources), 8)
        for source in sources:
            with self.subTest(source=source.name), tempfile.TemporaryDirectory() as directory:
                obj = Path(directory) / (source.name + '.obj')
                subprocess.run([compiler, '-c', str(source), '-o', str(obj)], check=True, capture_output=True)
                with self.assertRaisesRegex(Refusal, 'lacks its GCC call graph'):
                    d.generated_stack([obj], readelf)


class ImageSymbolTests(unittest.TestCase):
    def test_duplicate_image_symbols_refuse(self):
        import shutil
        compiler = shutil.which('cc')
        self.assertIsNotNone(compiler, 'a C compiler is required')
        name = 'atlas_image_' + '0' * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2):
                (root / f'{index}.c').write_text(f'static const unsigned {name}[1]={{0x73u}};\n'
                                                 f'const unsigned *image_{index}(void) {{ return {name}; }}\n')
            (root / 'main.c').write_text('int main(void) { return 0; }\n')
            for sources, refused in ((['0.c', 'main.c'], False), (['0.c', '1.c', 'main.c'], True)):
                with self.subTest(sources=sources):
                    subprocess.run([compiler, *sources, '-o', 'out'], cwd=root, check=True, capture_output=True)
                    raw = (root / 'out').read_bytes()
                    if refused:
                        with self.assertRaisesRegex(Refusal, 'duplicate instruction image symbol'):
                            d._elf_symbols(raw)
                    else:
                        self.assertIn(name, d._elf_symbols(raw)[1])


class HostArchiveTests(unittest.TestCase):
    def test_measured_host_binds_the_window_after_it(self):
        from atlas_mlir_native.soc import SOC
        ddr = SOC.memory_region('ddr')
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        out = Path(temporary.name) / 'deploy'
        for name in ('compiled', 'objects'):
            (out / name).mkdir(parents=True)
        context = dict(out=out)
        def objects(context):
            context['objects'] = dict(host_linked=dict(__stack_end=ddr.base + 27136, bytes=27136))
            return context['objects']
        arguments = ['-o', out / 'compiled']
        with patch.object(d, 'objects', side_effect=objects):
            bound = d.measure_host(context, arguments)
        self.assertEqual(bound, ['-o', str(out / 'compiled'), '--io-memory-bytes', str(ddr.capacity_bytes - 28672)])
        self.assertEqual(sorted(p.name for p in (out / 'measure').iterdir()), ['compiled', 'host.json', 'objects'])
        self.assertFalse((out / 'compiled').exists() or 'objects' in context)
        record = json.loads((out / 'measure/host.json').read_text())
        self.assertEqual((record['reserved_bytes'], record['window_base']), (28672, ddr.base + 28672))

    def onchip(self, host, staged=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        out = Path(temporary.name)
        (out / 'compiled/application').mkdir(parents=True)
        (out / 'objects').mkdir()
        deployment = dict(mode='bench_staged') if staged else dict(host_allocation=dict(usable_bytes=65536))
        (out / 'compiled/application/deployment.json').write_text(json.dumps(deployment))
        context = dict(out=out)
        def objects(context):
            context['objects'] = dict(host_linked=dict(__stack_end=0x08000000 + host, bytes=host))
            return context['objects']
        with patch.object(d, 'objects', side_effect=objects) as measured:
            bound = d.measure_onchip(context, ['-o', out / 'compiled'])
        return out, context, bound, measured

    def test_an_onchip_host_that_fits_keeps_its_measured_objects(self):
        out, context, bound, _ = self.onchip(65536)
        self.assertIsNone(bound)
        self.assertEqual(context['objects']['host_linked']['bytes'], 65536)
        self.assertTrue((out / 'compiled').is_dir() and not (out / 'measure').exists())

    def test_an_onchip_host_over_the_declared_host_compiles_again_with_its_size(self):
        out, context, bound, _ = self.onchip(113552)
        self.assertEqual(bound, ['-o', str(out / 'compiled'), '--measured-host-bytes', '113552'])
        self.assertEqual(sorted(p.name for p in (out / 'measure').iterdir()), ['compiled', 'host.json', 'objects'])
        self.assertFalse((out / 'compiled').exists() or 'objects' in context)
        record = json.loads((out / 'measure/host.json').read_text())
        self.assertEqual((record['host_linked']['bytes'], record['usable_bytes']), (113552, 65536))

    def test_a_staged_onchip_plan_is_not_measured_whole(self):
        _, context, bound, measured = self.onchip(113552, staged=True)
        self.assertIsNone(bound)
        measured.assert_not_called()

    def test_measured_host_over_the_declared_ddr_refuses_before_moving_outputs(self):
        from atlas_mlir_native.soc import SOC
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'compiled').mkdir()
            context = dict(out=out)
            size = SOC.memory_region('ddr').capacity_bytes
            with patch.object(d, 'objects', return_value=dict(host_linked=dict(bytes=size))):
                with self.assertRaisesRegex(Refusal, 'measured host exceeds the declared DDR'):
                    d.measure_host(context, [])
            self.assertTrue((out / 'compiled').is_dir())
            self.assertFalse((out / 'measure').exists())

    def test_declared_archive_must_be_supplied_by_hash_before_any_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            application = out / 'compiled' / 'application'
            (application / 'handoff').mkdir(parents=True)
            (application / 'handoff' / 'abi.json').write_text(json.dumps(dict(source='a.c', header='a.h')))
            archive = out / 'libmath.a'
            archive.write_bytes(b'archive')
            other = out / 'libother.a'
            other.write_bytes(b'other')
            (application / 'application-c.json').write_text(json.dumps(dict(host_linkage=[dict(
                schema='merlin.host_linkage_contract.v1', archive_sha256=digest(archive), symbols=['cosf'])])))
            context = dict(transport='jtag', out=out, host_archives=[other])
            prepared = dict(platform_deployment=dict(calls=[dict(placement='host')]))
            with patch.dict('os.environ', ATLAS_BAREMETAL_IDE=tmp), patch.object(d.shutil, 'which', return_value='tool'), \
                    patch('atlas_mlir.deployment_gate.prepare_binding', return_value=prepared), \
                    patch.object(d, '_run') as build:
                with self.assertRaisesRegex(Refusal, 'declared host math archive is not supplied'):
                    d.objects(context)
            build.assert_not_called()
            self.assertFalse((out / 'objects').exists())


class StartupTests(unittest.TestCase):
    def check(self, text, words=''):
        """Startup check of objdump text, with executable code at 0x8000404 given as 16- or 32-bit hex words."""
        result = SimpleNamespace(returncode=0, stdout=text)
        code = b''.join(int(word, 16).to_bytes(len(word) // 2, 'little') for word in words.split())
        with patch.object(d.shutil, 'which', return_value='objdump'), patch.object(d.subprocess, 'run', return_value=result), \
                patch.object(d, '_executable_code', return_value=[(0x8000404, code)]):
            return d._startup_check(Path('model.elf'))

    good = """08000000 <_start>:
  8000000: jal 8000100 <__wrap_main>
08000100 <__wrap_main>:
  8000100: jal 8000200 <uart_init>
  8000104: jal 8000300 <main>
08000200 <uart_init>:
08000300 <main>:
08000400 <trap_vector>:
  8000400: csrr a0,mcause
"""

    def test_platform_initializes_uart(self):
        self.assertTrue(self.check(self.good)['uart_init_reachable'])

    def test_unreachable_uart_refuses(self):
        with self.assertRaisesRegex(Refusal, 'UART'):
            self.check(self.good.replace('jal 8000200 <uart_init>', 'nop'))

    def test_unsupported_trap_csr_refuses(self):
        with self.assertRaisesRegex(Refusal, 'unsupported CSR'):
            self.check(self.good + '  8000404: csrrci a0,0x401,2\n')

    def test_records_clear_core_errata(self):
        self.assertTrue(self.check(self.good)['core_errata_clear'])

    def test_fence_i_refuses(self):
        refusal = r'fence.i at 0x8000404 in trap_vector.*issue 19'
        for words in ('0000100f', '0050100f'):  # fence.i, and the same opcode and funct3 with a nonzero immediate
            with self.subTest(words=words), self.assertRaisesRegex(Refusal, refusal):
                self.check(self.good, words)

    def test_vfmv_f_s_refuses(self):
        refusal = r'vfmv.f.s at 0x8000404 in trap_vector.*PR 22'
        for words in ('42801757', '40801757'):  # vfmv.f.s fa4,v8, and the same with vm clear
            with self.subTest(words=words), self.assertRaisesRegex(Refusal, refusal):
                self.check(self.good, words)
        self.check(self.good, '42802657')  # vmv.x.s a2,v8 writes no FP register

    def test_fp_write_then_same_register_load_refuses(self):
        # fsub.s, fmadd.d, fsqrt.d, fmv.d.x, and fld, each writing fa4; then fld fa4,0(a1).
        for writer in ('08b57753', '62b57743', '5a057753', 'f2050753', '0085b707'):
            refusal = r'writes fa4 at 0x8000404 in trap_vector.*issue 20'
            with self.subTest(writer=writer), self.assertRaisesRegex(Refusal, refusal):
                self.check(self.good, writer + ' 0005b707')

    def test_fp_load_without_same_packet_write_passes(self):
        # fsub.s fa5, fsd fa4, fmv.x.d a0, feq.d a0, add, and fcvt.w.d a4, each then fld fa4,0(a1).
        for first in ('08b577d3', '00e63027', 'e2070553', 'a2b72553', '00c58533', 'c2051753'):
            with self.subTest(first=first):
                self.check(self.good, first + ' 0005b707')
        self.check(self.good, '08b57753 00000013 0005a707')  # fsub.s fa4, nop, flw fa4
        self.check(self.good, '08b57753 0205e707')  # fsub.s fa4, vle32.v v14 loads no FP register

    def test_compressed_instruction_lengths_decode(self):
        fence, waw = r'fence.i at 0x8000406 in trap.*issue 19', r'writes fa4 at 0x80004(04|06) in trap.*issue 20'
        for words, refusal in (('0001 0000100f', fence),  # c.nop, then a fence.i the core fetches at +2
                               ('08b57753 2598', waw),  # fsub.s fa4, then c.fld fa4
                               ('2598 2598', waw),  # c.fld fa4 twice
                               ('0001 08b57753 0005a707', waw)):  # c.nop, fsub.s fa4, flw fa4 at +2 and +6
            with self.subTest(words=words), self.assertRaisesRegex(Refusal, refusal):
                self.check(self.good, words)

    def test_core_errata_read_real_disassembly(self):
        import shutil
        import subprocess
        compiler = shutil.which('riscv64-unknown-elf-gcc')
        self.assertTrue(compiler and shutil.which('riscv64-unknown-elf-objdump'), 'RISC-V GCC and objdump are required')
        cases = (('rv64gc_zifencei', 'fsub.s fa4,fa0,fa1\nflw fa4,0(a1)', 'issue 20'),
                 ('rv64gc_zifencei', 'fence.i', 'issue 19'),
                 ('rv64gcv', 'vfmv.f.s fa4,v8', 'PR 22'),
                 ('rv64gc', 'fcvt.d.w fa4,a0\nc.fldsp fa4,8(sp)', 'issue 20'),
                 ('rv64gc_zfh', 'fadd.h fa4,fa0,fa1\nflh fa4,0(a1)', 'issue 20'),
                 ('rv64gc_zfa', 'fli.d fa4,1.0\nfld fa4,0(a1)', 'issue 20'),
                 ('rv64gc', 'fsqrt.d fa4,fa0\n.globl next\nnext:\nfld fa4,0(a1)', 'issue 20'),
                 ('rv64imafd', '.word 0x0000100f', 'issue 19'),
                 ('rv64imafd', '.insn i 0x0f, 1, x0, x0, 5', 'issue 19'),
                 ('rv64imafd', 'fsub.s fa4,fa0,fa1\n.word 0x0005a707', 'issue 20'),
                 ('rv64imafd', 'fsub.s fa4,fa0,fa1\n.2byte 0x2598\n.2byte 0', 'issue 20'),
                 # Instructions that straddle data directives: c.nop then fence.i; c.nop, fsub.s fa4, then flw fa4.
                 ('rv64gc', '.word 0x100f0001\n.2byte 0', 'issue 19'),
                 ('rv64gc', '.word 0x77530001\n.word 0xa70708b5\n.2byte 5', 'issue 20'),
                 ('rv64gc', 'c.fsd fa4,0(a1)\nc.fld fa4,8(a1)', None),
                 ('rv64gc', 'fsflags a0\nfld fa4,0(a1)', None),
                 ('rv64imafd', 'fsub.s fa4,fa0,fa1\n.word 0x0205e707', None))
        startup = ('.globl _start, __wrap_main, uart_init, main, trap_vector\n_start:\nj __wrap_main\n'
                   '__wrap_main:\ncall uart_init\ncall main\nj __wrap_main\nuart_init:\nret\n'
                   'main:\n{}\nret\ntrap_vector:\nj trap_vector\n')
        for march, body, issue in cases:
            with self.subTest(march=march, body=body), tempfile.TemporaryDirectory() as directory:
                source, elf = Path(directory) / 'case.S', Path(directory) / 'case.elf'
                source.write_text(startup.format(body))
                subprocess.run([compiler, f'-march={march}', '-mabi=lp64d', '-nostdlib', '-o', str(elf), str(source)],
                               check=True, capture_output=True)
                if issue is None:
                    checked = d._startup_check(elf)
                    self.assertTrue(checked['core_errata_clear'])
                    # The same ELF in another folder is the same code, so its digest is unchanged.
                    moved = Path(directory) / 'other' / elf.name
                    moved.parent.mkdir()
                    shutil.copyfile(elf, moved)
                    self.assertEqual(d._startup_check(moved)['disassembly_sha256'], checked['disassembly_sha256'])
                else:
                    self.assertRaisesRegex(Refusal, r'at 0x[0-9a-f]+ in main.*' + issue, d._startup_check, elf)

    def test_platform_archives_are_byte_reproducible(self):
        cmake, compiler = shutil.which('cmake'), shutil.which('riscv64-unknown-elf-gcc')
        self.assertTrue(cmake and compiler, 'CMake and RISC-V GCC are required')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'lib').mkdir()
            (root / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.20)\nproject(t LANGUAGES ASM C)\n'
                                                 'add_subdirectory(lib)\n')
            (root / 'lib' / 'CMakeLists.txt').write_text('add_library(mixed STATIC a.c b.S)\n'
                                                         'add_library(asm STATIC b.S)\n')
            (root / 'lib' / 'a.c').write_text('int a(void) { return 1; }\n')
            (root / 'lib' / 'b.S').write_text('.globl b\nb: ret\n')
            (root / 'rules.cmake').write_text(d._DETERMINISTIC_ARCHIVES)
            (root / 'none.cmake').write_text('')

            def archives(rules, build):
                subprocess.run([cmake, '-S', root, '-B', root / build, '-DCMAKE_SYSTEM_NAME=Generic',
                                '-DCMAKE_C_COMPILER=' + compiler, '-DCMAKE_ASM_COMPILER=' + compiler,
                                '-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY',
                                '-DCMAKE_PROJECT_INCLUDE=' + str(root / rules)], check=True, capture_output=True)
                subprocess.run([cmake, '--build', root / build], check=True, capture_output=True)
                import time
                time.sleep(1.1)  # archive timestamps have one-second resolution
                return [(root / build / 'lib' / name).read_bytes() for name in ('libmixed.a', 'libasm.a')]

            self.assertEqual(archives('rules.cmake', 'one'), archives('rules.cmake', 'two'))
            # Control: without the rules the same builds differ, so the comparison can fail.
            self.assertNotEqual(archives('none.cmake', 'three'), archives('none.cmake', 'four'))

    def test_unused_platform_variables_refuse(self):
        with tempfile.TemporaryDirectory() as root:
            log = Path(root) / 'configure.log'
            log.write_text('Manually-specified variables were not used by the project:\n PLATFORM')
            with self.assertRaisesRegex(Refusal, 'unused platform'):
                d._configure_check(log)


class ResidentElfRecordTests(unittest.TestCase):
    """The link check of a resident deployment on a real linked RISC-V ELF: arena bytes, image set, and core errata."""
    HOST, WINDOW, RESIDENT = 0x80000000, 0x80010000, 0x80011000
    CLOCK = 100000000

    def setUp(self):
        import hashlib
        import shutil
        self.compiler = shutil.which('riscv64-unknown-elf-gcc')
        self.assertTrue(self.compiler and shutil.which('riscv64-unknown-elf-objdump'),
                        'RISC-V GCC and objdump are required')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sha = lambda data: hashlib.sha256(data).hexdigest()
        self.image = (0x73).to_bytes(4, 'little')  # the counted image linked as a symbol
        self.template = (0x40001057).to_bytes(4, 'little')  # a resident image's template, bound at run time
        self.arena = bytes(range(1, 33)) + bytes(32)

    def link(self, bind_body='nop', arena=None, resident_flags=4):
        """Link the host, one counted image in the window, and the arena; the resident bind runs bind_body."""
        image = 'atlas_image_' + self.sha(self.image)
        arena = self.arena if arena is None else arena
        words = lambda data: ', '.join(f'0x{byte:02x}' for byte in data)
        (self.root / 'case.S').write_text(
            '.globl _start, __wrap_main, uart_init, main, trap_vector, region_000_resident_image_bind\n'
            '.text\n_start:\nj __wrap_main\n__wrap_main:\ncall uart_init\ncall main\nj __wrap_main\n'
            'uart_init:\nret\nmain:\ncall region_000_resident_image_bind\nret\n'
            f'region_000_resident_image_bind:\n{bind_body}\nret\ntrap_vector:\nj trap_vector\n'
            f'.section .rodata\nregion_000_resident_image_template:\n.byte {words(self.template)}\n'
            '.data\n.globl sys_clk_freq\n.type sys_clk_freq, @object\n.size sys_clk_freq, 8\n'
            f'sys_clk_freq:\n.quad {self.CLOCK}\n'
            f'.section .window,"aw",@progbits\n.space 64\n.globl {image}\n.type {image}, @object\n'
            f'.size {image}, 4\n{image}:\n.byte {words(self.image)}\n'
            f'.section .atlas_resident,"a",@progbits\n.byte {words(arena)}\n')
        (self.root / 'chip.ld').write_text(
            f'MEMORY {{ HOST (rwx) : ORIGIN = {self.HOST}, LENGTH = 0x10000\n'
            f'WINDOW (rw) : ORIGIN = {self.WINDOW}, LENGTH = 0x1000\n'
            f'RESIDENT (r) : ORIGIN = {self.RESIDENT}, LENGTH = {len(self.arena)} }}\n'
            f'PHDRS {{ host PT_LOAD; window PT_LOAD; resident PT_LOAD FLAGS({resident_flags}); }}\nENTRY(_start)\n'
            'SECTIONS {\n'
            '.text : { *(.text*) } >HOST :host\n.rodata : { *(.rodata*) } >HOST :host\n'
            '.data : { *(.data*) } >HOST :host\n__heap_start = ALIGN(16);\n__heap_end = __heap_start + 256;\n'
            '__stack_start = __heap_end;\n__stack_end = __stack_start + 1024;\n__sp = __stack_end;\n'
            '.window : { KEEP(*(.window)) } >WINDOW :window\n'
            '.atlas_resident : { KEEP(*(.atlas_resident)) } >RESIDENT :resident\n}\n')
        elf = self.root / 'chip.elf'
        subprocess.run([self.compiler, '-march=rv64gcv', '-mabi=lp64d', '-nostdlib', '-T', 'chip.ld',
                        '-o', elf.name, 'case.S'], cwd=self.root, check=True, capture_output=True)
        return elf

    def deployment(self, resident_images=1):
        loads = [dict(name='parameter_a', address=self.RESIDENT, bytes=32, sha256=self.sha(self.arena[:32]))]
        images = [dict(sha256=self.sha(self.image), upload_convention='counted_ensure_loaded')]
        images += [dict(sha256=self.sha(self.template), upload_convention='resident_template_bind')] * resident_images
        return dict(host_allocation=dict(address=self.HOST, usable_bytes=0x10000),
                    window=dict(address=self.WINDOW, bytes=0x1000), window_required_bytes=64,
                    provider=dict(images=images), parameters=None,
                    resident=dict(address=self.RESIDENT, bytes=len(self.arena), loads=loads))

    def record(self, elf, deployment):
        return d._elf_record(elf, deployment, dict(clock_hz=self.CLOCK, receiver_baud=115200))

    def test_a_resident_elf_links_its_arena_read_only_and_its_counted_images_as_symbols(self):
        record = self.record(self.link(), self.deployment())
        self.assertEqual(record['resident_loads'], 1)
        self.assertEqual(list(record['instruction_images']), [self.sha(self.image)])
        self.assertTrue(record['startup_qualification']['core_errata_clear'])
        self.assertEqual(next(row['flags'] for row in record['loads'] if row['address'] == self.RESIDENT), 4)

    def test_a_changed_resident_byte_or_a_writable_arena_is_refused(self):
        changed = bytearray(self.arena)
        changed[5] ^= 1
        with self.assertRaisesRegex(Refusal, 'linked resident load differs from its packed asset: parameter_a'):
            self.record(self.link(arena=bytes(changed)), self.deployment())
        with self.assertRaisesRegex(Refusal, 'read-only parameters and resident arena'):
            self.record(self.link(resident_flags=6), self.deployment())

    def test_only_resident_images_are_exempt_from_the_linked_image_set(self):
        deployment = self.deployment()
        deployment['provider']['images'].append(dict(sha256=self.sha(b'unlinked'), upload_convention='counted_ensure_loaded'))
        with self.assertRaisesRegex(Refusal, 'linked instruction images differ from the deployment image set'):
            self.record(self.link(), deployment)

    def test_resident_host_code_gets_the_core_errata_scan(self):
        """A resident image is device code in read-only data; the host code that binds it is scanned like all code."""
        for body, issue in (('vfmv.f.s fa4,v8', 'vfmv.f.s at 0x[0-9a-f]+ in region_000_resident_image_bind.*PR 22'),
                            ('fence.i', 'fence.i at 0x[0-9a-f]+ in region_000_resident_image_bind.*issue 19')):
            with self.subTest(body=body), self.assertRaisesRegex(Refusal, issue):
                self.record(self.link(bind_body=body), self.deployment())
        # The template word matches vfmv.f.s, but it is never host code, so the scan does not decode it.
        self.assertEqual(self.template, (0x40001057).to_bytes(4, 'little'))
        self.record(self.link(), self.deployment())


class StepLinkTests(unittest.TestCase):
    """Step packages: zero-input binding, per-step host fit, and which refusals block a step."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_step_binding_uses_zero_inputs_and_the_packaging_clock(self):
        (self.root / 'deployment.json').write_text(json.dumps(dict(schema='merlin.onchip_deployment.v1', parameters=None,
            provider=dict(sys_clk_freq_hz=1), inputs=[dict(path='inputs/input_000.bin', bytes=3)])))
        with patch('merlin.compiler.build.handoff.bind_deployment_payloads', lambda d, p: (d, p)):
            bound = d.step_binding(self.root, dict(clock_hz=100000000))
        platform, payloads = bound['generated']
        self.assertEqual((platform['provider']['sys_clk_freq_hz'], payloads), (100000000, {'inputs/input_000.bin': bytes(3)}))
        self.assertEqual(bound['deployment']['provider']['sys_clk_freq_hz'], 1)

    def test_step_binding_refuses_a_parameter_blob(self):
        (self.root / 'deployment.json').write_text(json.dumps(dict(schema='merlin.onchip_deployment.v1', parameters=dict(path='p'))))
        with self.assertRaisesRegex(Refusal, 'parameter blob'):
            d.step_binding(self.root, {})

    def test_host_fit_is_judged_by_the_step_sizing_link(self):
        deployment = dict(host_allocation=dict(measurement='measured_sections', usable_bytes=65536))
        objects = lambda linked: dict(host_linked=dict(bytes=linked), window_allocated_bytes=8)
        self.assertEqual(d.host_fit(objects(65536), deployment, bytes(8))['host_linked_bytes'], 65536)
        with self.assertRaisesRegex(Refusal, r'exceed backed host capacity \(65537 > 65536 bytes\)'):
            d.host_fit(objects(65537), deployment, bytes(8))
        with self.assertRaisesRegex(Refusal, 'exceed backed host capacity'):
            d.host_fit(dict(window_allocated_bytes=8), deployment, bytes(8))
        with self.assertRaisesRegex(Refusal, 'window bytes differ'):
            d.host_fit(objects(4096), deployment, bytes(4))
        with self.assertRaisesRegex(Refusal, 'measured host sections'):
            d.host_fit(objects(4096), dict(host_allocation=dict(usable_bytes=65536)), bytes(8))

    def test_only_a_host_fit_refusal_blocks_a_step(self):
        out = self.root / 'out'
        (out / 'compiled/application').mkdir(parents=True)
        (out / 'compiled/application/deployment.json').write_text(json.dumps(dict(mode='bench_staged', steps=[
            dict(package='region_000'), dict(package='region_001'), dict(package='region_000')])))
        for package in ('region_000', 'region_001'):
            (out / 'compiled/application/steps' / package).mkdir(parents=True)
            (out / 'compiled/application/steps' / package / 'deployment.json').write_text('{}')
        context = dict(out=out, transport='jtag', settings={})
        deployment = dict(host_allocation={})
        binding = dict(deployment=deployment, generated={'handoff/window.bin': b''})
        fits = {'region_000': Refusal('linked host sections plus stack/heap exceed backed host capacity')}

        def fit(objects, *args):
            if objects['package'] in fits:
                raise fits[objects['package']]
            return {}
        with patch.object(d, 'step_binding', return_value=binding), patch.object(d, 'host_fit', fit), \
                patch.object(d, 'objects', lambda step: dict(package=step['application'].name, hashes={'h': step['application'].name})):
            self.assertTrue(d.staged(context))
            d.step_objects(context)
            self.assertEqual(context['blocked'], {'region_000': 'linked host sections plus stack/heap exceed backed host capacity'})
            self.assertEqual(list(context['steps']), ['region_000', 'region_001'])
        shutil.rmtree(out / 'steps')
        with patch.object(d, 'step_binding', return_value=binding), \
                patch.object(d, 'objects', side_effect=Refusal('platform build failed')):
            with self.assertRaisesRegex(Refusal, 'platform build failed'):
                d.step_objects(context)

    def test_each_step_link_checks_the_gate_over_its_own_objects(self):
        receipt = self.root / 'check.json'
        receipt.write_text(json.dumps(dict(bundle_inventory='SHA256SUMS')))

        class Gate:
            def __init__(self, receipt, hashes): self.receipt, self.hashes = receipt, hashes
            def verify(self): pass
        gate = Gate(receipt, {'a.o': '1', 'b.o': '2', str(receipt): '3', 'SHA256SUMS': '4'})
        steps = {'a': dict(objects=dict(hashes={'a.o': '1'}), bundle='a'), 'b': dict(objects=dict(hashes={'b.o': '2'}), bundle='b'),
                 'c': dict(blocked='code reservation exceeded')}
        seen = {}

        def link(step):
            seen[step['bundle']] = set(step['gate'].hashes)
            step['elf_record'] = {}
            return step['bundle'] + '.elf'
        with patch.object(d, 'link', link):
            self.assertEqual(d.link_steps(dict(out=self.root, settings={}, gate=gate, steps=steps)), {'a': 'a.elf', 'b': 'b.elf'})
        self.assertEqual(seen, {'a': {'a.o', str(receipt), 'SHA256SUMS'}, 'b': {'b.o', str(receipt), 'SHA256SUMS'}})
        self.assertNotIn('elf', steps['c'])


if __name__ == '__main__':
    unittest.main()
