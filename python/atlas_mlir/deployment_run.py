"""Ordered, single-attempt entry point for an authenticated chip kit."""
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from functools import cached_property
import hashlib
import io
import json
from pathlib import Path
import signal
import sys
import tempfile

from .deploy import Refusal, digest, safe_path


def _bound(kit):
    # The kit's capture_run.py bound (run.sh): 180 s startup, the UART deadline, 120 s per RPC command, and cleanup.
    # Temporary copy of capture_run.py's formula, kept because sharing it edits a staged bench/ file and so every kit;
    # replacement: kit_manifest.bound(kit), called from both, at the next round that repackages kits.
    return 300 + kit['settings']['timeout'] + 120 * (8 + len(kit['dep']['hash_ranges']))


def _composed_bound(bundle, manifest):
    # Above the composed runner's own waits, as _bound is for one kit: 300 s for startup and cleanup, then per step its
    # plan timeout and 120 s per Tcl command it can issue (ELF load and verify, load and verify per input and weight row,
    # resume, halt, and one dump per output and hash range), plus the session's reset and final halt.
    # Temporary, like _bound; replacement: kit_manifest.bound(kit) for both kit kinds.
    regions, timeout, commands = manifest['composed']['regions'], manifest['composed'].get('timeout_seconds', 120), 2
    for region in regions:
        rows = json.loads(safe_path(bundle / region['deployment']).read_text())['hash_ranges']
        commands += 4 + 2 * (len(region.get('inputs', [])) + len(region.get('weights', []))) + len(region['outputs']) + len(rows)
    return 300 + len(regions) * timeout + 120 * commands


class _Stop:
    """Stop one live rung at the wall-clock cap, SIGTERM, or SIGINT.

    The first stop raises TransportError inside the runner, whose own bounded
    cleanup then halts the hart and stops OpenOCD and the UART; later signals
    are ignored. So that cleanup always owns and stops OpenOCD, a stop is
    queued until its window ends when it arrives outside the runner; while
    the session is opening and has not yet stored its OpenOCD process (for a
    step ELF rung, also the transport that owns it); while the function that
    owns the session's cleanup runs its own bytecode rather than a call, which
    covers entering cleanup on a normal return or an exception; or while the
    session is closing. After close, nothing is queued or re-armed. SIGKILL
    cannot be cleaned up. `composed` selects the step ELF runner (composed
    plan over the composed Tcl transport) instead of the single-kit runner.
    """

    what = 'runner'

    def __init__(self, bound, composed=False):
        from .chiprunner import composed as plan, composed_tcl, run_kit, tcl_rpc, transport_log
        self.bound, self.reason, self.raised, self.closed = bound, None, False, False
        # Runner frames come from these modules; the innermost one decides whether a stop raises now.
        self.runners = {module.__name__ for module in (plan, composed_tcl, run_kit, tcl_rpc, transport_log)}
        # Opening windows: while a frame of this code runs and its object has not yet stored this attribute.
        # Owners: code whose try body opens the session and whose finally closes it.
        if composed:
            owner = composed_tcl.TclTransport
            self.opening = {owner.__init__.__code__: 'process', plan.Session._open.__code__: 'transport'}
            self.owners = {plan.execute.__code__}
            self.closing = {plan.Session.close.__code__, owner.close.__code__, owner._release.__code__}
        else:
            self.opening, self.owners = {run_kit.Session.open.__code__: 'process'}, {run_kit.execute.__code__}
            self.closing = {run_kit.Session.close.__code__}
        self.install()

    def install(self):
        self.previous = {signal.SIGALRM: signal.signal(signal.SIGALRM, self.handle)}
        if self.bound is not None:
            signal.setitimer(signal.ITIMER_REAL, self.bound)  # armed first, so it never replaces a stop queued below
        for number in (signal.SIGTERM, signal.SIGINT):
            self.previous[number] = signal.signal(number, self.handle)

    def deliverable(self, frame):
        innermost, guarded = None, False
        while frame is not None:
            if innermost is None and frame.f_globals.get('__name__') in self.runners:
                innermost = frame.f_code
            guarded |= frame.f_code in self.closing or (
                frame.f_code in self.opening and getattr(frame.f_locals.get('self'), self.opening[frame.f_code], None) is None)
            frame = frame.f_back
        return innermost is not None and innermost not in self.owners and not guarded

    def handle(self, number, frame):
        from .chiprunner.tcl_rpc import TransportError
        if self.raised or self.closed:
            return
        self.reason = self.reason or ('runner exceeded %d s and was stopped' % self.bound
                                      if number == signal.SIGALRM else f'{self.what} stopped by ' + signal.Signals(number).name)
        if not self.deliverable(frame):
            signal.setitimer(signal.ITIMER_REAL, .05)  # queue: deliver once the window ends
            return
        self.raised = True
        raise TransportError(self.reason)

    def close(self):
        self.closed = True  # a handler pending from here on cannot re-arm the timer
        signal.setitimer(signal.ITIMER_REAL, 0)
        for number, handler in self.previous.items():
            signal.signal(number, handler)


class _Resident(_Stop):
    """Stop a resident session on SIGTERM or SIGINT, from its first call until it has closed.

    The stop raises TransportError wherever the session is, including a wait
    for the next stdin line; the session's close then halts the hart and stops
    OpenOCD within its own bounded waits. As in _Stop, a stop that arrives
    while OpenOCD starts and before its owner stores it, during the close,
    while handlers are installed, or in the run's exit outside its closing
    reference call is queued until that window ends. Each call is bounded by
    the runner's own UART and RPC waits, so no wall-clock cap is armed.

    The run's final records are made from `seal()`, after which no stop is
    accepted; a first signal from then until `close()` is raised again once
    the previous handlers are restored, so it is never lost.
    """
    what = 'resident session'

    def __init__(self):
        from .chiprunner import composed, composed_tcl, run_kit
        self.bound, self.reason, self.raised, self.closed = None, None, False, False
        # Each opening frame, and the attribute that holds OpenOCD once its owner has stored it.
        self.opening = {run_kit.Session.open.__code__: 'process', composed.Session._open.__code__: 'transport'}
        self.guarded = {_Stop.install.__code__, run_kit.Session.close.__code__, composed.Session.close.__code__,
                        composed_tcl.TclTransport.close.__code__, composed_tcl.TclTransport._release.__code__}
        self.calls = {run_kit.Session.call.__code__, composed.Session.call.__code__}
        self.exiting = Run.__exit__.__code__
        self.sealed, self.late = False, None

    def handle(self, number, frame):
        if not self.sealed:
            if number == signal.SIGALRM and self.reason is None:
                self.reason = f'{self.what} stopped by SIGALRM'  # no cap is armed, so any first SIGALRM is external
            return super().handle(number, frame)
        if self.reason is None and number != signal.SIGALRM:
            self.late = self.late or number

    def seal(self):
        """Accept no further stop; return the final reason for the records."""
        self.sealed = True
        signal.setitimer(signal.ITIMER_REAL, 0)
        return self.reason

    def close(self):
        super().close()
        if self.late:
            signal.raise_signal(self.late)  # as if it arrived just after the handlers were restored

    def deliverable(self, frame):
        calling = False
        while frame is not None:
            code = frame.f_code
            if code in self.guarded or code is self.exiting and not calling or (
                    code in self.opening and getattr(frame.f_locals['self'], self.opening[code]) is None):
                return False
            calling |= code in self.calls
            frame = frame.f_back
        return True


def _rung(bundle, logs, *, negative=False, serial=()):
    from .chiprunner import run_kit, check_rung
    from .chiprunner.kit_manifest import load
    from .chiprunner.tcl_rpc import TransportError
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = logs / (('negative-' if negative else 'positive-') + stamp)
    arguments = [str(bundle), '--logs', str(directory), '--allow-transport', *serial]
    if negative:
        arguments += ['--elf', 'negative/chip.elf']
    terminal = logs / (directory.name + '-terminal.txt')
    bound = _bound(load(bundle, 'negative/chip.elf' if negative else 'chip.elf'))
    # Like run.sh: stdout and stderr stream into the terminal file, which ends with rc before the handlers are restored.
    rc, stop = 2, _Stop(bound)
    try:
        with terminal.open('x') as stream:
            try:
                with redirect_stdout(stream), redirect_stderr(stream):
                    rc = run_kit.main(arguments)
            except TransportError as error:
                stream.write(f'KIT TRANSPORT_ERROR {error}\n')
            finally:
                if stop.reason:
                    rc = 2  # like run.sh, any stop is a transport error, even after a numerical failure
                    if not stop.raised:
                        stream.write(f'KIT TRANSPORT_ERROR {stop.reason} after the runner returned\n')
                stream.write(f'rc={rc}\n')
                stream.flush()
    finally:
        stop.close()
    expected = 1 if negative else 0
    if rc != expected:
        raise Refusal(f'{directory.name}: runner returned {rc}, expected {expected}')
    result = check_rung.check(terminal, logs=directory, bundle=bundle,
                              negative=negative, run_stamp=stamp)
    (logs / (directory.name + '-checker.json')).write_text(json.dumps(result.checks, indent=2) + '\n')
    if result.verdict != 'PASS':
        raise Refusal(f'{directory.name}: checker {result.verdict}')


def _bench(bundle, serial=()):
    from .chiprunner import run_kit
    return run_kit.arguments([str(bundle), *serial])


def _composed_rung(bundle, logs, *, negative=False, serial=()):
    """One live rung of linked step ELFs: the plan, or its declared negative, which must fail only numerically.

    Like `_rung`: the cap and signal stops, and a terminal file (stdout and stderr, the KIT line) that ends with rc.
    """
    from .chiprunner import composed
    from .chiprunner.run_kit import NumericalError
    from .chiprunner.tcl_rpc import TransportError
    name = 'negative' if negative else 'positive'
    terminal = logs / (name + '-terminal.txt')
    bound = _composed_bound(bundle, composed.load_plan(bundle)[0])
    rc, failure, line, stop = 2, None, 'KIT TRANSPORT_ERROR runner raised an unexpected error', _Stop(bound, composed=True)
    try:
        with terminal.open('x') as stream:
            try:
                with redirect_stdout(stream), redirect_stderr(stream):
                    composed.execute(bundle, logs / name, backend='jtag', mutation=negative, bench=_bench(bundle, serial))
                rc, line = 0, 'KIT PASS every step boundary matches its reference'
            except composed.BoundaryError as error:
                rc, failure, line = 1, error, f'KIT FAIL {error}'
            except (TransportError, NumericalError, OSError, ValueError, KeyError) as error:
                # A step's nonzero UART status is not a boundary mismatch, and a cleanup error is a transport error
                # even after one, which it names.
                after = f' (after {error.__context__})' if isinstance(error.__context__, composed.BoundaryError) else ''
                rc, failure, line = 2, error, f'KIT TRANSPORT_ERROR {error}{after}'
            finally:
                stream.write(line + '\n')
                if stop.reason:
                    rc = 2  # any stop is a transport error, even after a numerical failure
                    if not stop.raised:
                        stream.write(f'KIT TRANSPORT_ERROR {stop.reason} after the runner returned\n')
                stream.write(f'rc={rc}\n')
                stream.flush()
    finally:
        stop.close()
    if rc == (1 if negative else 0):
        return
    if rc == 0:
        raise Refusal('negative: completed without a numerical boundary mismatch')
    raise Refusal(f'{name}: {failure if failure is not None else stop.reason}') from failure


def staged(manifest):
    """A staged plan: composed, and its manifest declares no tensor inputs (bind records only the untyped outputs)."""
    return 'composed' in manifest and 'inputs' not in manifest.get('deployment', {})


def staged_view(manifest, bound):
    """The plan a resident step session of a staged kit executes, with the deployment rows `atlascc run --inputs` reads.

    Bind declares a staged plan's application input as a file-backed value, which one step reads whole, and declares
    its outputs without types. The kit already inventories each step's deployment record, which types that step's
    input and output ranges. In the view, the step that reads each file-backed value reads that value's file directly
    (the same bytes, since no step writes a file-backed value), so the session's existing per-call input override
    replaces it; each such range becomes deployment input NNN in plan value order, typed by its step's record, and each
    output takes its producer step's type. The bench runner modules are unchanged, so existing kits keep their staged
    runner contract. A value read by more than one step, or not whole, is refused.
    """
    import copy
    from .chiprunner import composed
    view = copy.deepcopy(manifest)
    plan, records = view['composed'], {}
    def typed(region, role, row):
        name = region.get('deployment')
        if name is None:
            return None
        if name not in records:
            records[name] = json.loads(bound(name).read_text())
        found = [x for x in records[name].get(role, []) if (x.get('id'), x.get('address'), x.get('bytes')) == (row['id'], row['address'], row['bytes'])]
        return found[0].get('type') if len(found) == 1 else None
    inputs, windows = [], []
    for value in plan.get('values', []):
        if 'file' not in value:
            continue
        readers = [(region, index) for region in plan['regions'] for index, row in enumerate(region.get('inputs', []))
                   if 'source' in row and row['source'].get('value') == value['id']]
        if len(readers) != 1:
            raise ValueError(f'staged input {value["id"]} is read by {len(readers)} steps; --inputs needs exactly one reader')
        region, index = readers[0]
        row = region['inputs'][index]
        if row['bytes'] != value['bytes'] or composed.chunks(row['source'].get('range'), row['bytes'], value['bytes']) != [(0, value['bytes'])]:
            raise ValueError(f'staged input {value["id"]} is not read whole by step {region["id"]}')
        kind = typed(region, 'inputs', row)
        if kind is None:
            raise ValueError(f'staged input {value["id"]} has no type in the deployment record of step {region["id"]}')
        region['inputs'][index] = dict(id=row['id'], address=row['address'], bytes=row['bytes'], file=value['file'])
        inputs.append(dict(index=len(inputs), id=value['id'], address=row['address'], bytes=row['bytes'], type=kind))
        windows.append(region['window'])
    if any(window != windows[0] for window in windows):
        raise ValueError('staged inputs are read in different step windows')
    regions = {region['id']: region for region in plan['regions']}
    outputs = []
    for row in view.get('deployment', {}).get('outputs', []):
        if 'type' not in row:
            produced = [x for x in regions[row['source']['region']]['outputs'] if x['id'] == row['source']['output']]
            kind = typed(regions[row['source']['region']], 'outputs', produced[0]) if len(produced) == 1 else None
            if kind is not None:
                row = dict(row, type=kind)
        outputs.append(row)
    view['deployment'] = dict(view.get('deployment', {}), inputs=inputs, outputs=outputs)
    if windows:
        view['deployment']['window'] = windows[0]
    return view


@contextmanager
def _viewed(bundle):
    """While a resident session of a staged kit is made or called, its plan is the staged view (see staged_view)."""
    from .chiprunner import composed
    original, bundle = composed.load_plan, safe_path(bundle)
    def load_plan(root):
        manifest, members, bound = original(root)
        return (staged_view(manifest, bound) if safe_path(root) == bundle else manifest), members, bound
    composed.load_plan = load_plan
    try:
        yield
    finally:
        composed.load_plan = original


def kit_tensors(bundle):
    """The deployment's typed tensor rows, its packaged input bytes in row order, and the frozen output references (None
    if a plan output has none), for a single-ELF kit, a composed plan, or a staged plan (rows from its staged view)."""
    from .chiprunner import composed, step_inputs
    manifest = json.loads(safe_path(bundle / 'manifest.json').read_text())
    if 'composed' not in manifest:
        from .chiprunner.kit_manifest import load
        kit = load(bundle)
        return (kit['dep'], [item['data'] for item in step_inputs.window_inputs(kit['dep'], kit['window'].read_bytes())],
                [item['data'] for item in kit['references']])
    manifest, _, bound = composed.load_plan(bundle)
    if staged(manifest):
        manifest = staged_view(manifest, bound)
    deployment = manifest.get('deployment', {})
    plan = manifest['composed']
    rows = {(row['address'], row['bytes']): row for region in plan['regions']
            for row in region.get('inputs', []) if 'file' in row}
    inputs = [composed.gather(bound(rows[row['address'], row['bytes']]['file']).read_bytes(),
                              rows[row['address'], row['bytes']].get('range'), row['bytes'])
              for row in deployment.get('inputs', []) if (row['address'], row['bytes']) in rows]
    regions = {region['id']: region for region in plan['regions']}
    sources = [next(x for x in regions[row['source']['region']]['outputs'] if x['id'] == row['source']['output'])
               for row in deployment.get('outputs', [])]
    if any('reference' not in row for row in sources):
        return deployment, inputs, None
    return deployment, inputs, [bound(row['reference']).read_bytes() for row in sources]


def _packaged_record(bundle):
    """The evaluator record of the kit's own reference: packaged in the kit, or for a staged plan the deployment's
    reference/reference.json beside the kit, whose digest the inventoried manifest names."""
    packaged = safe_path(bundle / 'reference' / 'reference.json')
    if packaged.is_file():
        return json.loads(packaged.read_text())
    inventory = dict(reversed(line.split('  ', 1)) for line in safe_path(bundle / 'SHA256SUMS').read_text().splitlines())
    if inventory.get('manifest.json') != digest(bundle / 'manifest.json'):
        raise Refusal('kit manifest differs from its inventory')
    pinned = json.loads(safe_path(bundle / 'manifest.json').read_text()).get('staged', {}).get('reference_sha256')
    record = safe_path(bundle.parent / 'reference' / 'reference.json')
    if pinned is None or not record.is_file():
        raise Refusal('kit packages no evaluator reference to authenticate a supplied reference against')
    data = record.read_bytes()
    if hashlib.sha256(data).hexdigest() != pinned:
        raise Refusal('deployment reference differs from the one the staged kit names')
    return json.loads(data)


def _relocated(record):
    """The record with its compiler decision inputs named relative to the directory that holds them all.

    The evaluator names each decision input by its absolute path under the OUT it read, so a copied OUT (the bench's)
    records other paths for the same digests; only the digests and their places inside OUT are compared.
    """
    import os
    decisions = record.get('compiler_decision_inputs')
    identities = decisions.get('identities') if isinstance(decisions, dict) else None
    if not isinstance(identities, dict) or not identities or not all(os.path.isabs(path) for path in identities):
        return record
    root = os.path.commonpath(list(identities))
    if len(identities) == 1:
        root = os.path.dirname(root)
    return dict(record, compiler_decision_inputs=dict(decisions, identities={
        os.path.relpath(path, root): value for path, value in identities.items()}))


def _supplied_reference(bundle, deployment, staged, directory):
    """Expected output bytes from an evaluator reference for exactly these inputs, made as the kit's own reference was."""
    import numpy as np
    packaged = _packaged_record(bundle)
    root = safe_path(directory)
    record = json.loads(safe_path(root / 'reference.json').read_text())
    # Only the inputs and what they determine may differ from the packaged reference's record.
    per_inputs = {'inputs', 'outputs', 'expected_sha256', 'evaluator_members'}
    same = lambda item: {key: value for key, value in _relocated(item).items() if key not in per_inputs}
    if record.get('status') != 'EXPECTED' or not isinstance(record.get('inputs'), dict):
        raise Refusal(f'{directory}: not an EXPECTED evaluator reference for supplied inputs')
    if same(record) != same(packaged):
        raise Refusal(f'{directory}: not made from this kit\'s source, decisions, target rules, and evaluator')
    if record['inputs'] != {'input_%03d' % item['row']['index']: item['sha256'] for item in staged}:
        raise Refusal(f'{directory}: made from other inputs')
    data = safe_path(root / 'expected.npz').read_bytes()
    if hashlib.sha256(data).hexdigest() != record['expected_sha256']:
        raise Refusal(f'{directory}: expected outputs differ from its record')
    rows = deployment['outputs']
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if set(archive.files) != {f'output_{row["index"]}' for row in rows}:
            raise Refusal(f'{directory}: outputs differ from the deployment outputs')
        expected = [archive[f'output_{row["index"]}'].tobytes(order='C') for row in rows]
    if [len(item) for item in expected] != [row['bytes'] for row in rows]:
        raise Refusal(f'{directory}: output sizes differ from the deployment outputs')
    return expected


def _save_outputs(logs, rows, outputs):
    """Typed copies of a step's output bytes as output_NNN arrays, next to the raw output-<n>.bin dumps."""
    import numpy as np
    from .chiprunner.step_inputs import DTYPES
    np.savez(logs / 'outputs.npz', **{'output_%03d' % row['index']: np.frombuffer(data, np.dtype(DTYPES[row['type']['dtype']]).newbyteorder('<'))
                                      .reshape(row['type']['shape']) for row, data in zip(rows, outputs)})


class Run:
    """Qualify a packaged deployment once, then serve resident calls.

    Entering qualifies: a dry run preflights both rungs; a live run preflights
    and executes exactly one selected rung (positive, or the expected-fail
    negative), and the next rung needs a new invocation.
    Simulation runs the composed plan and its negative. Only an explicit `step`
    opens a resident session, which checks a reference call with the packaged
    inputs; exit repeats that check if a step was unchecked. For a composed plan
    with `steps`, that reference call is the positive (the rung is the same call
    on a fresh session), so entering runs nothing and exit refuses if no step
    ran. A live session stops on SIGTERM or SIGINT from its first call until its final records are made; its terminal record,
    session-terminal.txt, lists each step and always ends with rc=, and for
    a single kit it and session/summary.json carry the same final verdict. Each step
    re-verifies the bundle against the inventory digest pinned at construction. A step is checked
    against the packaged reference when its inputs equal the packaged ones,
    against a supplied evaluator reference made from exactly its inputs, or
    else not at all.
    """

    def __init__(self, out, *, dry_run=False, simulate=False, negative=False, serial=None, steps=False):
        from .deployment_kit import owned_tools, _verify_tools
        out = safe_path(out)
        status = json.loads(safe_path(out / 'deploy-status.json').read_text())
        if status.get('status') not in ('PACKAGED', 'PACKAGED_BLOCKED'):
            raise Refusal('deployment incomplete; bound EXACT, ELF, and packaging required')
        if status.get('transport') != 'jtag':
            raise Refusal(f'deployment declares transport {status.get("transport")!r}; atlascc run knows jtag and spi')
        if dry_run and simulate:
            raise Refusal('choose dry or simulated execution')
        if negative and (dry_run or simulate):
            raise Refusal('--negative selects one live rung')
        if status['status'] == 'PACKAGED_BLOCKED' and not (dry_run or simulate):
            raise Refusal('blocked steps have no linked ELF, so the kit only simulates: ' + ', '.join(status.get('blocked', ())))
        self.negative, self.simulate = negative, simulate
        self.serial = () if serial is None else ('--serial', serial)  # otherwise the runner's default UART
        self.bundle = safe_path(out / 'kit')
        self.dry_run = dry_run
        self.session, self.stop, self.terminal, self.steps, self.unchecked = None, None, None, 0, 0
        _verify_tools({name: safe_path(self.bundle / 'bench' / name) for name in owned_tools()})
        self.pin = digest(self.bundle / 'SHA256SUMS')
        self.manifest = json.loads(safe_path(self.bundle / 'manifest.json').read_text())
        self.composed = 'composed' in self.manifest
        self.in_session = self.composed and steps and not dry_run  # the resident reference call qualifies
        self.view = (lambda: _viewed(self.bundle)) if staged(self.manifest) else nullcontext
        if simulate and not self.composed:
            raise Refusal('simulation requires an authenticated composed plan; no transport opened')
        if self.composed and not (simulate or dry_run) and any(
                region['kind'] != 'elf' for region in self.manifest['composed']['regions']):
            raise Refusal('live composed execution needs a linked step ELF for every region')
        self.logs = Path(tempfile.mkdtemp(prefix='run-', dir=out))

    def __enter__(self):
        from .chiprunner import composed, run_kit, step_inputs
        if self.composed:
            composed.load_plan(self.bundle)
            if self.in_session:
                return self
            if self.simulate:
                step_inputs.qualify(self.bundle, self.logs / 'qualification')
            elif not self.dry_run:
                _composed_rung(self.bundle, self.logs, negative=self.negative, serial=self.serial)
            return self
        for negative in (False, True) if self.dry_run else (self.negative,):
            args = [str(self.bundle), '--dry-run'] + (['--elf', 'negative/chip.elf'] if negative else [])
            capture = io.StringIO()
            with redirect_stdout(capture):
                rc = run_kit.main(args)
            (self.logs / ('negative-preflight.log' if negative else 'preflight.log')).write_text(capture.getvalue())
            if rc:
                raise Refusal('fresh preflight failed')
            if not self.dry_run:
                _rung(self.bundle, self.logs, negative=negative, serial=self.serial)
        return self

    @cached_property
    def tensors(self):
        return kit_tensors(self.bundle)

    @property
    def deployment(self):
        return self.tensors[0]

    def validate(self, inputs, reference=None):
        """Staged inputs plus the expected outputs and their source, or (None, None) for an unchecked step."""
        from .chiprunner.step_inputs import DTYPES, load_inputs
        staged = load_inputs(self.deployment, inputs)
        rows = self.deployment.get('outputs')
        if not rows or any(row.get('type', {}).get('dtype') not in DTYPES for row in rows):
            raise Refusal('deployment declares no outputs, or an output without a supported dtype')
        if reference is not None:
            return staged, _supplied_reference(self.bundle, self.deployment, staged, reference), str(reference)
        packaged, expected = self.packaged
        if [item['data'] for item in staged] == packaged and expected is not None:
            return staged, expected, 'packaged'
        return staged, None, None

    @property
    def packaged(self):
        """The packaged input bytes and the frozen output references (None if a plan output has none)."""
        return self.tensors[1:]

    def _open(self):
        from .chiprunner import composed, run_kit, step_inputs
        if self.composed:
            live = {} if self.simulate else dict(backend='jtag', bench=_bench(self.bundle, self.serial))
            (self.logs / 'session').mkdir()  # the transport's work directory, as run_kit.live makes the single-kit one
            with self.view():
                self.session = composed.Session(self.bundle, self.logs / 'session', **live)
                self.session.call(self.logs / 'reference-open')
            return
        a = run_kit.arguments([str(self.bundle), '--allow-transport', *self.serial, '--logs', str(self.logs / 'session')])
        kit = run_kit.prepare(a)
        self.session = run_kit.Session(a, kit, run_kit.live(a, kit))
        self.session.open()
        self.reference = step_inputs.window_inputs(kit['dep'], self.session.window)
        self.session.call(self.logs / 'reference-open', self.reference, kit['references'])

    def step(self, inputs, reference=None):
        """One call with new tensors (an .npz of input_NNN arrays), optionally checked against a supplied reference.

        Returns its log directory, the output bytes per deployment output, and
        the reference it matched ('packaged', the supplied directory, or None
        for an unchecked step). A mismatch refuses.
        """
        if self.dry_run:
            raise Refusal('dry run executes no steps')
        if digest(self.bundle / 'SHA256SUMS') != self.pin:
            raise Refusal('bundle changed during session')
        if not self.composed:
            from .chiprunner.kit_manifest import load
            load(self.bundle)  # a composed call verifies its plan's inventory itself
        staged, expected, source = self.validate(inputs, reference)
        if self.session is None:
            if not self.simulate:
                self.terminal = (self.logs / 'session-terminal.txt').open('x')
                self.stop = _Resident()
                self.stop.install()
            self._open()
        logs = self.logs / ('step-%03d' % self.steps)
        self.steps += 1
        self.unchecked += source is None
        if self.composed:
            with self.view():
                self.session.call(logs, inputs=inputs)
        else:
            self.session.call(logs, staged)
        rows = self.deployment['outputs']
        outputs = [(logs / f'output-{row["index"]}.bin').read_bytes() for row in rows]
        _save_outputs(logs, rows, outputs)
        for row, data, want in zip(rows, outputs, expected or ()):
            if data != want:
                raise Refusal(f'{logs.name}: output {row["id"]} differs from the {source} reference')
        return logs, outputs, source

    def report(self, line):
        """Print one step's result line, and record it in a live session's terminal record."""
        print(line, flush=True)
        if self.terminal is not None:
            self.terminal.write(line + '\n'); self.terminal.flush()

    def __exit__(self, kind, error, traceback):
        from .chiprunner.tcl_rpc import TransportError
        session, self.session, stop = self.session, None, self.stop
        if session is None and error is None and self.in_session:
            raise Refusal('no step ran, so the resident session never qualified the kit')
        summary = session.transport.logs / 'summary.json' if session is not None and not self.composed else None
        try:
            # Steps checked against a reference need no repeat of the packaged check after them.
            if session is not None and error is None and self.unchecked and self.composed:
                with self.view():
                    session.call(self.logs / 'reference-close')
            elif session is not None and error is None and self.unchecked:
                session.call(self.logs / 'reference-close', self.reference, session.kit['references'])
        except Exception as failure:
            error = failure
            raise
        finally:
            failure = error or (stop and stop.reason and TransportError(stop.reason)) or None
            try:
                if session is not None and self.composed:
                    session.close()
                elif session is not None:
                    session.close(_verdict(failure), failure and str(failure))
            except BaseException as closing:
                error = error or closing
                raise
            finally:
                if stop is not None:
                    reason = self._finish(stop, error, summary)
        if stop is not None and reason and error is None:
            raise TransportError(reason)  # a stop queued until the session closed still fails the run

    def _finish(self, stop, error, summary):
        """Seal the stop, make the summary and the terminal record from its final state, then restore the handlers.

        The session's close wrote the summary before a stop it queued was
        known, so a differing verdict is rewritten here. Like run.sh, any
        accepted stop is rc=2, and the terminal record ends with rc= even if
        the close raised.
        """
        from .chiprunner.tcl_rpc import TransportError
        try:
            reason = stop.seal()
            failure = error or (reason and TransportError(reason)) or None
            verdict = _verdict(failure)
            message = ' '.join(([str(failure)] if failure else []) +
                               ([f'(then {reason})'] if reason and reason not in str(failure) else [])) or None
            try:
                if summary is not None and summary.exists():
                    record = json.loads(summary.read_text())
                    if (record['verdict'], record.get('error')) != (verdict, message):
                        record.update(verdict=verdict, error=message)
                        summary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
            finally:
                self.terminal.write(' '.join(['SESSION', verdict] + ([message] if message else [])) +
                                    f'\nrc={2 if reason else dict(PASS=0, FAIL=1, TRANSPORT_ERROR=2)[verdict]}\n')
                self.terminal.close()
        finally:
            stop.close()
        return reason


def _verdict(failure):
    from .chiprunner.tcl_rpc import TransportError
    return 'PASS' if failure is None else 'TRANSPORT_ERROR' if isinstance(failure, (TransportError, OSError)) else 'FAIL'


def _steps(inputs, references):
    """(inputs, reference) pairs: the listed files, or with `-` one `INPUTS [REFERENCE]` line per step from stdin."""
    if list(inputs) == ['-']:
        for line in sys.stdin:
            fields = line.split()
            if not 1 <= len(fields) <= 2:
                raise Refusal('each stdin line is INPUTS [REFERENCE]')
            yield fields[0], fields[1] if len(fields) == 2 else None
        return
    yield from zip(inputs, references or [None] * len(inputs))


def _board_ddr(out, *, dry_run=False, simulate=False, negative=False, serial=None, inputs=(), references=()):
    """A board DDR deployment: chip.elf, then its expected-fail negative, loaded into DDR and run (chiprunner/ddr.py).

    The deployment's declared transport chooses the link. Simulation runs both on the simulated board, whose DDR
    and load journal persist in OUT across runs; a dry run only checks the kit.
    """
    from .chiprunner import ddr
    from .chiprunner.run_kit import NumericalError
    if not (dry_run or simulate):
        raise Refusal('no SPI loader for this board yet; atlascc run OUT --simulate runs the deployment '
                      'on the simulated board')
    if negative or serial or inputs or references:
        raise Refusal('a board DDR deployment runs its packaged inputs, positive then negative')
    bundle = safe_path(out / 'kit')
    logs = Path(tempfile.mkdtemp(prefix='run-', dir=out))
    verdict = 'FAIL'
    try:
        if dry_run:
            for elf in ('chip.elf', 'negative/chip.elf'):
                ddr.plan(bundle, elf)
            verdict = 'DRY PASS'
            return logs
        from .chiprunner.ddr_sim import SimulatedBoard
        from .chiprunner.tcl_rpc import TransportError
        target = ddr.plan(bundle)
        board = SimulatedBoard(out / 'simulated-board', bundle, target['board'], target['ddr'])

        def stop(number, frame):  # the finally below then stops the simulator and releases the board
            raise TransportError('run stopped by ' + signal.Signals(number).name)
        previous = {number: signal.signal(number, stop) for number in (signal.SIGTERM, signal.SIGINT)}
        try:
            for negative in (False, True):
                name = 'negative' if negative else 'positive'
                try:
                    record = ddr.execute(bundle, board, logs / name, negative=negative)
                except NumericalError:
                    record = (logs / name / 'run.json').is_file() and json.loads((logs / name / 'run.json').read_text())
                    if not negative or not record or record['verdict'] != 'MISMATCH':
                        raise  # only the negative's packaged output mismatch is its expected failure
                else:
                    if negative:
                        raise Refusal('negative: completed without an output mismatch')
                print(f'{name.upper()} {record["verdict"]} ({record["board"]["link"]} board) load '
                      + json.dumps(record['load'], sort_keys=True))
        finally:
            board.close()
            for number, handler in previous.items():
                signal.signal(number, handler)
        verdict = 'SIM PASS'
        return logs
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        (logs / 'failure.txt').write_text(str(error) + '\n')
        raise Refusal(str(error)) from error
    finally:
        print(f'RUN {verdict} logs={logs}')


def execute(out, *, dry_run=False, simulate=False, negative=False, serial=None, inputs=(), references=()):
    status = json.loads(safe_path(Path(out) / 'deploy-status.json').read_text())
    if status.get('status') == 'PACKAGED' and status.get('transport') == 'spi':
        return _board_ddr(safe_path(out), dry_run=dry_run, simulate=simulate, negative=negative, serial=serial,
                          inputs=inputs, references=references)
    if inputs and negative:
        raise Refusal('--negative runs one rung without steps')
    if references and list(inputs) == ['-']:
        raise Refusal('with --inputs -, name each reference on its stdin line')
    if references and len(references) != len(inputs):
        raise Refusal('give one --reference per --inputs file')
    run = Run(out, dry_run=dry_run, simulate=simulate, negative=negative, serial=serial, steps=bool(inputs))
    verdict = 'FAIL'
    try:
        steps = _steps(inputs, references)
        if list(inputs) != ['-'] and not dry_run:
            # Every listed step and reference is validated before any transport opens.
            steps = list(steps)
            for item, reference in steps:
                run.validate(item, reference)
        with run:
            for item, reference in steps:
                if dry_run:
                    run.validate(item, reference)
                    continue
                logs, _, source = run.step(item, reference)
                run.report(f'STEP {run.steps - 1:03d} ' + (f'EXACT ({source} reference)' if source else 'UNCHECKED (no independent reference)')
                           + f' outputs={logs / "outputs.npz"}')
        verdict = ('DRY PASS' if dry_run else 'SIM PASS' if simulate else
                   'PASS (negative rejected as expected)' if negative else 'PASS')
        if run.unchecked:
            verdict += f' ({run.unchecked} of {run.steps} steps unchecked: no independent reference)'
        return run.logs
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        (run.logs / 'failure.txt').write_text(str(error) + '\n')
        raise Refusal(str(error)) from error
    finally:
        print(f'RUN {verdict} logs={run.logs}')
