"""Export-to-ELF stages for atlascc deploy/run. Unavailable stages refuse explicitly.

No environment switch enables stubs or transports. Tests inject adapters through
Python only; the installed command always uses the fail-closed adapters below.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .chiprunner.kit_manifest import system_path


class Refusal(ValueError):
    pass


def safe_path(value):
    # Reject links without following their targets, including parent links.
    if '..' in Path(value).parts:
        raise Refusal('parent path components are not accepted')
    path = system_path(Path(os.path.abspath(value)))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise Refusal('symlink paths are not accepted')
    return path


def board_settings(placement):
    """Settings from the declared carriers of this placement: their chip clock, and the SoC's UART0 baud."""
    from atlas_mlir_native import soc

    baud = soc.SOC.resource('penguin.uart0').baud
    try:
        hz, carriers = soc.carrier_clock(placement)
    except ValueError as error:
        raise Refusal(str(error)) from None
    if not baud <= hz <= 0xffffffff or hz % 1000 != 0:
        raise Refusal(f'declared clock {hz} Hz must be integral kHz from {-(-baud // 1000)} to 4294967 kHz')
    return dict(clock_hz=hz, mtime_hz=hz // 1000, uart_divisor=hz // baud - 1, receiver_baud=baud,
                carriers=list(carriers),
                clock_basis=f'declared by {soc.IDENTITY} carriers {", ".join(carriers)}, unmeasured')


def digest(path):
    return hashlib.sha256(safe_path(path).read_bytes()).hexdigest()


def check_hashes(hashes):
    if not hashes:
        raise Refusal('missing bound artifact hashes')
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise Refusal(f'hash drift: {path}')


@dataclass(frozen=True)
class Gate:
    # Internal adapter result, not a new public receipt format. Qualification of
    # the generic adapter is required before constructing this from real executions.
    receipt: Path
    hashes: dict

    def verify(self):
        check_hashes(self.hashes)
        if str(self.receipt) not in self.hashes:
            raise Refusal('EXACT receipt is not authenticated')
        record = json.loads(self.receipt.read_text())
        if (record.get('schema') != 'atlas.model_check.v1' or record.get('status') != 'PASS'
                or record.get('positive', {}).get('result') != 'EXACT'
                or record.get('negative', {}).get('result') != 'MISMATCH'
                or record.get('negative', {}).get('first_mismatch') is None
                or record.get('byte_flip', {}).get('changed_byte_count') != 1
                or record.get('same_application_library') is not True
                or record.get('same_instruction_images') is not True
                or record.get('independent_expected_unchanged') is not True):
            raise Refusal('bound positive EXACT and executed output-changing mutation required')


class Stages:
    """Replace each refusal only when its deployment adapter qualifies."""
    def __init__(self, *, jtag=False):
        self.jtag = jtag

    def admit(self, export, out):
        from .export_intake import prepare
        from .toolchain import ToolFailure, ToolchainError
        try:
            return prepare(export, out, board_memory=not self.jtag)
        except (ValueError, ToolFailure, ToolchainError) as error:
            raise Refusal(str(error)) from error

    def build(self, arguments):
        # This is the sole compilation route, including when tests stub it.
        subprocess.run([sys.executable, '-m', 'atlas_mlir.cli', 'build', *map(str, arguments)], check=True)

    def measure(self, context, arguments):
        from .deployment_link import measure_host, measure_onchip, staged
        if self.jtag:
            return measure_onchip(context, arguments)
        if staged(context):
            # A staged plan keeps every tensor on the bench host, which a board DDR program does not have.
            raise Refusal('a bench-staged plan has no board DDR deployment')
        return measure_host(context, arguments)

    def reference(self, context):
        from .deployment_stages import reference
        reference(context)

    # A staged plan takes the same stages per step package: objects and host fit, a composed
    # plan, its composed EXACT and negative, then step links and one staged kit.
    def objects(self, context):
        from .deployment_link import objects, staged, step_objects
        if 'objects' in context:
            # The on-chip host measurement already compiled and sized these objects.
            return context['objects']
        return (step_objects if staged(context) else objects)(context)

    def bind(self, context):
        from .deployment_gate import bind, bind_staged
        from .deployment_link import staged
        return (bind_staged if staged(context) else bind)(context)

    def exact(self, context):
        from .deployment_gate import exact, exact_staged
        from .deployment_link import staged
        return (exact_staged if staged(context) else exact)(context)

    def link(self, context):
        from .deployment_link import link, link_steps, staged
        return (link_steps if staged(context) else link)(context)

    def package(self, context):
        from .deployment_kit import package, package_staged
        from .deployment_link import staged
        return (package_staged if staged(context) else package)(context)


def deploy(export, *, out=None, jtag=False, stages=None):
    settings = board_settings('onchip' if jtag else 'ddr')
    export = safe_path(export)
    if not export.is_dir():
        raise Refusal('EXPORT must be an export directory')
    out = safe_path(out or (export.name + '-deployment'))
    if out.exists() or out.is_symlink():
        raise Refusal(f'output must be fresh: {out}')
    if export.is_relative_to(out):
        raise Refusal('inputs must be outside output')
    out.mkdir(parents=True)
    stages = stages or Stages(jtag=jtag)
    context = dict(export=export, out=out, settings=settings,
                   transport='jtag' if jtag else 'spi', board='sophialake')
    status = dict(status='INCOMPLETE', completed=[], settings=settings, transport=context['transport'])
    def step(name, action):
        result = action()
        status['completed'].append(name)
        return result
    try:
        # Admission authenticates the capture; build generates its policy.
        arguments = step('admit', lambda: stages.admit(export, out))
        # Admission builds the selected host-math archive; the link selects it by declared hash.
        archive = out / 'host-math' / 'rv64' / 'libmerlin-host-math.a'
        context['host_archives'] = [archive] if archive.is_file() else []
        step('atlascc build', lambda: stages.build(arguments))
        # Link the compiled host once. A board DDR host goes first at the DDR base, so the bound compile starts the
        # device window right after it; an on-chip host larger than the declared host memory compiles again as
        # bench-staged steps. An on-chip host that fits needs no second compile.
        bound = step('measure', lambda: stages.measure(context, arguments))
        if bound is not None:
            arguments = bound
            step('atlascc build', lambda: stages.build(arguments))
        step('reference', lambda: stages.reference(context))
        step('objects', lambda: stages.objects(context))
        step('bind', lambda: stages.bind(context))
        gate = step('exact', lambda: stages.exact(context))
        if not isinstance(gate, Gate):
            raise Refusal('missing bound EXACT evidence')
        gate.verify()
        context['gate'] = gate
        # Recheck directly at the link boundary. The link adapter must check during linking too.
        gate.verify()
        step('link', lambda: stages.link(context))
        gate.verify()
        step('package', lambda: stages.package(context))
        gate.verify()
        status['status'] = 'PACKAGED'
        if context.get('blocked'):
            # Blocked steps keep their step applications: the kit simulates, and live runs refuse.
            status.update(status='PACKAGED_BLOCKED', blocked=context['blocked'])
    except Exception as error:
        status['error'] = str(error)
        raise
    finally:
        (out / 'deploy-status.json').write_text(json.dumps(status, indent=2) + '\n')
    return out


def run(out, **options):
    from .deployment_run import execute
    return execute(out, **options)


def _deploy_command(args):
    print(deploy(export=args.export, out=args.out, jtag=args.jtag))
    return 0


def _run_command(args):
    options = {name: getattr(args, name) for name in
               ('dry_run', 'simulate', 'negative', 'serial', 'inputs', 'references') if getattr(args, name)}
    run(out=args.out, **options)
    return 0


def _reference_command(args):
    from .deployment_stages import new_inputs
    print(new_inputs(args.out, args.inputs, args.destination))
    return 0


def add_commands(commands):
    dp = commands.add_parser('deploy', help='deploy an authenticated export through atlascc build')
    dp.add_argument('export', metavar='EXPORT')
    dp.add_argument('--jtag', action='store_true', help='select small-module JTAG deployment')
    dp.add_argument('-o', dest='out', metavar='OUT', help='fresh deployment directory')
    dp.set_defaults(run=_deploy_command)
    rp = commands.add_parser('run', help='run a packaged deployment on the chip (--dry-run only checks it)')
    rp.add_argument('out', metavar='OUT')
    mode = rp.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='authenticate and validate without transport')
    mode.add_argument('--simulate', action='store_true',
                      help='execute offline: a composed bundle, or a board DDR deployment on the simulated board')
    rp.add_argument('--negative', action='store_true', help='run the expected-fail negative rung instead of the positive')
    rp.add_argument('--serial', metavar='PATH', help="the board's UART device (default: the runner's Linux FTDI if01 path)")
    rp.add_argument('--inputs', metavar='FILE', nargs='+', help='run one step per .npz of input_NNN arrays; - reads INPUTS [REFERENCE] lines from stdin')
    rp.add_argument('--reference', dest='references', metavar='DIR', nargs='+', help='check each step against an atlascc reference made from its inputs')
    rp.set_defaults(run=_run_command)
    fp = commands.add_parser('reference', help='evaluate a packaged deployment on new inputs, for atlascc run --reference')
    fp.add_argument('out', metavar='OUT')
    fp.add_argument('--inputs', metavar='FILE', required=True, help='.npz of input_NNN arrays, as atlascc run takes')
    fp.add_argument('-o', dest='destination', metavar='DIR', required=True, help='fresh reference directory outside OUT')
    fp.set_defaults(run=_reference_command)
