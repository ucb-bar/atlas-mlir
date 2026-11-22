"""Run a board DDR kit: load its one ELF into DDR, start the hart, and judge its UART records and output bytes.

The protocol is the same on every link; a transport only identifies the board, moves bytes, starts the hart at an
address, and streams the UART. Loading is chunked as `loader-manifest.json` describes: every chunk is read back and
must match its SHA-256, and a journal of verified chunks lets an interrupted load resume. The run passes when the UART
reports status 0, one hash per declared range, and completion; when the ELF links `tohost`, the exit word it writes;
when each hash range read back from DDR has its reported hash; and when the bytes equal the kit's references.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import time

if __package__:
    from .kit_manifest import safe_path
    from .run_kit import NumericalError, step_hashes
    from .tcl_rpc import TransportError
else:
    from kit_manifest import safe_path
    from run_kit import NumericalError, step_hashes
    from tcl_rpc import TransportError


class Transport:
    """One board link. Each method raises TransportError when the link fails.

    `journal` is the path of the board's one load journal, shared by every deployment run on that board.
    `requires_tohost` holds for a real hart: Shuttle's write-back L1 has no flush, so outputs reach DDR only through
    the tohost exit code's eviction, and an ELF without it is refused before any write.
    """
    journal = None
    requires_tohost = True

    def identify(self):
        """The board: dict(board, link, ddr=dict(address, bytes), session). `link` names the transport (recorded in
        each run, so a simulated run is never mistaken for a board run). `session` names one power-on of the board's DDR
        (equal only while its contents persist), or is None when the link cannot tell, so journaled chunks are
        read back before they are trusted."""
        raise NotImplementedError

    def write(self, address, data):
        raise NotImplementedError

    def read(self, address, count):
        raise NotImplementedError

    def start(self, entry, tohost):
        """Reset the hart and run it from `entry`; the UART is clear from here on. `tohost` is the address of the
        program's exit word, or None when it links none."""
        raise NotImplementedError

    def records(self, timeout, console, tag):
        """UART bytes, copied to `console`, until the `tag COMPLETE` line or, when the program links `tohost`, until it
        writes its exit word. A nonzero status raises NumericalError."""
        raise NotImplementedError

    def close(self):
        """Stop the hart and release the link; safe to call more than once."""


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _member(bundle, name):
    path = safe_path(bundle / name)
    if not path.is_file():
        raise ValueError('kit member missing: ' + name)
    return path


def verify_kit(bundle):
    """Every SHA256SUMS member matches; returns the inventory names."""
    names = set()
    for line in _member(bundle, 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if name in names or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('malformed kit inventory')
        names.add(name)
        if _sha(_member(bundle, name).read_bytes()) != expected:
            raise ValueError('kit integrity mismatch: ' + name)
    return names


def tohost(raw):
    """The `tohost` symbol's address in an ELF64, or None when it is not linked."""
    shoff, = struct.unpack_from('<Q', raw, 40)
    size, count = struct.unpack_from('<HH', raw, 58)
    sections = [struct.unpack_from('<IIQQQQIIQQ', raw, shoff + i * size) for i in range(count)] if shoff else []
    for section in sections:
        if section[1] != 2:  # SHT_SYMTAB
            continue
        strings = sections[section[6]]
        for offset in range(section[4], section[4] + section[5], 24):
            name, _, _, _, value, _ = struct.unpack_from('<IBBHQQ', raw, offset)
            start = strings[4] + name
            if raw[start:raw.index(b'\0', start)] == b'tohost':
                return value
    return None


def plan(bundle, elf='chip.elf'):
    """The kit's checked run plan: board DDR, entry, load rows (chunks, then zero-fill tails), and hash ranges."""
    names = verify_kit(bundle)
    if not {'loader-manifest.json', 'memory-plan.json', 'manifest.json', elf} <= names:
        raise ValueError('kit has no loader manifest and memory plan; redeploy it')
    loader = json.loads(_member(bundle, 'loader-manifest.json').read_text())
    memory = json.loads(_member(bundle, 'memory-plan.json').read_text())
    manifest = json.loads(_member(bundle, 'manifest.json').read_text())
    raw = _member(bundle, elf).read_bytes()
    if loader.get('schema') != 'atlas.loader_manifest.v1' or memory.get('schema') != 'atlas.deployment_memory.v1':
        raise ValueError('unsupported loader manifest or memory plan')
    base, size = memory['ddr']['address'], memory['ddr']['bytes']
    # Both records name the SoC declaration the kit was placed against; its identity names the board.
    if loader['soc'] != memory['soc'] or loader['entry'] != int.from_bytes(raw[24:32], 'little'):
        raise ValueError('loader manifest differs from the ELF')
    if elf == 'chip.elf' and loader['image']['sha256'] != _sha(raw):
        raise ValueError('loader manifest image differs from chip.elf')
    # The negative ELF has the same loads as chip.elf; its chunk hashes come from its own bytes.
    keys = ('address', 'file_bytes', 'memory_bytes', 'flags', 'file_offset')
    if _loads(raw) != [{key: segment[key] for key in keys} for segment in loader['segments']]:
        raise ValueError(f'{elf} loads differ from the loader manifest')
    rows = []
    for segment in loader['segments']:
        writable = bool(segment['flags'] & 2)
        for chunk in segment['chunks']:
            data = raw[chunk['file_offset']:chunk['file_offset'] + chunk['bytes']]
            if len(data) != chunk['bytes'] or elf == 'chip.elf' and _sha(data) != chunk['sha256']:
                raise ValueError(f'loader chunk at {chunk["address"]:#x} differs from the ELF')
            rows.append(dict(address=chunk['address'], bytes=chunk['bytes'], file_offset=chunk['file_offset'],
                             sha256=_sha(data), writable=writable))
        tail = segment['zero_fill']
        for start in range(0, tail['bytes'], loader['chunk_bytes']):
            count = min(loader['chunk_bytes'], tail['bytes'] - start)
            rows.append(dict(address=tail['address'] + start, bytes=count, file_offset=None,
                             sha256=_sha(bytes(count)), writable=True))
    for row in rows:
        if not base <= row['address'] < row['address'] + row['bytes'] <= base + size:
            raise ValueError(f'load at {row["address"]:#x} is outside board DDR')
    return dict(board=loader['soc']['identity'], ddr=dict(address=base, bytes=size), entry=loader['entry'], rows=rows, raw=raw,
                tohost=tohost(raw), hash_ranges=manifest['deployment']['hash_ranges'], tag=manifest['runner']['record_tag'],
                timeout=manifest['platform_assumptions']['timeout_seconds'])


class Journal:
    """Chunks verified at their destination on one board session, kept across invocations.

    A chunk is keyed by its address, length, and SHA-256, so a later ELF that shares chunks (the same weights) skips
    them. Any write over a chunk's range forgets it first. Starting the hart forgets every writable chunk, since the
    program may change it; read-only chunks are trusted, not enforced, across runs (nothing on the chip stops a
    program writing one), and a program that writes one fails its next run as a mismatch.
    """

    def __init__(self, path, identity):
        self.path, self.identity = path, dict(board=identity['board'], session=identity['session'])
        record = json.loads(path.read_text()) if path.is_file() else {}
        self.verified = record.get('verified', {}) if record.get('identity') == self.identity else {}

    @staticmethod
    def key(row):
        return f'{row["address"]:#x}:{row["bytes"]}:{row["sha256"]}'

    def forget(self, row):
        """Drop every chunk that overlaps `row`, before `row` is written."""
        def overlaps(key):
            address, count, _ = key.split(':')
            return int(address, 16) < row['address'] + row['bytes'] and row['address'] < int(address, 16) + int(count)
        self.verified = {key: writable for key, writable in self.verified.items() if not overlaps(key)}
        self.save()

    def add(self, row):
        self.verified[self.key(row)] = row['writable']
        self.save()

    def started(self):
        self.verified = {key: writable for key, writable in self.verified.items() if not writable}
        self.save()

    def save(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(identity=self.identity, verified=self.verified), indent=1) + '\n')
        os.replace(temporary, self.path)


def load(transport, run, journal, log):
    """Write and verify every load row not already verified on this board session; returns the counts."""
    counts = dict(rows=len(run['rows']), written=0, resumed=0, bytes_written=0)
    started = time.monotonic()
    for row in run['rows']:
        if Journal.key(row) in journal.verified:
            counts['resumed'] += 1
            continue
        if row['file_offset'] is None:
            data = bytes(row['bytes'])
        else:
            data = run['raw'][row['file_offset']:row['file_offset'] + row['bytes']]
        journal.forget(row)
        transport.write(row['address'], data)
        back = transport.read(row['address'], row['bytes'])
        if _sha(back) != row['sha256']:
            raise TransportError(f'load verification failed: {row["bytes"]} bytes at {row["address"]:#x} differ from '
                                 'their hash')
        journal.add(row)
        counts['written'] += 1
        counts['bytes_written'] += row['bytes']
        log.write(f'LOAD {row["address"]:#x} {row["bytes"]} {row["sha256"]} verified\n'); log.flush()
    counts['seconds'] = round(time.monotonic() - started, 3)
    return counts


def execute(bundle, transport, logs, *, negative=False):
    """One run of chip.elf (or the negative ELF) on `transport`; returns the run record.

    Raises NumericalError when the outputs differ from the references (the negative's expected failure), and
    TransportError or ValueError for any other failure.
    """
    elf = 'negative/chip.elf' if negative else 'chip.elf'
    run = plan(bundle, elf)
    if run['tohost'] is None and transport.requires_tohost:
        raise ValueError('board DDR runs need the tohost exit build: without it the outputs may stay in the cache')
    logs.mkdir(parents=True)
    identity = transport.identify()
    if identity['board'] != run['board'] or identity['ddr'] != run['ddr']:
        raise TransportError(f'board {identity["board"]} with DDR {identity["ddr"]} is not the kit target {run["board"]}')
    journal = Journal(transport.journal, identity)
    if identity['session'] is None:
        # Without a session the board may have lost its DDR since the journal was written: confirm each chunk.
        journal.verified = {key: writable for key, writable in journal.verified.items()
                            if _verified(transport, key)}
    with (logs / 'load.log').open('w') as log:
        counts = load(transport, run, journal, log)
    journal.started()
    transport.start(run['entry'], run['tohost'])
    with (logs / 'uart.log').open('wb') as console:
        text = transport.records(run['timeout'], console, run['tag']).decode('utf-8', 'replace')
    reported = step_hashes(run['hash_ranges'], text, run['tag'])
    exit_word = None
    if run['tohost'] is not None:
        exit_word = int.from_bytes(transport.read(run['tohost'], 8), 'little')
        if exit_word & 1 and exit_word != 1:
            raise NumericalError(f'program exited with status {exit_word >> 1} (tohost {exit_word:#x})')
        if exit_word != 1:
            raise TransportError(f'tohost is {exit_word:#x}, not an exit word')
    outputs, mismatch = [], None
    for row in run['hash_ranges']:
        data = transport.read(row['address'], row['bytes'])
        (logs / f'output-{row["index"]}.bin').write_bytes(data)
        if _sha(data) != reported[row['id']]:
            raise TransportError('DDR bytes differ from the UART hash: ' + row['id'])
        expected = _member(bundle, f'reference/output_{row["index"]:03d}.bin').read_bytes()
        outputs.append(dict(id=row['id'], sha256=_sha(data), expected_sha256=_sha(expected)))
        if data != expected and mismatch is None:
            mismatch = row['id']
        evidence = f'negative/negative-evidence/output_{row["index"]:03d}.bin'
        if negative and data != _member(bundle, evidence).read_bytes():
            raise ValueError(f'negative output {row["id"]} differs from its packaged evidence')
    record = dict(elf=elf, board=identity, load=counts, tohost=exit_word, outputs=outputs,
                  verdict='EXACT' if mismatch is None else 'MISMATCH')
    (logs / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    if mismatch is not None:
        raise NumericalError('output differs from the kit reference: ' + mismatch)
    return record


def _loads(raw):
    """An ELF64's PT_LOAD rows, as the loader manifest lists them."""
    phoff, = struct.unpack_from('<Q', raw, 32)
    size, count = struct.unpack_from('<HH', raw, 54)
    rows = []
    for i in range(count):
        kind, flags, offset, address, _, file_bytes, memory_bytes, _ = struct.unpack_from('<II6Q', raw, phoff + i * size)
        if kind == 1:
            rows.append(dict(address=address, file_bytes=file_bytes, memory_bytes=memory_bytes, flags=flags,
                             file_offset=offset))
    return rows


def _verified(transport, key):
    address, count, digest = key.split(':')
    return _sha(transport.read(int(address, 16), int(count))) == digest
