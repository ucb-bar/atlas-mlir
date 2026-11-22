"""File-only loader contract derived from the qualified single ELF.

Offsets refer to chip.elf. No second copy of resident weights is produced.
chiprunner/ddr.py loads, verifies, and resumes from it over a board transport.
"""
import hashlib
import struct

from .deploy import Refusal, safe_path

CHUNK_BYTES = 1024 * 1024


def describe(elf, deployment, record):
    """The memory plan and loader manifest of a board DDR ELF, rechecked against its bytes and the declared DDR."""
    from atlas_mlir_native import soc
    from .deployment_link import HOST_ALIGNMENT

    ddr = soc.SOC.memory_region('ddr')
    base, end = ddr.base, ddr.end
    declaration = dict(identity=soc.IDENTITY, sha256=soc.DECLARATION_SHA256, kit_member='bench/soc.json')
    host, window = deployment['host_allocation'], deployment['window']
    reserved = host['bytes']
    if (host['address'] != base or not 0 < reserved < ddr.capacity_bytes
            or reserved % HOST_ALIGNMENT or not 0 < host['usable_bytes'] <= reserved
            or window != dict(address=base + reserved, bytes=end - base - reserved)
            or record['entry'] != base):
        raise Refusal('loader memory plan differs from measured host placement')
    if record['host_reservations']['__stack_end'] > base + host['usable_bytes']:
        raise Refusal('loader host stack exceeds the measured host exclusion')

    with safe_path(elf).open('rb') as stream:
        stream.seek(0, 2)
        file_bytes = stream.tell()
        stream.seek(0)
        header = stream.read(64)
        if (len(header) != 64 or header[:6] != b'\x7fELF\x02\x01'
                or struct.unpack_from('<H', header, 18)[0] != 243
                or struct.unpack_from('<Q', header, 24)[0] != base):
            raise Refusal('loader requires a little-endian RV64 ELF at the DDR base')
        phoff, = struct.unpack_from('<Q', header, 32)
        phsize, phcount = struct.unpack_from('<HH', header, 54)
        if phsize != 56 or phoff + phsize * phcount > file_bytes:
            raise Refusal('loader ELF load table is truncated')
        stream.seek(phoff)
        table = stream.read(phsize * phcount)
        loads = []
        for i in range(phcount):
            kind, flags, offset, address, physical, size, memory, _ = struct.unpack_from('<II6Q', table, i * phsize)
            if kind != 1:
                continue
            if (physical != address or size > memory or offset + size > file_bytes
                    or not base <= address <= address + memory <= end
                    or any(address < r['address'] + r['memory_bytes'] and r['address'] < address + memory for r in loads)):
                raise Refusal('loader ELF load is invalid or outside DDR')
            loads.append(dict(address=address, file_bytes=size, memory_bytes=memory, flags=flags, file_offset=offset))
        if [{k: v for k, v in row.items() if k != 'file_offset'} for row in loads] != record['loads']:
            raise Refusal('loader ELF loads differ from the qualified link')

        def chunks(offset, address, size):
            rows, digest = [], hashlib.sha256()
            for start in range(0, size, CHUNK_BYTES):
                count = min(CHUNK_BYTES, size - start)
                stream.seek(offset + start)
                data = stream.read(count)
                if len(data) != count:
                    raise Refusal('loader payload is truncated')
                digest.update(data)
                rows.append(dict(file_offset=offset + start, address=address + start,
                                 bytes=count, sha256=hashlib.sha256(data).hexdigest()))
            return rows, digest.hexdigest()

        segments = []
        for row in loads:
            parts, sha = chunks(row['file_offset'], row['address'], row['file_bytes'])
            segments.append(dict(**row, sha256=sha, chunks=parts,
                                 zero_fill=dict(address=row['address'] + row['file_bytes'],
                                                bytes=row['memory_bytes'] - row['file_bytes'])))
        weights = deployment['parameters']
        containing = [r for r in loads if r['flags'] == 4 and r['address'] <= weights['address']
                      and weights['address'] + weights['bytes'] <= r['address'] + r['file_bytes']]
        if len(containing) != 1:
            raise Refusal('loader weights lack one read-only ELF backing')
        row, = containing
        offset = row['file_offset'] + weights['address'] - row['address']
        parts, sha = chunks(offset, weights['address'], weights['bytes'])
        if sha != weights['sha256']:
            raise Refusal('loader linked weights differ from bound parameters')
        stream.seek(0)
        digest = hashlib.sha256()
        while data := stream.read(CHUNK_BYTES):
            digest.update(data)
        if digest.hexdigest() != record['sha256']:
            raise Refusal('loader ELF differs from the qualified link')

    plan = dict(schema='atlas.deployment_memory.v1', soc=declaration,
                ddr=dict(address=base, bytes=ddr.capacity_bytes),
                host_reservation=dict(address=base, bytes=reserved,
                                      basis='linker-measured host code, data, heap, and stack; rounded to 4 KiB'),
                host=host, host_symbols=record['host_reservations'], window=window,
                allocations=deployment['allocations'], sections=record['allocated_sections'],
                instruction_images=record['instruction_images'], loads=loads,
                parameters=weights)
    manifest = dict(schema='atlas.loader_manifest.v1', soc=declaration,
                    image=dict(path='chip.elf', bytes=file_bytes, sha256=record['sha256']),
                    entry=base, chunk_bytes=CHUNK_BYTES, segments=segments,
                    weights=dict(path='chip.elf', file_offset=offset, address=weights['address'],
                                 bytes=weights['bytes'], sha256=sha, chunks=parts),
                    completion='verify every chunk at its destination and zero all tails before entry',
                    resume=('key verified chunks by address, length, and chunk sha256 on one board session; '
                            'any write over a chunk\'s range forgets it'),
                    transport_executed=False)
    return plan, manifest
