"""Bounded QMP physical reads while the caller holds QEMU stopped via RSP."""

import json
import socket
import time

from atlas_path_guard import checked


class PhysicalMemory:
    def __init__(self, sock, directory, deadline):
        self.deadline, self.directory = deadline, checked(directory)
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.counter = 0
        self.reader = None
        try:
            self.limit_wait()
            self.socket.connect(str(checked(sock)))
            self.reader = self.socket.makefile('rb')
            greeting = self.receive()
            if not isinstance(greeting.get('QMP'), dict):
                raise ValueError('QMP greeting missing')
            self.command('qmp_capabilities')
        except BaseException:
            self.close()
            raise

    def limit_wait(self):
        remaining = self.deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('QMP absolute deadline')
        self.socket.settimeout(remaining)

    def receive(self):
        self.limit_wait()
        raw = self.reader.readline(1024*1024+1)
        if not raw or len(raw) > 1024*1024 or not raw.endswith(b'\n'):
            raise ValueError('invalid or oversized QMP response')
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError('QMP response must be an object')
        return value

    def command(self, name, arguments=None):
        self.counter += 1
        identity = self.counter
        value = dict(execute=name, id=identity)
        if arguments is not None:
            value['arguments'] = arguments
        self.limit_wait()
        self.socket.sendall(json.dumps(value).encode()+b'\n')
        while True:
            result = self.receive()
            if 'event' in result:
                continue
            if result.get('id') != identity:
                raise ValueError('unexpected QMP response identity')
            if 'error' in result:
                raise ValueError('QMP command failed: '+str(result['error']))
            if 'return' not in result:
                raise ValueError('QMP response lacks result')
            return result['return']

    def read(self, address, length):
        if (type(address) is not int or type(length) is not int or not 0 <= address < 1 << 64
                or not 0 <= length <= 64*1024*1024 or address+length > 1 << 64):
            raise ValueError('invalid physical read range')
        if not length:
            return b''
        status = self.command('query-status')
        if status.get('running') is not False:
            raise ValueError('bulk physical reads require an all-stop guest')
        target = checked(self.directory/'physical-read.bin')
        target.unlink(missing_ok=True)
        filename = str(target)
        if any(value in filename for value in ('"', '\\', '\n', '\r')):
            raise ValueError('unsupported physical-read output filename')
        try:
            result = self.command('human-monitor-command',
                                  {'command-line':f'pmemsave 0x{address:x} 0x{length:x} "{filename}"'})
            if result != '':
                raise ValueError('QEMU physical dump failed: '+str(result))
            if target.stat().st_size != length:
                raise ValueError('short physical dump')
            return target.read_bytes()
        finally:
            target.unlink(missing_ok=True)

    def close(self):
        if self.reader is not None:
            self.reader.close()
        self.socket.close()
