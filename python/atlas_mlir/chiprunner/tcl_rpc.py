# Bring-up tooling: shared, fail-closed OpenOCD Tcl RPC transport.
import re
import socket
import time

# Jim catch returns the raw return code. OpenOCD 0.12 returns a failed handler's negative ERROR_* code as that code.
CATCH_CODE = re.compile(r'-?[0-9]{1,10}')


class TransportError(RuntimeError):
    pass


class CommandError(TransportError):
    """OpenOCD ran the command and it failed. The session is marked broken, as for every other failure."""

    def __init__(self, code, message):
        self.code, self.message = code, message
        super().__init__('OpenOCD command failed (catch rc=%d): %s' % (code, message.strip()[:400]))


def tcl_path(path):
    value = str(path)
    if any(c in value for c in '{}\n\r\x1a'):
        raise ValueError('unsupported Tcl path characters')
    return '{' + value + '}'


class TclRPC:
    def __init__(self, host='127.0.0.1', port=29666, timeout=120):
        self.timeout, self.broken = timeout, False
        self.sock = socket.create_connection((host, port), timeout=timeout)

    def _rpc(self, command):
        if self.broken:
            raise TransportError('Tcl session is broken')
        deadline = time.monotonic() + self.timeout
        try:
            self.sock.settimeout(self.timeout)
            self.sock.sendall(command.encode() + b'\x1a')
            data = b''
            while not data.endswith(b'\x1a'):
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError('RPC deadline')
                self.sock.settimeout(left)
                chunk = self.sock.recv(65536)
                if not chunk:
                    raise ConnectionError('OpenOCD closed Tcl connection')
                data += chunk
            if data.count(b'\x1a') != 1:
                raise ConnectionError('Tcl framing error')
            return data[:-1].decode('utf-8', 'strict')
        except (OSError, UnicodeError) as error:
            self.broken = True
            raise TransportError(str(error)) from error

    def command(self, command):
        rc = self._rpc('set chain_rc [catch {capture {%s}} chain_msg]' % command).strip()
        if not CATCH_CODE.fullmatch(rc):
            self.broken = True
            raise TransportError('invalid Tcl catch reply: %r' % rc[:80])
        message = self._rpc('set chain_msg')
        if int(rc) != 0:
            self.broken = True
            raise CommandError(int(rc), message)
        return message

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
