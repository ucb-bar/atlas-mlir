"""Small all-stop RSP client, with finite socket waits and exact transfer checks.

Protocol: https://sourceware.org/gdb/current/onlinedocs/gdb.html/Overview.html
Packets: https://sourceware.org/gdb/current/onlinedocs/gdb.html/Packets.html
"""

from atlas_tooling import sha
import json
import re
import socket
import time


def frame(payload):
    return b"$" + payload + b"#" + f"{sum(payload)&255:02x}".encode()


def decode_response(raw):
    output = bytearray()
    index = 0
    while index < len(raw):
        char = raw[index]
        index += 1
        if char == ord("}"):
            if index == len(raw):
                raise ValueError("truncated escape")
            output.append(raw[index] ^ 0x20)
            index += 1
        elif char == ord("*"):
            if not output or index == len(raw):
                raise ValueError("truncated RLE")
            count = raw[index] - 29
            index += 1
            if not 3 <= count <= 97:
                raise ValueError("invalid RLE count")
            output.extend([output[-1]] * count)
        else:
            output.append(char)
        if len(output) > 16 * 1024 * 1024:
            raise ValueError("remote packet exceeds bound")
    return bytes(output)


class Remote:
    def __init__(self, path, trace=None, timeout=1800, deadline=None):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.timeout = timeout
        self.deadline = deadline
        self._buffer = b""
        self._offset = 0
        self.limit_wait()
        self.socket.connect(path)
        self.trace = trace
        self.packet_limit = 1024

    def close(self):
        self.socket.close()

    def limit_wait(self):
        deadline = getattr(self, "deadline", None)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("absolute RSP deadline")
            self.socket.settimeout(min(getattr(self, "timeout", 1800), remaining))

    def send(self, data):
        self.limit_wait()
        self.socket.sendall(data)

    def byte(self):
        self.limit_wait()
        if getattr(self, "_offset", 0) >= len(getattr(self, "_buffer", b"")):
            self._buffer = self.socket.recv(65536)
            self._offset = 0
            if not self._buffer:
                raise EOFError("remote disconnected")
        result = self._buffer[self._offset : self._offset + 1]
        self._offset += 1
        return result

    def receive(self):
        while self.byte() != b"$":
            pass
        raw = bytearray()
        escaped = False
        repeat_count = False
        while True:
            char = self.byte()
            if char == b"#" and not escaped and not repeat_count:
                break
            raw.extend(char)
            if len(raw) > 16 * 1024 * 1024:
                raise ValueError("packet exceeds bound")
            if repeat_count:
                repeat_count = False
            elif escaped:
                escaped = False
            elif char == b"}":
                escaped = True
            elif char == b"*":
                repeat_count = True
        checksum = int(self.byte() + self.byte(), 16)
        if checksum != sum(raw) & 255:
            self.send(b"-")
            raise ValueError("remote checksum differs")
        self.send(b"+")
        return decode_response(raw)

    def request(self, command):
        command = command.encode() if isinstance(command, str) else command
        if any(c in command for c in (b"$", b"#", b"}")):
            raise ValueError("unescaped outgoing command")
        started = time.monotonic()
        for attempt in range(3):
            self.send(frame(command))
            ack = self.byte()
            if ack == b"+":
                break
            if ack != b"-":
                raise ValueError("missing remote acknowledgment")
        else:
            raise ValueError("remote refused packet repeatedly")
        reply = self.receive()
        if self.trace:
            self.trace.write(json.dumps(dict(command_prefix=command[:80].decode(errors="replace"),
                command_bytes=len(command), command_sha256=sha(command), response_bytes=len(reply),
                response_sha256=sha(reply), wall_seconds=time.monotonic() - started)) + "\n")
            self.trace.flush()
        if re.fullmatch(rb"E[0-9a-fA-F]{2}(:.*)?", reply):
            raise ValueError("remote error " + reply.decode())
        return reply

    def reg(self, index):
        assert 0 <= index <= 32
        value = self.request(f"p{index:x}")
        if not value:
            all_regs = self.request("g")
            assert len(all_regs) >= 33 * 16
            value = all_regs[index * 16 : (index + 1) * 16]
        raw = bytes.fromhex(value.decode())
        assert len(raw) == 8
        return int.from_bytes(raw, "little")

    def set_reg(self, index, value):
        assert 0 < index <= 32 and 0 <= value < 1 << 64
        value = value.to_bytes(8, "little").hex().encode()
        reply = self.request(f"P{index:x}=".encode() + value)
        if not reply:
            all_regs = self.request("g")
            assert len(all_regs) >= 33 * 16
            reply = self.request(b"G" + all_regs[: index * 16] + value + all_regs[(index + 1) * 16 :])
        assert reply == b"OK"

    def read(self, address, length):
        assert 0 <= address < 1 << 64 and 0 <= length <= 64 * 1024 * 1024 and address + length <= 1 << 64
        chunks = []
        for offset in range(0, length, self.packet_limit):
            count = min(self.packet_limit, length - offset)
            raw = bytes.fromhex(self.request(f"m{address+offset:x},{count:x}").decode())
            assert len(raw) == count, "short remote memory read"
            chunks.append(raw)
        return b"".join(chunks)

    def write(self, address, data):
        assert 0 <= address < 1 << 64 and len(data) <= 64 * 1024 * 1024 and address + len(data) <= 1 << 64
        for offset in range(0, len(data), self.packet_limit):
            block = data[offset : offset + self.packet_limit]
            assert self.request(f"M{address+offset:x},{len(block):x}:".encode() + block.hex().encode()) == b"OK"

    def breakpoint(self, address, length=2, kind=0):
        assert self.request(f"Z{kind:x},{address:x},{length:x}") == b"OK"

    def resume(self):
        response = self.request("c")
        assert response[:1] in (b"S", b"T") and response[1:3] == b"05", response
        return self.reg(32), response
