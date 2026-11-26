"""Offline OpenOCD Tcl client checks: a fake peer on a socketpair replays recorded and synthetic replies."""
import socket
import threading
import unittest

from atlas_mlir.chiprunner import tcl_rpc

WRAP = 'set chain_rc [catch {capture {%s}} chain_msg]'
# Recorded on a dry connect whose startup examine failed: the server logged "Error: Target not examined yet" for
# halt and echoed catch code -4 (ERROR_FAIL). The message text was never fetched; the logged line stands in for it.
RECORDED_HALT_RC, RECORDED_HALT_MSG = '-4', 'Target not examined yet\n'
RECORDED_REG_PC = 'pc (/64): 0x0000000008000734\n'


def frame(text):
    return [text.encode() + b'\x1a']


class Peer:
    """Reads 0x1a-terminated requests and answers each from script (request to raw chunks); unknown closes."""

    def __init__(self, script):
        self.script, self.requests = script, []
        self.client, self.server = socket.socketpair()
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        buffer = b''
        while True:
            try:
                data = self.server.recv(65536)
            except OSError:
                return
            if not data:
                return
            buffer += data
            while b'\x1a' in buffer:
                request, buffer = buffer.split(b'\x1a', 1)
                self.requests.append(request.decode())
                reply = self.script.get(request.decode())
                if reply is None:
                    self.server.close()
                    return
                for chunk in reply:
                    self.server.sendall(chunk)

    def rpc(self, timeout=5):
        rpc = tcl_rpc.TclRPC.__new__(tcl_rpc.TclRPC)
        rpc.timeout, rpc.broken, rpc.sock = timeout, False, self.client
        return rpc

    def close(self):
        self.client.close()
        self.server.close()
        self.thread.join(2)


class TclRPCTests(unittest.TestCase):
    def peer(self, script):
        peer = Peer(script)
        self.addCleanup(peer.close)
        return peer

    def test_recorded_failure_reports_code_and_message_then_stops(self):
        peer = self.peer({WRAP % 'halt': frame(RECORDED_HALT_RC), 'set chain_msg': frame(RECORDED_HALT_MSG)})
        rpc = peer.rpc()
        with self.assertRaisesRegex(tcl_rpc.CommandError,
                                    r'OpenOCD command failed \(catch rc=-4\): Target not examined yet') as caught:
            rpc.command('halt')
        self.assertIsInstance(caught.exception, tcl_rpc.TransportError)
        self.assertEqual((caught.exception.code, caught.exception.message), (-4, RECORDED_HALT_MSG))
        self.assertTrue(rpc.broken)
        self.assertEqual(peer.requests, [WRAP % 'halt', 'set chain_msg'])
        with self.assertRaisesRegex(tcl_rpc.TransportError, 'Tcl session is broken'):
            rpc.command('reg pc')
        self.assertEqual(len(peer.requests), 2)

    def test_success_returns_the_captured_text(self):
        peer = self.peer({WRAP % 'halt': frame('0'), 'set chain_msg': frame(''), WRAP % 'reg pc': frame('0')})
        rpc = peer.rpc()
        self.assertEqual(rpc.command('halt'), '')
        peer.script['set chain_msg'] = frame(RECORDED_REG_PC)
        self.assertEqual(rpc.command('reg pc'), RECORDED_REG_PC)
        self.assertFalse(rpc.broken)

    def test_other_nonzero_codes_are_command_failures(self):
        for code in ('1', '-601'):  # a Jim error, and OpenOCD's ERROR_COMMAND_SYNTAX_ERROR
            with self.subTest(code=code):
                rpc = self.peer({WRAP % 'bogus': frame(code), 'set chain_msg': frame('failure text')}).rpc()
                with self.assertRaisesRegex(tcl_rpc.CommandError, 'catch rc=%s' % code) as caught:
                    rpc.command('bogus')
                self.assertEqual(caught.exception.code, int(code))
                self.assertTrue(rpc.broken)

    def test_malformed_replies_are_transport_errors_without_a_second_request(self):
        cases = [('non-integer', frame('garbage'), 'invalid Tcl catch reply'),
                 ('empty', frame(''), 'invalid Tcl catch reply'),
                 ('double terminator', [b'0\x1a0\x1a'], 'Tcl framing error'),
                 ('closed', None, 'closed Tcl connection'), ('bad utf-8', [b'\xff\x1a'], 'utf-8')]
        for name, reply, text in cases:
            with self.subTest(name=name):
                peer = self.peer({WRAP % 'halt': reply} if reply is not None else {})
                rpc = peer.rpc()
                with self.assertRaisesRegex(tcl_rpc.TransportError, text) as caught:
                    rpc.command('halt')
                self.assertNotIsInstance(caught.exception, tcl_rpc.CommandError)
                self.assertTrue(rpc.broken)
                self.assertEqual(peer.requests, [WRAP % 'halt'])

    def test_split_reply_is_reassembled(self):
        rpc = self.peer({WRAP % 'reg pc': [b'0', b'\x1a'],
                         'set chain_msg': [b'pc (/64): ', b'0x0000000008000734\n\x1a']}).rpc()
        self.assertEqual(rpc.command('reg pc'), RECORDED_REG_PC)

    def test_silent_peer_times_out_and_breaks_the_session(self):
        rpc = self.peer({WRAP % 'halt': []}).rpc(timeout=0.2)
        with self.assertRaises(tcl_rpc.TransportError):
            rpc.command('halt')
        self.assertTrue(rpc.broken)


if __name__ == '__main__':
    unittest.main()
