"""Segment startup handles real UART initialization before the ready hook."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).absolute().parents[1]/"tools"
sys.path[:0] = [str(TOOLS), str(TOOLS/"host_iss")]
from segment_uart import Transport


class SegmentUARTTest(unittest.TestCase):
    def test_native_transport_preserves_existing_guest_serial(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = Transport(dict(kind="ns16550", base=0x10000000), False, mock.Mock(), Path(directory))
            self.assertEqual(transport.prepare(), "uart.log")
            self.assertFalse(transport.handle(mock.Mock(), 12))
            transport.register(mock.Mock(), mock.Mock(), {1,2,3})
            self.assertEqual(transport.summary()["backend"], "qemu-native-ns16550")
            transport.close()

    def test_sifive_initialization_precedes_ready_without_a_device(self):
        base = 0x10020000
        access = dict(base=5, reg=6, offset=8, store=True, signed=False, length=4,
                      uart_register=base+8, uart_role="initialize")
        with tempfile.TemporaryDirectory() as directory, mock.patch("segment_uart.sites", return_value={100:access}):
            transport = Transport(dict(kind="sifive", base=base), True, mock.Mock(), Path(directory))
            self.assertEqual(transport.prepare(), "qemu-native-uart.log")
            remote = mock.Mock()
            remote.reg.side_effect = {5:base, 6:1}.__getitem__
            image = mock.Mock()
            remote.read.return_value = image.bytes_at.return_value = b"abcd"
            transport.register(remote, image, {200})
            self.assertTrue(transport.handle(remote, 100))
            self.assertTrue(transport.summary()["tx_enabled"])
            remote.set_reg.assert_called_once_with(32, 104)
            with self.assertRaisesRegex(ValueError, "aliases"):
                transport.register(remote, image, {100})
            transport.close()
