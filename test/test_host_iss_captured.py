"""Controls for reply-only diagnostics; no accelerator is executed."""
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

TOOLS = Path(__file__).absolute().parents[1]/'tools'
sys.path[:0] = [str(TOOLS),str(TOOLS/'host_iss')]
from bundle import EPILOGUE
from captured_host import Replies
from qmp_memory import PhysicalMemory


class PhysicalMemoryTest(unittest.TestCase):
    def test_running_guest_is_refused_before_dump(self):
        memory = PhysicalMemory.__new__(PhysicalMemory)
        memory.command = Mock(return_value={'running':True})
        with self.assertRaisesRegex(ValueError,'all-stop'):
            memory.read(0x80000000,16)
        memory.command.assert_called_once_with('query-status')

    def test_stale_and_short_dump_are_refused_or_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PhysicalMemory.__new__(PhysicalMemory)
            memory.directory = Path(directory)
            target = memory.directory/'physical-read.bin'
            target.write_bytes(b'stale')
            def command(name, args=None):
                if name == 'query-status': return {'running':False}
                self.assertFalse(target.exists())
                target.write_bytes(b'bad')
                return ''
            memory.command = command
            with self.assertRaisesRegex(ValueError,'short'):
                memory.read(0x80000000,4)
            self.assertFalse(target.exists())


class ReplyControlTest(unittest.TestCase):
    def fixture(self, out):
        original = struct.pack('<I',0x73)
        item = dict(original='program',expected='expected',dram_base=0x100000000,output_offset=16,call=0)
        bundle = SimpleNamespace(kernel_count=1,kernels=[item],sums={'expected':'bound'},
            read={'program':original,'expected':b'r'*16}.__getitem__,initial_dram=lambda index:bytes(48))
        remote,physical = Mock(),Mock()
        physical.read.return_value = bytes(48)
        replies = Replies(remote,physical,Path(out),bundle)
        replies.sequence = 1
        replies.imem[:len(EPILOGUE)] = EPILOGUE
        replies.written = set(range(len(EPILOGUE)//4))
        return replies,remote,physical

    def test_bad_actual_staging_cannot_be_replaced_by_reply(self):
        with tempfile.TemporaryDirectory() as out:
            replies,remote,physical = self.fixture(out)
            physical.read.return_value = b'x'+bytes(47)
            with self.assertRaisesRegex(ValueError,'predecessor/staging'):
                replies.start()
            remote.write.assert_not_called()
            self.assertFalse(replies.completed)

    def test_good_reply_receipt_never_claims_live_device_or_cycles(self):
        with tempfile.TemporaryDirectory() as out:
            replies,remote,_ = self.fixture(out)
            replies.start()
            remote.write.assert_called_once_with(0x100000010,b'r'*16)
            result = replies.completed[0]
            self.assertTrue(result['captured_reply_injected'])
            self.assertFalse(result['device_execution'])
            self.assertNotIn('cycles',result)
            self.assertNotIn('drained',result)
            self.assertNotIn('exact',result)
            self.assertEqual(replies.marker,1)
            self.assertFalse(replies.written)
            self.assertEqual(json.loads((Path(out)/'reply-0000.json').read_text()),result)


if __name__ == '__main__': unittest.main()
