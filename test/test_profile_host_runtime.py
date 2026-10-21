"""Check runtime instrumentation independently of model timing or FPGA claims."""

import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from test_looped_mnk import ROOT
from tools.profile_host_runtime import HostRuntimeProfile


class State:
    halted = False
    reason = 0
    retired = 0

    def read_csrf(self, address):
        return (
            self.retired if address == 0xC01 else (self.reason << 1) | int(self.halted)
        )


def package():
    from merlin.compile.c_sources import c_source_pair

    binary = struct.pack("<2I", 123, 456)
    text = c_source_pair(
        "image",
        public_headers={"atlas_host.h": (ROOT / "runtime/atlas_host.h").read_text()},
        private_headers={},
        sources={
            "data.c": """
        const uint32_t image_program[]={123,456};
        const atlas_host_buffer image_device_buffers[]={{0x90000000u,32,ATLAS_HOST_DEVICE_READ}};
        """
        },
        declarations="""
        #define IMAGE_PROGRAM_WORDS 2u
        #define IMAGE_DEVICE_BUFFER_COUNT 1u
        extern const uint32_t image_program[2];
        extern const atlas_host_buffer image_device_buffers[1];
        """,
    )
    files = {name: value.encode() for name, value in text.items()}
    record = {
        "program_sha256": hashlib.sha256(binary).hexdigest(),
        "files": {
            key: hashlib.sha256(value).hexdigest() for key, value in files.items()
        },
    }
    return record, files, binary


class HostRuntimeProfileTest(unittest.TestCase):
    def test_cold_load_reuse_actual_csr_completion_and_invalid_results(self):
        record, files, binary = package()
        with tempfile.TemporaryDirectory(prefix="host-clock-") as path:
            root = Path(path)
            host = HostRuntimeProfile(root / "valid", record, files, binary, ROOT)
            for index in range(3):
                state = State()
                host.begin(state)
                with self.assertRaisesRegex(ValueError, "before the model halts"):
                    host.end(state)
                state.halted, state.reason, state.retired = True, 2, index + 2
                sample = host.end(state)
                self.assertEqual(sample["image_loads"], int(index == 0))
                self.assertEqual(sample["image_reuses"], int(index != 0))
                self.assertEqual(sample["words_written"], 2 if index == 0 else 0)
                self.assertEqual(sample["launches"], 1)
                self.assertTrue(all(value >= 0 for value in sample.values()))
            state = State()
            host.begin(state)
            state.halted, state.reason = True, 1
            with self.assertRaisesRegex(ValueError, "rejected model completion"):
                host.end(state)
            with self.assertRaisesRegex(ValueError, "different program"):
                HostRuntimeProfile(root / "wrong-image", record, files, b"bad!", ROOT)
            altered = {**files, "image.c": files["image.c"] + b"\n"}
            with self.assertRaisesRegex(ValueError, "member changed"):
                HostRuntimeProfile(
                    root / "changed-source", record, altered, binary, ROOT
                )


if __name__ == "__main__":
    unittest.main()
