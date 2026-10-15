import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location("program_preparation",Path(__file__).with_name("prepare_program.py"))
preparation=importlib.util.module_from_spec(spec);spec.loader.exec_module(preparation)


class PhysicalReplayControls(unittest.TestCase):
    def fixture(self,directory):
        root=Path(directory);payload=bytes(range(32));base=1<<32
        initial=payload+b"\xa5"*256+b"\xcd"*32+b"\x5a"*256
        (root/"initial.bin").write_bytes(initial);(root/"actual.bin").write_bytes(payload)
        (root/"program.bin").write_bytes(b"\x73\0\0\0")
        data={"memory":{"dram_base":base,"input_bytes":32,"output_offset":288,"output_bytes":32},
              "initial":str(root/"initial.bin"),"requested_dma":{"load_bytes":32,"store_bytes":32}}
        case={"id":"positive","expected":payload.hex(),"actual_oracle":payload.hex()}
        events=[{"cycle":0,"event":"host_a","csr":0,"address":0x20000,"data":0x73},
                {"cycle":1,"event":"host_d","csr":0},
                {"cycle":2,"event":"host_a","csr":1,"address":0x18,"data":1},
                {"cycle":3,"event":"host_d","csr":1},
                {"cycle":3,"event":"state","reset":0,"halted":0,"public_idle":1,"outstanding":0},
                {"cycle":4,"event":"dma_a","address":base,"source":0,"opcode":4,"bytes":list(payload)},
                {"cycle":5,"event":"dma_d","source":0,"opcode":1},
                {"cycle":6,"event":"dma_a","address":base+288,"source":1,"opcode":0,"bytes":list(payload)},
                {"cycle":7,"event":"dma_d","source":1,"opcode":0},
                {"cycle":8,"event":"status_a","address":8},
                {"cycle":9,"event":"status_d","data":5}]
        events += [{"cycle":cycle,"event":"state","reset":0,"halted":1,"public_idle":1,"outstanding":0}
                   for cycle in range(10,42)]
        return root,data,case,events

    def execute(self,root,data,case,events):
        (root/"trace.jsonl").write_text("".join(json.dumps(event)+"\n" for event in events))
        return preparation.replay(data,case,root/"trace.jsonl",root/"actual.bin",root/"program.bin")

    def test_physical_replay_accepts_complete_event_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root,data,case,events=self.fixture(directory)
            self.assertTrue(self.execute(root,data,case,events)["expected_equal"])

    def test_high_address_alias_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root,data,case,events=self.fixture(directory)
            events[5]["address"]=0
            with self.assertRaises(AssertionError):self.execute(root,data,case,events)

    def test_source_data_corruption_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root,data,case,events=self.fixture(directory)
            events[5]["bytes"][0]^=1
            with self.assertRaises(AssertionError):self.execute(root,data,case,events)

    def test_wrong_halt_refuses_positive_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root,data,case,events=self.fixture(directory)
            events[10]["data"]=7
            with self.assertRaises(AssertionError):self.execute(root,data,case,events)

    def test_missing_completion_quiet_margin_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root,data,case,events=self.fixture(directory)
            events[-1]["public_idle"]=0
            with self.assertRaises(AssertionError):self.execute(root,data,case,events)


if __name__=="__main__":unittest.main()
