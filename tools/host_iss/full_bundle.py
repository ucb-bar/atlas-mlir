"""Shared immutable weights and exact per-launch images for full-model hosts."""

import hashlib
import json
import re
import struct

from atlas_host_layout import fp8_output_offset, ATLAS_DRAM_BASE, ATLAS_IMEM_BYTES
from block_bundle import integer
from bundle import ChecksumBundle


class KernelAssets(ChecksumBundle):
    """Checksum-bound kernel sources only; never host/live admission."""

    def __init__(self, root, workload):
        self._metadata(root, workload, "kernels.json", "atlas.full_kernel_assets.v1")
        self.preparation_only = True
        self._validate_kernels()

    def _metadata(self, root, workload, filename, schema):
        super().__init__(root)
        self.metadata_name = filename
        self.provenance = json.loads(self.read(filename))
        if (workload not in ("tiny-full", "smol-full")
                or self.provenance.get("schema") != schema
                or self.provenance.get("workload") != workload):
            raise ValueError("unsupported full-model bundle")
        self.workload = workload
        self.kernel_count = integer(self.provenance.get("kernel_count"), 1)
        if self.kernel_count > 20000:
            raise ValueError("full-model launch count exceeds bound")
        self.kernels = self.provenance.get("kernels")
        if not isinstance(self.kernels, list) or len(self.kernels) != self.kernel_count:
            raise ValueError("full-model launch count differs")

    def _validate_kernels(self):
        for index, descriptor in enumerate(self.kernels):
            if (not isinstance(descriptor, dict) or type(descriptor.get("index")) is not int
                    or descriptor["index"] != index):
                raise ValueError("full-model launch descriptor order differs")
            self.validate_kernel(descriptor)
            self.initial_dram(index)

    def validate_kernel(self, item):
        if not isinstance(item, dict):
            raise ValueError("invalid full-model launch descriptor")
        for key in ("source", "original", "activation", "weights", "expected"):
            if not isinstance(item.get(key), str) or item[key] not in self.sums:
                raise ValueError("unlisted full-model kernel payload: " + key)
        dimensions = item.get("physical_mkn")
        if (not isinstance(dimensions, list) or len(dimensions) != 3
                or any(type(value) is not int or value <= 0 or value % 32 for value in dimensions)):
            raise ValueError("full-model physical dimensions must be positive tile multiples")
        m, k, n = dimensions
        offset = integer(item.get("output_offset"))
        total = integer(item.get("total_dram_bytes"), 1)
        if (item.get("dram_base") != ATLAS_DRAM_BASE
                or type(item.get("dram_base")) is not int
                or offset != fp8_output_offset(m, k, n)
                or total != offset + 2 * m * n + 256
                or total > 64 * 1024 * 1024):
            raise ValueError("full-model staging extent differs")
        if not isinstance(item.get("initial_sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", item["initial_sha256"]):
            raise ValueError("missing full-model initial image identity")
        original = self.read(item["original"])
        if (not original or len(original) % 4 or len(original) + 20 > ATLAS_IMEM_BYTES
                or original[-4:] != struct.pack("<I", 0x73)):
            raise ValueError("full-model program must fit IMEM and end in ECALL")
        for key, size in (("activation", m * k), ("weights", k * n), ("expected", 2 * m * n)):
            if len(self.read(item[key])) != size:
                raise ValueError("full-model payload extent differs: " + key)
        requested = item.get("requested_dma")
        if (not isinstance(requested, dict) or not {"load_bytes", "store_bytes"} <= requested.keys()
                or requested.keys() - {"load_bytes", "store_bytes", "response_beats"}):
            raise ValueError("invalid full-model requested DMA counters")
        for value in requested.values():
            integer(value)
        if requested["store_bytes"] != 2 * m * n:
            raise ValueError("full-model DMA store extent differs")

    def kernel_paths(self, index):
        integer(index)
        if index >= self.kernel_count:
            raise ValueError("full-model launch index exceeds contract")
        return dict(self.kernels[index], physical_mkn=list(self.kernels[index]["physical_mkn"]),
                    requested_dma=dict(self.kernels[index]["requested_dma"]))

    def initial_dram(self, index):
        item = self.kernel_paths(index)
        m, _, n = item["physical_mkn"]
        initial = (self.read(item["activation"]) + self.read(item["weights"])
                   + b"\xa5" * 256 + b"\xcd" * (2 * m * n) + b"\x5a" * 256)
        if (len(initial) != item["total_dram_bytes"]
                or hashlib.sha256(initial).hexdigest() != item["initial_sha256"]):
            raise ValueError("full-model reconstructed initial image differs")
        return initial


class FullBundle(KernelAssets):
    def __init__(self, root, workload):
        self._metadata(root, workload, "provenance.json", "atlas.full_bundle.v1")
        self.build = json.loads(self.read("host/build.json"))
        for name in ("host/host.elf", self.provenance.get("contract")):
            if not isinstance(name, str) or name not in self.sums:
                raise ValueError("unlisted full-model ELF or contract")
        contract = self.provenance["contract"]
        if self.sums[contract] != self.provenance.get("contract_sha256"):
            raise ValueError("full-model contract identity differs")
        self._validate_kernels()
