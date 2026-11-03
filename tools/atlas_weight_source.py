"""Explicit immutable FP8 weight transport, separate from dynamic arena inputs."""
from dataclasses import dataclass


@dataclass(frozen=True)
class WeightSource:
    base: int
    size_bytes: int
    arena_base: int

    @property
    def low(self):
        return self.base & 0xffffffff

    def validate(self, k, n):
        if type(self.base) is not int or self.base < 0 or self.base % 64 or self.base >= 2**64:
            raise ValueError("weight base must be a physical 64-byte-aligned address")
        if type(self.arena_base) is not int or self.arena_base < 0 or self.arena_base % 64 or self.arena_base >= 2**64:
            raise ValueError("arena base must be a physical 64-byte-aligned address")
        # dma.config is global in the model and RTL, not per channel.
        if self.base >> 32 != self.arena_base >> 32:
            raise ValueError("immutable weight source must share the arena upper32 DMA window")
        if type(self.size_bytes) is not int or self.size_bytes != k * n:
            raise ValueError("immutable weight extent must match K*N FP8 bytes")
        if self.low + self.size_bytes > 2**32 or self.base + self.size_bytes > 2**64:
            raise ValueError("immutable weights must fit one DMA low32 window")

    def setup(self, register, channels=(2, 3)):
        # The recipe's existing B pointer is dead here. No extra scratch is owned.
        return [f"li x{register}, {self.base >> 32}",
                *(f"dma.config x{register}, ch{channel}" for channel in channels)]

    def metadata(self, pointer_registers):
        return dict(base=self.base, low32=self.low, high32=self.base >> 32,
                    size_bytes=self.size_bytes, arena_base=self.arena_base, global_dma_upper32_shared=True, access="immutable_read", channels=[2, 3],
                    pointer_registers=list(pointer_registers), extra_scratch_registers=[])
