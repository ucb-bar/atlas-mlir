#include "Atlas/AtlasTarget.h"
#include <cstdio>

int main() {
  using mlir::atlas::atlasDMAMemorySpan;
  struct Case { uint64_t address, bytes; bool accepted; };
  const Case cases[] = {
      {0x40000, 32, false}, {0x07ffffe0, 32, false},
      {0x0800ffe0, 64, false}, {0x08010000, 32, false},
      {0x08000000, 0, false}, {0x90000000, 0, false},
      {0x08000000, 0x10000, true}, {0x0800ffe0, 32, true},
      {0x80000000, 32, true}, {0xffffffe0, 32, true},
      {0xffffffe0, 64, false}, {UINT64_MAX, 2, false},
      // VMEM banks 3-5 on the system bus; banks 0-2 stay the program's own.
      {0x200c0000, 0xc0000, true}, {0x2017ffe0, 32, true},
      {0x200bffe0, 32, false}, {0x200bffe0, 64, false}, {0x20000000, 32, false},
      {0x2017ffe0, 64, false}, {0x20180000, 32, false}};
  for (const auto &test : cases)
    if (atlasDMAMemorySpan(test.address, test.bytes) != test.accepted) {
      std::fprintf(stderr, "DMA span check failed at %llx\n",
                   static_cast<unsigned long long>(test.address));
      return 1;
    }
  return 0;
}
