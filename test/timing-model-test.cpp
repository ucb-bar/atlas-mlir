// Properties of the timing model that its users rely on.

#include "Atlas/AtlasTiming.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>
#include <vector>

using namespace mlir::atlas::timing;

int main() {
  unsigned checks = 0, failures = 0;

  // InOrderIssue::retire forgets an issued instruction once nextFree passes
  // its cycle + doneAge + 1, which is sound only if no dependence reaches
  // further. Every field naming the same register maximizes the overlap
  // between two instructions; 1 and 3 also select second MXU slots.
  std::vector<std::pair<Instr, Footprint>> instrs;
  for (const OpInfo &op : allOps())
    for (int reg : {0, 1, 2, 3})
      for (bool release : {false, true}) {
        if (release && op.opClass != OpClass::Csr)
          continue;
        Instr in{&op, reg, reg, reg, 0, release};
        Footprint f = footprintOf(in, unknownRegs());
        if (f.error.empty())
          instrs.push_back({in, f});
      }
  for (const auto &[a, fa] : instrs)
    for (const auto &[b, fb] : instrs) {
      ++checks;
      Dependence d = dependence(a, fa, b, fb);
      if (d.distance <= fa.doneAge + 1)
        continue;
      if (++failures <= 20)
        llvm::errs() << "FAIL: " << a.op->name << " r" << a.rd << " -> "
                     << b.op->name << " r" << b.rd << ": distance "
                     << d.distance << " exceeds doneAge + 1 = "
                     << fa.doneAge + 1 << " (" << d.reason << ")\n";
    }

  llvm::outs() << checks << " checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
