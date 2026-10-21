#include "Atlas/AtlasBoundaryABI.h"
#include "Atlas/AtlasOps.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::atlas;

static uint64_t slotIndex(Operation *op) {
  return op->getAttrOfType<IntegerAttr>("index").getValue().getZExtValue();
}

bool mlir::atlas::hasAtlasBoundaryTiles(func::FuncOp function) {
  return function
      .walk([](Operation *op) {
        return isa<VirtualInputFP8Op, VirtualInputBF16Op, VirtualOutputBF16Op>(op)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

FailureOr<AtlasBoundaryABI> AtlasBoundaryABI::read(func::FuncOp function,
                                                   bool diagnose) {
  auto error = [&](const char *message) -> FailureOr<AtlasBoundaryABI> {
    if (diagnose)
      function.emitOpError(message);
    return failure();
  };
  auto input = function->getAttrOfType<IntegerAttr>("atlas.input_dram_base");
  auto output = function->getAttrOfType<IntegerAttr>("atlas.output_dram_base");
  if (!input || !output)
    return error("DRAM boundary ABI requires explicit atlas.input_dram_base and "
                 "atlas.output_dram_base");
  int64_t in = input.getValue().getSExtValue();
  int64_t out = output.getValue().getSExtValue();
  auto aligned32 = [](int64_t base) {
    return base >= 0x80000000LL && base < (1LL << 32) &&
           base % kAtlasRegisterBytes == 0;
  };
  if (!aligned32(in) || !aligned32(out))
    return error("DRAM bases must be aligned 32-bit selected-memory addresses");
  AtlasBoundaryABI abi;
  abi.inputBase = uint64_t(in);
  abi.outputBase = uint64_t(out);
  if (Attribute raw = function->getAttr("atlas.output_dram_bytes")) {
    auto bytes = dyn_cast<IntegerAttr>(raw);
    if (!bytes || !bytes.getType().isSignlessInteger(64) ||
        bytes.getInt() <= 0 || bytes.getInt() % 32)
      return error("output DRAM extent must be positive 32-byte-aligned i64 bytes");
    abi.outputBytes = uint64_t(bytes.getInt());
  }
  function.walk([&](Operation *op) {
    if (isa<VirtualInputFP8Op, VirtualInputBF16Op>(op))
      abi.inputSlots = std::max(abi.inputSlots, slotIndex(op) + 1);
    else if (isa<VirtualOutputBF16Op>(op))
      abi.outputSlots = std::max(abi.outputSlots, slotIndex(op) + 1);
  });
  return abi;
}

LogicalResult AtlasBoundaryABI::verifySpans(func::FuncOp function,
                                            bool diagnose) const {
  if (inputEnd() <= (1ULL << 32) && outputEnd() <= (1ULL << 32) &&
      (inputBase >= outputEnd() || outputBase >= inputEnd()))
    return success();
  if (diagnose)
    function.emitOpError("input and output DRAM spans overlap or overflow");
  return failure();
}
