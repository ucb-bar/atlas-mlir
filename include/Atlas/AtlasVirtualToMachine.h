#ifndef ATLAS_VIRTUAL_TO_MACHINE_H
#define ATLAS_VIRTUAL_TO_MACHINE_H

#include "Atlas/AtlasVirtualAllocation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>
#include <string>

namespace mlir::atlas {

// The DRAM bindings of one lowered function.
struct VirtualLoweringABI {
  uint64_t inputBase = 0;
  uint64_t outputBase = 0;
  std::optional<uint64_t> controlBase;
};

// Read a function's DRAM bindings and check them against the lowering's VMEM
// windows. Emits a diagnostic on the function when they are missing or bad.
FailureOr<VirtualLoweringABI>
readVirtualLoweringABI(func::FuncOp function,
                       const FixedResourcePlacement &fixed);

// One machine operation of the lowered stream. A redirect names its target
// by label; the label's position becomes the encoded offset.
struct MachineStep {
  std::string name;
  SmallVector<NamedAttribute> attrs;
  Location loc;
  std::optional<unsigned> targetLabel;
};

// Receives the machine steps and labels of lowered virtual operations.
class MachineStepSink {
public:
  virtual ~MachineStepSink() = default;
  virtual void add(MachineStep step) = 0;
  virtual unsigned newLabel() = 0;
  // The next added step is the label's target.
  virtual void mark(unsigned label) = 0;
  virtual unsigned labelFor(Block *block) = 0;
};

struct VirtualLoweringOptions {
  // Follow asynchronous instructions with the fixed diagnostic DELAYs that
  // --verify-atlas-generated-schedule requires.
  bool diagnosticDelays = true;
};

// The steps every lowered program starts with: constant registers, DMA
// channel bases, and staged scalar arguments.
void lowerVirtualPrologue(func::FuncOp function,
                          const VirtualPlacement &placement,
                          const VirtualLoweringABI &abi,
                          VirtualLoweringOptions options,
                          MachineStepSink &sink);

// Lower one operation of a verified virtual function. Emits a diagnostic on
// the operation when it has no lowering.
LogicalResult lowerVirtualOperation(Operation &op,
                                    const VirtualPlacement &placement,
                                    const VirtualLoweringABI &abi,
                                    VirtualLoweringOptions options,
                                    MachineStepSink &sink);

// Replace a verified single-function virtual module with its machine stream.
LogicalResult lowerAtlasVirtualModule(ModuleOp module);

void registerLowerAtlasVirtualToMachinePass();

} // namespace mlir::atlas

#endif
