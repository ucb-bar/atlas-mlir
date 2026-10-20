#ifndef ATLAS_VIRTUAL_ALLOCATION_H
#define ATLAS_VIRTUAL_ALLOCATION_H

#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir::atlas {

struct MXUPlacement {
  unsigned unit;
  unsigned slot;
};

struct DMATransferPlacement {
  unsigned channel;
  unsigned halves;
  unsigned id;
  uint32_t stagingWord;
  unsigned stagingReg;
  unsigned dramReg;
  unsigned sizeReg;
};

// A PACK relayout row is 32 FP8 bytes: two 16-byte halves of four words,
// which lowerPack copies one word at a time. It is fixed by the selected PACK
// layout, not a parameter: lowerPack's loop is written for exactly this row.
constexpr unsigned kPackRelayoutRowWords = 8;

struct FixedResourcePlacement {
  unsigned tensorTemporary;
  unsigned scalarTemporary;
  unsigned halfSizeReg;
  unsigned haltReg;
  unsigned inputBaseReg;
  unsigned inputDramReg;
  unsigned outputBaseReg;
  unsigned outputDramReg;
  unsigned loadChannel;
  unsigned storeChannel;
  uint32_t inputWord;
  uint32_t outputWord;
  uint32_t mailboxWord;
  uint32_t inputWindowWords;
  uint32_t outputWindowWords;
  uint32_t packWord;
  uint32_t packRelayoutWord;
  uint32_t stagingWord;
  unsigned dmaBaseReg;
  unsigned dmaDramReg;
  unsigned dmaSizeReg;
  uint32_t spillWord;
  uint32_t spillWindowWords;
  unsigned scaleReg;
  std::array<unsigned, 2> packSourceRegs;
  unsigned packDestinationReg;
  unsigned packRowReg;
  unsigned packRowsReg;
  std::array<unsigned, 2> packTemporaryRegs;
  unsigned mxuWeightSlot;
  unsigned mxuAccSlot;
};

enum class RegisterKind { BF16, FP8, Scalar };
using RegisterCapacities = std::array<unsigned, 3>;

// The registers `value` takes once allocated: a pair for a BF16 tile, one
// tensor register for an FP8 tile, one scalar register for an i1/i32 control.
// None for a DMA immediate, which each launch builds (isVirtualDMAImmediate),
// a memory-resident entry control (isAtlasMemoryControl), or any other type.
// A loaded tile takes them from where it is staged (stagingPoint).
std::optional<RegisterKind> registerKindOf(Value value);

// One RegisterKind's demand in the last allocate(), against its capacity.
struct RegisterPressure {
  unsigned capacity = 0;
  // Peak simultaneous demand under the no-operand-overwrite policy, and an
  // operation where that many values need registers.
  unsigned peak = 0;
  Operation *peakAt = nullptr;
  // The registers the coloring used; empty when it did not fit.
  std::optional<unsigned> used;
};

// How many values of each RegisterKind allocate can keep in registers at once
// in `function`: BF16 pairs, FP8 registers and scalars.
RegisterCapacities registerCapacities(func::FuncOp function);

// `loc` as file:line:col, without the file's directory, where it has one,
// else as MLIR prints it.
std::string locationText(Location loc);

class VirtualAllocationPlan {
public:
  VirtualAllocationPlan();

  // Requires a verified virtual CFG. Rebuild after changing its values or order;
  // placement accessors require successful allocation and live source IR.
  // Query only mapped value kinds; placement references/views expire on rebuild.
  LogicalResult allocate(func::FuncOp function);
  LogicalResult verify() const;
  unsigned tile(Value value) const;
  unsigned fp8(Value value) const;
  unsigned scalar(Value value) const;
  MXUPlacement mxu(Value value) const;
  const DMATransferPlacement &dma(Value value) const;
  llvm::ArrayRef<int32_t> scalarArguments() const;
  const FixedResourcePlacement &fixed() const;
  // After allocate(), whether or not it succeeded: register pressure per kind,
  // each value's placement, MXU slots, DMA channels and the VMEM windows the
  // lowering uses. Values are named as the function prints.
  llvm::json::Object report() const;
  // The operation before which lowering stages the loaded tile `tile` into
  // its registers, if any; query after allocate().
  Operation *stagingPoint(Value tile) const;
  // Typed feedback for pressure recovery; query after allocate().
  const RegisterPressure &registerPressure(RegisterKind kind) const {
    return pressure[static_cast<unsigned>(kind)];
  }

private:
  LogicalResult colorValues(RegisterKind kind);
  LogicalResult placeMXU(Block &block);
  LogicalResult placeDMA(Block &block, unsigned &nextTransfer);
  void stageLoads();

  func::FuncOp function;
  // The loaded tiles whose staging VLOADs run right before each operation.
  llvm::DenseMap<Operation *, llvm::SmallVector<Value, 2>> stagedBefore;
  llvm::DenseMap<Value, unsigned> tileRegs, fp8Regs, scalarRegs;
  llvm::DenseMap<Value, MXUPlacement> mxuResources;
  llvm::DenseMap<Value, DMATransferPlacement> dmaTransfers;
  llvm::SmallVector<int32_t> scalarArgumentRegs;
  std::array<RegisterPressure, 3> pressure;
  bool mixedFp8 = false;
  bool hasPack = false;
  const FixedResourcePlacement fixedResources;
};

} // namespace mlir::atlas

#endif
