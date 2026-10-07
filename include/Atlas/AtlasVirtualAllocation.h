#ifndef ATLAS_VIRTUAL_ALLOCATION_H
#define ATLAS_VIRTUAL_ALLOCATION_H

#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <array>
#include <cstdint>

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

// One explicit DMA transfer in flight: a channel per direction, a staging
// window, and the registers the DMA reads when it completes.
struct DMASlotPlacement {
  unsigned loadChannel;
  unsigned storeChannel;
  uint32_t stagingWord;
  unsigned baseReg;
  unsigned dramReg;
  unsigned sizeReg;
};

struct FixedResourcePlacement {
  unsigned tensorTemporary;
  unsigned scalarTemporary;
  unsigned oneReg;
  unsigned zeroReg;
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
  std::array<DMASlotPlacement, kMaxPendingVirtualDMA> dmaSlots;
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

// How many values of `kind` can hold registers at once in `function`, as
// allocate colors them: BF16 pairs, FP8 registers, or scalars.
unsigned registerCapacity(func::FuncOp function, RegisterKind kind);

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
  // The channels the function's explicit transfers use, in first use order.
  llvm::ArrayRef<unsigned> dmaChannels() const { return usedDMAChannels; }

private:
  LogicalResult colorValues(RegisterKind kind);
  LogicalResult placeMXU(Block &block);
  LogicalResult placeDMA(Block &block, unsigned &nextTransfer);

  func::FuncOp function;
  llvm::DenseMap<Value, unsigned> tileRegs, fp8Regs, scalarRegs;
  llvm::DenseMap<Value, MXUPlacement> mxuResources;
  llvm::DenseMap<Value, DMATransferPlacement> dmaTransfers;
  llvm::SmallVector<int32_t> scalarArgumentRegs;
  bool mixedFp8 = false;
  bool hasPack = false;
  llvm::SmallVector<unsigned, 4> usedDMAChannels;
  const FixedResourcePlacement fixedResources;
};

} // namespace mlir::atlas

#endif
