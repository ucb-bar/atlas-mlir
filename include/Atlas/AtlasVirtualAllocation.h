#ifndef ATLAS_VIRTUAL_ALLOCATION_H
#define ATLAS_VIRTUAL_ALLOCATION_H

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
  uint32_t stagingWord;
  unsigned dmaBaseReg;
  unsigned dmaDramReg;
  unsigned dmaSizeReg;
  unsigned scaleReg;
  std::array<unsigned, 2> packSourceRegs;
  unsigned packDestinationReg;
  unsigned packRowReg;
  unsigned packRowsReg;
  std::array<unsigned, 2> packTemporaryRegs;
  unsigned mxuWeightSlot;
  unsigned mxuAccSlot;
};

// The physical location of every virtual value the lowering reads. The
// allocation plan answers with allocated registers; the scheduler answers
// with placeholder registers before allocation exists.
class VirtualPlacement {
public:
  virtual ~VirtualPlacement() = default;
  virtual unsigned tile(Value value) const = 0;
  virtual unsigned fp8(Value value) const = 0;
  virtual unsigned scalar(Value value) const = 0;
  virtual MXUPlacement mxu(Value value) const = 0;
  virtual const DMATransferPlacement &dma(Value value) const = 0;
  virtual const FixedResourcePlacement &fixed() const = 0;
};

enum class RegisterKind { BF16, FP8, Scalar };

// The registers one kind of virtual value may occupy: `count` locations from
// `first`, `stride` apart. A BF16 tile's location is an even/odd pair.
struct RegisterBudget {
  unsigned first;
  unsigned count;
  unsigned stride;

  unsigned reg(unsigned color) const { return first + stride * color; }
};

// With FP8 values present, FP8 tiles keep m0-m31 and BF16 pairs move above
// them; a pack keeps x10-x17 for its relayout loop.
RegisterBudget registerBudget(RegisterKind kind, bool mixedFp8, bool hasPack);

class VirtualAllocationPlan : public VirtualPlacement {
public:
  VirtualAllocationPlan();

  // Place MXU handles and DMA transfers without coloring registers. Requires
  // a verified virtual CFG.
  LogicalResult placeResources(func::FuncOp function);
  // placeResources, then color every register value. Rebuild after changing
  // the CFG's values or order; placement accessors require successful
  // allocation and live source IR. Query only mapped value kinds; placement
  // references/views expire on rebuild.
  LogicalResult allocate(func::FuncOp function);
  LogicalResult verify() const;
  unsigned tile(Value value) const override;
  unsigned fp8(Value value) const override;
  unsigned scalar(Value value) const override;
  MXUPlacement mxu(Value value) const override;
  const DMATransferPlacement &dma(Value value) const override;
  llvm::ArrayRef<int32_t> scalarArguments() const;
  const FixedResourcePlacement &fixed() const override;
  // The function facts that select register budgets.
  bool fp8Present() const { return mixedFp8; }
  bool packPresent() const { return hasPack; }

private:
  LogicalResult colorValues(RegisterKind kind);

  func::FuncOp function;
  llvm::DenseMap<Value, unsigned> tileRegs, fp8Regs, scalarRegs;
  llvm::DenseMap<Value, MXUPlacement> mxuResources;
  llvm::DenseMap<Value, DMATransferPlacement> dmaTransfers;
  llvm::SmallVector<int32_t> scalarArgumentRegs;
  bool mixedFp8 = false;
  bool hasPack = false;
  const FixedResourcePlacement fixedResources;
};

} // namespace mlir::atlas

#endif
