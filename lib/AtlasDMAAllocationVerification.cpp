#include "Atlas/AtlasDMAAllocationVerification.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace mlir;
using namespace mlir::atlas;

namespace {
// Selected RTL: ScalarCore.scala AtlasMemMap, DmaParams.scala defaults, and
// LSU.scala VLOAD/VSTORE assertions. DMA bases are VMEM word indices, as wired
// by AtlasCore.scala; every staging half is one aligned 1 KiB tensor transfer.
constexpr uint64_t kVMEMWords = 0x180000 / 4;
constexpr unsigned kDMAChannels = 8;
constexpr unsigned kHalfWords = 1024 / 4;

bool isLaunch(Operation *op) {
  return isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op,
             VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op);
}

class DMAAllocationVerifier {
public:
  DMAAllocationVerifier(func::FuncOp function,
                        ArrayRef<VirtualDMAAssignment> assignments)
      : function(function), assignments(assignments), assemblyState(function) {}

  LogicalResult verify() {
    known = sourceValues(function);
    for (Block &block : function.getBody())
      for (Operation &op : block)
        if (isLaunch(&op))
          transfers.push_back(op.getResult(0));

    llvm::DenseSet<unsigned> transferIds;
    for (const VirtualDMAAssignment &assignment : assignments) {
      Value transfer = assignment.transfer;
      // Establish membership before dereferencing caller-supplied Values.
      if (!transfer || !known.contains(transfer))
        return function.emitOpError("DMA assignment refers to a foreign, stale, or untracked value");
      Operation *launch = transfer.getDefiningOp();
      if (!launch || !isLaunch(launch) || launch->getResult(0) != transfer)
        return assignmentError(transfer, "DMA assignment does not refer to a DMA operation's state result");
      if (placements.contains(transfer))
        return assignmentError(transfer, "duplicate DMA assignment");
      if (failed(verifyGeometry(transfer, assignment.placement)))
        return failure();
      if (!transferIds.insert(assignment.placement.id).second)
        return assignmentError(transfer, "duplicate DMA transfer id across static launches");
      placements[transfer] = &assignment.placement;
    }
    for (Value transfer : transfers)
      if (!placements.contains(transfer))
        return assignmentError(transfer, "missing DMA assignment");

    // Atomic virtual DMA has no pending interval here. insert-atlas-dma-waits
    // completes a transfer before its channel or staging window is reused,
    // and the generated DMA checks verify every overlap in the timed stream.
    return success();
  }

private:
  void noteTransfer(InFlightDiagnostic &diagnostic, Value transfer,
                    StringRef label) {
    std::string operand;
    llvm::raw_string_ostream stream(operand);
    transfer.printAsOperand(stream, assemblyState);
    auto &note = diagnostic.attachNote(transfer.getLoc());
    note << label << " " << stream.str();
    if (const DMATransferPlacement *placement = placements.lookup(transfer)) {
      uint64_t end = uint64_t(placement->stagingWord) + placement->halves * kHalfWords;
      note << "; channel " << placement->channel << ", VMEM words ["
           << placement->stagingWord << ", " << end << ")";
    }
  }

  LogicalResult assignmentError(Value transfer, StringRef message) {
    auto diagnostic = function.emitOpError(message);
    noteTransfer(diagnostic, transfer, "assigned transfer");
    return failure();
  }

  LogicalResult verifyGeometry(Value transfer,
                               const DMATransferPlacement &placement) {
    Operation *launch = transfer.getDefiningOp();
    unsigned halves = isa<VirtualDMALoadBF16Op, VirtualDMAStoreBF16Op>(launch) ? 2 : 1;
    if (placement.halves != halves)
      return assignmentError(transfer, "DMA placement halves do not match the transfer format");
    if (placement.channel >= kDMAChannels)
      return assignmentError(transfer, "DMA channel is outside [0, 7]");
    if (placement.id > 0x7FFFFFFFU)
      return assignmentError(transfer, "DMA transfer id must fit a nonnegative i32");
    if (placement.stagingWord % kHalfWords)
      return assignmentError(transfer, "DMA staging word address must be aligned to 1 KiB");
    uint64_t end = uint64_t(placement.stagingWord) + halves * kHalfWords;
    if (end > kVMEMWords)
      return assignmentError(transfer, "DMA staging range exceeds selected VMEM capacity");
    for (unsigned reg : {placement.stagingReg, placement.dramReg, placement.sizeReg})
      if (reg == 0 || reg >= 32)
        return assignmentError(transfer, "DMA helper register must be in x1..x31");
    if (placement.stagingReg == placement.dramReg || placement.stagingReg == placement.sizeReg || placement.dramReg == placement.sizeReg)
      return assignmentError(transfer, "DMA helper registers must be distinct within a transfer");
    return success();
  }

  func::FuncOp function;
  ArrayRef<VirtualDMAAssignment> assignments;
  llvm::DenseSet<Value> known;
  SmallVector<Value> transfers;
  llvm::DenseMap<Value, const DMATransferPlacement *> placements;
  AsmState assemblyState;
};
} // namespace

LogicalResult mlir::atlas::verifyAtlasDMAAllocation(
    func::FuncOp function, ArrayRef<VirtualDMAAssignment> assignments) {
  return DMAAllocationVerifier(function, assignments).verify();
}
