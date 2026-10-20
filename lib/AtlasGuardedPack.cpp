#include "Atlas/AtlasGuardedPack.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::atlas;

namespace {
// Deliberately small, NaN-safe proof rules. In particular, finite operands do
// not prove finite products/sums, reciprocals, exponentials or accumulations.
bool provenFinite(Value value, llvm::DenseMap<Value, bool> &facts) {
  auto found = facts.find(value);
  if (found != facts.end())
    return found->second;
  bool finite = false;
  if (auto splat = value.getDefiningOp<VirtualVPUSplatOp>())
    finite = (splat.getBits() & 0x7f80) != 0x7f80;
  else if (value.getDefiningOp<VirtualMXUMatmulOp>())
    finite = true;
  else if (auto readout = value.getDefiningOp<VirtualMXUReadoutBF16Op>()) {
    Value acc = readout.getAcc();
    // Both MXUs interpret every raw E4M3 code as bounded integer fields,
    // including exponent 15. A reset contracts exactly 32 terms: each product
    // is below 2^18 and the absolute sum, including BF16 rounding, is below
    // 2^25. No BF16 seed or subsequent accumulation is covered by this rule.
    if (acc.getDefiningOp<VirtualMXUResetOp>())
      finite = true;
    else if (auto copy = acc.getDefiningOp<VirtualMXULoadAccBF16Op>())
      finite = provenFinite(copy.getSrc(), facts);
  } else if (auto unary = value.getDefiningOp<VirtualVPUUnaryOp>())
    finite = (unary.getKind() == "mov" || unary.getKind() == "relu") &&
             provenFinite(unary.getSrc(), facts);
  else if (auto binary = value.getDefiningOp<VirtualVPUBinaryOp>())
    finite = (binary.getKind() == "min" || binary.getKind() == "max") &&
             provenFinite(binary.getLhs(), facts) &&
             provenFinite(binary.getRhs(), facts);
  facts[value] = finite;
  return finite;
}

struct LowerAtlasGuardedPackPass
    : PassWrapper<LowerAtlasGuardedPackPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAtlasGuardedPackPass)
  LowerAtlasGuardedPackPass() = default;
  LowerAtlasGuardedPackPass(const LowerAtlasGuardedPackPass &other)
      : PassWrapper(other) {}
  StringRef getArgument() const final { return "lower-atlas-guarded-pack"; }
  StringRef getDescription() const final {
    return "Opt in to exact finite BF16 PACK via MXU copy/pop";
  }
  Option<unsigned> unit{*this, "unit", llvm::cl::desc("Conversion MXU (0 or 1)"),
                        llvm::cl::init(0)};
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (unit > 1) {
      module.emitError("guarded PACK conversion unit must be 0 or 1");
      return signalPassFailure();
    }
    if (failed(verifyAtlasVirtualModule(module)))
      return signalPassFailure();
    for (auto function : module.getOps<func::FuncOp>()) {
      // Limit conversion to a single straight-line resource lifetime.
      // The verifier admits only i1/i32 entry arguments, so accumulators
      // cannot enter the block without an operation counted below.
      if (!llvm::hasSingleElement(function.getBody()))
        continue;
      llvm::DenseMap<Value, bool> facts;
      for (Block &block : function.getBody()) {
        Value state;
        unsigned liveAcc = 0;
        for (Operation &op : llvm::make_early_inc_range(block)) {
          // Existing resource handles remain authoritative. A conversion
          // cannot borrow a slot while both selected-unit accumulators are live.
          Value createdAcc;
          if (auto load = dyn_cast<VirtualMXULoadAccFP8Op>(op))
            createdAcc = load.getAcc();
          else if (auto load = dyn_cast<VirtualMXULoadAccBF16Op>(op))
            createdAcc = load.getAcc();
          else if (auto reset = dyn_cast<VirtualMXUResetOp>(op))
            createdAcc = reset.getAcc();
          if (createdAcc && cast<VirtualMXUAccType>(createdAcc.getType()).getUnit() == unit)
            ++liveAcc;
          if (isa<VirtualMXUReadoutBF16Op, VirtualMXUReadoutFP8Op>(op) &&
              cast<VirtualMXUAccType>(virtualMXUReadoutAcc(&op).getType()).getUnit() == unit)
            --liveAcc;
          if (op.getNumResults() &&
              isa<VirtualStateType>(op.getResult(0).getType()))
            state = op.getResult(0);
          auto pack = dyn_cast<VirtualPackFP8Op>(op);
          if (!pack || !state ||
              liveAcc >= kVirtualMXUSlots ||
              !provenFinite(pack.getSrc(), facts))
            continue;
          OpBuilder builder(pack);
          Location loc = pack.getLoc();
          Type bf16 = VirtualBF16Type::get(module.getContext());
          Type token = VirtualStateType::get(module.getContext());
          unsigned code = pack.getScaleCode();
          // Rematerialize bounds at each use. Block-wide sharing extends every
          // scale's two BF16 tiles across unrelated conversions and can exhaust
          // the mixed BF16/FP8 register partition.
          Value low, high, scale;
          if (code <= 246) {
            unsigned cap = ((code + 8) << 7) | 96;
            low = VirtualVPUSplatOp::create(builder, loc, bf16, cap | 0x8000);
            high = VirtualVPUSplatOp::create(builder, loc, bf16, cap);
          }
          // The retained signed shift saturates at code 254 (255 aliases it).
          unsigned inverse = 254 - std::min(code, 254u);
          scale = VirtualScaleConstantOp::create(
              builder, loc, VirtualScaleType::get(module.getContext()), inverse);
          Value clamped = pack.getSrc();
          if (code <= 246) {
            clamped = VirtualVPUBinaryOp::create(builder, loc, bf16, clamped, low, "max");
            clamped = VirtualVPUBinaryOp::create(builder, loc, bf16, clamped, high, "min");
          }
          auto load = VirtualMXULoadAccBF16Op::create(
              builder, loc, token, VirtualMXUAccType::get(module.getContext(), unit),
              state, clamped, unit);
          auto pop = VirtualMXUReadoutFP8Op::create(
              builder, loc, token, pack.getResult().getType(),
              load.getNext(), load.getAcc(), scale);
          state.replaceAllUsesExcept(pop.getNext(), load);
          state = pop.getNext();
          pack.getResult().replaceAllUsesWith(pop.getValue());
          pack.erase();
        }
      }
    }
    if (failed(verifyAtlasVirtualModule(module)))
      signalPassFailure();
  }
};
} // namespace

void mlir::atlas::registerLowerAtlasGuardedPackPass() {
  PassRegistration<LowerAtlasGuardedPackPass>();
}

std::unique_ptr<Pass> mlir::atlas::createLowerAtlasGuardedPackPass() {
  return std::make_unique<LowerAtlasGuardedPackPass>();
}
