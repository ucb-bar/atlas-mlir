#include "Atlas/AtlasGuardedPack.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::atlas;

namespace {
// Deliberately small, NaN-safe proof rules. In particular, finite operands do
// not prove finite products/sums, reciprocals, exponentials or contractions.
bool provenFinite(Value value, llvm::DenseMap<Value, bool> &facts) {
  auto found = facts.find(value);
  if (found != facts.end())
    return found->second;
  bool finite = false;
  if (auto splat = value.getDefiningOp<VirtualVPUSplatOp>())
    finite = (splat.getBits() & 0x7f80) != 0x7f80;
  else if (auto unary = value.getDefiningOp<VirtualVPUUnaryOp>())
    finite = (unary.getKind() == "mov" || unary.getKind() == "relu") &&
             provenFinite(unary.getSrc(), facts);
  else if (auto binary = value.getDefiningOp<VirtualVPUBinaryOp>())
    finite = (binary.getKind() == "min" || binary.getKind() == "max") &&
             provenFinite(binary.getLhs(), facts) &&
             provenFinite(binary.getRhs(), facts);
  facts[value] = finite;
  return finite;
}

// Use the actual allocator, including fixed reservations and coloring, rather
// than a live-value estimate. Failed speculative allocation emits no error.
bool allocates(func::FuncOp function) {
  ScopedDiagnosticHandler handler(function.getContext(), [](Diagnostic &) {
    return success();
  });
  VirtualAllocationPlan plan;
  return succeeded(plan.allocate(function)) && succeeded(plan.verify());
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
      // Keep this proposal within a single straight-line resource lifetime.
      // The verifier admits only i1/i32 entry arguments, so accumulators
      // cannot enter the block without an operation counted below.
      if (!llvm::hasSingleElement(function.getBody()))
        continue;
      OwningOpRef<func::FuncOp> original(cast<func::FuncOp>(function->clone()));
      bool changed = false;
      llvm::DenseMap<Value, bool> facts;
      for (Block &block : function.getBody()) {
        Value state;
        unsigned liveAcc = 0;
        for (Operation &op : llvm::make_early_inc_range(block)) {
          // Existing resource handles remain authoritative. A conversion
          // cannot borrow a slot while both selected-unit accumulators are live.
          if (isa<VirtualMXULoadAccFP8Op, VirtualMXULoadAccBF16Op,
                  VirtualMXUResetOp>(op) &&
              cast<VirtualMXUAccType>(op.getResult(1).getType()).getUnit() == unit)
            ++liveAcc;
          if (isa<VirtualMXUReadoutBF16Op, VirtualMXUReadoutFP8Op>(op) &&
              cast<VirtualMXUAccType>(op.getOperand(1).getType()).getUnit() == unit)
            --liveAcc;
          if (op.getNumResults() &&
              isa<VirtualStateType>(op.getResult(0).getType()))
            state = op.getResult(0);
          auto pack = dyn_cast<VirtualPackFP8Op>(op);
          if (!pack || !state ||
              liveAcc >= kVirtualMXUSlots ||
              (!pack.getFiniteSrcAttr() && !provenFinite(pack.getSrc(), facts)))
            continue;
          OpBuilder builder(pack);
          Location loc = pack.getLoc();
          auto create = [&](StringRef name, ValueRange operands,
                            TypeRange types, ArrayRef<NamedAttribute> attrs = {}) {
            OperationState operation(loc, name);
            operation.addOperands(operands);
            operation.addTypes(types);
            operation.addAttributes(attrs);
            return builder.create(operation);
          };
          Type bf16 = VirtualBF16Type::get(module.getContext());
          Type token = VirtualStateType::get(module.getContext());
          unsigned code = pack.getScaleCode();
          // Rematerialize bounds at each use. Block-wide sharing extends every
          // scale's two BF16 tiles across unrelated conversions and can exhaust
          // the mixed BF16/FP8 register partition.
          Value low, high, scale;
          if (code <= 246) {
            unsigned cap = ((code + 8) << 7) | 96;
            low = create("atlas.virtual_vpu_splat", {}, {bf16},
                         {builder.getNamedAttr("bits", builder.getI32IntegerAttr(cap | 0x8000))})->getResult(0);
            high = create("atlas.virtual_vpu_splat", {}, {bf16},
                          {builder.getNamedAttr("bits", builder.getI32IntegerAttr(cap))})->getResult(0);
          }
          // The retained signed shift saturates at code 254 (255 aliases it).
          unsigned inverse = 254 - std::min(code, 254u);
          scale = create("atlas.virtual_scale_constant", {},
                         {VirtualScaleType::get(module.getContext())},
                         {builder.getNamedAttr("code", builder.getI32IntegerAttr(inverse))})->getResult(0);
          Value clamped = pack.getSrc();
          if (code <= 246) {
            clamped = create("atlas.virtual_vpu_binary", {clamped, low}, {bf16},
                             {builder.getNamedAttr("kind", builder.getStringAttr("max"))})->getResult(0);
            clamped = create("atlas.virtual_vpu_binary", {clamped, high}, {bf16},
                             {builder.getNamedAttr("kind", builder.getStringAttr("min"))})->getResult(0);
          }
          Operation *load = create("atlas.virtual_mxu_load_acc_bf16", {state, clamped},
                                   {token, VirtualMXUAccType::get(module.getContext(), unit)},
                                   {builder.getNamedAttr("unit", builder.getI32IntegerAttr(unit))});
          Operation *pop = create("atlas.virtual_mxu_readout_fp8",
                                  {load->getResult(0), load->getResult(1), scale},
                                  {token, pack.getResult().getType()});
          state.replaceAllUsesExcept(pop->getResult(0), load);
          state = pop->getResult(0);
          pack.getResult().replaceAllUsesWith(pop->getResult(1));
          pack.erase();
          changed = true;
        }
      }
      // Keep the native body if this rewrite introduces an allocation failure.
      // Whole-function rollback preserves all state and SSA uses together. If
      // neither body allocates, retain the candidate for the usual diagnostic.
      if (changed && !allocates(function) && allocates(*original)) {
        function.getBody().takeBody(original->getBody());
        function.emitRemark("guarded PACK kept native body to preserve allocation");
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
