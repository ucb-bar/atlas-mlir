#include "Atlas/AtlasGuardedPack.h"
#include "Atlas/AtlasOps.h"
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
// Facts describe every lane of the exact SSA value. Nonnegative includes +Inf
// and either zero sign; it does NOT imply finite. Unit interval implies both.
// No input samples, SSA names or graph-specific producer list enters the proof.
struct ValueFacts {
  bool finite = false;
  bool nonnegative = false;
  bool unitInterval = false;
  StringRef reason = "unknown producer/domain";
  // Closed interval in retained min/max's total order over raw BF16 bits.
  // Unknown spans negative NaNs through positive NaNs, not merely reals.
  unsigned low = 0, high = 65535;
  void classify() {
    finite = low > 0x007f && high < 0xff80;
    nonnegative = low >= 0x7fff && high <= 0xff80;
    unitInterval = low >= 0x7fff && high <= 0xbf80;
  }
};
unsigned orderedBF16(unsigned bits) {
  return bits ^ ((bits & 0x8000) ? 0xffff : 0x8000);
}

class BF16Facts {
public:
  ValueFacts get(Value value) {
    auto found = facts.find(value);
    if (found != facts.end())
      return found->second;
    ValueFacts result;
    if (!isa<VirtualBF16Type>(value.getType()))
      return result;
    if (auto splat = value.getDefiningOp<VirtualVPUSplatOp>()) {
      unsigned bits = splat.getBits();
      result.low = result.high = orderedBF16(bits);
      result.classify();
      result.reason = "raw BF16 splat classification";
    } else if (auto unary = value.getDefiningOp<VirtualVPUUnaryOp>()) {
      ValueFacts src = get(unary.getSrc());
      if (unary.getKind() == "mov") {
        result = src;
      } else if (unary.getKind() == "relu") {
        // Only the admitted finite source domain is used here. Do not assume
        // anything about ReLU's NaN/Inf behavior from a real-number formula.
        if (src.finite) {
          result.low = std::max(src.low, 0x8000u);
          result.high = std::max(src.high, 0x8000u);
          result.classify();
        }
        result.reason = "finite BF16 ReLU sign clamp";
      }
    } else if (auto binary = value.getDefiningOp<VirtualVPUBinaryOp>()) {
      if (binary.getKind() == "min" || binary.getKind() == "max") {
        ValueFacts lhs = get(binary.getLhs()), rhs = get(binary.getRhs());
        // Total-order interval transfer also admits clamps of unknown values:
        // min(max(x, finiteLo), finiteHi) excludes even signed NaN tails.
        bool minimum = binary.getKind() == "min";
        result.low = minimum ? std::min(lhs.low, rhs.low)
                             : std::max(lhs.low, rhs.low);
        result.high = minimum ? std::min(lhs.high, rhs.high)
                              : std::max(lhs.high, rhs.high);
        result.classify();
        result.reason = "ordered-bit min/max interval transfer";
      }
    }
    else if (auto reduce = value.getDefiningOp<VirtualVPUReduceOp>()) {
      if (reduce.getKind() == "row_max" || reduce.getKind() == "row_min") {
        result = get(reduce.getSrc());
        result.reason = "ordered-bit row extremum selects an input lane";
      }
    } else if (value.getDefiningOp<VirtualMXUMatmulOp>()) {
      result.low = orderedBF16(0xcc00); // -2^25 (conservative closed bound).
      result.high = orderedBF16(0x4c00);
      result.classify();
      result.reason = "32-term raw FP8 reset contraction bounded below 2^25";
    } else if (auto readout = value.getDefiningOp<VirtualMXUReadoutBF16Op>()) {
      Value acc = readout.getAcc();
      if (acc.getDefiningOp<VirtualMXUResetOp>()) {
        result.low = orderedBF16(0xcc00);
        result.high = orderedBF16(0x4c00);
        result.classify();
        result.reason = "32-term raw FP8 reset contraction bounded below 2^25";
      } else if (auto load = acc.getDefiningOp<VirtualMXULoadAccBF16Op>()) {
        result = get(load.getSrc());
        result.reason = "BF16 accumulator copy preserves source domain";
      } else if (acc.getDefiningOp<VirtualMXULoadAccFP8Op>()) {
        result.low = orderedBF16(0xc3e0); // -448
        result.high = orderedBF16(0x43e0); // +448
        result.classify();
        result.reason = "raw FP8 accumulator expansion flushes specials";
      }
    }
    // Finite inputs alone do not prove finite add/mul/exp/reciprocal or an
    // arbitrary BF16 contraction/accumulation. Only the bounded raw FP8 reset
    // cases above qualify; accumulation deliberately stays unknown.
    facts[value] = result;
    return result;
  }
private:
  llvm::DenseMap<Value, ValueFacts> facts;
};

// Parameterized rule: the dialect types carry exactly 32x32 BF16/E4M3 tiles;
// typed verification rejects other dtypes/shapes and scale codes outside u8.
struct PackRule {
  unsigned inverse;
  bool lowerClamp, upperClamp;
  bool derivedFinite;
  ValueFacts source;
};
PackRule selectPackRule(unsigned scale, ValueFacts source) {
  return {254 - std::min(scale, 254u),
          scale <= 246 && !source.nonnegative,
          scale <= 246 && !(source.unitInterval && scale >= 119),
          source.finite, source};
}

struct PackConstants {
  Value low, high, scale;
};

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
  Option<bool> annotateFacts{*this, "annotate-facts",
      llvm::cl::desc("Annotate source facts and their proof/assumption provenance"),
      llvm::cl::init(false)};
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
      BF16Facts facts;
      for (Block &block : function.getBody()) {
        Value state;
        llvm::DenseMap<unsigned, PackConstants> constants;
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
          if (!pack)
            continue;
          PackRule rule = selectPackRule(pack.getScaleCode(), facts.get(pack.getSrc()));
          OpBuilder builder(pack);
          auto proofAttrs = [&]() {
            return SmallVector<NamedAttribute>{
              builder.getNamedAttr("atlas.derived_finite", builder.getBoolAttr(rule.source.finite)),
              builder.getNamedAttr("atlas.derived_nonnegative", builder.getBoolAttr(rule.source.nonnegative)),
              builder.getNamedAttr("atlas.derived_unit_interval", builder.getBoolAttr(rule.source.unitInterval)),
              builder.getNamedAttr("atlas.caller_contract", builder.getBoolAttr(bool(pack.getFiniteSrcAttr()))),
              builder.getNamedAttr("atlas.proof_reason", builder.getStringAttr(rule.source.reason)),
              builder.getNamedAttr("atlas.pack_finite_proof", builder.getStringAttr(
                  rule.derivedFinite ? "derived" : pack.getFiniteSrcAttr() ? "caller_finite_src" : "unproved"))};
          };
          if (annotateFacts)
            for (auto attr : proofAttrs())
              pack->setAttr(attr.getName(), attr.getValue());
          if (!state || liveAcc >= kVirtualMXUSlots ||
              (!pack.getFiniteSrcAttr() && !rule.derivedFinite))
            continue;
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
          auto &values = constants[code];
          unsigned cap = ((code + 8) << 7) | 96;
          if (rule.lowerClamp && !values.low)
            values.low = create("atlas.virtual_vpu_splat", {}, {bf16},
                {builder.getNamedAttr("bits", builder.getI32IntegerAttr(cap | 0x8000))})->getResult(0);
          if (rule.upperClamp && !values.high)
            values.high = create("atlas.virtual_vpu_splat", {}, {bf16},
                {builder.getNamedAttr("bits", builder.getI32IntegerAttr(cap))})->getResult(0);
          if (!values.scale) {
            // The retained signed shift saturates at code 254 (255 aliases it).
            values.scale = create("atlas.virtual_scale_constant", {},
                {VirtualScaleType::get(module.getContext())},
                {builder.getNamedAttr("code", builder.getI32IntegerAttr(rule.inverse))})->getResult(0);
          }
          Value clamped = pack.getSrc();
          if (rule.lowerClamp)
            clamped = create("atlas.virtual_vpu_binary", {clamped, values.low}, {bf16},
                             {builder.getNamedAttr("kind", builder.getStringAttr("max"))})->getResult(0);
          if (rule.upperClamp)
            clamped = create("atlas.virtual_vpu_binary", {clamped, values.high}, {bf16},
                             {builder.getNamedAttr("kind", builder.getStringAttr("min"))})->getResult(0);
          Operation *load = create("atlas.virtual_mxu_load_acc_bf16", {state, clamped},
                                   {token, VirtualMXUAccType::get(module.getContext(), unit)},
                                   {builder.getNamedAttr("unit", builder.getI32IntegerAttr(unit))});
          if (annotateFacts)
            for (auto attr : proofAttrs())
              load->setAttr(attr.getName(), attr.getValue());
          Operation *pop = create("atlas.virtual_mxu_readout_fp8",
                                  {load->getResult(0), load->getResult(1), values.scale},
                                  {token, pack.getResult().getType()});
          state.replaceAllUsesExcept(pop->getResult(0), load);
          state = pop->getResult(0);
          pack.getResult().replaceAllUsesWith(pop->getResult(1));
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
