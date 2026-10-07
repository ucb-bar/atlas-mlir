#include "Atlas/AtlasVirtualScheduleTrace.h"
#include "Atlas/AtlasOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace mlir::atlas;

std::string mlir::atlas::describeVirtualOperation(Operation *op) {
  StringRef name = op->getName().getStringRef();
  name.consume_front("atlas.virtual_");
  name.consume_front("atlas.");
  std::string out = name.str();
  auto unitOf = [](Type type) -> std::optional<unsigned> {
    if (auto weight = dyn_cast<VirtualMXUWeightType>(type))
      return weight.getUnit();
    if (auto acc = dyn_cast<VirtualMXUAccType>(type))
      return acc.getUnit();
    return std::nullopt;
  };
  if (auto unary = dyn_cast<VirtualVPUUnaryOp>(op))
    out += " " + unary.getKind().str();
  else if (auto binary = dyn_cast<VirtualVPUBinaryOp>(op))
    out += " " + binary.getKind().str();
  else if (auto input = dyn_cast<VirtualInputBF16Op>(op))
    out += " #" + std::to_string(input.getIndex());
  else if (auto input = dyn_cast<VirtualInputFP8Op>(op))
    out += " #" + std::to_string(input.getIndex());
  else if (auto output = dyn_cast<VirtualOutputBF16Op>(op))
    out += " #" + std::to_string(output.getIndex());
  else if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(op))
    out += " u" + std::to_string(matmul.getUnit());
  else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    if (auto value = dyn_cast<IntegerAttr>(constant.getValue()))
      out += " " + std::to_string(value.getValue().getSExtValue());
  } else {
    SmallVector<Type> types(op->getOperandTypes());
    llvm::append_range(types, op->getResultTypes());
    for (Type type : types)
      if (auto unit = unitOf(type)) {
        out += " u" + std::to_string(*unit);
        break;
      }
  }
  if (auto loc = dyn_cast<FileLineColLoc>(op->getLoc()))
    out += " (L" + std::to_string(loc.getLine()) + ")";
  return out;
}

namespace {
void writeTimeline(llvm::json::OStream &j, const ScheduleTimeline &t) {
  j.attribute("cycles", t.cycles);
  j.attributeArray("order", [&] {
    for (unsigned node : t.order)
      j.value(static_cast<int64_t>(node));
  });
  j.attributeArray("spans", [&] {
    for (const ScheduleTimeline::Span &span : t.spans)
      j.object([&] {
        j.attribute("node", static_cast<int64_t>(span.node));
        j.attribute("start", span.start);
        j.attribute("end", span.end);
      });
  });
  j.attributeArray("instrs", [&] {
    for (const ScheduleTimeline::Instruction &in : t.instructions)
      j.object([&] {
        j.attribute("node", static_cast<int64_t>(in.node));
        j.attribute("mnemonic", in.mnemonic);
        j.attribute("engine", in.engine);
        j.attribute("issue", in.issue);
        j.attribute("end", in.end);
      });
  });
  j.attributeArray("pressure", [&] {
    for (const std::array<int, 3> &counts : t.live)
      j.array([&] {
        for (int count : counts)
          j.value(count);
      });
  });
  j.attributeArray("peak", [&] {
    for (int count : t.peak)
      j.value(count);
  });
}

void writeBlock(llvm::json::OStream &j, const BlockScheduleRecord &b) {
  j.attribute("index", static_cast<int64_t>(b.index));
  j.attributeArray("caps", [&] {
    for (int cap : b.caps)
      j.value(cap);
  });
  j.attribute("strategy", b.strategy);
  j.attribute("kept_source", b.keptReason.has_value());
  j.attribute("reason", b.keptReason.value_or(""));
  j.attributeArray("labels", [&] {
    for (const std::string &label : b.labels)
      j.value(label);
  });
  j.attributeArray("predecessors", [&] {
    for (const std::vector<unsigned> &preds : b.predecessors)
      j.array([&] {
        for (unsigned pred : preds)
          j.value(static_cast<int64_t>(pred));
      });
  });
  for (auto [key, timeline] : {std::pair{"source", &b.source},
                               std::pair{"scheduled", &b.scheduled}}) {
    if (*timeline)
      j.attributeObject(key, [&] { writeTimeline(j, **timeline); });
    else
      j.attribute(key, nullptr);
  }
}
} // namespace

void mlir::atlas::writeScheduleTrace(
    llvm::raw_ostream &os, llvm::ArrayRef<FunctionScheduleRecord> functions) {
  llvm::json::OStream j(os, 1);
  j.object([&] {
    j.attributeArray("functions", [&] {
      for (const FunctionScheduleRecord &f : functions)
        j.object([&] {
          j.attribute("name", f.name);
          j.attribute("kept_source", f.keptReason.has_value());
          j.attribute("reason", f.keptReason.value_or(""));
          j.attributeArray("blocks", [&] {
            for (const BlockScheduleRecord &b : f.blocks)
              j.object([&] { writeBlock(j, b); });
          });
        });
    });
  });
  os << "\n";
}
