#include "Atlas/AtlasRTLSelection.h"
#include "Atlas/AtlasEncoding.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

LogicalResult mlir::atlas::checkSelectedRTLProgramSize(ModuleOp module) {
  SmallVector<uint32_t> words;
  if (failed(collectAtlasWords(module, words, /*llvmBlock=*/false)))
    return failure();
  if (words.size() > RTLEvidence::maximumProgramWords())
    return module.emitError("selected RTL program exceeds the 32768-word instruction memory");
  return success();
}

FailureOr<AtlasStream> mlir::atlas::readTimedStream(ModuleOp module,
                                                    TargetTiming &target) {
  auto evidence = getSelectedRTLEvidence(module);
  if (failed(evidence) ||
      (*evidence && failed(checkSelectedRTLProgramSize(module))))
    return failure();
  target = *evidence ? (*evidence)->targetTiming() : TargetTiming();
  FailureOr<AtlasStream> stream = readAtlasStream(module);
  if (failed(stream))
    return failure();
  if (target) {
    if (stream->starts.size() != 1 || !stream->endsInHalt(0))
      return module.emitError("selected RTL timing requires one straight-line "
                              "stream ending in ECALL");
    for (Instr &in : stream->instrs)
      if (in.op->opClass == OpClass::Csr)
        in.release = true; // publish the completion marker only once drained
  }
  if (failed(checkAtlasStream(*stream, target)))
    return failure();
  return stream;
}

FailureOr<std::shared_ptr<RTLEvidence>>
mlir::atlas::getSelectedRTLEvidence(ModuleOp module) {
  Attribute raw = module->getAttr("atlas.rtl_evidence");
  if (!raw)
    return std::shared_ptr<RTLEvidence>();
  auto attr = dyn_cast<DictionaryAttr>(raw);
  if (!attr)
    return module.emitError("atlas.rtl_evidence must be a selection dictionary");
  auto path = attr.getAs<StringAttr>("op_timing");
  auto hash = attr.getAs<StringAttr>("op_timing_sha256");
  auto hardware = attr.getAs<StringAttr>("hardware_ir_sha256");
  auto resolver = attr.getAs<StringAttr>("resolver");
  if (!path || !hash || !hardware || !resolver)
    return module.emitError("incomplete RTL evidence selection");
  bool dma = false;
  if (Attribute policy = attr.get("dma")) {
    auto text = dyn_cast<StringAttr>(policy);
    if (!text || text.getValue() != "wait")
      return module.emitError("unsupported DMA policy; only dma=wait is defined");
    dma = true;
  }
  if (resolver.getValue() != RTLEvidence::resolverID())
    return module.emitError("unsupported RTL evidence resolver");
  auto loaded = loadRTLTimingFacts(path.getValue(), hash.getValue(), dma);
  if (!loaded)
    return module.emitError("RTL evidence selection failed: ")
           << llvm::toString(loaded.takeError());
  if (loaded->hardwareIRSha256() != hardware.getValue())
    return module.emitError("selected facts describe a different hardware IR than recorded");
  return std::make_shared<RTLEvidence>(std::move(*loaded));
}

namespace {
struct SelectAtlasRTLEvidencePass
    : PassWrapper<SelectAtlasRTLEvidencePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectAtlasRTLEvidencePass)
  SelectAtlasRTLEvidencePass() = default;
  SelectAtlasRTLEvidencePass(const SelectAtlasRTLEvidencePass &other)
      : PassWrapper(other) {}
  Option<std::string> opTiming{*this, "op-timing",
      llvm::cl::desc("RTL-computed operation timing facts (merlin.op_timing.v1)")};
  Option<std::string> opTimingHash{*this, "op-timing-sha256",
      llvm::cl::desc("Expected SHA-256 of the facts file")};
  Option<std::string> dmaPolicy{*this, "dma",
      llvm::cl::desc("Optional DMA policy: wait")};
  StringRef getArgument() const final { return "select-atlas-rtl-evidence"; }
  StringRef getDescription() const final {
    return "Select SHA-256-identified RTL-computed timing facts";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!dmaPolicy.empty() && dmaPolicy != "wait") {
      module.emitError("unsupported DMA policy; only dma=wait is defined");
      signalPassFailure();
      return;
    }
    auto loaded = loadRTLTimingFacts(opTiming, opTimingHash, dmaPolicy == "wait");
    if (!loaded) {
      module.emitError("RTL evidence selection failed: ") << llvm::toString(loaded.takeError());
      signalPassFailure();
      return;
    }
    Builder builder(module.getContext());
    NamedAttrList fields;
    fields.set("op_timing", builder.getStringAttr(opTiming));
    fields.set("op_timing_sha256", builder.getStringAttr(opTimingHash));
    fields.set("hardware_ir_sha256", builder.getStringAttr(loaded->hardwareIRSha256()));
    if (loaded->dmaWait())
      fields.set("dma", builder.getStringAttr("wait"));
    fields.set("resolver", builder.getStringAttr(RTLEvidence::resolverID()));
    module->setAttr("atlas.rtl_evidence", fields.getDictionary(module.getContext()));
  }
};
} // namespace

void mlir::atlas::registerSelectAtlasRTLEvidencePass() {
  PassRegistration<SelectAtlasRTLEvidencePass>();
}
