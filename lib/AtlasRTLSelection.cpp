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

FailureOr<std::shared_ptr<RTLEvidence>>
mlir::atlas::getSelectedRTLEvidence(ModuleOp module) {
  Attribute raw = module->getAttr("atlas.rtl_evidence");
  if (!raw && module->hasAttr("atlas.rtl_qualification"))
    return module.emitError("RTL qualification annotation has no selected evidence");
  if (!raw)
    return std::shared_ptr<RTLEvidence>();
  if (auto status = module->getAttr("atlas.rtl_qualification")) {
    auto text = dyn_cast<StringAttr>(status);
    if (!text || text.getValue() != "conditional")
      return module.emitError("selected RTL evidence has conditional qualification only");
  }
  auto attr = dyn_cast<DictionaryAttr>(raw);
  if (!attr)
    return module.emitError("atlas.rtl_evidence must be a selection dictionary");
  auto path = attr.getAs<StringAttr>("path");
  auto report = attr.getAs<StringAttr>("evidence_sha256");
  auto manifest = attr.getAs<StringAttr>("manifest_sha256");
  auto hardware = attr.getAs<StringAttr>("hardware_ir_sha256");
  auto conditional = attr.getAs<BoolAttr>("allow_conditional");
  auto resolver = attr.getAs<StringAttr>("resolver");
  if (!path || !report || !manifest || !hardware || !conditional || !resolver)
    return module.emitError("incomplete RTL evidence selection");
  auto dmaPath = attr.getAs<StringAttr>("dma_path");
  auto dmaHash = attr.getAs<StringAttr>("dma_evidence_sha256");
  auto xluPath = attr.getAs<StringAttr>("xlu_path");
  auto xluHash = attr.getAs<StringAttr>("xlu_evidence_sha256");
  auto vmulPath = attr.getAs<StringAttr>("vmul_path");
  auto vmulHash = attr.getAs<StringAttr>("vmul_evidence_sha256");
  if (bool(dmaPath) != bool(dmaHash) ||
      (attr.get("dma_path") && !dmaPath) ||
      (attr.get("dma_evidence_sha256") && !dmaHash))
    return module.emitError("incomplete DMA evidence selection");
  if (bool(xluPath) != bool(xluHash) ||
      (attr.get("xlu_path") && !xluPath) ||
      (attr.get("xlu_evidence_sha256") && !xluHash))
    return module.emitError("incomplete XLU evidence selection");
  if (bool(vmulPath) != bool(vmulHash) ||
      (attr.get("vmul_path") && !vmulPath) ||
      (attr.get("vmul_evidence_sha256") && !vmulHash))
    return module.emitError("incomplete VMUL evidence selection");
  const char *expectedResolver = RTLEvidence::resolverIDFor(
      bool(dmaPath), bool(xluPath), bool(vmulPath));
  if (resolver.getValue() != expectedResolver)
    return module.emitError("unsupported RTL evidence resolver or capability selection");
  ExpectedEvidenceIdentity identity{report.getValue().str(),
                                    manifest.getValue().str(),
                                    hardware.getValue().str()};
  auto loaded = loadRTLEvidence(path.getValue(), identity, conditional.getValue());
  if (!loaded)
    return module.emitError("RTL evidence selection failed: ")
           << llvm::toString(loaded.takeError());
  if (dmaPath)
    if (auto error = loadDMAEvidence(*loaded, dmaPath.getValue(), dmaHash.getValue()))
      return module.emitError("DMA evidence selection failed: ")
             << llvm::toString(std::move(error));
  if (xluPath)
    if (auto error = loadXLUEvidence(*loaded, xluPath.getValue(), xluHash.getValue()))
      return module.emitError("XLU evidence selection failed: ")
             << llvm::toString(std::move(error));
  if (vmulPath)
    if (auto error = loadVmulEvidence(*loaded, vmulPath.getValue(), vmulHash.getValue()))
      return module.emitError("VMUL evidence selection failed: ")
             << llvm::toString(std::move(error));
  return std::make_shared<RTLEvidence>(std::move(*loaded));
}

namespace {
struct SelectAtlasRTLEvidencePass
    : PassWrapper<SelectAtlasRTLEvidencePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectAtlasRTLEvidencePass)
  SelectAtlasRTLEvidencePass() = default;
  SelectAtlasRTLEvidencePass(const SelectAtlasRTLEvidencePass &other)
      : PassWrapper(other) {}
  Option<std::string> evidence{*this, "evidence", llvm::cl::desc("Selected replay report")};
  Option<std::string> evidenceHash{*this, "evidence-sha256", llvm::cl::desc("Expected replay report SHA-256")};
  Option<std::string> manifestHash{*this, "manifest-sha256", llvm::cl::desc("Expected retention manifest SHA-256")};
  Option<std::string> hardwareHash{*this, "hardware-ir-sha256", llvm::cl::desc("Expected HW/Comb/Seq SHA-256")};
  Option<std::string> dmaEvidence{*this, "dma-evidence", llvm::cl::desc("Optional selected DMA replay report")};
  Option<std::string> dmaEvidenceHash{*this, "dma-evidence-sha256", llvm::cl::desc("Expected DMA replay SHA-256")};
  Option<std::string> xluEvidence{*this, "xlu-evidence", llvm::cl::desc("Optional selected XLU replay report")};
  Option<std::string> xluEvidenceHash{*this, "xlu-evidence-sha256", llvm::cl::desc("Expected XLU replay SHA-256")};
  Option<std::string> vmulEvidence{*this, "vmul-evidence", llvm::cl::desc("Optional selected BF16 multiply replay report")};
  Option<std::string> vmulEvidenceHash{*this, "vmul-evidence-sha256", llvm::cl::desc("Expected BF16 multiply replay SHA-256")};
  Option<bool> allowConditional{*this, "allow-conditional", llvm::cl::init(false),
      llvm::cl::desc("Explicitly accept a conditional experimental scope")};
  StringRef getArgument() const final { return "select-atlas-rtl-evidence"; }
  StringRef getDescription() const final {
    return "Select identity-checked conditional RTL evidence for bounded timing consumers";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    NamedAttrList fields;
    fields.set("path", builder.getStringAttr(evidence));
    fields.set("evidence_sha256", builder.getStringAttr(evidenceHash));
    fields.set("manifest_sha256", builder.getStringAttr(manifestHash));
    fields.set("hardware_ir_sha256", builder.getStringAttr(hardwareHash));
    fields.set("allow_conditional", builder.getBoolAttr(allowConditional));
    if (dmaEvidence.empty() != dmaEvidenceHash.empty()) {
      module.emitError("DMA evidence path and SHA-256 must be selected together");
      signalPassFailure();
      return;
    }
    if (!dmaEvidence.empty()) {
      fields.set("dma_path", builder.getStringAttr(dmaEvidence));
      fields.set("dma_evidence_sha256", builder.getStringAttr(dmaEvidenceHash));
    }
    if (xluEvidence.empty() != xluEvidenceHash.empty()) {
      module.emitError("XLU evidence path and SHA-256 must be selected together");
      signalPassFailure();
      return;
    }
    if (!xluEvidence.empty()) {
      fields.set("xlu_path", builder.getStringAttr(xluEvidence));
      fields.set("xlu_evidence_sha256", builder.getStringAttr(xluEvidenceHash));
    }
    if (vmulEvidence.empty() != vmulEvidenceHash.empty()) {
      module.emitError("VMUL evidence path and SHA-256 must be selected together");
      signalPassFailure();
      return;
    }
    if (!vmulEvidence.empty()) {
      fields.set("vmul_path", builder.getStringAttr(vmulEvidence));
      fields.set("vmul_evidence_sha256", builder.getStringAttr(vmulEvidenceHash));
    }
    fields.set("resolver", builder.getStringAttr(RTLEvidence::resolverIDFor(
        !dmaEvidence.empty(), !xluEvidence.empty(), !vmulEvidence.empty())));
    module->setAttr("atlas.rtl_evidence", fields.getDictionary(module.getContext()));
    if (failed(getSelectedRTLEvidence(module))) {
      signalPassFailure();
      return;
    }
    module->setAttr("atlas.rtl_qualification", builder.getStringAttr("conditional"));
  }
};
} // namespace

void mlir::atlas::registerSelectAtlasRTLEvidencePass() {
  PassRegistration<SelectAtlasRTLEvidencePass>();
}
