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
  if (resolver.getValue() != RTLEvidence::resolverID())
    return module.emitError("unsupported RTL evidence resolver");
  ExpectedEvidenceIdentity identity{report.getValue().str(),
                                    manifest.getValue().str(),
                                    hardware.getValue().str()};
  auto loaded = loadRTLEvidence(path.getValue(), identity, conditional.getValue());
  if (!loaded)
    return module.emitError("RTL evidence selection failed: ")
           << llvm::toString(loaded.takeError());
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
    fields.set("resolver", builder.getStringAttr(RTLEvidence::resolverID()));
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
