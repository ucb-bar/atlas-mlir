#ifndef ATLAS_MODEL_REPLAY_H
#define ATLAS_MODEL_REPLAY_H

#include "Atlas/AtlasSearchPolicy.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace mlir::atlas::timing {

// A source-checked clock and explicit fixed compiler projection policy.
// Machine implements the declared recurrence; this descriptor carries no code.
struct ModelReplayProgram {
  llvm::json::Object document;

  static llvm::Expected<ModelReplayProgram> parse(const llvm::json::Object &object) {
    auto error = [](const char *why) -> llvm::Error {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                    "unsupported projected-Core replay: %s", why);
    };
    if (object.size() != 3 || object.getString("schema") != "atlas.projected_core_replay.v2" ||
        !object.getObject("source") || !object.getObject("projection_policy"))
      return error("schema/source/policy");
    const auto &source = *object.getObject("source");
    if (source.size() != 7 || source.getString("schema") != "atlas.core_edges.v1" ||
        source.getInteger("clock_step") != 1 || source.getString("dma_wait_observes") != "previous_edge_flags" ||
        source.getBoolean("implicit_exu_stalls") != false ||
        source.getString("source_contract") != "atlas.model_frontend.v1" ||
        !source.getArray("phases") || *source.getArray("phases") !=
          llvm::json::Array{"s1_previous_state", "execution_units", "fetch"} ||
        !source.getArray("checked_methods") || *source.getArray("checked_methods") !=
          llvm::json::Array{"Core.tick", "Core._tick_s1"})
      return error("source edge contract");
    auto templateValue = llvm::json::parse(search_policy::kReplayProjectionJson);
    if (!templateValue) return templateValue.takeError();
    auto *policy = templateValue->getAsObject();
    auto *projection = policy ? policy->getObject("projection_policy") : nullptr;
    if (!projection || projection->getString("algorithm") != "fixed-start-capacity-place-v1")
      return error("compiled projection policy");
    (*projection)["policy_sha256"] = search_policy::kReplayProjectionSHA256;
    if (*object.getObject("projection_policy") != *projection)
      return error("descriptor differs from declared projection policy");
    return ModelReplayProgram{object};
  }
};
} // namespace mlir::atlas::timing
#endif
