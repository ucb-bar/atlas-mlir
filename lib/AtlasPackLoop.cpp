#include "Atlas/AtlasPackLoop.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasOps.h"

using namespace mlir;
using namespace mlir::atlas;

bool mlir::atlas::isAtlasPackRelayoutLoop(ArrayRef<Operation *> ops,
                                         const AtlasLoop &loop) {
  auto source = contractTag(ops[loop.branch], kAtlasTagCFGSource);
  if (!source)
    return false;
  for (size_t pc = loop.begin; pc <= loop.branch; ++pc) {
    Operation *op = ops[pc];
    if (isa<DelayOp>(op))
      continue;
    auto helper = op->getAttrOfType<StringAttr>(kAtlasTagCFGHelper);
    if (!helper || helper.getValue() != "pack" ||
        contractTag(op, kAtlasTagCFGSource) != source)
      return false;
  }
  return true;
}
