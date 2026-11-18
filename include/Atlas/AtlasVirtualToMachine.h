#ifndef ATLAS_VIRTUAL_TO_MACHINE_H
#define ATLAS_VIRTUAL_TO_MACHINE_H

#include "mlir/Pass/Pass.h"

namespace mlir::atlas {
void registerLowerAtlasVirtualToMachinePass();
std::unique_ptr<Pass>
createLowerAtlasVirtualToMachinePass(llvm::StringRef allocationReport = "",
                                    unsigned programSizeLimit = 0);
}

#endif
