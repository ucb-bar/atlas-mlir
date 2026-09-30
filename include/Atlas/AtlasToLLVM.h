#ifndef ATLAS_TO_LLVM_H
#define ATLAS_TO_LLVM_H

namespace mlir::atlas {
void registerConvertAtlasToLLVMPass();
void registerConvertAtlasToLLVMCallsPass();
void registerFinalizeAtlasLLVMCallsPass();
}

#endif
