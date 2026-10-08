#include "Atlas/AtlasDelayInsertion.h"
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasScheduling.h"
#include "Atlas/AtlasToLLVM.h"
#include "Atlas/AtlasStreamVerification.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "Atlas/AtlasVirtualScheduling.h"
#include "Atlas/AtlasVirtualToMachine.h"
#include "Atlas/AtlasGeneratedSchedule.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::atlas::AtlasDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect>();
  mlir::atlas::registerConvertAtlasToLLVMPass();
  mlir::atlas::registerConvertAtlasToLLVMCallsPass();
  mlir::atlas::registerFinalizeAtlasLLVMCallsPass();
  mlir::atlas::registerVerifyAtlasMachineStreamPass();
  mlir::atlas::registerInsertAtlasDelaysPass();
  mlir::atlas::registerScheduleAtlasStreamPass();
  mlir::atlas::registerVerifyAtlasVirtualStreamPass();
  mlir::atlas::registerScheduleAtlasVirtualPass();
  mlir::atlas::registerLowerAtlasVirtualToMachinePass();
  mlir::atlas::registerVerifyAtlasGeneratedSchedulePass();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Atlas dialect verifier\n", registry));
}
