// Shared checks for the C++ verification suites. One executable runs the suite named by its argument; each suite
// gets a fresh context whose diagnostics (with notes) accumulate in `diagnostics` instead of being printed.
#ifndef ATLAS_TEST_VERIFICATIONTESTSUPPORT_H
#define ATLAS_TEST_VERIFICATIONTESTSUPPORT_H

#include "Atlas/AtlasDelayInsertion.h"
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasVirtualToMachine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <utility>

namespace atlas_test {
using namespace mlir;

inline unsigned checks = 0, failures = 0;
inline std::string diagnostics;

inline void check(bool condition, llvm::StringRef name, llvm::StringRef detail = {}) {
  ++checks;
  if (condition)
    return;
  ++failures;
  llvm::errs() << "FAIL: " << name << '\n';
  if (!detail.empty())
    llvm::errs() << detail << '\n';
}

inline bool diagnosed(llvm::StringRef message) { return llvm::StringRef(diagnostics).contains(message); }

// Runs `verifier` (returning LogicalResult); a valid case emits no diagnostics, and every fragment must appear.
template <typename Verifier>
void expectVerified(llvm::StringRef name, bool valid, llvm::ArrayRef<llvm::StringRef> fragments, Verifier &&verifier) {
  diagnostics.clear();
  bool accepted = succeeded(verifier());
  check(accepted == valid, name, diagnostics);
  if (valid)
    check(diagnostics.empty(), name, diagnostics);
  for (llvm::StringRef fragment : fragments)
    check(diagnosed(fragment), name, "missing diagnostic fragment '" + fragment.str() + "':\n" + diagnostics);
}

inline int64_t integer(DictionaryAttr record, llvm::StringRef name) { return record.getAs<IntegerAttr>(name).getInt(); }

// A signless i32 field equal to `expected` modulo 2^32.
inline void field(DictionaryAttr record, llvm::StringRef name, int64_t expected) {
  auto value = record.getAs<IntegerAttr>(name);
  check(value && value.getType().isSignlessInteger(32) && value.getValue().getZExtValue() == uint32_t(expected), name);
}

inline std::string replace(std::string text, llvm::StringRef from, llvm::StringRef to) {
  size_t position = text.find(from.str());
  check(position != std::string::npos, "fixture contains " + from.str());
  if (position != std::string::npos)
    text.replace(position, from.size(), to.str());
  return text;
}

// Parses typed source that must verify.
inline OwningOpRef<ModuleOp> parse(MLIRContext &context, llvm::StringRef text, llvm::StringRef name) {
  auto module = parseSourceString<ModuleOp>(text, &context);
  check(module && succeeded(verify(*module)), name);
  return module;
}

using Suite = void (*)(MLIRContext &);
void runCFGContract(MLIRContext &);
void runDMAAllocation(MLIRContext &);
void runDMAContract(MLIRContext &);
void runDMAHelper(MLIRContext &);
void runMXUAllocation(MLIRContext &);
void runMXUContract(MLIRContext &);
void runSourceMemoryContract(MLIRContext &);
void runTileContract(MLIRContext &);
void runTimingProvider(MLIRContext &);
void runVerificationContext(MLIRContext &);

inline int runSuite(int argc, char **argv, llvm::ArrayRef<std::pair<llvm::StringRef, Suite>> suites) {
  const std::pair<llvm::StringRef, Suite> *selected = nullptr;
  for (const auto &suite : suites)
    if (argc == 2 && suite.first == argv[1])
      selected = &suite;
  if (!selected) {
    llvm::errs() << "usage: " << argv[0] << " <suite>; suites:";
    for (const auto &suite : suites)
      llvm::errs() << ' ' << suite.first;
    llvm::errs() << '\n';
    return 2;
  }
  DialectRegistry registry;
  registry.insert<atlas::AtlasDialect, arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect>();
  atlas::registerLowerAtlasVirtualToMachinePass();
  atlas::registerInsertAtlasDelaysPass();
  MLIRContext context(registry);
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &diagnostic) {
    llvm::raw_string_ostream stream(diagnostics);
    diagnostic.print(stream);
    stream << '\n';
    for (const Diagnostic &note : diagnostic.getNotes()) {
      note.print(stream);
      stream << '\n';
    }
    return success();
  });
  selected->second(context);
  llvm::outs() << checks << ' ' << selected->first << " checks, " << failures << " failures\n";
  return failures != 0;
}
} // namespace atlas_test

#endif
