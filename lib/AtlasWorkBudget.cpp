#include "Atlas/AtlasWorkBudget.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

using namespace mlir;
using namespace mlir::atlas;
namespace {
llvm::cl::opt<uint64_t> maxWork(
    "atlas-max-work", llvm::cl::init(100000000),
    llvm::cl::desc("Maximum counted work per named Atlas algorithm invocation"));
llvm::cl::opt<uint64_t> maxTotalWork(
    "atlas-max-total-work", llvm::cl::init(1000000000),
    llvm::cl::desc("Maximum total counted Atlas work in this process"));
std::atomic<uint64_t> totalWork{0};
std::atomic<bool> workExhausted{false};
llvm::cl::opt<bool> reportWork(
    "atlas-report-work", llvm::cl::init(false),
    llvm::cl::desc("Report deterministic Atlas algorithm work as JSON on stderr"));
struct Counts {
  uint64_t invocations = 0, used = 0, maximum = 0, exhausted = 0;
};
// Deliberately live through the exit reporter, regardless of static teardown.
struct Report {
  std::mutex mutex;
  std::map<std::string, Counts> counts;
};
Report &report() { static auto *r = new Report; return *r; }
void emitReport() {
  auto &r = report();
  std::lock_guard<std::mutex> lock(r.mutex);
  llvm::json::Object algorithms;
  for (const auto &[name, c] : r.counts)
    algorithms[name] = llvm::json::Object{
        {"invocations", c.invocations}, {"used", c.used},
        {"maximum_used", c.maximum}, {"exhaustions", c.exhausted}};
  llvm::errs() << llvm::json::Value(llvm::json::Object{
      {"schema", "atlas-work-budget-v1"}, {"limit", uint64_t(maxWork)},
      {"total_limit", uint64_t(maxTotalWork)}, {"total_used", totalWork.load()},
      {"units", "record, candidate, edge, fixed-point and reservation probe visits"},
      {"algorithms", std::move(algorithms)}}) << '\n';
}
void enableReport() {
  static const bool enabled = [] { std::atexit(emitReport); return true; }();
  (void)enabled;
}
} // namespace
AtlasWorkBudget::AtlasWorkBudget(Operation *operation, llvm::StringRef name, bool *failureFlag)
    : operation(operation), name(name), limit(maxWork), failureFlag(failureFlag) {
  if (reportWork) enableReport();
}
AtlasWorkBudget::~AtlasWorkBudget() {
  if (!reportWork) return;
  auto &r = report();
  std::lock_guard<std::mutex> lock(r.mutex);
  auto &c = r.counts[name.str()];
  ++c.invocations;
  c.used += consumed;
  c.maximum = std::max(c.maximum, consumed);
  c.exhausted += exceeded;
}
LogicalResult AtlasWorkBudget::charge(uint64_t count) {
  if (exceeded) return failure();
  bool processLimit = false;
  uint64_t total = totalWork.load(std::memory_order_relaxed);
  if (count <= limit - consumed) {
    for (;;) {
      if (count > uint64_t(maxTotalWork) - total) {
        processLimit = true;
        break;
      }
      if (totalWork.compare_exchange_weak(total, total + count,
                                         std::memory_order_relaxed)) {
        consumed += count;
        return success();
      }
    }
  }
  exceeded = true;
  workExhausted.store(true, std::memory_order_relaxed);
  if (failureFlag) *failureFlag = true;
  operation->emitError() << (processLimit ? "Atlas process work budget exhausted: "
                                         : "Atlas work budget exhausted: ") << name
                         << " (used=" << (processLimit ? total : consumed)
                         << ", requested=" << count
                         << ", limit=" << (processLimit ? uint64_t(maxTotalWork) : limit) << ")";
  return failure();
}
bool AtlasWorkBudget::anyExhausted() {
  return workExhausted.load(std::memory_order_relaxed);
}
