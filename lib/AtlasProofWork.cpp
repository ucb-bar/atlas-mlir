#include "Atlas/AtlasProofWork.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>

using namespace mlir::atlas;
namespace {
constexpr unsigned kinds = unsigned(AtlasProofKind::Count);
struct Counts {
  std::atomic<uint64_t> invocations{0}, records{0}, transfers{0};
};
std::array<Counts, kinds> counters;
constexpr const char *names[] = {"dma", "mxu", "tile", "cfg", "source_memory", "buffer",
    "source_address", "captured_address", "source_initialization", "captured_initialization", "initialization_reachability"};
void report() {
  uint64_t contractWork = 0, auxiliaryWork = 0;
  std::fputs("{\"schema\":\"atlas-proof-work-v1\",\"units\":{\"record_visits\":\"contract records derived/examined or instruction tag scans\",\"transfer_visits\":\"abstract instruction/scalar-expression transfers including repeated visits\"},\"checks\":{", stderr);
  for (unsigned i = 0; i < kinds; ++i) {
    uint64_t invocations = counters[i].invocations.load(std::memory_order_relaxed);
    uint64_t records = counters[i].records.load(std::memory_order_relaxed);
    uint64_t transfers = counters[i].transfers.load(std::memory_order_relaxed);
    std::fprintf(stderr, "%s\"%s\":{\"invocations\":%" PRIu64 ",\"record_visits\":%" PRIu64 ",\"transfer_visits\":%" PRIu64 "}",
                 i ? "," : "", names[i], invocations, records, transfers);
    (i < 6 ? contractWork : auxiliaryWork) += records + transfers;
  }
  std::fprintf(stderr, "},\"six_contract_work\":%" PRIu64 ",\"auxiliary_work\":%" PRIu64 ",\"total_work\":%" PRIu64 "}\n",
               contractWork, auxiliaryWork, contractWork + auxiliaryWork);
}
bool enable() {
  const char *value = std::getenv("ATLAS_REPORT_PROOF_WORK");
  if (!value || std::strcmp(value, "1")) return false;
  std::atexit(report);
  return true;
}
} // namespace
const bool mlir::atlas::atlasProofWorkReportingEnabled = enable();
void mlir::atlas::recordAtlasProofWork(AtlasProofKind kind, AtlasProofEvent event, uint64_t count) {
  Counts &c = counters[unsigned(kind)];
  auto &counter = event == AtlasProofEvent::Invocation ? c.invocations :
                  event == AtlasProofEvent::Record ? c.records : c.transfers;
  counter.fetch_add(count, std::memory_order_relaxed);
}
