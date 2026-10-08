// Runs against a caller-selected replay receipt. This is a compiler API test;
// the receipt's conditional hardware experiment remains a separate validation.
#include "Atlas/AtlasRTLEvidence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <functional>
#include <iostream>

using namespace mlir::atlas::timing;
namespace fs = std::filesystem;

static int failures = 0;
static void check(bool good, const char *description) {
  if (!good) { ++failures; std::cerr << "FAIL: " << description << '\n'; }
}
static std::string digest(llvm::StringRef bytes) {
  auto hash = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  return llvm::toHex(llvm::ArrayRef<uint8_t>(hash), true);
}
static bool rejected(llvm::Expected<RTLEvidence> result) {
  if (result) return false;
  llvm::consumeError(result.takeError());
  return true;
}
static Instr instruction(const char *name, int rd = 0, int rs1 = 1,
                         long long imm = 0) {
  Instr result;
  result.op = findOp(name); result.rd = rd; result.rs1 = rs1; result.imm = imm;
  return result;
}
static void mutation(const llvm::json::Value &original, const fs::path &base,
                     ExpectedEvidenceIdentity identity,
                     const std::function<void(llvm::json::Object &)> &edit,
                     const char *description) {
  llvm::json::Value copy = original;
  edit(*copy.getAsObject());
  std::string bytes;
  llvm::raw_string_ostream stream(bytes);
  stream << copy; stream.flush();
  identity.evidenceSha256 = digest(bytes);
  std::error_code ec;
  llvm::raw_fd_ostream file((base / "mutated.json").string(), ec);
  if (ec) { check(false, "write mutation fixture"); return; }
  file << bytes; file.close();
  check(rejected(loadRTLEvidence((base / "mutated.json").string(), identity, true)),
        description);
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: rtl-evidence-test REPORT REPORT_SHA MANIFEST_SHA HW_SHA\n";
    return 2;
  }
  ExpectedEvidenceIdentity identity{argv[2], argv[3], argv[4]};
  auto result = loadRTLEvidence(argv[1], identity, true);
  if (!result) { std::cerr << llvm::toString(result.takeError()) << '\n'; return 1; }
  const RTLEvidence &evidence = *result;
  check(std::string(evidence.resolverID()) == "atlas.vls.conservative.v1",
        "versioned conservative resolver");
  check(std::string(evidence.qualificationStatus()) == "conditional", "conditional status");
  check(rejected(loadRTLEvidence(argv[1], identity, false)), "explicit conditional opt-in");
  for (int field = 0; field < 3; ++field) {
    auto bad = identity;
    (field == 0 ? bad.evidenceSha256 : field == 1 ? bad.manifestSha256 :
                                                  bad.hardwareIRSha256) = std::string(64, '0');
    check(rejected(loadRTLEvidence(argv[1], bad, true)), "selected identity mismatch");
  }
  RegValues regs = unknownRegs();
  regs[1] = 0;
  auto load = instruction("vload", 4);
  auto store = instruction("vstore", 4);
  Footprint fl = evidence.resolve(load, regs), fs = evidence.resolve(store, regs);
  check(fl.error.empty() && fs.error.empty(), "supported VLS instances");
  check(fl.doneAge == 34 && fs.doneAge == 34 && !fl.writeDuringRead,
        "observed release and conservative overwrite policy");
  check(fl.accesses.size() == 3 && fs.accesses.size() == 3 &&
        fl.holds.size() == 3 && fs.holds.size() == 3, "resolved resources");
  check(dependence(load, fl, store, fs).distance == 35,
        "same-register load/store release boundary");
  auto other = instruction("vstore", 5);
  regs[1] = 65536;
  Footprint different = evidence.resolve(other, regs);
  ReservationTable table;
  table.reserve(load, fl, 0);
  check(!table.conflict(other, different, 34).empty() &&
        table.conflict(other, different, 35).empty(),
        "all VLS serialized even across different banks/registers");
  regs = unknownRegs();
  check(!evidence.resolve(load, regs).error.empty(), "unknown base rejected");
  for (uint32_t base : {8u, 393216u}) {
    regs[1] = base;
    check(!evidence.resolve(load, regs).error.empty(), "unsupported effective address");
  }
  for (uint32_t base : {7u, 524288u}) {
    regs[1] = base;
    check(evidence.resolve(load, regs).error.empty(), "hardware masks before address admission");
  }
  regs[1] = 0;
  check(!evidence.resolve(instruction("vload", 64), regs).error.empty(), "invalid MREG rejected");
  check(!evidence.resolve(instruction("vload", 4, 1, -1), regs).error.empty(), "negative transformed tile rejected");
  check(!evidence.resolve(instruction("vload", 4, 1, 8192), regs).error.empty(), "unencoded immediate rejected");
  for (const char *name : {"lw", "dma.wait.ch0", "vmov", "fence", "jal"})
    check(!evidence.resolve(instruction(name), regs).error.empty(), "unsupported operation rejected");
  for (const char *name : {"addi", "lui", "delay", "ecall"})
    check(evidence.resolve(instruction(name), regs).error.empty(), "bounded scalar support");
  auto marker = instruction("csrrw", 0, 1, 0xC10);
  check(evidence.resolve(marker, regs).error.empty(), "exact synchronous debug CSR marker");
  marker.imm = 0xC11;
  check(!evidence.resolve(marker, regs).error.empty(), "other CSR address rejected");
  marker.imm = 0xC10; marker.rd = 1;
  check(!evidence.resolve(marker, regs).error.empty(), "CSR readback outside scope");

  auto buffer = llvm::MemoryBuffer::getFile(argv[1]);
  if (!buffer) return 1;
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) { llvm::consumeError(parsed.takeError()); return 1; }
  auto *hardwareID = parsed->getAsObject()->getObject("inputs")->getObject("hardware_ir");
  auto hardwarePath = hardwareID->getString("path");
  if (!hardwarePath) return 1;
  fs::path selectedPath(hardwarePath->str());
  if (!selectedPath.is_absolute()) selectedPath = fs::path(argv[1]).parent_path() / selectedPath;
  // The initial successful load checked this selected path and its symlinks.
  auto hardwareBuffer = llvm::MemoryBuffer::getFile(selectedPath.string());
  if (!hardwareBuffer) return 1;
  std::string hardware = (*hardwareBuffer)->getBuffer().str();
  auto compatibility = checkRTLSemanticCompatibility(hardware);
  check(!compatibility, "reviewed full AtlasCore module closure");
  if (compatibility) llvm::consumeError(std::move(compatibility));
  for (const char *name : {"ScalarCore", "ScalarDecoder", "ScalarALU", "PcControl",
                           "CSRFile_1", "Vmem", "MregFile", "MregBankTracker",
                           "LSU", "AtlasCore"}) {
    std::string changed = hardware;
    auto header = changed.find(std::string("\n  hw.module private @") + name + "(");
    if (header == std::string::npos) {
      check(false, "locate reviewed mutation module"); continue;
    }
    auto constant = changed.find("hw.constant ", header);
    if (constant == std::string::npos) {
      check(false, "locate semantic mutation constant"); continue;
    }
    auto value = constant + std::string("hw.constant ").size();
    auto end = changed.find(' ', value);
    std::string before = changed.substr(value, end - value);
    changed.replace(value, end - value, before == "true" ? "false" :
                    before == "false" ? "true" : before == "0" ? "1" : "0");
    auto rejectedVariant = checkRTLSemanticCompatibility(changed);
    check(bool(rejectedVariant), "changed full-HW semantics rejected independently of byte selector");
    if (rejectedVariant) llvm::consumeError(std::move(rejectedVariant));
  }
  std::string relocated = hardware;
  auto scalarHeader = relocated.find("\n  hw.module private @ScalarCore(");
  auto alias = relocated.find("loc(#loc", scalarHeader);
  auto aliasEnd = relocated.find(')', alias);
  relocated.replace(alias + 8, aliasEnd - alias - 8, "123456789");
  auto locationOnly = checkRTLSemanticCompatibility(relocated);
  check(!locationOnly, "location alias renumbering preserves reviewed semantics");
  if (locationOnly) llvm::consumeError(std::move(locationOnly));
  llvm::SmallString<128> temporary;
  if (llvm::sys::fs::createUniqueDirectory("/tmp/atlas-rtl-evidence-test", temporary)) return 1;
  fs::path temp(temporary.str().str());
  std::error_code ec;
  fs::copy_file(fs::path(argv[1]).parent_path() / "retention-manifest.json",
                temp / "retention-manifest.json", ec);
  if (ec) { fs::remove_all(temp, ec); return 1; }
  mutation(*parsed, temp, identity, [](auto &r) { r["schema"] = "atlas.conditional_vls_hw_check.v1"; }, "unknown schema rejected");
  mutation(*parsed, temp, identity, [](auto &r) { r["state"] = "failed"; }, "failed receipt rejected");
  mutation(*parsed, temp, identity, [](auto &r) { r.erase("scope"); }, "missing conditions rejected");
  mutation(*parsed, temp, identity, [](auto &r) {
    r["resolver_bindings"] = llvm::json::Array{"atlas.vls.conservative.v2"};
  }, "unknown resolver binding rejected");
  mutation(*parsed, temp, identity, [](auto &r) {
    (*r.getObject("scope"))["integrated_target_execution_qualified"] = true;
  }, "overstated qualification rejected");
  mutation(*parsed, temp, identity, [](auto &r) {
    auto *rows = r.getArray("footprint_comparisons");
    for (auto &row : *rows) {
      auto *item = row.getAsObject();
      if (item->getString("comparison") == "releases_and_path") {
        (*item)["observed_first_free_age"] = 34; break;
      }
    }
  }, "premature measured release rejected with matching selector digest");
  mutation(*parsed, temp, identity, [](auto &r) {
    auto *rows = r.getArray("footprint_comparisons");
    for (auto &row : *rows) {
      auto *item = row.getAsObject();
      if (auto *events = item->getArray("observed_absolute")) {
        (*(*events)[0].getAsObject())["cycle"] = 9; break;
      }
    }
  }, "modified measured stream rejected with matching selector digest");
  mutation(*parsed, temp, identity, [](auto &r) {
    r["footprint_comparisons"] = llvm::json::Array{};
  }, "missing stream evidence rejected");
  fs::remove_all(temp, ec);
  if (failures) return 1;
  std::cout << "RTL evidence loader/resolver checks passed\n";
  return 0;
}
