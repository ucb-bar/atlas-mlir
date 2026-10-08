#include "Atlas/AtlasRTLEvidence.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <algorithm>
#include <cctype>
#include <deque>
#include <filesystem>
#include <map>
#include <set>

using namespace mlir::atlas::timing;
namespace {
namespace fs = std::filesystem;
using Object = llvm::json::Object;
using Array = llvm::json::Array;

llvm::Error failure(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "RTL evidence: " + message);
}

bool hashSyntax(llvm::StringRef value) {
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

std::string digest(llvm::StringRef bytes) {
  auto hash = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  return llvm::toHex(llvm::ArrayRef<uint8_t>(hash), true);
}

// Keep strings, semantic attributes and newline bytes exactly. Remove only
// horizontal whitespace followed by a location-alias reference, outside
// quoted strings. This matches fingerprint-rtl-modules.py's v1 normalizer.
std::string normalizeModule(llvm::StringRef raw) {
  std::string out;
  bool quoted = false, escaped = false;
  for (size_t i = 0; i < raw.size();) {
    char c = raw[i];
    if (quoted) {
      out.push_back(c); ++i;
      if (escaped) escaped = false;
      else if (c == '\\') escaped = true;
      else if (c == '"') quoted = false;
      continue;
    }
    if (c == '"') { quoted = true; out.push_back(c); ++i; continue; }
    if (c == ' ' || c == '\t') {
      size_t end = i;
      while (end < raw.size() && (raw[end] == ' ' || raw[end] == '\t')) ++end;
      if (raw.substr(end).starts_with("loc(#loc")) {
        size_t digits = end + 8, tail = digits;
        while (tail < raw.size() && raw[tail] >= '0' && raw[tail] <= '9') ++tail;
        if (tail > digits && tail < raw.size() && raw[tail] == ')') {
          i = tail + 1; continue;
        }
      }
      out.append(raw.substr(i, end - i).str()); i = end; continue;
    }
    out.push_back(c); ++i;
  }
  return out;
}

bool forbidden(const fs::path &path) {
  for (const auto &part : path) {
    std::string name = part.string();
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (name.find("hammer") != std::string::npos ||
        name.find("vlsi") != std::string::npos)
      return true;
  }
  return false;
}

// Resolve one component at a time, inspecting a symlink's spelling before
// accessing its target. real_path/canonical would traverse unchecked targets.
llvm::Expected<std::string> safePath(llvm::StringRef name,
                                     const fs::path &base) {
  fs::path supplied(name.str());
  if (supplied.empty() || forbidden(supplied) || forbidden(base))
    return failure("empty or prohibited artifact path");
  fs::path full = supplied.is_absolute() ? supplied : base / supplied;
  if (!full.is_absolute()) {
    std::error_code ec;
    fs::path cwd = fs::current_path(ec);
    if (ec || forbidden(cwd)) return failure("unavailable artifact base");
    full = cwd / full;
  }
  std::deque<fs::path> remaining;
  for (const auto &part : full.relative_path()) remaining.push_back(part);
  fs::path resolved = full.root_path();
  unsigned links = 0;
  while (!remaining.empty()) {
    fs::path part = remaining.front();
    remaining.pop_front();
    if (part == ".") continue;
    if (part == "..") { resolved = resolved.parent_path(); continue; }
    fs::path candidate = resolved / part;
    if (forbidden(candidate)) return failure("prohibited artifact path");
    std::error_code ec;
    auto status = fs::symlink_status(candidate, ec);
    if (ec || status.type() == fs::file_type::not_found)
      return failure("artifact path does not exist: " + candidate.string());
    if (fs::is_symlink(status)) {
      if (++links > 64) return failure("artifact symlink cycle");
      fs::path target = fs::read_symlink(candidate, ec);
      if (ec || forbidden(target)) return failure("prohibited symlink target");
      if (target.is_absolute()) resolved = target.root_path();
      auto relative = target.is_absolute() ? target.relative_path() : target;
      std::vector<fs::path> parts;
      for (const auto &next : relative) parts.push_back(next);
      for (auto it = parts.rbegin(); it != parts.rend(); ++it)
        remaining.push_front(*it);
    } else {
      resolved = candidate;
    }
  }
  return resolved.string();
}

struct Reader {
  std::string error;
  void reject(const std::string &why) { if (error.empty()) error = why; }
  const Object *object(const Object *parent, llvm::StringRef key) {
    const Object *result = parent ? parent->getObject(key) : nullptr;
    if (!result) reject("missing/malformed object " + key.str());
    return result;
  }
  const Array *array(const Object *parent, llvm::StringRef key) {
    const Array *result = parent ? parent->getArray(key) : nullptr;
    if (!result) reject("missing/malformed array " + key.str());
    return result;
  }
  std::string str(const Object *parent, llvm::StringRef key) {
    auto result = parent ? parent->getString(key) : std::nullopt;
    if (!result) { reject("missing/malformed string " + key.str()); return {}; }
    return result->str();
  }
  int64_t integer(const Object *parent, llvm::StringRef key) {
    auto result = parent ? parent->getInteger(key) : std::nullopt;
    if (!result) { reject("missing/malformed integer " + key.str()); return -1; }
    return *result;
  }
  void stringIs(const Object *parent, llvm::StringRef key, llvm::StringRef value) {
    if (str(parent, key) != value) reject("unsupported " + key.str());
  }
  void boolIs(const Object *parent, llvm::StringRef key, bool value) {
    auto result = parent ? parent->getBoolean(key) : std::nullopt;
    if (!result || *result != value) reject("missing/unsupported " + key.str());
  }
  std::string read(llvm::StringRef path, const fs::path &base) {
    if (!error.empty()) return {};
    auto safe = safePath(path, base);
    if (!safe) { reject(llvm::toString(safe.takeError())); return {}; }
    auto buffer = llvm::MemoryBuffer::getFile(*safe);
    if (!buffer) { reject("cannot read artifact " + *safe); return {}; }
    return (*buffer)->getBuffer().str();
  }
  std::string artifact(const Object *identity, const fs::path &base) {
    std::string path = str(identity, "path");
    std::string sha = str(identity, "sha256");
    int64_t size = integer(identity, "bytes");
    if (!hashSyntax(sha) || size < 0) reject("invalid artifact identity");
    auto bytes = read(path, base);
    if (error.empty() && (size != static_cast<int64_t>(bytes.size()) ||
                          digest(bytes) != sha))
      reject("artifact hash/size mismatch: " + path);
    return bytes;
  }
};

bool sameIdentity(const Object *a, const Object *b) {
  return a && b && a->getString("sha256") == b->getString("sha256") &&
         a->getInteger("bytes") == b->getInteger("bytes");
}

struct TraceFacts {
  int sourceAge = 0, destinationAge = 0, rows = 0, step = 0;
  int busyLast = 0, sourceLast = 0, destinationLast = 0;
  std::vector<std::pair<int, int>> source, destination;
  std::vector<int> busy;
};

TraceFacts parseTrace(Reader &r, llvm::StringRef bytes, bool load) {
  TraceFacts out;
  std::set<int> cycles;
  int complete = 0;
  std::string sourcePrefix = load ? "vmem_read" : "mreg_read";
  std::string destinationPrefix = load ? "mreg_write" : "vmem_write";
  while (!bytes.empty()) {
    auto line = bytes.split('\n'); bytes = line.second;
    if (!line.first.starts_with("{")) continue;
    auto parsed = llvm::json::parse(line.first);
    if (!parsed) { r.reject(llvm::toString(parsed.takeError())); continue; }
    const Object *row = parsed->getAsObject();
    if (!row) { r.reject("malformed trace row"); continue; }
    if (row->getBoolean("complete").value_or(false)) {
      ++complete;
      r.boolIs(row, "data_and_guards_pass", true);
      if (r.integer(row, "reads") != 32 || r.integer(row, "writes") != 32)
        r.reject("standalone completion count mismatch");
      continue;
    }
    int64_t cycle = r.integer(row, "cycle");
    if (cycle < 0 || cycle > 1000 || !cycles.insert(cycle).second)
      r.reject("invalid/duplicate trace cycle");
    auto append = [&](const std::string &prefix, bool vmem,
                      std::vector<std::pair<int, int>> &stream) {
      int64_t active = r.integer(row, prefix);
      if (active != 0 && active != 1) r.reject("invalid trace request flag");
      if (!active) return;
      int64_t element = vmem ? r.integer(row, prefix + "_line")
                            : r.integer(row, prefix + "_reg") * 32 +
                                  r.integer(row, prefix + "_row");
      if (element < 0 || element > 65535) r.reject("invalid trace element");
      stream.emplace_back(cycle, element);
    };
    append(sourcePrefix, load, out.source);
    append(destinationPrefix, !load, out.destination);
    int64_t busy = r.integer(row, load ? "load_busy" : "store_busy");
    if (busy == 1) out.busy.push_back(cycle);
    else if (busy != 0) r.reject("invalid busy flag");
  }
  if (complete != 1 || out.source.size() != 32 ||
      out.destination.size() != 32 || out.busy.size() != 34) {
    r.reject("incomplete standalone stream/busy evidence"); return out;
  }
  for (int i = 0; i < 32; ++i) {
    if (out.source[i] != std::make_pair(1 + i, (load ? 64 : 96) + i) ||
        out.destination[i] != std::make_pair(3 + i, (load ? 96 : 64) + i))
      r.reject("unsupported observed stream domain");
  }
  for (int i = 0; i < 34; ++i)
    if (out.busy[i] != i + 1) r.reject("unsupported observed busy interval");
  out.rows = out.source.size();
  out.sourceAge = out.source.front().first;
  out.destinationAge = out.destination.front().first;
  out.sourceLast = out.source.back().first;
  out.destinationLast = out.destination.back().first;
  out.step = out.source[1].first - out.source[0].first;
  out.busyLast = out.busy.back();
  return out;
}

void compareObserved(Reader &r, const Array *comparisons,
                     const TraceFacts &facts, bool load) {
  int streams = 0, releases = 0, coverage = 0;
  int sourceStreams = 0, destinationStreams = 0;
  if (!comparisons) return;
  for (const auto &value : *comparisons) {
    const Object *item = value.getAsObject();
    if (!item) { r.reject("malformed comparison"); continue; }
    if (r.str(item, "operation") != (load ? "vload" : "vstore")) continue;
    if (auto category = item->getString("comparison")) {
      if (*category == "stream_coverage") {
        ++coverage; r.boolIs(item, "matches_required_stream_coverage", true);
      } else if (*category == "releases_and_path") {
        ++releases; r.boolIs(item, "matches_release_and_holds", true);
        if (r.integer(item, "observed_last_source_access") != facts.sourceLast ||
            r.integer(item, "observed_last_destination_access") != facts.destinationLast ||
            r.integer(item, "observed_first_free_age") != facts.busyLast + 1)
          r.reject("observation/trace release mismatch");
        const Array *busy = r.array(item, "observed_busy_ages");
        if (busy && busy->size() == facts.busy.size()) {
          for (size_t i = 0; i < busy->size(); ++i)
            if ((*busy)[i].getAsInteger() != facts.busy[i])
              r.reject("observation/trace busy mismatch");
        } else r.reject("missing observed busy ages");
      } else r.reject("unknown observation kind");
      continue;
    }
    ++streams; r.boolIs(item, "matches_normalized_stream", true);
    auto resource = r.str(item, "resource");
    auto write = item->getBoolean("write");
    if (!write || (resource != "MReg" && resource != "Vmem") ||
        *write != (resource == "MReg" ? load : !load))
      r.reject("invalid observation role");
    bool isSource = resource == (load ? "Vmem" : "MReg");
    if (isSource) ++sourceStreams;
    else ++destinationStreams;
    const auto &stream = isSource ? facts.source : facts.destination;
    const Array *events = r.array(item, "observed_absolute");
    if (!events || events->size() != stream.size()) {
      r.reject("missing observed stream"); continue;
    }
    for (size_t i = 0; i < stream.size(); ++i) {
      const Object *event = (*events)[i].getAsObject();
      if (r.integer(event, "cycle") != stream[i].first ||
          r.integer(event, "element") != stream[i].second)
        r.reject("observation/trace stream mismatch");
    }
  }
  if (streams != 2 || sourceStreams != 1 || destinationStreams != 1 ||
      releases != 1 || coverage != 1)
    r.reject("missing/duplicate observation coverage");
}
} // namespace

llvm::Error mlir::atlas::timing::checkRTLSemanticCompatibility(
    llvm::StringRef hardwareIR) {
  struct Binding { const char *module; const char *sha256; };
  static const Binding reviewed[] = {
#include "AtlasRTLModuleFingerprints.inc"
  };
  std::map<std::string, std::string> expected;
  for (const auto &binding : reviewed)
    expected.emplace(binding.module, binding.sha256);
  std::set<std::string> seen;
  size_t begin = 0;
  while (begin < hardwareIR.size()) {
    size_t end = hardwareIR.find('\n', begin);
    if (end == llvm::StringRef::npos) end = hardwareIR.size();
    else ++end;
    llvm::StringRef line = hardwareIR.substr(begin, end - begin);
    if (!line.starts_with("  hw.module")) { begin = end; continue; }
    size_t symbol = line.find('@');
    if (symbol == llvm::StringRef::npos) { begin = end; continue; }
    size_t tail = symbol + 1;
    while (tail < line.size() && line[tail] != '(' && line[tail] != '<' &&
           line[tail] != ' ' && line[tail] != '\t' && line[tail] != '\n') ++tail;
    std::string name = line.substr(symbol + 1, tail - symbol - 1).str();
    auto binding = expected.find(name);
    if (binding == expected.end()) { begin = end; continue; }
    if (!seen.insert(name).second)
      return failure("duplicate reviewed hardware module " + name);
    bool internal = line.starts_with("  hw.module ");
    size_t moduleEnd = end;
    if (internal) {
      bool closed = false;
      while (moduleEnd < hardwareIR.size()) {
        size_t next = hardwareIR.find('\n', moduleEnd);
        if (next == llvm::StringRef::npos) next = hardwareIR.size();
        else ++next;
        llvm::StringRef bodyLine = hardwareIR.substr(moduleEnd, next - moduleEnd);
        moduleEnd = next;
        if (bodyLine.starts_with("  }")) { closed = true; break; }
      }
      if (!closed) return failure("unterminated reviewed hardware module " + name);
    }
    auto normalized = normalizeModule(hardwareIR.substr(begin, moduleEnd - begin));
    if (digest(normalized) != binding->second)
      return failure("unreviewed hardware semantics for " + name +
                     " (atlas.vls.conservative.v1 compatibility mismatch)");
    begin = moduleEnd;
  }
  for (const auto &binding : expected)
    if (!seen.count(binding.first))
      return failure("missing reviewed hardware module " + binding.first);
  return llvm::Error::success();
}

llvm::Expected<RTLEvidence> mlir::atlas::timing::loadRTLEvidence(
    llvm::StringRef path, const ExpectedEvidenceIdentity &expected,
    bool conditionalOptIn) {
  if (!conditionalOptIn) return failure("conditional evidence requires explicit opt-in");
  if (!hashSyntax(expected.evidenceSha256) ||
      !hashSyntax(expected.manifestSha256) || !hashSyntax(expected.hardwareIRSha256))
    return failure("all three expected SHA-256 identities are required");
  Reader r;
  auto safe = safePath(path, fs::path());
  if (!safe) return safe.takeError();
  fs::path base = fs::path(*safe).parent_path();
  std::string bytes = r.read(*safe, base);
  if (!r.error.empty()) return failure(r.error);
  if (digest(bytes) != expected.evidenceSha256)
    return failure("selected evidence hash mismatch");
  auto parsed = llvm::json::parse(bytes);
  if (!parsed) return failure(llvm::toString(parsed.takeError()));
  const Object *report = parsed->getAsObject();
  if (!report) return failure("report must be an object");
  r.stringIs(report, "schema", "atlas.conditional_vls_hw_check.v0");
  r.stringIs(report, "state", "conditional_checks_passed");
  r.stringIs(report, "target_config", "EE290SimConfig");
  constexpr const char *reference = "https://github.com/ucb-ee194-tapeout/bringup-chipyard/blob/main/generators/chipyard/src/main/scala/config/AtlasConfigs.scala#L12-L25";
  r.stringIs(report, "target_reference", reference);
  r.boolIs(report, "rtl_rules_enabled", false);
  r.boolIs(report, "conditional_replay_expectations_met", true);
  r.boolIs(report, "compiler_comparison_matches", true);
  const Array *bindings = r.array(report, "resolver_bindings");
  if (bindings && !bindings->empty()) r.reject("unexpected report resolver binding");
  const Object *scope = r.object(report, "scope");
  r.stringIs(scope, "execution", "selected standalone LSU hardware with injected one-cycle synchronous SRAM responses");
  r.stringIs(scope, "other_engines", "quiescent");
  r.stringIs(scope, "external_traffic", "none");
  r.stringIs(scope, "addresses", "known aligned in-range 32-line tiles wholly in one VMEM bank");
  r.stringIs(scope, "mregs", "distinct logical registers during overlap; same-register load then store only at gap35 after busy release");
  r.stringIs(scope, "cycle_origin", "command-valid/op sampled on rising edge age zero; pre-edge observation");
  r.boolIs(scope, "opaque_sram_implementation_qualified", false);
  r.boolIs(scope, "integrated_target_execution_qualified", false);
  const Object *extraction = r.object(report, "extraction");
  r.stringIs(extraction, "module", "LSU");
  r.boolIs(extraction, "assertions_preserved", true);
  if (r.integer(extraction, "instances") != 0) r.reject("non-standalone LSU");
  const Object *inputs = r.object(report, "inputs");
  const Object *manifestID = r.object(inputs, "manifest");
  const Object *hwID = r.object(inputs, "hardware_ir");
  if (r.str(manifestID, "sha256") != expected.manifestSha256 ||
      r.str(hwID, "sha256") != expected.hardwareIRSha256)
    r.reject("selected manifest/hardware identity mismatch");
  // Use the immutable manifest copy retained beside the replay rather than a
  // mutable producer path; its byte identity must equal the selected manifest.
  std::string manifestBytes = r.read("retention-manifest.json", base);
  if (r.error.empty() && (digest(manifestBytes) != expected.manifestSha256 ||
      static_cast<int64_t>(manifestBytes.size()) != r.integer(manifestID, "bytes")))
    r.reject("retained manifest snapshot mismatch");
  if (!r.error.empty()) return failure(r.error);
  auto manifestJSON = llvm::json::parse(manifestBytes);
  if (!manifestJSON) return failure(llvm::toString(manifestJSON.takeError()));
  const Object *manifest = manifestJSON->getAsObject();
  r.stringIs(manifest, "schema", "atlas.retained_hw_ir.v0");
  r.stringIs(manifest, "state", "verified");
  r.stringIs(manifest, "target_config", "EE290SimConfig");
  r.stringIs(manifest, "target_reference", reference);
  r.boolIs(manifest, "original_inputs_unchanged", true);
  const Object *manifestHW = r.object(manifest, "hardware_ir");
  if (!sameIdentity(hwID, manifestHW)) r.reject("hardware representation crosslink mismatch");
  fs::path manifestBase = fs::path(r.str(manifestID, "path")).parent_path();
  if (!manifestBase.is_absolute()) manifestBase = base / manifestBase;
  std::string hardware = r.artifact(hwID, base);
  if (!r.error.empty()) return failure(r.error);
  if (auto mismatch = checkRTLSemanticCompatibility(hardware)) return mismatch;
  const Object *originals = r.object(manifest, "inputs");
  const Object *snapshots = r.object(manifest, "snapshots");
  for (const char *key : {"firrtl", "annotations", "lowering_options"}) {
    const Object *original = r.object(originals, key);
    const Object *snapshot = r.object(snapshots, key);
    if (!sameIdentity(original, snapshot)) r.reject("source/snapshot identity mismatch");
    r.artifact(snapshot, manifestBase);
  }
  if (originals && snapshots && (originals->get("chisel_annotations") ||
                                  snapshots->get("chisel_annotations"))) {
    const Object *original = r.object(originals, "chisel_annotations");
    const Object *snapshot = r.object(snapshots, "chisel_annotations");
    if (!sameIdentity(original, snapshot)) r.reject("source/snapshot identity mismatch");
    r.artifact(snapshot, manifestBase);
  }
  r.artifact(r.object(manifest, "prepared_annotations"), manifestBase);
  const Array *producerCommands = r.array(manifest, "commands");
  std::set<std::string> producerStages;
  if (producerCommands) for (const auto &value : *producerCommands) {
    const Object *command = value.getAsObject();
    std::string stage = r.str(command, "stage");
    if (!producerStages.insert(stage).second ||
        r.integer(command, "returncode") != 0)
      r.reject("invalid retained-HW producer command");
    r.artifact(r.object(command, "log"), manifestBase);
  }
  if (producerStages != std::set<std::string>{"lower", "verify"})
    r.reject("missing retained-HW lowering/verification provenance");
  const Object *structure = r.object(manifest, "structure");
  const Object *requiredModules = r.object(structure, "required_modules");
  for (const char *name : {"AtlasCore", "ScalarCore", "DmaEngine"})
    r.boolIs(requiredModules, name, true);
  const Object *sequential = r.object(structure, "sequential_op_occurrences");
  if (r.integer(sequential, "seq.firreg") < 0 ||
      r.integer(sequential, "seq.compreg") < 0 ||
      r.integer(sequential, "seq.firreg") + r.integer(sequential, "seq.compreg") == 0)
    r.reject("missing retained sequential boundaries");
  const Object *sources = r.object(inputs, "sources");
  const Object *copies = r.object(report, "snapshots");
  for (const char *key : {"checker", "indexer", "replay_harness", "compiler_probe", "timing_cpp", "timing_header"}) {
    const Object *source = r.object(sources, key);
    const Object *snapshot = r.object(copies, key);
    if (!sameIdentity(source, snapshot)) r.reject("replay source snapshot mismatch");
    r.artifact(snapshot, base);
  }
  const Object *artifacts = r.object(report, "artifacts");
  std::string raw = r.artifact(r.object(artifacts, "LSU.raw.mlir"), base);
  std::string preamble = r.artifact(r.object(artifacts, "preamble.raw.mlir"), base);
  r.artifact(r.object(artifacts, "LSU.mlir"), base);
  std::string verilog = r.artifact(r.object(artifacts, "LSU.sv"), base);
  if (raw.empty() || hardware.find(raw) == std::string::npos ||
      preamble.empty() || hardware.find(preamble) == std::string::npos ||
      raw.find("sv.fatal") == std::string::npos || raw.find("sv.error") == std::string::npos ||
      verilog.find("$fatal") == std::string::npos ||
      verilog.find("VLOAD and VSTORE target same VMEM bank") == std::string::npos)
    r.reject("hardware slice/assertion provenance mismatch");
  const Array *commands = r.array(report, "commands");
  std::map<std::string, const Object *> commandByStage;
  if (commands) for (const auto &value : *commands) {
    const Object *command = value.getAsObject();
    std::string stage = r.str(command, "stage");
    if (!commandByStage.emplace(stage, command).second) r.reject("duplicate command stage");
    r.artifact(r.object(command, "stdout"), base);
    r.artifact(r.object(command, "stderr"), base);
  }
  auto exportIt = commandByStage.find("export-lsu");
  if (exportIt == commandByStage.end()) r.reject("missing assertion-preserving export");
  else {
    const Array *argv = r.array(exportIt->second, "argv");
    bool assertions = false;
    if (argv) for (const auto &arg : *argv)
      if (arg.getAsString() == "--lower-verif-to-sv") assertions = true;
    if (!assertions || r.integer(exportIt->second, "returncode") != 0)
      r.reject("invalid assertion-preserving export");
  }
  TraceFacts facts[2];
  int isolated[2] = {0, 0};
  const Array *cases = r.array(report, "replay_cases");
  if (!cases || cases->size() != 20) r.reject("incomplete replay case coverage");
  if (cases) for (const auto &value : *cases) {
    const Object *record = value.getAsObject();
    r.boolIs(record, "expectation_met", true);
    std::string sequence = r.str(record, "sequence");
    std::string expectedResult = r.str(record, "expected");
    if ((sequence != "L" && sequence != "S") || expectedResult != "pass") continue;
    int which = sequence == "L" ? 0 : 1;
    ++isolated[which];
    if (r.integer(record, "response_delay_cycles") != 1 ||
        r.integer(record, "returncode") != 0) r.reject("unsupported standalone replay environment");
    std::string stage = r.str(record, "log_stage");
    auto it = commandByStage.find(stage);
    if (it == commandByStage.end()) { r.reject("missing replay trace command"); continue; }
    if (r.integer(it->second, "returncode") != 0) r.reject("failed standalone replay");
    std::string trace = r.artifact(r.object(it->second, "stdout"), base);
    facts[which] = parseTrace(r, trace, which == 0);
  }
  if (isolated[0] != 1 || isolated[1] != 1) r.reject("missing/duplicate isolated replay");
  const Array *identityChecks = r.array(report, "identity_rejection_checks");
  std::set<std::string> rejectedFields;
  if (identityChecks) for (const auto &value : *identityChecks) {
    const Object *check = value.getAsObject();
    rejectedFields.insert(r.str(check, "field"));
    r.boolIs(check, "rejected_before_output_creation", true);
  }
  if (!identityChecks || identityChecks->size() != 2 ||
      rejectedFields != std::set<std::string>{"sha256", "bytes"})
    r.reject("incomplete input identity rejection evidence");
  const Array *comparisons = r.array(report, "footprint_comparisons");
  compareObserved(r, comparisons, facts[0], true);
  compareObserved(r, comparisons, facts[1], false);
  if (!r.error.empty()) return failure(r.error);
  RTLEvidence result;
  result.identity = expected;
  auto copy = [](const TraceFacts &f) {
    return RTLEvidence::Stream{f.sourceAge, f.destinationAge, f.step, f.rows,
                               f.busyLast, f.sourceLast, f.destinationLast};
  };
  result.load = copy(facts[0]); result.store = copy(facts[1]);
  return result;
}

llvm::json::Object RTLEvidence::applicability() const {
  return Object{
      {"scope", "straight_line_serialized_vls_with_scalar_setup_and_completion"},
      {"supported_operations", Array{"addi", "lui", "vload", "vstore", "delay",
                                     "csrrw", "ecall"}},
      {"operand_domain", Object{
          {"scalar_registers", Object{{"first", 0}, {"count", 32}}},
          {"mreg_registers", Object{{"first", 0}, {"count", 64}}},
          {"vmem_banks", kVmemBanks}, {"vmem_bank_bytes", kVmemBankBytes},
          {"vmem_line_bytes", kLineBytes}, {"tile_rows", load.rows},
          {"vmem_start_line_alignment", 32},
          {"vmem_last_start_line", kVmemBytes / kLineBytes - 32},
          {"base_unit", "32_bit_word"},
          {"effective_line", "(((base + 32 * sext12(offset)) mod 2^32) >> 3) & 0xffff"},
          {"known_base_required", true}, {"single_bank_required", true},
          {"mlir_offset_min", -2048}, {"mlir_offset_max", 2047},
          {"csr_constraint", "csrrw x0,0xC10,rs"},
          {"release_annotation_supported", false}}},
      {"environment_assumptions", Array{
          "reset_deasserted", "accelerator_quiescent_at_entry",
          "other_engines_and_competing_memory_requesters_quiescent",
          "sram_read_response_one_cycle_after_request",
          "instructions_immutable_during_execution",
          "host_does_not_access_vmem_or_rewrite_dbg0_during_execution",
          "caller_supplies_required_initial_operand_data"}},
      {"admission", Object{
          {"vls_paths_serialized", true},
          {"minimum_vls_issue_gap", std::max(load.busyLast, store.busyLast) + 1},
          {"marker_and_terminal_require_prior_writes_complete", true},
          {"delay_immediately_before_terminal_supported", false},
          {"maximum_program_words", static_cast<int64_t>(maximumProgramWords())}}},
      {"unsupported", Array{"dma_and_dynamic_completion", "other_compute_engines",
          "branches_and_loops", "concurrent_engine_or_host_memory_traffic",
          "alternative_memory_implementations_without_review"}},
      {"domain_qualification", "conditional; finite_observations_do_not_qualify_entire_domain"}};
}

Footprint RTLEvidence::resolve(const Instr &in, const RegValues &regs) const {
  Footprint f;
  auto reject = [&](const char *why) { f.error = std::string(resolverID()) + ": " + why; };
  if (!in.op || in.rd < 0 || in.rd > 63 || in.rs1 < 0 || in.rs1 > 31 ||
      in.rs2 < 0 || in.rs2 > 31) { reject("invalid instruction operands"); return f; }
  const std::string &name = in.op->name;
  auto x = [&](int reg, bool write) {
    if (reg) f.accesses.push_back({Res::XReg, write, reg, 1, 0, 1});
  };
  if (name == "addi" || name == "lui") {
    if (in.rd > 31 || in.release) { reject("unsupported scalar operands"); return f; }
    if (name == "addi") x(in.rs1, false);
    x(in.rd, true);
    if (in.rd) f.holds.push_back({Unit::ScalarWriteback, 0, 0, 0});
    return f;
  }
  if (name == "delay" || name == "ecall") {
    if (in.release) reject("unsupported completion annotation");
    return f;
  }
  if (name == "csrrw" && in.rd == 0 && in.imm == 0xC10) {
    x(in.rs1, false);
    return f;
  }
  bool isLoad = name == "vload";
  if (!isLoad && name != "vstore") { reject("operation is outside selected bounded scope"); return f; }
  if (in.release) { reject("unsupported VLS completion annotation"); return f; }
  if (in.imm < -2048 || in.imm > 4095) {
    reject("VLS immediate is not a 12-bit encoding"); return f;
  }
  if (!regs[in.rs1]) { reject("unknown VLS base cannot establish bounded address domain"); return f; }
  uint32_t effectiveWords = *regs[in.rs1] +
      static_cast<uint32_t>(signExtend(in.imm, 12) * 32);
  uint32_t line = (effectiveWords >> 3) & 0xffff;
  if (line % 32 || line + 32 > static_cast<uint32_t>(kVmemBytes / kLineBytes) ||
      line / 8192 != (line + 31) / 8192) {
    reject("VLS effective tile is unaligned, out of range, or crosses a bank"); return f;
  }
  const Stream &s = isLoad ? load : store;
  x(in.rs1, false);
  f.accesses.push_back({Res::Vmem, !isLoad, static_cast<int>(line), s.rows,
                       isLoad ? s.sourceAge : s.destinationAge, s.step});
  f.accesses.push_back({Res::MReg, isLoad, in.rd * 32, s.rows,
                       isLoad ? s.destinationAge : s.sourceAge, s.step});
  f.holds.push_back({Unit::VloadPath, 0, 0, s.busyLast});
  f.holds.push_back({Unit::VstorePath, 0, 0, s.busyLast});
  f.holds.push_back({Unit::VmemBank, static_cast<int>(line / 8192),
                     isLoad ? s.sourceAge : s.destinationAge,
                     isLoad ? s.sourceLast : s.destinationLast});
  if (isLoad) f.mregWrites = {in.rd};
  else f.mregReads = {in.rd};
  f.readRelease = isLoad ? 0 : s.busyLast;
  f.writeRelease = std::max(s.busyLast, s.destinationLast);
  f.writeDuringRead = false;
  f.doneAge = std::max(f.readRelease, f.writeRelease);
  return f;
}
