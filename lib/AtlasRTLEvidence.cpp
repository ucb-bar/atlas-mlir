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

namespace {
llvm::Expected<llvm::json::Value> readEvidenceJSON(Reader &r, const Object *id,
                                                 const fs::path &base) {
  std::string bytes = r.artifact(id, base);
  if (!r.error.empty()) return failure(r.error);
  auto parsed = llvm::json::parse(bytes);
  if (!parsed) return failure(llvm::toString(parsed.takeError()));
  if (!parsed->getAsObject()) return failure("artifact must contain a JSON object");
  return std::move(*parsed);
}

bool phaseInput(const Object *phase, const Object *identity) {
  auto *inputs = phase ? phase->getArray("inputs_before") : nullptr;
  if (inputs) for (const auto &value : *inputs) {
    auto *entry = value.getAsObject();
    if (entry && sameIdentity(entry->getObject("identity"), identity)) return true;
  }
  return false;
}

bool phaseOutput(const Object *phase, const Object *identity) {
  auto *outputs = phase ? phase->getArray("outputs") : nullptr;
  if (outputs) for (const auto &value : *outputs) {
    auto *entry = value.getAsObject();
    auto *files = entry ? entry->getArray("files") : nullptr;
    if (files) for (const auto &member : *files) {
      auto *file = member.getAsObject();
      if (file && sameIdentity(file->getObject("identity"), identity)) return true;
    }
  }
  return false;
}

void checkCapturedPhase(Reader &r, const Object *phase, const fs::path &base,
                        llvm::StringRef kind) {
  r.stringIs(phase, "schema", "atlas.ee290_captured_phase.v0");
  r.stringIs(phase, "state", "phase_completed");
  r.stringIs(phase, "target_config", "EE290SimConfig");
  r.stringIs(phase, "kind", kind);
  r.boolIs(phase, "scheduling_qualified", false);
  r.boolIs(phase, "inputs_stable", true);
  auto *command = r.object(phase, "command");
  if (r.integer(command, "returncode") != 0) r.reject("failed captured command");
  r.boolIs(command, "timed_out", false);
  r.artifact(r.object(command, "stdout"), base);
  r.artifact(r.object(command, "stderr"), base);
  auto *failures = r.array(phase, "failures");
  if (failures && !failures->empty()) r.reject("captured phase has failures");
  auto *before = r.array(phase, "inputs_before");
  auto *after = r.array(phase, "inputs_after");
  if (!before || !after || before->empty() || before->size() != after->size()) {
    r.reject("captured input list mismatch"); return;
  }
  const Object *tool = nullptr;
  for (size_t i = 0; i < before->size(); ++i) {
    auto *a = (*before)[i].getAsObject(), *b = (*after)[i].getAsObject();
    auto *identity = r.object(a, "identity");
    if (r.str(a, "role") != r.str(b, "role") ||
        !sameIdentity(identity, r.object(b, "identity")))
      r.reject("captured inputs changed");
    r.artifact(identity, base);
    if (r.str(a, "role") == "tool") {
      if (tool) r.reject("multiple captured tools");
      tool = identity;
    }
  }
  auto *argv = r.array(command, "argv");
  if (!tool || !argv || argv->empty() ||
      (*argv)[0].getAsString() != r.str(tool, "path"))
    r.reject("captured command/tool mismatch");
  auto *snapshots = r.object(phase, "snapshots");
  for (const char *name : {"producer", "path_checker_dependency"}) {
    auto *copy = r.object(snapshots, name);
    if (!sameIdentity(copy, r.object(phase, name)))
      r.reject("captured producer snapshot mismatch");
    r.artifact(copy, base);
  }
}

// Independent checks of the compact observed event stream. The producer also
// checks the VCD against the decoded program and numerical fixture. Neither
// check assigns a finite transfer latency to subsequent compiler programs.
void checkDMAEvents(Reader &r, const Object *transcript, const Object *record,
                    const fs::path &base, bool configOnly) {
  r.stringIs(transcript, "schema", "atlas.ee290_dma_events.v0");
  if (!sameIdentity(r.object(transcript, "trace"), r.object(record, "trace")) ||
      !sameIdentity(r.object(transcript, "words"), r.object(record, "words")))
    r.reject("DMA transcript/trace/program mismatch");
  auto *events = r.array(transcript, "events");
  std::string encoded = r.artifact(r.object(record, "words"), base);
  llvm::StringRef remaining(encoded);
  std::vector<uint32_t> words;
  while (!remaining.empty()) {
    auto line = remaining.split('\n'); remaining = line.second;
    auto text = line.first.trim();
    if (text.empty()) continue;
    uint32_t word = 0;
    if (text.size() != 8 || text.getAsInteger(16, word)) {
      r.reject("invalid DMA emitted word"); return;
    }
    words.push_back(word);
  }
  auto *instructions = r.array(transcript, "instructions");
  if (words.empty() || words.size() > RTLEvidence::maximumProgramWords() ||
      words.back() != 0x73 || !instructions || instructions->size() + 1 != words.size()) {
    r.reject("DMA transcript does not execute the complete selected program"); return;
  }
  std::map<std::pair<int64_t, std::string>, const Object *> commands;
  if (events) for (const auto &value : *events) {
    auto *event = value.getAsObject();
    auto kind = r.str(event, "kind");
    if (kind == "launch" || kind == "config" || kind == "wait" || kind == "marker")
      if (!commands.emplace(std::make_pair(r.integer(event, "edge"), kind), event).second)
        r.reject("duplicate DMA instruction event");
  }
  RegValues registers = unknownRegs();
  uint64_t configuredBase = 0;
  int64_t priorInstruction = -1;
  for (size_t index = 0; index < instructions->size(); ++index) {
    auto *instruction = (*instructions)[index].getAsObject();
    auto edge = r.integer(instruction, "edge");
    uint32_t word = words[index];
    if (r.integer(instruction, "word_index") != int64_t(index) ||
        r.integer(instruction, "word_u32") != word || edge <= priorInstruction)
      r.reject("DMA observed instruction differs from selected program");
    priorInstruction = edge;
    unsigned opcode = word & 127, rd = (word >> 7) & 31,
             rs1 = (word >> 15) & 31, rs2 = (word >> 20) & 31;
    auto command = [&](const char *kind) -> const Object * {
      auto it = commands.find({edge, kind});
      if (it == commands.end()) { r.reject("missing observed DMA instruction event"); return nullptr; }
      const Object *event = it->second;
      commands.erase(it);
      if (llvm::StringRef(kind) != "marker" &&
          (r.integer(event, "word_index") != int64_t(index) ||
           r.integer(event, "word_u32") != word))
        r.reject("DMA event word identity mismatch");
      return event;
    };
    if (opcode == 0x13 && ((word >> 12) & 7) == 0) {
      if (rd) registers[rd] = registers[rs1] ?
          std::optional<uint32_t>(*registers[rs1] + uint32_t(signExtend(word >> 20, 12))) : std::nullopt;
    } else if (opcode == 0x37) {
      if (rd) registers[rd] = word & 0xfffff000;
    } else if (opcode == 0x7b && (word >> 25) <= 1) {
      auto *event = command("launch");
      bool load = (word >> 25) == 0;
      auto vm = registers[load ? rd : rs1], offset = registers[load ? rs1 : rd], size = registers[rs2];
      if (!vm || !offset || !size) { r.reject("unknown captured DMA operands"); continue; }
      if (r.integer(event, "channel") != ((word >> 12) & 7) ||
          r.str(event, "op") != (load ? "load" : "store") ||
          r.integer(event, "vmem_word") != *vm || r.integer(event, "size") != *size ||
          r.integer(event, "dram_address") != int64_t((configuredBase << 32) | *offset))
        r.reject("DMA launch differs from decoded captured operands");
    } else if (opcode == 0x7f && (word >> 25) <= 1) {
      if (word >> 25) {
        if (r.integer(command("wait"), "channel") != ((word >> 12) & 7))
          r.reject("DMA wait differs from decoded channel");
      } else {
        auto *event = command("config");
        if (!registers[rs1] || *registers[rs1] > 31 ||
            r.integer(event, "base") != *registers[rs1])
          r.reject("DMA configuration differs from decoded scalar operand");
        else configuredBase = *registers[rs1];
      }
    } else if (opcode == 0x73 && (word >> 20) == 0xc10 &&
               ((word >> 12) & 7) == 1 && rd == 0 && registers[rs1] == 1) {
      command("marker");
    } else {
      r.reject("unsupported instruction in finite DMA capture witness");
    }
  }
  if (!commands.empty()) r.reject("unbound DMA instruction events");
  int64_t lastEdge = -1, launchEdge = -1, channel = -1, address = 0;
  int64_t beats = 0, requests = 0, responses = 0, launches = 0, waits = 0;
  int markers = 0, halts = 0;
  std::string direction;
  std::map<int64_t, int64_t> outstanding;
  if (events) for (const auto &value : *events) {
    const Object *event = value.getAsObject();
    const auto edge = r.integer(event, "edge");
    if (edge < 0 || edge < lastEdge || edge > 1000000 || halts)
      r.reject("invalid DMA event ordering");
    lastEdge = edge;
    const std::string kind = r.str(event, "kind");
    if (kind == "launch") {
      if (channel != -1) r.reject("overlapping DMA launches in serialized witness");
      channel = r.integer(event, "channel");
      address = r.integer(event, "dram_address");
      const auto words = r.integer(event, "vmem_word"), size = r.integer(event, "size");
      if (channel < 0 || channel > 7 || address < 0 || address % 32 ||
          size < 32 || size > 4096 || size % 32 ||
          address + size > (int64_t(1) << 37) ||
          (address & 0xffffffffLL) + size > (int64_t(1) << 32) ||
          words < 0 || words % 8 || words + size / 4 > kVmemBytes / 4 ||
          words / (kVmemBankBytes / 4) != (words + size / 4 - 1) / (kVmemBankBytes / 4))
        r.reject("DMA witness operands outside admitted domain");
      direction = r.str(event, "op");
      if (direction != "load" && direction != "store") r.reject("invalid DMA direction");
      beats = size / 32; requests = responses = 0; launchEdge = edge;
      outstanding.clear(); ++launches;
    } else if (kind == "config") {
      auto upper = r.integer(event, "base");
      if (upper < 0 || upper > 31) r.reject("DMA configuration outside selected address domain");
    } else if (kind == "a") {
      auto source = r.integer(event, "source");
      if (channel == -1 || edge <= launchEdge || requests >= beats || source < 0 ||
          source >= 64 || outstanding.count(source) ||
          r.integer(event, "address") != address + 32 * requests ||
          r.integer(event, "op") != (direction == "load" ? 4 : 0))
        r.reject("accepted DMA request does not match captured launch");
      outstanding[source] = edge; ++requests;
    } else if (kind == "d") {
      auto source = r.integer(event, "source");
      auto request = outstanding.find(source);
      if (channel == -1 || request == outstanding.end())
        r.reject("DMA response has no accepted request");
      else {
        if (edge - request->second < 43) r.reject("DMA witness lacks its declared delayed response");
        outstanding.erase(request);
      }
      ++responses;
    } else if (kind == "wait") {
      if (channel == -1 || r.integer(event, "channel") != channel ||
          requests != beats || responses != beats)
        r.reject("DMA wait precedes its matching transfer completion");
      if (!outstanding.empty()) r.reject("DMA wait leaves outstanding responses");
      channel = -1; ++waits;
    } else if (kind == "marker" || kind == "halt") {
      if (channel != -1) r.reject("DMA completion published while transfer pending");
      if (kind == "marker") ++markers;
      else {
        if (markers != 1 || edge <= priorInstruction)
          r.reject("DMA halt requires the complete observed instruction stream");
        ++halts;
      }
    } else {
      r.reject("unsupported DMA event kind");
    }
  }
  const auto name = r.str(record, "name");
  int expected = configOnly ? 0 : name == "waited_channel_reuse" ? 4 : 2;
  if (launches != expected || waits != expected || channel != -1 || markers != 1 || halts != 1)
    r.reject("incomplete DMA launch/wait/completion observations");
}
} // namespace

llvm::Error mlir::atlas::timing::loadDMAEvidence(RTLEvidence &evidence,
    llvm::StringRef path, llvm::StringRef expectedSha256) {
  if (!hashSyntax(expectedSha256)) return failure("DMA evidence SHA-256 is required");
  Reader r;
  auto safe = safePath(path, fs::path());
  if (!safe) return safe.takeError();
  const fs::path base = fs::path(*safe).parent_path();
  auto bytes = r.read(*safe, base);
  if (!r.error.empty()) return failure(r.error);
  if (digest(bytes) != expectedSha256) return failure("selected DMA evidence hash mismatch");
  auto json = llvm::json::parse(bytes);
  if (!json) return failure(llvm::toString(json.takeError()));
  const Object *report = json->getAsObject();
  r.stringIs(report, "schema", "atlas.ee290_dma_replay.v0");
  r.stringIs(report, "state", "finite_dma_cases_passed");
  r.stringIs(report, "target_config", "EE290SimConfig");
  r.stringIs(report, "qualification", "conditional");
  r.boolIs(report, "scheduling_qualified", false);
  auto *manifest = r.object(report, "manifest"), *hardware = r.object(report, "hardware_ir");
  if (r.str(manifest, "sha256") != evidence.manifestSha256() ||
      r.str(hardware, "sha256") != evidence.hardwareIRSha256())
    r.reject("DMA/VLS selected hardware identity mismatch");
  r.artifact(manifest, base); r.artifact(hardware, base);
  auto coreJSON = readEvidenceJSON(r, r.object(report, "selected_core_replay"), base);
  if (!coreJSON) return coreJSON.takeError();
  auto *core = coreJSON->getAsObject();
  r.stringIs(core, "schema", "atlas.selected_atlascore_replay.v0");
  r.stringIs(core, "state", "numerical_and_boundary_replay_passed");
  r.stringIs(core, "target_config", "EE290SimConfig");
  std::set<std::string> originalHashes;
  if (auto *originals = r.array(core, "original_inputs"))
    for (const auto &input : *originals) originalHashes.insert(r.str(input.getAsObject(), "sha256"));
  if (!originalHashes.count(evidence.manifestSha256()) ||
      !originalHashes.count(evidence.hardwareIRSha256()))
    r.reject("selected-core replay does not reference selected retained hardware");
  auto compileJSON = readEvidenceJSON(r, r.object(report, "compile_phase"), base);
  if (!compileJSON) return compileJSON.takeError();
  auto *compile = compileJSON->getAsObject();
  checkCapturedPhase(r, compile, base, "dma_verilator_compile");
  auto *model = r.object(report, "model");
  r.artifact(model, base);
  if (!phaseOutput(compile, model)) r.reject("DMA model is not a captured compiler output");
  std::set<std::string> coreSources, dmaSources;
  if (auto *inputs = r.array(core, "rtl_snapshots"))
    for (const auto &input : *inputs)
      coreSources.insert(r.str(r.object(input.getAsObject(), "snapshot"), "sha256"));
  if (auto *inputs = r.array(report, "rtl_snapshots"))
    for (const auto &input : *inputs) {
      auto *copy = r.object(input.getAsObject(), "snapshot");
      if (!sameIdentity(copy, r.object(input.getAsObject(), "original")) ||
          !phaseInput(compile, copy)) r.reject("DMA RTL snapshot/build mismatch");
      r.artifact(copy, base); dmaSources.insert(r.str(copy, "sha256"));
    }
  if (coreSources.empty() || coreSources != dmaSources)
    r.reject("DMA model does not use the selected AtlasCore source closure");
  auto *harness = r.object(report, "harness"), *harnessCopy = r.object(harness, "snapshot");
  if (!sameIdentity(harnessCopy, r.object(harness, "original")) || !phaseInput(compile, harnessCopy))
    r.reject("DMA replay harness/build mismatch");
  r.artifact(harnessCopy, base);
  auto *producerCopies = r.array(report, "producer_snapshots");
  if (!producerCopies || producerCopies->empty()) r.reject("missing DMA producer snapshots");
  if (producerCopies) for (const auto &entry : *producerCopies) {
    auto *copy = r.object(entry.getAsObject(), "snapshot");
    if (!sameIdentity(copy, r.object(entry.getAsObject(), "original")))
      r.reject("DMA producer snapshot mismatch");
    r.artifact(copy, base);
  }
  auto *argv = r.array(r.object(compile, "command"), "argv");
  bool assertions = false, atlasCore = false;
  if (argv) for (size_t i = 0; i < argv->size(); ++i) {
    assertions |= (*argv)[i].getAsString() == "--assert";
    atlasCore |= i > 0 && (*argv)[i-1].getAsString() == "--top-module" &&
                 (*argv)[i].getAsString() == "AtlasCore";
  }
  if (!assertions || !atlasCore) r.reject("DMA replay requires assertion-preserving AtlasCore compilation");
  std::set<std::string> names;
  if (auto *cases = r.array(report, "cases")) for (const auto &value : *cases) {
    auto *record = value.getAsObject();
    auto name = r.str(record, "name");
    if (!names.insert(name).second) r.reject("duplicate DMA replay case");
    auto *program = r.object(record, "program"), *words = r.object(record, "words"),
         *trace = r.object(record, "trace");
    r.artifact(program, base); r.artifact(words, base); r.artifact(trace, base);
    auto emissionJSON = readEvidenceJSON(r, r.object(record, "emission_phase"), base);
    if (!emissionJSON) return emissionJSON.takeError();
    auto *emission = emissionJSON->getAsObject();
    checkCapturedPhase(r, emission, base, "dma_program_emit");
    if (!phaseInput(emission, program) ||
        !phaseInput(emission, r.object(r.object(report, "tool_identities"), "atlas_emit")) ||
        !phaseOutput(emission, words))
      r.reject("DMA program/emitter/encoded-word mismatch");
    auto executionJSON = readEvidenceJSON(r, r.object(record, "execution_phase"), base);
    if (!executionJSON) return executionJSON.takeError();
    auto *execution = executionJSON->getAsObject();
    checkCapturedPhase(r, execution, base, "dma_execution");
    if (!phaseInput(execution, model) || !phaseInput(execution, words) ||
        !phaseOutput(execution, trace)) r.reject("DMA execution/model/words/trace mismatch");
    auto *command = r.object(execution, "command");
    auto *runArgv = r.array(command, "argv");
    if (!runArgv || runArgv->empty() || (*runArgv)[0].getAsString() != r.str(model, "path"))
      r.reject("DMA captured execution did not invoke the selected model");
    auto log = r.artifact(r.object(command, "stdout"), base);
    if (log.find("EE290_DMA_PASSED ") == std::string::npos ||
        log.find("status=5 marker=1 illegal_pc=0") == std::string::npos)
      r.reject("missing DMA numerical completion witness");
    auto *numerical = r.object(record, "numerical_checks");
    for (const char *key : {"completion", "captured_source", "output", "guards"})
      r.boolIs(numerical, key, true);
    auto transcript = readEvidenceJSON(r, r.object(record, "events"), base);
    if (!transcript) return transcript.takeError();
    checkDMAEvents(r, transcript->getAsObject(), record, base, name == "config_only");
  }
  if (names != std::set<std::string>{"config_only", "capture_after_launch",
      "before_launch_control", "capture_scalar_and_base_pending", "waited_channel_reuse"})
    r.reject("incomplete DMA replay case coverage");
  if (!r.error.empty()) return failure(r.error);
  evidence.dmaIdentity = expectedSha256.str();
  return llvm::Error::success();
}

TargetTiming RTLEvidence::targetTiming() const {
  TargetTiming target([evidence = *this](const Instr &in, const RegValues &regs) {
    return evidence.resolve(in, regs);
  });
  if (hasVmulEvidence())
    target.computePolicies.insert(TargetTiming::ComputePolicy::VmulBf16);
  return target;
}

llvm::json::Object RTLEvidence::applicability() const {
  Object result{
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
      {"unsupported", Array{"dma_and_dynamic_completion", "unselected_compute_operations",
          "branches_and_loops", "concurrent_engine_or_host_memory_traffic",
          "alternative_memory_implementations_without_review"}},
      {"domain_qualification", "conditional; finite_observations_do_not_qualify_entire_domain"}};
  if (hasDMAEvidence()) {
    result["scope"] = "straight_line_serialized_vls_and_wait_governed_dma";
    auto *operations = result.getArray("supported_operations");
    for (const char *name : {"dma.load.ch0..7", "dma.store.ch0..7",
                             "dma.config.ch0..7", "dma.wait.ch0..7"})
      operations->push_back(name);
    result["dma_domain"] = Object{
        {"global_pending_limit", 1}, {"channels", 8},
        {"vmem_pointer_unit", "32_bit_word"}, {"vmem_word_alignment", 8},
        {"size_unit", "byte"}, {"size_min", 32}, {"size_max", 4096},
        {"size_alignment", 32}, {"single_vmem_bank_required", true},
        {"dram_offset_alignment", 32}, {"dram_base_max", 31},
        {"dram_offset_wrap_supported", false},
        {"dram_address", "(configured_base << 32) | captured_offset"},
        {"scalar_and_base_reads", "captured_at_launch"},
        {"memory_lifetime", "launch_through_matching_wait"},
        {"completion_latency", nullptr},
        {"vmem_competition_while_pending", false},
        {"publication_requires_matching_wait", true}};
    result.getArray("environment_assumptions")->push_back(
        "fresh_host_START_resets_global_dma_base_to_zero; DMA_queues_already_quiescent");
    result.getArray("environment_assumptions")->push_back(
        "external_memory_responds_correctly_and_eventually; no_fixed_latency_bound");
    result["unsupported"] = Array{"multiple_pending_dma_transfers", "unselected_compute_operations",
        "branches_and_loops", "concurrent_engine_or_host_memory_traffic",
        "alternative_memory_implementations_without_review"};
  }
  if (hasXLUEvidence()) {
    result["scope"] = hasDMAEvidence() ? "straight_line_serialized_vls_xlu_and_wait_governed_dma" :
                                        "straight_line_serialized_vls_and_xlu";
    result.getArray("supported_operations")->push_back("vtrpose.xlu");
    result["xlu_domain"] = Object{
        {"source_register_min", 0}, {"source_register_max", 63},
        {"destination_register_min", 0}, {"destination_register_max", 63},
        {"in_place_supported", true}, {"physical_bank_alias_supported_when_serialized", true},
        {"source_age", transpose.sourceAge}, {"destination_age", transpose.destinationAge},
        {"rows", transpose.rows}, {"first_free_age", transpose.busyLast + 1},
        {"vls_and_xlu_serialized", true}, {"requires_dma_quiescent", true},
        {"mreg_response_cycles", 1}};
  }
  if (hasVmulEvidence()) {
    result["scope"] = hasDMAEvidence() ?
        "straight_line_serialized_selected_engines_and_wait_governed_dma" :
        "straight_line_serialized_selected_engines";
    result.getArray("supported_operations")->push_back("vmul.bf16");
    result["vmul_domain"] = Object{
        {"pair_base_min", 0}, {"pair_base_max", 62}, {"pair_base_alignment", 2},
        {"three_distinct_physical_pairs_required", true},
        {"physical_pair", "pair_base_modulo_32"},
        {"source_age", multiply.sourceAge}, {"destination_age", multiply.destinationAge},
        {"rows", multiply.rows}, {"first_free_age", multiply.busyLast + 1},
        {"read_release_age", multiply.sourceLast + 1},
        {"mreg_response_cycles", 1}, {"selected_engines_serialized", true},
        {"requires_dma_quiescent", true},
        {"numerical_witness", "finite_signed_normal_bf16_powers_of_two"},
        {"full_bf16_arithmetic_qualified", false}};
    result.getArray("environment_assumptions")->push_back(
        "vmul_timing_is_data_independent; finite_numerical_witness_is_not_full_arithmetic_validation");
  }
  return result;
}

Footprint RTLEvidence::resolve(const Instr &in, const RegValues &regs) const {
  Footprint f;
  auto reject = [&](const char *why) { f.error = std::string(selectedResolverID()) + ": " + why; };
  bool transposeOp = in.op && in.op->opClass == OpClass::Transpose;
  bool multiplyOp = in.op && in.op->engine == Engine::Vpu && in.op->name == "vmul.bf16";
  if (!in.op || in.rd < 0 || in.rd > 63 || in.rs1 < 0 ||
      in.rs1 > (transposeOp || multiplyOp ? 63 : 31) ||
      in.rs2 < 0 || in.rs2 > (multiplyOp ? 63 : 31)) {
    reject("invalid instruction operands"); return f;
  }
  const std::string &name = in.op->name;
  if (multiplyOp && hasVmulEvidence()) {
    if (in.release || in.imm != 0 || (in.rd & 1) || (in.rs1 & 1) || (in.rs2 & 1) ||
        (in.rd & 31) == (in.rs1 & 31) || (in.rd & 31) == (in.rs2 & 31) ||
        (in.rs1 & 31) == (in.rs2 & 31)) {
      reject("VMUL requires three distinct even physical register pairs and canonical encoding");
      return f;
    }
    const Stream &s = multiply;
    for (int source : {in.rs1, in.rs2})
      f.accesses.push_back({Res::MReg, false, source * 32, s.rows, s.sourceAge, s.step});
    f.accesses.push_back({Res::MReg, true, in.rd * 32, s.rows, s.destinationAge, s.step});
    f.mregReads = {in.rs1, in.rs1 + 1, in.rs2, in.rs2 + 1};
    f.mregWrites = {in.rd, in.rd + 1};
    f.holds.push_back({Unit::Vpu, 0, 0, s.busyLast});
    f.holds.push_back({Unit::VloadPath, 0, 0, s.busyLast});
    f.holds.push_back({Unit::VstorePath, 0, 0, s.busyLast});
    f.readRelease = s.sourceLast + 1; // Retain operands through their synchronous response.
    f.writeRelease = s.destinationLast;
    f.doneAge = s.busyLast;
    f.serializeWithDMA = true;
    return f;
  }
  if (transposeOp && hasXLUEvidence()) {
    if (in.release || in.rs2 != 0 || in.imm != 0) {
      reject("unsupported XLU encoding or release annotation"); return f;
    }
    const Stream &s = transpose;
    f.accesses.push_back({Res::MReg, false, in.rs1 * 32, s.rows, s.sourceAge, s.step});
    f.accesses.push_back({Res::MReg, true, in.rd * 32, s.rows, s.destinationAge, s.step});
    f.mregReads = {in.rs1}; f.mregWrites = {in.rd};
    f.holds.push_back({Unit::Xlu, 0, 0, s.busyLast});
    f.holds.push_back({Unit::VloadPath, 0, 0, s.busyLast});
    f.holds.push_back({Unit::VstorePath, 0, 0, s.busyLast});
    f.readRelease = s.sourceLast + 1;
    f.writeRelease = s.destinationLast;
    f.doneAge = s.busyLast;
    f.serializeWithDMA = true;
    return f;
  }
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
  if (hasDMAEvidence() && in.op->engine == Engine::Dma) {
    if (in.op->channel < 0 || in.op->channel >= 8 || in.release) {
      reject("unsupported DMA channel or completion annotation"); return f;
    }
    if (in.op->opClass == OpClass::DmaWait)
      return f; // The stream verifier checks the matching pending generation.
    if (in.op->opClass == OpClass::DmaConfig) {
      if (!regs[in.rs1] || *regs[in.rs1] > 31) {
        reject("DMA configuration requires a known 5-bit upper DRAM base"); return f;
      }
      x(in.rs1, false);
      f.accesses.push_back({Res::DmaBase, true, 0, 1, 0, 1});
      return f;
    }
    if (in.op->opClass != OpClass::DmaLoad && in.op->opClass != OpClass::DmaStore) {
      reject("unsupported DMA operation"); return f;
    }
    if (in.rd > 31 || !regs[in.rd] || !regs[in.rs1] || !regs[in.rs2]) {
      reject("DMA requires known scalar VMEM, DRAM and size operands"); return f;
    }
    const bool loadDMA = in.op->opClass == OpClass::DmaLoad;
    const uint64_t words = *regs[loadDMA ? in.rd : in.rs1],
                   offset = *regs[loadDMA ? in.rs1 : in.rd], bytes = *regs[in.rs2];
    if (bytes < 32 || bytes > 4096 || bytes % 32 || words % 8 ||
        words + bytes / 4 > uint64_t(kVmemBytes / 4) ||
        words / (kVmemBankBytes / 4) != (words + bytes / 4 - 1) / (kVmemBankBytes / 4) ||
        offset % 32 || offset + bytes > (uint64_t(1) << 32)) {
      reject("DMA transfer is unaligned, out of range, oversized, or crosses a bank/address boundary");
      return f;
    }
    x(in.rd, false); x(in.rs1, false); x(in.rs2, false);
    f.accesses.push_back({Res::DmaBase, false, 0, 1, 0, 1});
    f.accesses.push_back({Res::Vmem, loadDMA, int(words / 8), int(bytes / 32),
                          0, 0, false, true});
    // Until configured-base state becomes a shared range fact, conservatively
    // retain the entire external address space. This is not an alias proof.
    f.accesses.push_back({Res::Dram, !loadDMA, 0, 1, 0, 0, true, true});
    f.dmaAsync = true;
    f.exclusiveVmemUntilWait = true;
    return f; // Unknown external completion is never converted to a cycle estimate.
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

llvm::Error mlir::atlas::timing::loadXLUEvidence(RTLEvidence &evidence,
    llvm::StringRef path, llvm::StringRef expectedSha256) {
  if (!hashSyntax(expectedSha256)) return failure("XLU evidence SHA-256 is required");
  Reader r;
  auto safe = safePath(path, fs::path());
  if (!safe) return safe.takeError();
  const fs::path base = fs::path(*safe).parent_path();
  std::string bytes = r.read(*safe, base);
  if (!r.error.empty()) return failure(r.error);
  if (digest(bytes) != expectedSha256) return failure("selected XLU evidence hash mismatch");
  auto json = llvm::json::parse(bytes);
  if (!json) return failure(llvm::toString(json.takeError()));
  const Object *report = json->getAsObject();
  r.stringIs(report, "schema", "atlas.conditional_xlu_hw_check.v0");
  r.stringIs(report, "state", "conditional_checks_passed");
  r.stringIs(report, "target_config", "EE290SimConfig");
  r.boolIs(report, "rtl_rules_enabled", false);
  r.boolIs(report, "scheduling_qualified", false);
  r.boolIs(report, "integrated_target_execution_qualified", false);
  r.boolIs(report, "conditional_replay_expectations_met", true);
  const Object *scope = r.object(report, "scope");
  r.stringIs(scope, "execution", "selected standalone XluEngine");
  r.stringIs(scope, "cycle_origin", "capture edge age zero; pre-edge observation");
  r.stringIs(scope, "memory", "injected ordered synchronous MREG response, one cycle except discriminating delay2 case");
  r.stringIs(scope, "physical_mapping", "bank=id&31; row={id[5],logicalRow[4:0]}");
  r.stringIs(scope, "other_engines_and_external_traffic", "absent");
  const Object *inputs = r.object(report, "inputs");
  const Object *manifestID = r.object(inputs, "manifest");
  const Object *hardwareID = r.object(inputs, "hardware_ir");
  if (r.str(manifestID, "sha256") != evidence.manifestSha256() ||
      r.str(hardwareID, "sha256") != evidence.hardwareIRSha256() ||
      r.str(hardwareID, "sha256") != "49fa1794b3b389941dc816bf23a1e16b0190c51dd5277347c1353736a05a3ac6")
    r.reject("XLU/VLS selected hardware identity mismatch");
  std::string hardware = r.artifact(hardwareID, base);
  const Object *artifacts = r.object(report, "artifacts");
  const Object *retainedManifest = r.object(artifacts, "retention-manifest.json");
  if (!sameIdentity(manifestID, retainedManifest)) r.reject("XLU retained manifest mismatch");
  auto manifestJSON = readEvidenceJSON(r, retainedManifest, base);
  if (!manifestJSON) return manifestJSON.takeError();
  const Object *manifest = manifestJSON->getAsObject();
  r.stringIs(manifest, "schema", "atlas.retained_hw_ir.v0");
  r.stringIs(manifest, "state", "verified");
  r.stringIs(manifest, "target_config", "EE290SimConfig");
  if (!sameIdentity(hardwareID, r.object(manifest, "hardware_ir")))
    r.reject("XLU hardware/manifest mismatch");

  // Check the executed producer copies, rather than mutable working-tree files.
  // The pinned harness computes a byte transpose independently and compares the
  // entire physical memory, so its numerical/guard summary has a concrete origin.
  const Object *sources = r.object(inputs, "sources");
  const Object *copies = r.object(report, "snapshots");
  const std::map<std::string, std::string> producerPins = {
    {"checker", "19a840a9324c976cb9bda6b2e0cca1a981c305950a49088e5b6591998613fa4f"},
    {"replay_harness", "80ca1867251e55849b119ae0c729c1df8601fa5b52695aa5777514f6a41aca00"},
    {"vls_helpers", "0b79186a40f7eb9b6091c54c5ce15f0a6235fae68418900c18682f06dbe0a6d3"},
    {"indexer", "5205e3909a32f708d385818be3e09818e783f3aa58e580af1589b5f790e74d2e"}};
  if (!sources || !copies || sources->size() != producerPins.size() ||
      copies->size() != producerPins.size()) r.reject("incomplete XLU producer snapshots");
  for (const auto &pin : producerPins) {
    const Object *source = r.object(sources, pin.first);
    const Object *copy = r.object(copies, pin.first);
    if (!sameIdentity(source, copy) || r.str(copy, "sha256") != pin.second)
      r.reject("unreviewed XLU replay producer");
    r.artifact(copy, base);
  }
  const Object *inventoryID = r.object(inputs, "source_inventory");
  if (r.str(inventoryID, "sha256") != "939fee9cb515f3f1b62a8c6d6743a312284856efd9e8a51140f83c24b7782a23")
    r.reject("unreviewed XLU source inventory");
  auto inventoryJSON = readEvidenceJSON(r, inventoryID, base);
  if (!inventoryJSON) return inventoryJSON.takeError();
  const Object *sourceIDs = r.object(inputs, "source_snapshots");
  const Object *sourceCopies = r.object(report, "source_snapshot_artifacts");
  const std::map<std::string, std::string> sourcePins = {
    {"XLU.scala", "9015a665d5bdd11fb94766e32353b4b394fc0f561e1c24415c13003b0277a03a"},
    {"MregFile.scala", "4b9349ccfff146b5322180ab9e88e7a3adcc777a993beee8b863404ad3ddb79f"},
    {"MregParams.scala", "006b5190cbd5f457718d699222b1b304a6c5c78981706a6af4d3c8c123a1a93c"}};
  if (!sourceIDs || !sourceCopies || sourceIDs->size() != sourcePins.size() ||
      sourceCopies->size() != sourcePins.size()) r.reject("incomplete reviewed XLU sources");
  std::set<std::string> inventoried;
  if (const Array *entries = r.array(inventoryJSON->getAsObject(), "sources"))
    for (const auto &value : *entries) {
      const Object *entry = value.getAsObject();
      std::string name = fs::path(r.str(entry, "path")).filename().string();
      auto pin = sourcePins.find(name);
      if (pin == sourcePins.end()) continue;
      if (!inventoried.insert(name).second || r.str(entry, "sha256") != pin->second ||
          !sameIdentity(r.object(entry, "captured_copy"), r.object(sourceIDs, name)))
        r.reject("XLU source inventory/snapshot mismatch");
    }
  if (inventoried.size() != sourcePins.size()) r.reject("missing reviewed XLU inventory members");
  for (const auto &pin : sourcePins) {
    const Object *source = r.object(sourceIDs, pin.first);
    const Object *copy = r.object(sourceCopies, pin.first);
    if (!sameIdentity(source, copy) || r.str(copy, "sha256") != pin.second)
      r.reject("unreviewed XLU/MREG source");
    r.artifact(copy, base);
  }

  // Reconstruct the complete selected slice, including its builtin wrapper and
  // preamble, then remove only location references using the shared normalizer.
  std::string wrapper, selectedRaw, selectedPreamble;
  llvm::StringRef remaining(hardware);
  bool sawWrapper = false, firstModule = false, collecting = false, closed = false;
  int lineNumber = 0, firstLine = 0, lastLine = 0;
  while (!remaining.empty()) {
    auto parts = remaining.split('\n'); remaining = parts.second;
    llvm::StringRef line = parts.first;
    std::string exact = line.str() + "\n";
    ++lineNumber;
    if (line.starts_with("module ") || line == "module {") {
      if (sawWrapper) r.reject("duplicate XLU builtin wrapper");
      sawWrapper = true; wrapper = exact; continue;
    }
    if (line.starts_with("  hw.module")) firstModule = true;
    if (sawWrapper && !firstModule) selectedPreamble += exact;
    if (line.starts_with("  hw.module private @XluEngine(")) {
      if (firstLine) r.reject("duplicate selected XluEngine");
      collecting = true; firstLine = lineNumber;
    }
    if (collecting) {
      selectedRaw += exact;
      if (line.starts_with("  }")) {
        collecting = false; closed = true; lastLine = lineNumber;
      }
    }
  }
  std::string raw = r.artifact(r.object(artifacts, "XluEngine.raw.mlir"), base);
  std::string preamble = r.artifact(r.object(artifacts, "preamble.raw.mlir"), base);
  std::string standalone = r.artifact(r.object(artifacts, "XluEngine.mlir"), base);
  std::string verilog = r.artifact(r.object(artifacts, "XluEngine.sv"), base);
  r.artifact(r.object(artifacts, "XluEngine.lowered.mlir"), base);
  const Object *model = r.object(artifacts, "obj/VXLU");
  r.artifact(model, base);
  const Object *extraction = r.object(report, "extraction");
  r.stringIs(extraction, "module", "XluEngine");
  r.boolIs(extraction, "assertions_preserved", true);
  bool assertions = raw.find("sv.fatal") != std::string::npos ||
                    raw.find("sv.error") != std::string::npos ||
                    raw.find("verif.assert") != std::string::npos;
  r.boolIs(extraction, "selected_module_has_assertions", assertions);
  if (!sawWrapper || !closed || firstLine != r.integer(extraction, "first_line") ||
      lastLine != r.integer(extraction, "last_line") || raw != selectedRaw ||
      preamble != selectedPreamble ||
      standalone != normalizeModule(wrapper + preamble + raw + "}\n") ||
      r.integer(extraction, "instances") != 0 ||
      raw.find("hw.instance") != std::string::npos ||
      (assertions && verilog.find("$fatal") == std::string::npos))
    r.reject("selected XLU slice/assertion provenance mismatch");
  if (!r.error.empty()) return failure(r.error);

  const Object *tools = r.object(report, "tools");
  for (const char *name : {"circt_opt", "verilator", "verilator_executed", "cxx", "make", "ar"})
    r.artifact(r.object(tools, name), base);
  std::map<std::string, const Object *> commands;
  if (const Array *array = r.array(report, "commands")) for (const auto &value : *array) {
    const Object *command = value.getAsObject();
    if (!commands.emplace(r.str(command, "stage"), command).second ||
        r.integer(command, "returncode") != 0) r.reject("duplicate/failed XLU command");
    r.artifact(r.object(command, "stdout"), base);
    r.artifact(r.object(command, "stderr"), base);
  }
  auto recipe = [&](const std::string &stage, const std::vector<std::string> &expected) {
    auto it = commands.find(stage);
    if (it == commands.end()) { r.reject("missing XLU command " + stage); return; }
    const Array *argv = r.array(it->second, "argv");
    if (!argv || argv->size() != expected.size()) {
      r.reject("XLU command recipe mismatch " + stage); return;
    }
    for (size_t i = 0; i < expected.size(); ++i)
      if ((*argv)[i].getAsString() != expected[i]) r.reject("XLU command recipe mismatch " + stage);
  };
  auto artifactPath = [&](llvm::StringRef name) { return r.str(r.object(artifacts, name), "path"); };
  auto toolPath = [&](llvm::StringRef name) { return r.str(r.object(tools, name), "path"); };
  for (const char *name : {"circt_opt", "verilator", "cxx", "make", "ar"})
    recipe(std::string("version-") + name,
           {toolPath(std::string(name) == "verilator" ? "verilator_executed" : name), "--version"});
  recipe("export-xlu", {toolPath("circt_opt"), artifactPath("XluEngine.mlir"),
      "--verify-each", "--lower-seq-to-sv", "--lower-verif-to-sv", "--export-verilog",
      "-o", artifactPath("XluEngine.lowered.mlir")});
  auto exported = commands.find("export-xlu");
  if (exported == commands.end() ||
      !sameIdentity(r.object(exported->second, "stdout"), r.object(artifacts, "XluEngine.sv")))
    r.reject("XLU exported Verilog identity mismatch");
  const std::string modelPath = r.str(model, "path");
  const std::string objectDir = fs::path(modelPath).parent_path().string();
  recipe("verilate-xlu", {toolPath("verilator_executed"), "--cc", "--exe", "--assert",
      "--top-module", "XluEngine", "--prefix", "VXLU", "--Mdir", objectDir,
      "-Wno-fatal", artifactPath("XluEngine.sv"), r.str(r.object(copies, "replay_harness"), "path")});
  auto build = commands.find("build-xlu");
  if (build == commands.end()) r.reject("missing XLU model build");
  else {
    const Array *argv = r.array(build->second, "argv");
    if (!argv || argv->size() != 10) r.reject("invalid XLU build recipe");
    else {
      auto python = (*argv)[7].getAsString();
      if (!python || !python->starts_with("PYTHON3=")) r.reject("missing XLU build Python");
      recipe("build-xlu", {toolPath("make"), "-C", objectDir, "-f", "VXLU.mk",
          "CXX=" + toolPath("cxx"), "LINK=" + toolPath("cxx"),
          python ? python->str() : "", "AR=" + toolPath("ar"), "-j4"});
    }
  }

  struct Case { const char *name; int src, dst, gap, src2, dst2, delay; bool second; };
  const Case required[] = {
    {"distinct", 3, 7, -1, 22, 24, 1, false},
    {"in-place", 5, 5, -1, 22, 24, 1, false},
    {"paired-physical-bank", 3, 35, -1, 22, 24, 1, false},
    {"endpoints", 0, 63, -1, 22, 24, 1, false},
    {"reverse-endpoints", 63, 0, -1, 22, 24, 1, false},
    {"busy-unrelated-age1", 3, 7, 1, 22, 24, 1, false},
    {"busy-unrelated-age65", 3, 7, 65, 22, 24, 1, false},
    {"reuse-age66", 3, 7, 66, 7, 24, 1, true},
    {"two-cycle-response", 3, 7, -1, 22, 24, 2, false}};
  const Array *cases = r.array(report, "replay_cases");
  if (!cases || cases->size() != 9) r.reject("incomplete XLU replay coverage");
  std::map<std::string, const Object *> records;
  if (cases) for (const auto &value : *cases) {
    const Object *record = value.getAsObject();
    if (!records.emplace(r.str(record, "name"), record).second) r.reject("duplicate XLU replay case");
  }
  RTLEvidence::Stream observed;
  for (const Case &test : required) {
    auto found = records.find(test.name);
    if (found == records.end()) { r.reject("missing XLU replay case"); continue; }
    const Object *record = found->second;
    if (r.integer(record, "src") != test.src || r.integer(record, "dst") != test.dst ||
        r.integer(record, "gap") != test.gap || r.integer(record, "second_src") != test.src2 ||
        r.integer(record, "second_dst") != test.dst2 || r.integer(record, "delay") != test.delay ||
        r.integer(record, "returncode") != 0) r.reject("unsupported XLU replay case parameters");
    r.boolIs(record, "accept_second", test.second);
    r.boolIs(record, "expectation_met", true);
    const Object *checks = r.object(record, "checks");
    if (!checks || checks->size() != 12) r.reject("incomplete XLU declared checks");
    for (const char *name : {"contiguous_cycles", "source_stream", "response_stream",
        "destination_stream", "source_lifetime", "destination_and_engine_lifetime",
        "capture_events", "launch_events", "busy_launch_loss", "numerical_and_guard_result",
        "access_and_capture_counts", "busy_launch_count"}) r.boolIs(checks, name, true);
    const std::string stage = std::string("replay-") + test.name;
    r.stringIs(record, "trace_log_stage", stage);
    recipe(stage, {modelPath, std::to_string(test.src), std::to_string(test.dst),
        std::to_string(test.gap), std::to_string(test.src2), std::to_string(test.dst2),
        std::to_string(test.delay), test.second ? "1" : "0"});
    auto command = commands.find(stage);
    if (command == commands.end()) continue;
    std::string trace = r.artifact(r.object(command->second, "stdout"), base);
    std::vector<std::pair<int, int>> reads, writes, readActive, writeActive;
    std::vector<int> responses, launches, captures, losses;
    int expectedCycle = 0, completions = 0;
    llvm::StringRef log(trace);
    while (!log.empty()) {
      auto parts = log.split('\n'); log = parts.second;
      if (parts.first.empty()) continue;
      auto rowJSON = llvm::json::parse(parts.first);
      if (!rowJSON) { r.reject(llvm::toString(rowJSON.takeError())); continue; }
      const Object *row = rowJSON->getAsObject();
      if (!row) { r.reject("malformed XLU trace row"); continue; }
      if (row->get("complete")) {
        ++completions;
        r.boolIs(row, "complete", true); r.boolIs(row, "data_and_guards_pass", true);
        const Object *summary = r.object(record, "completion");
        r.boolIs(summary, "complete", true); r.boolIs(summary, "data_and_guards_pass", true);
        for (const char *field : {"reads", "writes", "captures", "busy_launches"})
          if (r.integer(row, field) != r.integer(summary, field)) r.reject("XLU completion/trace mismatch");
        const int count = test.second ? 2 : 1;
        if (r.integer(row, "reads") != 32 * count || r.integer(row, "writes") != 32 * count ||
            r.integer(row, "captures") != count ||
            r.integer(row, "busy_launches") != (test.gap >= 0 && !test.second ? 1 : 0))
          r.reject("XLU numerical/access/capture count mismatch");
        continue;
      }
      if (completions || r.integer(row, "cycle") != expectedCycle++) r.reject("noncontiguous XLU trace cycles");
      const int age = expectedCycle - 1;
      auto flag = [&](const char *key) {
        const int64_t value = r.integer(row, key);
        if (value != 0 && value != 1) r.reject("invalid XLU trace flag");
        return value == 1;
      };
      if (flag("launch")) launches.push_back(age);
      if (flag("capture")) captures.push_back(age);
      if (flag("busy_launch")) losses.push_back(age);
      if (flag("mreg_response")) responses.push_back(age);
      auto append = [&](const char *active, const char *id, const char *index,
                        std::vector<std::pair<int, int>> &out) {
        if (!flag(active)) return;
        const auto reg = r.integer(row, id);
        const auto element = index ? r.integer(row, index) : 0;
        if (reg < 0 || reg > 63 || element < 0 || element > 31) r.reject("invalid XLU register/row");
        out.emplace_back(age, index ? reg * 32 + element : reg);
      };
      append("mreg_read", "mreg_read_reg", "mreg_read_row", reads);
      append("mreg_write", "mreg_write_reg", "mreg_write_row", writes);
      append("read_active", "read_active_reg", nullptr, readActive);
      append("write_active", "write_active_reg", nullptr, writeActive);
    }
    std::vector<std::pair<int, int>> expectedReads, expectedWrites, expectedReadActive, expectedWriteActive;
    std::vector<int> expectedResponses, expectedCaptures = {0}, expectedLaunches = {0}, expectedLosses;
    auto transfer = [&](int origin, int src, int dst) {
      for (int i = 0; i < 32; ++i) {
        expectedReads.emplace_back(origin + 1 + i, src * 32 + i);
        expectedWrites.emplace_back(origin + 33 + test.delay + i, dst * 32 + i);
        expectedResponses.push_back(origin + 1 + test.delay + i);
      }
      for (int i = 1; i < 33 + test.delay; ++i) expectedReadActive.emplace_back(origin + i, src);
      for (int i = 1; i < 65 + test.delay; ++i) expectedWriteActive.emplace_back(origin + i, dst);
    };
    transfer(0, test.src, test.dst);
    if (test.gap >= 0) expectedLaunches.push_back(test.gap);
    if (test.second) { expectedCaptures.push_back(test.gap); transfer(test.gap, test.src2, test.dst2); }
    else if (test.gap >= 0) expectedLosses.push_back(test.gap);
    if (completions != 1 || expectedCycle != std::max(test.gap, 0) + 75 ||
        reads != expectedReads || writes != expectedWrites || responses != expectedResponses ||
        readActive != expectedReadActive || writeActive != expectedWriteActive ||
        captures != expectedCaptures || launches != expectedLaunches || losses != expectedLosses)
      r.reject("XLU observed stream/capture/release mismatch");
    if (std::string(test.name) == "distinct" && reads.size() == 32 && writes.size() == 32 &&
        writeActive.size() == 65)
      observed = {reads.front().first, writes.front().first,
          reads[1].first - reads[0].first, static_cast<int>(reads.size()),
          writeActive.back().first, reads.back().first, writes.back().first};
  }
  if (commands.size() != 17) r.reject("incomplete/extra XLU execution commands");
  const Object *facts = r.object(report, "resolved_facts");
  if (r.integer(facts, "read_age") != observed.sourceAge ||
      r.integer(facts, "write_age") != observed.destinationAge ||
      r.integer(facts, "read_count") != observed.rows ||
      r.integer(facts, "write_count") != observed.rows ||
      r.integer(facts, "read_step") != observed.step ||
      r.integer(facts, "write_step") != observed.step ||
      r.integer(facts, "read_release") != observed.sourceLast + 1 ||
      r.integer(facts, "write_release") != observed.destinationLast ||
      r.integer(facts, "first_free_age") != observed.busyLast + 1 ||
      r.integer(r.object(facts, "engine_hold"), "from") != 0 ||
      r.integer(r.object(facts, "engine_hold"), "to") != observed.busyLast)
    r.reject("XLU resolved facts disagree with observed events");
  if (!r.error.empty()) return failure(r.error);
  evidence.transpose = observed;
  evidence.xluIdentity = expectedSha256.str();
  return llvm::Error::success();
}

llvm::Error mlir::atlas::timing::loadVmulEvidence(RTLEvidence &evidence,
    llvm::StringRef path, llvm::StringRef expectedSha256) {
  if (!hashSyntax(expectedSha256)) return failure("VMUL evidence SHA-256 is required");
  Reader r;
  auto safe = safePath(path, fs::path());
  if (!safe) return safe.takeError();
  const fs::path base = fs::path(*safe).parent_path();
  std::string bytes = r.read(*safe, base);
  if (!r.error.empty()) return failure(r.error);
  if (digest(bytes) != expectedSha256) return failure("selected VMUL evidence hash mismatch");
  auto json = llvm::json::parse(bytes);
  if (!json) return failure(llvm::toString(json.takeError()));
  const Object *report = json->getAsObject();
  r.stringIs(report, "schema", "atlas.ee290_compute_replay.v0");
  r.stringIs(report, "state", "finite_compute_cases_passed");
  r.stringIs(report, "target_config", "EE290SimConfig");
  r.stringIs(report, "qualification", "conditional");
  r.boolIs(report, "scheduling_qualified", false);
  const Object *manifest = r.object(report, "manifest"), *hardware = r.object(report, "hardware_ir");
  if (r.str(manifest, "sha256") != evidence.manifestSha256() ||
      r.str(hardware, "sha256") != evidence.hardwareIRSha256() ||
      r.str(hardware, "sha256") != "49fa1794b3b389941dc816bf23a1e16b0190c51dd5277347c1353736a05a3ac6")
    r.reject("VMUL/VLS selected hardware identity mismatch");
  r.artifact(manifest, base); r.artifact(hardware, base);
  // The reviewed retained core receipt closes the hardware-to-Verilog link;
  // caller rehashing of a changed outer report cannot replace that closure.
  if (r.str(r.object(report, "selected_core_replay"), "sha256") !=
      "d02049fc151158df58658009aa757b9edf4ff1c3ea97c0a294d1577110600a01")
    r.reject("unreviewed VMUL selected-core source closure");
  auto coreJSON = readEvidenceJSON(r, r.object(report, "selected_core_replay"), base);
  if (!coreJSON) return coreJSON.takeError();
  const Object *core = coreJSON->getAsObject();
  r.stringIs(core, "schema", "atlas.selected_atlascore_replay.v0");
  r.stringIs(core, "state", "numerical_and_boundary_replay_passed");
  r.stringIs(core, "target_config", "EE290SimConfig");
  std::set<std::string> originalHashes;
  if (const Array *originals = r.array(core, "original_inputs"))
    for (const auto &value : *originals) originalHashes.insert(r.str(value.getAsObject(), "sha256"));
  if (!originalHashes.count(evidence.manifestSha256()) ||
      !originalHashes.count(evidence.hardwareIRSha256()))
    r.reject("VMUL selected-core replay does not reference selected hardware");
  auto compileJSON = readEvidenceJSON(r, r.object(report, "compile_phase"), base);
  if (!compileJSON) return compileJSON.takeError();
  const Object *compile = compileJSON->getAsObject();
  checkCapturedPhase(r, compile, base, "compute_verilator_compile");
  const Object *model = r.object(report, "model");
  r.artifact(model, base);
  if (!phaseOutput(compile, model)) r.reject("VMUL model is not a captured compiler output");
  std::set<std::pair<std::string, int64_t>> coreSources, computeSources;
  if (const Array *inputs = r.array(core, "rtl_snapshots")) for (const auto &value : *inputs) {
    const Object *copy = r.object(value.getAsObject(), "snapshot");
    coreSources.emplace(r.str(copy, "sha256"), r.integer(copy, "bytes"));
  }
  if (const Array *inputs = r.array(report, "rtl_snapshots")) for (const auto &value : *inputs) {
    const Object *entry = value.getAsObject(), *copy = r.object(entry, "snapshot");
    if (!sameIdentity(copy, r.object(entry, "original")) || !phaseInput(compile, copy) ||
        !computeSources.emplace(r.str(copy, "sha256"), r.integer(copy, "bytes")).second)
      r.reject("VMUL RTL snapshot/build mismatch");
    r.artifact(copy, base);
  }
  if (coreSources.empty() || computeSources != coreSources)
    r.reject("VMUL model does not use the selected AtlasCore source closure");
  const Object *harness = r.object(report, "harness"), *harnessCopy = r.object(harness, "snapshot");
  if (!sameIdentity(harnessCopy, r.object(harness, "original")) || !phaseInput(compile, harnessCopy) ||
      r.str(harnessCopy, "sha256") != "42015729dd089719fa3601aebbb8a933482f854435cd222828d42423a169ceb0")
    r.reject("unreviewed VMUL harness/build mismatch");
  r.artifact(harnessCopy, base);
  const std::map<std::string, std::string> producerPins = {
    {"replay-ee290-compute.py", "45685598a5c1ca8fbbb4b805ff975b4947646745d3fd60403cfe6a742b01a73f"},
    {"replay-ee290-dma.py", "6a86fe0c31047d77782b60c7b8235724368734ffe6feaa9233d77275151fc11d"},
    {"observe-ee290-vls.py", "e6b65d324005d71d2bf556f4ec217f56fa61fa621e9a909a68fdc315782e2190"},
    {"ee290_build_capture.py", "da3412786d5ff977a5980056e96724fa6380407132aff8f8c26a97b663d15efc"},
    {"check-ee290-provenance.py", "542d2b3de6d882107c282a00c53a70aceeae8ef5de515b3fff36935dd656b8e7"},
    {"fingerprint-rtl-modules.py", "9fad779a54b396254a5c5f27e74636f4178ae834c41937fe3ef3532950f24d3d"},
    {"index-retained-hw.py", "5205e3909a32f708d385818be3e09818e783f3aa58e580af1589b5f790e74d2e"}};
  std::set<std::string> producers;
  if (const Array *inputs = r.array(report, "producer_snapshots")) for (const auto &value : *inputs) {
    const Object *entry = value.getAsObject(), *copy = r.object(entry, "snapshot");
    const std::string name = fs::path(r.str(copy, "path")).filename().string();
    auto pin = producerPins.find(name);
    if (!producers.insert(name).second || pin == producerPins.end() ||
        r.str(copy, "sha256") != pin->second || !sameIdentity(copy, r.object(entry, "original")))
      r.reject("unreviewed VMUL producer snapshot");
    r.artifact(copy, base);
  }
  if (producers.size() != producerPins.size()) r.reject("missing VMUL producer snapshots");
  const Object *tools = r.object(report, "tool_identities");
  for (const char *name : {"verilator", "cxx", "make", "ar"})
    if (!phaseInput(compile, r.object(tools, name))) r.reject("VMUL compilation/tool mismatch");
  const Array *compileArgv = r.array(r.object(compile, "command"), "argv");
  bool assertions = false, selectedTop = false;
  if (compileArgv) for (size_t i = 0; i < compileArgv->size(); ++i) {
    assertions |= (*compileArgv)[i].getAsString() == "--assert";
    selectedTop |= i && (*compileArgv)[i - 1].getAsString() == "--top-module" &&
                   (*compileArgv)[i].getAsString() == "AtlasCore";
  }
  if (!assertions || !selectedTop) r.reject("VMUL requires assertion-preserving selected AtlasCore");
  std::set<std::pair<std::string, int64_t>> compiledSources;
  std::set<std::string> capturedSourcePaths, commandSourcePaths;
  std::vector<std::string> capturedSources;
  int capturedHarnesses = 0;
  if (const Array *inputs = r.array(compile, "inputs_before")) for (const auto &value : *inputs) {
    const Object *input = value.getAsObject(), *id = r.object(input, "identity");
    const std::string role = r.str(input, "role");
    if (role != "selected_rtl" && role != "harness") continue;
    if (!capturedSourcePaths.insert(r.str(id, "path")).second)
      r.reject("duplicate VMUL captured source path");
    capturedSources.push_back(r.str(id, "path"));
    if (role == "selected_rtl") {
      if (!compiledSources.emplace(r.str(id, "sha256"), r.integer(id, "bytes")).second)
        r.reject("duplicate VMUL captured RTL identity");
    } else if (++capturedHarnesses != 1 || !sameIdentity(id, harnessCopy))
      r.reject("VMUL compilation captured an unreviewed harness");
  }
  if (compileArgv) for (const auto &value : *compileArgv) {
    auto arg = value.getAsString();
    if (!arg) { r.reject("malformed VMUL compile argument"); continue; }
    const auto suffix = fs::path(arg->str()).extension().string();
    if (suffix != ".sv" && suffix != ".v" && suffix != ".cpp") continue;
    if (!commandSourcePaths.insert(arg->str()).second)
      r.reject("duplicate VMUL compilation source argument");
  }
  if (compiledSources != computeSources || capturedHarnesses != 1 ||
      capturedSourcePaths != commandSourcePaths)
    r.reject("VMUL compilation/source argument closure mismatch");
  // A copied --assert flag is insufficient when contradictory or unreviewed
  // switches can change the generated model. Admit the captured recipe whole.
  std::vector<std::string> reviewedCompile = {
      r.str(r.object(tools, "verilator"), "path"), "--cc", "--exe", "--build",
      "--top-module", "AtlasCore", "--prefix", "VAtlasCore", "--Mdir",
      fs::path(r.str(model, "path")).parent_path().string(), "--trace",
      "--trace-depth", "3", "--assert", "--output-split", "20000",
      "--output-split-cfuncs", "500", "-Wno-fatal", "-j", "4", "-CFLAGS",
      "-std=c++17", "-MAKEFLAGS",
      "CXX=" + r.str(r.object(tools, "cxx"), "path") + " LINK=" +
          r.str(r.object(tools, "cxx"), "path") + " AR=" +
          r.str(r.object(tools, "ar"), "path")};
  reviewedCompile.insert(reviewedCompile.end(), capturedSources.begin(), capturedSources.end());
  if (!compileArgv || compileArgv->size() != reviewedCompile.size())
    r.reject("unsupported VMUL compilation recipe");
  else for (size_t i = 0; i < reviewedCompile.size(); ++i)
    if ((*compileArgv)[i].getAsString() != reviewedCompile[i])
      r.reject("unsupported VMUL compilation recipe");

  std::set<std::string> names;
  std::optional<llvm::json::Value> vmulJSON;
  std::string vmulWords;
  if (const Array *cases = r.array(report, "cases")) for (const auto &value : *cases) {
    const Object *record = value.getAsObject();
    const std::string name = r.str(record, "name");
    if (!names.insert(name).second) r.reject("duplicate compute replay case");
    const Object *program = r.object(record, "program"), *words = r.object(record, "words"),
                 *trace = r.object(record, "trace");
    r.artifact(program, base);
    std::string encoded = r.artifact(words, base);
    r.artifact(trace, base);
    auto emissionJSON = readEvidenceJSON(r, r.object(record, "emission_phase"), base);
    if (!emissionJSON) return emissionJSON.takeError();
    const Object *emission = emissionJSON->getAsObject();
    checkCapturedPhase(r, emission, base, "compute_program_emit");
    if (!phaseInput(emission, program) || !phaseInput(emission, r.object(tools, "atlas_emit")) ||
        !phaseOutput(emission, words)) r.reject("compute program/emitter/word mismatch");
    const Array *emissionArgv = r.array(r.object(emission, "command"), "argv");
    const std::string emitScript = "import subprocess,sys; subprocess.run([sys.argv[1],sys.argv[2]],stdout=open(sys.argv[3],'wb'),check=True)";
    const Object *emittedWords = nullptr;
    if (const Array *outputs = r.array(emission, "outputs")) for (const auto &output : *outputs)
      if (const Array *files = r.array(output.getAsObject(), "files")) for (const auto &file : *files) {
        const Object *id = r.object(file.getAsObject(), "identity");
        if (sameIdentity(id, words)) {
          if (emittedWords) r.reject("duplicate compute emitted-word artifact");
          emittedWords = id;
        }
      }
    if (!emissionArgv || emissionArgv->size() != 6 ||
        (*emissionArgv)[1].getAsString() != "-c" ||
        (*emissionArgv)[2].getAsString() != emitScript ||
        (*emissionArgv)[3].getAsString() != r.str(r.object(tools, "atlas_emit"), "path") ||
        (*emissionArgv)[4].getAsString() != r.str(program, "path") ||
        !emittedWords || (*emissionArgv)[5].getAsString() != r.str(emittedWords, "path"))
      r.reject("compute emission command does not bind program/emitter/words");
    auto executionJSON = readEvidenceJSON(r, r.object(record, "execution_phase"), base);
    if (!executionJSON) return executionJSON.takeError();
    const Object *execution = executionJSON->getAsObject();
    checkCapturedPhase(r, execution, base, "compute_execution");
    if (!phaseInput(execution, model) || !phaseInput(execution, words) || !phaseOutput(execution, trace))
      r.reject("compute execution/model/words/trace mismatch");
    const Object *command = r.object(execution, "command");
    const Array *argv = r.array(command, "argv");
    if (!argv || argv->size() != 5 || (*argv)[0].getAsString() != r.str(model, "path") ||
        (*argv)[1].getAsString() != r.str(words, "path") ||
        (*argv)[2].getAsString() != r.str(trace, "path") ||
        (*argv)[3].getAsString() != "100000" || (*argv)[4].getAsString() != name)
      r.reject("compute execution command does not match selected fixture");
    std::string log = r.artifact(r.object(command, "stdout"), base);
    if (log.find("EE290_COMPUTE_PASSED mode=" + name + " checked_vmem_bytes=8192") == std::string::npos ||
        log.find("status=5 marker=1 illegal_pc=0") == std::string::npos)
      r.reject("missing compute numerical/drain witness");
    for (const char *key : {"completion", "output", "guards"})
      r.boolIs(r.object(record, "numerical_checks"), key, true);
    for (const char *key : {"selected_words", "data_streams", "observed_release", "composed_memory", "terminal_drain"})
      r.boolIs(r.object(record, "boundary_checks"), key, true);
    auto transcript = readEvidenceJSON(r, r.object(record, "events"), base);
    if (!transcript) return transcript.takeError();
    const Object *events = transcript->getAsObject();
    r.stringIs(events, "schema", "atlas.ee290_compute_events.v0");
    r.stringIs(events, "validator", "atlas.ee290.compute.fixed.v1");
    r.stringIs(events, "target_config", "EE290SimConfig");
    r.stringIs(events, "data_encoding", "little_endian_bytes_hex");
    r.stringIs(events, "sampling", "values immediately before rising clock timestamp");
    r.stringIs(events, "mode", name);
    if (!sameIdentity(r.object(events, "trace"), trace) || !sameIdentity(r.object(events, "words"), words))
      r.reject("compute event transcript/trace/word mismatch");
    if (name == "vmul") { vmulWords = std::move(encoded); vmulJSON = std::move(*transcript); }
  }
  if (names != std::set<std::string>{"xlu", "vmul", "dma_xlu"} || !vmulJSON)
    r.reject("incomplete compute replay coverage");
  if (!r.error.empty()) return failure(r.error);

  const Object *events = vmulJSON->getAsObject();
  std::vector<uint32_t> words;
  llvm::StringRef encoded(vmulWords);
  while (!encoded.empty()) {
    auto parts = encoded.split('\n'); encoded = parts.second;
    auto text = parts.first.trim();
    if (text.empty()) continue;
    uint32_t word = 0;
    if (text.size() != 8 || text.getAsInteger(16, word)) r.reject("invalid VMUL emitted word");
    words.push_back(word);
  }
  const Array *instructions = r.array(events, "instructions"), *commands = r.array(events, "commands");
  if (words.empty() || words.size() > RTLEvidence::maximumProgramWords() || words.back() != 0x73 ||
      !instructions || instructions->size() + 1 != words.size() || !commands || commands->size() != 7)
    return failure("incomplete finite VMUL instruction/command stream");
  std::map<int64_t, const Object *> commandByWord;
  for (const auto &value : *commands) {
    const Object *command = value.getAsObject();
    if (!commandByWord.emplace(r.integer(command, "word_index"), command).second)
      r.reject("duplicate compute command index");
  }
  using Row = std::vector<uint8_t>;
  std::vector<uint8_t> memory(8192);
  for (size_t i = 0; i < memory.size(); ++i) memory[i] = (0xa5 ^ (13 * i)) & 255;
  for (int i = 0; i < 1024; ++i) for (int side = 0; side < 2; ++side) {
    int exponent = side ? i % 3 - 1 : i % 5 - 2;
    bool negative = side ? i % 11 == 0 : i % 7 == 0;
    uint16_t value = (negative ? 0x8000 : 0) | ((127 + exponent) << 7);
    const int offset = side * 2048 + 2 * i;
    memory[offset] = value & 255; memory[offset + 1] = value >> 8;
  }
  const std::vector<uint8_t> initial = memory;
  std::map<int, Row> mregs;
  auto rowData = [&](const Object *event) {
    std::string hex = r.str(event, "data_hex");
    Row result(32);
    if (hex.size() != 64) { r.reject("malformed compute row data"); return result; }
    for (int i = 0; i < 32; ++i) {
      unsigned value = 0;
      auto byte = llvm::StringRef(hex).substr(2 * i, 2);
      if (byte.getAsInteger(16, value)) r.reject("malformed compute row byte");
      result[i] = value;
    }
    return result;
  };
  auto multiply = [&](const Row &a, const Row &b) {
    Row result(32);
    for (int i = 0; i < 32; i += 2) {
      const unsigned lhs = a[i] | (unsigned(a[i + 1]) << 8), rhs = b[i] | (unsigned(b[i + 1]) << 8);
      const int le = (lhs >> 7) & 255, re = (rhs >> 7) & 255, exponent = le + re - 127;
      if ((lhs & 127) || (rhs & 127) || le < 1 || le > 254 || re < 1 || re > 254 ||
          exponent < 1 || exponent > 254) r.reject("outside finite normal BF16 power reference");
      const unsigned value = ((lhs ^ rhs) & 0x8000) | (unsigned(exponent) << 7);
      result[i] = value & 255; result[i + 1] = value >> 8;
    }
    return result;
  };
  auto observedRow = [&](const Array *stream, int index, int64_t edge, int age,
                         int id, int row, int port, const Row &expected) {
    if (!stream || index < 0 || size_t(index) >= stream->size()) { r.reject("missing compute row event"); return; }
    const Object *event = (*stream)[index].getAsObject();
    if (r.integer(event, "edge") != edge + age || r.integer(event, "age") != age ||
        r.integer(event, "id") != id || r.integer(event, "row") != row ||
        (port >= 0 && r.integer(event, "port") != port) || rowData(event) != expected)
      r.reject("compute row identity/age/numerical mismatch");
  };
  std::array<std::optional<uint32_t>, 32> regs{}; regs[0] = 0;
  int64_t priorEdge = -1, release = -1, scalarAvailable = -1;
  int vlsCount = 0, vmulCount = 0;
  RTLEvidence::Stream observed;
  for (size_t index = 0; index < instructions->size(); ++index) {
    const Object *instruction = (*instructions)[index].getAsObject();
    const int64_t edge = r.integer(instruction, "edge");
    const uint32_t word = words[index];
    if (edge < 0 || edge > 1000000) return failure("invalid VMUL observed instruction edge");
    if (r.integer(instruction, "word_index") != int64_t(index) ||
        r.integer(instruction, "word_u32") != word || edge <= priorEdge || edge < scalarAvailable)
      r.reject("VMUL instruction/word/order mismatch");
    priorEdge = edge; scalarAvailable = edge + 1;
    const unsigned opcode = word & 127, rd = (word >> 7) & 31, rs1 = (word >> 15) & 31;
    if (opcode == 0x13 && ((word >> 12) & 7) == 0) {
      int32_t immediate = int32_t(word) >> 20;
      if (!regs[rs1]) r.reject("unknown captured scalar ADDI input");
      else if (rd) regs[rd] = *regs[rs1] + immediate;
    } else if (opcode == 0x37) {
      if (rd) regs[rd] = word & 0xfffff000;
    } else if (opcode == 0x67 && ((word >> 12) & 7) == 1) {
      scalarAvailable = edge + (word >> 20) + 1;
    } else if (opcode == 0x73) {
      if (word != 0xc1009073 || !regs[1] || *regs[1] != 1)
        r.reject("unsupported finite compute completion marker");
    } else if (opcode == 0x07 || opcode == 0x57) {
      auto found = commandByWord.find(index);
      if (found == commandByWord.end()) { r.reject("missing decoded compute command"); continue; }
      const Object *command = found->second;
      if (r.integer(command, "edge") != edge || r.integer(command, "word_u32") != word || edge < release)
        r.reject("compute command/capture/drain mismatch");
      const Array *reads = r.array(command, "reads"), *responses = r.array(command, "responses"),
                  *writes = r.array(command, "writes");
      if (opcode == 0x07) {
        r.stringIs(command, "engine", "vls");
        const bool store = word & (1 << 13);
        const int id = (word >> 7) & 63;
        const int32_t immediate = int32_t(word) >> 20;
        if (word & (1 << 14) || !regs[rs1]) { r.reject("unsupported/unknown raw VLS source"); continue; }
        const uint32_t address = *regs[rs1] + uint32_t(32 * immediate);
        const int line = (address >> 3) & 65535;
        if (line % 32 || line < 0 || line + 32 > 256) { r.reject("VLS outside finite VMEM"); continue; }
        r.stringIs(command, "op", store ? "store" : "load");
        if (r.integer(command, "mreg") != id || r.integer(command, "line") != line ||
            r.integer(command, "release_age") != 35 || r.integer(command, "release_edge") != edge + 35 ||
            !reads || reads->size() != 32 || !responses || responses->size() != 32 || !writes || writes->size() != 32)
          r.reject("VLS decoded operands/count/release mismatch");
        for (int row = 0; row < 32; ++row) {
          Row data;
          if (store) {
            auto foundRow = mregs.find(id * 32 + row);
            if (foundRow == mregs.end()) { r.reject("VSTORE reads uninitialized MREG"); data.resize(32); }
            else data = foundRow->second;
          } else data.assign(memory.begin() + 32 * (line + row), memory.begin() + 32 * (line + row + 1));
          observedRow(reads, row, edge, 1 + row, store ? id : line + row, row, -1, data);
          observedRow(responses, row, edge, 2 + row, store ? id : line + row, row, -1, data);
          observedRow(writes, row, edge, 3 + row, store ? line + row : id, row, -1, data);
          if (store) std::copy(data.begin(), data.end(), memory.begin() + 32 * (line + row));
          else mregs[id * 32 + row] = data;
        }
        release = edge + 35; ++vlsCount;
      } else {
        r.stringIs(command, "engine", "vmul"); r.stringIs(command, "op", "mul.bf16");
        const int lhs = (word >> 13) & 63, rhs = (word >> 19) & 63, dst = (word >> 7) & 63;
        if (word >> 25 != 3 || lhs % 2 || rhs % 2 || dst % 2 ||
            (lhs & 31) == (rhs & 31) || (dst & 31) == (lhs & 31) ||
            (dst & 31) == (rhs & 31) || r.integer(command, "lhs") != lhs ||
            r.integer(command, "rhs") != rhs || r.integer(command, "dst") != dst ||
            r.integer(command, "release_age") != 66 || r.integer(command, "release_edge") != edge + 66 ||
            !reads || reads->size() != 128 || !responses || responses->size() != 128 || !writes || writes->size() != 64)
          r.reject("VMUL decoded pair/count/release mismatch");
        for (int row = 0; row < 64; ++row) {
          Row inputs[2];
          for (int port = 0; port < 2; ++port) {
            const int id = (port ? rhs : lhs) + row / 32;
            auto foundRow = mregs.find(id * 32 + row % 32);
            if (foundRow == mregs.end()) { r.reject("VMUL reads uninitialized pair"); inputs[port].resize(32); }
            else inputs[port] = foundRow->second;
            observedRow(reads, 2 * row + port, edge, row, id, row % 32, port, inputs[port]);
            observedRow(responses, 2 * row + port, edge, row + 1, id, row % 32, port, inputs[port]);
          }
          Row result = multiply(inputs[0], inputs[1]);
          observedRow(writes, row, edge, row + 2, dst + row / 32, row % 32, 0, result);
          mregs[(dst + row / 32) * 32 + row % 32] = std::move(result);
        }
        // Derive enabled ages from the checked observed streams, not summaries.
        if (reads && reads->size() == 128 && writes && writes->size() == 64)
          observed = {int(r.integer((*reads)[0].getAsObject(), "age")),
              int(r.integer((*writes)[0].getAsObject(), "age")),
              int(r.integer((*reads)[2].getAsObject(), "age") - r.integer((*reads)[0].getAsObject(), "age")),
              int(writes->size()), int(r.integer((*writes)[63].getAsObject(), "age")),
              int(r.integer((*reads)[126].getAsObject(), "age")), int(r.integer((*writes)[63].getAsObject(), "age"))};
        release = edge + 66; ++vmulCount;
      }
      commandByWord.erase(found);
    } else r.reject("unsupported instruction in finite VMUL transcript");
  }
  if (!commandByWord.empty() || vlsCount != 6 || vmulCount != 1)
    r.reject("incomplete decoded VMUL command coverage");
  std::vector<uint8_t> expected = initial;
  for (int row = 0; row < 64; ++row) {
    Row a(initial.begin() + 32 * row, initial.begin() + 32 * (row + 1));
    Row b(initial.begin() + 2048 + 32 * row, initial.begin() + 2048 + 32 * (row + 1));
    Row result = multiply(a, b);
    std::copy(result.begin(), result.end(), expected.begin() + 4096 + 32 * row);
  }
  if (memory != expected || r.integer(events, "full_memory_checked_bytes") != 8192)
    r.reject("VMUL full-memory numerical/guard mismatch");
  const Array *endpoints = r.array(events, "endpoints");
  int64_t marker = -1, publication = -1, halt = -1;
  if (!endpoints || endpoints->size() != 3) r.reject("incomplete VMUL completion endpoints");
  if (endpoints) for (const auto &value : *endpoints) {
    const Object *endpoint = value.getAsObject();
    const std::string kind = r.str(endpoint, "kind");
    const int64_t edge = r.integer(endpoint, "edge");
    if (kind == "marker_publication") { if (publication != -1) r.reject("duplicate marker publication"); publication = edge; continue; }
    const Object *state = r.object(endpoint, "engine_state");
    if (!state || state->size() != 10) r.reject("incomplete terminal engine state");
    for (const char *name : {"vls_load_busy", "vls_store_busy", "xlu_read_active", "xlu_write_active",
        "vpu_read_mask", "vpu_write_mask", "dma_scalar_mask", "dma_engine_mask", "dma_a_valid", "dma_d_valid"})
      if (r.integer(state, name) != 0) r.reject("terminal precedes observed engine quiescence");
    if (kind == "marker") {
      if (marker != -1 || r.integer(endpoint, "word_index") != int64_t(words.size()) - 2 ||
          r.integer(endpoint, "word_u32") != 0xc1009073 || edge != priorEdge || edge < release)
        r.reject("marker precedes complete VMUL instruction/data drain");
      marker = edge;
    } else if (kind == "halt") {
      if (halt != -1 || r.integer(endpoint, "word_index") != int64_t(words.size()) - 1 ||
          r.integer(endpoint, "word_u32") != 0x73 || edge <= priorEdge || edge < release)
        r.reject("halt precedes complete VMUL instruction/data drain");
      halt = edge;
    } else r.reject("unsupported VMUL completion endpoint");
  }
  // This fixture performs no DMA; retained DMA busy observations must agree
  // with that decoded command stream rather than copied boundary flags.
  const Array *busy = r.array(events, "busy_transitions");
  if (!busy || busy->size() != 1 ||
      r.integer((*busy)[0].getAsObject(), "scalar_mask") != 0 ||
      r.integer((*busy)[0].getAsObject(), "engine_mask") != 0 ||
      r.integer((*busy)[0].getAsObject(), "edge") < 0 ||
      r.integer((*busy)[0].getAsObject(), "edge") > marker)
    r.reject("VMUL fixture has unexpected DMA busy transitions");
  const int64_t sampled = r.integer(events, "sampled_edges"), drain = r.integer(events, "terminal_drain_edges");
  if (marker < 0 || publication != marker + 1 || halt < publication || sampled <= halt ||
      drain != sampled - halt || drain < 2)
    r.reject("missing observed VMUL post-halt drain");
  if (!r.error.empty()) return failure(r.error);
  evidence.multiply = observed;
  evidence.vmulIdentity = expectedSha256.str();
  return llvm::Error::success();
}
