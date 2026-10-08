// Diagnostic inspection of the existing compiler timing API, not an RTL bound
// provider. No expected timing values are asserted here: a separately bound RTL
// replay must compare this report and account for its declared assumptions.
#include "Atlas/AtlasTiming.h"
#include <iostream>
#include <optional>
#include <string>

using namespace mlir::atlas::timing;

static void quoted(const std::string &s) {
  std::cout << '"';
  for (unsigned char c : s) {
    switch (c) {
    case '"': std::cout << "\\\""; break;
    case '\\': std::cout << "\\\\"; break;
    case '\n': std::cout << "\\n"; break;
    case '\r': std::cout << "\\r"; break;
    case '\t': std::cout << "\\t"; break;
    default:
      if (c < 32) {
        const char hex[] = "0123456789abcdef";
        std::cout << "\\u00" << hex[c >> 4] << hex[c & 15];
      } else {
        std::cout << static_cast<char>(c);
      }
    }
  }
  std::cout << '"';
}

static const char *resourceName(Res r) {
  switch (r) {
  case Res::XReg: return "XReg";
  case Res::EReg: return "EReg";
  case Res::MReg: return "MReg";
  case Res::Acc: return "Acc";
  case Res::Weight: return "Weight";
  case Res::Vmem: return "Vmem";
  case Res::DmaBase: return "DmaBase";
  }
  return "invalid";
}

static void ints(const std::vector<int> &values) {
  std::cout << '[';
  bool comma = false;
  for (int value : values) {
    if (comma) std::cout << ',';
    comma = true;
    std::cout << value;
  }
  std::cout << ']';
}

static Instr instruction(const char *operation, int mreg = 0,
                         long long offset = 0) {
  Instr in;
  in.op = findOp(operation);
  in.rd = mreg;
  in.rs1 = 1;
  in.imm = offset;
  return in;
}

static RegValues registers(std::optional<uint32_t> base) {
  RegValues regs = unknownRegs();
  regs[1] = base;
  return regs;
}

static void footprint(const Footprint &f) {
  std::cout << "{\"error\":";
  quoted(f.error);
  std::cout << ",\"accesses\":[";
  bool comma = false;
  for (const Access &a : f.accesses) {
    if (comma) std::cout << ',';
    comma = true;
    std::cout << "{\"resource\":";
    quoted(resourceName(a.res));
    std::cout << ",\"write\":" << (a.write ? "true" : "false")
              << ",\"first\":" << a.first << ",\"count\":" << a.count
              << ",\"age\":" << a.age << ",\"step\":" << a.step
              << ",\"last_age\":" << a.lastAge()
              << ",\"anywhere\":" << (a.anywhere ? "true" : "false")
              << ",\"at_completion\":"
              << (a.atCompletion ? "true" : "false") << '}';
  }
  std::cout << "],\"holds\":[";
  comma = false;
  for (const Hold &h : f.holds) {
    if (comma) std::cout << ',';
    comma = true;
    std::cout << "{\"unit\":";
    quoted(unitName(h.unit));
    std::cout << ",\"unit_id\":" << static_cast<int>(h.unit)
              << ",\"index\":" << h.index << ",\"from\":" << h.from
              << ",\"to\":" << h.to << ",\"alt\":" << h.alt << '}';
  }
  std::cout << "],\"mreg_reads\":";
  ints(f.mregReads);
  std::cout << ",\"mreg_writes\":";
  ints(f.mregWrites);
  std::cout << ",\"read_release\":" << f.readRelease
            << ",\"write_release\":" << f.writeRelease
            << ",\"write_during_read\":"
            << (f.writeDuringRead ? "true" : "false")
            << ",\"done_age\":" << f.doneAge << '}';
}

static void instance(const Instr &in, std::optional<uint32_t> base) {
  std::cout << "{\"operation\":";
  quoted(in.op->name);
  std::cout << ",\"mreg\":" << in.rd << ",\"base_word\":";
  if (base) std::cout << *base;
  else std::cout << "null";
  std::cout << ",\"offset\":" << in.imm << ",\"footprint\":";
  footprint(footprintOf(in, registers(base)));
  std::cout << '}';
}

int main(int argc, char **argv) {
  if (argc != 1 && !(argc == 2 && std::string(argv[1]) == "--suite")) {
    std::cerr << "usage: vls-timing-probe [--suite]\n";
    return 2;
  }
  std::cout << "{\"schema\":\"atlas.compiler_vls_probe.v0\","
               "\"source_api\":\"AtlasTiming footprintOf/dependence/ReservationTable\","
               "\"rtl_rules_enabled\":false,"
               "\"gap_unit\":\"issue cycles\","
               "\"hold_endpoints\":\"inclusive\","
               "\"geometry\":{\"vmem_bytes\":" << kVmemBytes
            << ",\"vmem_bank_bytes\":" << kVmemBankBytes
            << ",\"vmem_banks\":" << kVmemBanks
            << ",\"line_bytes\":" << kLineBytes << "},\"instances\":[";
  bool comma = false;
  for (const char *operation : {"vload", "vstore"}) {
    // These cases expose actual error coverage, including cases where current
    // model code accepts a conservative unknown instead of rejecting it.
    struct Case { int mreg; std::optional<uint32_t> base; long long offset; };
    for (const Case &c : {
           Case{0, 0, 0}, Case{1, 256, 0}, Case{32, 65536, 0},
           Case{63, 327680, 0}, Case{0, std::nullopt, 0},
           Case{0, 8, 0}, Case{0, 393216, 0},
           Case{0, 0, -1}, Case{0, 0, 8}, Case{0, 7, 0},
           Case{0, 524288, 0}}) {
      if (comma) std::cout << ',';
      comma = true;
      instance(instruction(operation, c.mreg, c.offset), c.base);
    }
  }
  std::cout << "],\"pair_matrix\":[";
  comma = false;
  struct AddressCase { const char *name; uint32_t base; };
  struct RegisterCase { const char *name; int mreg; };
  for (const char *firstOp : {"vload", "vstore"})
    for (const char *secondOp : {"vload", "vstore"})
      for (const AddressCase &addr : {
             AddressCase{"same_range", 0},
             AddressCase{"same_bank_disjoint_range", 256},
             AddressCase{"different_bank", 65536}})
        for (const RegisterCase &reg : {
               RegisterCase{"same_register", 0},
               RegisterCase{"different_register", 1},
               RegisterCase{"physical_bank_alias", 32}}) {
          if (comma) std::cout << ',';
          comma = true;
          Instr a = instruction(firstOp), b = instruction(secondOp, reg.mreg);
          Footprint fa = footprintOf(a, registers(0));
          Footprint fb = footprintOf(b, registers(addr.base));
          Dependence d = dependence(a, fa, b, fb);
          ReservationTable table;
          std::string firstAlone = table.conflict(a, fa, 0);
          std::string secondAlone = table.conflict(b, fb, 0);
          table.reserve(a, fa, 0);
          std::cout << "{\"first_operation\":";
          quoted(firstOp);
          std::cout << ",\"second_operation\":";
          quoted(secondOp);
          std::cout << ",\"address_relation\":";
          quoted(addr.name);
          std::cout << ",\"register_relation\":";
          quoted(reg.name);
          std::cout << ",\"first_base_word\":0,\"second_base_word\":"
                    << addr.base << ",\"first_mreg\":0,\"second_mreg\":"
                    << reg.mreg << ",\"dependence_distance\":" << d.distance
                    << ",\"dependence_kind\":";
          quoted(edgeKindName(d.kind));
          std::cout << ",\"dependence_reason\":";
          quoted(d.reason);
          std::cout << ",\"first_intrinsic_conflict\":";
          quoted(firstAlone);
          std::cout << ",\"second_intrinsic_conflict\":";
          quoted(secondAlone);
          std::cout << ",\"gaps\":[";
          for (int gap = 0; gap <= 36; ++gap) {
            if (gap) std::cout << ',';
            std::string conflict = table.conflict(b, fb, gap);
            bool dataOK = gap >= d.distance;
            // A single frontend cannot issue two instructions at age zero;
            // that admission fact is intentionally separate from these APIs.
            bool apiOK = dataOK && conflict.empty() && firstAlone.empty() &&
                         secondAlone.empty() && fa.error.empty() && fb.error.empty();
            std::cout << "{\"gap\":" << gap << ",\"dependence_satisfied\":"
                      << (dataOK ? "true" : "false")
                      << ",\"reservation_conflict\":";
            quoted(conflict);
            std::cout << ",\"compiler_api_allows\":"
                      << (apiOK ? "true" : "false") << '}';
          }
          std::cout << "]}";
        }
  std::cout << "]}\n";
}
