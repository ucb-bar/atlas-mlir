// Standalone selected-LSU replay with an explicitly one-cycle memory environment.
// This exercises the extracted hardware, not an integrated Atlas/SoC simulator.
#include "VLSU.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using Row = std::array<uint32_t, 8>;
struct Command { bool load; int cycle, reg, line; };

template <typename T> static void put(T &port, const Row &row) {
  for (int i = 0; i < 8; ++i) port[i] = row[i];
}
template <typename T> static Row get(const T &port) {
  Row row{};
  for (int i = 0; i < 8; ++i) row[i] = port[i];
  return row;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  if (argc != 5 && argc != 6) return 2;
  const std::string sequence = argv[1];
  const int gap = std::stoi(argv[2]);
  const bool sameBank = std::stoi(argv[3]);
  const bool sameRegister = std::stoi(argv[4]);
  const int responseDelay = argc == 6 ? std::stoi(argv[5]) : 1;
  if (sequence.empty() || sequence.size() > 2 || gap < 1 || gap > 80)
    return 2;
  if (responseDelay != 1 && responseDelay != 2) return 2;
  std::vector<Command> commands;
  for (size_t i = 0; i < sequence.size(); ++i) {
    if (sequence[i] != 'L' && sequence[i] != 'S') return 2;
    commands.push_back({sequence[i] == 'L', i ? gap : 0,
                        i && !sameRegister ? 7 : 3,
                        i ? (sameBank ? 128 : 8192 + 128) : 64});
  }
  std::vector<Row> vmem(6 * 8192), mreg(64 * 32);
  for (size_t line = 0; line < vmem.size(); ++line)
    for (int word = 0; word < 8; ++word)
      vmem[line][word] = 0x13570000u ^ uint32_t(line * 97 + word * 13);
  for (size_t line = 0; line < mreg.size(); ++line)
    for (int word = 0; word < 8; ++word)
      mreg[line][word] = 0xACE10000u ^ uint32_t(line * 101 + word * 17);
  auto expectedVmem = vmem;
  auto expectedMreg = mreg;
  // Independent whole-tile sequential specification. Same-register overlap is
  // a deliberate negative/unsupported domain, not a forwarding specification.
  for (const auto &command : commands)
    for (int row = 0; row < 32; ++row)
      if (command.load)
        expectedMreg[command.reg * 32 + row] = expectedVmem[command.line + row];
      else
        expectedVmem[command.line + row] = expectedMreg[command.reg * 32 + row];

  VLSU dut;
  dut.clock = 0;
  dut.reset = 1;
  dut.io_cmd_valid = 0;
  dut.io_scalarCmd_valid = 0;
  dut.io_scalarCmd_bits_isStore = 0;
  dut.io_scalarCmd_bits_byteAddr = 0;
  dut.io_scalarCmd_bits_wdata = 0;
  dut.io_scalarCmd_bits_wmask = 0;
  dut.io_vmemScalarReadData_valid = 0;
  dut.io_vmemVecReadData_valid = 0;
  dut.io_mregReadResp_valid = 0;
  Row zero{};
  put(dut.io_vmemScalarReadData_bits, zero);
  put(dut.io_vmemVecReadData_bits, zero);
  put(dut.io_mregReadResp_bits, zero);
  for (int i = 0; i < 3; ++i) {
    dut.clock = 0; dut.eval(); dut.clock = 1; dut.eval();
  }
  dut.reset = 0;
  std::vector<std::optional<Row>> vmemResponses(gap + 48), mregResponses(gap + 48);
  int reads = 0, writes = 0;
  for (int age = 0; age < gap + 45; ++age) {
    dut.clock = 0;
    dut.io_cmd_valid = 0;
    dut.io_vmemVecReadData_valid = vmemResponses[age].has_value();
    dut.io_mregReadResp_valid = mregResponses[age].has_value();
    put(dut.io_vmemVecReadData_bits, vmemResponses[age].value_or(zero));
    put(dut.io_mregReadResp_bits, mregResponses[age].value_or(zero));
    for (const auto &command : commands)
      if (command.cycle == age) {
        dut.io_cmd_valid = 1;
        dut.io_cmd_bits_op = command.load ? 1 : 2;
        dut.io_cmd_bits_mregBank = command.reg;
        dut.io_cmd_bits_vmemLineAddr = command.line;
      }
    dut.eval();
    bool launchBusy = dut.io_cmd_valid &&
        (dut.io_cmd_bits_op == 1 ? dut.io_vloadBusy : dut.io_vstoreBusy);
    bool logicalWriteHazard = dut.io_cmd_valid && dut.io_activeMregWrite_valid &&
        dut.io_cmd_bits_mregBank == dut.io_activeMregWrite_bits;
    // The integrated scalar frontend asserts this condition. It is not a
    // ready signal and is not implemented as a stall in this LSU slice.
    std::cout << "{\"cycle\":" << age
              << ",\"launch\":" << unsigned(dut.io_cmd_valid)
              << ",\"load_busy\":" << unsigned(dut.io_vloadBusy)
              << ",\"store_busy\":" << unsigned(dut.io_vstoreBusy)
              << ",\"read_active\":" << unsigned(dut.io_activeMregRead_valid)
              << ",\"write_active\":" << unsigned(dut.io_activeMregWrite_valid)
              << ",\"vmem_read\":" << unsigned(dut.io_vmemVecRead_valid)
              << ",\"vmem_read_line\":" << ((dut.io_vmemVecRead_bits_bankIdx << 13) | dut.io_vmemVecRead_bits_bankAddr)
              << ",\"vmem_write\":" << unsigned(dut.io_vmemVecWrite_valid)
              << ",\"vmem_write_line\":" << ((dut.io_vmemVecWrite_bits_bankIdx << 13) | dut.io_vmemVecWrite_bits_bankAddr)
              << ",\"mreg_read\":" << unsigned(dut.io_mregReadReq_valid)
              << ",\"mreg_read_reg\":" << unsigned(dut.io_mregReadReq_bits_mregId)
              << ",\"mreg_read_row\":" << unsigned(dut.io_mregReadReq_bits_row)
              << ",\"mreg_write\":" << unsigned(dut.io_mregWriteReq_valid)
              << ",\"mreg_write_reg\":" << unsigned(dut.io_mregWriteReq_bits_mregId)
              << ",\"mreg_write_row\":" << unsigned(dut.io_mregWriteReq_bits_row)
              << ",\"frontend_busy_violation\":" << launchBusy
              << ",\"frontend_logical_write_hazard\":" << logicalWriteHazard << "}\n";
    std::cout.flush();
    if (launchBusy || logicalWriteHazard) return 3;
    if (dut.io_vmemVecRead_valid) {
      const int line = (dut.io_vmemVecRead_bits_bankIdx << 13) | dut.io_vmemVecRead_bits_bankAddr;
      if (line >= int(vmem.size())) return 4;
      vmemResponses[age + responseDelay] = vmem.at(line);
      ++reads;
    }
    if (dut.io_mregReadReq_valid) {
      mregResponses[age + responseDelay] = mreg.at(dut.io_mregReadReq_bits_mregId * 32 + dut.io_mregReadReq_bits_row);
      ++reads;
    }
    if (dut.io_vmemVecWrite_valid) {
      const int line = (dut.io_vmemVecWrite_bits_bankIdx << 13) | dut.io_vmemVecWrite_bits_bankAddr;
      vmem.at(line) = get(dut.io_vmemVecWrite_bits_data);
      ++writes;
    }
    if (dut.io_mregWriteReq_valid) {
      mreg.at(dut.io_mregWriteReq_bits_mregId * 32 + dut.io_mregWriteReq_bits_row) = get(dut.io_mregWriteReq_bits_data);
      ++writes;
    }
    dut.clock = 1; dut.eval();
  }
  const bool correct = vmem == expectedVmem && mreg == expectedMreg &&
      reads == int(commands.size() * 32) && writes == int(commands.size() * 32);
  std::cout << "{\"complete\":true,\"data_and_guards_pass\":"
            << (correct ? "true" : "false") << ",\"reads\":" << reads
            << ",\"writes\":" << writes << "}\n";
  return correct ? 0 : 5;
}
