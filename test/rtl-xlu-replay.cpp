// Selected XLU slice with an explicit synchronous MREG response environment.
// Physical bank/row mapping is modeled; no other memory port is active.
#include "VXLU.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>

using Row = std::array<uint32_t, 8>;
using Memory = std::array<Row, 32 * 64>;
static int address(int id, int row) {
  return (id & 31) * 64 + (id >> 5) * 32 + row;
}
template <typename T> static void put(T &port, const Row &row) {
  for (int i = 0; i < 8; ++i) port[i] = row[i];
}
template <typename T> static Row get(const T &port) {
  Row row{};
  for (int i = 0; i < 8; ++i) row[i] = port[i];
  return row;
}
static uint8_t byte(const Row &row, int column) {
  return uint8_t(row[column / 4] >> (8 * (column % 4)));
}
static void transpose(Memory &memory, int source, int destination) {
  std::array<Row, 32> input{};
  for (int row = 0; row < 32; ++row)
    input[row] = memory[address(source, row)];
  for (int row = 0; row < 32; ++row) {
    Row output{};
    for (int column = 0; column < 32; ++column)
      output[column / 4] |= uint32_t(byte(input[column], row))
                            << (8 * (column % 4));
    memory[address(destination, row)] = output;
  }
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  if (argc != 8) return 2;
  const int source = std::stoi(argv[1]), destination = std::stoi(argv[2]);
  const int gap = std::stoi(argv[3]);
  const int secondSource = std::stoi(argv[4]);
  const int secondDestination = std::stoi(argv[5]);
  const int delay = std::stoi(argv[6]);
  const bool secondExpected = std::stoi(argv[7]);
  if (source < 0 || source >= 64 || destination < 0 || destination >= 64 ||
      gap < -1 || gap > 100 || secondSource < 0 || secondSource >= 64 ||
      secondDestination < 0 || secondDestination >= 64 || delay < 1 || delay > 2)
    return 2;
  Memory memory{};
  for (int id = 0; id < 64; ++id)
    for (int row = 0; row < 32; ++row)
      for (int column = 0; column < 32; ++column)
        memory[address(id, row)][column / 4] |=
            uint32_t(uint8_t(id * 29 + row * 7 + column * 11 + (row ^ column)))
              << (8 * (column % 4));
  Memory expected = memory;
  transpose(expected, source, destination);
  if (secondExpected) transpose(expected, secondSource, secondDestination);

  VXLU dut;
  Row zero{};
  dut.clock = 0;
  dut.reset = 1;
  dut.io_cmd_valid = 0;
  dut.io_cmd_bits_srcMregId = source;
  dut.io_cmd_bits_dstMregId = destination;
  dut.io_mregReadResp_valid = 0;
  put(dut.io_mregReadResp_bits, zero);
  for (int i = 0; i < 3; ++i) {
    dut.clock = 0; dut.eval(); dut.clock = 1; dut.eval();
  }
  dut.reset = 0;
  const int end = (gap < 0 ? 0 : gap) + 75;
  std::vector<std::optional<Row>> responses(end + 3);
  int reads = 0, writes = 0, captures = 0, busyLaunches = 0;
  for (int age = 0; age < end; ++age) {
    dut.clock = 0;
    dut.io_cmd_valid = age == 0 || age == gap;
    // Change inactive command operands after capture to distinguish metadata
    // latching from live port use, without creating another command.
    dut.io_cmd_bits_srcMregId = age == 0 ? source : secondSource;
    dut.io_cmd_bits_dstMregId = age == 0 ? destination : secondDestination;
    dut.io_mregReadResp_valid = responses[age].has_value();
    put(dut.io_mregReadResp_bits, responses[age].value_or(zero));
    dut.eval();
    const bool busy = dut.io_activeMregWrite_valid;
    const bool capture = dut.io_cmd_valid && !busy;
    const bool busyLaunch = dut.io_cmd_valid && busy;
    captures += capture;
    busyLaunches += busyLaunch;
    std::cout << "{\"cycle\":" << age
              << ",\"launch\":" << unsigned(dut.io_cmd_valid)
              << ",\"capture\":" << capture
              << ",\"busy_launch\":" << busyLaunch
              << ",\"read_active\":" << unsigned(dut.io_activeMregRead_valid)
              << ",\"read_active_reg\":" << unsigned(dut.io_activeMregRead_bits)
              << ",\"write_active\":" << unsigned(dut.io_activeMregWrite_valid)
              << ",\"write_active_reg\":" << unsigned(dut.io_activeMregWrite_bits)
              << ",\"mreg_read\":" << unsigned(dut.io_mregReadReq_valid)
              << ",\"mreg_read_reg\":" << unsigned(dut.io_mregReadReq_bits_mregId)
              << ",\"mreg_read_row\":" << unsigned(dut.io_mregReadReq_bits_row)
              << ",\"mreg_response\":" << unsigned(dut.io_mregReadResp_valid)
              << ",\"mreg_write\":" << unsigned(dut.io_mregWriteReq_valid)
              << ",\"mreg_write_reg\":" << unsigned(dut.io_mregWriteReq_bits_mregId)
              << ",\"mreg_write_row\":" << unsigned(dut.io_mregWriteReq_bits_row)
              << "}\n";
    if (dut.io_mregReadReq_valid) {
      responses[age + delay] = memory.at(address(
          dut.io_mregReadReq_bits_mregId, dut.io_mregReadReq_bits_row));
      ++reads;
    }
    if (dut.io_mregWriteReq_valid) {
      memory.at(address(dut.io_mregWriteReq_bits_mregId,
                        dut.io_mregWriteReq_bits_row)) =
          get(dut.io_mregWriteReq_bits_data);
      ++writes;
    }
    dut.clock = 1;
    dut.eval();
  }
  const int expectedCaptures = 1 + int(secondExpected);
  const bool correct = memory == expected && reads == expectedCaptures * 32 &&
                       writes == expectedCaptures * 32 &&
                       captures == expectedCaptures;
  std::cout << "{\"complete\":true,\"data_and_guards_pass\":"
            << (correct ? "true" : "false") << ",\"reads\":" << reads
            << ",\"writes\":" << writes << ",\"captures\":" << captures
            << ",\"busy_launches\":" << busyLaunches << "}\n";
  return correct ? 0 : 5;
}
