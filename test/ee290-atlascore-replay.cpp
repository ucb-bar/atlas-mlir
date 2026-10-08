// Drive the selected AtlasCore TileLink ports without a Chipyard/CPU harness.
// This replay deliberately does not claim full EE290 system qualification.
#include "VAtlasCore.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class Replay {
  VerilatedContext context;
  VAtlasCore dut{&context};
  VerilatedVcdC trace;
  uint64_t time = 0, cycles = 0, limit;

  void evaluate() {
    dut.eval();
    trace.dump(time++);
    if (context.gotFinish())
      throw std::runtime_error("RTL stopped or asserted");
  }

public:
  Replay(const char *wave, uint64_t cycleLimit) : limit(cycleLimit) {
    context.traceEverOn(true);
    dut.trace(&trace, 3);
    trace.open(wave);
    dut.clock = 0;
    dut.reset = 1;
    dut.io_imemTL_a_valid = dut.io_csrTL_a_valid = dut.io_vmemTL_a_valid = 0;
    dut.io_imemTL_d_ready = dut.io_csrTL_d_ready = dut.io_vmemTL_d_ready = 1;
    dut.io_dmaTL_a_ready = 1;
    dut.io_dmaTL_d_valid = 0;
    dut.io_dmaTL_d_bits_source = 0;
    for (unsigned i = 0; i < 8; ++i) dut.io_dmaTL_d_bits_data[i] = 0;
    evaluate();
    for (unsigned i = 0; i < 4; ++i) tick();
    dut.reset = 0;
    tick();
  }

  ~Replay() { dut.final(); trace.close(); }

  void tick() {
    if (++cycles > limit) throw std::runtime_error("cycle bound exceeded");
    dut.clock = 0;
    evaluate(); // distinct timestamp: inputs settle before the rising edge
    dut.clock = 1;
    evaluate();
    dut.clock = 0;
    evaluate();
    if (dut.io_dmaTL_a_valid) throw std::runtime_error("unexpected DMA request");
  }

  uint32_t scalar(bool csr, bool write, uint32_t address, uint32_t data = 0) {
    if (csr) {
      dut.io_csrTL_a_bits_opcode = write ? 0 : 4;
      dut.io_csrTL_a_bits_size = 2;
      dut.io_csrTL_a_bits_source = 0;
      dut.io_csrTL_a_bits_address = address;
      dut.io_csrTL_a_bits_data = data;
      dut.io_csrTL_a_valid = 1;
    } else {
      dut.io_imemTL_a_bits_opcode = write ? 0 : 4;
      dut.io_imemTL_a_bits_size = 2;
      dut.io_imemTL_a_bits_source = 0;
      dut.io_imemTL_a_bits_address = address;
      dut.io_imemTL_a_bits_data = data;
      dut.io_imemTL_a_valid = 1;
    }
    dut.clock = 0;
    evaluate();
    unsigned waits = 0;
    while (!(csr ? dut.io_csrTL_a_ready : dut.io_imemTL_a_ready)) {
      if (++waits > 100) throw std::runtime_error("scalar A handshake timeout");
      tick();
    }
    tick();
    if (csr) dut.io_csrTL_a_valid = 0;
    else dut.io_imemTL_a_valid = 0;
    evaluate();
    waits = 0;
    while (!(csr ? dut.io_csrTL_d_valid : dut.io_imemTL_d_valid)) {
      if (++waits > 100) throw std::runtime_error("scalar D handshake timeout");
      tick();
    }
    uint32_t result = csr ? dut.io_csrTL_d_bits_data : dut.io_imemTL_d_bits_data;
    auto opcode = csr ? dut.io_csrTL_d_bits_opcode : dut.io_imemTL_d_bits_opcode;
    if (opcode != (write ? 0 : 1)) throw std::runtime_error("scalar response opcode mismatch");
    tick();
    return result;
  }

  std::array<uint32_t, 8> memory(bool write, unsigned line,
                               const std::array<uint32_t, 8> &data = {}) {
    dut.io_vmemTL_a_bits_opcode = write ? 0 : 4;
    dut.io_vmemTL_a_bits_size = 5; // 32 bytes: one full physical line
    dut.io_vmemTL_a_bits_source = 0;
    dut.io_vmemTL_a_bits_address = 0x20000000U + 32 * line;
    dut.io_vmemTL_a_bits_mask = 0xffffffffU;
    for (unsigned i = 0; i < 8; ++i) dut.io_vmemTL_a_bits_data[i] = data[i];
    dut.io_vmemTL_a_valid = 1;
    dut.clock = 0;
    evaluate();
    unsigned waits = 0;
    while (!dut.io_vmemTL_a_ready) {
      if (++waits > 100) throw std::runtime_error("VMEM A handshake timeout");
      tick();
    }
    tick();
    dut.io_vmemTL_a_valid = 0;
    evaluate();
    waits = 0;
    while (!dut.io_vmemTL_d_valid) {
      if (++waits > 100) throw std::runtime_error("VMEM D handshake timeout");
      tick();
    }
    std::array<uint32_t, 8> result;
    for (unsigned i = 0; i < 8; ++i) result[i] = dut.io_vmemTL_d_bits_data[i];
    if (dut.io_vmemTL_d_bits_opcode != (write ? 0 : 1))
      throw std::runtime_error("VMEM response opcode mismatch");
    tick();
    return result;
  }

  static uint32_t pattern(unsigned index, unsigned panel) {
    uint32_t result = 0;
    for (unsigned byte = 0; byte < 4; ++byte) {
      unsigned n = 4 * index + byte;
      unsigned value = panel == 0 ? n : panel == 1 ? 37 * n + 19 :
          ((n & 1) ? 0x80U : 0U) ^ (n / 32);
      result |= (value & 255U) << (8 * byte);
    }
    return result;
  }

  static uint32_t initial(unsigned index, unsigned panel) {
    return index >= 256 && index < 512 ? pattern(index - 256, panel) :
        0xa5a55a5aU ^ (index * 0x01010101U);
  }

  void run(const std::vector<uint32_t> &program) {
    scalar(true, true, 0x40018, 0);
    if (!(scalar(true, false, 0x40008) & 1)) throw std::runtime_error("initial halt missing");
    for (unsigned i = 0; i < program.size(); ++i) scalar(false, true, 0x20000 + 4 * i, program[i]);
    for (unsigned i = 0; i < program.size(); ++i)
      if (scalar(false, false, 0x20000 + 4 * i) != program[i]) throw std::runtime_error("IMEM readback mismatch");
    for (unsigned panel = 0; panel < 3; ++panel) {
      for (unsigned line = 0; line < 192; ++line) {
        std::array<uint32_t, 8> data;
        for (unsigned word = 0; word < 8; ++word) data[word] = initial(8 * line + word, panel);
        memory(true, line, data);
      }
      for (uint32_t offset : {0x10U, 0x14U, 0U, 4U}) scalar(true, true, 0x40000 + offset, 0);
      scalar(true, true, 0x40018, 1);
      uint32_t status = 0, marker = 0;
      unsigned poll = 0;
      for (; poll < 100000; ++poll) {
        status = scalar(true, false, 0x40008);
        marker = scalar(true, false, 0x40010);
        if ((status & 1) && ((status >> 1) & 3)) break;
      }
      uint32_t illegal = scalar(true, false, 0x4000c);
      std::cout << "ATLASCORE_VLS_OBSERVED panel=" << panel << " status=" << status
                << " marker=" << marker << " illegal_pc=" << illegal << " polls=" << poll << '\n';
      if (status != 5 || marker != 1 || illegal != 0 || poll == 100000)
        throw std::runtime_error("settled completion mismatch");
      for (unsigned line = 0; line < 192; ++line) {
        auto actual = memory(false, line);
        for (unsigned word = 0; word < 8; ++word) {
          unsigned index = 8 * line + word;
          uint32_t expected = index >= 768 && index < 1024 ? pattern(index - 768, panel) : initial(index, panel);
          if (actual[word] != expected) {
            std::cerr << "memory mismatch panel=" << panel << " word=" << index << " expected="
                      << std::hex << expected << " actual=" << actual[word] << std::dec << '\n';
            throw std::runtime_error("output/guard mismatch");
          }
        }
      }
      std::cout << "ATLASCORE_VLS_PANEL_PASSED panel=" << panel << " output_words=256 preserved_words=1280\n";
    }
    std::cout << "ATLASCORE_VLS_PASSED panels=3 cycles=" << cycles << '\n';
  }
};

int main(int argc, char **argv) {
  if (argc != 4) { std::cerr << "words.hex trace.vcd max-cycles required\n"; return 2; }
  try {
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error("cannot read selected program words");
    std::vector<uint32_t> words;
    uint64_t word;
    while (input >> std::hex >> word) {
      if (word > 0xffffffffULL || words.size() >= 8192) throw std::runtime_error("invalid program word");
      words.push_back(word);
    }
    if (!input.eof() || words.empty()) throw std::runtime_error("malformed/empty program words");
    Replay replay(argv[2], std::stoull(argv[3]));
    replay.run(words);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ATLASCORE_VLS_FAILED " << error.what() << '\n';
    return 1;
  }
}
