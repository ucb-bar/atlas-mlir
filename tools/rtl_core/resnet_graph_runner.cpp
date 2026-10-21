// Drive the selected AtlasCore TileLink ports without a Chipyard/CPU harness.
// This replay deliberately does not claim full EE290 system qualification.
#include "VAtlasCore.h"
#include "atlas_host.h"
extern "C" {
#include "assets.h"
int atlas_resnet_graph_run(const float *, float *, void *,
                           int (*)(void *, unsigned, const uint8_t *, size_t, int, uint8_t *, size_t));
}
#include <cstring>
#include <iterator>

#include "verilated.h"
#include "verilated_vcd_c.h"
#include <array>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

class Replay {
  VerilatedContext context;
  VAtlasCore dut{&context};
  VerilatedVcdC trace;
  bool traceEnabled = false;
  bool dmaTrace = false;
  uint64_t time = 0, cycles = 0, limit;
  struct Response {
    uint64_t due;
    unsigned source;
    std::array<uint32_t, 8> data;
  };
  std::deque<Response> pending;
  std::map<uint64_t, uint8_t> dram;
  unsigned reads = 0, writes = 0;
  void service() {
    dut.io_dmaTL_a_ready = cycles % 11 < 4;
    dut.io_dmaTL_d_valid = !pending.empty() && pending.front().due <= cycles;
    if (dut.io_dmaTL_d_valid) {
      dut.io_dmaTL_d_bits_source = pending.front().source;
      for (unsigned i = 0; i < 8; ++i) dut.io_dmaTL_d_bits_data[i] = pending.front().data[i];
    }
  }
  void accept() {
    bool response = dut.io_dmaTL_d_valid && dut.io_dmaTL_d_ready;
    if (response) pending.pop_front();
    if (dut.io_dmaTL_a_valid && dut.io_dmaTL_a_ready) {
      auto addr = uint64_t(dut.io_dmaTL_a_bits_address);
      auto op = unsigned(dut.io_dmaTL_a_bits_opcode);
      if ((addr & 31) || (op != 0 && op != 4)) throw std::runtime_error("unsupported DMA request");
      Response r{cycles + 43, unsigned(dut.io_dmaTL_a_bits_source), {}};
      for (unsigned i = 0; i < 32; ++i) {
        if (!dram.count(addr + i)) throw std::runtime_error("DMA escaped initialized memory");
        if (op == 4)
          r.data[i / 4] |= uint32_t(dram.at(addr + i)) << (8 * (i % 4));
        else
          dram[addr + i] = (dut.io_dmaTL_a_bits_data[i / 4] >> (8 * (i % 4))) & 255;
      }
      if (op == 4)
        ++reads;
      else
        ++writes;
      pending.push_back(r);
      if (dmaTrace)
        std::cout << "DMA_A cycle=" << cycles << " opcode=" << op << " address=" << addr << " source=" << r.source << '\n';
    }
    if (response && dmaTrace) std::cout << "DMA_D cycle=" << cycles << '\n';
  }
  void evaluate() {
    dut.eval();
    if (traceEnabled) trace.dump(time);
    ++time;
    if (context.gotFinish()) throw std::runtime_error("RTL stopped or asserted");
  }
  void exchange(CData &valid, const CData &ready, const CData &response, const char *port) {
    auto wait = [&](const CData &signal, const char *phase) {
      unsigned waits = 0;
      while (!signal) {
        if (++waits > 100) throw std::runtime_error(std::string(port) + phase + " handshake timeout");
        tick();
      }
    };
    valid = 1;
    dut.clock = 0;
    evaluate();
    wait(ready, " A");
    tick();
    valid = 0;
    evaluate();
    wait(response, " D");
  }

public:
  Replay(const char *wave, uint64_t cycleLimit, bool logDma = false) : dmaTrace(logDma), limit(cycleLimit) {
    traceEnabled = std::string(wave) != "/dev/null";
    if (traceEnabled) {
      context.traceEverOn(true);
      dut.trace(&trace, 3);
      trace.open(wave);
    }
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
  ~Replay() {
    dut.final();
    if (traceEnabled) trace.close();
  }
  void tick() {
    if (++cycles > limit) throw std::runtime_error("cycle bound exceeded");
    dut.clock = 0;
    service();
    evaluate(); // inputs settle before the rising edge
    accept();
    dut.clock = 1;
    evaluate();
    dut.clock = 0;
    evaluate();
  }
  uint32_t scalar(bool csr, bool write, uint32_t address, uint32_t data = 0) {
    (csr ? dut.io_csrTL_a_bits_opcode : dut.io_imemTL_a_bits_opcode) = write ? 0 : 4;
    (csr ? dut.io_csrTL_a_bits_size : dut.io_imemTL_a_bits_size) = 2;
    (csr ? dut.io_csrTL_a_bits_source : dut.io_imemTL_a_bits_source) = 0;
    (csr ? dut.io_csrTL_a_bits_address : dut.io_imemTL_a_bits_address) = address;
    (csr ? dut.io_csrTL_a_bits_data : dut.io_imemTL_a_bits_data) = data;
    exchange(csr ? dut.io_csrTL_a_valid : dut.io_imemTL_a_valid, csr ? dut.io_csrTL_a_ready : dut.io_imemTL_a_ready,
             csr ? dut.io_csrTL_d_valid : dut.io_imemTL_d_valid, "scalar");
    uint32_t result = csr ? dut.io_csrTL_d_bits_data : dut.io_imemTL_d_bits_data;
    auto opcode = csr ? dut.io_csrTL_d_bits_opcode : dut.io_imemTL_d_bits_opcode;
    if (opcode != (write ? 0 : 1)) throw std::runtime_error("scalar response opcode mismatch");
    tick();
    return result;
  }
  std::array<uint32_t, 8> memory(bool write, unsigned line, const std::array<uint32_t, 8> &data = {}) {
    dut.io_vmemTL_a_bits_opcode = write ? 0 : 4;
    dut.io_vmemTL_a_bits_size = 5; // 32 bytes: one full physical line
    dut.io_vmemTL_a_bits_source = 0;
    dut.io_vmemTL_a_bits_address = ATLAS_VMEM_BASE + 32 * line;
    dut.io_vmemTL_a_bits_mask = 0xffffffffU;
    for (unsigned i = 0; i < 8; ++i) dut.io_vmemTL_a_bits_data[i] = data[i];
    exchange(dut.io_vmemTL_a_valid, dut.io_vmemTL_a_ready, dut.io_vmemTL_d_valid, "VMEM");
    std::array<uint32_t, 8> result;
    for (unsigned i = 0; i < 8; ++i) result[i] = dut.io_vmemTL_d_bits_data[i];
    if (dut.io_vmemTL_d_bits_opcode != (write ? 0 : 1)) throw std::runtime_error("VMEM response opcode mismatch");
    tick();
    return result;
  }
  // Memory contents come from the case image; every initialized byte is checked.
  struct Region {
    uint64_t base;
    std::vector<uint8_t> bytes;
  };
  std::vector<Region> vmemInit, dramInit, vmemExpect, dramExpect;
  unsigned expectedReads = 0, expectedWrites = 0;
  static std::vector<uint8_t> hex(const std::string &text) {
    if (text.size() % 2) throw std::runtime_error("odd image hex length");
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < text.size(); i += 2) bytes.push_back(std::stoul(text.substr(i, 2), nullptr, 16));
    return bytes;
  }
  void load(const char *path) {
    // New image views do not clear the retained device or off-chip memory.
    vmemInit.clear();
    dramInit.clear();
    vmemExpect.clear();
    dramExpect.clear();
    expectedReads = expectedWrites = 0;
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read case image");
    std::string kind, data;
    uint64_t base;
    while (input >> kind) {
      if (kind == "dma") {
        input >> expectedReads >> expectedWrites;
        continue;
      }
      if (!(input >> base >> data)) throw std::runtime_error("malformed case image");
      Region region{base, hex(data)};
      bool vmem = kind == "vmem" || kind == "expect_vmem";
      if (vmem && (region.bytes.empty() || region.bytes.size() % 32))
        throw std::runtime_error("VMEM region is not whole lines");
      if (kind == "vmem")
        vmemInit.push_back(region);
      else if (kind == "dram")
        dramInit.push_back(region);
      else if (kind == "expect_vmem")
        vmemExpect.push_back(region);
      else if (kind == "expect_dram")
        dramExpect.push_back(region);
      else
        throw std::runtime_error("unknown case image record");
    }
    if (!input.eof()) throw std::runtime_error("malformed case image");
  }
  static uint32_t hostRead(void *context, uintptr_t address) {
    auto *self = static_cast<Replay *>(context);
    return self->scalar(address >= ATLAS_CSR_BASE, false, uint32_t(address));
  }
  static void hostWrite(void *context, uintptr_t address, uint32_t value) {
    auto *self = static_cast<Replay *>(context);
    self->scalar(address >= ATLAS_CSR_BASE, true, uint32_t(address), value);
  }
  void memoryWrite(unsigned line, const std::array<uint32_t, 8> &data) {
    memory(true, line, data);
  }

  static std::vector<uint8_t> fileBytes(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open block input " + path);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
  }
  struct GraphDriverState {
    Replay *self;
    atlas_graph_device_context *device;
    std::vector<std::string> images, programs;
    unsigned stage = 0;
    unsigned firstReads = 0, firstWrites = 0;
    uint64_t firstCycle = 0;
  };
  static int graphSync(void *context, const atlas_host_buffer *buffer, int before) {
    auto &s = *static_cast<GraphDriverState *>(context);
    auto &r = *s.self;
    if (s.stage >= 54 || buffer->address != ATLAS_DRAM_BASE || buffer->size_bytes > s.device->dram_capacity)
      throw std::runtime_error("fullgraph memory mapping mismatch");
    if (before) {
      r.load(s.images[s.stage].c_str());
      const auto program = fileBytes(s.programs[s.stage]);
      const auto &k = s.device->kernels[s.stage];
      if (program.size() != k.program_words * 4 || std::memcmp(program.data(), s.device->program_buffer, program.size()))
        throw std::runtime_error("actual compiled backend patched program differs from npu program");
      if (r.dramInit.size() != 1 || r.dramExpect.size() != 1 || r.dramInit[0].base != buffer->address ||
          r.dramInit[0].bytes.size() != buffer->size_bytes || r.dramExpect[0].bytes.size() != buffer->size_bytes ||
          (s.stage && !r.vmemInit.empty()))
        throw std::runtime_error("fullgraph complete image mismatch");
      if (std::memcmp(s.device->dram, r.dramInit[0].bytes.data(), buffer->size_bytes))
        throw std::runtime_error("actual full C activation/B/guards differ from model");
      // Reallocate only the host responder view; device/VMEM remain retained.
      // accept() now rejects every byte outside this exact current descriptor.
      r.dram.clear();
      for (size_t i = 0; i < buffer->size_bytes; ++i) r.dram[buffer->address + i] = s.device->dram[i];
      for (const auto &region : r.vmemInit)
        for (size_t line = 0; line < region.bytes.size() / 32; ++line) {
          std::array<uint32_t, 8> data{};
          for (unsigned i = 0; i < 32; ++i) data[i / 4] |= uint32_t(region.bytes[32 * line + i]) << (8 * (i % 4));
          r.memoryWrite(region.base + line, data);
        }
      s.firstCycle = r.cycles;
      s.firstReads = r.reads;
      s.firstWrites = r.writes;
    } else {
      if (r.scalar(true, false, ATLAS_CSR_BASE + ATLAS_CSR_STATUS) != 5 ||
          r.scalar(true, false, ATLAS_CSR_BASE + ATLAS_CSR_DBG0) != s.stage + 1 ||
          r.scalar(true, false, ATLAS_CSR_BASE + ATLAS_CSR_DBG1) != s.stage + 1 ||
          r.scalar(true, false, ATLAS_CSR_BASE + ATLAS_CSR_PC) || !r.pending.empty())
        throw std::runtime_error("fullgraph fresh marker/halt/drain mismatch");
      if (r.reads - s.firstReads != r.expectedReads || r.writes - s.firstWrites != r.expectedWrites)
        throw std::runtime_error("fullgraph DMA count mismatch");
      for (size_t i = 0; i < buffer->size_bytes; ++i) {
        auto actual = r.dram.at(buffer->address + i);
        if (actual != r.dramExpect[0].bytes[i]) throw std::runtime_error("fullgraph DRAM/input/output/guard mismatch");
        s.device->dram[i] = actual;
      }
      for (const auto &region : r.vmemExpect)
        for (size_t line = 0; line < region.bytes.size() / 32; ++line) {
          auto data = r.memory(false, region.base + line);
          for (unsigned i = 0; i < 32; ++i)
            if (((data[i / 4] >> (8 * (i % 4))) & 255) != region.bytes[32 * line + i])
              throw std::runtime_error("fullgraph VMEM guard mismatch");
        }
      for (unsigned i = 0; i < 32; ++i) r.tick();
      if (!r.pending.empty() || r.reads - s.firstReads != r.expectedReads || r.writes - s.firstWrites != r.expectedWrites)
        throw std::runtime_error("fullgraph quiet margin failed");
      std::cout << "C_GRAPH_STAGE_PASSED stage=" << s.stage << " marker=" << s.stage + 1
                << " checked_dram_bytes=" << buffer->size_bytes << " reads=" << r.reads - s.firstReads
                << " writes=" << r.writes - s.firstWrites << " launch=" << s.firstCycle << " completion=" << r.cycles
                << " observed_interval=" << r.cycles - s.firstCycle << '\n'
                << std::flush;
      ++s.stage;
    }
    return 0;
  }
  void runActualGraph(const std::string &planPath) {
    std::ifstream plan(planPath);
    std::string version;
    if (!(plan >> version) || version != "atlas_resnet_graph_v1") throw std::runtime_error("bad fullgraph plan");
    std::vector<std::string> images, programs;
    std::string image, program;
    for (unsigned i = 0; i < 54; ++i) {
      if (!(plan >> image >> program)) throw std::runtime_error("incomplete fullgraph plan");
      images.push_back(image);
      programs.push_back(program);
    }
    if (plan >> image) throw std::runtime_error("extra fullgraph plan record");
    std::vector<uint8_t> cpu(ATLAS_GRAPH_DRAM_BYTES);
    std::vector<uint32_t> patch(8192);
    std::vector<float> logits(1000);
    atlas_host host{};
    atlas_graph_device_context device{};
    device.host = &host;
    device.kernels = graph_kernels;
    device.rescale_words = graph_rescale;
    device.dram = cpu.data();
    device.dram_capacity = cpu.size();
    device.max_polls = 100000000;
    device.program_buffer = patch.data();
    device.program_capacity_words = patch.size();
    GraphDriverState state{this, &device, images, programs};
    if (atlas_host_init(&host, {ATLAS_IMEM_BASE, ATLAS_CSR_BASE, 128 * 1024}, {hostRead, hostWrite, this},
                        {0, graphSync, &state}) != ATLAS_HOST_OK)
      throw std::runtime_error("fullgraph host init failed");
    const int result = atlas_resnet_graph_run(graph_entry, logits.data(), &device, atlas_resnet_graph_device);
    if (result || device.completed != 54 || state.stage != 54 || device.failed || host.active)
      throw std::runtime_error("complete C graph/driver returned failure " + std::to_string(result) + " at " +
                               std::to_string(device.failing_index));
    if (std::memcmp(logits.data(), graph_expected, 1000 * 4))
      throw std::runtime_error("full Cgraph final target logits differ");
    for (unsigned i = 0; i < 54; ++i) {
      if (device.sequence[i] != i + 1) throw std::runtime_error("fullgraph completion ledger mismatch");
      std::cout << "C_GRAPH_LEDGER stage=" << i << " marker=" << device.sequence[i] << " atlas_cycles=" << device.cycles[i]
                << " scale=" << device.activation_scale[i] << " rescale=" << device.rescale_exponent[i] << '\n';
    }
    std::cout << "C_GRAPH_PASSED kernels=54 resets=1 final_words=1000 cycles=" << cycles << '\n' << std::flush;
  }
};

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "resnet_graph_runner graph.plan max-cycles [--dma-trace] [--vcd path]\n";
    return 2;
  }
  try {
    bool dmaTrace = false;
    std::string wave = "/dev/null";
    for (int i = 3; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--dma-trace")
        dmaTrace = true;
      else if (option == "--vcd" && i + 1 < argc)
        wave = argv[++i];
      else
        throw std::runtime_error("unknown or incomplete option");
    }
    Replay replay(wave.c_str(), std::stoull(argv[2]), dmaTrace);
    replay.runActualGraph(argv[1]);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "RESNET_GRAPH_FAILED " << e.what() << '\n';
    return 1;
  }
}
