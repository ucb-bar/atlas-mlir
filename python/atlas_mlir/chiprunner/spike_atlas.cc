// Spike devices for a linked platform ELF in simulation. No tensor arithmetic is implemented here.
//
// atlas_vmem: the Atlas VMEM bus window, backed by a file the step worker shares.
// atlas_mmio: Atlas IMEM and CSR words, forwarded in order to the step worker, whose pinned
//   functional model executes each launch over the shared VMEM file.
// atlas_uart: the platform UART transmitter; the run ends at the harness's COMPLETE or failing
//   STATUS record, which the worker then checks against the dumped outputs.
#include <riscv/abstract_device.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace {

reg_t number(const std::vector<std::string>& args, size_t index) {
  if (index >= args.size()) throw std::runtime_error("missing device argument");
  return std::stoull(args[index], nullptr, 0);
}

class atlas_vmem_t : public abstract_device_t {
 public:
  atlas_vmem_t(const std::string& path, reg_t bytes) : bytes_(bytes) {
    int fd = open(path.c_str(), O_RDWR);
    if (fd < 0) throw std::runtime_error("VMEM file is unavailable");
    void* map = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (map == MAP_FAILED) throw std::runtime_error("VMEM file cannot be mapped");
    data_ = static_cast<uint8_t*>(map);
  }
  bool load(reg_t addr, size_t len, uint8_t* bytes) override {
    if (addr + len > bytes_) return false;
    memcpy(bytes, data_ + addr, len);
    return true;
  }
  bool store(reg_t addr, size_t len, const uint8_t* bytes) override {
    if (addr + len > bytes_) return false;
    memcpy(data_ + addr, bytes, len);
    return true;
  }
  reg_t size() override { return bytes_; }

 private:
  uint8_t* data_;
  reg_t bytes_;
};

void exact(int fd, void* data, size_t bytes, bool writing) {
  auto* cursor = static_cast<uint8_t*>(data);
  while (bytes) {
    ssize_t n = writing ? write(fd, cursor, bytes) : read(fd, cursor, bytes);
    if (n <= 0) _exit(70);  // the worker is gone: nothing may continue silently
    cursor += n;
    bytes -= static_cast<size_t>(n);
  }
}

class atlas_mmio_t : public abstract_device_t {
 public:
  atlas_mmio_t(reg_t base, int requests, int responses) : base_(base), requests_(requests), responses_(responses) {}
  bool load(reg_t addr, size_t len, uint8_t* bytes) override {
    if (len != 4 || addr % 4) return false;
    uint8_t message[13] = {'R'};
    uint64_t address = base_ + addr;
    memcpy(message + 1, &address, 8);
    exact(requests_, message, sizeof(message), true);
    exact(responses_, bytes, 4, false);
    return true;
  }
  bool store(reg_t addr, size_t len, const uint8_t* bytes) override {
    if (len != 4 || addr % 4) return false;
    uint8_t message[13] = {'W'};
    uint64_t address = base_ + addr;
    memcpy(message + 1, &address, 8);
    memcpy(message + 9, bytes, 4);
    exact(requests_, message, sizeof(message), true);
    return true;
  }
  reg_t size() override { return 0x21000; }

 private:
  reg_t base_;
  int requests_, responses_;
};

class atlas_uart_t : public abstract_device_t {
 public:
  atlas_uart_t(const std::string& log, const std::string& tag) : tag_(tag) {
    log_ = fopen(log.c_str(), "wb");
    if (!log_) throw std::runtime_error("UART log is unavailable");
  }
  bool load(reg_t addr, size_t len, uint8_t* bytes) override {
    if (addr + len > sizeof(registers_)) return false;
    uint32_t view[7];
    memcpy(view, registers_, sizeof(view));
    view[0] = 0;            // TXDATA: never full
    view[1] = 1u << 31;     // RXDATA: always empty
    view[5] = 1;            // IP: transmit watermark pending
    memcpy(bytes, reinterpret_cast<uint8_t*>(view) + addr, len);
    return true;
  }
  bool store(reg_t addr, size_t len, const uint8_t* bytes) override {
    if (addr + len > sizeof(registers_)) return false;
    memcpy(reinterpret_cast<uint8_t*>(registers_) + addr, bytes, len);
    if (addr == 0) transmit(static_cast<char>(bytes[0]));
    return true;
  }
  reg_t size() override { return 0x1000; }

 private:
  void transmit(char c) {
    fputc(c, log_);
    if (c != '\n') {
      line_ += c;
      return;
    }
    fflush(log_);
    std::string line = line_;
    line_.clear();
    if (line == tag_ + " COMPLETE") finish(0);
    std::string status = tag_ + " STATUS ";
    if (line.compare(0, status.size(), status) == 0 && line.compare(status.size(), 2, "0 ") != 0) finish(3);
  }
  void finish(int code) {
    fclose(log_);
    _exit(code);
  }
  FILE* log_;
  std::string tag_, line_;
  uint32_t registers_[7] = {};
};

atlas_vmem_t* vmem_parse(const void*, const sim_t*, reg_t* base, const std::vector<std::string>& args) {
  *base = number(args, 0);
  return new atlas_vmem_t(args.at(2), number(args, 1));
}
atlas_mmio_t* mmio_parse(const void*, const sim_t*, reg_t* base, const std::vector<std::string>& args) {
  *base = number(args, 0);
  return new atlas_mmio_t(*base, static_cast<int>(number(args, 1)), static_cast<int>(number(args, 2)));
}
atlas_uart_t* uart_parse(const void*, const sim_t*, reg_t* base, const std::vector<std::string>& args) {
  *base = number(args, 0);
  return new atlas_uart_t(args.at(1), args.at(2));
}
std::string no_dts(const sim_t*, const std::vector<std::string>&) { return ""; }

}  // namespace

REGISTER_DEVICE(atlas_vmem, vmem_parse, no_dts)
REGISTER_DEVICE(atlas_mmio, mmio_parse, no_dts)
REGISTER_DEVICE(atlas_uart, uart_parse, no_dts)
