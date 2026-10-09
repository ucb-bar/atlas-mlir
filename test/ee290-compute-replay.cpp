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
#include <deque>
#include <map>

class Replay {
  VerilatedContext context;
  VAtlasCore dut{&context};
  VerilatedVcdC trace;
  uint64_t time = 0, cycles = 0, limit;
  struct Response { uint64_t due; unsigned source; std::array<uint32_t,8> data; };
  std::deque<Response> pending;
  std::map<uint64_t,uint8_t> dram;
  unsigned reads=0,writes=0,astalls=0,dwaits=0;
  unsigned expected=0, repeats=1;
  static constexpr uint64_t A=0x90000000ULL,B=0x90001000ULL,O=0x90000400ULL;
  uint8_t byte(unsigned i,unsigned which) { return which ? (53*i+79)&255 : (37*i+11)&255; }
  void service() {
    dut.io_dmaTL_a_ready = cycles%11 < 4;
    dut.io_dmaTL_d_valid = !pending.empty() && pending.front().due <= cycles;
    if(dut.io_dmaTL_d_valid) {
      dut.io_dmaTL_d_bits_source=pending.front().source;
      for(unsigned i=0;i<8;++i) dut.io_dmaTL_d_bits_data[i]=pending.front().data[i];
    }
  }
  void accept() {
    if(dut.io_dmaTL_a_valid && !dut.io_dmaTL_a_ready) ++astalls;
    if(!pending.empty() && !dut.io_dmaTL_d_valid) ++dwaits;
    bool response=dut.io_dmaTL_d_valid && dut.io_dmaTL_d_ready;
    if(response) pending.pop_front();
    if(dut.io_dmaTL_a_valid && dut.io_dmaTL_a_ready) {
      auto addr=uint64_t(dut.io_dmaTL_a_bits_address);
      auto op=unsigned(dut.io_dmaTL_a_bits_opcode);
      if((addr&31) || (op!=0 && op!=4)) throw std::runtime_error("unsupported DMA request");
      Response r{cycles+43,unsigned(dut.io_dmaTL_a_bits_source),{}};
      for(unsigned i=0;i<32;++i) {
        if(!dram.count(addr+i)) throw std::runtime_error("DMA escaped initialized memory");
        if(op==4) r.data[i/4]|=uint32_t(dram.at(addr+i))<<(8*(i%4));
        else dram[addr+i]=(dut.io_dmaTL_a_bits_data[i/4]>>(8*(i%4)))&255;
      }
      if(op==4) ++reads; else ++writes;
      pending.push_back(r);
      std::cout<<"DMA_A cycle="<<cycles<<" opcode="<<op<<" address="<<addr<<" source="<<r.source<<'\n';
    }
    if(response) std::cout<<"DMA_D cycle="<<cycles<<'\n';
  }

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
    service();
    evaluate(); // distinct timestamp: inputs settle before the rising edge
    accept();
    dut.clock = 1;
    evaluate();
    dut.clock = 0;
    evaluate();

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

  uint8_t raw(unsigned i) { return (37*i+11+3*(i/32))&255; }
  uint16_t bf(unsigned i,bool rhs) {
    int e=rhs?int(i%3)-1:int(i%5)-2;
    bool negative=rhs?i%11==0:i%7==0;
    return (negative?0x8000:0)|uint16_t((127+e)<<7);
  }
  uint16_t product(unsigned i) {
    int e=int(i%5)-2+int(i%3)-1;
    bool negative=(i%11==0)!=(i%7==0);
    return (negative?0x8000:0)|uint16_t((127+e)<<7);
  }
  void run(const std::vector<uint32_t> &program,const std::string &mode) {
    if(mode!="xlu"&&mode!="vmul"&&mode!="dma_xlu") throw std::runtime_error("unknown mode");
    std::vector<uint8_t> initial(8192),expectedMemory;
    for(unsigned i=0;i<initial.size();++i) initial[i]=(0xa5^(13*i))&255;
    if(mode=="vmul") {
      for(unsigned i=0;i<1024;++i) {
        auto lhs=bf(i,false),rhs=bf(i,true);
        initial[2*i]=lhs&255;initial[2*i+1]=lhs>>8;
        initial[2048+2*i]=rhs&255;initial[2049+2*i]=rhs>>8;
      }
    } else for(unsigned i=0;i<1024;++i) initial[i]=raw(i);
    expectedMemory=initial;
    if(mode=="vmul") {
      for(unsigned i=0;i<1024;++i) { auto p=product(i);expectedMemory[4096+2*i]=p&255;expectedMemory[4097+2*i]=p>>8; }
    } else {
      if(mode=="dma_xlu") for(unsigned i=0;i<128;++i) expectedMemory[i]=byte(i,0);
      for(unsigned r=0;r<32;++r) for(unsigned c=0;c<32;++c) expectedMemory[1024+32*r+c]=expectedMemory[32*c+r];
    }
    for(unsigned i=0;i<128;++i) { dram[A+i]=byte(i,0);dram[B+i]=byte(i,1);dram[A+(1ULL<<32)+i]=0xe7; }
    for(unsigned i=0;i<192;++i) dram[O-32+i]=0xa5;
    scalar(true,true,0x40018,0);
    if(!(scalar(true,false,0x40008)&1)) throw std::runtime_error("initial halt missing");
    for(unsigned i=0;i<program.size();++i) scalar(false,true,0x20000+4*i,program[i]);
    for(unsigned i=0;i<program.size();++i) if(scalar(false,false,0x20000+4*i)!=program[i]) throw std::runtime_error("IMEM mismatch");
    for(unsigned line=0;line<256;++line) {
      std::array<uint32_t,8> d{};
      for(unsigned i=0;i<32;++i) d[i/4]|=uint32_t(initial[line*32+i])<<(8*(i%4));
      memory(true,line,d);
    }
    for(uint32_t offset:{0x10U,0x14U,0U,4U}) scalar(true,true,0x40000+offset,0);
    scalar(true,true,0x40018,1);
    uint32_t status=0,marker=0;unsigned poll=0;
    for(;poll<10000;++poll) { status=scalar(true,false,0x40008);marker=scalar(true,false,0x40010);if((status&1)&&((status>>1)&3)) break; }
    auto illegal=scalar(true,false,0x4000c);
    if(status!=5||marker!=1||illegal||poll==10000||!pending.empty()) throw std::runtime_error("completion mismatch");
    unsigned transactions=mode=="dma_xlu"?4:0;
    if(reads!=transactions||writes!=transactions) throw std::runtime_error("transaction count mismatch");
    for(unsigned i=0;i<128;++i) {
      if(dram[A+i]!=byte(i,0)||dram[B+i]!=byte(i,1)||dram[A+(1ULL<<32)+i]!=0xe7) throw std::runtime_error("source mutation");
      if(dram[O+i]!=(transactions?expectedMemory[1024+i]:0xa5)) throw std::runtime_error("DMA composition output mismatch");
    }
    for(unsigned i=0;i<32;++i) if(dram[O-32+i]!=0xa5||dram[O+128+i]!=0xa5) throw std::runtime_error("DRAM guard mismatch");
    for(unsigned line=0;line<256;++line) {
      auto d=memory(false,line);
      for(unsigned i=0;i<32;++i) if(((d[i/4]>>(8*(i%4)))&255)!=expectedMemory[32*line+i]) {
        std::cerr<<"VMEM byte mismatch mode="<<mode<<" byte="<<32*line+i<<'\n';throw std::runtime_error("compute output/guard mismatch");
      }
    }
    std::cout<<"EE290_COMPUTE_PASSED mode="<<mode<<" checked_vmem_bytes=8192 reads="<<reads<<" writes="<<writes<<" a_stalls="<<astalls<<" delayed_response_cycles="<<dwaits<<" status="<<status<<" marker="<<marker<<" illegal_pc="<<illegal<<" cycles="<<cycles<<'\n';
  }

};

int main(int argc, char **argv) {
  if (argc != 5) { std::cerr << "words.hex trace.vcd max-cycles mode required\n"; return 2; }
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
    replay.run(words,argv[4]);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "EE290_COMPUTE_FAILED " << error.what() << '\n';
    return 1;
  }
}
