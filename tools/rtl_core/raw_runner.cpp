// Current emitted AtlasCore only. Functional TL responder, not a memory timing model.
#include "VAtlasCore.h"
#include "verilated.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
static void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
static fs::path admitted(const std::string& value) {
    fs::path path = fs::absolute(value);
    for (const auto& component : path) {
        std::string name = component.string();
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c){return std::tolower(c);});
        require(name != ".." && name.find("hammer") == std::string::npos &&
                name.find("vlsi") == std::string::npos, "Unadmitted lexical component");
    }
    fs::path current;
    for (const auto& component : path) {
        current /= component;
        require(!fs::is_symlink(fs::symlink_status(current)), "Symlink component refused");
    }
    return path;
}
static std::vector<uint8_t> read_file(const std::string& value) {
    std::ifstream input(admitted(value), std::ios::binary);
    require(input.good(), "Cannot open input");
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
}
static uint64_t number(const char* value) {
    size_t end = 0;
    uint64_t result = std::stoull(value, &end, 0);
    require(value[end] == '\0', "Malformed integer");
    return result;
}
struct Response { uint64_t due, address; uint32_t source, opcode; std::vector<uint8_t> data; };

class Runner {
public:
    VAtlasCore top;
    std::vector<uint8_t> memory, initial;
    std::deque<Response> responses;
    std::ofstream trace;
    uint64_t cycle = 0, base, input_bytes, output_offset, output_bytes;
    uint64_t load_bytes = 0, store_bytes = 0, responses_served = 0;
    bool active = false, seen_running = false;

    Runner(std::vector<uint8_t> image, const std::string& log, uint64_t b, uint64_t inputs,
           uint64_t offset, uint64_t outputs): memory(image), initial(image), trace(admitted(log)),
           base(b), input_bytes(inputs), output_offset(offset), output_bytes(outputs) {
        require(trace.good(), "Cannot open trace");
        require(base == (uint64_t(1) << 32), "Require explicit non-aliasing 4GiB DRAM base");
        require(output_offset >= input_bytes && output_offset + output_bytes <= memory.size(), "Invalid memory extent");
        top.clock = 0; top.reset = 1;
        top.io_imemTL_a_valid = 0; top.io_imemTL_d_ready = 1;
        top.io_csrTL_a_valid = 0; top.io_csrTL_d_ready = 1;
        top.io_vmemTL_a_valid = 0; top.io_vmemTL_d_ready = 1;
        top.io_vmemTL_a_bits_opcode = 0; top.io_vmemTL_a_bits_param = 0;
        top.io_vmemTL_a_bits_size = 5; top.io_vmemTL_a_bits_source = 0;
        top.io_vmemTL_a_bits_address = 0; top.io_vmemTL_a_bits_mask = 0;
        top.io_vmemTL_a_bits_corrupt = 0;
        for (int i=0;i<8;++i) top.io_vmemTL_a_bits_data[i] = 0;
        top.io_imemTL_a_bits_opcode = 0; top.io_imemTL_a_bits_param = 0;
        top.io_imemTL_a_bits_size = 2; top.io_imemTL_a_bits_source = 0;
        top.io_imemTL_a_bits_address = 0; top.io_imemTL_a_bits_mask = 15;
        top.io_imemTL_a_bits_data = 0; top.io_imemTL_a_bits_corrupt = 0;
        top.io_csrTL_a_bits_opcode = 0; top.io_csrTL_a_bits_param = 0;
        top.io_csrTL_a_bits_size = 2; top.io_csrTL_a_bits_source = 0;
        top.io_csrTL_a_bits_address = 0; top.io_csrTL_a_bits_mask = 15;
        top.io_csrTL_a_bits_data = 0; top.io_csrTL_a_bits_corrupt = 0;
        top.io_dmaTL_a_ready = 1; top.io_dmaTL_d_valid = 0;
        top.io_dmaTL_d_bits_opcode = 0; top.io_dmaTL_d_bits_param = 0;
        top.io_dmaTL_d_bits_size = 5; top.io_dmaTL_d_bits_source = 0;
        top.io_dmaTL_d_bits_sink = 0; top.io_dmaTL_d_bits_denied = 0;
        top.io_dmaTL_d_bits_corrupt = 0;
        for (int i=0;i<8;++i) top.io_dmaTL_d_bits_data[i] = 0;
    }

    void tick() {
        top.clock = 0;
        const bool pending = !responses.empty() && responses.front().due <= cycle;
        top.io_dmaTL_d_valid = pending;
        if (pending) {
            const auto& response = responses.front();
            top.io_dmaTL_d_bits_opcode = response.opcode;
            top.io_dmaTL_d_bits_source = response.source;
            for (int i=0;i<8;++i) {
                uint32_t word=0;
                for (int j=0;j<4;++j) word |= uint32_t(response.data[4*i+j]) << (8*j);
                top.io_dmaTL_d_bits_data[i] = word;
            }
        }
        top.eval(); // Settle external drives before sampling any fire.
        const bool afire = !top.reset && top.io_dmaTL_a_valid && top.io_dmaTL_a_ready;
        const bool dfire = !top.reset && top.io_dmaTL_d_valid && top.io_dmaTL_d_ready;
        Response next{};
        if (afire) {
            uint64_t address = top.io_dmaTL_a_bits_address;
            uint32_t opcode = top.io_dmaTL_a_bits_opcode;
            require(top.io_dmaTL_a_bits_size == 5 && address % 32 == 0 &&
                    top.io_dmaTL_a_bits_mask == 0xffffffffu && !top.io_dmaTL_a_bits_corrupt &&
                    top.io_dmaTL_a_bits_param == 0, "Unexpected DMA request fields");
            require(address >= base && address-base <= memory.size() && memory.size()-(address-base) >= 32,
                    "Unmapped DMA address (no modulo aliasing)");
            const uint64_t offset = address-base;
            require(opcode == 0 || opcode == 4, "Unsupported DMA opcode");
            std::vector<uint8_t> data(32);
            if (opcode == 4) {
                require(offset + 32 <= input_bytes, "DMA read outside declared inputs");
                std::copy_n(memory.begin()+offset,32,data.begin()); load_bytes += 32;
            } else {
                require(offset >= output_offset && offset-output_offset <= output_bytes &&
                        output_bytes-(offset-output_offset) >= 32, "DMA write outside declared outputs");
                for (int i=0;i<32;++i) data[i] = uint8_t(top.io_dmaTL_a_bits_data[i/4] >> (8*(i%4)));
                std::copy(data.begin(),data.end(),memory.begin()+offset); store_bytes += 32;
            }
            next = {cycle+1,address,uint32_t(top.io_dmaTL_a_bits_source),opcode==4?1u:0u,data};
            for (const auto& old:responses) require(old.source != next.source ||
                (dfire && &old == &responses.front()), "DMA source reused while outstanding");
            trace << "{\"cycle\":"<<cycle<<",\"event\":\"dma_a\",\"address\":"<<address
                  <<",\"source\":"<<next.source<<",\"opcode\":"<<opcode<<",\"bytes\":[";
            for (int i=0;i<32;++i) trace<<(i?",":"")<<unsigned(data[i]);
            trace << "]}\n";
        }
        if (dfire) {
            trace << "{\"cycle\":"<<cycle<<",\"event\":\"dma_d\",\"source\":"
                  << responses.front().source << ",\"opcode\":" << responses.front().opcode <<"}\n";
            responses.pop_front(); ++responses_served;
        }
        if (afire) responses.push_back(next);
        if (active && !top.io_halted) seen_running = true;
        trace << "{\"cycle\":"<<cycle<<",\"event\":\"state\",\"reset\":"<<unsigned(top.reset)
              <<",\"halted\":"<<unsigned(top.io_halted)<<",\"public_idle\":"<<observed_idle()
              <<",\"outstanding\":"<<responses.size()<<"}\n";
        if (cycle%256==0) trace.flush();
        top.clock = 1; top.eval(); ++cycle;
    }

    void put(bool csr, uint32_t address, uint32_t data) {
        if (csr) {top.io_csrTL_a_bits_address=address;top.io_csrTL_a_bits_data=data;top.io_csrTL_a_valid=1;}
        else {top.io_imemTL_a_bits_address=address;top.io_imemTL_a_bits_data=data;top.io_imemTL_a_valid=1;}
        bool sent=false,ack=false;
        for(int i=0;i<100 && !ack;++i) {
            top.clock=0;top.eval();
            const bool fire=!sent && (csr?top.io_csrTL_a_ready:top.io_imemTL_a_ready);
            const bool response=csr?top.io_csrTL_d_valid:top.io_imemTL_d_valid;
            if(fire) trace<<"{\"cycle\":"<<cycle<<",\"event\":\"host_a\",\"csr\":"<<csr
                         <<",\"address\":"<<address<<",\"data\":"<<data<<"}\n";
            if(response) {
                require(sent || fire,"Unsolicited host response");
                require((csr?top.io_csrTL_d_bits_opcode:top.io_imemTL_d_bits_opcode)==0 &&
                        !(csr?top.io_csrTL_d_bits_denied:top.io_imemTL_d_bits_denied) &&
                        !(csr?top.io_csrTL_d_bits_corrupt:top.io_imemTL_d_bits_corrupt),"Bad host response");
                require((csr?top.io_csrTL_d_bits_source:top.io_imemTL_d_bits_source)==0 &&
                        (csr?top.io_csrTL_d_bits_size:top.io_imemTL_d_bits_size)==2 &&
                        (csr?top.io_csrTL_d_bits_param:top.io_imemTL_d_bits_param)==0,"Bad host echo fields");
                trace<<"{\"cycle\":"<<cycle<<",\"event\":\"host_d\",\"csr\":"<<csr<<"}\n";
                ack=true;
            }
            tick();
            if(fire) {sent=true;if(csr)top.io_csrTL_a_valid=0;else top.io_imemTL_a_valid=0;}
        }
        require(sent && ack,"Host TileLink timeout");
    }

    // AtlasCore's public debug omits VPU busy; this observes only exported engines.
    bool observed_idle() const {
        return responses.empty() && !top.io_dbg_dmaBusy_0 && !top.io_dbg_dmaBusy_1 &&
               !top.io_dbg_dmaBusy_2 && !top.io_dbg_dmaBusy_3 && !top.io_dbg_dmaBusy_4 &&
               !top.io_dbg_dmaBusy_5 && !top.io_dbg_dmaBusy_6 && !top.io_dbg_dmaBusy_7 &&
               !top.io_dbg_mxu0DataBusy && !top.io_dbg_mxu0CompBusy &&
               !top.io_dbg_mxu1DataBusy && !top.io_dbg_mxu1CompBusy &&
               !top.io_dbg_xluBusy && !top.io_dbg_lsuBusy;
    }
};

static uint32_t read_status(Runner& run) {
    auto& top=run.top;
    top.io_csrTL_a_bits_opcode=4;top.io_csrTL_a_bits_address=8;
    top.io_csrTL_a_bits_data=0;top.io_csrTL_a_valid=1;
    bool sent=false,ack=false;uint32_t status=0;
    for(int i=0;i<100 && !ack;++i) {
        top.clock=0;top.eval();
        const bool fire=!sent && top.io_csrTL_a_ready;
        if(fire) run.trace<<"{\"cycle\":"<<run.cycle<<",\"event\":\"status_a\",\"address\":8}\n";
        if(top.io_csrTL_d_valid) {
            require(sent||fire,"Unsolicited status response");
            require(top.io_csrTL_d_bits_opcode==1 && top.io_csrTL_d_bits_source==0 &&
                    top.io_csrTL_d_bits_size==2 && top.io_csrTL_d_bits_param==0 &&
                    !top.io_csrTL_d_bits_denied && !top.io_csrTL_d_bits_corrupt,"Bad status response");
            status=top.io_csrTL_d_bits_data;ack=true;
            run.trace<<"{\"cycle\":"<<run.cycle<<",\"event\":\"status_d\",\"data\":"<<status<<"}\n";
        }
        run.tick();
        if(fire){sent=true;top.io_csrTL_a_valid=0;}
    }
    require(sent&&ack,"Status Get timeout");top.io_csrTL_a_bits_opcode=0;
    return status;
}

int main(int argc,char**argv) {
    try {
        require(argc==12,"Usage: raw_runner program initial actual trace seed maxcycles base inputbytes offset outputbytes summary");
        Verilated::randReset(2); Verilated::randSeed(number(argv[5]));
        auto program=read_file(argv[1]);
        require(!program.empty() && program.size()%4==0 && program.size()<=16384,"Bad IMEM image");
        Runner run(read_file(argv[2]),argv[4],number(argv[7]),number(argv[8]),number(argv[9]),number(argv[10]));
        for(int i=0;i<4;++i) run.tick();
        run.top.reset=0;run.tick();
        for(size_t i=0;i<program.size();i+=4) {
            uint32_t word=0;for(int j=0;j<4;++j)word|=uint32_t(program[i+j])<<(8*j);
            run.put(false,0x20000+i,word);
        }
        const uint64_t started=run.cycle;
        run.active=true;run.put(true,0x18,1);
        const uint64_t limit=number(argv[6]);
        while(run.cycle-started<limit && !(run.seen_running && run.top.io_halted && run.observed_idle()))run.tick();
        require(run.seen_running && run.top.io_halted && run.observed_idle(),"Program failed to halt with public engines idle");
        const uint64_t halted=run.cycle;
        run.tick(); // Commit the combinational halt event into CSR halt_reason.
        const uint32_t status=read_status(run);
        require(status==5,"CSR status is not halted with ECALL reason2");
        const uint64_t writes=run.store_bytes,reads=run.load_bytes;
        for(int i=0;i<32;++i){run.tick();require(run.top.io_halted && run.observed_idle() &&
            run.store_bytes==writes && run.load_bytes==reads,"Activity after completion");}
        require(std::equal(run.memory.begin(),run.memory.begin()+run.output_offset,run.initial.begin()) &&
                std::equal(run.memory.begin()+run.output_offset+run.output_bytes,run.memory.end(),
                           run.initial.begin()+run.output_offset+run.output_bytes),"Input/guard corruption");
        std::ofstream actual(admitted(argv[3]),std::ios::binary);
        actual.write(reinterpret_cast<const char*>(run.memory.data()+run.output_offset),run.output_bytes);
        require(actual.good(),"Output write failed");
        std::ofstream summary(admitted(argv[11]));
        summary<<"{\"halted\":true,\"cycles_from_start_csr\":"<<halted-started
               <<",\"quiet_cycles\":32,\"load_bytes\":"<<reads<<",\"store_bytes\":"<<writes
               <<",\"responses\":"<<run.responses_served
               <<",\"csr_status\":"<<status<<",\"halt_reason\":\"ecall\""
               <<",\"guards_preserved\":true,\"public_busy_idle\":true,\"vpu_busy_observed\":false}\n";
        require(summary.good(),"Summary write failed");
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
