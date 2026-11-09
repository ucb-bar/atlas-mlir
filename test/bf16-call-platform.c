/* Controlled fake MMIO transport. Does not execute or verify device arithmetic. */
#include "platform.h"
#include <string.h>
static struct {
 const atlas_bf16_program *program;
 uint32_t words[32768];
 uint8_t input[2048], response[2048];
 unsigned mode,state,starts,effects,retired;
} device;
static uint32_t read32(void *context, uintptr_t address) {
 (void)context; ++device.effects;
 if(address==0x40004u) return device.retired;
 if(address!=0x40008u) return 0;
 if(device.state==1) { if(device.mode==1) return 0; device.state=2; return 0; }
 if(device.state==2) { device.state=0; ++device.retired; }
 return 5;
}
static void write32(void *context, uintptr_t address, uint32_t word) {
 (void)context; ++device.effects;
 if(address>=0x20000u && address<0x40000u) { device.words[(address-0x20000u)/4]=word; return; }
 if(address==0x40018u && word==1) { ++device.starts; device.state=1; }
}
static int cache(void *context, const atlas_host_buffer *buffer, int before) {
 (void)context; (void)buffer; ++device.effects;
 return !before && device.mode==2;
}
static int write_memory(void *context, uintptr_t address, const uint8_t *data, size_t bytes) {
 (void)context; ++device.effects;
 if(address!=device.program->input_address || bytes!=sizeof(device.input)) return 1;
 memcpy(device.input,data,bytes); return 0;
}
static int read_memory(void *context, uintptr_t address, uint8_t *data, size_t bytes) {
 (void)context; ++device.effects;
 if(address!=device.program->output_address || bytes!=sizeof(device.response)) return 1;
 if(device.mode==3) { memcpy(data,device.response,4); return 1; }
 memcpy(data,device.response,bytes); return 0;
}
int test_platform_initialize(atlas_bf16_call *context, const atlas_bf16_program *program,
                            void *workspace, size_t bytes) {
 device.program=program;
 atlas_host_map map={0x20000u,0x40000u,131072u};
 atlas_host_mmio mmio={read32,write32,0};
 atlas_host_memory memory={0,cache,0};
 atlas_tile_memory_io io={write_memory,read_memory,0};
 return atlas_bf16_call_init(context,program,map,mmio,memory,io,4,workspace,bytes);
}
void test_platform_mode(unsigned mode) { device.mode=mode; }
void test_platform_response(const uint8_t *source, size_t bytes) {
 if(bytes==sizeof(device.response)) memcpy(device.response,source,bytes);
}
unsigned test_platform_starts(void) { return device.starts; }
unsigned test_platform_effects(void) { return device.effects; }
int test_platform_loaded(const uint32_t *words, size_t count) {
 return count<=32768 && !memcmp(words,device.words,count*sizeof(uint32_t));
}
const uint8_t *test_platform_input(void) { return device.input; }
