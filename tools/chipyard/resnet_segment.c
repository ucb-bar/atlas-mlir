/* One ResNet graph segment on the EE290 SoC: LOADMEM places live-in tensors and the captured DMA arena; this host
 * executes the segment's graph statements with the unchanged runtime, then checks the end state word by word. */
#include "atlas_resnet_graph_device.h"
#include "atlas_host_layout.h"
#include "atlas_segment.h"

__attribute__((section(".tohost"), aligned(64))) volatile uint64_t tohost;
__attribute__((section(".tohost"), aligned(64))) volatile uint64_t fromhost;
extern const atlas_graph_kernel graph_kernels[54];
extern const uint32_t graph_rescale[];
static atlas_host host;
static atlas_graph_device_context context;
__attribute__((aligned(64))) static uint32_t program_buffer[ATLAS_GRAPH_PROGRAM_WORDS];
__attribute__((aligned(64))) float logits[1000];
__attribute__((aligned(64))) static volatile uint64_t magic[8];
static char line[256];
static size_t length;
static uint64_t last_cycle, last_instret;

void *memset(void *p, int value, size_t n) {
  unsigned char *b = p;
  for (size_t i = 0; i < n; ++i) b[i] = (unsigned char)value;
  return p;
}
void *memcpy(void *p, const void *q, size_t n) {
  unsigned char *a = p;
  const unsigned char *b = q;
  for (size_t i = 0; i < n; ++i) a[i] = b[i];
  return p;
}
static inline uint64_t cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
}
static inline uint64_t instret(void) {
  uint64_t value;
  __asm__ volatile("rdinstret %0" : "=r"(value));
  return value;
}

/* HTIF proxied SYS_write(1, s, n), as libgloss does; the QEMU checker intercepts this function. */
__attribute__((noinline)) void atlas_segment_write(const char *s, size_t n) {
  magic[0] = 64;
  magic[1] = 1;
  magic[2] = (uintptr_t)s;
  magic[3] = n;
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  tohost = (uintptr_t)magic;
  while (!fromhost) {
  }
  fromhost = 0;
  __asm__ volatile("fence iorw,iorw" ::: "memory");
}
__attribute__((noinline, noreturn)) void atlas_segment_exit(uint32_t code) {
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  tohost = ((uint64_t)code << 1) | 1;
  for (;;) __asm__ volatile("wfi");
}
static void text(const char *s) {
  while (*s && length < sizeof line - 1) line[length++] = *s++;
}
static void number(const char *key, int64_t value) {
  char digits[24];
  unsigned n = 0;
  uint64_t magnitude = value < 0 ? -(uint64_t)value : (uint64_t)value;
  do digits[n++] = (char)('0' + magnitude % 10); while (magnitude /= 10);
  text(" ");
  text(key);
  text(value < 0 ? "=-" : "=");
  while (n && length < sizeof line - 1) line[length++] = digits[--n];
}
static void flush(void) {
  line[length++] = '\n';
  atlas_segment_write(line, length);
  length = 0;
}

void atlas_segment_mark(unsigned statement) {
  uint64_t c = cycles(), i = instret();
  text("SEG_OP");
  number("segment", atlas_segment.index);
  number("statement", statement);
  number("cpu_cycles", (int64_t)(c - last_cycle));
  number("instret", (int64_t)(i - last_instret));
  flush();
  last_cycle = cycles();
  last_instret = instret();
}
void atlas_block_trap_report(uint64_t cause, uint64_t pc, uint64_t value) {
  text("SEG_TRAP");
  number("mcause", (int64_t)cause);
  number("mepc", (int64_t)pc);
  number("mtval", (int64_t)value);
  flush();
  atlas_segment_exit(0xff);
}

static uint64_t compare(const atlas_segment_region *r, int64_t *first) {
  const volatile uint64_t *actual = r->actual;
  const uint64_t *expected = r->expected;
  uint64_t bad = 0;
  *first = -1;
  for (size_t i = 0; i < r->bytes / 8; ++i)
    if (actual[i] != expected[i]) {
      if (!bad) *first = (int64_t)i;
      ++bad;
    }
  return bad;
}

void atlas_workload_main(void) {
  const unsigned k = atlas_segment.index;
  const uint64_t begin = cycles(), begin_instret = instret();
  text("SEG_BEGIN");
  number("segment", k);
  number("statements", atlas_segment.statements);
  number("regions", atlas_segment.regions);
  flush();
  context = (atlas_graph_device_context){.host = &host,
                                         .kernels = graph_kernels,
                                         .rescale_words = graph_rescale,
                                         .dram = (uint8_t *)(uintptr_t)ATLAS_DRAM_BASE,
                                         .dram_capacity = ATLAS_GRAPH_DRAM_BYTES,
                                         .max_polls = ATLAS_GRAPH_MAX_POLLS,
                                         .program_buffer = program_buffer,
                                         .program_capacity_words = ATLAS_GRAPH_PROGRAM_WORDS,
                                         .completed = k};
  atlas_host_result r = atlas_host_init(&host, (atlas_host_map){ATLAS_IMEM_BASE, ATLAS_CSR_BASE, ATLAS_IMEM_BYTES},
                                        (atlas_host_mmio){atlas_host_direct_read32, atlas_host_direct_write32, 0},
                                        (atlas_host_memory){1, 0, 0});
  /* Runtime state at the segment boundary: k launches completed with sequences 1..k. */
  host.invocation_sequence = k;
  last_cycle = cycles();
  last_instret = instret();
  int result = r == ATLAS_HOST_OK ? atlas_resnet_segment_run(atlas_segment.input, logits, &context,
                                                              atlas_resnet_graph_device) : -2;
  const uint64_t run_cycles = cycles() - begin, run_instret = instret() - begin_instret;
  const uint32_t dbg0 = atlas_host_direct_read32(0, ATLAS_CSR_BASE + ATLAS_CSR_DBG0);
  const uint32_t dbg1 = atlas_host_direct_read32(0, ATLAS_CSR_BASE + ATLAS_CSR_DBG1);
  text("SEG_RUN");
  number("segment", k);
  number("result", result);
  number("completed", context.completed);
  number("failed", context.failed);
  number("device_result", context.result);
  number("runtime_result", context.runtime_result);
  number("sequence", host.invocation_sequence);
  number("scale", context.activation_scale[k]);
  number("exponent", context.rescale_exponent[k]);
  number("dbg0", dbg0);
  number("dbg1", dbg1);
  number("atlas_cycles", context.cycles[k]);
  number("cpu_cycles", (int64_t)run_cycles);
  number("instret", (int64_t)run_instret);
  flush();
  uint64_t mismatches = 0;
  for (unsigned i = 0; i < atlas_segment.regions; ++i) {
    int64_t first;
    const atlas_segment_region *region = &atlas_segment.region[i];
    uint64_t bad = compare(region, &first);
    mismatches += bad;
    text("SEG_CHECK region=");
    text(region->name);
    number("words", (int64_t)(region->bytes / 8));
    number("mismatches", (int64_t)bad);
    number("first", first);
    flush();
  }
  const unsigned want = k + 1;
  const int state_ok = !result && !context.failed && context.completed == want && host.invocation_sequence == want &&
                       context.sequence[k] == want && dbg0 == want && dbg1 == want &&
                       context.activation_scale[k] == atlas_segment.activation_scale &&
                       context.rescale_exponent[k] == atlas_segment.rescale_exponent;
  text(state_ok && !mismatches ? "SEG_PASS" : "SEG_FAIL");
  number("segment", k);
  number("state_ok", state_ok);
  number("mismatches", (int64_t)mismatches);
  number("atlas_cycles", context.cycles[k]);
  number("cpu_cycles", (int64_t)run_cycles);
  number("total_cpu_cycles", (int64_t)(cycles() - begin));
  flush();
  atlas_segment_exit(state_ok && !mismatches ? 0 : 1);
}
