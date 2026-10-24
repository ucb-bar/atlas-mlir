#!/usr/bin/env python3
"""Native real host graph/runtime adapter against exact saved device replies."""
from atlas_tooling import write_json, run_command

import argparse
import json
from pathlib import Path
import subprocess
import time
from atlas_tooling import Admission, sha
from atlas_path_guard import checked

FIXTURE = r"""
#include "assets.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
@EXTERNS@
static const unsigned char *const encoded[] = {@ENCODED@};
static const unsigned char *const replies[] = {@REPLIES@};
__attribute__((aligned(64))) static unsigned char dram[ATLAS_GRAPH_DRAM_BYTES];
__attribute__((aligned(64))) static uint32_t patch[8192], table[254];
static atlas_graph_kernel kernels[54];
static atlas_graph_device_context context;
static uint32_t imem[32768], status = 1, done, command, ticks;
static unsigned launches, mode;
static float logits[1000];
int atlas_resnet_graph_run(const float *, float *, void *,
                           int (*)(void *, unsigned, const uint8_t *, size_t, int, uint8_t *, size_t));

static uint32_t read32(void *opaque, uintptr_t address) {
  (void)opaque;
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_CYCLES)
    return ++ticks;
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_STATUS)
    return status;
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_DBG0)
    return done;
  assert(0);
  return 0;
}

static void write32(void *opaque, uintptr_t address, uint32_t value) {
  (void)opaque;
  if (address >= ATLAS_IMEM_BASE && address < ATLAS_IMEM_BASE + ATLAS_IMEM_BYTES) {
    assert(!(address & 3));
    imem[(address - ATLAS_IMEM_BASE) / 4] = value;
    return;
  }
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_DBG0) {
    done = value;
    return;
  }
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_DBG1) {
    command = value;
    return;
  }
  assert(address == ATLAS_CSR_BASE + ATLAS_CSR_EXEC_CONTROL && value == 1 && launches < 54);
  const atlas_graph_kernel *k = &graph_kernels[launches];
  assert(memcmp(dram, encoded[launches], k->activation_bytes) == 0);
  assert(memcmp(dram + k->activation_bytes, k->weights, k->weight_bytes) == 0);
  for (size_t i = 0; i < k->program_words - 1; ++i) {
    uint32_t expected = i == k->patch_word ? table[@EXPONENTS@[launches] + 126] : k->program[i];
    assert(imem[i] == expected);
  }
  static const uint32_t tail[] = {0x13, 0xc1102ff3, 0x13, 0xc10f9073, 0x13, 0x73};
  assert(memcmp(imem + k->program_words - 1, tail, sizeof tail) == 0);
  assert(command == launches + 1 && done == 0);
  ++launches;
  if (mode == 7) {
    status = 0;
    return;
  }
  memcpy(dram + k->output_offset, replies[launches - 1], k->output_bytes);
  if (mode == 5)
    dram[k->output_offset - 1] ^= 1;
  if (mode == 6)
    dram[0] ^= 1;
  done = command;
  status = 5; /* device completes before first host poll */
}

static int invoke(void *w, unsigned i, const uint8_t *a, size_t n, int e, uint8_t *out, size_t b) {
  if (mode == 1)
    ++i;
  if (mode == 3)
    e = 127;
  if (mode == 9)
    out = dram;
  return atlas_resnet_graph_device(w, i, a, n, e, out, b);
}

int main(int argc, char **argv) {
  assert(argc == 3);
  mode = (unsigned)atoi(argv[1]);
  atlas_host host;
  assert(atlas_host_init(&host, (atlas_host_map){ATLAS_IMEM_BASE, ATLAS_CSR_BASE, ATLAS_IMEM_BYTES},
                         (atlas_host_mmio){read32, write32, 0}, (atlas_host_memory){1, 0, 0}) == ATLAS_HOST_OK);
  memcpy(kernels, graph_kernels, sizeof kernels);
  memcpy(table, graph_rescale, sizeof table);
  context = (atlas_graph_device_context){
      .host = &host, .kernels = kernels, .rescale_words = table,
      .dram = dram, .dram_capacity = sizeof dram, .max_polls = 3,
      .program_buffer = patch, .program_capacity_words = 8192};
  if (mode == 2)
    context.dram_capacity = 1;
  if (mode == 3)
    kernels[0].weight_scale = 127;
  if (mode == 4)
    table[126] ^= 1;
  uint32_t *changed = 0;
  if (mode == 8 || mode == 10) {
    changed = malloc(kernels[0].program_words * 4);
    assert(changed);
    memcpy(changed, kernels[0].program, kernels[0].program_words * 4);
    changed[mode == 8 ? kernels[0].patch_word : kernels[0].program_words - 1] ^= 1;
    kernels[0].program = changed;
  }
  if (mode == 11)
    context.program_buffer = (uint32_t *)dram;
  if (mode == 12)
    ++kernels[0].weight_bytes;
  int result = atlas_resnet_graph_run(graph_entry, logits, &context, invoke);
  if (!mode) {
    assert(!result && launches == 54 && context.completed == 54 && !host.active);
    assert(memcmp(logits, graph_expected, sizeof logits) == 0);
    for (unsigned i = 0; i < 54; ++i) {
      assert(context.sequence[i] == i + 1);
      assert(context.rescale_exponent[i] == @EXPONENTS@[i]);
    }
    FILE *out = fopen(argv[2], "wb");
    assert(out);
    assert(fwrite(logits, 1, sizeof logits, out) == sizeof logits);
    assert(!fclose(out));
  } else {
    assert(result && context.failed && context.completed == 0);
    unsigned expected = mode == 3 ? ATLAS_GRAPH_DEVICE_SCALE : mode == 5 ? ATLAS_GRAPH_DEVICE_GUARD :
        mode == 6 ? ATLAS_GRAPH_DEVICE_INPUT :
        (mode == 7 || mode == 10) ? ATLAS_GRAPH_DEVICE_RUNTIME : ATLAS_GRAPH_DEVICE_INVALID;
    assert(context.result == expected);
    assert(launches == ((mode >= 5 && mode <= 7) ? 1u : 0u));
    if (mode == 7) {
      assert(host.active && host.active_buffers == &context.active_buffer && context.runtime_result == ATLAS_HOST_TIMEOUT);
    } else {
      assert(!host.active);
    }
    assert(atlas_resnet_graph_device(&context, 0, encoded[0], kernels[0].activation_bytes,
                                     0, (uint8_t *)logits, kernels[0].output_bytes) == (int)expected);
  }
  free(changed);
  printf("{\"mode\":%u,\"launches\":%u,\"result\":%u,\"status\":\"PASS\"}\n",
         mode, launches, (unsigned)context.result);
  return 0;
}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("build", "reference", "out", "cc"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    for value in vars(args).values():
        checked(value)
    if args.out.exists():
        raise ValueError("fresh gate output required")
    if any(ch in str(args.out) for ch in ('"', "\\", "\n", "\r")):
        raise ValueError("unsafe assembly output path")
    ad = Admission()
    ad.read(Path(__file__).absolute())
    ad.read(args.cc)
    build = json.loads(ad.read(args.build / "build.json"))
    if build["returncode"] or not build["source_bookends_equal"]:
        raise ValueError("invalid bound target build")
    ad.read(args.build / "host.elf", build["elf_sha256"])
    for path, digest in build["inputs_sha256"].items():
        ad.read(path, digest)
    for path, digest in build["generated_inputs_sha256"].items():
        ad.read(path, digest)
    reference = json.loads(ad.read(args.reference / "receipt.json"))
    if len(reference["kernels"]) != 54:
        raise ValueError("54 reference replies required")
    args.out.mkdir(parents=True)
    asm = ['.section .rodata,"a",@progbits']
    externs = []
    encoded = []
    replies = []
    exponents = []
    for i, k in enumerate(reference["kernels"]):
        for label, key in [("encoded", "encoded_a_tiled"), ("reply", "expected_physical_output")]:
            item = k[key]
            raw = ad.read(item["path"], item["sha256"])
            dest = args.out / (label + "_" + str(i) + ".bin")
            dest.write_bytes(raw)
            asm += [".balign 64", f".global {label}_{i}", f"{label}_{i}:", f'.incbin "{dest}"',
                f".size {label}_{i},.-{label}_{i}"]
            externs.append(f"extern const unsigned char {label}_{i}[];")
        encoded.append("encoded_" + str(i))
        replies.append("reply_" + str(i))
        exponents.append(str(k["rescale_exponent"]))
    asm += ['.section .note.GNU-stack,"",@progbits']
    (args.out / "replies.S").write_text("\n".join(asm) + "\n")
    body = (FIXTURE.replace("@EXTERNS@",
        "\n".join(externs) + "\nstatic const int expected_exponents[]={" + ",".join(exponents) + "};") .replace("@ENCODED@",
        ",".join(encoded)) .replace("@REPLIES@", ",".join(replies)) .replace("@EXPONENTS@", "expected_exponents"))
    (args.out / "fixture.c").write_text(body)
    command = ([str(args.cc), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off", "-fno-fast-math",
        "-I" + str(args.build), "-I" + str(args.build / "runtime"), str(args.out / "fixture.c"),
        str(args.out / "replies.S")] + [str(args.build / n) for n in ("assets.c", "images.S", "graph/graph.c",
        "graph/constants.S", "runtime/atlas_host.c", "runtime/atlas_resnet_graph_device.c",
        "runtime/atlas_resnet_graph_math.c")] + ["-o", str(args.out / "native-test")])
    generated = {str(path): sha(path.read_bytes()) for path in [args.out / "fixture.c",
        args.out / "replies.S"] + [args.out / (label + "_" + str(i) + ".bin") for i in range(54) for label in ("encoded",
        "reply")]}
    result = run_command(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (args.out / "build.log").write_text(result.stdout)
    if result.returncode:
        raise ValueError("native compile failed: " + result.stdout)
    cases = []
    for mode in range(13):
        command_case = [str(args.out / "native-test"), str(mode), str(args.out / "actual-final.f32.bin")]
        before = time.monotonic()
        r = run_command(command_case, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (args.out / f"case-{mode}.log").write_text(r.stdout)
        cases.append(dict(command=command_case, returncode=r.returncode, wall_seconds=time.monotonic() - before,
            stdout=r.stdout))
        if r.returncode:
            break
    final = (sha((args.out / "actual-final.f32.bin").read_bytes()) if (args.out / "actual-final.f32.bin").exists() else None)
    bookends = ad.recheck() and all(sha(checked(Path(path)).read_bytes()) == digest for path, digest in generated.items())
    passed = (len(cases) == 13 and all(not c["returncode"] for c in cases) and bookends
        and final == reference["final_logits"]["sha256"])
    report = dict(
        status="PASS" if passed else "FAIL",
        scope=("Actual native C graph+device adapter+raw runtime; exact saved reply mock, "
               "no instruction-model/RTL/RV64 execution"),
        command=command, cases=cases, all54_live_A_weights_patched_programs_exact=passed,
        all54_sequences_exact=passed, final_1000_words_exact=passed, final_sha256=final,
        inputs_sha256=ad.inputs, generated_inputs_sha256=generated, source_bookends_equal=bookends,
        executable_sha256=sha((args.out / "native-test").read_bytes()))
    write_json(args.out / "gate.json", report)
    print(json.dumps({key: report[key] for key in ("status", "final_sha256", "source_bookends_equal")}))
    raise SystemExit(not passed)


if __name__ == "__main__":
    main()
