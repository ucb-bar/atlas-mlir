"""The IMEM residency cost rule replays the host runtime word for word."""

import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from atlas_mlir_native.imem_residency import (call_programs, choose, launch_order, one_image_words, relocatable,
                                              resident_words)

ROOT = Path(__file__).resolve().parents[1]
ECALL, NOP = 0x73, 0x13

# A fake device: records IMEM writes and, at each start, the image reached from word 0.
HARNESS = r'''
#include "atlas_host.h"
#include <stdio.h>
static uint32_t imem[%(capacity)d], copy[%(capacity)d];
static unsigned long writes;
static atlas_host_image_slot slots[%(slots)d];
static uint32_t read32(void *c, uintptr_t a) { (void)c; (void)a; return 5; }
static void write32(void *c, uintptr_t a, uint32_t w) {
  (void)c;
  if (a == ATLAS_CSR_BASE + ATLAS_CSR_EXEC_CONTROL) return;
  imem[(a - ATLAS_IMEM_BASE) / 4] = w; ++writes;
}
%(images)s
static const uint32_t *order[] = {%(order)s};
static const size_t counts[] = {%(counts)s};
int main(void) {
  atlas_host h;
  if (atlas_host_init(&h, (atlas_host_map){ATLAS_IMEM_BASE, ATLAS_CSR_BASE, %(capacity)du * 4u},
                      (atlas_host_mmio){read32, write32, 0}, (atlas_host_memory){1, 0, 0})) return 1;
  if (%(resident)d && atlas_host_keep_images(&h, copy, %(capacity)du, slots, %(slots)du)) return 2;
  for (size_t i = 0; i < sizeof order / sizeof order[0]; ++i) {
    if (atlas_host_ensure_loaded(&h, order[i], counts[i])) return 3;
    size_t entry = 0;
    if (%(resident)d) {
      if (imem[1] != 0x13u || (imem[0] & 0xfffu) != 0x6fu || imem[0] != atlas_host_entry_jump(h.image_selected)) return 4;
      entry = h.image_selected;
    }
    for (size_t j = 0; j < counts[i]; ++j) if (imem[entry + j] != order[i][j]) return 5;
    printf("%%zu ", entry);
  }
  printf("\n%%lu %%llu\n", writes, (unsigned long long)h.imem_words_written);
  return 0;
}
'''


def images_and_order(seed):
    """Three trips over five templates whose trips differ in one or two address words, plus a shared image."""
    rng = random.Random(seed)
    templates = [[rng.randrange(1 << 12) << 20 | NOP for _ in range(length - 1)] + [ECALL]
                 for length in (5, 9, 5, 12, 7)]
    words, order = {}, []
    for trip in range(3):
        for index, template in enumerate(templates):
            image = list(template)
            for site in range(1 + index % 2):
                image[site] = (trip * 64 + index) << 20 | NOP
            words[f"t{trip}i{index}"] = image
            order += [f"t{trip}i{index}", "shared"]
    words["shared"] = [0x02a00093, ECALL]
    return words, order


class ImemResidencyTest(unittest.TestCase):
    def run_runtime(self, words, order, capacity, resident):
        compiler = shutil.which("cc")
        self.assertIsNotNone(compiler, "a C compiler is required")
        names = {key: f"image_{i}" for i, key in enumerate(words)}
        source = HARNESS % dict(
            capacity=capacity, slots=max(1, resident_words(order, words, capacity)[1]), resident=int(resident),
            images="\n".join(f"static const uint32_t {names[key]}[] = {{{','.join(f'{w}u' for w in image)}}};"
                             for key, image in words.items()),
            order=",".join(names[key] for key in order), counts=",".join(str(len(words[key])) for key in order))
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "main.c").write_text(source)
            subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "runtime"),
                            "main.c", str(ROOT / "runtime/atlas_host.c"), "-o", "check"],
                           cwd=directory, check=True, capture_output=True)
            result = subprocess.run([str(Path(directory, "check"))], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, "an image reached from word 0 differs from the request")
        entries, counters = result.stdout.splitlines()
        mmio, counted = map(int, counters.split())
        self.assertEqual(mmio, counted)
        return mmio, [int(entry) for entry in entries.split()]

    def test_rule_predicts_the_runtime_word_for_word(self):
        words, order = images_and_order(1)
        # 48 words hold one trip, so trips replace each other and a full IMEM starts again at word 2.
        for capacity in (48, 32):
            with self.subTest(capacity=capacity):
                predicted, _, _, entries = resident_words(order, words, capacity)
                self.assertEqual(self.run_runtime(words, order, capacity, True), (predicted, entries))
                self.assertEqual(self.run_runtime(words, order, capacity, False),
                                 (one_image_words(order, words, ()), [0] * len(order)))
        record = choose(order, words, (), dict.fromkeys(words, True), 48)
        self.assertLess(record["imem_words"]["resident"], record["imem_words"]["one_image"])
        self.assertEqual(record["imem_words"]["resident"], resident_words(order, words, 48)[0])
        # The published schedule names each start's image and the entry word the runtime selects.
        self.assertEqual([record["images"][image] for image, _ in record["starts"]], order)
        self.assertEqual([entry for _, entry in record["starts"]], resident_words(order, words, 48)[3])

    def test_rule_keeps_one_image_unless_it_writes_fewer_words(self):
        words, order = images_and_order(2)
        self.assertIsNone(choose(order[:2], words, (), dict.fromkeys(words, True), 48))
        self.assertIsNone(choose(order, words, (), {**dict.fromkeys(words, True), "shared": False}, 48))
        self.assertIsNone(choose(order, words, (), dict.fromkeys(words, True), 13))

    def test_launch_order_and_relocation_facts(self):
        calls = [dict(launches_per_invocation=0), dict(launches_per_invocation=2),
                 dict(launches_per_invocation=3, device_launch_schedule=dict(
                     images=["b", "c"], groups=[dict(repeat=1, runs=[dict(image=0, count=1), dict(image=1, count=2)])]))]
        self.assertEqual(launch_order(calls, [[], ["a"], ["b", "c"]]), ["a", "a", "b", "c", "c"])
        # Each launching call's program path per launch; a launch whose program is not determined disables the form.
        calls = [dict(context_index=i, **call) for i, call in enumerate(calls)]
        programs = [[], [("p/a", "a")], [("p/b", "b"), ("p/c", "c")]]
        self.assertEqual(call_programs(calls, programs), [[1, ["p/a", "p/a"]], [2, ["p/b", "p/c", "p/c"]]])
        self.assertIsNone(call_programs(calls, [[], [("p/a", "a"), ("p/d", "d")], programs[2]]))
        self.assertIsNone(call_programs(calls, [[], programs[1], [("p/b", "b")]]))
        jump = dict(operation="atlas.jump", fields=dict(kind="jalr"))
        upper = dict(operation="atlas.upper", fields=dict(kind="auipc"))
        self.assertTrue(relocatable(dict(instructions=[dict(operation="atlas.upper", fields=dict(kind="lui"))])))
        self.assertFalse(relocatable(dict(instructions=[jump])))
        self.assertFalse(relocatable(dict(instructions=[upper])))


if __name__ == "__main__":
    unittest.main()
