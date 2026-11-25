"""The checker's resident-image form: every start is bound to its published image and entry word.

The images here write no memory, so every output buffer holds the same bytes whichever image runs, and a refusal can
only come from the start's binding, never from an output comparison. npu_model is imported in each test, so an
interpreter that cannot import it fails these tests instead of collection.
"""
import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = 0x90000000
IMEM, START, STATUS = 0x20000, 0x40018, 0x40008
PLAN = dict(output_addresses=[BASE + 64], output_bytes=4, input_addresses=[BASE], input_payload_bytes=[4])


def isa():
    from npu_model.configs import isa_definition
    return isa_definition


def encode(*instructions):
    return [instruction.to_bytecode() for instruction in instructions]


def raw(words):
    return b"".join(word.to_bytes(4, "little") for word in words)


def images():
    """Trip 0 and trip 1 of one program, differing in two operand-address words, and a program with a relocated
    branch and its delay slot, plus programs that leave their body backward, into the entry sequence, and forward."""
    i = isa()
    trip = lambda low, high: encode(i.ADDI(rd=5, rs1=0, imm=low), i.ADDI(rd=6, rs1=0, imm=high), i.ECALL())
    return dict(
        a0=trip(64, 128), a1=trip(320, 384),
        # The BEQ at body word 1 is taken to body word 4; body word 2 is its delay slot and word 3 is skipped.
        branch=encode(i.ADDI(rd=1, rs1=0, imm=1), i.BEQ(rs1=0, rs2=0, imm=6), i.ADDI(rd=1, rs1=1, imm=2),
                      i.ADDI(rd=1, rs1=1, imm=100), i.ECALL()),
        backward=encode(i.BEQ(rs1=0, rs2=0, imm=-2), i.ADDI(rd=0, rs1=0, imm=0), i.ECALL()),
        forward=encode(i.BEQ(rs1=0, rs2=0, imm=6), i.ADDI(rd=0, rs1=0, imm=0), i.ECALL()))


def jump(entry):
    return isa().JAL(rd=0, imm=2 * entry).to_bytecode()


NOP = 0x13


class Host:
    """Drives the runtime's MMIO callbacks the way the host runtime does."""

    def __init__(self, runtime):
        self.runtime = runtime

    def put(self, index, words):
        for offset, word in enumerate(words):
            self.runtime.write(None, IMEM + 4 * (index + offset), word)

    def select(self, entry, delay=NOP):
        self.put(1, [delay])
        self.put(0, [jump(entry)])

    def start(self):
        before = len(self.runtime.errors)
        self.runtime.write(None, START, 1)
        while self.runtime.running:
            self.runtime.read(None, STATUS)
        return self.runtime.errors[before:]


@pytest.fixture
def make(tmp_path, monkeypatch):
    import functional_runtime
    import model_runtime

    monkeypatch.setattr(model_runtime, "authenticate_model", lambda _: {})
    monkeypatch.setattr(functional_runtime, "authenticate_model", lambda _: {})
    words = images()
    digests = {name: hashlib.sha256(raw(image)).hexdigest() for name, image in words.items()}
    counter = iter(range(1000))

    def build(schedule=None, imem_words=64):
        published = [raw(image) for image in words.values()]
        resident = None if schedule is None else dict(
            starts=[(digests[name], entry) for name, entry in schedule], imem_words=imem_words)
        runtime = functional_runtime.FunctionalModelRuntime(
            ROOT, ROOT, published, tmp_path / f"execution-{next(counter)}", memory_base=BASE, memory_bytes=256,
            image_plans={digest: PLAN for digest in digests.values()}, resident_images=resident)
        return runtime, Host(runtime), words, digests

    return build


def test_multi_trip_execution_with_changing_addresses_and_a_relocated_branch(make):
    a0_entry, branch_entry = 2, 5
    runtime, host, words, digests = make([("a0", a0_entry), ("branch", branch_entry), ("a1", a0_entry),
                                          ("branch", branch_entry)])
    host.put(a0_entry, words["a0"])
    host.put(branch_entry, words["branch"])
    host.select(a0_entry)
    assert host.start() == []
    host.select(branch_entry)
    assert host.start() == []
    # Trip 1 rewrites only its two operand-address words in trip 0's slot.
    host.put(a0_entry, words["a1"][:2])
    host.select(a0_entry)
    assert host.start() == []
    host.select(branch_entry)
    assert host.start() == []
    record = runtime.finish()
    assert [(row["image_sha256"], row["imem_entry"]) for row in record["launches"]] == [
        (digests["a0"], 2), (digests["branch"], 5), (digests["a1"], 2), (digests["branch"], 5)]
    # Jump, delay slot, and body: the branch program executes ADDI, BEQ, its delay slot, and ECALL.
    assert [row["instructions"] for row in record["launches"]] == [5, 6, 5, 6]
    assert record["imem_residency"]["imem_words_written"] == 3 + 5 + 2 * 4 + 2


def test_stale_earlier_trip_image_refuses_before_execution(make):
    runtime, host, words, _ = make([("a0", 2), ("a1", 2)])
    host.put(2, words["a0"])
    host.select(2)
    assert host.start() == []
    errors = host.start()  # trip 1 expects a1, and IMEM still holds trip 0's published a0
    assert len(errors) == 1 and "differs from this start's compiled image" in errors[0]
    assert len(runtime.launches) == 1


def test_jump_to_another_resident_slot_refuses(make):
    _, host, words, _ = make([("a1", 2)])
    host.put(2, words["a1"])
    host.put(5, words["a1"])  # an identical published body at another offset
    host.select(5)
    errors = host.start()
    assert len(errors) == 1 and "word 0 is not the jump to this start's published entry" in errors[0]


def test_entry_off_by_one_refuses(make):
    _, host, words, _ = make([("a0", 2), ("a0", 2)])
    host.put(2, words["a0"])
    host.select(3)
    assert "word 0 is not the jump" in host.start()[0]
    host.select(1)
    assert "word 0 is not the jump" in host.start()[0]


def test_half_updated_address_row_refuses(make):
    _, host, words, _ = make([("a0", 2), ("a1", 2)])
    host.put(2, words["a0"])
    host.select(2)
    assert host.start() == []
    host.put(2, words["a1"][:1])  # one of the two operand-address words
    assert "differs from this start's compiled image" in host.start()[0]


def test_overwrite_between_starts_refuses_until_the_exact_words_return(make):
    runtime, host, words, _ = make([("a0", 2), ("a0", 2), ("a0", 2), ("branch", 5)])
    host.put(2, words["a0"])
    host.put(5, words["branch"])
    host.select(2)
    assert host.start() == []
    host.put(3, words["a1"][1:2])  # a published image's word, written over the cached body
    assert "differs from this start's compiled image" in host.start()[0]
    host.put(3, words["a0"][1:2])
    host.put(7, [NOP])  # corrupt the unselected branch body: caught only when it is selected
    assert host.start() == []
    host.select(5)
    assert "differs from this start's compiled image" in host.start()[0]
    assert len(runtime.launches) == 2


def test_delay_slot_that_is_not_a_nop_refuses(make):
    _, host, words, _ = make([("a0", 2)])
    host.put(2, words["a0"])
    host.select(2, delay=isa().ADDI(rd=1, rs1=0, imm=1).to_bytecode())
    assert "delay slot, is not a NOP" in host.start()[0]


@pytest.mark.parametrize("name,entry,message", [
    ("backward", 4, "left the launched image backward"),
    ("backward", 2, "re-entered the IMEM entry sequence"),
    ("forward", 4, "failed to halt normally"),
])
def test_execution_leaving_the_body_refuses(make, name, entry, message):
    _, host, words, _ = make([(name, entry)])
    host.put(entry, words[name])
    host.select(entry)
    errors = host.start()
    assert len(errors) == 1 and message in errors[0]


def test_missing_and_extra_starts_refuse(make):
    runtime, host, words, _ = make([("a0", 2), ("a0", 2)])
    host.put(2, words["a0"])
    host.select(2)
    assert host.start() == []
    with pytest.raises(AssertionError, match="1 starts differ from the published resident schedule's 2"):
        runtime.finish()
    runtime, host, words, _ = make([("a0", 2)])
    host.put(2, words["a0"])
    host.select(2)
    assert host.start() == []
    assert "start beyond the published resident schedule" in host.start()[0]


def test_entry_and_body_are_bounded_by_imem(make):
    _, host, words, _ = make([("a0", 62)], imem_words=64)
    host.put(62, words["a0"])
    assert host.runtime.errors == ["instruction memory write beyond the declared IMEM"]
    host.select(62)
    assert "published resident entry or image lies outside IMEM" in host.start()[0]


def test_word_zero_form_refuses_on_a_resident_kit(make):
    _, host, words, _ = make([("a0", 2)])
    host.put(0, words["a0"])
    assert "word 0 is not the jump" in host.start()[0]


def test_a_kit_without_residency_keeps_the_word_zero_rule(make):
    runtime, host, words, digests = make()
    host.put(0, words["a0"])
    assert host.start() == []
    host.put(0, words["a1"])  # a word-0 write clears IMEM, so the shorter image replaces it whole
    assert host.start() == []
    host.put(2, words["a0"])
    host.select(2)  # the jump form is not admitted without a declaration
    assert "launched image differs from compiled publication" in host.start()[0]
    assert [row["image_sha256"] for row in runtime.launches] == [digests["a0"], digests["a1"]]
    assert all("imem_entry" not in row for row in runtime.launches)


def deployment(images, starts, regions, calls, imem_bytes=256):
    """A provider record whose images are published at path "p/<digest>" and whose calls name those paths."""
    return dict(provider=dict(host_map=dict(imem_bytes=imem_bytes),
                              images=[dict(path=f"p/{digest}", sha256=digest) for digest in sorted(set(images))],
                              regions=regions, imem_residency=dict(
                                  schema="atlas.imem_residency.v1", runtime="atlas_host_keep_images",
                                  copy_words=imem_bytes // 4, launches=len(starts), images=images, starts=starts,
                                  calls=[[index, [f"p/{digest}" for digest in sequence]] for index, sequence in calls])))


# Two calls without a launch schedule, one launch each, then a call whose schedule launches "a" twice.
REGIONS = [dict(context_index=0, launches_per_invocation=0), dict(context_index=1, launches_per_invocation=1),
           dict(context_index=2, launches_per_invocation=1),
           dict(context_index=3, launches_per_invocation=2, device_launch_schedule=dict(
               images=["b", "a"], groups=[dict(repeat=2, runs=[dict(image=1, count=1)])]))]
CALLS = [(1, ["b"]), (2, ["c"]), (3, ["a", "a"])]
IMAGES = ["b", "c", "a"]
STARTS = [[0, 2], [1, 5], [2, 8], [2, 8]]


def test_schedule_binds_every_launching_call_to_its_compiled_programs():
    from functional_runtime import resident_schedule

    assert resident_schedule(deployment(IMAGES, STARTS, REGIONS, CALLS)) == dict(
        starts=[("b", 2), ("c", 5), ("a", 8), ("a", 8)], imem_words=64)
    assert resident_schedule(dict(provider=dict(regions=REGIONS))) is None
    refusals = [
        # The unscheduled calls' images swapped: counts and outputs are equal, the starts no longer match the calls.
        (deployment(IMAGES, [[1, 5], [0, 2], [2, 8], [2, 8]], REGIONS, CALLS), "differs from the compiled calls"),
        # A scheduled call whose published programs differ from its launch schedule.
        (deployment(IMAGES, [[0, 2], [1, 5], [0, 2], [2, 8]], REGIONS, [(1, ["b"]), (2, ["c"]), (3, ["b", "a"])]),
         "compiled launch schedule"),
        # Missing, short, misattributed, or unpublished program identity.
        (deployment(IMAGES, STARTS, REGIONS, CALLS[:2]), "lacks a launching call's program identity"),
        (deployment(IMAGES, STARTS, REGIONS, [(1, []), (2, ["c"]), (3, ["a", "a"])]), "lacks a launching call"),
        (deployment(IMAGES, STARTS, REGIONS, [(2, ["b"]), (1, ["c"]), (3, ["a", "a"])]), "lacks a launching call"),
        (deployment(IMAGES, STARTS, REGIONS, [(1, ["x"]), (2, ["c"]), (3, ["a", "a"])]), "lacks a launching call"),
        # An extra start, and a start naming no published image.
        (deployment(IMAGES, STARTS + [[2, 8]], REGIONS, CALLS), "differs from the compiled calls"),
        (deployment(IMAGES, [[0, 2], [1, 5], [2, 8], [3, 8]], REGIONS, CALLS), "record differs from the deployment"),
    ]
    missing = deployment(IMAGES, STARTS, REGIONS, CALLS)
    del missing["provider"]["imem_residency"]["calls"]
    refusals.append((missing, "lacks a launching call's program identity"))
    for record, message in refusals:
        with pytest.raises(AssertionError, match=message):
            resident_schedule(record)


def test_host_that_swaps_two_unscheduled_calls_images_refuses(make):
    """The host runs call 2's image for call 1 and the reverse: equal counts and equal outputs, refused at start 1."""
    from functional_runtime import resident_schedule

    runtime, host, words, digests = make([("a0", 2), ("branch", 5)])
    regions = [dict(context_index=1, launches_per_invocation=1), dict(context_index=2, launches_per_invocation=1)]
    record = deployment([digests["a0"], digests["branch"]], [[0, 2], [1, 5]], regions,
                        [(1, [digests["a0"]]), (2, [digests["branch"]])])
    assert resident_schedule(record)["starts"] == [(digests["a0"], 2), (digests["branch"], 5)]
    host.put(2, words["branch"])
    host.put(7, words["a0"])
    host.select(2)
    assert "differs from this start's compiled image" in host.start()[0]
    assert runtime.launches == []


def test_counters_must_equal_the_published_record():
    spec = importlib.util.spec_from_file_location("check_model_counters", ROOT / "python/atlas_mlir/exactness/check_model.py")
    check_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_model)
    published = dict(imem_words=dict(resident=10), launches=3, image_loads=2)
    counters = dict(imem_words_written=10, launches=3, image_loads=2, image_reuses=1)
    check_model.check_resident_counters("case", published, dict(imem_residency=dict(imem_words_written=10)), counters)
    for key, value in (("imem_words_written", 11), ("launches", 4), ("image_loads", 1)):
        with pytest.raises(AssertionError, match="resident IMEM counters differ"):
            check_model.check_resident_counters("case", published, dict(imem_residency=dict(imem_words_written=10)),
                                                {**counters, key: value})
    with pytest.raises(AssertionError, match="resident IMEM counters differ"):
        check_model.check_resident_counters("case", published, dict(imem_residency=dict(imem_words_written=9)), counters)
