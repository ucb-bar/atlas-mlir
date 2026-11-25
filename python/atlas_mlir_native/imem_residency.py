"""Choose between one image at IMEM word 0 and several resident images, by IMEM words written.

Both rules replay the host runtime (runtime/atlas_host.c) over a deployment's static
launch order: atlas_host_ensure_loaded keeps one image at word 0, and
atlas_host_keep_images keeps several, rewriting only differing words when an image
replaces the least recently used one of its length. A loop over repeated layers is
what makes images recur; nothing here depends on the loop or the model.
"""
from __future__ import annotations


def launch_order(calls, program_images):
    """Image digests in launch order; program_images maps each call's programs to digests."""
    order = []
    for call, programs in zip(calls, program_images):
        schedule = call.get("device_launch_schedule")
        if schedule is None:
            order += programs[:1] * call["launches_per_invocation"]
            continue
        for group in schedule["groups"]:
            for _ in range(group["repeat"]):
                for run in group["runs"]:
                    order += [schedule["images"][run["image"]]] * run["count"]
    return order


def call_programs(calls, programs):
    """Each launching call's program path per launch, or None when a launch's program is not determined.

    programs lists each call's (path, digest) pairs. A call with a launch schedule names each launch's image by
    digest, which must be one of its own programs; a call without one must have exactly one program.
    """
    rows = []
    for call, owned in zip(calls, programs, strict=True):
        count = call["launches_per_invocation"]
        if not count:
            continue
        schedule = call.get("device_launch_schedule")
        if schedule is None:
            if len(owned) != 1:
                return None
            rows.append([call["context_index"], [owned[0][0]] * count])
            continue
        paths = {}
        for path, digest in owned:
            paths.setdefault(digest, path)
        sequence = [paths.get(schedule["images"][run["image"]])
                    for group in schedule["groups"] for _ in range(group["repeat"])
                    for run in group["runs"] for _ in range(run["count"])]
        if None in sequence or len(sequence) != count:
            return None
        rows.append([call["context_index"], sequence])
    return rows


def one_image_words(order, words, always_loaded):
    """atlas_host_ensure_loaded reuses only the image at word 0; atlas_host_load always writes."""
    written, resident = 0, None
    for digest in order:
        if digest in always_loaded:
            written, resident = written + len(words[digest]), None
        elif digest != resident:
            written, resident = written + len(words[digest]), digest
    return written


def resident_words(order, words, capacity_words):
    """IMEM words, peak resident images, image loads, and each start's entry word under atlas_host_keep_images."""
    copy = [None] * capacity_words
    slots, end, selected, written, loads, peak, entries = [], 2, None, 0, 0, 0, []
    for use, digest in enumerate(order):
        image = words[digest]
        count = len(image)
        same = [slot for slot in slots if slot[1] == count]
        slot = next((slot for slot in same if copy[slot[0]:slot[0] + count] == image), None)
        if slot is None:
            loads += 1
            if same:
                slot = min(same, key=lambda row: row[2])
                written += sum(old != new for old, new in zip(copy[slot[0]:slot[0] + count], image))
            else:
                if count > capacity_words - end:
                    slots, end, selected = [], 2, None
                slot = [end, count, 0]
                slots.append(slot)
                end += count
                written += count
            copy[slot[0]:slot[0] + count] = image
        slot[2] = use
        if selected != slot[0]:
            written, selected = written + 2, slot[0]
        peak = max(peak, len(slots))
        entries.append(slot[0])
    return written, peak, loads, entries


def choose(order, words, always_loaded, relocatable, capacity_words):
    """The resident record when every launched image is relocatable and it writes fewer words."""
    if (not order or capacity_words > 0x7ffff
            or any(not relocatable[digest] or len(words[digest]) > capacity_words - 2 for digest in order)):
        return None
    today = one_image_words(order, words, always_loaded)
    written, slots, loads, entries = resident_words(order, words, capacity_words)
    if written >= today:
        return None
    # The checker binds every start to its compiled image and entry word: word 0 holds JAL x0 to the entry,
    # word 1 a NOP, and the image's words follow from the entry.
    images = list(dict.fromkeys(order))
    index = {digest: i for i, digest in enumerate(images)}
    return dict(schema="atlas.imem_residency.v1", runtime="atlas_host_keep_images",
                copy_words=capacity_words, slots=slots, launches=len(order), image_loads=loads,
                imem_words=dict(one_image=today, resident=written),
                images=images, starts=[[index[digest], entry] for digest, entry in zip(order, entries)])


def relocatable(physical_program):
    """No register-indirect jump or PC-relative upper immediate; branches are encoded relative.

    A row that does not name its operation and fields cannot be shown relocatable."""
    return all(type(row.get("operation")) is str and isinstance(row.get("fields"), dict)
               and not (row["operation"] == "atlas.jump" and row["fields"].get("kind") == "jalr")
               and not (row["operation"] == "atlas.upper" and row["fields"].get("kind") == "auipc")
               for row in physical_program["instructions"])
