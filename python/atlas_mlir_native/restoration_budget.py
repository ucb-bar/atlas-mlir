"""Fail-closed dynamic instruction bound for generated restoration images.

The source proves loop trips from unsigned, constant seeded affine inductions.
The verified final machine CFG must have exactly those counted loops. Every
encoded word is charged on every visit to its owning block, including untaken
edge copies, both numeric arms, spills, waits, and delay slots. This intentionally
overcounts branches; it does not depend on observed activation exponents.
"""
from collections import Counter, defaultdict
from math import prod
import re


MAX_INSTRUCTIONS = 1_000_000
# Full scalar admission remains available for small images. Larger images have
# an explicit zero fallback budget and report a domain/resource refusal instead.
MAX_SCALAR_TILES = 2


def fallback_tiles(plan):
    count = prod(plan.batch_shape) * ((plan.m + 31)//32) * ((plan.n + 31)//32)
    return count if count <= MAX_SCALAR_TILES else 0


def _natural_loops(successors, entry):
    nodes = set(successors)
    predecessors = {node: set() for node in nodes}
    for node, targets in successors.items():
        for target in targets:
            if target not in nodes:
                raise ValueError('instruction bound: CFG target is absent')
            predecessors[target].add(node)
    reachable, pending = set(), [entry]
    while pending:
        node = pending.pop()
        if node in reachable:
            continue
        reachable.add(node)
        pending.extend(successors[node])
    if reachable != nodes:
        raise ValueError('instruction bound: unreachable CFG block')
    dominators = {node: ({entry} if node == entry else set(nodes)) for node in nodes}
    while True:
        updated = {node: ({entry} if node == entry else
                   {node} | set.intersection(*(dominators[p] for p in predecessors[node])))
                   for node in nodes}
        if updated == dominators:
            break
        dominators = updated
    loops, backedges = {}, set()
    for tail, targets in successors.items():
        for header in targets:
            if header not in dominators[tail]:
                continue
            backedges.add((tail, header))
            members = {header, tail}
            pending = [] if tail == header else [tail]
            while pending:
                node = pending.pop()
                for predecessor in predecessors[node]:
                    if predecessor not in members:
                        members.add(predecessor)
                        pending.append(predecessor)
            loops.setdefault(header, set()).update(members)
    # Removing all certified backedges must leave a DAG. This excludes hidden
    # irreducible cycles whose visits would otherwise be charged only once.
    active, done = set(), set()
    def visit(node):
        if node in active:
            raise ValueError('instruction bound: uncounted CFG cycle')
        if node in done:
            return
        active.add(node)
        for target in successors[node]:
            if (node, target) not in backedges:
                visit(target)
        active.remove(node)
        done.add(node)
    visit(entry)
    for header, members in loops.items():
        if any(header not in dominators[node] for node in members):
            raise ValueError('instruction bound: loop has an uncounted entry')
        targets = successors[header]
        if len(targets) != 2 or targets[0] not in members or targets[1] in members:
            raise ValueError('instruction bound: loop exit is not its counted false arm')
    for first in loops.values():
        for second in loops.values():
            if first & second and not (first <= second or second <= first):
                raise ValueError('instruction bound: overlapping nonnested loops')
    return loops, predecessors


def _source_loop_trips(source):
    block = '__entry__'
    successors, arguments, incoming = {block: []}, {block: []}, defaultdict(list)
    constants, additions, comparisons, conditions = {}, {}, {}, {}
    for raw in source.splitlines():
        line = raw.strip()
        label = re.match(r'\^([\w.$-]+)\((.*)\):', line)
        if label:
            block = label[1]
            successors[block] = []
            arguments[block] = re.findall(r'(%[\w.$-]+)\s*:', label[2])
        constant = re.match(r'(%[\w.$-]+) = arith.constant (-?\d+) : i32', line)
        if constant:
            constants[constant[1]] = int(constant[2]) & 0xffffffff
        addition = re.match(r'(%[\w.$-]+) = arith.addi (%[\w.$-]+), (%[\w.$-]+)', line)
        if addition:
            additions[addition[1]] = (addition[2], addition[3])
        comparison = re.match(r'(%[\w.$-]+) = arith.cmpi (\w+), (%[\w.$-]+), (%[\w.$-]+)', line)
        if comparison:
            comparisons[comparison[1]] = (comparison[2], comparison[3], comparison[4])
        if line.startswith(('cf.br ', 'cf.cond_br ')):
            edges = re.findall(r'\^([\w.$-]+)\(([^)]*)\)', line)
            successors[block] = [target for target, _ in edges]
            for target, values in edges:
                incoming[target].append((block, re.findall(r'%[\w.$-]+', values.split(' : ')[0])))
            if line.startswith('cf.cond_br '):
                conditions[block] = re.search(r'cf.cond_br (%[\w.$-]+)', line)[1]
    loops, _ = _natural_loops(successors, '__entry__')
    result = []
    for header, members in loops.items():
        predicate, induction, bound = comparisons[conditions[header]]
        if predicate != 'ult' or bound not in constants or induction not in arguments[header]:
            raise ValueError('instruction bound: source loop is not an unsigned constant-bound induction')
        index = arguments[header].index(induction)
        seeds, steps = set(), set()
        for predecessor, values in incoming[header]:
            value = values[index]
            if predecessor not in members:
                seeds.add(constants[value])
            else:
                lhs, rhs = additions[value]
                if induction not in (lhs, rhs):
                    raise ValueError('instruction bound: source induction update differs')
                steps.add(constants[rhs if lhs == induction else lhs])
        if len(seeds) != 1 or len(steps) != 1 or not next(iter(steps)):
            raise ValueError('instruction bound: ambiguous loop seed or step')
        seed, step, bound = next(iter(seeds)), next(iter(steps)), constants[bound]
        trips = max(0, (bound-seed+step-1)//step)
        if seed+trips*step > 0xffffffff:
            raise ValueError('instruction bound: source induction wraps')
        result.append((bound, trips))
    return result


def _fresh_counter_read(header, members, induction, old, *, restores, effects, successors):
    number, _, slot = restores[induction]
    old_number, old_block, old_slot = restores[old]
    if old_block not in members or old_slot != slot:
        return False
    # Writer-set equality alone conflates dynamic loop iterations. Start
    # after this header's actual induction read, cut its backedges, and
    # reject any path that writes the counter before the update's read.
    # A nested-loop or hoisted stale reload cannot masquerade as this value.
    reached, pending, seen = False, [(header, False)], set()
    while pending:
        block, dirty = pending.pop()
        if (block, dirty) in seen:
            continue
        seen.add((block, dirty))
        started = block != header
        for kind, identifier, cell, value in effects[block]:
            if not started:
                if identifier == number:
                    started = True
                    if old_number == number:
                        reached = True
                elif (kind == 'save' and cell == slot) or (kind == 'copy' and value == slot):
                    # Predecessor seeds/updates do not describe a counter
                    # overwritten by its own header before the comparison.
                    return False
                continue
            if kind == 'restore' and identifier == old_number:
                if dirty:
                    return False
                reached = True
            if (kind == 'save' and cell == slot) or (kind == 'copy' and value == slot):
                dirty = True
        pending.extend((target, dirty) for target in successors[block]
                       if target != header and target in members)
    return reached

def instruction_bound(source, machine, program):
    """Return a conservative physical bound or raise on an unproved image.

    Call only after the existing CFG, control, address and timing verifiers.
    Their source contract ties every machine comparison to its captured scalar
    producer; this routine adds the stricter per-launch resource admission.
    """
    source_loops = _source_loop_trips(source)
    trips_by_bound = defaultdict(set)
    for bound, trips in source_loops:
        trips_by_bound[bound].add(trips)
    contract = machine.split('atlas.virtual_cfg_contract = ', 1)[1]
    block_text, remaining = contract.split('blocks = [', 1)[1].split('], edges = [', 1)
    edge_text, remaining = remaining.split('], operations = [', 1)
    operation_text = remaining.split('], output_buffer = ', 1)[0]
    value_text = remaining.split('values = [', 1)[1].split(']}', 1)[0]
    def records(text):
        return re.findall(r'\{([^{}]*)\}', text)
    def integer(text, key):
        found = re.search(r'\b'+key+r' = (-?\d+) : i32', text)
        if not found:
            raise ValueError('instruction bound: missing '+key)
        return int(found[1])
    def array(text, key):
        found = re.search(r'\b'+key+r' = array<i32(?:: ([\d, -]+))?>', text)
        if not found:
            raise ValueError('instruction bound: missing '+key+' array')
        return [] if found[1] is None else [int(value) for value in found[1].split(',')]
    successors, conditions, arguments, block_operations = {}, {}, {}, {}
    for record in records(block_text):
        block = integer(record, 'id')
        successors[block] = []
        conditions[block] = integer(record, 'condition')
        arguments[block] = array(record, 'args')
        block_operations[block] = array(record, 'operations')
    incoming = defaultdict(list)
    for record in records(edge_text):
        predecessor, target = integer(record, 'from'), integer(record, 'to')
        successors[predecessor].append(target)
        incoming[target].append((predecessor, array(record, 'incoming')))
    values = {integer(record, 'id'): record for record in records(value_text)}
    operations = {integer(record, 'id'): record for record in records(operation_text)}
    loops, predecessors = _natural_loops(successors, 0)
    # Reconstruct scalar spill reaching definitions from the captured contract.
    # An indexed BF16 load is never a counted induction origin. Slot copies
    # forward the exact writer set, and an uninitialized writer cannot prove a
    # constant loop seed or an update from the same induction cell.
    scalar_slots = set()
    effects = defaultdict(list)
    stores, restores = {}, {}
    for block, identifiers in block_operations.items():
        for identifier in identifiers:
            record = operations[identifier]
            if not re.search(r'slot_kind = "i(?:1|32)"', record):
                continue
            slot = integer(record, 'slot_word')
            scalar_slots.add(slot)
            if 'name = "atlas.virtual_save"' in record:
                operands = array(record, 'operands')
                if len(operands) != 1:
                    raise ValueError('instruction bound: malformed scalar save')
                stores[identifier] = operands[0]
                effects[block].append(('save', identifier, slot, operands[0]))
            elif 'name = "atlas.virtual_restore"' in record:
                results = array(record, 'results')
                if len(results) != 1:
                    raise ValueError('instruction bound: malformed scalar restore')
                restores[results[0]] = (identifier, block, slot)
                effects[block].append(('restore', identifier, slot, results[0]))
            elif 'name = "atlas.virtual_slot_copy"' in record:
                destination = integer(record, 'destination_word')
                scalar_slots.add(destination)
                effects[block].append(('copy', identifier, slot, destination))
            else:
                raise ValueError('instruction bound: unknown scalar memory effect')
    bottom = {slot: frozenset() for slot in scalar_slots}
    out_memory = {block: dict(bottom) for block in successors}
    read_writers = {}
    for _ in range(10000):
        changed = False
        for block in successors:
            state = ({slot: frozenset({'uninitialized'}) for slot in scalar_slots} if block == 0 else
                     {slot: frozenset().union(*(out_memory[p][slot] for p in predecessors[block])) for slot in scalar_slots})
            for kind, identifier, slot, value in effects[block]:
                if kind == 'save':
                    state[slot] = frozenset({identifier})
                elif kind == 'copy':
                    state[value] = state[slot]
                else:
                    read_writers[identifier] = state[slot]
            if state != out_memory[block]:
                out_memory[block] = state
                changed = True
        if not changed:
            break
    else:
        raise ValueError('instruction bound: scalar reaching definitions did not converge')
    def constant(value):
        return integer(values[value], 'constant') & 0xffffffff
    def addition(value):
        record = values[value]
        operands = array(record, 'operands')
        if 'def = "arith.addi"' not in record or len(operands) != 2:
            raise ValueError('instruction bound: induction update is not addition')
        return operands
    loop_trips, final_bounds = {}, []
    final_inventory = []
    final_proofs = {}
    for header, members in loops.items():
        comparison = values[conditions[header]]
        # MLIR CmpIPredicate::ult is 6. Every current generated loop retains
        # its original comparison bound in the verifier's captured contract.
        if 'def = "arith.cmpi"' not in comparison or integer(comparison, 'predicate') != 6:
            raise ValueError('instruction bound: final loop condition is not ult')
        operands = array(comparison, 'operands')
        if len(operands) != 2:
            raise ValueError('instruction bound: final loop has no captured operands')
        induction, bound_id = operands
        bound = constant(bound_id)
        if bound not in trips_by_bound:
            raise ValueError('instruction bound: final loop bound differs from source')
        seeds, steps = set(), set()
        if induction in arguments[header]:
            index = arguments[header].index(induction)
            for predecessor, edge_values in incoming[header]:
                value = edge_values[index]
                if predecessor not in members:
                    seeds.add(constant(value))
                else:
                    lhs, rhs = addition(value)
                    if induction not in (lhs, rhs):
                        raise ValueError('instruction bound: final induction update differs')
                    steps.add(constant(rhs if lhs == induction else lhs))
        else:
            if induction not in restores or restores[induction][1] != header:
                raise ValueError('instruction bound: unknown final induction origin')
            number, _, slot = restores[induction]
            for predecessor in predecessors[header]:
                writers = out_memory[predecessor][slot]
                if not writers or any(type(writer) is not int for writer in writers):
                    raise ValueError('instruction bound: undefined final induction cell')
                for writer in writers:
                    value = stores[writer]
                    if predecessor not in members:
                        seeds.add(constant(value))
                    else:
                        pair = addition(value)
                        constants = [item for item in pair if 'constant =' in values[item]]
                        if len(constants) != 1:
                            raise ValueError('instruction bound: final memory induction has no constant step')
                        steps.add(constant(constants[0]))
                        old = pair[1] if pair[0] == constants[0] else pair[0]
                        if (old not in restores or restores[old][2] != slot or
                                read_writers[restores[old][0]] != read_writers[number] or
                                not _fresh_counter_read(header, members, induction, old,
                                                        restores=restores, effects=effects, successors=successors)):
                            raise ValueError('instruction bound: final memory induction reads another counter')
        if len(seeds) != 1 or len(steps) != 1 or not next(iter(steps)):
            raise ValueError('instruction bound: ambiguous or zero final induction step')
        seed, step = next(iter(seeds)), next(iter(steps))
        trips = max(0, (bound-seed+step-1)//step)
        if seed+trips*step > 0xffffffff:
            raise ValueError('instruction bound: final induction wraps')
        loop_trips[header] = trips
        final_inventory.append((bound, trips))
        final_proofs[header] = dict(seed=seed, step=step, bound=bound)
        final_bounds.append(bound)
    if Counter(final_inventory) != Counter(source_loops):
        raise ValueError('instruction bound: final/source counted loop inventories differ')
    visits = {block: prod((loop_trips[header]+1 if block == header else loop_trips[header])
                         for header, members in loops.items() if block in members)
              for block in successors}
    words = Counter()
    for index, instruction in enumerate(program['instructions']):
        fields = instruction.get('fields', {})
        if 'atlas.virtual_cfg_helper' in fields:
            raise ValueError('instruction bound: internal helper control is not admitted')
        block = fields.get('atlas.virtual_cfg_block')
        if instruction['word_index'] != index or block not in visits:
            raise ValueError('instruction bound: encoded word lacks unique CFG ownership')
        words[block] += 1
    bound = sum(words[block]*visits[block] for block in successors)
    return dict(schema='atlas.restoration_instruction_bound.v1', max_instructions=MAX_INSTRUCTIONS,
                instruction_bound=bound, admitted=bound <= MAX_INSTRUCTIONS,
                method='sum_all_encoded_words_times_proved_maximum_block_visits',
                source_loop_inventory=[dict(bound=bound, trip_count=trips) for bound, trips in sorted(source_loops)],
                loops=[dict(header=header, trip_count=loop_trips[header], **final_proofs[header], blocks=sorted(members))
                       for header, members in sorted(loops.items())],
                blocks=[dict(block=block, encoded_words=words[block], maximum_visits=visits[block])
                        for block in sorted(successors)])
