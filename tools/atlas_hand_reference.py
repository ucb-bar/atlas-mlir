"""Independent directed rewriting control for the bounded Atlas experiment.

This intentionally hand-written comparator applies the same qualified local
families, then enumerates native/unit0/unit1 per finite PACK. It is not a second
extensible framework. All decoded candidates still need the normal compiler
and measured byte gates. No claim of global optimality follows from its cost
shortlist or its greedy directed simplification.
"""
from dataclasses import replace
from itertools import product
from atlas_egglog_ir import Node, parse_program, pure_roots, decode_program
from atlas_egglog_facts import analyze_node, analyze_program
from atlas_egglog_pack import recipe, choices_from_roots, materialize


def simplify(root, program, facts):
    memo={}
    def visit(node):
        if node in memo:return memo[node]
        current=Node(node.op,tuple(visit(x) for x in node.operands),node.attrs,node.type)
        def kind(value, wanted):
            return value.op=='virtual_vpu_binary' and value.attrs==(('kind','"'+wanted+'"'),)
        def information(value):return analyze_node(value,program_facts=facts,program=program)
        operands=current.operands
        if current.op=='virtual_vpu_unary' and current.attrs==(('kind','"mov"'),):current=operands[0]
        elif current.op=='virtual_xlu_transpose' and not current.attrs:
            inner=operands[0]
            if inner.op=='virtual_xlu_transpose' and not inner.attrs:current=inner.operands[0]
        elif current.op=='virtual_vpu_unary' and current.attrs==(('kind','"relu"'),):
            inner=operands[0]
            if information(inner).raw_nonnegative:current=inner
            elif inner.op=='virtual_vpu_unary' and inner.attrs==current.attrs and information(inner.operands[0]).finite:current=inner
        elif kind(current,'min') or kind(current,'max'):
            left,right=operands; minimum=kind(current,'min'); opposite='max' if minimum else 'min'
            if left==right:current=left
            elif kind(right,opposite) and right.operands[0]==left:current=left
            elif kind(left,opposite) and left.operands[0]==right:current=right
            else:
                lf,rf=information(left),information(right)
                if lf.interval is not None and rf.interval is not None:
                    if lf.interval[1]<=rf.interval[0]:current=left if minimum else right
                    elif rf.interval[1]<=lf.interval[0]:current=right if minimum else left
        memo[node]=current
        return current
    return visit(root)


def enumerate_hand(program, *, max_sites=6):
    """Yield all 3^N assignments for N<=max_sites, plus original native control.

    Directed pure simplification is fixed across assignments. Enumerating the
    complete engine/native product does not enumerate every algebraic shape.
    """
    facts=analyze_program(program)
    roots={name:simplify(node,program,facts) for name,node in pure_roots(program).items()}
    simplified=parse_program(decode_program(program,roots))
    facts=analyze_program(simplified)
    sites=[op.results[0] for op in simplified.operations if op.op=='virtual_pack_fp8' and facts.values[op.operands[0]].finite]
    if len(sites)>max_sites:raise ValueError('hand product exceeds explicit site budget')
    roots=pure_roots(simplified)
    yield 'native',''.join(program.lines),(),None
    for index,assignment in enumerate(product((-1,0,1),repeat=len(sites))):
        selected=dict(roots)
        for name,unit in zip(sites,assignment):
            if unit>=0:selected[name]=recipe(simplified.nodes[name],unit)
        choices=choices_from_roots(simplified,selected)
        choices=tuple(replace(choice,clamped=simplify(choice.clamped,simplified,facts)) for choice in choices)
        try:
            converted=parse_program(materialize(simplified,choices))
            source=decode_program(converted,pure_roots(converted),prune_dead_pure=True)
            yield f'hand-exhaustive-{index:03}',source,assignment,None
        except ValueError as error:
            yield f'hand-exhaustive-{index:03}',None,assignment,str(error)
