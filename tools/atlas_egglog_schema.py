"""Atlas node adapters for Merlin's optional target-neutral egglog forest.

Add the explicitly selected Merlin checkout's src directory to PYTHONPATH.
State/resource and anchor legality remains in the Atlas decoder.
"""
from merlin.semantic_compiler.egraph_terms import Value, Inputs, attrs_json
from merlin.semantic_compiler.egraph_terms import encode_node as to_egglog
from merlin.semantic_compiler.egraph_terms import decode_node as _decode
from atlas_egglog_ir import Node, IllegalExtraction


def from_egglog(value):
    try:
        return _decode(value, Node)
    except ValueError as error:
        raise IllegalExtraction(str(error)) from error


encode_node = to_egglog
decode_node = from_egglog
