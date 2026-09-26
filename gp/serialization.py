"""
Serialización de individuos a JSON y viceversa.

Formato:
{
  "root": {
    "name": "IF",
    "feature_name": null,
    "children": [...]
  }
}
"""

from __future__ import annotations

import json
from pathlib import Path

from gp.evaluator import TreeNode
from gp.grammar import DEFAULT_GRAMMAR, Grammar
from gp.individual import Individual
from gp.nodes import Node
from gp.types import GPType


def _build_node_index(grammar: Grammar) -> dict[str, Node]:
    """Indexa todos los nodos de la gramática por nombre."""
    index: dict[str, Node] = {}
    for t in (GPType.POSITION, GPType.BOOLEAN, GPType.REAL):
        for node in grammar.nodes_for(t):
            index[node.name] = node
    return index


def tree_to_dict(tree: TreeNode) -> dict:
    return {
        "name": tree.node.name,
        "feature_name": tree.feature_name,
        "children": [tree_to_dict(c) for c in tree.children],
    }


def dict_to_tree(
    d: dict,
    node_index: dict[str, Node],
) -> TreeNode:
    name = d["name"]
    if name not in node_index:
        raise ValueError(f"Nodo desconocido en serialización: '{name}'")
    node = node_index[name]

    children = [dict_to_tree(c, node_index) for c in d.get("children", [])]
    feature_name = d.get("feature_name")

    tree = TreeNode(node=node, children=children, feature_name=feature_name)
    return tree


def individual_to_dict(ind: Individual) -> dict:
    return {"root": tree_to_dict(ind.root)}


def dict_to_individual(
    d: dict,
    grammar: Grammar = DEFAULT_GRAMMAR,
) -> Individual:
    node_index = _build_node_index(grammar)
    root = dict_to_tree(d["root"], node_index)
    ind = Individual(root=root)
    ind.validate()
    return ind


def save_individual(ind: Individual, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(individual_to_dict(ind), f, indent=2)


def load_individual(
    path: Path | str,
    grammar: Grammar = DEFAULT_GRAMMAR,
) -> Individual:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return dict_to_individual(d, grammar)