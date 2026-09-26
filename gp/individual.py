"""
Individual: un árbol GP con metadatos.

Un Individual envuelve un TreeNode raíz y expone:
- evaluate(ctx) -> array de señales
- tamaño, profundidad, número de nodos por tipo
- construcción aleatoria que respeta la gramática

La construcción aleatoria usa la gramática para elegir nodos compatibles
con el tipo objetivo y respeta un límite de profundidad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.settings import DEFAULT_GP_LIMITS, GPLimits
from gp.evaluator import EvalContext, TreeNode, check_types, evaluate
from gp.grammar import DEFAULT_GRAMMAR, Grammar
from gp.nodes import Node
from gp.types import GPType


@dataclass
class Individual:
    """
    Un individuo GP.

    root: raíz del árbol.
    metadata: información opcional (id, generación, fitness, etc.).
    """
    root: TreeNode
    metadata: dict = field(default_factory=dict)

    def evaluate(self, ctx: EvalContext) -> np.ndarray:
        return evaluate(self.root, ctx)

    @property
    def size(self) -> int:
        return self.root.size()

    @property
    def depth(self) -> int:
        return self.root.depth()

    def node_counts(self) -> dict[str, int]:
        return self.root.count_nodes_by_name()

    def n_features(self) -> int:
        """Número de terminales de feature distintos en el árbol."""
        return len(self._collect_features())

    def n_conditions(self) -> int:
        """Número de nodos GT y LT (comparadores)."""
        counts = self.node_counts()
        return counts.get("GT", 0) + counts.get("LT", 0)

    def _collect_features(self) -> set[str]:
        out: set[str] = set()
        self._collect_recursive(self.root, out)
        return out

    @staticmethod
    def _collect_recursive(tree: TreeNode, out: set[str]) -> None:
        if tree.is_feature_terminal() and tree.feature_name:
            out.add(tree.feature_name)
        for c in tree.children:
            Individual._collect_recursive(c, out)

    def validate(self) -> None:
        """Verifica tipos y consistencia estructural."""
        check_types(self.root)


# ---------------------------------------------------------------------------
# Construcción aleatoria
# ---------------------------------------------------------------------------

def _min_depth_for_type(grammar: Grammar, target_type: GPType) -> int:
    """
    Profundidad mínima de un subárbol del tipo dado.

    - REAL: 1 (terminal de feature)
    - POSITION: 1 (terminal LONG/SHORT/FLAT)
    - BOOLEAN: 2 (GT o LT con dos REALs terminales)
    """
    if target_type in (GPType.REAL, GPType.POSITION):
        return 1
    if target_type == GPType.BOOLEAN:
        return 2
    raise ValueError(f"Tipo desconocido: {target_type}")


def _eligible_nodes(
    grammar: Grammar,
    target_type: GPType,
    remaining_depth: int,
) -> list[Node]:
    """
    Devuelve los nodos de `target_type` cuyo subárbol mínimo cabe en
    `remaining_depth`.

    remaining_depth es el número de niveles que aún podemos usar,
    contando el nodo actual como nivel 1.
    """
    out = []
    for node in grammar.nodes_for(target_type):
        # Profundidad mínima de este nodo = 1 + max(min_depth de sus hijos)
        if node.arity == 0:
            node_min = 1
        else:
            child_mins = [
                _min_depth_for_type(grammar, arg_t) for arg_t in node.arg_types
            ]
            node_min = 1 + max(child_mins)
        if node_min <= remaining_depth:
            out.append(node)
    return out


def _random_tree_recursive(
    grammar: Grammar,
    target_type: GPType,
    remaining_depth: int,
    rng: np.random.Generator,
) -> TreeNode:
    """
    Construye un subárbol aleatorio de tipo `target_type`.

    `remaining_depth` es cuántos niveles quedan disponibles (incluyendo este).
    Si remaining_depth <= 1, solo se pueden usar terminales.

    Lanza RuntimeError si no hay nodos elegibles, lo cual indicaría un
    problema en la gramática o en el cálculo de min_depth.
    """
    eligible = _eligible_nodes(grammar, target_type, remaining_depth)
    if not eligible:
        raise RuntimeError(
            f"No hay nodos elegibles para tipo {target_type.name} "
            f"con remaining_depth={remaining_depth}"
        )

    node = eligible[rng.integers(0, len(eligible))]

    # Terminal de feature (REAL con func=None)
    if node.func is None and node.arity == 0:
        return TreeNode(node=node, children=[], feature_name=node.name)

    # Terminal POSITION o cualquier terminal
    if node.arity == 0:
        return TreeNode(node=node, children=[], feature_name=None)

    # Nodo interno: construir hijos con remaining_depth - 1
    children = [
        _random_tree_recursive(grammar, arg_type, remaining_depth - 1, rng)
        for arg_type in node.arg_types
    ]
    return TreeNode(node=node, children=children, feature_name=None)


def random_individual(
    grammar: Grammar = DEFAULT_GRAMMAR,
    limits: GPLimits = DEFAULT_GP_LIMITS,
    rng: Optional[np.random.Generator] = None,
) -> Individual:
    """
    Construye un individuo aleatorio.

    La raíz es siempre de tipo POSITION (es la salida del sistema).
    """
    if rng is None:
        rng = np.random.default_rng()

    root = _random_tree_recursive(
        grammar=grammar,
        target_type=GPType.POSITION,
        remaining_depth=limits.max_depth,
        rng=rng,
    )

    ind = Individual(root=root)
    ind.validate()
    return ind