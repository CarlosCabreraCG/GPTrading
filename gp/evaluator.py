"""
Evaluador de árboles GP.

Dado un árbol (estructura de nodos) y un contexto de features, devuelve
el array de valores del tipo raíz.

El evaluador es recursivo. Para árboles pequeños (max_depth <= 6, max_nodes <= 40)
el coste de recursión es despreciable.

Convenciones:
- Los terminales de feature (REAL, func is None) leen del contexto por nombre.
- Los terminales POSITION devuelven un array constante del tamaño n.
- Los nodos internos evalúan sus hijos y aplican func.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gp.nodes import Node
from gp.types import GPType


@dataclass
class EvalContext:
    """
    Contexto de evaluación.

    features: dict {feature_name: np.ndarray} con arrays alineados por índice temporal.
    n: número de filas (debe coincidir con len de cada array de features).
    """
    features: dict[str, np.ndarray]
    n: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(f"n debe ser > 0, recibido {self.n}")
        for name, arr in self.features.items():
            if len(arr) != self.n:
                raise ValueError(
                    f"Feature '{name}' tiene longitud {len(arr)}, esperado {self.n}"
                )


# ---------------------------------------------------------------------------
# Estructura del árbol
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """
    Nodo concreto del árbol (instancia, no definición).

    node: definición (Node) del catálogo.
    children: lista de TreeNode hijos, en el orden de node.arg_types.
    feature_name: solo para terminales de feature (REAL, func is None).
    """
    node: Node
    children: list["TreeNode"]
    feature_name: str | None = None

    @property
    def arity(self) -> int:
        return len(self.children)

    def is_feature_terminal(self) -> bool:
        return self.node.func is None and self.feature_name is not None

    def size(self) -> int:
        """Número total de nodos en el subárbol (incluyendo este)."""
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        """Profundidad máxima del subárbol (hoja = 1)."""
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def count_nodes_by_name(self) -> dict[str, int]:
        """Cuenta ocurrencias de cada nombre de nodo en el subárbol."""
        counts: dict[str, int] = {}
        self._count_recursive(counts)
        return counts

    def _count_recursive(self, counts: dict[str, int]) -> None:
        counts[self.node.name] = counts.get(self.node.name, 0) + 1
        for c in self.children:
            c._count_recursive(counts)


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluate(tree: TreeNode, ctx: EvalContext) -> np.ndarray:
    """
    Evalúa el árbol sobre el contexto.

    Devuelve un array de dtype:
    - float64 si el tipo raíz es REAL
    - bool si el tipo raíz es BOOLEAN
    - int8 si el tipo raíz es POSITION
    """
    result = _eval_recursive(tree, ctx)
    return result


def _eval_recursive(tree: TreeNode, ctx: EvalContext) -> np.ndarray:
    # Caso 1: terminal de feature
    if tree.is_feature_terminal():
        name = tree.feature_name
        if name not in ctx.features:
            raise KeyError(
                f"Feature '{name}' no está en el contexto. "
                f"Disponibles: {list(ctx.features.keys())[:5]}..."
            )
        return ctx.features[name]

    # Caso 2: nodo sin hijos (terminal POSITION)
    if not tree.children:
        return tree.node.func(ctx.n)

    # Caso 3: nodo interno
    child_values = [_eval_recursive(c, ctx) for c in tree.children]
    return tree.node.func(*child_values)


# ---------------------------------------------------------------------------
# Verificación de tipos
# ---------------------------------------------------------------------------

def check_types(tree: TreeNode) -> None:
    """
    Verifica recursivamente que el árbol respeta la gramática tipada.

    Lanza ValueError si encuentra una violación.
    """
    _check_recursive(tree, path="root")


def _check_recursive(tree: TreeNode, path: str) -> None:
    node = tree.node

    # Verificar aridad
    if len(tree.children) != node.arity:
        raise ValueError(
            f"En {path}: nodo '{node.name}' espera {node.arity} hijos, "
            f"tiene {len(tree.children)}"
        )

    # Verificar que terminal de feature tiene feature_name
    if node.func is None:
        if node.arity != 0:
            raise ValueError(
                f"En {path}: nodo '{node.name}' tiene func=None pero arity={node.arity}"
            )
        if tree.feature_name is None:
            raise ValueError(
                f"En {path}: nodo terminal '{node.name}' sin feature_name"
            )
        return

    # Verificar que nodo no-terminal no tiene feature_name
    if tree.feature_name is not None:
        raise ValueError(
            f"En {path}: nodo '{node.name}' tiene feature_name pero func no es None"
        )

    # Verificar tipos de hijos
    for i, (child, expected_type) in enumerate(zip(tree.children, node.arg_types)):
        actual_type = child.node.return_type
        if actual_type != expected_type:
            raise ValueError(
                f"En {path}.child[{i}]: nodo '{node.name}' espera hijo de tipo "
                f"{expected_type.name}, recibido {actual_type.name} "
                f"(nodo '{child.node.name}')"
            )
        _check_recursive(child, path=f"{path}.child[{i}]")