"""
Operadores genéticos: cruce y mutación.

Ambos respetan la gramática tipada y los límites de profundidad/nodos.

Cruce tipado: se eligen dos subárboles del MISMO tipo, uno en cada padre,
y se intercambian.

Mutación tipada: se elige un subárbol, se reemplaza por un subárbol aleatorio
del MISMO tipo, construido con el presupuesto de profundidad restante.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from config.settings import DEFAULT_GP_LIMITS, GPLimits
from gp.evaluator import TreeNode
from gp.grammar import DEFAULT_GRAMMAR, Grammar
from gp.individual import Individual, _random_tree_recursive
from gp.types import GPType


# ---------------------------------------------------------------------------
# Utilidades de navegación del árbol
# ---------------------------------------------------------------------------

def _collect_subtrees(
    tree: TreeNode,
    target_type: GPType,
    parent: Optional[TreeNode] = None,
    child_index: int = -1,
) -> list[tuple[TreeNode, Optional[TreeNode], int]]:
    """
    Devuelve una lista de (subárbol, padre, índice_en_padre) para todos los
    subárboles de `tree` cuyo tipo sea `target_type`.

    El subárbol raíz (tree) se incluye con parent=None e índice=-1.
    """
    out = []
    if tree.node.return_type == target_type:
        out.append((tree, parent, child_index))
    for i, child in enumerate(tree.children):
        out.extend(_collect_subtrees(child, target_type, tree, i))
    return out


def _replace_subtree(
    root: TreeNode,
    target: TreeNode,
    replacement: TreeNode,
) -> TreeNode:
    """
    Devuelve una copia de `root` donde el nodo `target` ha sido reemplazado
    por `replacement`.

    Compara por identidad de objeto (is), no por igualdad estructural.
    """
    if root is target:
        return copy.deepcopy(replacement)

    new_root = TreeNode(
        node=root.node,
        children=[],
        feature_name=root.feature_name,
    )
    for child in root.children:
        new_root.children.append(_replace_subtree(child, target, replacement))
    return new_root


# ---------------------------------------------------------------------------
# Cruce
# ---------------------------------------------------------------------------

def crossover(
    parent1: Individual,
    parent2: Individual,
    limits: GPLimits = DEFAULT_GP_LIMITS,
    rng: np.random.Generator | None = None,
) -> tuple[Individual, Individual]:
    """
    Cruce tipado entre dos padres.

    Estrategia:
    1. Elegir un tipo al azar entre los tipos presentes en ambos padres.
    2. Elegir un subárbol de ese tipo en cada padre.
    3. Intercambiarlos, generando dos hijos.
    4. Si algún hijo excede max_nodes, devolver copias de los padres sin cambio.
    """
    if rng is None:
        rng = np.random.default_rng()

    types_to_try = [GPType.POSITION, GPType.BOOLEAN, GPType.REAL]
    rng.shuffle(types_to_try)

    for target_type in types_to_try:
        subs1 = _collect_subtrees(parent1.root, target_type)
        subs2 = _collect_subtrees(parent2.root, target_type)
        if not subs1 or not subs2:
            continue

        sub1, _, _ = subs1[rng.integers(0, len(subs1))]
        sub2, _, _ = subs2[rng.integers(0, len(subs2))]

        # Hijo 1: parent1 con sub1 reemplazado por sub2
        child1_root = _replace_subtree(parent1.root, sub1, sub2)
        # Hijo 2: parent2 con sub2 reemplazado por sub1
        child2_root = _replace_subtree(parent2.root, sub2, sub1)

        child1 = Individual(root=child1_root)
        child2 = Individual(root=child2_root)

        # Verificar límites
        if child1.size > limits.max_nodes or child2.size > limits.max_nodes:
            continue

        try:
            child1.validate()
            child2.validate()
        except ValueError:
            continue

        return child1, child2

    # Fallback: copias de los padres
    return copy.deepcopy(parent1), copy.deepcopy(parent2)


# ---------------------------------------------------------------------------
# Mutación
# ---------------------------------------------------------------------------

def mutate(
    individual: Individual,
    grammar: Grammar = DEFAULT_GRAMMAR,
    limits: GPLimits = DEFAULT_GP_LIMITS,
    rng: np.random.Generator | None = None,
    mutation_rate: float = 0.1,
) -> Individual:
    """
    Mutación tipada.

    Con probabilidad `mutation_rate` por invocación, reemplaza un subárbol
    aleatorio por otro del mismo tipo. Si el resultado excede límites,
    devuelve el original sin cambio.
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() > mutation_rate:
        return copy.deepcopy(individual)

    types_to_try = [GPType.REAL, GPType.BOOLEAN, GPType.POSITION]
    rng.shuffle(types_to_try)

    for target_type in types_to_try:
        subs = _collect_subtrees(individual.root, target_type)
        if not subs:
            continue

        target_sub, _, _ = subs[rng.integers(0, len(subs))]

        # Presupuesto de profundidad: usamos max_depth completo (conservador).
        # Una versión más fina calcularía la profundidad restante real.
        new_sub = _random_tree_recursive(
            grammar=grammar,
            target_type=target_type,
            remaining_depth=limits.max_depth,
            rng=rng,
        )

        new_root = _replace_subtree(individual.root, target_sub, new_sub)
        new_ind = Individual(root=new_root)

        if new_ind.size > limits.max_nodes:
            continue

        try:
            new_ind.validate()
        except ValueError:
            continue

        return new_ind

    return copy.deepcopy(individual)