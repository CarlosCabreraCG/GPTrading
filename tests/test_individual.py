"""
Tests de Individual y del evaluador.

Fase 3b-1: construcción aleatoria, evaluación, y verificación de tipos.
No hay evolución todavía.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gp.evaluator import EvalContext, TreeNode, check_types, evaluate
from gp.individual import Individual, random_individual
from gp.nodes import (
    NODE_AND,
    NODE_GT,
    NODE_IF,
    POSITION_FLAT,
    POSITION_LONG,
    POSITION_SHORT,
)
from gp.types import GPType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_context() -> EvalContext:
    """Contexto con dos features sintéticas."""
    n = 10
    rng = np.random.default_rng(42)
    return EvalContext(
        features={
            "return_1": rng.normal(0, 0.001, n),
            "rsi_14": rng.uniform(20, 80, n),
        },
        n=n,
    )


@pytest.fixture
def real_context() -> EvalContext:
    """Contexto con TODAS las features del registry, con datos sintéticos."""
    from features.registry import feature_names
    n = 100
    rng = np.random.default_rng(0)
    features = {name: rng.normal(0, 1, n) for name in feature_names()}
    return EvalContext(features=features, n=n)


# ---------------------------------------------------------------------------
# Tests del evaluador
# ---------------------------------------------------------------------------

def test_evaluate_position_terminal_long(simple_context):
    tree = TreeNode(node=POSITION_LONG, children=[], feature_name=None)
    result = evaluate(tree, simple_context)
    np.testing.assert_array_equal(result, [1] * simple_context.n)


def test_evaluate_position_terminal_short(simple_context):
    tree = TreeNode(node=POSITION_SHORT, children=[], feature_name=None)
    result = evaluate(tree, simple_context)
    np.testing.assert_array_equal(result, [-1] * simple_context.n)


def test_evaluate_position_terminal_flat(simple_context):
    tree = TreeNode(node=POSITION_FLAT, children=[], feature_name=None)
    result = evaluate(tree, simple_context)
    np.testing.assert_array_equal(result, [0] * simple_context.n)


def test_evaluate_feature_terminal(simple_context):
    from gp.grammar import FEATURE_TERMINALS
    node = FEATURE_TERMINALS["return_1"]
    tree = TreeNode(node=node, children=[], feature_name="return_1")
    result = evaluate(tree, simple_context)
    np.testing.assert_array_equal(result, simple_context.features["return_1"])


def test_evaluate_if_simple(simple_context):
    """
    IF(GT(return_1, 0), LONG, FLAT)
    """
    from gp.grammar import FEATURE_TERMINALS

    ret_node = FEATURE_TERMINALS["return_1"]

    # Construir: GT(return_1, return_1)  -> siempre False
    # Pero queremos GT(return_1, 0). No tenemos terminal constante.
    # Usamos GT(return_1, NEG(return_1)) = GT(x, -x) = x > -x = x > 0 para x != 0.
    # Es una forma de construir "es positivo" sin constante.
    from gp.nodes import NODE_NEG

    neg_tree = TreeNode(
        node=NODE_NEG,
        children=[TreeNode(node=ret_node, children=[], feature_name="return_1")],
        feature_name=None,
    )
    gt_tree = TreeNode(
        node=NODE_GT,
        children=[
            TreeNode(node=ret_node, children=[], feature_name="return_1"),
            neg_tree,
        ],
        feature_name=None,
    )
    if_tree = TreeNode(
        node=NODE_IF,
        children=[
            gt_tree,
            TreeNode(node=POSITION_LONG, children=[], feature_name=None),
            TreeNode(node=POSITION_FLAT, children=[], feature_name=None),
        ],
        feature_name=None,
    )

    result = evaluate(if_tree, simple_context)
    ret_vals = simple_context.features["return_1"]
    expected = np.where(ret_vals > -ret_vals, 1, 0).astype(np.int8)
    np.testing.assert_array_equal(result, expected)


def test_evaluate_missing_feature_raises(simple_context):
    from gp.grammar import FEATURE_TERMINALS
    node = FEATURE_TERMINALS["atr_14_norm"]  # no está en simple_context
    tree = TreeNode(node=node, children=[], feature_name="atr_14_norm")
    with pytest.raises(KeyError):
        evaluate(tree, simple_context)


# ---------------------------------------------------------------------------
# Tests de check_types
# ---------------------------------------------------------------------------

def test_check_types_ok_for_terminal():
    tree = TreeNode(node=POSITION_LONG, children=[], feature_name=None)
    check_types(tree)  # no debe lanzar


def test_check_types_detects_wrong_child_type():
    """IF espera (BOOLEAN, POSITION, POSITION). Si le damos LONG como condición, falla."""
    tree = TreeNode(
        node=NODE_IF,
        children=[
            TreeNode(node=POSITION_LONG, children=[], feature_name=None),  # mal: POSITION
            TreeNode(node=POSITION_LONG, children=[], feature_name=None),
            TreeNode(node=POSITION_FLAT, children=[], feature_name=None),
        ],
        feature_name=None,
    )
    with pytest.raises(ValueError, match="espera hijo de tipo BOOLEAN"):
        check_types(tree)


def test_check_types_detects_wrong_arity():
    """IF necesita 3 hijos. Con 2 debe fallar."""
    tree = TreeNode(
        node=NODE_IF,
        children=[
            TreeNode(node=POSITION_LONG, children=[], feature_name=None),
            TreeNode(node=POSITION_FLAT, children=[], feature_name=None),
        ],
        feature_name=None,
    )
    with pytest.raises(ValueError, match="espera 3 hijos"):
        check_types(tree)


def test_check_types_detects_feature_without_name():
    from gp.grammar import FEATURE_TERMINALS
    node = FEATURE_TERMINALS["return_1"]
    tree = TreeNode(node=node, children=[], feature_name=None)
    with pytest.raises(ValueError, match="sin feature_name"):
        check_types(tree)


# ---------------------------------------------------------------------------
# Tests de Individual
# ---------------------------------------------------------------------------

def test_random_individual_validates():
    rng = np.random.default_rng(42)
    for _ in range(20):
        ind = random_individual(rng=rng)
        ind.validate()  # no debe lanzar


def test_random_individual_root_is_position():
    rng = np.random.default_rng(1)
    ind = random_individual(rng=rng)
    assert ind.root.node.return_type == GPType.POSITION


def test_random_individual_respects_max_depth():
    rng = np.random.default_rng(7)
    for _ in range(20):
        ind = random_individual(rng=rng)
        assert ind.depth <= 6, f"Depth {ind.depth} excede max_depth=6"


def test_random_individual_evaluates(real_context):
    rng = np.random.default_rng(123)
    for _ in range(10):
        ind = random_individual(rng=rng)
        result = ind.evaluate(real_context)
        assert len(result) == real_context.n
        assert result.dtype == np.int8
        # Solo valores válidos
        assert set(np.unique(result)).issubset({-1, 0, 1})


def test_random_individual_deterministic_with_seed():
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)
    ind1 = random_individual(rng=rng1)
    ind2 = random_individual(rng=rng2)
    assert ind1.size == ind2.size
    assert ind1.depth == ind2.depth
    assert ind1.node_counts() == ind2.node_counts()


def test_individual_size_and_depth():
    tree = TreeNode(node=POSITION_LONG, children=[], feature_name=None)
    ind = Individual(root=tree)
    assert ind.size == 1
    assert ind.depth == 1


def test_individual_n_conditions():
    """Un árbol sin comparadores tiene 0 condiciones."""
    tree = TreeNode(node=POSITION_LONG, children=[], feature_name=None)
    ind = Individual(root=tree)
    assert ind.n_conditions() == 0


def test_random_individual_diversity():
    """20 individuos aleatorios no deben ser todos idénticos."""
    rng = np.random.default_rng(5)
    individuals = [random_individual(rng=rng) for _ in range(20)]
    signatures = {ind.size for ind in individuals}
    # Al menos 2 tamaños distintos
    assert len(signatures) > 1, "Los individuos aleatorios son demasiado homogéneos"


def test_context_validates_lengths():
    with pytest.raises(ValueError, match="longitud"):
        EvalContext(features={"a": np.zeros(5)}, n=10)


def test_context_rejects_zero_n():
    with pytest.raises(ValueError, match="n debe ser"):
        EvalContext(features={}, n=0)