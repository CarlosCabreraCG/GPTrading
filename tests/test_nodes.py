"""
Tests de nodos, operadores protegidos y gramática.

Fase 3a: solo verifica correctitud matemática de los operadores y las
reglas de la gramática. No hay evolución todavía.
"""

from __future__ import annotations

import numpy as np
import pytest

from config.settings import EPSILON, MAX_ABS_VALUE
from gp import nodes as N
from gp.grammar import DEFAULT_GRAMMAR, FEATURE_TERMINALS
from gp.types import GPType


# ---------------------------------------------------------------------------
# Tests de operadores protegidos
# ---------------------------------------------------------------------------

def test_protected_division_normal():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 5.0])
    result = N.protected_division(a, b)
    np.testing.assert_allclose(result, [0.5, 0.5, 0.6])


def test_protected_division_by_zero_returns_zero():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 0.0, 5.0])
    result = N.protected_division(a, b)
    assert result[0] == pytest.approx(0.5)
    assert result[1] == 0.0
    assert result[2] == pytest.approx(0.6)


def test_protected_division_by_tiny_returns_zero():
    a = np.array([1.0])
    b = np.array([EPSILON / 10])
    result = N.protected_division(a, b)
    assert result[0] == 0.0


def test_protected_division_clamps_extreme():
    a = np.array([1e12])
    b = np.array([1e-3])
    result = N.protected_division(a, b)
    assert result[0] <= MAX_ABS_VALUE


def test_protected_sqrt_negative_returns_zero():
    a = np.array([4.0, -1.0, 9.0])
    result = N.protected_sqrt(a)
    assert result[0] == pytest.approx(2.0)
    assert result[1] == 0.0
    assert result[2] == pytest.approx(3.0)


def test_protected_log_nonpositive_returns_zero():
    a = np.array([1.0, 0.0, -1.0, np.e])
    result = N.protected_log(a)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == 0.0
    assert result[2] == 0.0
    assert result[3] == pytest.approx(1.0)


def test_protected_inverse_zero_returns_zero():
    a = np.array([2.0, 0.0, 4.0])
    result = N.protected_inverse(a)
    assert result[0] == pytest.approx(0.5)
    assert result[1] == 0.0
    assert result[2] == pytest.approx(0.25)


def test_safe_mul_handles_nan_and_inf():
    a = np.array([1e300, 0.0, 1.0])
    b = np.array([1e300, np.inf, 2.0])
    result = N.safe_mul(a, b)
    assert np.isfinite(result).all()
    assert (np.abs(result) <= MAX_ABS_VALUE).all()


def test_safe_add_clamps():
    a = np.array([MAX_ABS_VALUE, -MAX_ABS_VALUE])
    b = np.array([MAX_ABS_VALUE, -MAX_ABS_VALUE])
    result = N.safe_add(a, b)
    assert result[0] == MAX_ABS_VALUE
    assert result[1] == -MAX_ABS_VALUE


# ---------------------------------------------------------------------------
# Tests de nodos POSITION
# ---------------------------------------------------------------------------

def test_if_position_selects_branch():
    cond = np.array([True, False, True, False])
    then_val = np.array([1, 1, 1, 1], dtype=np.int8)
    else_val = np.array([-1, -1, -1, -1], dtype=np.int8)
    result = N.NODE_IF.func(cond, then_val, else_val)
    np.testing.assert_array_equal(result, [1, -1, 1, -1])


def test_position_terminal_long():
    result = N.POSITION_LONG.func(5)
    np.testing.assert_array_equal(result, [1, 1, 1, 1, 1])


def test_position_terminal_short():
    result = N.POSITION_SHORT.func(3)
    np.testing.assert_array_equal(result, [-1, -1, -1])


def test_position_terminal_flat():
    result = N.POSITION_FLAT.func(4)
    np.testing.assert_array_equal(result, [0, 0, 0, 0])


# ---------------------------------------------------------------------------
# Tests de operadores BOOLEAN
# ---------------------------------------------------------------------------

def test_boolean_and_or_not():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    np.testing.assert_array_equal(N.NODE_AND.func(a, b), [True, False, False, False])
    np.testing.assert_array_equal(N.NODE_OR.func(a, b), [True, True, True, False])
    np.testing.assert_array_equal(N.NODE_NOT.func(a), [False, False, True, True])


def test_gt_lt():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 2.0, 1.0])
    np.testing.assert_array_equal(N.NODE_GT.func(a, b), [False, False, True])
    np.testing.assert_array_equal(N.NODE_LT.func(a, b), [True, False, False])


# ---------------------------------------------------------------------------
# Tests de la gramática
# ---------------------------------------------------------------------------

def test_grammar_has_all_types():
    for t in (GPType.POSITION, GPType.BOOLEAN, GPType.REAL):
        assert len(DEFAULT_GRAMMAR.nodes_for(t)) > 0


def test_grammar_position_nodes():
    names = {n.name for n in DEFAULT_GRAMMAR.nodes_for(GPType.POSITION)}
    assert "IF" in names
    assert "LONG" in names
    assert "SHORT" in names
    assert "FLAT" in names


def test_grammar_boolean_nodes():
    names = {n.name for n in DEFAULT_GRAMMAR.nodes_for(GPType.BOOLEAN)}
    assert {"AND", "OR", "NOT", "GT", "LT"}.issubset(names)


def test_grammar_real_nodes_include_features():
    names = {n.name for n in DEFAULT_GRAMMAR.nodes_for(GPType.REAL)}
    # Algunas features clave deben estar
    assert "return_1" in names
    assert "rsi_14" in names
    assert "atr_14_norm" in names
    # Y operadores
    assert {"ADD", "SUB", "MUL", "DIV", "SQRT", "LOG", "ABS", "NEG", "INV"}.issubset(names)


def test_grammar_feature_terminals_match_registry():
    from features.registry import feature_names
    grammar_features = set(DEFAULT_GRAMMAR.terminal_features())
    registry_features = set(feature_names())
    assert grammar_features == registry_features


def test_all_nodes_have_consistent_arity():
    """Verifica que arity == len(arg_types) para todos los nodos."""
    all_nodes = []
    for t in (GPType.POSITION, GPType.BOOLEAN, GPType.REAL):
        all_nodes.extend(DEFAULT_GRAMMAR.nodes_for(t))
    for node in all_nodes:
        assert node.arity == len(node.arg_types), f"Nodo {node.name} inconsistente"


def test_feature_terminals_have_no_args():
    for name, node in FEATURE_TERMINALS.items():
        assert node.arity == 0
        assert node.return_type == GPType.REAL