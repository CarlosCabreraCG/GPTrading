"""
Nodos del árbol GP.

Un nodo tiene:
- name: identificador único
- return_type: GPType
- arg_types: tupla de GPType que espera como hijos
- func: función que evalúa el nodo dados los valores de sus hijos
- arity: número de hijos (= len(arg_types))

Los nodos terminales (features, LONG/SHORT/FLAT) tienen arity=0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from config.settings import EPSILON, MAX_ABS_VALUE
from gp.types import GPType


# ---------------------------------------------------------------------------
# Operadores protegidos
# ---------------------------------------------------------------------------

def protected_division(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """División protegida: si |b| < EPSILON, devuelve 0. Clamp a ±MAX_ABS_VALUE."""
    safe_b = np.where(np.abs(b) < EPSILON, np.nan, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / safe_b
    result = np.where(np.abs(b) < EPSILON, 0.0, result)
    result = np.nan_to_num(result, nan=0.0, posinf=MAX_ABS_VALUE, neginf=-MAX_ABS_VALUE)
    return np.clip(result, -MAX_ABS_VALUE, MAX_ABS_VALUE)


def protected_sqrt(a: np.ndarray) -> np.ndarray:
    """Raíz cuadrada protegida: solo para a >= 0. Para a < 0, devuelve 0."""
    result = np.where(a >= 0, np.sqrt(np.maximum(a, 0.0)), 0.0)
    return np.clip(result, -MAX_ABS_VALUE, MAX_ABS_VALUE)


def protected_log(a: np.ndarray) -> np.ndarray:
    """Logaritmo protegido: solo para a > 0. Para a <= 0, devuelve 0."""
    result = np.where(a > 0, np.log(np.maximum(a, EPSILON)), 0.0)
    result = np.nan_to_num(result, nan=0.0, posinf=MAX_ABS_VALUE, neginf=-MAX_ABS_VALUE)
    return np.clip(result, -MAX_ABS_VALUE, MAX_ABS_VALUE)


def protected_inverse(a: np.ndarray) -> np.ndarray:
    """Inverso protegido: si |a| < EPSILON, devuelve 0."""
    return protected_division(np.ones_like(a), a)


def safe_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = a + b
    return np.clip(result, -MAX_ABS_VALUE, MAX_ABS_VALUE)


def safe_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = a - b
    return np.clip(result, -MAX_ABS_VALUE, MAX_ABS_VALUE)


def safe_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = a * b
    result = np.nan_to_num(result, nan=0.0, posinf=MAX_ABS_VALUE, neginf=-MAX_ABS_VALUE)
    return np.clip(result, -MAX_ABS_VALUE, MAX_ABS_VALUE)


def safe_abs(a: np.ndarray) -> np.ndarray:
    return np.clip(np.abs(a), 0.0, MAX_ABS_VALUE)


def safe_neg(a: np.ndarray) -> np.ndarray:
    return np.clip(-a, -MAX_ABS_VALUE, MAX_ABS_VALUE)


# ---------------------------------------------------------------------------
# Nodo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Node:
    """
    Definición de un nodo del árbol GP.

    name: identificador único (usado en serialización y en el registry).
    return_type: tipo que devuelve este nodo.
    arg_types: tipos esperados de los hijos.
    func: función que evalúa el nodo. Para terminales, recibe solo el contexto.
    """
    name: str
    return_type: GPType
    arg_types: tuple[GPType, ...]
    func: Callable

    @property
    def arity(self) -> int:
        return len(self.arg_types)


# ---------------------------------------------------------------------------
# Terminales POSITION
# ---------------------------------------------------------------------------

# Los terminales POSITION son símbolos atómicos. Su evaluación devuelve
# un vector constante del valor correspondiente.
# La función recibe el número de filas (n) y devuelve un array de ese tamaño.

def _make_position_terminal(value: int):
    def fn(n: int) -> np.ndarray:
        return np.full(n, value, dtype=np.int8)
    return fn


POSITION_LONG = Node(
    name="LONG",
    return_type=GPType.POSITION,
    arg_types=(),
    func=_make_position_terminal(1),
)

POSITION_SHORT = Node(
    name="SHORT",
    return_type=GPType.POSITION,
    arg_types=(),
    func=_make_position_terminal(-1),
)

POSITION_FLAT = Node(
    name="FLAT",
    return_type=GPType.POSITION,
    arg_types=(),
    func=_make_position_terminal(0),
)


# ---------------------------------------------------------------------------
# Nodo POSITION: IF(cond, then, else)
# ---------------------------------------------------------------------------

def _if_position(cond: np.ndarray, then_val: np.ndarray, else_val: np.ndarray) -> np.ndarray:
    """Selecciona then_val donde cond es True, else_val donde es False."""
    return np.where(cond, then_val, else_val).astype(np.int8)


NODE_IF = Node(
    name="IF",
    return_type=GPType.POSITION,
    arg_types=(GPType.BOOLEAN, GPType.POSITION, GPType.POSITION),
    func=_if_position,
)


# ---------------------------------------------------------------------------
# Operadores BOOLEAN
# ---------------------------------------------------------------------------

NODE_AND = Node(
    name="AND",
    return_type=GPType.BOOLEAN,
    arg_types=(GPType.BOOLEAN, GPType.BOOLEAN),
    func=lambda a, b: np.logical_and(a, b),
)

NODE_OR = Node(
    name="OR",
    return_type=GPType.BOOLEAN,
    arg_types=(GPType.BOOLEAN, GPType.BOOLEAN),
    func=lambda a, b: np.logical_or(a, b),
)

NODE_NOT = Node(
    name="NOT",
    return_type=GPType.BOOLEAN,
    arg_types=(GPType.BOOLEAN,),
    func=lambda a: np.logical_not(a),
)

NODE_GT = Node(
    name="GT",
    return_type=GPType.BOOLEAN,
    arg_types=(GPType.REAL, GPType.REAL),
    func=lambda a, b: a > b,
)

NODE_LT = Node(
    name="LT",
    return_type=GPType.BOOLEAN,
    arg_types=(GPType.REAL, GPType.REAL),
    func=lambda a, b: a < b,
)


# ---------------------------------------------------------------------------
# Operadores REAL
# ---------------------------------------------------------------------------

NODE_ADD = Node(
    name="ADD",
    return_type=GPType.REAL,
    arg_types=(GPType.REAL, GPType.REAL),
    func=safe_add,
)

NODE_SUB = Node(
    name="SUB",
    return_type=GPType.REAL,
    arg_types=(GPType.REAL, GPType.REAL),
    func=safe_sub,
)

NODE_MUL = Node(
    name="MUL",
    return_type=GPType.REAL,
    arg_types=(GPType.REAL, GPType.REAL),
    func=safe_mul,
)

NODE_DIV = Node(
    name="DIV",
    return_type=GPType.REAL,
    arg_types=(GPType.REAL, GPType.REAL),
    func=protected_division,
)

NODE_SQRT = Node(
    name="SQRT",
    return_type=GPType.REAL,
    arg_types=(GPType.REAL,),
    func=protected_sqrt,
)

NODE_LOG = Node(
    name="LOG",
    return_type=GPType.REAL,
    arg_types=(GPType.REAL,),
    func=protected_log,
)

NODE_ABS = Node(
    name="ABS",
    return_type=GPType.REAL,
    arg_types=(GPType.REAL,),
    func=safe_abs,
)

NODE_NEG = Node(
    name="NEG",
    return_type=GPType.REAL,
    arg_types=(GPType.REAL,),
    func=safe_neg,
)

NODE_INV = Node(
    name="INV",
    return_type=GPType.REAL,
    arg_types=(GPType.REAL,),
    func=protected_inverse,
)