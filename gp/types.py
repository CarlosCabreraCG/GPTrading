"""
Tipos del GP.

Solo hay tres tipos conceptuales. Cada nodo declara su tipo de retorno,
y la gramática verifica que los hijos tengan tipos compatibles con el padre.
"""

from __future__ import annotations

from enum import Enum, auto


class GPType(Enum):
    """Tipos posibles en el árbol GP."""
    REAL = auto()
    BOOLEAN = auto()
    POSITION = auto()


#: Conjunto de tipos que consideramos "terminales" (hojas sin hijos).
LEAF_TYPES = frozenset({GPType.REAL, GPType.POSITION})