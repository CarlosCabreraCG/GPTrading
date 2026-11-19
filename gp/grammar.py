"""
Gramática tipada del GP.

Dado un tipo objetivo, la gramática sabe qué nodos pueden usarse como raíz
de un subárbol de ese tipo. También conoce los terminales (features) de tipo REAL.

La gramática NO construye árboles. Solo declara las reglas de qué es válido.
La construcción viene en 3b (individual.py, population.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from features.registry import feature_names, ohlc_feature_names
from gp.nodes import (
    NODE_ABS,
    NODE_ADD,
    NODE_AND,
    NODE_DIV,
    NODE_GT,
    NODE_IF,
    NODE_INV,
    NODE_LOG,
    NODE_LT,
    NODE_MUL,
    NODE_NEG,
    NODE_NOT,
    NODE_OR,
    NODE_SQRT,
    NODE_SUB,
    POSITION_FLAT,
    POSITION_LONG,
    POSITION_SHORT,
    Node,
)
from gp.types import GPType


# ---------------------------------------------------------------------------
# Nodos terminales REAL: las features
# ---------------------------------------------------------------------------

def _make_feature_terminal(name: str) -> Node:
    """
    Crea un nodo terminal REAL que lee la feature `name` del contexto.

    El contexto es un dict {feature_name: np.ndarray}. La función de evaluación
    recibe n (número de filas) pero en realidad ignora n y devuelve el array
    de la feature. Para que la interfaz sea uniforme con los terminales POSITION,
    guardamos el contexto en un atributo especial.
    """
    # Nota: este terminal necesita acceso al contexto de features.
    # La firma func(n) no basta. Usaremos un wrapper que el evaluador manejará.
    return Node(
        name=name,
        return_type=GPType.REAL,
        arg_types=(),
        func=None,  # se resuelve en el evaluador vía feature_name
    )


#: Terminales de TODAS las features registradas (OHLC + cross-asset).
#: Una gramática concreta usa solo un subconjunto (ver build_grammar).
FEATURE_TERMINALS: dict[str, Node] = {
    name: _make_feature_terminal(name) for name in feature_names()
}


# ---------------------------------------------------------------------------
# Gramática
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Grammar:
    """
    Reglas de la gramática tipada.

    production[type] = lista de nodos que pueden usarse como raíz de un
    subárbol de ese tipo.
    """
    productions: dict[GPType, tuple[Node, ...]]

    def nodes_for(self, return_type: GPType) -> tuple[Node, ...]:
        return self.productions[return_type]

    def terminal_features(self) -> tuple[str, ...]:
        """Features que esta gramática puede usar como terminales REAL."""
        return tuple(
            n.name for n in self.productions[GPType.REAL] if n.name in FEATURE_TERMINALS
        )


def build_grammar(features: Iterable[str] | None = None) -> Grammar:
    """
    Construye una gramática con las features dadas como terminales REAL.

    features: nombres de features permitidas. Si es None, solo features OHLC
        (las que no requieren columnas cross-asset). Para usar cross-asset,
        pasar `available_features(df.columns)`.

    POSITION:
        - IF
        - LONG, SHORT, FLAT
    BOOLEAN:
        - AND, OR, NOT
        - GT, LT
    REAL:
        - features terminales
        - ADD, SUB, MUL, DIV
        - SQRT, LOG, ABS, NEG, INV
    """
    position_nodes = (NODE_IF, POSITION_LONG, POSITION_SHORT, POSITION_FLAT)
    boolean_nodes = (NODE_AND, NODE_OR, NODE_NOT, NODE_GT, NODE_LT)
    names = ohlc_feature_names() if features is None else list(features)
    unknown = [n for n in names if n not in FEATURE_TERMINALS]
    if unknown:
        raise ValueError(f"Features no registradas: {unknown}")
    if not names:
        raise ValueError("La gramática necesita al menos una feature")

    real_nodes = tuple(FEATURE_TERMINALS[n] for n in names) + (
        NODE_ADD, NODE_SUB, NODE_MUL, NODE_DIV,
        NODE_SQRT, NODE_LOG, NODE_ABS, NODE_NEG, NODE_INV,
    )

    return Grammar(
        productions={
            GPType.POSITION: position_nodes,
            GPType.BOOLEAN: boolean_nodes,
            GPType.REAL: real_nodes,
        }
    )


#: Gramática por defecto: solo features OHLC.
DEFAULT_GRAMMAR = build_grammar()