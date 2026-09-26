"""
Penalizaciones de complejidad.

Todos los valores devueltos son >= 0. El fitness final los RESTA.
"""

from __future__ import annotations

from dataclasses import dataclass

from gp.individual import Individual


@dataclass(frozen=True)
class ComplexityWeights:
    """Pesos de cada término de complejidad."""
    size: float = 0.001        # por nodo
    conditions: float = 0.05   # por comparador (GT/LT)
    features: float = 0.02     # por feature distinta


def complexity_penalty(
    ind: Individual,
    weights: ComplexityWeights = ComplexityWeights(),
) -> float:
    """
    Penalización total por complejidad.

    Términos:
    - size: número total de nodos
    - conditions: número de GT + LT
    - features: número de features distintas usadas

    Se normaliza por tamaño típico para que los pesos sean interpretables.
    """
    size_term = weights.size * ind.size
    cond_term = weights.conditions * ind.n_conditions()
    feat_term = weights.features * ind.n_features()
    return float(size_term + cond_term + feat_term)