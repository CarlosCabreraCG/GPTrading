"""
Población de individuos.

Una Population es una lista de Individual con utilidades para:
- selección por torneo
- elitismo (preservar los mejores)
- estadísticas agregadas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from gp.individual import Individual, random_individual
from config.settings import DEFAULT_GP_LIMITS, GPLimits
from gp.grammar import DEFAULT_GRAMMAR, Grammar


@dataclass
class Population:
    """
    Conjunto de individuos.

    fitness_values: lista paralela a individuals. Se asigna al evaluar.
    """
    individuals: list[Individual] = field(default_factory=list)
    fitness_values: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)

    def add(self, ind: Individual, fitness: float | None = None) -> None:
        self.individuals.append(ind)
        self.fitness_values.append(fitness if fitness is not None else float("-inf"))

    def set_fitness(self, idx: int, fitness: float) -> None:
        self.fitness_values[idx] = fitness

    def best(self) -> tuple[Individual, float]:
        if not self.individuals:
            raise ValueError("Población vacía")
        idx = int(np.argmax(self.fitness_values))
        return self.individuals[idx], self.fitness_values[idx]

    def worst(self) -> tuple[Individual, float]:
        if not self.individuals:
            raise ValueError("Población vacía")
        idx = int(np.argmin(self.fitness_values))
        return self.individuals[idx], self.fitness_values[idx]

    def mean_fitness(self) -> float:
        if not self.individuals:
            return 0.0
        return float(np.mean(self.fitness_values))

    def std_fitness(self) -> float:
        if len(self.individuals) < 2:
            return 0.0
        return float(np.std(self.fitness_values))

    def sorted_by_fitness(self) -> list[tuple[Individual, float]]:
        paired = list(zip(self.individuals, self.fitness_values))
        return sorted(paired, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------

def random_population(
    size: int,
    grammar: Grammar = DEFAULT_GRAMMAR,
    limits: GPLimits = DEFAULT_GP_LIMITS,
    rng: np.random.Generator | None = None,
) -> Population:
    """Crea una población inicial aleatoria."""
    if rng is None:
        rng = np.random.default_rng()

    pop = Population()
    for _ in range(size):
        ind = random_individual(grammar=grammar, limits=limits, rng=rng)
        pop.add(ind, fitness=float("-inf"))
    return pop


# ---------------------------------------------------------------------------
# Selección
# ---------------------------------------------------------------------------

def tournament_select(
    pop: Population,
    tournament_size: int,
    rng: np.random.Generator,
) -> Individual:
    """
    Selección por torneo.

    Elige `tournament_size` individuos al azar y devuelve el de mayor fitness.
    """
    n = len(pop)
    if n == 0:
        raise ValueError("Población vacía")
    if tournament_size < 1:
        raise ValueError(f"tournament_size debe ser >= 1, recibido {tournament_size}")
    if tournament_size > n:
        tournament_size = n

    indices = rng.choice(n, size=tournament_size, replace=False)
    best_idx = max(indices, key=lambda i: pop.fitness_values[i])
    return pop.individuals[best_idx]


def select_elites(
    pop: Population,
    n_elites: int,
) -> list[Individual]:
    """
    Devuelve los n_elites mejores individuos (copia de la lista ordenada).
    """
    if n_elites <= 0:
        return []
    sorted_pairs = pop.sorted_by_fitness()
    return [ind for ind, _ in sorted_pairs[:n_elites]]