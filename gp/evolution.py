"""
Bucle evolutivo.

Interfaz de fitness: un callable que recibe un Individual y devuelve un float
(mayor = mejor). En Fase 4 se enchufará el fitness real con backtester.
En esta fase se puede usar cualquier función dummy para verificar el bucle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from config.settings import DEFAULT_GP_LIMITS, GPLimits
from gp.grammar import DEFAULT_GRAMMAR, Grammar
from gp.individual import Individual, random_individual
from gp.operators import crossover, mutate
from gp.population import Population, random_population, select_elites, tournament_select


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class EvolutionConfig:
    population_size: int = 100
    n_generations: int = 20
    tournament_size: int = 5
    n_elites: int = 2
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    limits: GPLimits = field(default_factory=lambda: DEFAULT_GP_LIMITS)
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

@dataclass
class EvolutionResult:
    best_individual: Individual
    best_fitness: float
    history_best: list[float] = field(default_factory=list)
    history_mean: list[float] = field(default_factory=list)
    history_std: list[float] = field(default_factory=list)
    generations: int = 0


# ---------------------------------------------------------------------------
# Bucle evolutivo
# ---------------------------------------------------------------------------

def evolve(
    fitness_fn: Callable[[Individual], float],
    config: EvolutionConfig,
    grammar: Grammar = DEFAULT_GRAMMAR,
) -> EvolutionResult:
    """
    Ejecuta el bucle evolutivo.

    Pasos por generación:
    1. Evaluar el fitness de cada individuo.
    2. Registrar estadísticas.
    3. Preservar élites.
    4. Generar nueva población por torneo + cruce + mutación.
    """
    rng = np.random.default_rng(config.seed)

    # Población inicial
    pop = random_population(
        size=config.population_size,
        grammar=grammar,
        limits=config.limits,
        rng=rng,
    )

    history_best: list[float] = []
    history_mean: list[float] = []
    history_std: list[float] = []

    best_ind: Individual | None = None
    best_fit: float = float("-inf")

    for gen in range(config.n_generations):
        # 1. Evaluar
        for i, ind in enumerate(pop.individuals):
            f = fitness_fn(ind)
            pop.set_fitness(i, f)

        # Filtramos -inf 
        finite_fits = [f for f in pop.fitness_values if np.isfinite(f)]

        # 2. Estadísticas
        gen_best = max(pop.fitness_values)
        gen_mean = float(np.mean(finite_fits)) if finite_fits else float("-inf")
        gen_std = float(np.std(finite_fits)) if len(finite_fits) > 1 else 0.0
        history_best.append(gen_best)
        history_mean.append(gen_mean)
        history_std.append(gen_std)

        if gen_best > best_fit:
            best_fit = gen_best
            best_ind = pop.best()[0]

        # Última generación: no reproducir
        if gen == config.n_generations - 1:
            break

        # 3. Élites
        elites = select_elites(pop, config.n_elites)

        # 4. Nueva población
        new_pop = Population()
        for elite in elites:
            new_pop.add(elite, fitness=float("-inf"))

        while len(new_pop) < config.population_size:
            parent1 = tournament_select(pop, config.tournament_size, rng)
            parent2 = tournament_select(pop, config.tournament_size, rng)

            if rng.random() < config.crossover_rate:
                child1, child2 = crossover(
                    parent1, parent2, limits=config.limits, rng=rng
                )
            else:
                import copy
                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)

            child1 = mutate(
                child1, grammar=grammar, limits=config.limits,
                rng=rng, mutation_rate=config.mutation_rate,
            )
            child2 = mutate(
                child2, grammar=grammar, limits=config.limits,
                rng=rng, mutation_rate=config.mutation_rate,
            )

            new_pop.add(child1, fitness=float("-inf"))
            if len(new_pop) < config.population_size:
                new_pop.add(child2, fitness=float("-inf"))

        pop = new_pop

    if best_ind is None:
        # Ningún individuo superó el fitness. Devolvemos el primero de la
        # población inicial con best_fitness=-inf. Esto permite que el WFO
        # registre el fold como fallido y siga, en lugar de abortar.
        fallback = pop.individuals[0] if pop.individuals else None
        if fallback is None:
            raise RuntimeError(
                "La evolución no produjo ningún individuo. Población vacía."
            )
        return EvolutionResult(
            best_individual=fallback,
            best_fitness=float("-inf"),
            history_best=history_best,
            history_mean=history_mean,
            history_std=history_std,
            generations=config.n_generations,
        )

    return EvolutionResult(
        best_individual=best_ind,
        best_fitness=best_fit,
        history_best=history_best,
        history_mean=history_mean,
        history_std=history_std,
        generations=config.n_generations,
    )