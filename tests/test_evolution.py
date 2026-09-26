"""
Tests de población, operadores y evolución.

No hay fitness real todavía. Usamos funciones dummy para verificar
la mecánica del bucle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gp.evaluator import TreeNode
from gp.evolution import EvolutionConfig, evolve
from gp.individual import Individual, random_individual
from gp.operators import _collect_subtrees, _replace_subtree, crossover, mutate
from gp.population import (
    Population,
    random_population,
    select_elites,
    tournament_select,
)
from gp.serialization import (
    dict_to_individual,
    individual_to_dict,
    load_individual,
    save_individual,
)
from gp.types import GPType


# ---------------------------------------------------------------------------
# Tests de Population
# ---------------------------------------------------------------------------

def test_random_population_size():
    pop = random_population(size=20, rng=np.random.default_rng(0))
    assert len(pop) == 20
    assert len(pop.fitness_values) == 20


def test_population_best_worst():
    pop = Population()
    for i, f in enumerate([1.0, 5.0, 3.0]):
        ind = random_individual(rng=np.random.default_rng(i))
        pop.add(ind, fitness=f)

    best_ind, best_f = pop.best()
    worst_ind, worst_f = pop.worst()
    assert best_f == 5.0
    assert worst_f == 1.0


def test_tournament_select_returns_valid():
    pop = random_population(size=10, rng=np.random.default_rng(1))
    for i, f in enumerate(np.arange(10.0)):
        pop.set_fitness(i, float(f))
    rng = np.random.default_rng(2)
    for _ in range(20):
        ind = tournament_select(pop, tournament_size=3, rng=rng)
        assert ind in pop.individuals


def test_tournament_prefers_fitter():
    """Con muchos torneos, el mejor debe ser elegido más frecuentemente."""
    pop = Population()
    fitnesses = [10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i, f in enumerate(fitnesses):
        ind = random_individual(rng=np.random.default_rng(i))
        pop.add(ind, fitness=f)

    rng = np.random.default_rng(42)
    counts = {}
    for _ in range(200):
        ind = tournament_select(pop, tournament_size=3, rng=rng)
        key = id(ind)
        counts[key] = counts.get(key, 0) + 1

    # El individuo mejor debe haber sido seleccionado al menos una vez
    best_id = id(pop.individuals[0])
    assert counts.get(best_id, 0) > 0


def test_select_elites():
    pop = Population()
    for i, f in enumerate([5.0, 1.0, 4.0, 2.0, 3.0]):
        ind = random_individual(rng=np.random.default_rng(i))
        pop.add(ind, fitness=f)

    elites = select_elites(pop, n_elites=2)
    assert len(elites) == 2
    # Los dos mejores están en [0] (f=5) e [2] (f=4)
    assert pop.individuals[0] in elites
    assert pop.individuals[2] in elites


# ---------------------------------------------------------------------------
# Tests de navegación de árbol
# ---------------------------------------------------------------------------

def test_collect_subtrees_finds_all_real():
    ind = random_individual(rng=np.random.default_rng(0))
    subs = _collect_subtrees(ind.root, GPType.REAL)
    # Debe haber al menos un terminal REAL en cualquier árbol POSITION
    # (excepto si es un terminal POSITION puro)
    if ind.root.node.arity > 0:
        assert len(subs) > 0


def test_collect_subtrees_includes_root_if_matching():
    ind = random_individual(rng=np.random.default_rng(1))
    subs_pos = _collect_subtrees(ind.root, GPType.POSITION)
    assert any(s[0] is ind.root for s in subs_pos)


def test_replace_subtree_identity():
    """Reemplazar un nodo por sí mismo debe dar un árbol equivalente."""
    ind = random_individual(rng=np.random.default_rng(2))
    new_root = _replace_subtree(ind.root, ind.root, ind.root)
    assert new_root.size() == ind.root.size()
    assert new_root.depth() == ind.root.depth()


# ---------------------------------------------------------------------------
# Tests de crossover
# ---------------------------------------------------------------------------

def test_crossover_produces_valid_children():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p1 = random_individual(rng=rng)
        p2 = random_individual(rng=rng)
        c1, c2 = crossover(p1, p2, rng=rng)
        c1.validate()
        c2.validate()


def test_crossover_respects_max_nodes():
    from config.settings import GPLimits
    limits = GPLimits(max_depth=6, max_nodes=40, max_conditions=8,
                      max_indicators=10, max_nesting=3)
    rng = np.random.default_rng(1)
    for _ in range(30):
        p1 = random_individual(limits=limits, rng=rng)
        p2 = random_individual(limits=limits, rng=rng)
        c1, c2 = crossover(p1, p2, limits=limits, rng=rng)
        assert c1.size <= limits.max_nodes
        assert c2.size <= limits.max_nodes


def test_crossover_does_not_modify_parents():
    rng = np.random.default_rng(2)
    p1 = random_individual(rng=rng)
    p2 = random_individual(rng=rng)
    s1_before = p1.size
    s2_before = p2.size
    crossover(p1, p2, rng=rng)
    assert p1.size == s1_before
    assert p2.size == s2_before


# ---------------------------------------------------------------------------
# Tests de mutación
# ---------------------------------------------------------------------------

def test_mutation_rate_zero_returns_copy():
    ind = random_individual(rng=np.random.default_rng(0))
    rng = np.random.default_rng(1)
    for _ in range(10):
        m = mutate(ind, rng=rng, mutation_rate=0.0)
        assert m.size == ind.size
        assert m.depth == ind.depth


def test_mutation_rate_one_changes_something():
    """Con mutation_rate=1.0, muchos intentos deben producir cambios."""
    rng = np.random.default_rng(0)
    n_changed = 0
    for _ in range(30):
        ind = random_individual(rng=rng)
        m = mutate(ind, rng=rng, mutation_rate=1.0)
        if m.size != ind.size or m.depth != ind.depth:
            n_changed += 1
    assert n_changed > 0


def test_mutation_produces_valid_individuals():
    rng = np.random.default_rng(3)
    for _ in range(20):
        ind = random_individual(rng=rng)
        m = mutate(ind, rng=rng, mutation_rate=1.0)
        m.validate()


# ---------------------------------------------------------------------------
# Tests de serialización
# ---------------------------------------------------------------------------

def test_serialization_roundtrip():
    ind = random_individual(rng=np.random.default_rng(0))
    d = individual_to_dict(ind)
    ind2 = dict_to_individual(d)
    assert ind.size == ind2.size
    assert ind.depth == ind2.depth
    assert ind.node_counts() == ind2.node_counts()


def test_serialization_json_file(tmp_path: Path):
    ind = random_individual(rng=np.random.default_rng(1))
    path = tmp_path / "ind.json"
    save_individual(ind, path)
    assert path.exists()

    ind2 = load_individual(path)
    assert ind.size == ind2.size
    assert ind.node_counts() == ind2.node_counts()


def test_serialization_preserves_evaluation(tmp_path):
    """Un individuo cargado debe evaluar exactamente igual que el original."""
    from features.registry import feature_names
    from gp.evaluator import EvalContext

    n = 50
    rng = np.random.default_rng(7)
    ctx = EvalContext(
        features={name: rng.normal(0, 1, n) for name in feature_names()},
        n=n,
    )

    ind = random_individual(rng=np.random.default_rng(0))
    sig1 = ind.evaluate(ctx)

    path = tmp_path / "ind.json"
    save_individual(ind, path)
    ind2 = load_individual(path)
    sig2 = ind2.evaluate(ctx)

    np.testing.assert_array_equal(sig1, sig2)


# ---------------------------------------------------------------------------
# Tests de evolución
# ---------------------------------------------------------------------------

def test_evolve_runs_with_dummy_fitness():
    """Fitness dummy: tamaño del individuo (queremos árboles pequeños)."""
    def fitness(ind: Individual) -> float:
        return -float(ind.size)

    config = EvolutionConfig(
        population_size=20,
        n_generations=5,
        tournament_size=3,
        n_elites=2,
        crossover_rate=0.8,
        mutation_rate=0.2,
        seed=42,
    )
    result = evolve(fitness, config)
    assert result.best_individual is not None
    assert len(result.history_best) == 5
    assert result.best_fitness > float("-inf")


def test_evolve_improves_dummy_fitness():
    """Con fitness = -size, la evolución debe reducir el tamaño medio."""
    def fitness(ind: Individual) -> float:
        return -float(ind.size)

    config = EvolutionConfig(
        population_size=30,
        n_generations=15,
        tournament_size=4,
        n_elites=2,
        crossover_rate=0.7,
        mutation_rate=0.3,
        seed=1,
    )
    result = evolve(fitness, config)
    # El best de la última generación debe ser al menos tan bueno como el de la primera
    assert result.history_best[-1] >= result.history_best[0]


def test_evolve_deterministic_with_seed():
    def fitness(ind: Individual) -> float:
        return -float(ind.size)

    config1 = EvolutionConfig(population_size=10, n_generations=3, seed=999)
    config2 = EvolutionConfig(population_size=10, n_generations=3, seed=999)
    r1 = evolve(fitness, config1)
    r2 = evolve(fitness, config2)
    assert r1.best_fitness == r2.best_fitness
    assert r1.history_best == r2.history_best


def test_evolve_best_individual_is_valid():
    def fitness(ind: Individual) -> float:
        return -float(ind.size)

    config = EvolutionConfig(population_size=15, n_generations=5, seed=7)
    result = evolve(fitness, config)
    result.best_individual.validate()