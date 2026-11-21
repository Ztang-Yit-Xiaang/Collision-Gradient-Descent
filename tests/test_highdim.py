import csv
import gzip
import json

import numpy as np
import pytest

from collision_optim import CountedObjective, HybridConfig, HybridContact, Quadratic, Rosenbrock, State, run
from collision_optim import highdim
from collision_optim.highdim_objectives import (
    CoupledDoubleWell, CoupledRosenbrock, DiagonalEllipsoid, StructuredRotation,
)


def test_implicit_rotation_is_orthogonal_and_invertible():
    rotation = StructuredRotation(1000)
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(2, 1000))
    np.testing.assert_allclose(rotation.inverse(rotation.forward(x)), x, atol=1e-13)
    assert rotation.forward(x) @ rotation.forward(y) == pytest.approx(x @ y, abs=1e-11)


@pytest.mark.parametrize("constructor", [DiagonalEllipsoid, CoupledDoubleWell, CoupledRosenbrock])
def test_highdim_gradient_and_complete_drift_polynomial(constructor):
    objective = constructor(9)
    rng = np.random.default_rng(8)
    x = rng.normal(0, 0.3, 9)
    v = rng.normal(0, 0.05, 9)
    numeric = []
    eps = 1e-6
    for axis in np.eye(9):
        numeric.append((objective.value(x + eps*axis)-objective.value(x-eps*axis))/(2*eps))
    np.testing.assert_allclose(objective.gradient(x), numeric, rtol=2e-6, atol=2e-7)
    coefficients = objective.path_coefficients(x, v)
    for t in np.linspace(0, 1, 15):
        assert np.polynomial.polynomial.polyval(t, coefficients) == pytest.approx(objective.value(x+t*v), abs=1e-10)


def test_double_well_normalization_keeps_energy_and_force_norm_comparable():
    values, gradients, positions = [], [], []
    for d in [10, 1000, 100000]:
        objective = CoupledDoubleWell(d)
        x = objective.from_coordinates(np.full(d, 0.8))
        values.append(objective.value(x))
        gradients.append(np.linalg.norm(objective.gradient(x)))
        positions.append(np.linalg.norm(x))
    np.testing.assert_allclose(values, values[0], rtol=1e-12)
    np.testing.assert_allclose(gradients, gradients[0], rtol=1e-12)
    np.testing.assert_allclose(positions, positions[0], rtol=1e-12)


def test_neighbor_coupling_is_real_not_just_a_rotation():
    independent = CoupledDoubleWell(8, coupling=0)
    coupled = CoupledDoubleWell(8, coupling=0.2)
    x = coupled.from_coordinates(np.array([-1, 1]*4))
    assert coupled.value(x) > independent.value(x)
    assert np.linalg.norm(coupled.gradient(x)-independent.gradient(x)) > 0.1
    optimum = min(np.roots([1, 0, -1, 0.05]).real)
    at_minimum = coupled.from_coordinates(np.full(8, optimum))
    assert coupled.value(at_minimum) == pytest.approx(coupled.minimum, abs=1e-13)


def test_scaled_rosenbrock_matches_original_chain_definition():
    objective = CoupledRosenbrock(8)
    r = np.linspace(-1.1, 1.0, 8)
    x = objective.from_coordinates(r)
    original = Rosenbrock()
    assert objective.value(x) == pytest.approx(original.value(r)/(100*7))
    expected_gradient = objective.rotation.inverse(original.gradient(r)*np.sqrt(8)/(100*7))
    np.testing.assert_allclose(objective.gradient(x), expected_gradient, atol=1e-12)


def test_default_quadratic_does_not_allocate_a_dense_identity(monkeypatch):
    monkeypatch.setattr(np, "eye", lambda *args, **kwargs: pytest.fail("dense identity allocation"))
    x, v = np.ones(100000), np.ones(100000)*0.1
    objective = Quadratic()
    assert objective.value(x) == pytest.approx(50000)
    np.testing.assert_array_equal(objective.gradient(x), x)
    assert len(objective.path_coefficients(x, v)) == 3


def test_compact_storage_does_not_change_dynamics_or_oracle_cost():
    initial = State([0.7], [0.3], z=0.9, vz=-0.3)
    full = run(HybridContact(HybridConfig(h=0.02)), CountedObjective(Quadratic()), initial, 80)
    compact = run(HybridContact(HybridConfig(h=0.02)), CountedObjective(Quadratic()), initial, 80, store_states=False)
    np.testing.assert_array_equal(full.final_state.x, compact.final_state.x)
    np.testing.assert_array_equal(full.final_state.velocity, compact.final_state.velocity)
    assert full.counts == compact.counts
    assert full.best_value == compact.best_value
    assert "x" not in compact.rows[0] and "v" not in compact.rows[0]
    assert compact.contacts
    assert "point" not in compact.contacts[0]
    assert "horizontal_energy_transfer" in compact.contacts[0]


def test_highdim_preflight_exports_scalar_traces_and_all_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(highdim, "_plots", lambda *args, **kwargs: None)
    result = highdim.run_highdim(tmp_path, dimensions=[8, 16], seeds=1, steps=12)
    assert result["total_runs"] == 40
    assert result["maximum_dimension"] == 16
    assert result["checks"]["finite_outcomes"]
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["total_runs"] == 40
    first = next((tmp_path / "traces").glob("*.csv.gz"))
    with gzip.open(first, "rt") as stream:
        reader = csv.DictReader(stream)
        assert "x" not in reader.fieldnames and "v" not in reader.fieldnames
        assert "relative_best_gap" in reader.fieldnames


def test_initial_total_horizontal_kinetic_energy_does_not_grow_with_dimension():
    for name in highdim.CASES:
        energies = []
        for d in [10, 1000]:
            objective = highdim.make_case(name, d)
            x, v = highdim.initialize_case(name, objective, seed=0)
            energies.append(0.5*(v@v))
            assert np.count_nonzero(x) == d
        assert energies[0] == pytest.approx(energies[1], abs=1e-12)

