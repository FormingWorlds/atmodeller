#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Tests for the EquilibriumModel solve error contract.

These tests cover the multistart non-convergence guard in
:meth:`atmodeller.classes.EquilibriumModel.solve`. When no attempt in the batch converges (the
oxidising corner of the composition grid can exhaust the solver step budget for every attempt),
solve must raise a RuntimeError that names the batch size and step budget rather than publishing an
empty output. The tests also cover the step-summary log branch that reports "n/a" when every
attempt exhausts the step budget, so the summary never coerces an infinite step count to an integer.

The batch solver itself is replaced so the error path is exercised without running the nonlinear
solve. The replacement returns a genuine ``MultiAttemptSolution`` wrapping a hand-built optimistix
solution with a non-successful result, so it satisfies the type contract solve annotates while
reporting an all-failed batch.
"""

import logging
from typing import cast

import jax.numpy as jnp
import optimistix as optx
import pytest
from jaxmod.solvers import MultiAttemptSolution
from optimistix import RESULTS

from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, Planet, SpeciesNetwork
from atmodeller.output import Output
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

# max_steps far below this sentinel, so every attempt reads as having exhausted the step budget.
_STEPS_EXHAUSTED: int = 10**6
# max_steps far above this sentinel, so every attempt reads as having finished under the budget.
_STEPS_FINITE: int = 7


def _all_failed_solution(batch: int, num_steps: int) -> MultiAttemptSolution:
    """Builds an all-failed MultiAttemptSolution for a batch of ``batch`` attempts.

    The wrapped optimistix solution carries a non-successful ``nonlinear_max_steps_reached`` result,
    which is the failure the multistart cliff produces, so ``solver_success`` is False for every
    attempt. ``num_steps`` is reported uniformly so a caller can steer the step-summary log branch.

    Parameters
    ----------
    batch
        Number of attempts in the batch; sets the leading solution dimension.
    num_steps
        Uniform step count reported for every attempt.
    """
    solution: optx.Solution = optx.Solution(
        value=jnp.zeros((batch, 1)),
        result=RESULTS.nonlinear_max_steps_reached,
        aux=None,
        stats={"num_steps": jnp.full((batch,), num_steps, dtype=jnp.int32)},
        state=None,
    )
    return MultiAttemptSolution(solution)


def _single_species_model() -> EquilibriumModel:
    """Builds a one-species (H2O) equilibrium model exposing the solve entry point."""
    species: SpeciesNetwork = SpeciesNetwork((ChemicalSpecies.create_gas("H2O"),))
    return EquilibriumModel(species)


def _mass_constraints() -> dict:
    """Two-ocean hydrogen budget, enough to make Parameters.create well posed."""
    return {"H": earth_oceans_to_hydrogen_mass(2)}


def _install_failed_solver(model: EquilibriumModel, *, num_steps: int) -> EquilibriumModel:
    """Replaces ``model._solver`` with a stub reporting an all-failed batch.

    Parameters
    ----------
    model
        Model whose compiled solver is replaced in place.
    num_steps
        Step count reported for every attempt. Use :data:`_STEPS_EXHAUSTED` to drive the step
        summary into its "no model solved below max_steps" branch, or :data:`_STEPS_FINITE` to
        keep every attempt under the budget so the summary reports a finite maximum.

    Returns
    -------
    EquilibriumModel
        The same model, for call chaining.
    """

    def _solver(base_solution_array, parameters, subkey):
        return _all_failed_solution(parameters.batch_size, num_steps)

    model._solver = _solver
    return model


def test_solve_raises_when_no_model_converges() -> None:
    """solve raises a diagnostic RuntimeError, not a silent return, on zero convergence.

    A non-solution must not be mistaken for a converged state. The guard turns an all-failed batch
    into an explicit error that names the two operational knobs a user would tune (batch size and
    step budget), and leaves the output unset so no downstream code can read a stale result.
    """
    model: EquilibriumModel = _install_failed_solver(
        _single_species_model(), num_steps=_STEPS_EXHAUSTED
    )

    with pytest.raises(RuntimeError) as excinfo:
        model.solve(state=Planet(), mass_constraints=_mass_constraints(), solver="basic")

    message: str = str(excinfo.value)
    assert "No multistart model converged" in message
    assert "solver_max_steps" in message
    # A scalar constraint gives a batch of one; the count must be reported, not omitted.
    assert "1 attempt(s)" in message
    # No output may be published when the solve did not converge.
    assert model._output is None


def test_solve_raises_regardless_of_step_count() -> None:
    """Convergence, not step count, gates the error: a finite-step all-failed batch still raises.

    Here every attempt fails but stays under the step budget, so the step summary takes its finite
    branch instead of the "n/a" branch. The RuntimeError must still fire, confirming the guard keys
    off ``num_successful_models``, not off whether the budget was exhausted.
    """
    model: EquilibriumModel = _install_failed_solver(
        _single_species_model(), num_steps=_STEPS_FINITE
    )

    with pytest.raises(RuntimeError, match="No multistart model converged"):
        model.solve(state=Planet(), mass_constraints=_mass_constraints(), solver="basic")

    assert model._output is None


def test_step_summary_reports_na_when_all_attempts_exhaust_budget(caplog) -> None:
    """The step summary logs 'n/a' rather than coercing an infinite maximum to an integer.

    When no attempt finishes under the step budget the maximum sub-budget step count is -inf.
    Coercing that to int would raise and mask the real non-convergence error, so the summary must
    log a warning and skip the coercion. The finite info line must be absent in this case.
    """
    model: EquilibriumModel = _install_failed_solver(
        _single_species_model(), num_steps=_STEPS_EXHAUSTED
    )

    with caplog.at_level(logging.INFO, logger="atmodeller.classes"):
        with pytest.raises(RuntimeError):
            model.solve(state=Planet(), mass_constraints=_mass_constraints(), solver="basic")

    messages: list[str] = [record.getMessage() for record in caplog.records]
    assert any("no model solved below max_steps" in m for m in messages)
    # The finite branch (a bare integer maximum with no "n/a") must not have fired.
    assert not any(m.startswith("Solver steps (max) =") and "n/a" not in m for m in messages)


def test_step_summary_reports_finite_max_when_attempts_stay_under_budget(caplog) -> None:
    """The step summary reports the finite maximum when attempts finish under the budget.

    This is the complement of the "n/a" case: every attempt reports a finite sub-budget step count,
    so the summary logs that maximum as an integer. The exact sentinel is asserted to prove the
    finite branch, not the "n/a" branch, was taken; the guard still raises because none converged.
    """
    model: EquilibriumModel = _install_failed_solver(
        _single_species_model(), num_steps=_STEPS_FINITE
    )

    with caplog.at_level(logging.INFO, logger="atmodeller.classes"):
        with pytest.raises(RuntimeError):
            model.solve(state=Planet(), mass_constraints=_mass_constraints(), solver="basic")

    messages: list[str] = [record.getMessage() for record in caplog.records]
    assert any(m == f"Solver steps (max) = {_STEPS_FINITE}" for m in messages)
    assert not any("no model solved below max_steps" in m for m in messages)


def test_failed_resolve_clears_output_from_prior_success() -> None:
    """A failed solve drops output stored by an earlier successful solve on the same model.

    solve promises no output is stored when the batch fails to converge. On a model reused after an
    earlier convergence this is not automatic: unless the guard clears it, the prior output stays in
    place and is read as a result matching the new, non-converged constraints. This pins that a
    failed re-solve both clears the private slot and makes the public accessor refuse. Without the
    clear, the private slot would still hold the prior output and the first assertion below fails.
    """
    model: EquilibriumModel = _install_failed_solver(
        _single_species_model(), num_steps=_STEPS_EXHAUSTED
    )
    # Stand in for output left by a prior successful solve; only its non-None presence matters here.
    model._output = cast(Output, object())
    assert model._output is not None

    with pytest.raises(RuntimeError, match="No multistart model converged"):
        model.solve(state=Planet(), mass_constraints=_mass_constraints(), solver="basic")

    assert model._output is None
    with pytest.raises(AttributeError, match="Output has not been set"):
        _ = model.output


def test_failed_solve_clears_output_before_convergence_guard(monkeypatch) -> None:
    """A solve that fails while building the solver still drops output from a prior success.

    The no-stale-output promise must hold on every failure path, not only the non-convergence
    guard. The solver is constructed before the batch runs, so a build error exits solve early,
    ahead of the convergence check. This pins that the output is cleared at the top of solve, so a
    stale solution from an earlier convergence cannot survive an early exit. Were the clear placed
    only in the guard, the prior output would still be readable here and the first assertion below
    would fail.
    """
    model: EquilibriumModel = _single_species_model()
    assert model._solver is None
    # Stand in for output left by a prior successful solve; only its non-None presence matters here.
    model._output = cast(Output, object())
    assert model._output is not None

    def _raise_on_build(parameters):
        raise ValueError("solver build failed before the batch ran")

    # The basic dispatch arm builds the solver through this factory; make it fail so solve exits
    # during construction, strictly before the convergence guard runs.
    monkeypatch.setattr("atmodeller.classes.make_independent_solver", _raise_on_build)

    with pytest.raises(ValueError, match="solver build failed before the batch ran"):
        model.solve(state=Planet(), mass_constraints=_mass_constraints(), solver="basic")

    assert model._output is None
    with pytest.raises(AttributeError, match="Output has not been set"):
        _ = model.output


def test_basic_solver_dispatch_builds_independent_solver(monkeypatch) -> None:
    """solver="basic" routes through make_independent_solver and still guards non-convergence.

    The other tests pre-install a stub on the model, which bypasses solve's solver-construction
    branch entirely. This one leaves the solver unset and intercepts the factory, so the basic
    dispatch arm (build and record the independent solver) is exercised rather than skipped. The
    built solver reports an all-failed batch, so the convergence guard must still raise, the
    factory must have been invoked exactly once, and the recorded selection must flip from a
    pre-seeded "robust" to "basic" to prove the dispatch arm did the recording (not the default).
    """
    model: EquilibriumModel = _single_species_model()
    assert model._solver is None
    # Seed the opposite selection so the post-solve "basic" assertion cannot pass on the default.
    model._selected_solver = "robust"

    built: dict = {"calls": 0}

    def _fake_factory(parameters):
        built["calls"] += 1

        def _solver(base_solution_array, parameters, subkey):
            return _all_failed_solution(parameters.batch_size, _STEPS_EXHAUSTED)

        return _solver

    monkeypatch.setattr("atmodeller.classes.make_independent_solver", _fake_factory)

    with pytest.raises(RuntimeError, match="No multistart model converged"):
        model.solve(state=Planet(), mass_constraints=_mass_constraints(), solver="basic")

    # The dispatch arm built the solver exactly once and recorded the basic selection.
    assert built["calls"] == 1
    assert model._selected_solver == "basic"
