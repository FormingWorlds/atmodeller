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
"""Classes"""

import logging
import pprint
from collections.abc import Callable, Mapping
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, PRNGKeyArray

from atmodeller.constants import INITIAL_LOG_NUMBER_MOLES, INITIAL_LOG_STABILITY
from atmodeller.containers import Parameters, SolverParameters, SpeciesNetwork
from atmodeller.interfaces import FugacityConstraintProtocol, ThermodynamicStateProtocol
from atmodeller.output import Output, OutputDisequilibrium, OutputSolution
from atmodeller.solvers import MultiAttemptSolution, make_independent_solver, make_solver
from atmodeller.type_aliases import NpFloat

logger: logging.Logger = logging.getLogger(__name__)


class EquilibriumModel:
    """An equilibrium model

    This is the main class that the user interacts with to build equilibrium models, solve them,
    and retrieve the results.

    Args:
        species_network: Species network
    """

    _solver: Optional[Callable] = None
    _output: Optional[Output] = None
    _selected_solver: Literal["basic", "robust"] = "basic"

    def __init__(self, species_network: SpeciesNetwork):
        self.species_network: SpeciesNetwork = species_network
        logger.info("species_network = %s", str(self.species_network))
        temperature_min, temperature_max = self.species_network.get_temperature_range()
        logger.info(
            "Thermodynamic data requires temperatures between %d K and %d K",
            np.ceil(temperature_min),
            np.floor(temperature_max),
        )
        logger.info(
            "reactions = %s", pprint.pformat(self.species_network.get_reaction_dictionary())
        )

    @property
    def output(self) -> Output:
        if self._output is None:
            raise AttributeError("Output has not been set.")

        return self._output

    def calculate_disequilibrium(
        self, *, state: ThermodynamicStateProtocol, log_number_moles: ArrayLike
    ) -> None:
        """Computes the Gibbs free energy disequilibrium.

        This method calculates the Gibbs free energy difference (ΔG) for each considered reaction
        relative to equilibrium, based on the current state of the system. A value of zero
        indicates a reaction at equilibrium, while positive or negative values indicate departures
        from equilibrium in terms of energetic favourability.

        Args:
            state: Thermodynamic state
            log_number_moles: Log number of moles
        """
        parameters: Parameters = Parameters.create(self.species_network, state)
        solution_array: Array = broadcast_initial_solution(
            log_number_moles, None, self.species_network.number_species, parameters.batch_size
        )
        # jax.debug.print("solution_array = {out}", out=solution_array)

        self._output = OutputDisequilibrium(parameters, solution_array)

    def solve(
        self,
        *,
        initial_log_number_moles: Optional[ArrayLike] = None,
        initial_log_stability: Optional[ArrayLike] = None,
        state: Optional[ThermodynamicStateProtocol] = None,
        fugacity_constraints: Optional[Mapping[str, FugacityConstraintProtocol]] = None,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        solver_parameters: Optional[SolverParameters] = None,
        solver: Literal["basic", "robust"] = "robust",
        solver_recompile: bool = False,
    ) -> None:
        """Runs the nonlinear solver and initialises the output state.

        This method executes the compiled equilibrium solver produced by :meth:`set_solver` and
        stores the resulting solution for downstream processing. It optionally accepts updated
        planetary/environmental constraints and initial guesses for the nonlinear system. After
        successful convergence, an internal ``Output`` instance is created to expose number
        densities, activities, stabilities, and post-processed diagnostic quantities.

        If :meth:`set_solver` has not been called, a suitable solver will be constructed and
        JIT-compiled automatically. Repeated calls to :meth:`solve` with compatible shapes will be
        fast and will reuse cached compilation artifacts.

        Args:
            initial_log_number_moles: Initial log number of moles. Defaults to ``None``.
            initial_log_stability: Initial log stability. Defaults to ``None``.
            state: Thermodynamic state. Defaults to ``None``.
            fugacity_constraints: Fugacity constraints. Defaults to ``None``.
            mass_constraints: Mass constraints. Defaults to ``None``.
            solver_parameters: Solver parameters. Defaults to ``None``.
            solver: Build a ``basic`` (faster compile time) or a ``robust`` (slower compile time)
                solver. Defaults to ``robust``.
            solver_recompile: Force recompilation of the solver. Defaults to ``False``.

        Raises:
            RuntimeError: If no model in the multistart batch converges. The message reports the
                batch size and solver step budget, and no output is stored, so a non-solution is
                never mistaken for a converged state.
        """
        # Clear any output from a prior solve on this reused model before attempting a new one,
        # so no failure path (unknown solver, a solver-build error, or a non-converged batch) can
        # leave a stale, non-matching solution readable through the output property. The success
        # path restores the output once convergence is reached.
        self._output = None

        parameters: Parameters = Parameters.create(
            self.species_network, state, fugacity_constraints, mass_constraints, solver_parameters
        )
        base_solution_array: Array = broadcast_initial_solution(
            initial_log_number_moles,
            initial_log_stability,
            self.species_network.number_species,
            parameters.batch_size,
        )
        # jax.debug.print("base_solution_array = {out}", out=base_solution_array)

        key: PRNGKeyArray = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)  # Split the key for use in this function

        if self._solver is None or solver_recompile:
            if solver == "basic":
                self._solver = make_independent_solver(parameters)
                # Alternatively, could use the batch solver
                # self._solver = make_batch_solver(parameters)
            elif solver == "robust":
                self._solver = make_solver(parameters)
            else:
                raise ValueError(f"Unknown solver type: {solver}")
            self._selected_solver = solver

        multi_sol: MultiAttemptSolution = self._solver(base_solution_array, parameters, subkey)

        num_successful_models: int = jnp.count_nonzero(multi_sol.solver_success).item()
        num_failed_models: int = jnp.count_nonzero(~multi_sol.solver_success).item()

        logger.info(
            "Solve (%s) complete: %d (%0.2f%%) successful model(s)",
            self._selected_solver,
            num_successful_models,
            num_successful_models * 100 / parameters.batch_size,
        )
        if num_failed_models > 0:
            logger.warning(
                "%d (%0.2f%%) model(s) still failed",
                num_failed_models,
                num_failed_models * 100 / parameters.batch_size,
            )

        # Count unique values and their frequencies
        unique_vals, counts = jnp.unique(multi_sol.attempts, return_counts=True)
        for val, count in zip(unique_vals.tolist(), counts.tolist()):
            logger.info(
                "Multistart summary: %d (%0.2f%%) models(s) required %d attempt(s)",
                count,
                count * 100 / parameters.batch_size,
                val,
            )

        # Want the maximum number of steps for cases that solved
        mask_num_steps: Bool[Array, " batch"] = (
            multi_sol.num_steps < parameters.solver_parameters.max_steps
        )
        # Replace invalid values with -inf so they never win in the max
        max_less_than_max: Array = jnp.where(mask_num_steps, multi_sol.num_steps, -jnp.inf).max()
        if bool(jnp.isfinite(max_less_than_max).item()):
            logger.info("Solver steps (max) = %d", int(max_less_than_max.item()))
        else:
            logger.warning("Solver steps (max) = n/a (no model solved below max_steps)")

        if num_successful_models == 0:
            raise RuntimeError(
                "No multistart model converged "
                f"({parameters.batch_size} attempt(s), solver_max_steps="
                f"{parameters.solver_parameters.max_steps}); check the initial "
                "condition (temperature, fO2, and real-gas EOS validity range)."
            )

        self._output = OutputSolution(parameters, multi_sol.value, multi_sol)


def _broadcast_component(
    component: Optional[ArrayLike], default_value: float, dim: int, batch_size: int, name: str
) -> NpFloat:
    """Broadcasts a scalar, 1D, or 2D input array to shape ``(batch_size, dim)``.

    This function standardizes inputs that may be:
        - ``None`` (in which case ``default_value`` is used),
        - a scalar (promoted to a 1D array of length ``dim``),
        - a 1D array of shape ``(dim,)`` (broadcast across the batch),
        - or a 2D array of shape ``(batch_size``, dim)`` (used as-is).

    Args:
        component: The input data (or ``None``), representing either a scalar, 1D array, or 2D array
        default_value: The default scalar value to use if ``component`` is ``None``
        dim: The number of features or dimensions per batch item
        batch_size: The number of batch items
        name: Name of the component (used for error messages)

    Returns:
        A numpy array of shape ``(batch_size, dim)``, with values broadcast as needed

    Raises:
        ValueError: If the input array has an unexpected shape or inconsistent dimensions
    """
    if component is None:
        base: NpFloat = np.full((dim,), default_value, dtype=np.float64)
    else:
        component = np.asarray(component, dtype=np.float64)
        if component.ndim == 0:
            base = np.full((dim,), component.item(), dtype=np.float64)
        elif component.ndim == 1:
            if component.shape[0] != dim:
                raise ValueError(f"{name} should have shape ({dim},), got {component.shape}")
            base = component
        elif component.ndim == 2:
            if component.shape[0] != batch_size or component.shape[1] != dim:
                raise ValueError(
                    f"{name} should have shape ({batch_size}, {dim}), got {component.shape}"
                )
            # Replace NaNs with default_value
            component = np.where(np.isnan(component), default_value, component)
            return component
        else:
            raise ValueError(
                f"{name} must be a scalar, 1D, or 2D array, got shape {component.shape}"
            )

    # Promote 1D base to (batch_size, dim)
    return np.broadcast_to(base[None, :], (batch_size, dim))


def broadcast_initial_solution(
    initial_log_number_moles: Optional[ArrayLike],
    initial_log_stability: Optional[ArrayLike],
    number_of_species: int,
    batch_size: int,
) -> Float[Array, " batch_size solution"]:
    """Creates and broadcasts the initial solution to shape ``(batch_size, solution)``

    ``D = number_of_species + number_of_stability``, i.e. the total number of solution quantities

    Args:
        initial_log_number_moles: Initial log number moles or ``None``
        initial_log_stability: Initial log stability or ``None``
        number_of_species: Number of species
        batch_size: Batch size

    Returns:
        Initial solution with shape ``(batch_size, solution)``
    """
    number_moles: NpFloat = _broadcast_component(
        initial_log_number_moles,
        INITIAL_LOG_NUMBER_MOLES,
        number_of_species,
        batch_size,
        name="initial_log_number_moles",
    )
    stability: NpFloat = _broadcast_component(
        initial_log_stability,
        INITIAL_LOG_STABILITY,
        number_of_species,
        batch_size,
        name="initial_log_stability",
    )

    return jnp.concatenate((number_moles, stability), axis=-1)
