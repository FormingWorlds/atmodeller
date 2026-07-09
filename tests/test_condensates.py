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
"""Tests for C-H-O systems with stable or unstable condensates"""

import logging

import numpy as np
import pytest
from jaxtyping import ArrayLike
from molmass import Formula

from atmodeller import debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, Planet, SpeciesNetwork, ThermodynamicState
from atmodeller.interfaces import FugacityConstraintProtocol
from atmodeller.output import Output
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)

# End-to-end tests that build and solve a full EquilibriumModel.
pytestmark = pytest.mark.integration

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""
TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage and FastChem"""

species: SpeciesNetwork = SpeciesNetwork.create(
    ("H2_g", "H2O_g", "CO_g", "CO2_g", "CH4_g", "O2_g", "C_cd")
)
CHO_model: EquilibriumModel = EquilibriumModel(species)


def test_graphite_stable(helper) -> None:
    """Tests graphite stable with around 50% condensed C mass fraction"""

    planet: Planet = Planet(temperature=873)
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(np.nan)
    }
    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 5 * h_kg
    o_kg: ArrayLike = 2.73159e19
    mass_constraints = {"C": c_kg, "H": h_kg, "O": o_kg}

    CHO_model.solve(
        state=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
        solver="basic",
    )
    output: Output = CHO_model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "O2_g": 1.27e-25,
        "H2_g": 14.564,
        "CO_g": 0.07276,
        "H2O_g": 4.527,
        "CO2_g": 0.061195,
        "CH4_g": 96.74,
        "C_cd_activity": 1.0,
        "mass_C_cd": 3.54162e20,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_graphite_unstable(helper) -> None:
    """Tests C-H-O system at IW+0.5 with graphite unstable

    Similar to :cite:p:`BHS22{Table E, row 2}`
    """

    planet: Planet = Planet(temperature=1400)
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(0.5)}
    oceans: ArrayLike = 3
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 1 * h_kg
    mass_constraints = {"C": c_kg, "H": h_kg}

    CHO_model.solve(
        state=planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )
    output: Output = CHO_model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "O2_g": 4.11e-13,
        "H2_g": 236.98,
        "CO_g": 46.42,
        "H2O_g": 337.16,
        "CO2_g": 30.88,
        "CH4_g": 28.66,
        "C_cd_activity": 0.12202,
        "mass_C_cd": 0.0,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_water_stable(helper) -> None:
    """Condensed water at 10 bar"""

    species: SpeciesNetwork = SpeciesNetwork.create(("H2_g", "H2O_g", "O2_g", "H2O_cd"))
    planet: Planet = Planet(temperature=411.75)
    model: EquilibriumModel = EquilibriumModel(species)

    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    o_kg: float = 1.14375e21
    mass_constraints = {"H": h_kg, "O": o_kg}

    model.solve(state=planet, mass_constraints=mass_constraints, solver="robust")
    output: Output = model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "H2O_g": 3.3596,
        "H2_g": 6.5604,
        "O2_g": 5.6433e-58,
        "H2O_cd_activity": 1.0,
        "mass_H2O_cd": 1.247201e21,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_graphite_water_stable(helper) -> None:
    """Tests C and water in equilibrium at 430 K and 10 bar"""

    species: SpeciesNetwork = SpeciesNetwork.create(
        ("H2O_g", "H2_g", "O2_g", "CO_g", "CO2_g", "CH4_g", "H2O_cd", "C_cd")
    )
    planet: Planet = Planet(temperature=430)
    model: EquilibriumModel = EquilibriumModel(species)

    h_kg: float = 3.10e20
    c_kg: float = 1.08e20
    o_kg: float = 2.48298883581636e21
    mass_constraints = {"C": c_kg, "H": h_kg, "O": o_kg}

    model.solve(state=planet, mass_constraints=mass_constraints, solver="basic")
    output: Output = model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "CH4_g": 0.3241,
        "CO2_g": 4.3064,
        "CO_g": 2.77e-6,
        "C_cd_activity": 1.0,
        "H2O_g": 5.3672,
        "H2O_cd_activity": 1.0,
        "H2_g": 0.0023,
        "O2_g": 4.74e-48,
        "mass_C_cd": 8.75101e19,
        "mass_H2O_cd": 2.74821e21,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_impose_stable(helper) -> None:
    """Tests a user-imposed stable condensate"""

    # Since in this example we do not provide carbon in the injected gas stream, we cannot solve
    # for the stability of any carbon-bearing products because in order to do so requires
    # specification of the mass of carbon in the system.
    C_cr = ChemicalSpecies.create_condensed("C", solve_for_stability=False)

    # Define allowable gas species at equilibrium
    H2_g = ChemicalSpecies.create_gas("H2")
    N2_g = ChemicalSpecies.create_gas("N2")
    CH4_g = ChemicalSpecies.create_gas("CH4")
    CHN_g = ChemicalSpecies.create_gas("CHN")
    H_g = ChemicalSpecies.create_gas("H")

    species = SpeciesNetwork((C_cr, H2_g, N2_g, CH4_g, CHN_g, H_g))

    model = EquilibriumModel(species)

    # Set the temperature and pressure
    state = ThermodynamicState(temperature=1773.15, melt_fraction=0, pressure=1)

    # Define the mole fractions of input gases
    mole_fractions = {"H2": 0.5, "N2": 0.5}

    # Define the composition of the input gas mixture by mass
    mass_constraints = {key: value * Formula(key).mass for key, value in mole_fractions.items()}

    # Solve
    model.solve(state=state, mass_constraints=mass_constraints, solver="basic")

    output: Output = model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "CH4_g": 0.000194708,
        "C_cd_activity": 1.0,
        "H_g": 0.000201266,
        "H2_g": 0.49807992,
        "N2_g": 0.49866269,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
