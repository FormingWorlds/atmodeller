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
"""Utilities for tests"""

import logging

import numpy as np
import pytest
from jaxtyping import ArrayLike
from pytest import approx

from atmodeller.interfaces import SolubilityProtocol

logger: logging.Logger = logging.getLogger("atmodeller.tests.solubility")

# Tolerances to compare the test results with target output.
# RTOL: float = 1.0e-8
RTOL: float = 0.61
"""Relative tolerance"""
# ATOL: float = 1.0e-8
ATOL: float = 0.61
"""Absolute tolerance"""


class CheckValues:
    """Helper to check and confirm values"""

    @classmethod
    def concentration(
        cls,
        function_name: str,
        solubility_model: SolubilityProtocol,
        expected_concentration: ArrayLike,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
        *,
        rtol=RTOL,
        atol=ATOL,
    ) -> None:
        concentration: ArrayLike = solubility_model.concentration(
            fugacity, temperature=temperature, pressure=pressure, fO2=fO2
        )
        logger.debug("%s, concentration = %s ppmw", function_name, concentration)

        assert np.asarray(concentration) == approx(np.asarray(expected_concentration), rtol, atol)


@pytest.fixture(scope="module")
def check_values():
    return CheckValues()
