"""FactSage and FastChem benchmark formulation, element-balance, and K_p closure tests.

Verifies that across all 9 active FactSage (Gibbs energy minimisation) and FastChem
(law of mass action) benchmark states in ``test_CHO.py``, ``test_HO.py``, and
``test_condensates.py``, plus the 3 skipped FactSage CHO benchmarks, ``atmodeller``
simultaneously:

1. Matches the FactSage and FastChem reference gas partial pressures, condensate
   activities, and condensate total masses in log10 space within ``0.05 dex``
   (``|log10(actual) - log10(target)| <= 0.05``) for condensate masses and major
   species (``>= 1.0e-2`` bar), and within ``0.10 dex`` for trace species
   (``< 1.0e-2`` bar).
2. Conserves elemental mass summed from solved gas, condensate, and dissolved
   melt species moles against the input mass constraints to ``SolverParameters.atol``
   (``1.0e-6``).
3. Satisfies reaction-quotient mass-action equilibrium closure
   ``|reaction_matrix @ log_activity - log_Kp| <= 1.0e-6`` and Gibbs-to-Kp
   consistency ``get_log_Kp == -reaction_matrix @ (G^0 / RT)``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from molmass import Formula

from atmodeller import (
    ChemicalSpecies,
    EquilibriumModel,
    Planet,
    SpeciesNetwork,
    ThermodynamicState,
)
from atmodeller.containers import SolverParameters
from atmodeller.engine import get_log_Kp
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

pytestmark = pytest.mark.integration

SPECIES_STOICH: dict[str, dict[str, int]] = {
    "H2_g": {"H": 2},
    "H2O_g": {"H": 2, "O": 1},
    "H2O_cd": {"H": 2, "O": 1},
    "CO_g": {"C": 1, "O": 1},
    "CO2_g": {"C": 1, "O": 2},
    "CH4_g": {"C": 1, "H": 4},
    "O2_g": {"O": 2},
    "C_cd": {"C": 1},
    "N2_g": {"N": 2},
    "CHN_g": {"C": 1, "H": 1, "N": 1},
    "H_g": {"H": 1},
}

# Log10 tolerance [dex] applied directly as |log10(actual) - log10(target)| <= tol_dex.
LOG10_TOL_MAJOR_DEX = 0.05
LOG10_TOL_TRACE_DEX = 0.10


def _target_element_masses_from_input(mass_constraints: dict[str, Any]) -> dict[str, float]:
    """Convert input mass_constraints (elemental or molecular keys) to per-element mass [kg]."""
    elem_targets: dict[str, float] = {}
    for key, raw_val in mass_constraints.items():
        mass_kg = float(np.squeeze(raw_val))
        if not math.isfinite(mass_kg) or mass_kg <= 0.0:
            continue
        sp_lookup = f"{key}_g"
        if sp_lookup in SPECIES_STOICH:
            stoich = SPECIES_STOICH[sp_lookup]
            m_sp = sum(n * Formula(el).mass * 1.0e-3 for el, n in stoich.items())
            for el, count in stoich.items():
                el_mm = Formula(el).mass * 1.0e-3
                elem_targets[el] = elem_targets.get(el, 0.0) + mass_kg * (count * el_mm / m_sp)
        else:
            elem_targets[key] = elem_targets.get(key, 0.0) + mass_kg
    return elem_targets


def _build_benchmark_cases() -> list[dict[str, Any]]:
    """Construct all 12 FactSage/FastChem benchmark cases from test_CHO, test_HO, test_condensates."""
    cho_gas_species = SpeciesNetwork.create(("H2_g", "H2O_g", "CO_g", "CO2_g", "CH4_g", "O2_g"))
    cho_cond_species = SpeciesNetwork.create(
        ("H2_g", "H2O_g", "CO_g", "CO2_g", "CH4_g", "O2_g", "C_cd")
    )
    h_3eo = earth_oceans_to_hydrogen_mass(3)
    h_1eo = earth_oceans_to_hydrogen_mass(1)

    c_cr = ChemicalSpecies.create_condensed("C", solve_for_stability=False)
    impose_net = SpeciesNetwork(
        (
            c_cr,
            ChemicalSpecies.create_gas("H2"),
            ChemicalSpecies.create_gas("N2"),
            ChemicalSpecies.create_gas("CH4"),
            ChemicalSpecies.create_gas("CHN"),
            ChemicalSpecies.create_gas("H"),
        )
    )
    mole_fractions = {"H2": 0.5, "N2": 0.5}
    impose_mc = {k: v * Formula(k).mass for k, v in mole_fractions.items()}

    return [
        {
            "case_id": "test_CHO_reduced",
            "species_network": cho_gas_species,
            "state": Planet(temperature=1400),
            "fugacity_constraints": {"O2_g": IronWustiteBuffer(-2)},
            "mass_constraints": {"H": h_3eo, "C": 1 * h_3eo},
            "targets": {
                "FactSage": {
                    "H2_g": 175.5,
                    "H2O_g": 13.8,
                    "CO_g": 6.21,
                    "CO2_g": 0.228,
                    "CH4_g": 38.07,
                    "O2_g": 1.25e-15,
                }
            },
        },
        {
            "case_id": "test_CHO_IW",
            "species_network": cho_gas_species,
            "state": Planet(temperature=1400),
            "fugacity_constraints": {"O2_g": IronWustiteBuffer(0.5)},
            "mass_constraints": {"H": h_3eo, "C": 1 * h_3eo},
            "targets": {
                "FactSage": {
                    "CH4_g": 28.66,
                    "CO2_g": 30.88,
                    "CO_g": 46.42,
                    "H2O_g": 337.16,
                    "H2_g": 236.98,
                    "O2_g": 4.11e-13,
                },
                "FastChem": {
                    "CH4_g": 29.61919788,
                    "CO2_g": 29.82548282,
                    "CO_g": 45.94958264,
                    "H2O_g": 332.03616807,
                    "H2_g": 236.73845646,
                    "O2_g": 3.96475584e-13,
                },
            },
        },
        {
            "case_id": "test_CHO_oxidised",
            "species_network": cho_gas_species,
            "state": Planet(temperature=1400),
            "fugacity_constraints": {"O2_g": IronWustiteBuffer(2)},
            "mass_constraints": {"H": h_1eo, "C": 0.1 * h_1eo},
            "targets": {
                "FactSage": {
                    "CH4_g": 0.00129,
                    "CO2_g": 3.25,
                    "CO_g": 0.873,
                    "H2O_g": 218.48,
                    "H2_g": 27.40,
                    "O2_g": 1.29e-11,
                }
            },
        },
        {
            "case_id": "test_CHO_highly_oxidised",
            "species_network": cho_gas_species,
            "state": Planet(temperature=1400),
            "fugacity_constraints": {"O2_g": IronWustiteBuffer(4)},
            "mass_constraints": {"H": h_1eo, "C": 5 * h_1eo},
            "targets": {
                "FactSage": {
                    "CH4_g": 7.13e-05,
                    "CO2_g": 357.23,
                    "CO_g": 10.21,
                    "H2O_g": 432.08,
                    "H2_g": 5.78,
                    "O2_g": 1.14e-09,
                }
            },
        },
        {
            "case_id": "test_CHO_middle_temperature",
            "species_network": cho_gas_species,
            "state": Planet(temperature=873),
            "fugacity_constraints": {"O2_g": IronWustiteBuffer()},
            "mass_constraints": {"C": 1 * h_1eo, "H": h_1eo},
            "targets": {
                "FactSage": {
                    "H2_g": 59.066,
                    "H2O_g": 18.320,
                    "CO_g": 8.91e-4,
                    "CO2_g": 7.48e-4,
                    "CH4_g": 19.548,
                    "O2_g": 1.27e-25,
                }
            },
        },
        {
            "case_id": "test_CHO_low_temperature",
            "species_network": cho_gas_species,
            "state": Planet(temperature=450),
            "fugacity_constraints": {"O2_g": IronWustiteBuffer(np.nan)},
            "mass_constraints": {"C": 1 * h_1eo, "H": h_1eo, "O": 1.02999e20},
            "targets": {
                "FactSage": {
                    "H2_g": 55.475,
                    "H2O_g": 8.0,
                    "CO2_g": 1.24e-14,
                    "O2_g": 7.85e-54,
                    "CH4_g": 16.037,
                    "CO_g": 2.12e-16,
                }
            },
        },
        {
            "case_id": "test_H_O",
            "species_network": SpeciesNetwork.create(("H2_g", "H2O_g", "O2_g")),
            "state": Planet(),
            "mass_constraints": {"H": h_1eo, "O": 6.25774e20},
            "targets": {
                "FastChem": {
                    "H2O_g": 76.45861543,
                    "H2_g": 73.84378192,
                    "O2_g": 8.91399329e-08,
                }
            },
        },
        {
            "case_id": "test_graphite_stable",
            "species_network": cho_cond_species,
            "state": Planet(temperature=873),
            "fugacity_constraints": {"O2_g": IronWustiteBuffer(np.nan)},
            "mass_constraints": {"C": 5 * h_1eo, "H": h_1eo, "O": 2.73159e19},
            "targets": {
                "FactSage": {
                    "O2_g": 1.27e-25,
                    "H2_g": 14.564,
                    "CO_g": 0.07276,
                    "H2O_g": 4.527,
                    "CO2_g": 0.061195,
                    "CH4_g": 96.74,
                    "C_cd_activity": 1.0,
                    "mass_C_cd": 3.54162e20,
                }
            },
        },
        {
            "case_id": "test_graphite_unstable",
            "species_network": cho_cond_species,
            "state": Planet(temperature=1400),
            "fugacity_constraints": {"O2_g": IronWustiteBuffer(0.5)},
            "mass_constraints": {"C": 1 * h_3eo, "H": h_3eo},
            "targets": {
                "FactSage": {
                    "O2_g": 4.11e-13,
                    "H2_g": 236.98,
                    "CO_g": 46.42,
                    "H2O_g": 337.16,
                    "CO2_g": 30.88,
                    "CH4_g": 28.66,
                    "C_cd_activity": 0.12202,
                    "mass_C_cd": 0.0,
                }
            },
        },
        {
            "case_id": "test_water_stable",
            "species_network": SpeciesNetwork.create(("H2_g", "H2O_g", "O2_g", "H2O_cd")),
            "state": Planet(temperature=411.75),
            "mass_constraints": {"H": h_1eo, "O": 1.14375e21},
            "solver": "robust",
            "targets": {
                "FactSage": {
                    "H2O_g": 3.3596,
                    "H2_g": 6.5604,
                    "O2_g": 5.6433e-58,
                    "H2O_cd_activity": 1.0,
                    "mass_H2O_cd": 1.247201e21,
                }
            },
        },
        {
            "case_id": "test_graphite_water_stable",
            "species_network": SpeciesNetwork.create(
                ("H2O_g", "H2_g", "O2_g", "CO_g", "CO2_g", "CH4_g", "H2O_cd", "C_cd")
            ),
            "state": Planet(temperature=430),
            "mass_constraints": {
                "C": 1.08e20,
                "H": 3.10e20,
                "O": 2.48298883581636e21,
            },
            "targets": {
                "FactSage": {
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
            },
        },
        {
            "case_id": "test_impose_stable",
            "species_network": impose_net,
            "state": ThermodynamicState(temperature=1773.15, melt_fraction=0, pressure=1),
            "mass_constraints": impose_mc,
            "targets": {
                "FactSage": {
                    "CH4_g": 0.000194708,
                    "C_cd_activity": 1.0,
                    "H_g": 0.000201266,
                    "H2_g": 0.49807992,
                    "N2_g": 0.49866269,
                }
            },
        },
    ]


def test_factsage_and_fastchem_formulation_element_and_k_closure() -> None:
    """Verify species match, element conservation, and K_p closure across all 12 benchmarks."""
    cases = _build_benchmark_cases()
    default_atol = SolverParameters().atol
    assert default_atol == pytest.approx(1.0e-6, abs=0.0)

    for case in cases:
        case_id = case["case_id"]
        model = EquilibriumModel(case["species_network"])
        model.solve(
            state=case["state"],
            fugacity_constraints=case.get("fugacity_constraints"),
            mass_constraints=case["mass_constraints"],
            solver=case.get("solver", "basic"),
        )
        output = model.output
        assert bool(np.squeeze(output.multi_attempt_solution.solver_success)), (
            f"{case_id}: solver did not converge"
        )
        quick = output.quick_look()
        asdict = output.asdict()
        input_elem_targets = _target_element_masses_from_input(case["mass_constraints"])

        # 1. Formulation species and condensate mass match in direct log10 space [dex].
        for ref_code, target_dict in case["targets"].items():
            for sp_key, target_val in target_dict.items():
                if sp_key.startswith("mass_"):
                    cond_sp = sp_key[len("mass_") :]
                    actual_mass_kg = float(np.squeeze(asdict[cond_sp]["total_mass"]))
                    if target_val > 0.0:
                        assert math.isfinite(actual_mass_kg) and actual_mass_kg > 0.0
                        d_dex = abs(math.log10(actual_mass_kg) - math.log10(float(target_val)))
                        assert d_dex <= LOG10_TOL_MAJOR_DEX, (
                            f"{case_id} ({ref_code}) condensate {cond_sp} total_mass: "
                            f"|d_log10|={d_dex:.6f} dex > {LOG10_TOL_MAJOR_DEX} dex"
                        )
                    else:
                        # At stability floor tau = 1e-25, inactive condensate mass is O(tau) of budget.
                        cond_el = next(iter(SPECIES_STOICH[cond_sp]))
                        el_budget = input_elem_targets[cond_el]
                        assert actual_mass_kg / el_budget < 1.0e-23, (
                            f"{case_id} ({ref_code}) inactive {cond_sp} mass ratio "
                            f"{actual_mass_kg / el_budget!r}"
                        )
                else:
                    assert sp_key in quick, f"{case_id}: missing {sp_key} in quick_look"
                    actual_val = float(np.squeeze(quick[sp_key]))
                    assert math.isfinite(actual_val) and actual_val > 0.0
                    tol_dex = (
                        LOG10_TOL_MAJOR_DEX if float(target_val) >= 1.0e-2 else LOG10_TOL_TRACE_DEX
                    )
                    d_dex = abs(math.log10(actual_val) - math.log10(float(target_val)))
                    assert d_dex <= tol_dex, (
                        f"{case_id} ({ref_code}) species {sp_key}: "
                        f"|d_log10|={d_dex:.6f} dex > {tol_dex} dex"
                    )

        # 2. Stoichiometric species-summed elemental mass conservation against input constraints.
        elem_masses_recovered: dict[str, float] = {}
        melt_moles_arr = output.species_number_moles_in_melt()[0]
        for sp_idx, sp in enumerate(output.species):
            sp_name = sp.name
            n_moles = float(output.number_moles[0, sp_idx])
            n_melt = float(melt_moles_arr[sp_idx])
            stoich = SPECIES_STOICH[sp_name]
            for el, count in stoich.items():
                el_molar_mass_kg = Formula(el).mass * 1.0e-3
                total_sp_moles = (
                    n_moles + n_melt if bool(output.gas_species_mask[sp_idx]) else n_moles
                )
                elem_masses_recovered[el] = (
                    elem_masses_recovered.get(el, 0.0) + total_sp_moles * count * el_molar_mass_kg
                )

        assert len(input_elem_targets) >= 1
        for el, target_kg in input_elem_targets.items():
            rec_kg = elem_masses_recovered.get(el, 0.0)
            rel_err = abs(rec_kg - target_kg) / target_kg
            assert rel_err <= default_atol, (
                f"{case_id} element {el} conservation failure: "
                f"recovered={rec_kg!r} kg, target={target_kg!r} kg, rel_err={rel_err!r}"
            )

        # 3. Gibbs-to-Kp consistency and reaction-quotient closure |ln Q - ln K_p| <= 1e-6.
        rxn_matrix = np.asarray(output.parameters.species_network.reaction_matrix, dtype=float)
        t_arr = output.parameters.state.temperature
        gibbs_over_rt = np.array(
            [
                float(np.squeeze(sp.data.get_gibbs_over_RT(t_arr)))
                for sp in output.parameters.species_network
            ],
            dtype=float,
        )
        expected_log_kp = -1.0 * (rxn_matrix @ gibbs_over_rt)
        code_log_kp = np.asarray(get_log_Kp(output.parameters), dtype=float)
        assert np.allclose(code_log_kp, expected_log_kp, rtol=1.0e-10, atol=1.0e-10), (
            f"{case_id}: get_log_Kp mismatch vs -reaction_matrix @ (G^0/RT)"
        )

        log_act = np.asarray(output.log_activity()[0], dtype=float)
        ln_q_minus_ln_kp = np.abs((rxn_matrix @ log_act) - expected_log_kp)
        assert float(np.max(ln_q_minus_ln_kp)) <= default_atol, (
            f"{case_id}: reaction-quotient residual {float(np.max(ln_q_minus_ln_kp))!r} > 1e-6"
        )
