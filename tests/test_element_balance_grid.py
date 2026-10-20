"""Elemental mass-balance consistency tests over the 75-case parameter grid.

Verifies stoichiometric species-to-element mass recovery across gas, dissolved
melt, and condensate phases against the input elemental inventory over the
60-point buffered grid (5 fO2 x 4 T x 3 inventories), the 12-point
authoritative-oxygen grid (4 T x 3 inventories), 2 volatile-inventory
configurations using elemental budgets from PROTEUS reference inputs at
T = 2000 K, and a 5-noble-gas complement configuration.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from molmass import Formula

from atmodeller import (
    ChemicalSpecies,
    EquilibriumModel,
    Planet,
    SpeciesNetwork,
)
from atmodeller.containers import SolverParameters
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata import IronWustiteBuffer

pytestmark = pytest.mark.integration

ATMODELLER_SPECIES_STOICH: dict[str, dict[str, int]] = {
    "H2O_g": {"H": 2, "O": 1},
    "H2_g": {"H": 2},
    "O2_g": {"O": 2},
    "CO2_g": {"C": 1, "O": 2},
    "CO_g": {"C": 1, "O": 1},
    "CH4_g": {"C": 1, "H": 4},
    "N2_g": {"N": 2},
    "H3N_g": {"N": 1, "H": 3},
    "S2_g": {"S": 2},
    "O2S_g": {"S": 1, "O": 2},
    "H2S_g": {"H": 2, "S": 1},
    "He_g": {"He": 1},
    "Ne_g": {"Ne": 1},
    "Ar_g": {"Ar": 1},
    "Kr_g": {"Kr": 1},
    "Xe_g": {"Xe": 1},
    "C_cd": {"C": 1},
}

# Solver relative residual limit matching SolverParameters.atol = 1.0e-6.
SOLVER_RESIDUAL_TOL = 1.0e-6


def _reconstruct_atmodeller_element_masses(
    out_dict: dict,
    elements: tuple[str, ...],
    batch_idx: int = 0,
) -> dict[str, dict[str, float]]:
    """Reconstruct per-element gas, melt, and condensate masses [kg] from species outputs."""
    el_mmw = {el: Formula(el).mass * 1e-3 for el in elements}
    res: dict[str, dict[str, float]] = {
        el: {"gas_kg": 0.0, "liq_kg": 0.0, "cond_kg": 0.0, "total_kg": 0.0} for el in elements
    }
    for sp_key, stoich in ATMODELLER_SPECIES_STOICH.items():
        if sp_key not in out_dict:
            continue
        sp_data = out_dict[sp_key]
        m_sp = sum(n * Formula(el).mass * 1e-3 for el, n in stoich.items())
        if sp_key.endswith("_cd"):
            sp_cond = float(np.atleast_1d(sp_data["total_mass"])[batch_idx])
            for el, n_atom in stoich.items():
                if el in res:
                    w_frac = (n_atom * el_mmw[el]) / m_sp
                    res[el]["cond_kg"] += sp_cond * w_frac
        else:
            sp_gas = float(np.atleast_1d(sp_data["gas_mass"])[batch_idx])
            sp_liq = float(np.atleast_1d(sp_data["dissolved_mass"])[batch_idx])
            for el, n_atom in stoich.items():
                if el in res:
                    w_frac = (n_atom * el_mmw[el]) / m_sp
                    res[el]["gas_kg"] += sp_gas * w_frac
                    res[el]["liq_kg"] += sp_liq * w_frac
    for el in elements:
        res[el]["total_kg"] = res[el]["gas_kg"] + res[el]["liq_kg"] + res[el]["cond_kg"]
    return res


def _build_grid_cases() -> list[dict]:
    """Build the 75-case parameter grid (60 buffered + 12 authoritative O + 3 reference-budget/noble)."""
    mass_ocean = 1.55e20
    m_earth = 5.972e24
    r_earth = 6.335439e6
    cmf = 0.325
    m_mantle_default = (1.0 - cmf) * m_earth

    m_h2 = 0.00201588
    m_o = 0.015999
    m_c = 0.012011
    m_s = 0.032065

    inventories: dict[str, dict[str, float]] = {}
    for label, scale in (("low", 0.1), ("default", 1.0), ("high", 10.0)):
        h_kg = scale * mass_ocean
        c_kg = 1.0 * h_kg
        n_kg = 0.5 * h_kg
        s_kg = 1.0 * h_kg
        o_stoich_kg = h_kg * (m_o / m_h2) + c_kg * (2.0 * m_o / m_c) + s_kg * (2.0 * m_o / m_s)
        inventories[label] = {
            "H": h_kg,
            "C": c_kg,
            "N": n_kg,
            "S": s_kg,
            "O": o_stoich_kg,
        }

    fo2_grid = (-4.0, -2.0, 0.0, 2.0, 4.0)
    t_grid = (1500.0, 2000.0, 2500.0, 3000.0)
    inv_labels = ("low", "default", "high")

    cases: list[dict] = []
    for inv_label in inv_labels:
        inv = inventories[inv_label]
        for t_k in t_grid:
            for fo2 in fo2_grid:
                cases.append(
                    {
                        "case_id": f"buffered_{inv_label}_T{int(t_k)}_IW{fo2:+.0f}",
                        "mode": "buffered",
                        "T_K": t_k,
                        "fO2_shift_IW": fo2,
                        "M_planet": m_earth,
                        "R_int": r_earth,
                        "Phi_global": 1.0,
                        "targets": {k: inv[k] for k in ("H", "C", "N", "S")},
                    }
                )
    for inv_label in inv_labels:
        inv = inventories[inv_label]
        for t_k in t_grid:
            cases.append(
                {
                    "case_id": f"authO_{inv_label}_T{int(t_k)}",
                    "mode": "authoritative_O",
                    "T_K": t_k,
                    "fO2_shift_IW": 2.0,
                    "M_planet": m_earth,
                    "R_int": r_earth,
                    "Phi_global": 1.0,
                    "targets": {k: inv[k] for k in ("H", "C", "N", "S", "O")},
                }
            )
    # Volatile budgets and fO2_shift_IW from PROTEUS reference inputs evaluated at T = 2000 K.
    cases.append(
        {
            "case_id": "budget_tutorial_earth_T2000_IW+2",
            "mode": "buffered",
            "T_K": 2000.0,
            "fO2_shift_IW": 2.0,
            "M_planet": m_earth,
            "R_int": r_earth,
            "Phi_global": 1.0,
            "targets": {"H": 4.7e20, "C": 2.73e20, "N": 0.0, "S": 0.0},
        }
    )
    cases.append(
        {
            "case_id": "budget_cmb_singleimpact_T2000_IW+1",
            "mode": "buffered",
            "T_K": 2000.0,
            "fO2_shift_IW": 1.0,
            "M_planet": 0.5 * m_earth,
            "R_int": (0.5 ** (1.0 / 3.0)) * r_earth,
            "Phi_global": 1.0,
            "targets": {"H": 2.4e20, "C": 1.4e20, "N": 4.0e18, "S": 1.0e20},
        }
    )
    noble_mass = 1.0e-6 * m_mantle_default
    noble_targets = {k: inventories["default"][k] for k in ("H", "C", "N", "S")}
    for ng in ("He", "Ne", "Ar", "Kr", "Xe"):
        noble_targets[ng] = noble_mass
    cases.append(
        {
            "case_id": "budget_noble_gas_complement_T2000_IW+2",
            "mode": "buffered_noble",
            "T_K": 2000.0,
            "fO2_shift_IW": 2.0,
            "M_planet": m_earth,
            "R_int": r_earth,
            "Phi_global": 1.0,
            "targets": noble_targets,
        }
    )
    return cases


def _solve_and_assert_group(
    model: EquilibriumModel,
    group_cases: list[dict],
    checked_elements: tuple[str, ...],
    authoritative_o: bool,
    solver_params: SolverParameters,
) -> tuple[bool, bool]:
    """Solve a homogeneous batch of grid cases and assert per-element mass balance."""
    n_batch = len(group_cases)
    cmf = 0.325
    planet = Planet(
        planet_mass=np.array([c["M_planet"] for c in group_cases], dtype=float),
        core_mass_fraction=np.full(n_batch, cmf, dtype=float),
        mantle_melt_fraction=np.array([c["Phi_global"] for c in group_cases], dtype=float),
        surface_radius=np.array([c["R_int"] for c in group_cases], dtype=float),
        temperature=np.array([c["T_K"] for c in group_cases], dtype=float),
        pressure=np.full(n_batch, np.nan, dtype=float),
    )
    fug = (
        None
        if authoritative_o
        else {
            "O2_g": IronWustiteBuffer(
                np.array([c["fO2_shift_IW"] for c in group_cases], dtype=float)
            )
        }
    )
    mass_constraints = {
        el: np.array([c["targets"][el] for c in group_cases], dtype=float)
        for el in checked_elements
    }

    model.solve(
        state=planet,
        fugacity_constraints=fug,
        mass_constraints=mass_constraints,
        solver="robust",
        solver_parameters=solver_params,
    )
    solver_success = np.asarray(model.output.multi_attempt_solution.solver_success, dtype=bool)
    assert np.all(solver_success), (
        f"Solver failed on {[group_cases[i]['case_id'] for i in np.where(~solver_success)[0]]}"
    )

    out_dict = model.output.asdict()
    reconstruct_elems = checked_elements if authoritative_o else checked_elements + ("O",)
    saw_liq = False
    saw_cond = False

    for idx, case in enumerate(group_cases):
        case_id = case["case_id"]
        rec = _reconstruct_atmodeller_element_masses(out_dict, reconstruct_elems, batch_idx=idx)
        if not authoritative_o:
            assert math.isfinite(rec["O"]["total_kg"]), f"{case_id}: O total_kg not finite"
            assert rec["O"]["total_kg"] > 0.0, f"{case_id}: O total_kg <= 0"
        for el in checked_elements:
            if rec[el]["liq_kg"] > 0.0:
                saw_liq = True
            if rec[el]["cond_kg"] > 0.0:
                saw_cond = True
            t_kg = float(case["targets"][el])
            rec_tot = rec[el]["total_kg"]
            abs_err = abs(rec_tot - t_kg)
            solver_tol_kg = SOLVER_RESIDUAL_TOL * t_kg
            assert abs_err <= solver_tol_kg, (
                f"{case_id} {el}: |recovered - target| = {abs_err!r} > solver tol {solver_tol_kg!r}"
            )
    return saw_liq, saw_cond


def test_element_balance_across_75_case_grid():
    """Verify species-reconstructed elemental mass conservation across all 75 grid cases."""
    cases = _build_grid_cases()
    assert len(cases) == 75

    sol_models = get_solubility_models()
    base_species = (
        ChemicalSpecies.create_gas("H2O", solubility=sol_models["H2O_peridotite_sossi23"]),
        ChemicalSpecies.create_gas("H2", solubility=sol_models["H2_basalt_hirschmann12"]),
        ChemicalSpecies.create_gas("O2"),
        ChemicalSpecies.create_gas("CO2", solubility=sol_models["CO2_basalt_dixon95"]),
        ChemicalSpecies.create_gas("CO", solubility=sol_models["CO_basalt_yoshioka19"]),
        ChemicalSpecies.create_gas("CH4", solubility=sol_models["CH4_basalt_ardia13"]),
        ChemicalSpecies.create_gas("N2", solubility=sol_models["N2_basalt_dasgupta22"]),
        ChemicalSpecies.create_gas("H3N"),
        ChemicalSpecies.create_gas("S2", solubility=sol_models["S2_sulfide_basalt_boulliung23"]),
        ChemicalSpecies.create_gas("O2S"),
        ChemicalSpecies.create_gas("H2S"),
        ChemicalSpecies.create_condensed("C"),
    )
    model_base = EquilibriumModel(SpeciesNetwork(base_species))
    model_autho = EquilibriumModel(SpeciesNetwork(base_species))

    hco_species = base_species[:6] + base_species[-1:]
    model_hco = EquilibriumModel(SpeciesNetwork(hco_species))

    noble_species = base_species + tuple(
        ChemicalSpecies.create_gas(ng, solubility=sol_models[f"{ng}_basalt_jambon86"])
        for ng in ("He", "Ne", "Ar", "Kr", "Xe")
    )
    model_noble = EquilibriumModel(SpeciesNetwork(noble_species))

    solver_params = SolverParameters(atol=1.0e-6, rtol=1.0e-4, max_steps=1024, multistart=10)

    buf_4el_cases = [
        c
        for c in cases
        if c["mode"] == "buffered" and not c["case_id"].startswith("budget_tutorial_earth")
    ]
    autho_cases = [c for c in cases if c["mode"] == "authoritative_O"]
    hco_cases = [c for c in cases if c["case_id"].startswith("budget_tutorial_earth")]
    noble_cases = [c for c in cases if c["mode"] == "buffered_noble"]

    assert len(buf_4el_cases) + len(autho_cases) + len(hco_cases) + len(noble_cases) == 75

    liq_b, cond_b = _solve_and_assert_group(
        model_base, buf_4el_cases, ("H", "C", "N", "S"), False, solver_params
    )
    liq_a, cond_a = _solve_and_assert_group(
        model_autho, autho_cases, ("H", "C", "N", "S", "O"), True, solver_params
    )
    liq_h, cond_h = _solve_and_assert_group(model_hco, hco_cases, ("H", "C"), False, solver_params)
    liq_n, cond_n = _solve_and_assert_group(
        model_noble,
        noble_cases,
        ("H", "C", "N", "S", "He", "Ne", "Ar", "Kr", "Xe"),
        False,
        solver_params,
    )
    assert any((liq_b, liq_a, liq_h, liq_n)), "Dissolved melt branch was never exercised"
    assert any((cond_b, cond_a, cond_h, cond_n)), "Graphite condensate branch was never exercised"


def test_engine_element_moles_and_mass_residual_against_stoichiometry():
    """Verify get_element_moles, get_element_moles_in_melt, and objective_function against stoichiometry."""
    sol_models = get_solubility_models()
    species = (
        ChemicalSpecies.create_gas("H2O", solubility=sol_models["H2O_peridotite_sossi23"]),
        ChemicalSpecies.create_gas("H2", solubility=sol_models["H2_basalt_hirschmann12"]),
        ChemicalSpecies.create_gas("O2"),
        ChemicalSpecies.create_gas("CO2", solubility=sol_models["CO2_basalt_dixon95"]),
        ChemicalSpecies.create_gas("CO", solubility=sol_models["CO_basalt_yoshioka19"]),
        ChemicalSpecies.create_gas("CH4", solubility=sol_models["CH4_basalt_ardia13"]),
        ChemicalSpecies.create_gas("N2", solubility=sol_models["N2_basalt_dasgupta22"]),
        ChemicalSpecies.create_gas("H3N"),
        ChemicalSpecies.create_gas("S2", solubility=sol_models["S2_sulfide_basalt_boulliung23"]),
        ChemicalSpecies.create_gas("O2S"),
        ChemicalSpecies.create_gas("H2S"),
        ChemicalSpecies.create_condensed("C"),
    )
    model = EquilibriumModel(SpeciesNetwork(species))
    planet = Planet(
        planet_mass=5.972e24,
        core_mass_fraction=0.325,
        mantle_melt_fraction=1.0,
        surface_radius=6.335439e6,
        temperature=2000.0,
        pressure=np.nan,
    )
    targets = {"H": 1.55e20, "C": 1.55e20, "N": 7.75e19, "S": 1.55e20}
    model.solve(
        state=planet,
        fugacity_constraints={"O2_g": IronWustiteBuffer(-2.0)},
        mass_constraints=targets,
        solver="basic",
        solver_parameters=SolverParameters(atol=1.0e-6, rtol=1.0e-4, max_steps=1024),
    )
    assert bool(np.squeeze(model.output.multi_attempt_solution.solver_success))

    params = model.output.parameters
    log_n_batch = jnp.asarray(model.output.log_number_moles)
    el_moles_gas_cond = np.squeeze(np.asarray(model.output.vmapf.get_element_moles(log_n_batch)))
    el_moles_melt = np.squeeze(
        np.asarray(model.output.vmapf.get_element_moles_in_melt(log_n_batch))
    )
    obj_res = np.squeeze(
        np.asarray(model.output.vmapf.objective_function(jnp.asarray(model.output.solution)))
    )
    assert np.all(np.abs(obj_res) <= SOLVER_RESIDUAL_TOL)
    assert np.all(el_moles_gas_cond > 0.0)
    assert np.all(el_moles_melt > 0.0)

    out_dict = model.output.asdict()
    unique_elements = params.species_network.unique_elements
    for idx, el in enumerate(unique_elements):
        el_mmw = Formula(el).mass * 1e-3
        expected_gas_cond_moles = 0.0
        expected_melt_moles = 0.0
        for sp_key, stoich in ATMODELLER_SPECIES_STOICH.items():
            if sp_key not in out_dict or el not in stoich:
                continue
            m_sp = sum(n * Formula(e).mass * 1e-3 for e, n in stoich.items())
            if sp_key.endswith("_cd"):
                sp_cond_kg = float(np.squeeze(out_dict[sp_key]["total_mass"]))
                expected_gas_cond_moles += (sp_cond_kg / m_sp) * stoich[el]
            else:
                sp_gas_kg = float(np.squeeze(out_dict[sp_key]["gas_mass"]))
                sp_liq_kg = float(np.squeeze(out_dict[sp_key]["dissolved_mass"]))
                expected_gas_cond_moles += (sp_gas_kg / m_sp) * stoich[el]
                expected_melt_moles += (sp_liq_kg / m_sp) * stoich[el]
        assert el_moles_gas_cond[idx] == pytest.approx(
            expected_gas_cond_moles, rel=1.0e-12, abs=0.0
        ), f"gas+cond moles mismatch for {el}"
        assert el_moles_melt[idx] == pytest.approx(expected_melt_moles, rel=1.0e-12, abs=0.0), (
            f"melt moles mismatch for {el}"
        )
        if el in targets:
            target_moles = targets[el] / el_mmw
            total_moles = el_moles_gas_cond[idx] + el_moles_melt[idx]
            rel_err = abs(total_moles / target_moles - 1.0)
            assert rel_err <= SOLVER_RESIDUAL_TOL, f"{el} relative closure error {rel_err}"
