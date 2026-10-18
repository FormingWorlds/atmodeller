"""Cross-code thermodynamic and gas-only equilibrium consistency tests.

Verifies:
1. atmodeller standard-state Gibbs reaction equilibrium constants (get_log_Kp in
   atmodeller/engine.py:276-299) and gas-only equilibrium reaction quotients against
   CALLIOPE's JANAF and Schaefer & Fegley (2004) analytical fits across
   T in {1500, 2000, 2500, 3000} K within the propagated reaction-data tolerances.
2. Live 72-case gas-only equilibrium cross-comparison against CALLIOPE (60 buffered
   cases with matched fO2 plus 12 authoritative-O cases) on the shared volatile
   species, gated via pytest.importorskip('calliope') when run outside the coupled
   environment.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import scipy.optimize as opt

from atmodeller import (
    ChemicalSpecies,
    EquilibriumModel,
    Planet,
    SpeciesNetwork,
)
from atmodeller.containers import (
    FixedFugacityConstraint,
    Parameters,
    SolverParameters,
)
from atmodeller.engine import get_log_Kp
from atmodeller.thermodata import IronWustiteBuffer

pytestmark = pytest.mark.integration

# Per-species relative tolerances derived from the JANAF/Schaefer vs NASA Glenn
# thermodynamic data spread (with stoichiometric power 0.5 for doubled reactions SO2 and NH3).
REACTION_QUOTIENT_RTOL: dict[str, float] = {
    "H2": 2.0e-2,
    "CO": 4.0e-2,
    "SO2": 2.5e-2,
    "NH3": 1.8e-1,
    "CH4": 2.8e-1,
}


def _calliope_log10_k_ref(t_k: float) -> dict[str, float]:
    """Evaluate CALLIOPE JANAF and Schaefer & Fegley (2004) log10(K) formulas (src/calliope/chemistry.py)."""
    return {
        "janaf_H2": -13152.47778038 / t_k + 3.03859429,
        "janaf_CO": -14467.51139876 / t_k + 4.34813634,
        "schaefer_CH4": -16276.0 / t_k - 5.4738,
        "janaf_SO2": 37774.0 / t_k - 7.6128,
        "janaf_H2S": 13462.03094 / t_k - 7.24546062,
        "janaf_NH3": 5328.03125 / t_k - 11.98476042,
    }


def test_atmodeller_get_log_kp_and_gas_only_speciation_vs_janaf_schaefer():
    """Verify atmodeller get_log_Kp and gas-only reaction quotients against CALLIOPE JANAF/Schaefer fits."""
    species_names = (
        "H2O",
        "H2",
        "O2",
        "CO2",
        "CO",
        "CH4",
        "N2",
        "H3N",
        "S2",
        "O2S",
        "H2S",
    )
    gas_species = tuple(ChemicalSpecies.create_gas(sp) for sp in species_names)
    network = SpeciesNetwork(gas_species)
    model = EquilibriumModel(network)

    t_grid = np.array([1500.0, 2000.0, 2500.0, 3000.0], dtype=float)
    n_batch = len(t_grid)
    planet = Planet(
        planet_mass=np.full(n_batch, 5.972e24),
        core_mass_fraction=np.full(n_batch, 0.325),
        mantle_melt_fraction=np.zeros(n_batch),
        surface_radius=np.full(n_batch, 6.335439e6),
        temperature=t_grid,
        pressure=np.full(n_batch, np.nan),
    )
    targets = {
        "H": np.full(n_batch, 1.55e20),
        "C": np.full(n_batch, 1.55e20),
        "N": np.full(n_batch, 7.75e19),
        "S": np.full(n_batch, 1.55e20),
    }
    model.solve(
        state=planet,
        fugacity_constraints={"O2_g": IronWustiteBuffer(np.zeros(n_batch))},
        mass_constraints=targets,
        solver="robust",
        solver_parameters=SolverParameters(
            atol=1.0e-6, rtol=1.0e-4, max_steps=1024, multistart=10
        ),
    )

    # 1. Verify get_log_Kp directly against reaction_matrix @ (-G_0 / RT) and reaction quotients ln Q_r
    rxn_mat = np.asarray(model.output.parameters.species_network.reaction_matrix)
    asdict = model.output.asdict()
    sp_order = tuple(sp.name for sp in model.output.species)
    log_act_out = np.asarray(model.output.log_activity(), dtype=float)

    for idx, t_k in enumerate(t_grid):
        params_t = Parameters.create(
            species_network=network,
            state=Planet(temperature=float(t_k)),
        )
        code_log_kp = np.asarray(get_log_Kp(params_t), dtype=float)
        gibbs_over_rt = np.array(
            [
                float(np.squeeze(sp.data.get_gibbs_over_RT(params_t.state.temperature)))
                for sp in network
            ],
            dtype=float,
        )
        expected_log_kp = -1.0 * (rxn_mat @ gibbs_over_rt)
        assert np.allclose(code_log_kp, expected_log_kp, rtol=1.0e-10, atol=1.0e-10)

        log_act = np.array([math.log(float(asdict[sp]["fugacity"][idx])) for sp in sp_order])
        assert np.allclose(log_act, log_act_out[idx], rtol=1.0e-12, atol=1.0e-12)
        ln_q = rxn_mat @ log_act
        assert np.allclose(ln_q, code_log_kp, rtol=0.0, atol=1.0e-6)

        p_h2o = float(asdict["H2O_g"]["pressure"][idx])
        p_h2 = float(asdict["H2_g"]["pressure"][idx])
        p_o2 = float(asdict["O2_g"]["pressure"][idx])
        p_co2 = float(asdict["CO2_g"]["pressure"][idx])
        p_co = float(asdict["CO_g"]["pressure"][idx])
        p_ch4 = float(asdict["CH4_g"]["pressure"][idx])
        p_n2 = float(asdict["N2_g"]["pressure"][idx])
        p_nh3 = float(asdict["H3N_g"]["pressure"][idx])
        p_s2 = float(asdict["S2_g"]["pressure"][idx])
        p_so2 = float(asdict["O2S_g"]["pressure"][idx])

        q_h2 = p_h2 * (p_o2**0.5) / p_h2o
        q_co = p_co * (p_o2**0.5) / p_co2
        q_so2 = p_so2 / ((p_s2**0.5) * p_o2)
        q_nh3 = p_nh3 / ((p_n2**0.5) * (p_h2**1.5))
        q_ch4 = p_ch4 * p_o2 / (p_co2 * (p_h2**2))

        cal_ref = _calliope_log10_k_ref(float(t_k))
        assert q_h2 == pytest.approx(10.0 ** cal_ref["janaf_H2"], rel=REACTION_QUOTIENT_RTOL["H2"])
        assert q_co == pytest.approx(10.0 ** cal_ref["janaf_CO"], rel=REACTION_QUOTIENT_RTOL["CO"])
        assert q_so2 == pytest.approx(
            10.0 ** (0.5 * cal_ref["janaf_SO2"]), rel=REACTION_QUOTIENT_RTOL["SO2"]
        )
        assert q_nh3 == pytest.approx(
            10.0 ** (0.5 * cal_ref["janaf_NH3"]), rel=REACTION_QUOTIENT_RTOL["NH3"]
        )
        assert q_ch4 == pytest.approx(
            10.0 ** cal_ref["schaefer_CH4"], rel=REACTION_QUOTIENT_RTOL["CH4"]
        )


def test_atmodeller_vs_calliope_72_case_gas_only_grid():
    """Verify atmodeller vs CALLIOPE cross-code agreement over all 72 gas-only grid cases."""
    pytest.importorskip("calliope")
    from calliope.constants import molar_mass, noble_gases, ocean_moles
    from calliope.oxygen_fugacity import OxygenFugacity
    from calliope.solve import (
        equilibrium_atmosphere,
        equilibrium_atmosphere_authoritative_O,
        func,
        func_authoritative_O,
    )

    mass_ocean = float(ocean_moles * molar_mass["H2"])
    m_earth = 5.972e24
    r_earth = 6.335439e6
    cmf = 0.325
    m_mantle = (1.0 - cmf) * m_earth
    g_surf = 9.81

    inventories: dict[str, dict[str, float]] = {}
    for label, scale in (("low", 0.1), ("default", 1.0), ("high", 10.0)):
        h_kg = scale * mass_ocean
        c_kg = 1.0 * h_kg
        n_kg = 0.5 * h_kg
        s_kg = 1.0 * h_kg
        o_stoich_kg = (
            h_kg * (molar_mass["O"] / molar_mass["H2"])
            + c_kg * (2.0 * molar_mass["O"] / molar_mass["C"])
            + s_kg * (2.0 * molar_mass["O"] / molar_mass["S"])
        )
        inventories[label] = {
            "H": h_kg,
            "C": c_kg,
            "N": n_kg,
            "S": s_kg,
            "O": o_stoich_kg,
        }

    buffered_cases: list[dict[str, Any]] = []
    for inv_label in ("low", "default", "high"):
        inv = inventories[inv_label]
        for t_k in (1500.0, 2000.0, 2500.0, 3000.0):
            for fo2 in (-4.0, -2.0, 0.0, 2.0, 4.0):
                buffered_cases.append(
                    {
                        "case_id": f"buffered_{inv_label}_T{int(t_k)}_IW{fo2:+.0f}",
                        "T_K": t_k,
                        "fO2_shift_IW": fo2,
                        "targets": {k: inv[k] for k in ("H", "C", "N", "S")},
                    }
                )
    autho_cases: list[dict[str, Any]] = []
    for inv_label in ("low", "default", "high"):
        inv = inventories[inv_label]
        for t_k in (1500.0, 2000.0, 2500.0, 3000.0):
            autho_cases.append(
                {
                    "case_id": f"authO_{inv_label}_T{int(t_k)}",
                    "T_K": t_k,
                    "fO2_shift_IW": 2.0,
                    "targets": {k: inv[k] for k in ("H", "C", "N", "S", "O")},
                }
            )

    species_names = (
        "H2O",
        "H2",
        "O2",
        "CO2",
        "CO",
        "CH4",
        "N2",
        "H3N",
        "S2",
        "O2S",
        "H2S",
    )
    gas_species = tuple(ChemicalSpecies.create_gas(sp) for sp in species_names)
    solver_params = SolverParameters(atol=1.0e-6, rtol=1.0e-4, max_steps=1024, multistart=10)
    fo2_fischer = OxygenFugacity()

    n_buf = len(buffered_cases)
    planet_buf = Planet(
        planet_mass=np.full(n_buf, m_earth),
        core_mass_fraction=np.full(n_buf, cmf),
        mantle_melt_fraction=np.zeros(n_buf),
        surface_radius=np.full(n_buf, r_earth),
        temperature=np.array([c["T_K"] for c in buffered_cases]),
        pressure=np.full(n_buf, np.nan),
    )
    f_o2_arr = np.array(
        [float(10.0 ** fo2_fischer(c["T_K"], c["fO2_shift_IW"])) for c in buffered_cases]
    )
    mdl_buf = EquilibriumModel(SpeciesNetwork(gas_species))
    mdl_buf.solve(
        state=planet_buf,
        fugacity_constraints={"O2_g": FixedFugacityConstraint(f_o2_arr)},
        mass_constraints={
            el: np.array([c["targets"][el] for c in buffered_cases]) for el in ("H", "C", "N", "S")
        },
        solver="robust",
        solver_parameters=solver_params,
    )
    asdict_buf = mdl_buf.output.asdict()
    ptot_buf = np.asarray(mdl_buf.output.total_pressure())

    n_auth = len(autho_cases)
    planet_auth = Planet(
        planet_mass=np.full(n_auth, m_earth),
        core_mass_fraction=np.full(n_auth, cmf),
        mantle_melt_fraction=np.zeros(n_auth),
        surface_radius=np.full(n_auth, r_earth),
        temperature=np.array([c["T_K"] for c in autho_cases]),
        pressure=np.full(n_auth, np.nan),
    )
    mdl_auth = EquilibriumModel(SpeciesNetwork(gas_species))
    mdl_auth.solve(
        state=planet_auth,
        fugacity_constraints=None,
        mass_constraints={
            el: np.array([c["targets"][el] for c in autho_cases])
            for el in ("H", "C", "N", "S", "O")
        },
        solver="robust",
        solver_parameters=solver_params,
    )
    asdict_auth = mdl_auth.output.asdict()
    ptot_auth = np.asarray(mdl_auth.output.total_pressure())

    def solve_cal(
        target: dict[str, float], t_k: float, fo2_shift: float, autho: bool
    ) -> dict[str, float]:
        ddict: dict[str, float | int] = {
            "M_mantle": m_mantle,
            "gravity": g_surf,
            "radius": r_earth,
            "Phi_global": 0.0,
            "T_magma": t_k,
            "fO2_shift_IW": fo2_shift,
        }
        for sp in (
            "H2O",
            "CO2",
            "N2",
            "S2",
            "SO2",
            "H2S",
            "NH3",
            "H2",
            "CH4",
            "CO",
            "O2",
        ):
            ddict[f"{sp}_included"] = 1
            ddict[f"{sp}_initial_bar"] = 0.0
        for ng in noble_gases:
            ddict[f"{ng}_included"] = 0
            ddict[f"{ng}_initial_bar"] = 0.0
            ddict[f"{ng}_ppmw"] = 0.0
        scales = np.array(
            [target[k] for k in (("H", "C", "N", "S", "O") if autho else ("H", "C", "N", "S"))]
        )
        if autho:

            def f_log_a(y: np.ndarray) -> np.ndarray:
                yc = np.clip(y[:4], -80.0, 14.0)
                fo2_c = float(np.clip(y[4], -11.99, 11.99))
                x = [
                    float(np.exp(yc[0])),
                    float(np.exp(yc[1])),
                    float(np.exp(yc[2])),
                    float(np.exp(yc[3])),
                    fo2_c,
                ]
                return np.asarray(func_authoritative_O(x, ddict, target)) / scales

            res = opt.least_squares(
                f_log_a,
                [5.0, 4.0, 2.0, -20.0, 6.0],
                bounds=(
                    [-80.0, -80.0, -80.0, -80.0, -11.99],
                    [14.0, 14.0, 14.0, 14.0, 11.99],
                ),
                ftol=1.0e-14,
                xtol=1.0e-14,
                gtol=1.0e-14,
            )
            p_g = {
                "H2O": float(np.exp(res.x[0])),
                "CO2": float(np.exp(res.x[1])),
                "N2": float(np.exp(res.x[2])),
                "S2": float(np.exp(res.x[3])),
            }
            return equilibrium_atmosphere_authoritative_O(
                target,
                ddict,
                fO2_hint=float(res.x[4]),
                p_guess=p_g,
                rtol=1.0e-5,
                atol=1.0e10,
                xtol=1.0e-8,
                nguess=5,
                opt_solver=False,
                random_seed=42,
                print_result=False,
            )

        y0 = [4.0, 3.0, 1.0, -15.0 if fo2_shift > 0 else 1.0]

        def f_log_b(y: np.ndarray) -> np.ndarray:
            yc = np.clip(y, -80.0, 14.0)
            x = [
                float(np.exp(yc[0])),
                float(np.exp(yc[1])),
                float(np.exp(yc[2])),
                float(np.exp(yc[3])),
            ]
            return np.asarray(func(x, ddict, target)) / scales

        res = opt.least_squares(
            f_log_b,
            y0,
            bounds=([-80.0] * 4, [14.0] * 4),
            ftol=1.0e-14,
            xtol=1.0e-14,
            gtol=1.0e-14,
        )
        p_g = {
            "H2O": float(np.exp(res.x[0])),
            "CO2": float(np.exp(res.x[1])),
            "N2": float(np.exp(res.x[2])),
            "S2": float(np.exp(res.x[3])),
        }
        return equilibrium_atmosphere(
            target,
            ddict,
            p_guess=p_g,
            rtol=1.0e-5,
            atol=1.0e10,
            xtol=1.0e-8,
            nguess=5,
            opt_solver=False,
            print_result=False,
        )

    for idx, case in enumerate(buffered_cases):
        cal_out = solve_cal(case["targets"], case["T_K"], case["fO2_shift_IW"], autho=False)
        assert float(ptot_buf[idx]) == pytest.approx(float(cal_out["P_surf"]), rel=3.5e-2)
        assert float(asdict_buf["O2_g"]["pressure"][idx]) == pytest.approx(
            float(cal_out["O2_bar"]), rel=1.0e-3
        )
        assert float(asdict_buf["H2O_g"]["pressure"][idx]) == pytest.approx(
            float(cal_out["H2O_bar"]), rel=5.0e-2
        )
        assert float(asdict_buf["N2_g"]["pressure"][idx]) == pytest.approx(
            float(cal_out["N2_bar"]), rel=5.0e-2
        )
        assert float(asdict_buf["H2_g"]["pressure"][idx]) == pytest.approx(
            float(cal_out["H2_bar"]), rel=5.0e-2
        )
        assert float(asdict_buf["H3N_g"]["pressure"][idx]) == pytest.approx(
            float(cal_out["NH3_bar"]), rel=1.8e-1
        )
        assert float(asdict_buf["CH4_g"]["pressure"][idx]) == pytest.approx(
            float(cal_out["CH4_bar"]), rel=2.8e-1
        )
        c_rtol = 5.0e-2 if case["fO2_shift_IW"] >= 2.0 else 2.0e-1
        assert float(asdict_buf["CO2_g"]["pressure"][idx]) == pytest.approx(
            float(cal_out["CO2_bar"]), rel=c_rtol
        )
        assert float(asdict_buf["CO_g"]["pressure"][idx]) == pytest.approx(
            float(cal_out["CO_bar"]), rel=c_rtol
        )

    for idx, case in enumerate(autho_cases):
        cal_out = solve_cal(case["targets"], case["T_K"], case["fO2_shift_IW"], autho=True)
        assert float(ptot_auth[idx]) == pytest.approx(float(cal_out["P_surf"]), rel=3.5e-2)
        assert float(asdict_auth["H2O_g"]["pressure"][idx]) == pytest.approx(
            float(cal_out["H2O_bar"]), rel=5.0e-2
        )
        assert float(asdict_auth["CO2_g"]["pressure"][idx]) == pytest.approx(
            float(cal_out["CO2_bar"]), rel=5.0e-2
        )
        assert float(asdict_auth["N2_g"]["pressure"][idx]) == pytest.approx(
            float(cal_out["N2_bar"]), rel=5.0e-2
        )
        assert float(asdict_auth["O2S_g"]["pressure"][idx]) == pytest.approx(
            float(cal_out["SO2_bar"]), rel=2.5e-2
        )
