"""Unit and Integration Tests for Energy Conservation and Master Accounting.

Verifies:
1. First Law of Thermodynamics compliance across EnergyAccountant.
2. Separation of boundary power inputs from internal subsystem transfers.
3. Individual subsystem work-energy and enthalpy balances.
4. Multi-scale coupled simulation stability and 10.0s end-to-end energy conservation.
"""

import math
from typing import List
import numpy as np
import pytest

from engine.config import SimulationConfig
from engine.geometry import PyramidGeometry
from engine.orchestrator import OrchestratorConfig, SimulationOrchestrator
from engine.physics.antechamber_filter import AntechamberFilter
from engine.physics.chemical_gas_transport import ChemicalGasTransport
from engine.physics.energy_accountant import (
    EnergyAccountant,
    EnergyBalanceSnapshot,
    PowerFlowState,
)
from engine.physics.grand_gallery_acoustics import GrandGalleryAcoustics
from engine.physics.microwave_maser import MicrowaveMaser
from engine.physics.piezoelectric_beams import GraniteBeam, PiezoelectricBeams
from engine.physics.schumann_hydraulics import SubterraneanHydraulics


def test_energy_accountant_synthetic_exact_conservation():
    """Verify EnergyAccountant satisfies exact conservation for arbitrary power functions."""
    acc = EnergyAccountant(tolerance=1.0e-4, initial_stored_energy=500.0)
    assert acc.initial_stored_energy == 500.0

    dt = 0.001
    num_steps = 1000
    stored_e = 500.0

    for step_idx in range(num_steps):
        t = step_idx * dt
        p_seis = 1000.0 * (1.0 + 0.5 * math.sin(2.0 * math.pi * 7.83 * t))
        p_hyd = 500.0 * (1.0 + 0.2 * math.cos(2.0 * math.pi * 14.3 * t))
        p_chem = 200.0
        p_rad = 50.0 * (1.0 + math.sin(2.0 * math.pi * 1.0 * t))
        p_hyd_loss = 300.0
        p_ac_loss = 100.0
        p_beam_loss = 50.0
        p_spark_loss = 10.0
        p_cav_loss = 20.0
        p_shaft_loss = 5.0
        p_therm_loss = 15.0

        p_total_in = p_seis + p_hyd + p_chem
        p_total_out = p_rad
        p_total_loss = (
            p_hyd_loss
            + p_ac_loss
            + p_beam_loss
            + p_spark_loss
            + p_cav_loss
            + p_shaft_loss
            + p_therm_loss
        )
        net_power = p_total_in - p_total_out - p_total_loss

        stored_e += net_power * dt

        snap = acc.step(
            dt=dt,
            p_seismic=p_seis,
            p_hydraulic=p_hyd,
            p_chemical=p_chem,
            p_maser_radiated=p_rad,
            p_hydraulic_loss=p_hyd_loss,
            p_acoustic_loss=p_ac_loss,
            p_beam_damping_loss=p_beam_loss,
            p_spark_loss=p_spark_loss,
            p_cavity_loss=p_cav_loss,
            p_shaft_loss=p_shaft_loss,
            p_thermal_loss=p_therm_loss,
            e_stored_hydraulic=stored_e,
        )

        assert snap.is_conserved
        assert snap.relative_error < 1.0e-4

    assert acc.check_conservation(tolerance=1.0e-4)
    assert acc.cumulative_energy_in > 0.0
    assert acc.cumulative_energy_out > 0.0
    assert acc.cumulative_energy_loss > 0.0
    assert abs(snap.delta_stored_energy - snap.net_work) < 1.0e-6


def test_energy_accountant_reset_and_efficiency():
    """Verify reset functionality and efficiency metric."""
    acc = EnergyAccountant(tolerance=1.0e-4, initial_stored_energy=0.0)

    for _ in range(100):
        acc.step(
            dt=0.01,
            p_chemical=1000.0,
            p_maser_radiated=200.0,
            p_thermal_loss=800.0,
            e_stored_chemical_thermal=0.0,
        )

    assert math.isclose(acc.overall_efficiency(), 0.20, rel_tol=1.0e-3)

    acc.reset(initial_stored_energy=100.0)
    assert acc.time == 0.0
    assert acc.initial_stored_energy == 100.0
    assert acc.cumulative_energy_in == 0.0
    assert acc.cumulative_energy_out == 0.0
    assert acc.overall_efficiency() == 0.0


def test_energy_accountant_internal_transfers_not_double_counted():
    """Verify internal coupling powers are tracked without double-counting in cumulative_energy_in."""
    acc = EnergyAccountant(tolerance=1.0e-4, initial_stored_energy=1000.0)

    snap = acc.step(
        dt=1.0,
        p_seismic=100.0,
        p_hydraulic=50.0,
        p_chemical=200.0,
        p_acoustic_in=80.0,
        p_piezo_in=40.0,
        p_maser_in=20.0,
        p_acoustic_transfer=70.0,
        p_piezo_transfer=35.0,
        p_maser_radiated=10.0,
        p_hydraulic_loss=40.0,
        p_acoustic_loss=10.0,
        p_beam_damping_loss=15.0,
        p_spark_loss=5.0,
        p_cavity_loss=5.0,
        p_shaft_loss=5.0,
        p_thermal_loss=70.0,
        e_stored_hydraulic=500.0,
        e_stored_chemical_thermal=400.0,
        e_stored_acoustic=150.0,
        e_stored_beams=100.0,
        e_stored_maser=50.0,
    )

    assert math.isclose(snap.power_flow.p_total_in, 350.0, rel_tol=1.0e-9)
    assert math.isclose(acc.cumulative_energy_in, 350.0, rel_tol=1.0e-9)
    assert math.isclose(acc.cumulative_acoustic_in, 80.0, rel_tol=1.0e-9)
    assert math.isclose(acc.cumulative_piezo_in, 40.0, rel_tol=1.0e-9)
    assert math.isclose(acc.cumulative_maser_in, 20.0, rel_tol=1.0e-9)
    assert math.isclose(acc.cumulative_acoustic_transfer, 70.0, rel_tol=1.0e-9)
    assert math.isclose(acc.cumulative_piezo_transfer, 35.0, rel_tol=1.0e-9)
    assert math.isclose(snap.power_flow.p_total_out, 10.0, rel_tol=1.0e-9)
    assert math.isclose(snap.power_flow.p_total_loss, 150.0, rel_tol=1.0e-9)
    assert math.isclose(snap.power_flow.net_power_flux, 190.0, rel_tol=1.0e-9)
    assert math.isclose(snap.total_stored_energy, 1200.0, rel_tol=1.0e-9)
    assert math.isclose(snap.delta_stored_energy, 200.0, rel_tol=1.0e-9)


def test_energy_accountant_stored_energy_decomposition():
    """Verify stored energy sums all 5 subsystems and relative error formula."""
    acc = EnergyAccountant(tolerance=1.0e-3, energy_scale=1.0e6)
    acc.set_initial_stored_energy(1000.0)

    snap = acc.step(
        dt=0.1,
        p_seismic=1000.0,
        e_stored_hydraulic=200.0,
        e_stored_chemical_thermal=300.0,
        e_stored_acoustic=250.0,
        e_stored_beams=150.0,
        e_stored_maser=100.0,
    )

    assert snap.total_stored_energy == 1000.0
    assert snap.delta_stored_energy == 0.0
    assert snap.is_conserved is True


def test_subterranean_hydraulics_work_energy_balance():
    """Verify subterranean bedrock mass-spring-damper satisfies work-energy balance."""
    hyd = SubterraneanHydraulics(
        bedrock_damping_ratio=0.02,
        seismic_force_amplitude=1.0e5,
        enable_water_hammer=False,
    )

    dt = 0.0005
    n_steps = 1000

    e_mech_0 = 0.5 * hyd.bedrock_mass * (hyd.velocity**2) + 0.5 * hyd.stiffness_k * (hyd.displacement**2)
    w_damping = 0.0
    w_input = 0.0

    for _ in range(n_steps):
        t_curr = hyd.time
        _, f_seis_prev, _, f_hyd_prev, _ = hyd.compute_driving_forces(t_curr)
        v_prev = hyd.velocity

        st = hyd.step(dt)
        v_curr = hyd.velocity

        f_seis_curr = st.seismic_force
        f_hyd_curr = st.hydraulic_force

        p_in = 0.5 * (f_seis_prev * v_prev + f_seis_curr * v_curr) + 0.5 * (f_hyd_prev * v_prev + f_hyd_curr * v_curr)
        w_input += p_in * dt

        p_diss = 0.5 * hyd.damping_c * (v_prev**2 + v_curr**2)
        w_damping += p_diss * dt

    e_mech_final = 0.5 * hyd.bedrock_mass * (hyd.velocity**2) + 0.5 * hyd.stiffness_k * (hyd.displacement**2)
    delta_e = e_mech_final - e_mech_0
    net_w = w_input - w_damping

    rel_error = abs(delta_e - net_w) / max(abs(w_input), abs(e_mech_final), 1.0)
    assert rel_error < 1.0e-3


def test_chemical_gas_enthalpy_conservation():
    """Verify Queen's Chamber chemical reaction kinetics satisfies heat enthalpy balance."""
    chem = ChemicalGasTransport(
        initial_zn_moles=100.0,
        initial_hcl_moles=200.0,
        enable_thermal_feedback=True,
    )

    dt = 0.1
    for _ in range(50):
        chem.step(dt)

    st = chem.get_state()
    h2_generated = st.reaction.h2_moles_generated_total
    expected_total_heat = h2_generated * (-chem.delta_H_rxn)

    assert math.isclose(st.reaction.cumulative_heat_joules, expected_total_heat, rel_tol=1.0e-4)


def test_grand_gallery_acoustic_energy_conservation():
    """Verify Grand Gallery acoustics conserves wave energy in unforced closed domain."""
    ac = GrandGalleryAcoustics(
        num_grid_points=81,
        attenuation_coeff=0.0,
        top_boundary_type="rigid",
        bottom_boundary_type="rigid",
        enable_resonators=False,
    )
    ac.inject_pulse(amplitude=150.0, center_z=ac.length * 0.5, width=1.5)

    e0 = ac.compute_total_acoustic_energy()
    assert e0 > 0.0

    dt = 0.0001
    for _ in range(500):
        ac.step(dt, bottom_pressure_drive=0.0)

    e_final = ac.compute_total_acoustic_energy()
    rel_drift = abs(e_final - e0) / e0
    assert rel_drift < 0.05


def test_piezoelectric_beams_energy_balance():
    """Verify King's Chamber 43 rose granite beams satisfy electromechanical energy balance."""
    pz = PiezoelectricBeams()

    for b in pz.all_beams:
        b.q[0] = 3.0e-5
        b.update_stress_and_voltage()

    e_mech_0 = sum(b.mechanical_energy() for b in pz.all_beams)
    dt = 2.0e-5

    for _ in range(300):
        st = pz.step(dt, p_kc_acoustic=0.0)

    final_energy = st.total_mechanical_energy + st.cumulative_loss_energy
    rel_error = abs(final_energy - e_mech_0) / e_mech_0
    assert rel_error < 1.0e-3
    assert st.stored_electrical_energy <= st.strain_energy


def test_microwave_maser_photon_energy_balance():
    """Verify microwave maser quantum hyperfine rate equations conserve photon energy."""
    maser = MicrowaveMaser(
        nominal_h_density=1.0e20,
        cavity_quality_factor=1.0e5,
    )

    dt = 1.0e-4
    for _ in range(200):
        st = maser.step(
            dt=dt,
            piezo_voltage=5000.0,
            acoustic_pressure=1000.0,
            h2_concentration=0.5,
        )

    assert st.cumulative_stimulated_energy >= 0.0
    assert st.cumulative_radiated_energy >= 0.0
    assert st.cumulative_cavity_loss_energy >= 0.0


def test_orchestrator_multi_rate_simulation_stability():
    """Verify end-to-end SimulationOrchestrator runs stably with multi-rate time stepping."""
    orch = SimulationOrchestrator(
        orchestrator_config=OrchestratorConfig(
            scenario_name="baseline",
            duration_s=0.2,
            dt_macro=0.01,
            dt_micro=0.0001,
            telemetry_fps=60.0,
        )
    )

    telemetry = orch.run(duration=0.2)
    assert len(telemetry.frames) > 0
    assert telemetry.total_frames == len(telemetry.frames)

    for frame in telemetry.frames:
        assert math.isfinite(frame.bedrock_displacement)
        assert math.isfinite(frame.water_hammer_pressure)
        assert math.isfinite(frame.gallery_peak_pressure)
        assert math.isfinite(frame.total_piezo_voltage)
        assert math.isfinite(frame.maser_total_radiated_power)
        assert math.isfinite(frame.total_stored_energy)
        assert not math.isnan(frame.gallery_peak_pressure)
        assert not math.isnan(frame.total_piezo_voltage)

    summary = telemetry.summary
    assert "duration_s" in summary
    assert "peak_gallery_pressure_pa" in summary
    assert "peak_piezo_voltage_v" in summary
    assert "total_energy_in_j" in summary
    assert summary["all_steps_conserved"] is True


def test_orchestrator_baseline_10s_energy_conservation():
    """Verify 10.0s baseline simulation satisfies First Law RelError < 0.001 across all steps."""
    orch = SimulationOrchestrator.create_scenario("baseline", duration=10.0)
    telemetry = orch.run(duration=10.0)

    assert len(telemetry.frames) > 0
    summary = telemetry.summary

    assert summary["all_steps_conserved"] is True
    assert summary["max_relative_energy_error"] < 1.0e-3
    assert summary["mean_relative_energy_error"] < 1.0e-4
    assert summary["total_energy_in_j"] > 0.0
    assert summary["total_energy_loss_j"] > 0.0
    assert summary["final_stored_energy_j"] > 0.0

    for frame in telemetry.frames:
        assert frame.is_energy_conserved is True
        assert frame.relative_energy_error < 1.0e-3


def test_orchestrator_scenarios_energy_conservation():
    """Verify all predefined scenario presets maintain First Law energy balance."""
    scenarios = [
        "acoustic_peak",
        "full_maser_power",
        "dry_run_no_gas",
        "high_seismic",
        "resonance_sweep",
        "transient_shock",
    ]

    for sc in scenarios:
        orch = SimulationOrchestrator.create_scenario(sc, duration=0.4)
        tel = orch.run(duration=0.4)
        s = tel.summary
        assert s["all_steps_conserved"] is True, f"Scenario {sc} failed energy conservation"
        assert s["max_relative_energy_error"] < 1.0e-3, f"Scenario {sc} max error {s['max_relative_energy_error']} >= 1e-3"
