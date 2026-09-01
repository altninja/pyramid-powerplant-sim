"""Comprehensive test suite for Rose Granite Piezoelectric Transduction Module.

Tests:
1. Beam natural bending frequency verification against analytical Euler-Bernoulli theory.
2. Piezoelectric linear open-circuit voltage response vs acoustic driving pressure.
3. Collective 5-tier voltage array stacking and multi-kilovolt potential generation.
4. Electromechanical energy conservation and strain-to-electrostatic conversion audit.
5. Coffer spark gap ionization breakdown threshold and energy discharge.
6. Geometric and mass consistency across all 43 monolithic granite beams.
"""

import math
import numpy as np
import pytest

from engine.config import VACUUM_PERMITTIVITY_EPS0
from engine.physics.piezoelectric_beams import (
    CLAMPED_CLAMPED_BETA_L,
    TIER_BEAM_COUNTS,
    TIER_NAMES,
    GraniteBeam,
    PiezoelectricBeams,
)


def test_granite_beam_geometry_and_mass() -> None:
    """Verify single beam physical dimensions, mass, second moment of area, and capacitance."""
    beam = GraniteBeam(
        length=6.50,
        width=1.20,
        depth=1.50,
        density=2650.0,
        youngs_modulus=55.0e9,
        dielectric_permittivity=6.2,
    )

    expected_area = 1.20 * 1.50
    expected_i = 1.20 * (1.50 ** 3) / 12.0
    expected_mass = 2650.0 * expected_area * 6.50
    expected_c = (6.2 * VACUUM_PERMITTIVITY_EPS0 * (6.50 * 1.20)) / 1.50

    assert np.isclose(beam.cross_section_area, expected_area)
    assert np.isclose(beam.second_moment_area, expected_i)
    assert np.isclose(beam.mass, expected_mass)
    assert np.isclose(beam.capacitance, expected_c)
    assert beam.mass == pytest.approx(31005.0, rel=1.0e-4)


def test_beam_natural_bending_frequencies() -> None:
    """Verify mode 1-4 natural bending frequencies match Euler-Bernoulli analytical formula within 0.5%."""
    beam = GraniteBeam(
        length=6.50,
        width=1.20,
        depth=1.50,
        density=2650.0,
        youngs_modulus=55.0e9,
        num_modes=4,
    )

    e = 55.0e9
    i_moment = 0.3375
    rho = 2650.0
    area = 1.80
    span = 6.50

    wave_speed_factor = math.sqrt((e * i_moment) / (rho * area))

    analytical_omegas = [
        ((beta_l / span) ** 2) * wave_speed_factor
        for beta_l in CLAMPED_CLAMPED_BETA_L[:4]
    ]
    analytical_freqs_hz = [w / (2.0 * math.pi) for w in analytical_omegas]

    calc_freqs_hz = beam.natural_frequencies()

    for n in range(4):
        rel_error = abs(calc_freqs_hz[n] - analytical_freqs_hz[n]) / analytical_freqs_hz[n]
        assert rel_error < 0.005, f"Mode {n+1} frequency mismatch: {calc_freqs_hz[n]} vs {analytical_freqs_hz[n]}"

    theoretical_omega1 = ((4.73004074 / span) ** 2) * wave_speed_factor
    assert np.isclose(beam.omegas[0], theoretical_omega1, rtol=1.0e-3)


def test_dynamic_free_vibration_ringdown_frequency() -> None:
    """Verify numerical dynamic time-stepping reproduces fundamental oscillation frequency."""
    beam = GraniteBeam(
        length=6.50,
        width=1.20,
        depth=1.50,
        density=2650.0,
        youngs_modulus=55.0e9,
        quality_factor=1000.0,
    )

    f1_theor = beam.natural_frequencies()[0]
    period1 = 1.0 / f1_theor

    beam.q[0] = 1.0e-4
    beam.dq_dt.fill(0.0)

    dt = 1.0e-5
    sim_time = period1 * 10.0
    n_steps = int(sim_time / dt)

    zero_crossings = []
    prev_q = beam.q[0]
    zero_force = np.zeros(beam.num_modes, dtype=float)

    for step in range(n_steps):
        beam.step_verlet(dt, zero_force)
        curr_q = beam.q[0]
        if prev_q < 0.0 and curr_q >= 0.0:
            frac = -prev_q / (curr_q - prev_q)
            t_cross = (step - 1 + frac) * dt
            zero_crossings.append(t_cross)
        prev_q = curr_q

    assert len(zero_crossings) >= 8
    periods = np.diff(zero_crossings)
    mean_period = float(np.mean(periods))
    sim_freq = 1.0 / mean_period

    rel_error = abs(sim_freq - f1_theor) / f1_theor
    assert rel_error < 0.005


def test_piezoelectric_linear_voltage_response() -> None:
    """Verify open-circuit voltage scales linearly with acoustic driving pressure."""
    pressures = [100.0, 250.0, 500.0, 1000.0, 2000.0]
    peak_voltages = []

    dt = 1.0e-4
    n_steps = 100

    for p in pressures:
        beams_sys = PiezoelectricBeams()
        for _ in range(n_steps):
            state = beams_sys.step(dt, p_kc_acoustic=p)
        peak_voltages.append(state.total_voltage)

    ratios = [v / p for v, p in zip(peak_voltages, pressures)]
    mean_ratio = float(np.mean(ratios))

    for r in ratios:
        assert np.isclose(r, mean_ratio, rtol=1.0e-2)


def test_array_stacking_multi_kilovolt_resonant_response() -> None:
    """Verify 5-tier voltage aggregation produces multi-kilovolt potential under resonant acoustic excitation (> 1 kPa)."""
    beams_sys = PiezoelectricBeams(breakdown_voltage=100000.0)
    sample_beam = beams_sys.all_beams[0]
    f_res = sample_beam.natural_frequencies()[0]

    p0 = 1500.0
    dt = 5.0e-5
    duration = 0.05
    n_steps = int(duration / dt)

    max_total_voltage = 0.0
    tier_max_voltages = [0.0] * 5

    for step in range(n_steps):
        t = step * dt
        p_drive = p0 * math.sin(2.0 * math.pi * f_res * t)
        state = beams_sys.step(dt, p_kc_acoustic=p_drive)
        
        max_total_voltage = max(max_total_voltage, abs(state.total_voltage))
        for k, v_k in enumerate(state.tier_voltages):
            tier_max_voltages[k] = max(tier_max_voltages[k], abs(v_k))

    assert max_total_voltage > 1000.0, f"Total voltage {max_total_voltage} V failed to exceed 1 kV"
    assert np.isclose(sum(state.tier_voltages), state.total_voltage)
    assert len(state.tier_voltages) == 5


def test_energy_conservation_audit() -> None:
    """Verify electrostatic energy <= strain energy and total energy is conserved in free vibration."""
    beam = GraniteBeam(
        length=6.50,
        width=1.20,
        depth=1.50,
        density=2650.0,
        youngs_modulus=55.0e9,
        quality_factor=1000.0,
    )

    beam.q[0] = 5.0e-5
    beam.dq_dt.fill(0.0)
    beam.update_stress_and_voltage()

    initial_mech_energy = beam.mechanical_energy()
    initial_elec_energy = beam.electrical_energy()

    assert initial_elec_energy < initial_mech_energy
    assert initial_elec_energy > 0.0

    beams_sys = PiezoelectricBeams()
    for b in beams_sys.all_beams:
        b.q[0] = 2.0e-5
        b.update_stress_and_voltage()

    e_mech_0 = sum(b.mechanical_energy() for b in beams_sys.all_beams)

    dt = 2.0e-5
    n_steps = 500

    for _ in range(n_steps):
        state = beams_sys.step(dt, p_kc_acoustic=0.0)

    final_e_mech = state.total_mechanical_energy
    dissipated_loss = state.cumulative_loss_energy

    total_accounted_energy = final_e_mech + dissipated_loss
    rel_energy_drift = abs(total_accounted_energy - e_mech_0) / e_mech_0

    assert rel_energy_drift < 1.0e-3
    assert state.stored_electrical_energy <= state.strain_energy


def test_coffer_spark_breakdown_threshold() -> None:
    """Verify dielectric discharge triggers when voltage exceeds threshold and produces ionization pulse."""
    v_breakdown = 20000.0
    beams_sys = PiezoelectricBeams(breakdown_voltage=v_breakdown)

    sample_beam = beams_sys.all_beams[0]
    f_res = sample_beam.natural_frequencies()[0]

    p0 = 5000.0
    dt = 5.0e-5
    duration = 0.08
    n_steps = int(duration / dt)

    spark_detected = False
    max_spark_energy = 0.0
    max_ion_density = 0.0

    for step in range(n_steps):
        t = step * dt
        p_drive = p0 * math.sin(2.0 * math.pi * f_res * t)
        state = beams_sys.step(dt, p_kc_acoustic=p_drive)

        if state.spark_triggered:
            spark_detected = True
            max_spark_energy = max(max_spark_energy, state.spark_energy)
            max_ion_density = max(max_ion_density, state.ion_density)

    assert spark_detected, "Spark breakdown was not triggered under strong excitation"
    assert beams_sys.spark_count >= 1
    assert max_spark_energy > 0.0
    assert max_ion_density > 0.0
    assert state.cumulative_spark_energy > 0.0


def test_total_beam_count_and_tier_structure() -> None:
    """Verify the 43 beams are correctly distributed across the 5 relieving tiers."""
    beams_sys = PiezoelectricBeams()

    assert beams_sys.total_beam_count == 43
    assert len(beams_sys.tiers) == 5

    for k, (tier, expected_count, expected_name) in enumerate(
        zip(beams_sys.tiers, TIER_BEAM_COUNTS, TIER_NAMES)
    ):
        assert len(tier) == expected_count
        for b in tier:
            assert b.tier_index == k
            assert b.tier_name == expected_name


def test_rk4_and_verlet_integrators_consistency() -> None:
    """Verify Verlet and RK4 integrators produce matching trajectories."""
    sys_verlet = PiezoelectricBeams(integrator="verlet")
    sys_rk4 = PiezoelectricBeams(integrator="rk4")

    for b_v, b_r in zip(sys_verlet.all_beams, sys_rk4.all_beams):
        b_v.q[0] = 1.0e-5
        b_r.q[0] = 1.0e-5
        b_v.update_stress_and_voltage()
        b_r.update_stress_and_voltage()

    dt = 1.0e-4
    for _ in range(50):
        s_v = sys_verlet.step(dt, p_kc_acoustic=100.0)
        s_r = sys_rk4.step(dt, p_kc_acoustic=100.0)

    assert np.isclose(s_v.total_voltage, s_r.total_voltage, rtol=1.0e-2)
    assert np.isclose(s_v.total_mechanical_energy, s_r.total_mechanical_energy, rtol=1.0e-2)


def test_reset_functionality() -> None:
    """Verify reset restores all state variables to equilibrium zero."""
    beams_sys = PiezoelectricBeams()
    dt = 1.0e-4
    beams_sys.step(dt, p_kc_acoustic=500.0)

    assert beams_sys.time > 0.0
    assert abs(beams_sys.compute_total_voltage()) > 0.0

    beams_sys.reset()

    assert beams_sys.time == 0.0
    assert beams_sys.cumulative_input_work == 0.0
    assert beams_sys.cumulative_loss_energy == 0.0
    assert beams_sys.spark_count == 0
    assert beams_sys.compute_total_voltage() == 0.0
    for b in beams_sys.all_beams:
        assert np.allclose(b.q, 0.0)
        assert np.allclose(b.dq_dt, 0.0)
        assert b.voltage == 0.0


def test_43_individual_beam_modal_dynamics_and_spatial_stresses() -> None:
    """Verify all 43 rose granite beams have distinct modal responses, stresses, and voltages."""
    beams_sys = PiezoelectricBeams()
    dt = 1.0e-4
    state = beams_sys.step(dt, p_kc_acoustic=1000.0)

    assert len(state.all_beam_stresses_mpa) == 43
    assert len(state.all_beam_voltages_v) == 43
    assert len(state.all_beam_displacement_currents_a) == 43
    assert len(beams_sys.all_beams) == 43

    # Across the array, 38 non-nodal beams have unique voltages, while the 5 center beams at the acoustic node (y_b = L_KC / 2) have 0 V
    voltages = state.all_beam_voltages_v
    unique_voltages = set(round(v, 8) for v in voltages)
    assert len(unique_voltages) == 39, f"Expected 39 unique voltages (38 distinct + nodal 0.0 V), got {len(unique_voltages)}"

    # Within each individual tier, all beam voltages are distinct
    for k, (tier_start, count) in enumerate([(0, 9), (9, 9), (18, 9), (27, 9), (36, 7)]):
        tier_v = voltages[tier_start : tier_start + count]
        tier_unique_v = set(round(v, 8) for v in tier_v)
        assert len(tier_unique_v) == count, f"Tier {k} expected {count} unique voltages, got {len(tier_unique_v)}"

    # Check stress variations across longitudinal chamber coordinates y_b in Tier 0
    t0_stresses = state.all_beam_stresses_mpa[0:9]
    assert t0_stresses[0] > t0_stresses[4]  # End beam has higher stress than middle node beam
    assert np.isclose(t0_stresses[4], 0.0, atol=1.0e-12)  # Middle beam at node y_b = L_KC / 2


def test_displacement_current_calculus_relation() -> None:
    """Verify displacement current I_disp = C * dV/dt satisfies calculus relation within +-0.5%."""
    beams_sys = PiezoelectricBeams()
    c_beam = beams_sys.all_beams[0].capacitance

    # Drive with sinusoidal acoustic excitation: p(t) = p0 * sin(omega * t)
    f_drive = 438.0
    omega = 2.0 * math.pi * f_drive
    p0 = 2000.0
    dt = 1.0e-5
    duration = 0.01
    n_steps = int(duration / dt)

    current_errors = []
    for step in range(n_steps):
        t = step * dt
        p_t = p0 * math.sin(omega * t)
        state = beams_sys.step(dt, p_kc_acoustic=p_t)

        # Check total displacement current is sum of absolute individual currents
        assert np.isclose(
            state.total_displacement_current_a,
            sum(abs(i) for i in state.all_beam_displacement_currents_a),
            rtol=1.0e-9,
        )

        # For beam 0, verify I_disp matches finite difference C * (V_curr - V_prev) / dt
        b0 = beams_sys.all_beams[0]
        if step > 5:
            numerical_i = b0.displacement_current
            v_curr = b0.voltage
            v_prev = b0.prev_voltage
            expected_i = c_beam * (v_curr - v_prev) / dt
            if abs(expected_i) > 1.0e-9:
                rel_err = abs(numerical_i - expected_i) / abs(expected_i)
                current_errors.append(rel_err)

    assert len(current_errors) > 0
    assert max(current_errors) < 0.005, f"Displacement current calculus relation error {max(current_errors)} exceeded 0.5%"


def test_tier_capacitance_and_array_impedance() -> None:
    """Verify tier capacitance summation and AC capacitive impedance calculations."""
    beams_sys = PiezoelectricBeams()
    c_b = beams_sys.all_beams[0].capacitance
    tier_caps = beams_sys.compute_tier_capacitances()

    assert len(tier_caps) == 5
    for k, count in enumerate(TIER_BEAM_COUNTS):
        expected_c_tier = count * c_b
        assert np.isclose(tier_caps[k], expected_c_tier, rtol=1.0e-6)

    # 5-tier series equivalent capacitance: 1 / C_total = sum(1 / C_tier)
    inv_c_sum = sum(1.0 / c_k for c_k in tier_caps)
    expected_c_total = 1.0 / inv_c_sum
    assert np.isclose(beams_sys.total_capacitance, expected_c_total, rtol=1.0e-6)

    # AC impedance Z_array(f) at 438 Hz
    f_ref = 438.0
    expected_z = 1.0 / (2.0 * math.pi * f_ref * expected_c_total)
    calc_z = beams_sys.compute_array_impedance(f_ref)
    assert np.isclose(calc_z, expected_z, rtol=1.0e-6)

    state = beams_sys.step(1.0e-4, p_kc_acoustic=100.0)
    assert np.isclose(state.array_impedance_ohms, expected_z, rtol=1.0e-6)

    # Tier impedance
    for k in range(5):
        expected_z_tier = 1.0 / (2.0 * math.pi * f_ref * tier_caps[k])
        assert np.isclose(beams_sys.compute_tier_impedance(k, f_ref), expected_z_tier, rtol=1.0e-6)


def test_beam_stress_strain_tensors_and_shear_stress() -> None:
    """Verify peak bending stress, transverse shear stress, and quartz polarization."""
    beam = GraniteBeam(
        length=6.50,
        width=1.20,
        depth=1.50,
        density=2650.0,
        youngs_modulus=55.0e9,
        dielectric_permittivity=6.2,
        piezo_d33_eff=0.35e-12,
        piezo_g33_eff=0.012,
    )

    beam.q[0] = 1.0e-4
    beam.update_stress_and_voltage(dt=1.0e-4)

    # Peak fiber bending stress at center or ends
    assert beam.max_fiber_stress > 0.0
    assert beam.max_fiber_stress_mpa == pytest.approx(beam.max_fiber_stress / 1.0e6)
    assert beam.mean_fiber_stress_mpa == pytest.approx(beam.mean_fiber_stress / 1.0e6)

    # Transverse shear stress tau_xz
    assert beam.shear_stress > 0.0
    assert beam.shear_stress_mpa == pytest.approx(beam.shear_stress / 1.0e6)

    # Quartz polarization P_z = d33_eff * mean_stress
    expected_pol = beam.piezo_d33_eff * beam.mean_fiber_stress
    assert np.isclose(beam.polarization, expected_pol, rtol=1.0e-6)

    # Displacement current
    assert beam.displacement_current != 0.0 or beam.voltage != 0.0
