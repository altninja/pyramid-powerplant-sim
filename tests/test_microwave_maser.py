"""Comprehensive test suite for King's Chamber Microwave Maser & Shaft Beaming.

Tests:
1. Waveguide cutoff verification: sub-cutoff frequencies are evanescently attenuated,
   while 1.4204 GHz propagates freely with correct beta, phase velocity, and impedance.
2. Maser threshold condition: stimulated emission occurs if and only if pumping rate
   exceeds critical threshold W_pump > W_th.
3. Hydrogen gas dependency: zero stimulated emission and zero beam power when X_H2 = 0.0.
4. Shaft radiation power balance: emitted beam powers match waveguide transmission
   efficiencies and stimulated energy is strictly conserved.
5. Stability and steady-state saturation: population inversion clamps to threshold
   Delta N_th under high continuous pumping and matches analytical formulas.
6. Horn antenna aperture gain and directional beam pointing geometry.
7. Electromechanical-acoustic pumping rate scaling laws (quadratic in V_piezo, linear in p_KC).
"""

import math
import numpy as np
import pytest

from engine.config import SPEED_OF_LIGHT_C0
from engine.geometry import PyramidGeometry
from engine.physics.microwave_maser import (
    FREE_SPACE_IMPEDANCE_OHMS,
    HYPERFINE_FREQUENCY_HZ,
    PROPAGATION_BETA_RAD_M,
    MicrowaveMaser,
    WaveguideShaft,
)


def test_waveguide_shaft_cutoff_frequency() -> None:
    shaft = WaveguideShaft(width=0.22, height=0.22, length=71.0)
    expected_fc = SPEED_OF_LIGHT_C0 / (2.0 * 0.22)
    assert np.isclose(shaft.cutoff_frequency, expected_fc, rtol=1.0e-6)
    assert np.isclose(shaft.cutoff_frequency, 681.346495e6, rtol=1.0e-4)


def test_waveguide_evanescent_subcutoff_attenuation() -> None:
    shaft = WaveguideShaft(width=0.22, height=0.22, length=71.0, attenuation_np_per_m=0.005)
    subcutoff_freqs = [10.0e6, 100.0e6, 300.0e6, 500.0e6, 650.0e6]
    for f in subcutoff_freqs:
        gamma = shaft.propagation_constant(f)
        assert gamma.real > 1.0
        assert np.isclose(gamma.imag, 0.0)
        eff = shaft.transmission_efficiency(f)
        amp = shaft.transmission_amplitude(f)
        assert eff < 1.0e-30
        assert amp < 1.0e-15
        assert shaft.transmission_loss_db(f) >= 100.0


def test_waveguide_propagation_at_maser_hyperfine_frequency() -> None:
    shaft_north = WaveguideShaft(width=0.22, height=0.22, length=71.0, attenuation_np_per_m=0.005)
    shaft_south = WaveguideShaft(width=0.22, height=0.22, length=53.0, attenuation_np_per_m=0.005)
    f_maser = HYPERFINE_FREQUENCY_HZ

    gamma = shaft_north.propagation_constant(f_maser)
    beta = gamma.imag
    assert np.isclose(beta, PROPAGATION_BETA_RAD_M, rtol=1.0e-3)
    assert np.isclose(gamma.real, 0.005)

    lambda_g = shaft_north.guide_wavelength(f_maser)
    lambda_0 = SPEED_OF_LIGHT_C0 / f_maser
    assert lambda_g > lambda_0
    assert np.isclose(lambda_g, 2.0 * math.pi / beta, rtol=1.0e-6)

    vg = shaft_north.group_velocity(f_maser)
    vp = shaft_north.phase_velocity(f_maser)
    assert vg < SPEED_OF_LIGHT_C0
    assert vp > SPEED_OF_LIGHT_C0
    assert np.isclose(vg * vp, SPEED_OF_LIGHT_C0 ** 2, rtol=1.0e-5)

    z_te = shaft_north.wave_impedance(f_maser)
    assert z_te.imag == 0.0
    assert z_te.real > FREE_SPACE_IMPEDANCE_OHMS
    expected_zte = FREE_SPACE_IMPEDANCE_OHMS / math.sqrt(1.0 - (shaft_north.cutoff_frequency / f_maser) ** 2)
    assert np.isclose(z_te.real, expected_zte, rtol=1.0e-5)

    eff_north = shaft_north.transmission_efficiency(f_maser)
    expected_eff_north = math.exp(-2.0 * 0.005 * 71.0)
    assert np.isclose(eff_north, expected_eff_north, rtol=1.0e-5)
    assert 0.48 < eff_north < 0.50

    eff_south = shaft_south.transmission_efficiency(f_maser)
    expected_eff_south = math.exp(-2.0 * 0.005 * 53.0)
    assert np.isclose(eff_south, expected_eff_south, rtol=1.0e-5)
    assert 0.58 < eff_south < 0.60


def test_horn_aperture_gain_and_pointing_vectors() -> None:
    geo = PyramidGeometry()
    maser = MicrowaveMaser(geometry=geo)
    f_maser = HYPERFINE_FREQUENCY_HZ

    g_north = maser.north_shaft.aperture_gain(f_maser)
    g_north_dbi = maser.north_shaft.aperture_gain_dbi(f_maser)
    lambda_0 = SPEED_OF_LIGHT_C0 / f_maser
    area = 0.22 * 0.22
    expected_gain = (4.0 * math.pi / (lambda_0 ** 2)) * area * 0.70
    expected_gain_dbi = 10.0 * math.log10(expected_gain)

    assert np.isclose(g_north, expected_gain, rtol=1.0e-5)
    assert np.isclose(g_north_dbi, expected_gain_dbi, rtol=1.0e-5)
    assert 9.0 < g_north_dbi < 11.0

    u_north = maser.north_shaft.get_unit_vector()
    u_south = maser.south_shaft.get_unit_vector()

    assert np.isclose(u_north.magnitude(), 1.0)
    assert np.isclose(u_south.magnitude(), 1.0)
    assert u_north.x == 0.0
    assert u_north.y < 0.0
    assert u_north.z > 0.0
    assert u_south.x == 0.0
    assert u_south.y > 0.0
    assert u_south.z > 0.0

    rad_n = math.radians(32.46666666666667)
    assert np.isclose(u_north.y, -math.cos(rad_n), atol=1.0e-4)
    assert np.isclose(u_north.z, math.sin(rad_n), atol=1.0e-4)


def test_pumping_rate_electromechanical_scaling() -> None:
    maser = MicrowaveMaser(
        coupling_kappa_elec=10.0,
        coupling_kappa_acoust=5.0,
        voltage_norm=1000.0,
        pressure_norm=1000.0,
    )

    w1 = maser.compute_pumping_rate(piezo_voltage=1000.0, acoustic_pressure=0.0, h2_fraction=1.0)
    assert np.isclose(w1, 10.0)

    w2 = maser.compute_pumping_rate(piezo_voltage=2000.0, acoustic_pressure=0.0, h2_fraction=1.0)
    assert np.isclose(w2, 40.0)

    w3 = maser.compute_pumping_rate(piezo_voltage=0.0, acoustic_pressure=2000.0, h2_fraction=1.0)
    assert np.isclose(w3, 10.0)

    w4 = maser.compute_pumping_rate(piezo_voltage=2000.0, acoustic_pressure=2000.0, h2_fraction=0.5)
    assert np.isclose(w4, 0.5 * (40.0 + 10.0))


def test_zero_hydrogen_dependency() -> None:
    maser = MicrowaveMaser()
    state = maser.step(
        dt=0.01,
        piezo_voltage=50000.0,
        acoustic_pressure=10000.0,
        h2_concentration=0.0,
    )

    assert state.total_h_density == 0.0
    assert state.n1_population == 0.0
    assert state.n2_population == 0.0
    assert state.stimulated_power_total == 0.0
    assert state.total_radiated_power == 0.0
    assert not state.is_above_threshold


def test_maser_threshold_condition() -> None:
    maser = MicrowaveMaser(nominal_h_density=1.0e20)
    w_th = maser.threshold_pumping_rate(h2_fraction=1.0)
    assert w_th > 0.0
    assert np.isfinite(w_th)

    delta_n_th = maser.threshold_population_inversion()
    assert delta_n_th > 0.0

    state_sub = maser.calculate_steady_state(
        piezo_voltage=0.0,
        acoustic_pressure=0.0,
        h2_concentration=1.0,
    )
    assert state_sub.pumping_rate < w_th
    assert state_sub.stimulated_power_total == pytest.approx(0.0, abs=1.0e-9)
    assert state_sub.total_radiated_power == pytest.approx(0.0, abs=1.0e-9)
    assert not state_sub.is_above_threshold

    state_supra = maser.calculate_steady_state(
        piezo_voltage=5000.0,
        acoustic_pressure=5000.0,
        h2_concentration=1.0,
    )
    assert state_supra.pumping_rate > w_th
    assert state_supra.stimulated_power_total > 0.0
    assert state_supra.total_radiated_power > 0.0
    assert state_supra.is_above_threshold
    assert np.isclose(state_supra.population_inversion, delta_n_th, rtol=1.0e-4)


def test_shaft_radiation_power_balance() -> None:
    maser = MicrowaveMaser()
    maser.photon_energy_density = 1.0e-6
    state = maser.get_state(piezo_voltage=1000.0, acoustic_pressure=500.0, h2_concentration=1.0)

    eta_n = maser.north_shaft.transmission_efficiency(HYPERFINE_FREQUENCY_HZ)
    eta_s = maser.south_shaft.transmission_efficiency(HYPERFINE_FREQUENCY_HZ)

    assert np.isclose(state.north_shaft_beam_power, state.north_shaft_power_in * eta_n)
    assert np.isclose(state.south_shaft_beam_power, state.south_shaft_power_in * eta_s)
    assert np.isclose(
        state.total_radiated_power,
        state.north_shaft_beam_power + state.south_shaft_beam_power,
    )
    assert np.isclose(
        state.shaft_extracted_power,
        state.north_shaft_power_in + state.south_shaft_power_in,
    )


def test_dynamic_step_convergence_and_gain_clamping() -> None:
    maser = MicrowaveMaser(nominal_h_density=1.0e20)
    piezo_v = 3000.0
    p_kc = 2000.0
    x_h2 = 1.0

    dt = 1.0e-3
    for _ in range(500):
        state = maser.step(
            dt=dt,
            piezo_voltage=piezo_v,
            acoustic_pressure=p_kc,
            h2_concentration=x_h2,
        )

    delta_n_th = maser.threshold_population_inversion()
    assert state.is_above_threshold
    assert np.isclose(state.population_inversion, delta_n_th, rtol=0.01)
    assert state.stimulated_power_total > 0.0
    assert state.total_radiated_power > 0.0

    ss_state = maser.calculate_steady_state(piezo_v, p_kc, x_h2)
    assert np.isclose(state.photon_energy_density, ss_state.photon_energy_density, rtol=0.05)
    assert np.isclose(state.total_radiated_power, ss_state.total_radiated_power, rtol=0.05)


def test_energy_conservation_in_steady_state() -> None:
    maser = MicrowaveMaser()
    state = maser.calculate_steady_state(
        piezo_voltage=4000.0,
        acoustic_pressure=2000.0,
        h2_concentration=1.0,
    )

    p_in_field = state.stimulated_power_total
    p_out_field = state.cavity_loss_power + state.shaft_extracted_power

    rel_diff = abs(p_in_field - p_out_field) / max(1.0e-9, p_in_field)
    assert rel_diff < 1.0e-3

    p_shaft_dissipated = state.shaft_extracted_power - state.total_radiated_power
    assert p_shaft_dissipated >= 0.0
    assert np.isclose(
        state.stimulated_power_total,
        state.cavity_loss_power + state.total_radiated_power + p_shaft_dissipated,
        rtol=1.0e-4,
    )


def test_physical_range_under_baseline_resonant_drive() -> None:
    maser = MicrowaveMaser()
    
    # Baseline resonant multi-kilovolt drive: V_piezo = 5 kV, p_KC = 10 kPa, X_H2 = 1.0
    state_baseline = maser.calculate_steady_state(
        piezo_voltage=5000.0,
        acoustic_pressure=10000.0,
        h2_concentration=1.0,
    )
    
    assert state_baseline.is_above_threshold
    assert 1.0 <= state_baseline.total_radiated_power <= 5000.0
    assert state_baseline.stimulated_power_total > state_baseline.total_radiated_power
    assert state_baseline.total_erp_watts > state_baseline.total_radiated_power
    
    # Higher resonant drive: V_piezo = 12 kV, p_KC = 25 kPa, X_H2 = 0.5
    state_high = maser.calculate_steady_state(
        piezo_voltage=12000.0,
        acoustic_pressure=25000.0,
        h2_concentration=0.5,
    )
    assert 1.0 <= state_high.total_radiated_power <= 5000.0
    assert state_high.total_radiated_power > state_baseline.total_radiated_power

    # Low-concentration resonant drive: V_piezo = 8 kV, p_KC = 15 kPa, X_H2 = 0.05
    state_low_x = maser.calculate_steady_state(
        piezo_voltage=8000.0,
        acoustic_pressure=15000.0,
        h2_concentration=0.05,
    )
    assert state_low_x.is_above_threshold
    assert state_low_x.total_radiated_power >= 0.1
    assert state_low_x.total_radiated_power < state_baseline.total_radiated_power


def test_sub_threshold_strict_zero_power() -> None:
    maser = MicrowaveMaser()
    
    # 1. Zero hydrogen fraction
    state_zero_h2 = maser.calculate_steady_state(
        piezo_voltage=10000.0,
        acoustic_pressure=20000.0,
        h2_concentration=0.0,
    )
    assert state_zero_h2.pumping_rate == 0.0
    assert state_zero_h2.stimulated_power_total == 0.0
    assert state_zero_h2.total_radiated_power == 0.0
    assert state_zero_h2.north_shaft_beam_power == 0.0
    assert state_zero_h2.south_shaft_beam_power == 0.0
    assert state_zero_h2.total_erp_watts == 0.0
    assert not state_zero_h2.is_above_threshold

    # 2. Sub-threshold voltage (< 100 V) with zero acoustic pressure
    state_sub_v = maser.calculate_steady_state(
        piezo_voltage=80.0,
        acoustic_pressure=0.0,
        h2_concentration=1.0,
    )
    assert state_sub_v.pumping_rate == 0.0
    assert state_sub_v.stimulated_power_total == 0.0
    assert state_sub_v.total_radiated_power == 0.0
    assert state_sub_v.total_erp_watts == 0.0
    assert not state_sub_v.is_above_threshold

    # 3. Dynamic step with sub-threshold drive
    maser.reset()
    state_step = maser.step(
        dt=0.01,
        piezo_voltage=50.0,
        acoustic_pressure=0.0,
        h2_concentration=1.0,
    )
    assert state_step.stimulated_power_total == 0.0
    assert state_step.total_radiated_power == 0.0
    assert state_step.total_erp_watts == 0.0
    assert not state_step.is_above_threshold


def test_numerical_stability_long_duration_stepping() -> None:
    maser = MicrowaveMaser()
    dt = 0.01  # 10 ms macro time step
    total_time = 10.0  # 10 s total duration
    num_steps = int(total_time / dt)

    v_drive = 8000.0
    p_drive = 20000.0
    x_drive = 0.8

    delta_n_th = maser.threshold_population_inversion()

    for step_idx in range(num_steps):
        state = maser.step(
            dt=dt,
            piezo_voltage=v_drive,
            acoustic_pressure=p_drive,
            h2_concentration=x_drive,
        )

        assert np.isfinite(state.population_inversion)
        assert np.isfinite(state.photon_energy_density)
        assert np.isfinite(state.total_radiated_power)
        assert state.n1_population >= 0.0
        assert state.n2_population >= 0.0
        assert 0.0 <= state.population_inversion <= state.total_h_density
        assert state.photon_energy_density >= 0.0
        assert state.total_radiated_power > 0.0

    # Ensure final state clamped at saturation threshold
    assert np.isclose(state.population_inversion, delta_n_th, rtol=1.0e-3)
    assert 1.0 <= state.total_radiated_power <= 5000.0


def test_effective_radiated_power_calculation() -> None:
    maser = MicrowaveMaser()
    state = maser.calculate_steady_state(
        piezo_voltage=6000.0,
        acoustic_pressure=15000.0,
        h2_concentration=1.0,
    )

    g_north_lin = maser.north_shaft.aperture_gain(HYPERFINE_FREQUENCY_HZ)
    g_south_lin = maser.south_shaft.aperture_gain(HYPERFINE_FREQUENCY_HZ)

    assert np.isclose(maser.north_shaft.aperture_gain_dbi(HYPERFINE_FREQUENCY_HZ), 9.8018, atol=0.1)
    assert np.isclose(maser.south_shaft.aperture_gain_dbi(HYPERFINE_FREQUENCY_HZ), 9.8018, atol=0.1)

    expected_erp_n = state.north_shaft_beam_power * g_north_lin
    expected_erp_s = state.south_shaft_beam_power * g_south_lin
    expected_erp_tot = expected_erp_n + expected_erp_s

    assert np.isclose(state.north_shaft_erp_watts, expected_erp_n, rtol=1.0e-6)
    assert np.isclose(state.south_shaft_erp_watts, expected_erp_s, rtol=1.0e-6)
    assert np.isclose(state.total_erp_watts, expected_erp_tot, rtol=1.0e-6)
    assert np.isclose(state.north_erp, expected_erp_n, rtol=1.0e-6)
    assert np.isclose(state.south_erp, expected_erp_s, rtol=1.0e-6)
    assert np.isclose(state.total_erp, expected_erp_tot, rtol=1.0e-6)


def test_calibrated_default_pumping_thresholds() -> None:
    maser = MicrowaveMaser()
    assert maser.v_ref == 5000.0
    assert maser.p_ref == 100000.0
    assert maser.kappa_elec == 50.0
    assert maser.kappa_acoust == 10.0

    # At V = V_ref, p = p_ref, X_H2 = 1.0 -> W_pump = 50.0*(1)^2 + 10.0*(1) = 60.0 s^-1
    w_cal = maser.compute_pumping_rate(
        piezo_voltage=5000.0,
        acoustic_pressure=100000.0,
        h2_fraction=1.0,
    )
    assert np.isclose(w_cal, 60.0)
