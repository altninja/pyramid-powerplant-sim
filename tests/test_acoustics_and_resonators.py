import math
import numpy as np
import pytest

from engine.config import SimulationConfig
from engine.geometry import PyramidGeometry
from engine.physics.chemical_gas_transport import ChemicalGasTransport
from engine.physics.grand_gallery_acoustics import (
    GalleryAcousticState,
    GrandGalleryAcoustics,
    HelmholtzResonator,
    ResonatorBank,
)


def test_helmholtz_single_resonator_formula_and_frequency():
    c_sound = 343.2
    a_neck = 0.015
    l_eff = 0.12
    v_cav = 0.08

    theor_f = (c_sound / (2.0 * math.pi)) * math.sqrt(a_neck / (v_cav * l_eff))
    res = HelmholtzResonator(
        slot_index=0,
        position_z=10.0,
        target_frequency_hz=theor_f,
        neck_area=a_neck,
        neck_length=l_eff,
        cavity_volume=v_cav,
    )

    f_calc = res.frequency(c_sound)
    rel_error = abs(f_calc - theor_f) / theor_f
    assert rel_error <= 0.005

    res_438 = HelmholtzResonator(
        slot_index=1,
        position_z=5.0,
        target_frequency_hz=438.0,
        cavity_volume=0.0,
    )
    assert abs(res_438.frequency(c_sound) - 438.0) < 0.1


def test_helmholtz_single_resonator_dynamic_resonance_peak():
    c_sound = 343.2
    f_target = 438.0
    res = HelmholtzResonator(
        slot_index=0,
        position_z=0.0,
        target_frequency_hz=f_target,
        quality_factor=50.0,
        cavity_volume=0.0,
    )

    dt = 1.0e-5
    duration = 0.1
    test_freqs = [400.0, 420.0, 438.0, 455.0, 480.0]
    amplitudes = []

    for f_drive in test_freqs:
        res.reset()
        max_u = 0.0
        n_steps = int(duration / dt)
        for step in range(n_steps):
            t = step * dt
            p_drive = 100.0 * math.sin(2.0 * math.pi * f_drive * t)
            u, _, _ = res.step_rk4(p_drive, c_sound, 1.204, dt)
            if step > n_steps // 2:
                max_u = max(max_u, abs(u))
        amplitudes.append(max_u)

    peak_idx = int(np.argmax(amplitudes))
    assert test_freqs[peak_idx] == 438.0


def test_resonator_q_factor_ringdown():
    c_sound = 343.2
    f_target = 438.0
    q = 50.0
    omega_r = 2.0 * math.pi * f_target
    theor_tau = 2.0 * q / omega_r

    res = HelmholtzResonator(
        slot_index=0,
        position_z=0.0,
        target_frequency_hz=f_target,
        quality_factor=q,
        cavity_volume=0.0,
    )

    res.displacement = 1.0
    res.velocity = 0.0

    dt = 1.0e-5
    sim_time = theor_tau * 3.0
    n_steps = int(sim_time / dt)

    t_history = []
    u_history = []

    for step in range(n_steps):
        t = step * dt
        u, _, _ = res.step_rk4(0.0, c_sound, 1.204, dt)
        t_history.append(t)
        u_history.append(u)

    t_arr = np.array(t_history)
    u_arr = np.array(u_history)

    peaks_idx = []
    for i in range(1, len(u_arr) - 1):
        if u_arr[i] > 0 and u_arr[i] > u_arr[i - 1] and u_arr[i] > u_arr[i + 1]:
            peaks_idx.append(i)

    peak_times = t_arr[peaks_idx]
    peak_vals = u_arr[peaks_idx]

    log_vals = np.log(peak_vals)
    slope, _ = np.polyfit(peak_times, log_vals, 1)
    meas_tau = -1.0 / slope

    rel_error = abs(meas_tau - theor_tau) / theor_tau
    assert rel_error <= 0.005


def test_standing_wave_harmonic_peaks():
    sim = GrandGalleryAcoustics(
        num_grid_points=800,
        attenuation_coeff=0.005,
        top_boundary_type="rigid",
        bottom_boundary_type="driven",
        enable_resonators=True,
        coupling_gain=2.0,
    )

    dt = 5.0e-5
    duration = 0.4
    t_vals = np.arange(0, duration, dt)
    n_steps = len(t_vals)

    def delta_drive(t: float) -> float:
        t0 = 0.001
        sigma = 0.0002
        return 2000.0 * math.exp(-0.5 * ((t - t0) / sigma) ** 2)

    p_gallery = []
    res_displacements = []
    for t in t_vals:
        st = sim.step(dt, bottom_pressure_drive=delta_drive(t))
        p_gallery.append(st.pressure[len(st.pressure) // 3])
        res_displacements.append(st.resonator_displacements.copy())

    p_arr = np.array(p_gallery)
    freqs = np.fft.rfftfreq(n_steps, dt)
    fft_mag = np.abs(np.fft.rfft(p_arr))

    f_targets = [438.0, 876.0, 1314.0, 1752.0]
    for ft in f_targets:
        band = np.where((freqs >= ft - 30.0) & (freqs <= ft + 30.0))[0]
        assert len(band) > 0
        peak_idx = band[np.argmax(fft_mag[band])]
        peak_f = freqs[peak_idx]
        assert abs(peak_f - ft) <= 25.0

    res_arr = np.array(res_displacements)
    for r_idx in range(4):
        u_sig = res_arr[:, r_idx]
        fft_u = np.abs(np.fft.rfft(u_sig))
        target_f_res = sim.resonator_bank.resonators[r_idx].target_frequency_hz
        band_res = np.where(
            (freqs >= target_f_res - 30.0) & (freqs <= target_f_res + 30.0)
        )[0]
        assert len(band_res) > 0
        peak_f_res = freqs[band_res[np.argmax(fft_u[band_res])]]
        assert abs(peak_f_res - target_f_res) <= 10.0


def test_hydrogen_sound_speed_shift():
    c_air = 343.2
    c_h2 = 1290.0
    expected_ratio = c_h2 / c_air

    bank = ResonatorBank(
        num_stations=27,
        harmonic_frequencies=(438.0, 876.0, 1314.0, 1752.0),
    )

    freqs_air = bank.get_frequencies(sound_speed=c_air)
    freqs_h2 = bank.get_frequencies(sound_speed=c_h2)

    ratios = freqs_h2 / freqs_air
    for r in ratios:
        assert abs(r - expected_ratio) / expected_ratio < 0.001

    sim = GrandGalleryAcoustics()
    sim.set_gas_properties(sound_speed=c_air, gas_density=1.204)
    assert np.allclose(sim.c_grid, c_air)

    sim.set_gas_from_hydrogen_fraction(1.0)
    assert np.allclose(sim.c_grid, c_h2, atol=0.5)
    assert sim.rho_grid[0] < 0.1


def test_energy_conservation_and_cfl_stability():
    sim = GrandGalleryAcoustics(
        num_grid_points=100,
        attenuation_coeff=0.0,
        top_boundary_type="rigid",
        bottom_boundary_type="rigid",
        enable_resonators=False,
        cfl_safety_factor=0.8,
    )

    sim.inject_pulse(amplitude=200.0, center_z=sim.length * 0.5, width=2.0)

    initial_energy = sim.compute_total_acoustic_energy()
    assert initial_energy > 0.0

    dt = 1.0e-4
    duration = 1.0
    n_steps = int(duration / dt)

    energies = []
    for _ in range(n_steps):
        st = sim.step(dt, bottom_pressure_drive=0.0)
        energies.append(st.total_acoustic_energy)

    energies = np.array(energies)
    assert np.all(np.isfinite(energies))
    assert not np.any(np.isnan(energies))

    energy_min = np.min(energies)
    energy_max = np.max(energies)
    assert abs(energy_max - initial_energy) / initial_energy < 0.05
    assert abs(energy_min - initial_energy) / initial_energy < 0.05


def test_resonator_bank_geometry_and_slots():
    geom = PyramidGeometry()
    gallery_geom = geom.grand_gallery
    bank = ResonatorBank(gallery_length=gallery_geom.length_along_incline)

    assert len(bank.resonators) == 27
    positions = bank.get_positions()
    assert len(positions) == 27
    assert np.all(np.diff(positions) > 0)
    assert positions[-1] <= gallery_geom.length_along_incline

    freqs = [r.target_frequency_hz for r in bank.resonators]
    expected_pattern = [438.0, 876.0, 1314.0, 1752.0]
    for i, f in enumerate(freqs):
        assert f == expected_pattern[i % len(expected_pattern)]


def test_coupling_with_gas_transport():
    transport = ChemicalGasTransport()
    state = transport.step(dt=10.0)

    sim = GrandGalleryAcoustics()
    sim.update_from_gas_transport(state)
    assert sim.c_grid[0] >= 343.2
    assert sim.rho_grid[0] > 0.0

    # Also test passing transport instance directly
    sim2 = GrandGalleryAcoustics()
    sim2.update_from_gas_transport(transport)
    assert np.allclose(sim2.c_grid, sim.c_grid)


def test_boundary_conditions_and_power_flux():
    sim = GrandGalleryAcoustics(
        num_grid_points=100,
        top_boundary_type="matched",
        bottom_boundary_type="driven",
    )

    state = sim.step(dt=1.0e-4, bottom_pressure_drive=100.0)
    assert isinstance(state, GalleryAcousticState)
    assert state.bottom_pressure == 100.0
    assert state.total_acoustic_energy >= 0.0
    assert len(state.resonator_displacements) == 27
    assert len(state.resonator_velocities) == 27
    assert len(state.resonator_energies) == 27
    assert len(state.area_profile) == 100


def test_corbelled_area_profile_and_scaling():
    """Verify corbel cross-sectional area calculation and area profile updating."""
    sim = GrandGalleryAcoustics(num_grid_points=50)
    s_corbel = sim.compute_corbelled_cross_section_area(num_courses=7)
    assert 13.0 < s_corbel < 15.0
    assert np.allclose(sim.area_grid, s_corbel)

    # Test array area profile
    taper_area = np.linspace(14.0, 7.0, 50)
    sim.set_cross_section_area(taper_area)
    assert np.allclose(sim.area_grid, taper_area)
    assert sim.cross_section_area == pytest.approx(10.5, rel=1.0e-3)


def test_wave_propagation_in_varying_cross_section_duct():
    """Verify smooth acoustic wave propagation and energy conservation in area-varying duct."""
    sim = GrandGalleryAcoustics(
        num_grid_points=120,
        attenuation_coeff=0.0,
        top_boundary_type="rigid",
        bottom_boundary_type="rigid",
        enable_resonators=False,
        cfl_safety_factor=0.7,
    )
    # Define smooth expanding horn area profile
    s_horn = 10.0 + 4.0 * np.sin(np.pi * sim.z_grid / sim.length)
    sim.set_cross_section_area(s_horn)

    sim.inject_pulse(amplitude=150.0, center_z=sim.length * 0.4, width=2.0)
    e_initial = sim.compute_total_acoustic_energy()
    assert e_initial > 0.0

    dt = 1.0e-4
    n_steps = 500
    energies = []
    for _ in range(n_steps):
        st = sim.step(dt)
        energies.append(st.total_acoustic_energy)

    energies = np.array(energies)
    assert np.all(np.isfinite(energies))
    # Energy conserved in conservative area-varying Webster horn formulation
    assert abs(np.max(energies) - e_initial) / e_initial < 0.05
    assert abs(np.min(energies) - e_initial) / e_initial < 0.05


def test_resonator_spatial_source_area_normalization():
    """Verify Helmholtz resonator feedback source term scales inversely with local area S(z_m)."""
    bank = ResonatorBank(num_stations=1, slot_positions=[10.0])
    bank.resonators[0].acceleration = 5.0
    bank.resonators[0].num_units_per_slot = 2

    z_grid = np.linspace(0.0, 20.0, 201)
    dz = z_grid[1] - z_grid[0]
    rho_grid = np.full_like(z_grid, 1.204)

    # Source with area = 10 m^2
    s_grid_10 = bank.compute_coupling_source(z_grid, gallery_area=10.0, rho_grid=rho_grid)
    # Source with area = 20 m^2 (doubled area)
    s_grid_20 = bank.compute_coupling_source(z_grid, gallery_area=20.0, rho_grid=rho_grid)

    assert np.max(s_grid_10) > 0.0
    assert np.isclose(np.max(s_grid_10), 2.0 * np.max(s_grid_20), rtol=1.0e-4)

    # Verify integral over duct area equals total rho * vol_accel
    integral_force = np.sum(s_grid_10 * 10.0 * dz)
    expected_force = 1.204 * (5.0 * 2)
    assert np.isclose(integral_force, expected_force, rtol=1.0e-4)
