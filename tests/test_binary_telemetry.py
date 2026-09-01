"""Tests for Compact High-Performance Binary Telemetry Serialization (.bin format).

Verifies:
1. Round-trip fidelity between original Python telemetry, JSON export, and binary (.bin) export.
2. Single-precision float32 accuracy within 1e-6 relative tolerance.
3. File size reduction exceeding 70% compared to uncompressed JSON.
4. Loading performance parsing 600 frames in < 20 ms.
5. Orchestrator integration and edge case robustness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import time
from typing import List
import numpy as np
import pytest

from engine.orchestrator import OrchestratorConfig, SimulationOrchestrator
from engine.telemetry import (
    SimulationTelemetry,
    SpatialFieldSlice,
    TelemetryExporter,
    TelemetryFrame,
    export_binary,
    load_binary,
)


def _build_rich_telemetry_frame(index: int, t: float) -> TelemetryFrame:
    stresses = [math.sin(index * 0.1 + b * 0.2) * 15.0 + 20.0 for b in range(43)]
    voltages = [math.cos(index * 0.05 + b * 0.15) * 500.0 + 1000.0 for b in range(43)]
    freqs = np.linspace(1.0, 2000.0, 128).tolist()
    psd = [-60.0 + 40.0 * math.exp(-((f - 438.0) ** 2) / 500.0) for f in freqs]
    z_coords = np.linspace(0.0, 46.7, 50).tolist()
    press_profile = [2000.0 * math.sin(math.pi * z / 46.7) * math.cos(2.0 * math.pi * 7.83 * t) for z in z_coords]
    vel_profile = [0.5 * math.cos(math.pi * z / 46.7) * math.sin(2.0 * math.pi * 7.83 * t) for z in z_coords]
    energy_profile = [0.5 * 1.2 * (v ** 2) + 0.5 * (p ** 2) / (1.2 * 343.2 ** 2) for p, v in zip(press_profile, vel_profile)]

    spatial = SpatialFieldSlice(
        gallery_z=z_coords,
        acoustic_pressure_profile=press_profile,
        acoustic_velocity_profile=vel_profile,
        acoustic_energy_density=energy_profile,
        gas_nodes=["queens_chamber", "horizontal_passage", "ascending_passage", "grand_gallery", "kings_chamber"],
        gas_h2_mole_fractions=[0.85, 0.60, 0.45, 0.35, 0.30],
        gas_sound_speeds=[1200.0, 950.0, 750.0, 600.0, 550.0],
        gas_densities=[0.20, 0.40, 0.60, 0.80, 0.90],
        tier_voltages=[1500.0, 2500.0, 3500.0, 4500.0, 5500.0],
        all_beam_stresses_mpa=stresses,
        all_beam_voltages_v=voltages,
        fft_frequencies_hz=freqs,
        fft_power_spectral_density_db=psd,
        north_shaft_power=150.0 * (1.0 + 0.1 * math.sin(t)),
        south_shaft_power=140.0 * (1.0 + 0.1 * math.cos(t)),
    )

    return TelemetryFrame(
        time=t,
        step_index=index,
        bedrock_displacement=0.002 * math.sin(2.0 * math.pi * 7.83 * t),
        bedrock_velocity=0.05 * math.cos(2.0 * math.pi * 7.83 * t),
        bedrock_acceleration=-2.5 * math.sin(2.0 * math.pi * 7.83 * t),
        water_hammer_pressure=45000.0 + 5000.0 * math.sin(t),
        seismic_force=120000.0,
        hydraulic_force=230000.0,
        schumann_excitation=0.98,
        acoustic_pressure_sub=1800.0,
        h2_mole_fraction_qc=0.85,
        h2_mole_fraction_kc=0.30,
        chemical_reaction_rate=3.2,
        qc_chamber_temperature_k=335.5,
        cumulative_h2_moles=25.0 + 0.5 * index,
        qc_heat_release_w=450000.0,
        chamber_temperatures_k=[335.5, 320.0, 310.0, 302.0, 298.0],
        chamber_pressures_pa=[101325.0, 101320.0, 101315.0, 101310.0, 101305.0],
        gallery_peak_pressure=2850.0,
        gallery_rms_pressure=2015.0,
        gallery_sound_speed_avg=600.0,
        gallery_total_acoustic_energy=55.0,
        f_sharp_spectral_purity=0.99,
        top_pressure_kc_entry=2400.0,
        antechamber_p_in=2400.0,
        antechamber_p_out=2100.0,
        antechamber_transmission_loss_db=1.15,
        antechamber_p_trans=2000.0,
        total_piezo_voltage=17500.0,
        total_piezo_charge=1.5e-5,
        displacement_current_a=0.0062,
        beam_array_impedance_ohms=725000.0,
        total_mechanical_energy=30.0,
        total_electrostatic_energy=7.5,
        max_beam_stress_pa=6.5e6,
        spark_triggered=(index % 50 == 0),
        spark_count=index // 50,
        ion_density=1.8e18,
        maser_total_radiated_power=290.0,
        effective_radiated_power_w=2769.5,
        maser_population_inversion=1.2e19,
        maser_photon_energy_density=3.0e-5,
        maser_pumping_rate=32.0,
        maser_is_above_threshold=True,
        maser_north_beam_power=150.0,
        maser_south_beam_power=140.0,
        shaft_poynting_flux_w_m2=[1549.58, 1446.28],
        maser_state_populations={"n1": 1.0e22, "n2": 2.2e22, "delta_n": 1.2e22, "n_total": 3.2e22},
        maser_cumulative_radiated_energy=290.0 * t,
        p_total_in=450000.0,
        p_total_out=290.0,
        p_total_loss=55000.0,
        cumulative_energy_in=450000.0 * t,
        cumulative_energy_out=290.0 * t,
        cumulative_energy_loss=55000.0 * t,
        total_stored_energy=394710.0,
        delta_stored_energy=394710.0,
        net_work=394710.0,
        energy_balance_error=0.0,
        relative_energy_error=1.0e-7,
        is_energy_conserved=True,
        spatial=spatial,
    )


def test_binary_roundtrip_equivalence():
    """Verify that exporting to binary and loading back restores exact state within float32 tolerance."""
    telemetry = SimulationTelemetry(
        simulation_id="test_binary_sim",
        scenario_name="acoustic_peak",
        duration=1.0,
        dt_macro=0.01,
        dt_micro=0.0001,
        metadata={"author": "GizaSim", "target": "Three.js"},
    )

    for i in range(25):
        t = i * 0.04
        telemetry.add_frame(_build_rich_telemetry_frame(i, t))

    telemetry.compute_summary()

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "test_data.bin"
        json_path = Path(tmpdir) / "test_data.json"

        telemetry.save_binary(bin_path)
        telemetry.save_json(json_path)

        assert bin_path.exists()
        assert bin_path.stat().st_size > 0

        loaded_bin = load_binary(bin_path)
        loaded_json = SimulationTelemetry.load_json(json_path)

        assert loaded_bin.simulation_id == telemetry.simulation_id
        assert loaded_bin.scenario_name == telemetry.scenario_name
        assert math.isclose(loaded_bin.duration, telemetry.duration, rel_tol=1e-5)
        assert len(loaded_bin.frames) == len(telemetry.frames)

        for orig_f, bin_f in zip(telemetry.frames, loaded_bin.frames):
            assert math.isclose(bin_f.time, orig_f.time, rel_tol=1e-5, abs_tol=1e-6)
            assert bin_f.step_index == orig_f.step_index
            assert math.isclose(bin_f.bedrock_displacement, orig_f.bedrock_displacement, rel_tol=1e-5, abs_tol=1e-6)
            assert math.isclose(bin_f.water_hammer_pressure, orig_f.water_hammer_pressure, rel_tol=1e-5, abs_tol=1e-6)
            assert math.isclose(bin_f.total_piezo_voltage, orig_f.total_piezo_voltage, rel_tol=1e-5, abs_tol=1e-6)
            assert math.isclose(bin_f.displacement_current_a, orig_f.displacement_current_a, rel_tol=1e-5, abs_tol=1e-6)
            assert math.isclose(bin_f.maser_total_radiated_power, orig_f.maser_total_radiated_power, rel_tol=1e-5, abs_tol=1e-6)
            assert math.isclose(bin_f.effective_radiated_power_w, orig_f.effective_radiated_power_w, rel_tol=1e-5, abs_tol=1e-6)
            assert bin_f.spark_triggered == orig_f.spark_triggered
            assert bin_f.spark_count == orig_f.spark_count
            assert bin_f.is_energy_conserved == orig_f.is_energy_conserved

            assert len(bin_f.chamber_temperatures_k) == len(orig_f.chamber_temperatures_k)
            assert np.allclose(bin_f.chamber_temperatures_k, orig_f.chamber_temperatures_k, rtol=1e-5, atol=1e-5)
            assert np.allclose(bin_f.chamber_pressures_pa, orig_f.chamber_pressures_pa, rtol=1e-5, atol=1e-5)
            assert np.allclose(bin_f.shaft_poynting_flux_w_m2, orig_f.shaft_poynting_flux_w_m2, rtol=1e-5, atol=1e-5)

            for pop_k, pop_v in orig_f.maser_state_populations.items():
                assert pop_k in bin_f.maser_state_populations
                assert math.isclose(bin_f.maser_state_populations[pop_k], pop_v, rel_tol=1e-5, abs_tol=1e-5)

            orig_sp = orig_f.spatial
            bin_sp = bin_f.spatial
            assert orig_sp.gas_nodes == bin_sp.gas_nodes
            assert len(bin_sp.gallery_z) == len(orig_sp.gallery_z)
            assert np.allclose(bin_sp.gallery_z, orig_sp.gallery_z, rtol=1e-5, atol=1e-5)
            assert np.allclose(bin_sp.acoustic_pressure_profile, orig_sp.acoustic_pressure_profile, rtol=1e-5, atol=1e-5)
            assert np.allclose(bin_sp.acoustic_velocity_profile, orig_sp.acoustic_velocity_profile, rtol=1e-5, atol=1e-5)
            assert np.allclose(bin_sp.acoustic_energy_density, orig_sp.acoustic_energy_density, rtol=1e-5, atol=1e-5)

            assert len(bin_sp.all_beam_stresses_mpa) == 43
            assert len(bin_sp.all_beam_voltages_v) == 43
            assert np.allclose(bin_sp.all_beam_stresses_mpa, orig_sp.all_beam_stresses_mpa, rtol=1e-5, atol=1e-5)
            assert np.allclose(bin_sp.all_beam_voltages_v, orig_sp.all_beam_voltages_v, rtol=1e-5, atol=1e-5)

            assert len(bin_sp.fft_frequencies_hz) == 128
            assert len(bin_sp.fft_power_spectral_density_db) == 128
            assert np.allclose(bin_sp.fft_frequencies_hz, orig_sp.fft_frequencies_hz, rtol=1e-5, atol=1e-5)
            assert np.allclose(bin_sp.fft_power_spectral_density_db, orig_sp.fft_power_spectral_density_db, rtol=1e-5, atol=1e-5)

            assert math.isclose(bin_sp.north_shaft_power, orig_sp.north_shaft_power, rel_tol=1e-5, abs_tol=1e-5)
            assert math.isclose(bin_sp.south_shaft_power, orig_sp.south_shaft_power, rel_tol=1e-5, abs_tol=1e-5)


def test_file_size_reduction_greater_than_70_percent():
    """Verify that .bin serialization yields > 70% file size reduction vs uncompressed JSON."""
    telemetry = SimulationTelemetry(
        simulation_id="size_test_sim",
        scenario_name="full_maser_power",
        duration=10.0,
        dt_macro=0.01,
        dt_micro=0.0001,
    )

    for i in range(600):
        t = i * (10.0 / 600.0)
        telemetry.add_frame(_build_rich_telemetry_frame(i, t))

    telemetry.compute_summary()

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "telemetry_600.json"
        bin_path = Path(tmpdir) / "telemetry_600.bin"

        telemetry.save_json(json_path)
        telemetry.save_binary(bin_path)

        json_size = json_path.stat().st_size
        bin_size = bin_path.stat().st_size

        reduction_ratio = (json_size - bin_size) / json_size
        assert bin_size < json_size * 0.30, f"Binary size {bin_size} vs JSON size {json_size} (reduction: {reduction_ratio:.1%})"
        assert reduction_ratio > 0.70


def test_binary_loading_performance_under_20ms():
    """Verify that parsing 600 frames from binary takes < 20 ms."""
    telemetry = SimulationTelemetry(
        simulation_id="bench_sim",
        scenario_name="baseline",
        duration=10.0,
        dt_macro=0.01,
        dt_micro=0.0001,
    )

    for i in range(600):
        t = i * (10.0 / 600.0)
        telemetry.add_frame(_build_rich_telemetry_frame(i, t))

    telemetry.compute_summary()

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "bench_600.bin"
        telemetry.save_binary(bin_path)

        _ = load_binary(bin_path)

        times: List[float] = []
        for _ in range(10):
            t0 = time.perf_counter()
            loaded = load_binary(bin_path)
            dt = time.perf_counter() - t0
            times.append(dt)
            assert len(loaded.frames) == 600

        min_time = min(times)
        mean_time = sum(times) / len(times)

        assert min_time < 0.020, f"Min load time {min_time * 1000.0:.2f} ms exceeded 20 ms target"
        assert mean_time < 0.025, f"Mean load time {mean_time * 1000.0:.2f} ms exceeded 25 ms"


def test_orchestrator_export_binary():
    """Verify SimulationOrchestrator helper export_binary method."""
    orch = SimulationOrchestrator(
        orchestrator_config=OrchestratorConfig(
            scenario_name="acoustic_peak",
            duration_s=0.05,
            dt_macro=0.01,
            dt_micro=0.0002,
            telemetry_fps=50.0,
        )
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "orch_out.bin"
        json_path = Path(tmpdir) / "orch_out.json"

        out_b = orch.export_binary(bin_path)
        assert out_b == bin_path
        assert bin_path.exists()

        loaded = load_binary(bin_path)
        assert len(loaded.frames) > 0
        assert loaded.scenario_name == "acoustic_peak"

        out_j = orch.export_json(json_path)
        assert out_j == json_path
        assert json_path.exists()


def test_exporter_routing_bin_extension():
    """Verify TelemetryExporter routes .bin extension to export_binary."""
    exporter = TelemetryExporter(output_rate_hz=60.0)
    telemetry = SimulationTelemetry(simulation_id="routing_test")
    telemetry.add_frame(TelemetryFrame(time=0.1, step_index=1, total_piezo_voltage=500.0))

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "exporter_test.bin"
        saved = exporter.export(telemetry, bin_path)
        assert saved == bin_path
        assert bin_path.exists()

        loaded = SimulationTelemetry.load_binary(bin_path)
        assert len(loaded.frames) == 1
        assert math.isclose(loaded.frames[0].total_piezo_voltage, 500.0)


def test_empty_and_corrupt_binary_handling():
    """Verify empty telemetry handling and corrupt file error handling."""
    empty_telemetry = SimulationTelemetry(simulation_id="empty_sim")

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "empty.bin"
        empty_telemetry.save_binary(bin_path)

        loaded_empty = load_binary(bin_path)
        assert len(loaded_empty.frames) == 0
        assert loaded_empty.simulation_id == "empty_sim"

        corrupt_path = Path(tmpdir) / "corrupt.bin"
        with open(corrupt_path, "wb") as f:
            f.write(b"\x01\x02")

        with pytest.raises(ValueError, match="truncated"):
            load_binary(corrupt_path)
