"""Unit and Integration Tests for Telemetry Serialization and Exporter.

Verifies:
1. SpatialFieldSlice and TelemetryFrame serialization and deserialization.
2. SimulationTelemetry JSON, Gzip JSON, and NPZ binary export.
3. TelemetryExporter frame-rate filtering and decimation.
4. End-to-end telemetry fidelity between orchestrator simulation and exported files.
"""

import gzip
import json
import math
from pathlib import Path
import tempfile
import numpy as np
import pytest

from engine.orchestrator import OrchestratorConfig, SimulationOrchestrator
from engine.telemetry import (
    SimulationTelemetry,
    SpatialFieldSlice,
    TelemetryExporter,
    TelemetryFrame,
)


def test_spatial_field_slice_serialization():
    """Verify SpatialFieldSlice dictionary conversion and reconstruction."""
    stresses = [float(i) * 0.1 for i in range(43)]
    voltages = [float(i) * -10.0 for i in range(43)]
    freqs = np.linspace(0.0, 2000.0, 128).tolist()
    psd = [-20.0 + float(i) * 0.1 for i in range(128)]

    slice_obj = SpatialFieldSlice(
        gallery_z=[0.0, 1.5, 3.0],
        acoustic_pressure_profile=[10.0, 20.0, 30.0],
        acoustic_velocity_profile=[0.01, 0.02, 0.03],
        acoustic_energy_density=[0.1, 0.2, 0.3],
        gas_nodes=["queens_chamber", "kings_chamber"],
        gas_h2_mole_fractions=[0.8, 0.2],
        gas_sound_speeds=[1100.0, 600.0],
        gas_densities=[0.25, 0.85],
        tier_voltages=[100.0, 200.0, 300.0, 400.0, 500.0],
        all_beam_stresses_mpa=stresses,
        all_beam_voltages_v=voltages,
        fft_frequencies_hz=freqs,
        fft_power_spectral_density_db=psd,
        north_shaft_power=12.5,
        south_shaft_power=15.0,
    )

    data = slice_obj.to_dict()
    reconstructed = SpatialFieldSlice.from_dict(data)

    assert reconstructed.gallery_z == [0.0, 1.5, 3.0]
    assert reconstructed.acoustic_pressure_profile == [10.0, 20.0, 30.0]
    assert reconstructed.gas_nodes == ["queens_chamber", "kings_chamber"]
    assert math.isclose(reconstructed.north_shaft_power, 12.5)
    assert math.isclose(reconstructed.south_shaft_power, 15.0)
    assert len(reconstructed.all_beam_stresses_mpa) == 43
    assert len(reconstructed.all_beam_voltages_v) == 43
    assert len(reconstructed.fft_frequencies_hz) == 128
    assert len(reconstructed.fft_power_spectral_density_db) == 128
    assert math.isclose(reconstructed.all_beam_stresses_mpa[5], 0.5)
    assert math.isclose(reconstructed.all_beam_voltages_v[10], -100.0)


def test_telemetry_frame_serialization():
    """Verify TelemetryFrame round-trip serialization."""
    frame = TelemetryFrame(
        time=1.23,
        step_index=123,
        bedrock_displacement=0.005,
        bedrock_velocity=0.12,
        water_hammer_pressure=50000.0,
        seismic_force=100000.0,
        hydraulic_force=200000.0,
        schumann_excitation=0.95,
        acoustic_pressure_sub=1500.0,
        h2_mole_fraction_qc=0.45,
        h2_mole_fraction_kc=0.12,
        chemical_reaction_rate=2.5,
        qc_chamber_temperature_k=320.0,
        cumulative_h2_moles=15.0,
        qc_heat_release_w=380000.0,
        chamber_temperatures_k=[320.0, 310.0, 305.0, 300.0, 298.0],
        chamber_pressures_pa=[101325.0, 101320.0, 101315.0, 101310.0, 101300.0],
        gallery_peak_pressure=2500.0,
        gallery_rms_pressure=1800.0,
        gallery_sound_speed_avg=650.0,
        gallery_total_acoustic_energy=45.0,
        f_sharp_spectral_purity=0.98,
        top_pressure_kc_entry=2200.0,
        antechamber_p_in=2200.0,
        antechamber_p_out=1900.0,
        antechamber_transmission_loss_db=1.2,
        antechamber_p_trans=1800.0,
        total_piezo_voltage=15000.0,
        total_piezo_charge=1.2e-5,
        displacement_current_a=0.0055,
        beam_array_impedance_ohms=747589.9,
        total_mechanical_energy=25.0,
        total_electrostatic_energy=5.0,
        max_beam_stress_pa=5.0e6,
        spark_triggered=True,
        spark_count=3,
        ion_density=1.5e18,
        maser_total_radiated_power=150.0,
        effective_radiated_power_w=1432.5,
        maser_population_inversion=1.0e19,
        maser_photon_energy_density=2.5e-5,
        maser_pumping_rate=25.0,
        maser_is_above_threshold=True,
        maser_north_beam_power=75.0,
        maser_south_beam_power=75.0,
        shaft_poynting_flux_w_m2=[1549.58, 1549.58],
        maser_state_populations={"n1": 1.0e22, "n2": 2.0e22, "delta_n": 1.0e22, "n_total": 3.0e22},
        maser_cumulative_radiated_energy=300.0,
        p_total_in=400000.0,
        p_total_out=150.0,
        p_total_loss=50000.0,
        cumulative_energy_in=500000.0,
        cumulative_energy_out=300.0,
        cumulative_energy_loss=60000.0,
        total_stored_energy=440000.0,
        delta_stored_energy=440000.0,
        net_work=439700.0,
        energy_balance_error=300.0,
        relative_energy_error=0.0006,
        is_energy_conserved=True,
    )

    data = frame.to_dict()
    assert isinstance(data, dict)
    assert data["time"] == 1.23
    assert data["step_index"] == 123
    assert data["spark_triggered"] is True
    assert data["displacement_current_a"] == 0.0055
    assert data["beam_array_impedance_ohms"] == 747589.9
    assert data["effective_radiated_power_w"] == 1432.5
    assert len(data["chamber_temperatures_k"]) == 5
    assert len(data["chamber_pressures_pa"]) == 5
    assert len(data["shaft_poynting_flux_w_m2"]) == 2
    assert data["maser_state_populations"]["n1"] == 1.0e22

    reconstructed = TelemetryFrame.from_dict(data)
    assert reconstructed.time == 1.23
    assert reconstructed.step_index == 123
    assert reconstructed.total_piezo_voltage == 15000.0
    assert reconstructed.spark_triggered is True
    assert math.isclose(reconstructed.displacement_current_a, 0.0055)
    assert math.isclose(reconstructed.beam_array_impedance_ohms, 747589.9)
    assert math.isclose(reconstructed.effective_radiated_power_w, 1432.5)
    assert len(reconstructed.chamber_temperatures_k) == 5
    assert len(reconstructed.chamber_pressures_pa) == 5
    assert len(reconstructed.shaft_poynting_flux_w_m2) == 2
    assert reconstructed.maser_state_populations["n_total"] == 3.0e22


def test_simulation_telemetry_json_save_load():
    """Verify SimulationTelemetry JSON export, compression, and deserialization."""
    telemetry = SimulationTelemetry(
        simulation_id="test_sim_01",
        scenario_name="acoustic_peak",
        duration=1.0,
        dt_macro=0.01,
        dt_micro=0.0001,
    )

    for i in range(10):
        t = i * 0.1
        f = TelemetryFrame(
            time=t,
            step_index=i,
            gallery_peak_pressure=100.0 * (i + 1),
            total_piezo_voltage=1000.0 * (i + 1),
            maser_total_radiated_power=10.0 * (i + 1),
            cumulative_energy_in=500.0 * (i + 1),
            cumulative_energy_out=10.0 * (i + 1),
            cumulative_energy_loss=50.0 * (i + 1),
            total_stored_energy=440.0 * (i + 1),
            relative_energy_error=1.0e-5,
            is_energy_conserved=True,
        )
        telemetry.add_frame(f)

    telemetry.compute_summary()
    assert telemetry.summary["total_frames_recorded"] == 10
    assert telemetry.summary["peak_maser_radiated_power_w"] == 100.0
    assert telemetry.summary["peak_piezo_voltage_v"] == 10000.0

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_telemetry.json"
        telemetry.save_json(json_path, indent=2)

        assert json_path.exists()
        loaded = SimulationTelemetry.load_json(json_path)

        assert loaded.simulation_id == "test_sim_01"
        assert loaded.scenario_name == "acoustic_peak"
        assert len(loaded.frames) == 10
        assert loaded.frames[5].time == 0.5
        assert loaded.frames[5].gallery_peak_pressure == 600.0

        gz_path = Path(tmpdir) / "test_telemetry.json.gz"
        telemetry.save_json(gz_path, compress=True)
        assert gz_path.exists()

        loaded_gz = SimulationTelemetry.load_json(gz_path)
        assert len(loaded_gz.frames) == 10
        assert loaded_gz.simulation_id == "test_sim_01"


def test_simulation_telemetry_npz_export():
    """Verify SimulationTelemetry NPZ binary archive export."""
    telemetry = SimulationTelemetry(
        simulation_id="test_npz",
        scenario_name="baseline",
        duration=0.5,
    )

    for i in range(5):
        f = TelemetryFrame(
            time=i * 0.1,
            step_index=i,
            bedrock_displacement=0.001 * i,
            water_hammer_pressure=1000.0 * i,
            total_piezo_voltage=500.0 * i,
            maser_total_radiated_power=5.0 * i,
        )
        telemetry.add_frame(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "telemetry_data.npz"
        telemetry.save_npz(npz_path)

        assert npz_path.exists()
        npz_data = np.load(npz_path)

        assert "time" in npz_data
        assert "bedrock_disp" in npz_data
        assert "piezo_v" in npz_data
        assert "maser_p_rad" in npz_data
        assert len(npz_data["time"]) == 5
        assert np.allclose(npz_data["time"], [0.0, 0.1, 0.2, 0.3, 0.4])


def test_telemetry_exporter_rate_decimation():
    """Verify TelemetryExporter filters frames at target frame rate."""
    exporter = TelemetryExporter(output_rate_hz=60.0)
    assert math.isclose(exporter.frame_interval, 1.0 / 60.0, rel_tol=1.0e-5)

    recorded_times = []
    dt = 0.001
    for i in range(1000):
        t = i * dt
        if exporter.should_record_frame(t):
            recorded_times.append(t)

    assert 58 <= len(recorded_times) <= 62


def test_orchestrator_end_to_end_telemetry_export():
    """Verify full orchestrator simulation runs and exports valid JSON."""
    orch = SimulationOrchestrator(
        orchestrator_config=OrchestratorConfig(
            scenario_name="acoustic_peak",
            duration_s=0.1,
            dt_macro=0.01,
            dt_micro=0.0002,
            telemetry_fps=50.0,
        )
    )

    telemetry = orch.run()
    assert len(telemetry.frames) > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "orchestrator_output.json"
        orch.exporter.export(telemetry, out_path)

        assert out_path.exists()
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "frames" in data
        assert "summary" in data
        assert "metadata" in data
        assert len(data["frames"]) == len(telemetry.frames)

        for fr in data["frames"]:
            assert "spatial" in fr
            assert "gallery_z" in fr["spatial"]
            assert "acoustic_pressure_profile" in fr["spatial"]
            assert "tier_voltages" in fr["spatial"]
            assert len(fr["spatial"]["tier_voltages"]) == 5

            assert "all_beam_stresses_mpa" in fr["spatial"]
            assert len(fr["spatial"]["all_beam_stresses_mpa"]) == 43
            assert "all_beam_voltages_v" in fr["spatial"]
            assert len(fr["spatial"]["all_beam_voltages_v"]) == 43
            assert "fft_frequencies_hz" in fr["spatial"]
            assert len(fr["spatial"]["fft_frequencies_hz"]) == 128
            assert "fft_power_spectral_density_db" in fr["spatial"]
            assert len(fr["spatial"]["fft_power_spectral_density_db"]) == 128

            assert "chamber_temperatures_k" in fr
            assert len(fr["chamber_temperatures_k"]) == 5
            assert "chamber_pressures_pa" in fr
            assert len(fr["chamber_pressures_pa"]) == 5
            assert "displacement_current_a" in fr
            assert "beam_array_impedance_ohms" in fr
            assert "effective_radiated_power_w" in fr
            assert "maser_state_populations" in fr
            assert "n1" in fr["maser_state_populations"]
            assert "n2" in fr["maser_state_populations"]
            assert "delta_n" in fr["maser_state_populations"]
            assert "n_total" in fr["maser_state_populations"]
            assert "shaft_poynting_flux_w_m2" in fr
            assert len(fr["shaft_poynting_flux_w_m2"]) == 2


def test_fft_spectral_peak_resolution():
    orch = SimulationOrchestrator()
    dt = 0.0001
    t = np.arange(2000) * dt
    signal = 500.0 * np.sin(2.0 * np.pi * 7.83 * t) + 200.0 * np.sin(2.0 * np.pi * 438.0 * t)
    orch._acoustic_buffer = signal.tolist()

    freqs_list, psd_list = orch._compute_fft_spectrum(num_bins=128, max_freq_hz=2000.0)
    freqs = np.array(freqs_list)
    psd = np.array(psd_list)

    assert len(freqs) == 128
    assert len(psd) == 128
    assert math.isclose(freqs[0], 0.0, abs_tol=1.0e-5)
    assert math.isclose(freqs[-1], 2000.0, abs_tol=1.0e-5)

    idx_7 = int(np.argmin(np.abs(freqs - 7.83)))
    idx_438 = int(np.argmin(np.abs(freqs - 438.0)))
    idx_noise = int(np.argmin(np.abs(freqs - 1500.0)))

    assert psd[idx_7] > 20.0
    assert psd[idx_438] > 10.0
    assert psd[idx_noise] < -30.0
    assert (psd[idx_438] - psd[idx_noise]) > 30.0


def test_telemetry_frame_backward_compatibility():
    legacy_spatial = {
        "gallery_z": [0.0, 1.0],
        "acoustic_pressure_profile": [10.0, 20.0],
    }
    spatial = SpatialFieldSlice.from_dict(legacy_spatial)
    assert spatial.gallery_z == [0.0, 1.0]
    assert spatial.all_beam_stresses_mpa == []
    assert spatial.all_beam_voltages_v == []
    assert spatial.fft_frequencies_hz == []
    assert spatial.fft_power_spectral_density_db == []

    legacy_frame = {
        "time": 0.5,
        "step_index": 50,
        "gallery_peak_pressure": 500.0,
        "spatial": legacy_spatial,
    }
    frame = TelemetryFrame.from_dict(legacy_frame)
    assert frame.time == 0.5
    assert frame.step_index == 50
    assert frame.displacement_current_a == 0.0
    assert frame.beam_array_impedance_ohms == 0.0
    assert frame.effective_radiated_power_w == 0.0
    assert frame.chamber_temperatures_k == []
    assert frame.chamber_pressures_pa == []
    assert frame.shaft_poynting_flux_w_m2 == []
    assert frame.maser_state_populations == {}
