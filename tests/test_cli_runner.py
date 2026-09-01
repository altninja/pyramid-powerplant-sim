"""Unit and Integration Tests for Simulation CLI Runner & Scientific Scenario Presets.

Verifies:
1. Argument parser configuration, flags, defaults, and choices.
2. Preset scenario factory functions and orchestrator tuning.
3. Multi-scenario simulation execution and telemetry JSON/GZIP output validity.
4. Dry-run control behavior (H2 = 0, 0 W RF maser power).
5. CLI main entrypoint exit codes and diagnostic plot generation.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict

import numpy as np
import pytest

from engine.run_sim import (
    AVAILABLE_SCENARIOS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_PATH,
    SCENARIO_ACOUSTIC_PEAK,
    SCENARIO_BASELINE,
    SCENARIO_CATALOG,
    SCENARIO_DRY_RUN_NO_GAS,
    SCENARIO_FULL_MASER_POWER,
    SCENARIO_HIGH_SEISMIC,
    build_acoustic_peak_scenario,
    build_baseline_scenario,
    build_dry_run_no_gas_scenario,
    build_full_maser_power_scenario,
    build_high_seismic_scenario,
    build_parser,
    build_scenario,
    generate_diagnostic_plots,
    generate_scenario_manifest,
    main,
    run_all_scenarios,
    run_simulation,
    validate_telemetry_schema,
)


def test_build_parser_defaults_and_options():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.scenario == SCENARIO_BASELINE
    assert args.all_scenarios is False
    assert math.isclose(args.duration, 10.0)
    assert math.isclose(args.dt_macro, 0.01)
    assert math.isclose(args.dt_micro, 0.0001)
    assert math.isclose(args.fps, 60.0)
    assert args.out == "viewer/public/sample_telemetry.json"
    assert args.out_dir == "viewer/public/scenarios"
    assert args.format == "all"
    assert args.compress is False
    assert args.validate_schema is False
    assert args.plot is None
    assert args.quiet is False

    custom_args = parser.parse_args([
        "--scenario", "acoustic_peak",
        "--all-scenarios",
        "--duration", "5.0",
        "--dt", "0.005",
        "--dt-micro", "0.00005",
        "--fps", "30.0",
        "--out", "output/test.json.gz",
        "--out-dir", "custom/scenarios",
        "--format", "bin",
        "--compress",
        "--validate-schema",
        "--plot", "test_plot.png",
        "--quiet",
    ])
    assert custom_args.scenario == "acoustic_peak"
    assert custom_args.all_scenarios is True
    assert math.isclose(custom_args.duration, 5.0)
    assert math.isclose(custom_args.dt_macro, 0.005)
    assert math.isclose(custom_args.dt_micro, 0.00005)
    assert math.isclose(custom_args.fps, 30.0)
    assert custom_args.out == "output/test.json.gz"
    assert custom_args.out_dir == "custom/scenarios"
    assert custom_args.format == "bin"
    assert custom_args.compress is True
    assert custom_args.validate_schema is True
    assert custom_args.plot == "test_plot.png"
    assert custom_args.quiet is True


def test_scenario_builders():
    orch_base = build_baseline_scenario(duration=2.0)
    assert orch_base.orch_cfg.scenario_name == SCENARIO_BASELINE
    assert math.isclose(orch_base.orch_cfg.duration_s, 2.0)

    orch_peak = build_acoustic_peak_scenario(duration=2.0)
    assert orch_peak.orch_cfg.scenario_name == SCENARIO_ACOUSTIC_PEAK
    assert orch_peak.gallery_acoustics.coupling_gain >= 1.5
    for res in orch_peak.gallery_acoustics.resonator_bank.resonators:
        assert res.quality_factor >= 200.0

    orch_full = build_full_maser_power_scenario(duration=2.0)
    assert orch_full.orch_cfg.scenario_name == SCENARIO_FULL_MASER_POWER
    assert orch_full.microwave_maser.coupling_kappa_elec >= 25.0
    assert orch_full.gas_transport.initial_zn_moles >= 5000.0

    orch_dry = build_dry_run_no_gas_scenario(duration=2.0)
    assert orch_dry.orch_cfg.scenario_name == SCENARIO_DRY_RUN_NO_GAS
    assert orch_dry.gas_transport.initial_zn_moles == 0.0
    assert orch_dry.gas_transport.initial_hcl_moles == 0.0

    orch_seis = build_high_seismic_scenario(duration=2.0)
    assert orch_seis.orch_cfg.scenario_name == SCENARIO_HIGH_SEISMIC
    assert orch_seis.hydraulics.seismic_force_amplitude >= 5.0e5


def test_build_scenario_dispatch_and_error():
    for name in AVAILABLE_SCENARIOS:
        orch = build_scenario(name, duration=1.0)
        assert orch.orch_cfg.scenario_name == name

    orch_alias = build_scenario("standard", duration=1.0)
    assert orch_alias.orch_cfg.scenario_name == SCENARIO_BASELINE

    with pytest.raises(ValueError, match="Unknown scenario preset"):
        build_scenario("invalid_scenario_name")


def test_run_simulation_baseline():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "baseline_telem.json"
        telemetry = run_simulation(
            scenario=SCENARIO_BASELINE,
            duration=0.05,
            dt_macro=0.01,
            dt_micro=0.0001,
            fps=60.0,
            output_path=out_file,
            quiet=True,
        )

        assert out_file.exists()
        assert telemetry.scenario_name == SCENARIO_BASELINE
        assert len(telemetry.frames) >= 3

        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["scenario_name"] == SCENARIO_BASELINE
        assert len(data["frames"]) == len(telemetry.frames)
        assert "summary" in data
        assert "metadata" in data


def test_run_simulation_dry_run_no_gas():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "dry_run.json"
        telemetry = run_simulation(
            scenario=SCENARIO_DRY_RUN_NO_GAS,
            duration=0.05,
            dt_macro=0.01,
            dt_micro=0.0001,
            fps=60.0,
            output_path=out_file,
            quiet=True,
        )

        summary = telemetry.compute_summary()
        assert math.isclose(summary["final_h2_mole_fraction_qc"], 0.0, abs_tol=1e-12)
        assert math.isclose(summary["final_h2_mole_fraction_kc"], 0.0, abs_tol=1e-12)
        assert math.isclose(summary["peak_maser_radiated_power_w"], 0.0, abs_tol=1e-12)


def test_run_simulation_full_maser_power():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "full_maser.json"
        telemetry = run_simulation(
            scenario=SCENARIO_FULL_MASER_POWER,
            duration=0.05,
            dt_macro=0.01,
            dt_micro=0.0001,
            fps=60.0,
            output_path=out_file,
            quiet=True,
        )

        summary = telemetry.compute_summary()
        assert summary["final_h2_mole_fraction_qc"] > 0.0
        assert summary["total_energy_in_j"] > 0.0


def test_run_simulation_acoustic_peak():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "acoustic_peak.json"
        telemetry = run_simulation(
            scenario=SCENARIO_ACOUSTIC_PEAK,
            duration=0.05,
            dt_macro=0.01,
            dt_micro=0.0001,
            fps=60.0,
            output_path=out_file,
            quiet=True,
        )

        summary = telemetry.compute_summary()
        assert summary["total_frames_recorded"] >= 3
        assert summary["total_energy_in_j"] > 0.0


def test_run_simulation_high_seismic():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "high_seismic.json"
        telemetry = run_simulation(
            scenario=SCENARIO_HIGH_SEISMIC,
            duration=0.05,
            dt_macro=0.01,
            dt_micro=0.0001,
            fps=60.0,
            output_path=out_file,
            quiet=True,
        )

        summary = telemetry.compute_summary()
        assert summary["total_frames_recorded"] >= 3
        assert summary["total_energy_in_j"] > 0.0


def test_run_simulation_compressed_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "compressed_telem.json.gz"
        telemetry = run_simulation(
            scenario=SCENARIO_BASELINE,
            duration=0.05,
            dt_macro=0.01,
            dt_micro=0.0001,
            fps=60.0,
            output_path=out_file,
            compress=True,
            quiet=True,
        )

        assert out_file.exists()
        with gzip.open(out_file, "rt", encoding="utf-8") as f:
            data = json.load(f)

        assert data["scenario_name"] == SCENARIO_BASELINE
        assert len(data["frames"]) == len(telemetry.frames)


def test_cli_main_entrypoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = str(Path(tmpdir) / "main_out.json")
        exit_code = main([
            "--scenario", "baseline",
            "--duration", "0.03",
            "--dt", "0.01",
            "--out", out_file,
            "--quiet",
        ])
        assert exit_code == 0
        assert Path(out_file).exists()


def test_diagnostic_plot_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "plot_telem.json"
        plot_file = Path(tmpdir) / "plot.png"

        telemetry = run_simulation(
            scenario=SCENARIO_BASELINE,
            duration=0.05,
            dt_macro=0.01,
            dt_micro=0.0001,
            fps=60.0,
            output_path=out_file,
            plot_path=str(plot_file),
            quiet=True,
        )

        assert plot_file.exists()
        assert plot_file.stat().st_size > 1000


def test_validate_telemetry_schema():
    telemetry = run_simulation(
        scenario=SCENARIO_BASELINE,
        duration=0.03,
        dt_macro=0.01,
        dt_micro=0.0001,
        fps=60.0,
        output_path=None,
        quiet=True,
    )
    assert validate_telemetry_schema(telemetry) is True

    with pytest.raises(TypeError, match="Expected SimulationTelemetry"):
        validate_telemetry_schema({"not": "telemetry"})  # type: ignore

    telemetry.total_frames = 999999
    with pytest.raises(ValueError, match="total_frames mismatch"):
        validate_telemetry_schema(telemetry)
    telemetry.total_frames = len(telemetry.frames)

    saved_v = telemetry.frames[0].total_piezo_voltage
    telemetry.frames[0].total_piezo_voltage = float("nan")
    with pytest.raises(ValueError, match="invalid total_piezo_voltage"):
        validate_telemetry_schema(telemetry)
    telemetry.frames[0].total_piezo_voltage = saved_v


def test_generate_scenario_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        sample_scenarios = [
            {
                "id": "test_sc_1",
                "name": "Test Scenario 1",
                "description": "Test description 1",
                "duration": 5.0,
                "dt_macro": 0.01,
                "tags": ["test"],
                "binUrl": "./scenarios/test_sc_1.bin",
                "jsonUrl": "./scenarios/test_sc_1.json",
                "metadata": {"test_key": 123},
            }
        ]
        manifest_path = generate_scenario_manifest(sample_scenarios, out_dir, default_scenario_id="test_sc_1")
        assert manifest_path.exists()

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["version"] == "1.0.0"
        assert data["defaultScenarioId"] == "test_sc_1"
        assert "generated_at" in data
        assert len(data["scenarios"]) == 1
        assert data["scenarios"][0]["id"] == "test_sc_1"


def test_run_all_scenarios_batch_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        results = run_all_scenarios(
            duration=0.03,
            dt_macro=0.01,
            dt_micro=0.0001,
            fps=60.0,
            out_dir=out_dir,
            format_choice="all",
            validate_schema=True,
            quiet=True,
        )

        assert len(results) == 5
        for sc_name in AVAILABLE_SCENARIOS:
            assert sc_name in results
            json_file = out_dir / f"{sc_name}.json"
            bin_file = out_dir / f"{sc_name}.bin"
            assert json_file.exists(), f"Missing JSON file for {sc_name}"
            assert bin_file.exists(), f"Missing BIN file for {sc_name}"
            assert json_file.stat().st_size > 0
            assert bin_file.stat().st_size > 0

        manifest_file = out_dir / "manifest.json"
        assert manifest_file.exists()

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        assert manifest_data["version"] == "1.0.0"
        assert manifest_data["defaultScenarioId"] == SCENARIO_BASELINE
        assert len(manifest_data["scenarios"]) == 5

        for entry in manifest_data["scenarios"]:
            assert "id" in entry
            assert "name" in entry
            assert "description" in entry
            assert "binUrl" in entry
            assert "jsonUrl" in entry
            assert "metadata" in entry
            assert entry["id"] in AVAILABLE_SCENARIOS

        baseline_telem = results[SCENARIO_BASELINE]
        loaded_bin = baseline_telem.load_binary(out_dir / f"{SCENARIO_BASELINE}.bin")
        assert loaded_bin.scenario_name == SCENARIO_BASELINE
        assert len(loaded_bin.frames) == len(baseline_telem.frames)


def test_run_all_scenarios_format_filter():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "bin_only"
        results = run_all_scenarios(
            duration=0.02,
            dt_macro=0.01,
            dt_micro=0.0001,
            fps=60.0,
            out_dir=out_dir,
            format_choice="bin",
            validate_schema=False,
            quiet=True,
        )
        assert len(results) == 5
        for sc_name in AVAILABLE_SCENARIOS:
            assert (out_dir / f"{sc_name}.bin").exists()
            assert not (out_dir / f"{sc_name}.json").exists()


def test_cli_main_all_scenarios_entrypoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "cli_batch"
        exit_code = main([
            "--all-scenarios",
            "--duration", "0.02",
            "--dt", "0.01",
            "--dt-micro", "0.0001",
            "--fps", "60.0",
            "--out-dir", str(out_dir),
            "--format", "all",
            "--validate-schema",
            "--quiet",
        ])
        assert exit_code == 0
        assert (out_dir / "manifest.json").exists()
        assert (out_dir / "baseline.bin").exists()
        assert (out_dir / "baseline.json").exists()
