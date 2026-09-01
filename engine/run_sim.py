#!/usr/bin/env python3
"""Simulation CLI Runner & Scientific Scenario Presets.

Command-line entrypoint for the Christopher Dunn Giza Power Plant Multi-Physics Simulation.
Coordinates parameter configuration, scenario preset initialization, coupled time-stepping,
progress monitoring, diagnostic telemetry serialization, and static trajectory visualization.

Scenario Presets:
  - baseline: Standard balanced operation with 7.83 Hz seismic pulse, progressive H2 diffusion,
              resonant acoustic build-up, and microwave beaming.
  - acoustic_peak: Resonators sharply tuned to maximum Q, demonstrating peak acoustic
                   standing waves and granite beam flexural stress.
  - full_maser_power: High chemical generation + maximum piezoelectric drive demonstrating
                      peak stimulated emission and microwave RF beaming.
  - dry_run_no_gas: Control run with H2 = 0, demonstrating that without hydrogen, sound speed
                    stays at 343.2 m/s, resonance is mismatched, and microwave emission remains 0 W.
  - high_seismic: Intense ground motion and hydraulic water hammer surge demonstrating
                  heavy mechanical and acoustic coupling.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from engine.config import SimulationConfig
from engine.geometry import PyramidGeometry
from engine.orchestrator import OrchestratorConfig, SimulationOrchestrator
from engine.telemetry import SimulationTelemetry


# Scenario Preset Constants
SCENARIO_BASELINE = "baseline"
SCENARIO_ACOUSTIC_PEAK = "acoustic_peak"
SCENARIO_FULL_MASER_POWER = "full_maser_power"
SCENARIO_DRY_RUN_NO_GAS = "dry_run_no_gas"
SCENARIO_HIGH_SEISMIC = "high_seismic"

AVAILABLE_SCENARIOS: List[str] = [
    SCENARIO_BASELINE,
    SCENARIO_ACOUSTIC_PEAK,
    SCENARIO_FULL_MASER_POWER,
    SCENARIO_DRY_RUN_NO_GAS,
    SCENARIO_HIGH_SEISMIC,
]

DEFAULT_OUTPUT_PATH = "viewer/public/sample_telemetry.json"
DEFAULT_OUTPUT_DIR = "viewer/public/scenarios"

SCENARIO_CATALOG: Dict[str, Dict[str, Any]] = {
    SCENARIO_BASELINE: {
        "id": SCENARIO_BASELINE,
        "name": "Baseline Resonant Run (10s)",
        "description": "Standard 10-second multi-physics simulation with hydrogen injection, 7.83 Hz infrasonic drive, and quartz excitation.",
        "tags": ["baseline", "resonance", "maser"],
        "recommended": True,
    },
    SCENARIO_ACOUSTIC_PEAK: {
        "id": SCENARIO_ACOUSTIC_PEAK,
        "name": "Acoustic Peak Mode",
        "description": "High-amplitude acoustic lock in the Grand Gallery resonator rack with steep standing wave pressure gradients.",
        "tags": ["acoustic", "resonance", "high-pressure"],
        "recommended": False,
    },
    SCENARIO_FULL_MASER_POWER: {
        "id": SCENARIO_FULL_MASER_POWER,
        "name": "Full Maser Power Output",
        "description": "High hydrogen mole fraction and kilovolt piezoelectric pumping driving saturated 1.4204 GHz maser beam emission.",
        "tags": ["maser", "quantum", "high-power"],
        "recommended": False,
    },
    SCENARIO_DRY_RUN_NO_GAS: {
        "id": SCENARIO_DRY_RUN_NO_GAS,
        "name": "Dry Run (No Hydrogen Gas)",
        "description": "Acoustic and seismic drive in standard atmospheric air without chemical reaction, demonstrating sub-threshold behavior.",
        "tags": ["inert", "sub-threshold", "control"],
        "recommended": False,
    },
    SCENARIO_HIGH_SEISMIC: {
        "id": SCENARIO_HIGH_SEISMIC,
        "name": "High Seismic Transient",
        "description": "Strong water hammer shock and bedrock acceleration transient testing structural relieving beam stress limits.",
        "tags": ["seismic", "water-hammer", "transient"],
        "recommended": False,
    },
}


def build_baseline_scenario(
    duration: float = 10.0,
    dt_macro: float = 0.01,
    dt_micro: float = 0.0001,
    fps: float = 60.0,
    config: Optional[SimulationConfig] = None,
) -> SimulationOrchestrator:
    """Configure baseline scenario: standard balanced parameters."""
    cfg = config or SimulationConfig()
    orch_cfg = OrchestratorConfig(
        scenario_name=SCENARIO_BASELINE,
        duration_s=duration,
        dt_macro=dt_macro,
        dt_micro=dt_micro,
        telemetry_fps=fps,
    )
    orch = SimulationOrchestrator(config=cfg, orchestrator_config=orch_cfg)
    return orch


def build_acoustic_peak_scenario(
    duration: float = 10.0,
    dt_macro: float = 0.01,
    dt_micro: float = 0.0001,
    fps: float = 60.0,
    config: Optional[SimulationConfig] = None,
) -> SimulationOrchestrator:
    """Configure acoustic peak scenario: resonators sharply tuned to maximum Q."""
    cfg = config or SimulationConfig()
    orch_cfg = OrchestratorConfig(
        scenario_name=SCENARIO_ACOUSTIC_PEAK,
        duration_s=duration,
        dt_macro=dt_macro,
        dt_micro=dt_micro,
        telemetry_fps=fps,
    )
    orch = SimulationOrchestrator(config=cfg, orchestrator_config=orch_cfg)

    # Maximize Grand Gallery resonator Q factors and acoustic coupling
    orch.gallery_acoustics.coupling_gain = 2.0
    for res in orch.gallery_acoustics.resonator_bank.resonators:
        res.quality_factor = 250.0

    # Boost seismic input to drive acoustic cavity
    orch.hydraulics.seismic_force_amplitude = 2.5e5

    # Enhance beam piezoelectric resonance Q
    orch.piezo_beams.coupling_efficiency = 0.95
    for beam in orch.piezo_beams.all_beams:
        beam.quality_factor = 500.0

    return orch


def build_full_maser_power_scenario(
    duration: float = 10.0,
    dt_macro: float = 0.01,
    dt_micro: float = 0.0001,
    fps: float = 60.0,
    config: Optional[SimulationConfig] = None,
) -> SimulationOrchestrator:
    """Configure full maser power scenario: high chemical feed + maximum piezoelectric drive."""
    cfg = config or SimulationConfig()
    orch_cfg = OrchestratorConfig(
        scenario_name=SCENARIO_FULL_MASER_POWER,
        duration_s=duration,
        dt_macro=dt_macro,
        dt_micro=dt_micro,
        telemetry_fps=fps,
    )
    orch = SimulationOrchestrator(config=cfg, orchestrator_config=orch_cfg)

    # Maximize hydraulic/seismic drive
    orch.hydraulics.seismic_force_amplitude = 3.0e5

    # High chemical reaction generation
    orch.gas_transport.initial_zn_moles = 10000.0
    orch.gas_transport.initial_hcl_moles = 20000.0
    orch.gas_transport.reset(initial_zn_moles=10000.0, initial_hcl_moles=20000.0)

    # Maximum piezoelectric electromechanical coupling
    orch.piezo_beams.coupling_efficiency = 0.95

    # Maximum maser pump coupling and cavity quality factor
    orch.microwave_maser.coupling_kappa_elec = 50.0
    orch.microwave_maser.coupling_kappa_acoust = 25.0
    orch.microwave_maser.cavity_quality_factor = 1.0e5

    return orch


def build_dry_run_no_gas_scenario(
    duration: float = 10.0,
    dt_macro: float = 0.01,
    dt_micro: float = 0.0001,
    fps: float = 60.0,
    config: Optional[SimulationConfig] = None,
) -> SimulationOrchestrator:
    """Configure dry run scenario: control experiment with zero hydrogen (H2 = 0)."""
    cfg = config or SimulationConfig()
    orch_cfg = OrchestratorConfig(
        scenario_name=SCENARIO_DRY_RUN_NO_GAS,
        duration_s=duration,
        dt_macro=dt_macro,
        dt_micro=dt_micro,
        telemetry_fps=fps,
    )
    orch = SimulationOrchestrator(config=cfg, orchestrator_config=orch_cfg)

    # Deactivate chemical reactants: H2 concentration stays strictly 0
    orch.gas_transport.initial_zn_moles = 0.0
    orch.gas_transport.initial_hcl_moles = 0.0
    orch.gas_transport.reset(
        initial_zn_moles=0.0,
        initial_hcl_moles=0.0,
        initial_h2_concentrations=np.zeros(len(orch.gas_transport._node_names)),
    )

    return orch


def build_high_seismic_scenario(
    duration: float = 10.0,
    dt_macro: float = 0.01,
    dt_micro: float = 0.0001,
    fps: float = 60.0,
    config: Optional[SimulationConfig] = None,
) -> SimulationOrchestrator:
    """Configure high seismic scenario: intense bedrock ground motion and hydraulic surge."""
    cfg = config or SimulationConfig()
    orch_cfg = OrchestratorConfig(
        scenario_name=SCENARIO_HIGH_SEISMIC,
        duration_s=duration,
        dt_macro=dt_macro,
        dt_micro=dt_micro,
        telemetry_fps=fps,
    )
    orch = SimulationOrchestrator(config=cfg, orchestrator_config=orch_cfg)

    # Massive seismic ground acceleration and hydraulic flow velocity
    orch.hydraulics.seismic_force_amplitude = 1.0e6
    orch.hydraulics.nominal_flow_velocity = 5.0
    orch.hydraulics.pulse_duty_cycle = 0.6

    return orch


SCENARIO_BUILDERS: Dict[
    str,
    Callable[
        [float, float, float, float, Optional[SimulationConfig]],
        SimulationOrchestrator,
    ],
] = {
    SCENARIO_BASELINE: build_baseline_scenario,
    SCENARIO_ACOUSTIC_PEAK: build_acoustic_peak_scenario,
    SCENARIO_FULL_MASER_POWER: build_full_maser_power_scenario,
    SCENARIO_DRY_RUN_NO_GAS: build_dry_run_no_gas_scenario,
    SCENARIO_HIGH_SEISMIC: build_high_seismic_scenario,
}


def build_scenario(
    scenario_name: str,
    duration: float = 10.0,
    dt_macro: float = 0.01,
    dt_micro: float = 0.0001,
    fps: float = 60.0,
    config: Optional[SimulationConfig] = None,
) -> SimulationOrchestrator:
    """Instantiate and configure an orchestrator for a given scenario name."""
    name = scenario_name.lower().strip()
    if name in ("standard", "default"):
        name = SCENARIO_BASELINE

    builder = SCENARIO_BUILDERS.get(name)
    if builder is None:
        valid_names = ", ".join(repr(s) for s in AVAILABLE_SCENARIOS)
        raise ValueError(
            f"Unknown scenario preset: '{scenario_name}'. Available scenarios: {valid_names}"
        )

    return builder(duration, dt_macro, dt_micro, fps, config)


def generate_diagnostic_plots(
    telemetry: SimulationTelemetry,
    output_image_path: Union[str, Path],
) -> None:
    """Generate static multi-panel matplotlib diagnostic plots from simulation telemetry."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive background backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed. Skipping diagnostic plot generation.")
        return

    frames = telemetry.frames
    if not frames:
        print("[WARN] No telemetry frames available to plot.")
        return

    times = np.array([f.time for f in frames])
    bedrock_disp_mm = np.array([f.bedrock_displacement * 1000.0 for f in frames])
    water_hammer_kpa = np.array([f.water_hammer_pressure / 1000.0 for f in frames])
    h2_qc = np.array([f.h2_mole_fraction_qc for f in frames])
    h2_kc = np.array([f.h2_mole_fraction_kc for f in frames])
    gallery_p_kpa = np.array([f.gallery_peak_pressure / 1000.0 for f in frames])
    piezo_v_kv = np.array([f.total_piezo_voltage / 1000.0 for f in frames])
    maser_power_w = np.array([f.maser_total_radiated_power for f in frames])
    north_shaft_w = np.array([f.maser_north_beam_power for f in frames])
    south_shaft_w = np.array([f.maser_south_beam_power for f in frames])
    cum_e_in = np.array([f.cumulative_energy_in for f in frames])
    cum_e_out = np.array([f.cumulative_energy_out for f in frames])
    cum_e_loss = np.array([f.cumulative_energy_loss for f in frames])
    stored_e = np.array([f.total_stored_energy for f in frames])
    rel_error = np.array([f.relative_energy_error * 100.0 for f in frames])

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=150)
    fig.suptitle(
        f"Giza Power Plant Simulation - Scenario: {telemetry.scenario_name.upper()}\n"
        f"Duration: {telemetry.duration:.2f}s | dt_macro: {telemetry.dt_macro:.4f}s | dt_micro: {telemetry.dt_micro:.5f}s",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Subterranean Hydraulics & Seismic Drive
    ax0 = axes[0, 0]
    ax0.plot(times, bedrock_disp_mm, label="Bedrock Disp (mm)", color="#1f77b4", lw=1.2)
    ax0_twin = ax0.twinx()
    ax0_twin.plot(times, water_hammer_kpa, label="Water Hammer (kPa)", color="#ff7f0e", lw=1.0, alpha=0.7)
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Displacement (mm)", color="#1f77b4")
    ax0_twin.set_ylabel("Pressure (kPa)", color="#ff7f0e")
    ax0.set_title("Subterranean Hydraulics & Water Hammer Pulse", fontweight="semibold")
    ax0.grid(True, alpha=0.3)

    # 2. Chemical Hydrogen Generation & Transport
    ax1 = axes[0, 1]
    ax1.plot(times, h2_qc, label="Queen's Chamber (QC)", color="#2ca02c", lw=1.5)
    ax1.plot(times, h2_kc, label="King's Chamber (KC)", color="#9467bd", lw=1.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("H₂ Mole Fraction $X_{H2}$")
    ax1.set_title("Hydrogen Gas Generation & Chamber Diffusion", fontweight="semibold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # 3. Grand Gallery Acoustics
    ax2 = axes[1, 0]
    ax2.plot(times, gallery_p_kpa, label="Gallery Peak Pressure (kPa)", color="#d62728", lw=1.2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Acoustic Pressure (kPa)")
    ax2.set_title("Grand Gallery Resonant Acoustic Standing Waves", fontweight="semibold")
    ax2.grid(True, alpha=0.3)

    # 4. King's Chamber Piezoelectric Transduction
    ax3 = axes[1, 1]
    ax3.plot(times, piezo_v_kv, label="Total Piezo Potential (kV)", color="#e377c2", lw=1.2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Piezo Voltage (kV)")
    ax3.set_title("King's Chamber Granite Beams Piezoelectric Voltage", fontweight="semibold")
    ax3.grid(True, alpha=0.3)

    # 5. Microwave Maser Stimulated Emission & Beaming
    ax4 = axes[2, 0]
    ax4.plot(times, maser_power_w, label="Total Radiated Power", color="#8c564b", lw=1.5)
    ax4.plot(times, north_shaft_w, label="North Shaft (32°28')", color="#17becf", lw=1.0, ls="--")
    ax4.plot(times, south_shaft_w, label="South Shaft (45°00')", color="#bcbd22", lw=1.0, ls=":")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Radiated Power (W)")
    ax4.set_title("1.4204 GHz Hydrogen Microwave Maser Beaming", fontweight="semibold")
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)

    # 6. First Law Energy Conservation & Audit
    ax5 = axes[2, 1]
    ax5.plot(times, cum_e_in, label="Energy In", color="#2ca02c", lw=1.2)
    ax5.plot(times, cum_e_loss, label="Dissipated Losses", color="#d62728", lw=1.2)
    ax5.plot(times, stored_e, label="Stored Energy", color="#1f77b4", lw=1.2)
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Energy (J)")
    ax5.set_title("Global Energy Balance & Conservation", fontweight="semibold")
    ax5.legend(loc="best")
    ax5.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    p = Path(output_image_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved diagnostic plots to: {p.resolve()}")


def validate_telemetry_schema(telemetry: SimulationTelemetry) -> bool:
    """Validate that a SimulationTelemetry instance complies with the strict schema.

    Verifies:
    1. Top-level metadata integrity and required fields.
    2. Summary statistics presence and numerical validity.
    3. Telemetry frame array integrity and non-empty state.
    4. Frame scalar channels: types, finite numbers, no NaN / Inf.
    5. Spatial field slice profiles: grid dimensions, 43-beam arrays, 5-node gases, FFT bins.

    Returns True if valid; raises ValueError or TypeError otherwise.
    """
    if not isinstance(telemetry, SimulationTelemetry):
        raise TypeError(f"Expected SimulationTelemetry instance, got {type(telemetry).__name__}")

    if not telemetry.simulation_id or not isinstance(telemetry.simulation_id, str):
        raise ValueError(f"Invalid simulation_id: {telemetry.simulation_id!r}")
    if not telemetry.version or not isinstance(telemetry.version, str):
        raise ValueError(f"Invalid version: {telemetry.version!r}")
    if not telemetry.scenario_name or not isinstance(telemetry.scenario_name, str):
        raise ValueError(f"Invalid scenario_name: {telemetry.scenario_name!r}")
    if telemetry.duration < 0 or not math.isfinite(telemetry.duration):
        raise ValueError(f"Invalid duration: {telemetry.duration}")
    if telemetry.dt_macro <= 0 or not math.isfinite(telemetry.dt_macro):
        raise ValueError(f"Invalid dt_macro: {telemetry.dt_macro}")
    if telemetry.dt_micro <= 0 or not math.isfinite(telemetry.dt_micro):
        raise ValueError(f"Invalid dt_micro: {telemetry.dt_micro}")
    if telemetry.total_frames != len(telemetry.frames):
        raise ValueError(
            f"total_frames mismatch: recorded {telemetry.total_frames} vs {len(telemetry.frames)} actual frames"
        )

    summary = telemetry.summary
    if not isinstance(summary, dict) or not summary:
        summary = telemetry.compute_summary()
    required_summary_keys = [
        "duration_s",
        "total_frames_recorded",
        "peak_maser_radiated_power_w",
        "mean_maser_radiated_power_w",
        "peak_piezo_voltage_v",
        "peak_gallery_pressure_pa",
        "total_energy_in_j",
        "total_energy_out_j",
        "max_relative_energy_error",
    ]
    for k in required_summary_keys:
        if k not in summary:
            raise ValueError(f"Missing required summary metric: '{k}'")
        val = summary[k]
        if not isinstance(val, (int, float, np.number)) or not math.isfinite(float(val)):
            raise ValueError(f"Invalid summary metric value for '{k}': {val}")

    if not telemetry.frames:
        raise ValueError("Telemetry has 0 frames.")

    sample_indices = [0, len(telemetry.frames) // 2, len(telemetry.frames) - 1]
    for f_idx in sample_indices:
        frame = telemetry.frames[f_idx]
        if not isinstance(frame.time, (int, float, np.number)) or not math.isfinite(float(frame.time)):
            raise ValueError(f"Frame {f_idx} has invalid time: {frame.time}")
        if not math.isfinite(float(frame.total_piezo_voltage)):
            raise ValueError(f"Frame {f_idx} has invalid total_piezo_voltage: {frame.total_piezo_voltage}")
        if not math.isfinite(float(frame.maser_total_radiated_power)):
            raise ValueError(f"Frame {f_idx} has invalid maser_total_radiated_power: {frame.maser_total_radiated_power}")
        if not math.isfinite(float(frame.relative_energy_error)):
            raise ValueError(f"Frame {f_idx} has invalid relative_energy_error: {frame.relative_energy_error}")

        spatial = frame.spatial
        if spatial is not None:
            if spatial.gallery_z and len(spatial.acoustic_pressure_profile) != len(spatial.gallery_z):
                raise ValueError(f"Frame {f_idx} acoustic_pressure_profile length mismatch")
            if spatial.all_beam_stresses_mpa and len(spatial.all_beam_stresses_mpa) != 43:
                raise ValueError(
                    f"Frame {f_idx} all_beam_stresses_mpa must have 43 elements, got {len(spatial.all_beam_stresses_mpa)}"
                )
            if spatial.all_beam_voltages_v and len(spatial.all_beam_voltages_v) != 43:
                raise ValueError(
                    f"Frame {f_idx} all_beam_voltages_v must have 43 elements, got {len(spatial.all_beam_voltages_v)}"
                )
            if spatial.tier_voltages and len(spatial.tier_voltages) != 5:
                raise ValueError(
                    f"Frame {f_idx} tier_voltages must have 5 elements, got {len(spatial.tier_voltages)}"
                )
            if spatial.fft_frequencies_hz and len(spatial.fft_power_spectral_density_db) != len(spatial.fft_frequencies_hz):
                raise ValueError(f"Frame {f_idx} FFT arrays length mismatch")

    return True


def generate_scenario_manifest(
    scenarios_metadata: Sequence[Dict[str, Any]],
    out_dir: Union[str, Path],
    default_scenario_id: str = SCENARIO_BASELINE,
) -> Path:
    """Generate and write manifest.json cataloging generated simulation scenarios."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    manifest_path = out_p / "manifest.json"

    manifest_data = {
        "version": "1.0.0",
        "defaultScenarioId": default_scenario_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": list(scenarios_metadata),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_path


def run_all_scenarios(
    duration: float = 10.0,
    dt_macro: float = 0.01,
    dt_micro: float = 0.0001,
    fps: float = 60.0,
    out_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    format_choice: str = "all",
    compress: bool = False,
    validate_schema: bool = False,
    quiet: bool = False,
) -> Dict[str, SimulationTelemetry]:
    """Batch execute all 5 scientific preset scenarios and generate manifest.json.

    Exports scenario data in requested format (.json, .bin, or both) to out_dir
    and generates compliant manifest.json for standalone static viewer hosting.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: Dict[str, SimulationTelemetry] = {}
    manifest_entries: List[Dict[str, Any]] = []

    if not quiet:
        print("=" * 78)
        print("  GIZA POWER PLANT BATCH SCENARIO GENERATOR")
        print("=" * 78)
        print(f"  Scenarios to Run:   {len(AVAILABLE_SCENARIOS)} ({', '.join(AVAILABLE_SCENARIOS)})")
        print(f"  Duration per Run:   {duration:.2f} s")
        print(f"  Macro / Micro Step: {dt_macro:.4f} s / {dt_micro:.6f} s")
        print(f"  Telemetry FPS:      {fps:.1f} Hz")
        print(f"  Output Directory:   {out_path.resolve()}")
        print(f"  Format Choice:      {format_choice.upper()}")
        print(f"  Validate Schema:    {validate_schema}")
        print("=" * 78)

    t_batch_start = time.perf_counter()

    for idx, sc_name in enumerate(AVAILABLE_SCENARIOS, 1):
        if not quiet:
            print(f"\n[{idx}/{len(AVAILABLE_SCENARIOS)}] Executing preset: '{sc_name}'...")

        telemetry = run_simulation(
            scenario=sc_name,
            duration=duration,
            dt_macro=dt_macro,
            dt_micro=dt_micro,
            fps=fps,
            output_path=None,
            format_choice="none",
            compress=compress,
            validate_schema=validate_schema,
            plot_path=None,
            quiet=quiet,
        )

        json_rel_url: Optional[str] = None
        bin_rel_url: Optional[str] = None

        # Export JSON
        if format_choice in ("json", "all"):
            ext = ".json.gz" if compress else ".json"
            json_file = out_path / f"{sc_name}{ext}"
            telemetry.save_json(json_file, compress=compress, indent=2 if not compress else None)
            json_rel_url = f"./scenarios/{sc_name}{ext}"

        # Export Binary (.bin)
        if format_choice in ("bin", "all"):
            bin_file = out_path / f"{sc_name}.bin"
            telemetry.save_binary(bin_file)
            bin_rel_url = f"./scenarios/{sc_name}.bin"

        catalog_meta = SCENARIO_CATALOG.get(
            sc_name,
            {
                "id": sc_name,
                "name": sc_name.replace("_", " ").title(),
                "description": f"Simulation run for preset {sc_name}.",
                "tags": [sc_name],
                "recommended": False,
            },
        )

        entry: Dict[str, Any] = {
            "id": sc_name,
            "name": catalog_meta["name"],
            "description": catalog_meta["description"],
            "duration": float(duration),
            "dt_macro": float(dt_macro),
            "tags": catalog_meta.get("tags", []),
            "recommended": catalog_meta.get("recommended", False),
            "metadata": telemetry.summary,
        }
        if bin_rel_url:
            entry["binUrl"] = bin_rel_url
            entry["bin_url"] = bin_rel_url
        if json_rel_url:
            entry["jsonUrl"] = json_rel_url
            entry["json_url"] = json_rel_url

        manifest_entries.append(entry)
        results[sc_name] = telemetry

    # Write manifest.json
    manifest_file = generate_scenario_manifest(
        scenarios_metadata=manifest_entries,
        out_dir=out_path,
        default_scenario_id=SCENARIO_BASELINE,
    )

    t_batch_elapsed = time.perf_counter() - t_batch_start
    if not quiet:
        print("\n" + "=" * 78)
        print(f"  BATCH GENERATION COMPLETE: {len(results)} scenarios in {t_batch_elapsed:.2f} s")
        print(f"  Manifest written to: {manifest_file.resolve()}")
        print("=" * 78)

    return results


def run_simulation(
    scenario: str = SCENARIO_BASELINE,
    duration: float = 10.0,
    dt_macro: float = 0.01,
    dt_micro: float = 0.0001,
    fps: float = 60.0,
    output_path: Optional[Union[str, Path]] = DEFAULT_OUTPUT_PATH,
    format_choice: str = "json",
    compress: bool = False,
    validate_schema: bool = False,
    plot_path: Optional[str] = None,
    quiet: bool = False,
) -> SimulationTelemetry:
    """Run coupled multi-physics simulation and save structured telemetry."""
    t_start_wall = time.perf_counter()

    if not quiet:
        print("=" * 78)
        print("  GIZA POWER PLANT MULTI-PHYSICS SIMULATION ENGINE")
        print("  Christopher Dunn Coupled Acoustic-Piezo-Maser Theory")
        print("=" * 78)
        print(f"  Scenario:          {scenario.upper()}")
        print(f"  Duration:          {duration:.2f} s")
        print(f"  Macro Step (dt):   {dt_macro:.5f} s ({1.0 / dt_macro:.0f} Hz)")
        print(f"  Micro Step (dt):   {dt_micro:.6f} s ({1.0 / dt_micro:.0f} Hz)")
        print(f"  Telemetry FPS:     {fps:.1f} Hz")
        print(f"  Output Path:       {output_path if output_path else 'None'}")
        print(f"  Export Format:     {format_choice.upper()}")
        print(f"  Compression:       {'GZIP (.gz)' if compress else 'Raw JSON'}")
        print("-" * 78)

    # Instantiate orchestrator from scenario preset
    orchestrator = build_scenario(
        scenario_name=scenario,
        duration=duration,
        dt_macro=dt_macro,
        dt_micro=dt_micro,
        fps=fps,
    )

    last_print_time = 0.0

    def progress_callback(sim_time: float, total_time: float) -> None:
        nonlocal last_print_time
        if quiet:
            return
        now = time.perf_counter()
        if now - last_print_time >= 0.5 or sim_time >= total_time:
            last_print_time = now
            pct = (sim_time / total_time) * 100.0 if total_time > 0 else 100.0
            wall_elapsed = now - t_start_wall
            speedup = sim_time / wall_elapsed if wall_elapsed > 0 else 0.0
            sys.stdout.write(
                f"\r  [SIMULATING] {sim_time:6.2f}s / {total_time:6.2f}s [{pct:5.1f}%] "
                f"| Wall: {wall_elapsed:5.1f}s | Speed: {speedup:4.2f}x real-time"
            )
            sys.stdout.flush()

    telemetry = orchestrator.run(
        duration=duration,
        dt_macro=dt_macro,
        dt_micro=dt_micro,
        progress_callback=progress_callback,
    )

    t_end_wall = time.perf_counter()
    wall_duration = t_end_wall - t_start_wall

    if not quiet:
        sys.stdout.write("\n")
        sys.stdout.flush()

    summary = telemetry.compute_summary()

    # Validate schema if requested
    if validate_schema:
        validate_telemetry_schema(telemetry)

    # Save telemetry files if requested
    saved_files: List[Path] = []
    if output_path is not None and format_choice != "none":
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if format_choice in ("json", "all"):
            is_gz = compress or str(out_p).endswith(".gz")
            json_target = out_p if (str(out_p).endswith(".json") or str(out_p).endswith(".gz")) else out_p.with_suffix(".json.gz" if is_gz else ".json")
            sf = telemetry.save_json(json_target, compress=is_gz, indent=2 if not is_gz else None)
            saved_files.append(sf)

        if format_choice in ("bin", "all"):
            if str(out_p).endswith(".json.gz"):
                bin_target = out_p.parent / out_p.name.replace(".json.gz", ".bin")
            elif str(out_p).endswith(".json"):
                bin_target = out_p.with_suffix(".bin")
            elif str(out_p).endswith(".bin"):
                bin_target = out_p
            else:
                bin_target = out_p.with_suffix(".bin")
            sf = telemetry.save_binary(bin_target)
            saved_files.append(sf)

    # Generate diagnostic plot if requested
    if plot_path:
        generate_diagnostic_plots(telemetry, plot_path)

    if not quiet:
        print("-" * 78)
        print("  SIMULATION RUN COMPLETE - SUMMARY AUDIT")
        print("-" * 78)
        print(f"  Total Frames Exported:      {summary.get('total_frames_recorded', 0):,}")
        print(f"  Wall-clock Duration:        {wall_duration:.3f} s (Sim Speed: {duration / max(wall_duration, 1e-6):.2f}x)")
        print(f"  Peak Maser Radiated Power:  {summary.get('peak_maser_radiated_power_w', 0.0):.6e} W")
        print(f"  Mean Maser Radiated Power:  {summary.get('mean_maser_radiated_power_w', 0.0):.6e} W")
        print(f"  Peak Gallery Pressure:      {summary.get('peak_gallery_pressure_pa', 0.0):.2f} Pa")
        print(f"  Peak Piezoelectric Voltage: {summary.get('peak_piezo_voltage_v', 0.0):.2f} V")
        print(f"  Final H₂ Fraction (QC):     {summary.get('final_h2_mole_fraction_qc', 0.0):.4f}")
        print(f"  Final H₂ Fraction (KC):     {summary.get('final_h2_mole_fraction_kc', 0.0):.4f}")
        print(f"  Total Energy Input:         {summary.get('total_energy_in_j', 0.0):.4e} J")
        print(f"  Total Energy Loss:          {summary.get('total_energy_loss_j', 0.0):.4e} J")
        print(f"  Max Rel. Energy Error:      {summary.get('max_relative_energy_error', 0.0) * 100.0:.4f}%")
        print(f"  Energy Conserved:           {summary.get('all_steps_conserved', True)}")
        for sf in saved_files:
            file_size_kb = sf.stat().st_size / 1024.0
            print(f"  Saved File:                 {sf.resolve()} ({file_size_kb:.1f} KB)")
        print("=" * 78)

    return telemetry


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m engine.run_sim",
        description="Christopher Dunn Giza Power Plant Multi-Physics Simulation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--scenario",
        type=str,
        choices=AVAILABLE_SCENARIOS,
        default=SCENARIO_BASELINE,
        help="Scientific scenario preset condition to simulate.",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        default=False,
        help="Batch run and export all 5 scientific preset scenarios alongside manifest.json.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Total continuous simulation duration in physical seconds.",
    )
    parser.add_argument(
        "--dt-macro",
        "--dt",
        dest="dt_macro",
        type=float,
        default=0.01,
        help="Macro outer time step for chemistry, gas diffusion & thermodynamics (s).",
    )
    parser.add_argument(
        "--dt-micro",
        type=float,
        default=0.0001,
        help="Micro inner time step for acoustic waves & piezoelectric vibrations (s).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Telemetry recording frame rate in Hz (frames per second).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output destination path for single scenario telemetry dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for batch scenario generation and manifest.json.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "bin", "all"],
        default="all",
        help="Export format choice for simulation telemetry (json, bin, or all).",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="Enable gzip compression for the JSON output file (e.g. sample_telemetry.json.gz).",
    )
    parser.add_argument(
        "--validate-schema",
        action="store_true",
        default=False,
        help="Validate telemetry data structures against schema specifications.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        nargs="?",
        const="diagnostic_plots.png",
        default=None,
        help="Generate static matplotlib multi-panel diagnostic plots (optional output filename).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress terminal output and progress reporting.",
    )

    return parser


def main(args: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint function."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    try:
        if parsed_args.all_scenarios:
            run_all_scenarios(
                duration=parsed_args.duration,
                dt_macro=parsed_args.dt_macro,
                dt_micro=parsed_args.dt_micro,
                fps=parsed_args.fps,
                out_dir=parsed_args.out_dir,
                format_choice=parsed_args.format,
                compress=parsed_args.compress,
                validate_schema=parsed_args.validate_schema,
                quiet=parsed_args.quiet,
            )
            return 0

        run_simulation(
            scenario=parsed_args.scenario,
            duration=parsed_args.duration,
            dt_macro=parsed_args.dt_macro,
            dt_micro=parsed_args.dt_micro,
            fps=parsed_args.fps,
            output_path=parsed_args.out,
            format_choice=parsed_args.format,
            compress=parsed_args.compress,
            validate_schema=parsed_args.validate_schema,
            plot_path=parsed_args.plot,
            quiet=parsed_args.quiet,
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] Simulation failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
