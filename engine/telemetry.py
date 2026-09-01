"""Telemetry Exporter and Data Serialization for Multi-Physics Simulation.

Provides structured data containers, spatial field snapshots, time-series frame
storage, and JSON/NPZ serialization for downstream Three.js viewer replay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gzip
import json
import math
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


TELEMETRY_SCALAR_CHANNELS: List[str] = [
    "time",
    "step_index",
    "bedrock_displacement",
    "bedrock_velocity",
    "bedrock_acceleration",
    "water_hammer_pressure",
    "seismic_force",
    "hydraulic_force",
    "schumann_excitation",
    "acoustic_pressure_sub",
    "h2_mole_fraction_qc",
    "h2_mole_fraction_kc",
    "chemical_reaction_rate",
    "qc_chamber_temperature_k",
    "cumulative_h2_moles",
    "qc_heat_release_w",
    "gallery_peak_pressure",
    "gallery_rms_pressure",
    "gallery_sound_speed_avg",
    "gallery_total_acoustic_energy",
    "f_sharp_spectral_purity",
    "top_pressure_kc_entry",
    "antechamber_p_in",
    "antechamber_p_out",
    "antechamber_transmission_loss_db",
    "antechamber_p_trans",
    "total_piezo_voltage",
    "total_piezo_charge",
    "displacement_current_a",
    "beam_array_impedance_ohms",
    "total_mechanical_energy",
    "total_electrostatic_energy",
    "max_beam_stress_pa",
    "spark_triggered",
    "spark_count",
    "ion_density",
    "maser_total_radiated_power",
    "effective_radiated_power_w",
    "maser_population_inversion",
    "maser_photon_energy_density",
    "maser_pumping_rate",
    "maser_is_above_threshold",
    "maser_north_beam_power",
    "maser_south_beam_power",
    "maser_cumulative_radiated_energy",
    "p_total_in",
    "p_total_out",
    "p_total_loss",
    "cumulative_energy_in",
    "cumulative_energy_out",
    "cumulative_energy_loss",
    "total_stored_energy",
    "delta_stored_energy",
    "net_work",
    "energy_balance_error",
    "relative_energy_error",
    "is_energy_conserved",
]


def _to_serializable(val: Any) -> Any:
    """Recursively convert numpy types and dataclasses into json-serializable primitives."""
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.floating, float)):
        v = float(val)
        return 0.0 if math.isnan(v) or math.isinf(v) else v
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, np.ndarray):
        return [_to_serializable(x) for x in val.tolist()]
    if isinstance(val, (list, tuple)):
        return [_to_serializable(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _to_serializable(v) for k, v in val.items()}
    return val


@dataclass
class SpatialFieldSlice:
    """1D and multi-station spatial profiles captured at a telemetry frame."""

    gallery_z: List[float] = field(default_factory=list)
    acoustic_pressure_profile: List[float] = field(default_factory=list)
    acoustic_velocity_profile: List[float] = field(default_factory=list)
    acoustic_energy_density: List[float] = field(default_factory=list)
    gas_nodes: List[str] = field(default_factory=list)
    gas_h2_mole_fractions: List[float] = field(default_factory=list)
    gas_sound_speeds: List[float] = field(default_factory=list)
    gas_densities: List[float] = field(default_factory=list)
    tier_voltages: List[float] = field(default_factory=list)
    all_beam_stresses_mpa: List[float] = field(default_factory=list)
    all_beam_voltages_v: List[float] = field(default_factory=list)
    fft_frequencies_hz: List[float] = field(default_factory=list)
    fft_power_spectral_density_db: List[float] = field(default_factory=list)
    north_shaft_power: float = 0.0
    south_shaft_power: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gallery_z": [float(x) for x in self.gallery_z],
            "acoustic_pressure_profile": [float(x) for x in self.acoustic_pressure_profile],
            "acoustic_velocity_profile": [float(x) for x in self.acoustic_velocity_profile],
            "acoustic_energy_density": [float(x) for x in self.acoustic_energy_density],
            "gas_nodes": list(self.gas_nodes),
            "gas_h2_mole_fractions": [float(x) for x in self.gas_h2_mole_fractions],
            "gas_sound_speeds": [float(x) for x in self.gas_sound_speeds],
            "gas_densities": [float(x) for x in self.gas_densities],
            "tier_voltages": [float(x) for x in self.tier_voltages],
            "all_beam_stresses_mpa": [float(x) for x in self.all_beam_stresses_mpa],
            "all_beam_voltages_v": [float(x) for x in self.all_beam_voltages_v],
            "fft_frequencies_hz": [float(x) for x in self.fft_frequencies_hz],
            "fft_power_spectral_density_db": [float(x) for x in self.fft_power_spectral_density_db],
            "north_shaft_power": float(self.north_shaft_power),
            "south_shaft_power": float(self.south_shaft_power),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpatialFieldSlice:
        return cls(
            gallery_z=[float(x) for x in data.get("gallery_z", [])],
            acoustic_pressure_profile=[float(x) for x in data.get("acoustic_pressure_profile", [])],
            acoustic_velocity_profile=[float(x) for x in data.get("acoustic_velocity_profile", [])],
            acoustic_energy_density=[float(x) for x in data.get("acoustic_energy_density", [])],
            gas_nodes=[str(x) for x in data.get("gas_nodes", [])],
            gas_h2_mole_fractions=[float(x) for x in data.get("gas_h2_mole_fractions", [])],
            gas_sound_speeds=[float(x) for x in data.get("gas_sound_speeds", [])],
            gas_densities=[float(x) for x in data.get("gas_densities", [])],
            tier_voltages=[float(x) for x in data.get("tier_voltages", [])],
            all_beam_stresses_mpa=[float(x) for x in data.get("all_beam_stresses_mpa", [])],
            all_beam_voltages_v=[float(x) for x in data.get("all_beam_voltages_v", [])],
            fft_frequencies_hz=[float(x) for x in data.get("fft_frequencies_hz", [])],
            fft_power_spectral_density_db=[float(x) for x in data.get("fft_power_spectral_density_db", [])],
            north_shaft_power=float(data.get("north_shaft_power", 0.0)),
            south_shaft_power=float(data.get("south_shaft_power", 0.0)),
        )


@dataclass
class TelemetryFrame:
    """Unified telemetry frame capturing instantaneous state of all 6 physics domains."""

    time: float = 0.0
    step_index: int = 0

    bedrock_displacement: float = 0.0
    bedrock_velocity: float = 0.0
    bedrock_acceleration: float = 0.0
    water_hammer_pressure: float = 0.0
    seismic_force: float = 0.0
    hydraulic_force: float = 0.0
    schumann_excitation: float = 0.0
    acoustic_pressure_sub: float = 0.0

    h2_mole_fraction_qc: float = 0.0
    h2_mole_fraction_kc: float = 0.0
    chemical_reaction_rate: float = 0.0
    qc_chamber_temperature_k: float = 293.15
    cumulative_h2_moles: float = 0.0
    qc_heat_release_w: float = 0.0

    chamber_temperatures_k: List[float] = field(default_factory=list)
    chamber_pressures_pa: List[float] = field(default_factory=list)

    gallery_peak_pressure: float = 0.0
    gallery_rms_pressure: float = 0.0
    gallery_sound_speed_avg: float = 343.2
    gallery_total_acoustic_energy: float = 0.0
    f_sharp_spectral_purity: float = 0.0
    top_pressure_kc_entry: float = 0.0

    antechamber_p_in: float = 0.0
    antechamber_p_out: float = 0.0
    antechamber_transmission_loss_db: float = 0.0
    antechamber_p_trans: float = 0.0

    total_piezo_voltage: float = 0.0
    total_piezo_charge: float = 0.0
    displacement_current_a: float = 0.0
    beam_array_impedance_ohms: float = 0.0
    total_mechanical_energy: float = 0.0
    total_electrostatic_energy: float = 0.0
    max_beam_stress_pa: float = 0.0
    spark_triggered: bool = False
    spark_count: int = 0
    ion_density: float = 0.0

    maser_total_radiated_power: float = 0.0
    effective_radiated_power_w: float = 0.0
    maser_population_inversion: float = 0.0
    maser_photon_energy_density: float = 0.0
    maser_pumping_rate: float = 0.0
    maser_is_above_threshold: bool = False
    maser_north_beam_power: float = 0.0
    maser_south_beam_power: float = 0.0
    shaft_poynting_flux_w_m2: List[float] = field(default_factory=list)
    maser_state_populations: Dict[str, float] = field(default_factory=dict)
    maser_cumulative_radiated_energy: float = 0.0

    p_total_in: float = 0.0
    p_total_out: float = 0.0
    p_total_loss: float = 0.0
    cumulative_energy_in: float = 0.0
    cumulative_energy_out: float = 0.0
    cumulative_energy_loss: float = 0.0
    total_stored_energy: float = 0.0
    delta_stored_energy: float = 0.0
    net_work: float = 0.0
    energy_balance_error: float = 0.0
    relative_energy_error: float = 0.0
    is_energy_conserved: bool = True

    spatial: SpatialFieldSlice = field(default_factory=SpatialFieldSlice)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["spatial"] = self.spatial.to_dict()
        return _to_serializable(d)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TelemetryFrame:
        data_copy = dict(data)
        spatial_data = data_copy.pop("spatial", None)
        spatial = (
            SpatialFieldSlice.from_dict(spatial_data)
            if isinstance(spatial_data, dict)
            else (spatial_data if isinstance(spatial_data, SpatialFieldSlice) else SpatialFieldSlice())
        )
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {}
        for k, v in data_copy.items():
            if k in field_names:
                if k in ("chamber_temperatures_k", "chamber_pressures_pa", "shaft_poynting_flux_w_m2"):
                    filtered_data[k] = [float(x) for x in (v or [])]
                elif k == "maser_state_populations":
                    filtered_data[k] = {str(pk): float(pv) for pk, pv in (v or {}).items()}
                else:
                    filtered_data[k] = v
        return cls(spatial=spatial, **filtered_data)


@dataclass
class SimulationTelemetry:
    """Complete collection of time-series frames and metadata for a simulation run."""

    simulation_id: str = "pyramid_powerplant_sim"
    version: str = "1.0.0"
    scenario_name: str = "baseline"
    duration: float = 0.0
    dt_macro: float = 0.01
    dt_micro: float = 0.0001
    total_frames: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    frames: List[TelemetryFrame] = field(default_factory=list)

    def add_frame(self, frame: TelemetryFrame) -> None:
        self.frames.append(frame)
        self.total_frames = len(self.frames)

    def compute_summary(self) -> Dict[str, Any]:
        """Compute aggregate summary metrics across all recorded frames."""
        if not self.frames:
            self.summary = {}
            return self.summary

        times = [f.time for f in self.frames]
        maser_powers = [f.maser_total_radiated_power for f in self.frames]
        piezo_voltages = [abs(f.total_piezo_voltage) for f in self.frames]
        gallery_pressures = [f.gallery_peak_pressure for f in self.frames]
        energy_errors = [f.relative_energy_error for f in self.frames]
        h2_qc = [f.h2_mole_fraction_qc for f in self.frames]
        h2_kc = [f.h2_mole_fraction_kc for f in self.frames]

        last_f = self.frames[-1]

        self.summary = {
            "duration_s": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
            "total_frames_recorded": len(self.frames),
            "peak_maser_radiated_power_w": float(max(maser_powers)),
            "mean_maser_radiated_power_w": float(np.mean(maser_powers)),
            "peak_piezo_voltage_v": float(max(piezo_voltages)),
            "peak_gallery_pressure_pa": float(max(gallery_pressures)),
            "final_h2_mole_fraction_qc": float(h2_qc[-1]),
            "final_h2_mole_fraction_kc": float(h2_kc[-1]),
            "total_energy_in_j": float(last_f.cumulative_energy_in),
            "total_energy_out_j": float(last_f.cumulative_energy_out),
            "total_energy_loss_j": float(last_f.cumulative_energy_loss),
            "final_stored_energy_j": float(last_f.total_stored_energy),
            "max_relative_energy_error": float(max(energy_errors)),
            "mean_relative_energy_error": float(np.mean(energy_errors)),
            "all_steps_conserved": bool(all(f.is_energy_conserved for f in self.frames)),
            "total_sparks": int(last_f.spark_count),
        }
        return self.summary

    def to_dict(self) -> Dict[str, Any]:
        if not self.summary and self.frames:
            self.compute_summary()

        return {
            "simulation_id": self.simulation_id,
            "version": self.version,
            "scenario_name": self.scenario_name,
            "duration": float(self.duration),
            "dt_macro": float(self.dt_macro),
            "dt_micro": float(self.dt_micro),
            "total_frames": int(self.total_frames),
            "metadata": _to_serializable(self.metadata),
            "summary": _to_serializable(self.summary),
            "frames": [f.to_dict() for f in self.frames],
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, filepath: Union[str, Path], compress: bool = False, indent: Optional[int] = None) -> Path:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)

        if compress or str(filepath).endswith(".gz"):
            json_str = self.to_json(indent=None)
            with gzip.open(p, "wt", encoding="utf-8") as f:
                f.write(json_str)
        else:
            json_str = self.to_json(indent=indent)
            with open(p, "w", encoding="utf-8") as f:
                f.write(json_str)
        return p

    def save_npz(self, filepath: Union[str, Path]) -> Path:
        """Save telemetry arrays into a fast binary NumPy archive."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)

        arrays: Dict[str, np.ndarray] = {
            "time": np.array([f.time for f in self.frames], dtype=np.float64),
            "bedrock_disp": np.array([f.bedrock_displacement for f in self.frames], dtype=np.float64),
            "water_hammer_p": np.array([f.water_hammer_pressure for f in self.frames], dtype=np.float64),
            "h2_qc": np.array([f.h2_mole_fraction_qc for f in self.frames], dtype=np.float64),
            "h2_kc": np.array([f.h2_mole_fraction_kc for f in self.frames], dtype=np.float64),
            "gallery_peak_p": np.array([f.gallery_peak_pressure for f in self.frames], dtype=np.float64),
            "gallery_rms_p": np.array([f.gallery_rms_pressure for f in self.frames], dtype=np.float64),
            "antechamber_tl": np.array([f.antechamber_transmission_loss_db for f in self.frames], dtype=np.float64),
            "piezo_v": np.array([f.total_piezo_voltage for f in self.frames], dtype=np.float64),
            "maser_p_rad": np.array([f.maser_total_radiated_power for f in self.frames], dtype=np.float64),
            "pop_inversion": np.array([f.maser_population_inversion for f in self.frames], dtype=np.float64),
            "p_total_in": np.array([f.p_total_in for f in self.frames], dtype=np.float64),
            "p_total_out": np.array([f.p_total_out for f in self.frames], dtype=np.float64),
            "p_total_loss": np.array([f.p_total_loss for f in self.frames], dtype=np.float64),
            "total_stored_energy": np.array([f.total_stored_energy for f in self.frames], dtype=np.float64),
            "rel_energy_error": np.array([f.relative_energy_error for f in self.frames], dtype=np.float64),
            "displacement_current_a": np.array([f.displacement_current_a for f in self.frames], dtype=np.float64),
            "beam_array_impedance_ohms": np.array([f.beam_array_impedance_ohms for f in self.frames], dtype=np.float64),
            "effective_radiated_power_w": np.array([f.effective_radiated_power_w for f in self.frames], dtype=np.float64),
        }
        np.savez_compressed(p, **arrays)
        return p

    def save_binary(self, filepath: Union[str, Path]) -> Path:
        """Save telemetry into a packed little-endian binary (.bin) file."""
        return export_binary(self, filepath)

    @classmethod
    def load_binary(cls, filepath: Union[str, Path]) -> SimulationTelemetry:
        """Load telemetry from a packed little-endian binary (.bin) file."""
        return load_binary(filepath)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimulationTelemetry:
        frames_raw = data.get("frames", [])
        frames = [TelemetryFrame.from_dict(fr) for fr in frames_raw]
        return cls(
            simulation_id=data.get("simulation_id", "pyramid_powerplant_sim"),
            version=data.get("version", "1.0.0"),
            scenario_name=data.get("scenario_name", "baseline"),
            duration=float(data.get("duration", 0.0)),
            dt_macro=float(data.get("dt_macro", 0.01)),
            dt_micro=float(data.get("dt_micro", 0.0001)),
            total_frames=int(data.get("total_frames", len(frames))),
            metadata=dict(data.get("metadata", {})),
            summary=dict(data.get("summary", {})),
            frames=frames,
        )

    @classmethod
    def from_json(cls, json_str: str) -> SimulationTelemetry:
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> SimulationTelemetry:
        p = Path(filepath)
        if str(filepath).endswith(".gz"):
            with gzip.open(p, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        return cls.from_dict(data)


def export_binary(telemetry: SimulationTelemetry, output_path: Union[str, Path]) -> Path:
    """Export simulation telemetry to packed little-endian binary (.bin) format.

    Layout:
    - 4-byte little-endian unsigned integer `header_len`.
    - UTF-8 JSON header string padded with spaces to 4-byte alignment.
    - Contiguous IEEE 754 float32 binary payload storing columnar scalar channels
      and multi-dimensional / spatial time series profiles.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not telemetry.summary and telemetry.frames:
        telemetry.compute_summary()

    num_frames = len(telemetry.frames)

    gas_nodes: List[str] = []
    gallery_z: List[float] = []
    fft_frequencies_hz: List[float] = []
    maser_population_keys: List[str] = []

    if num_frames > 0:
        first_frame = telemetry.frames[0]
        gas_nodes = list(first_frame.spatial.gas_nodes)
        gallery_z = [float(x) for x in first_frame.spatial.gallery_z]
        fft_frequencies_hz = [float(x) for x in first_frame.spatial.fft_frequencies_hz]

        keys_set: Dict[str, bool] = {}
        for f in telemetry.frames:
            for k in f.maser_state_populations.keys():
                keys_set[k] = True
        maser_population_keys = sorted(keys_set.keys()) if keys_set else ["n1", "n2", "delta_n", "n_total"]

    channels_meta: Dict[str, Dict[str, Any]] = {}
    payload_byte_list: List[bytes] = []
    current_offset = 0

    def add_channel(name: str, arr: np.ndarray) -> None:
        nonlocal current_offset
        f32_arr = np.ascontiguousarray(arr, dtype="<f4")
        raw_bytes = f32_arr.tobytes()
        channels_meta[name] = {
            "offset_bytes": current_offset,
            "shape": list(f32_arr.shape),
            "count": int(f32_arr.size),
            "dtype": "float32",
        }
        payload_byte_list.append(raw_bytes)
        current_offset += len(raw_bytes)

    if num_frames > 0:
        for ch in TELEMETRY_SCALAR_CHANNELS:
            vals = np.array([float(getattr(f, ch)) for f in telemetry.frames], dtype="<f4")
            add_channel(ch, vals)

        max_temps = max((len(f.chamber_temperatures_k) for f in telemetry.frames), default=0)
        if max_temps > 0:
            temps_mat = np.zeros((num_frames, max_temps), dtype="<f4")
            for i, f in enumerate(telemetry.frames):
                if f.chamber_temperatures_k:
                    temps_mat[i, : len(f.chamber_temperatures_k)] = f.chamber_temperatures_k
            add_channel("chamber_temperatures_k", temps_mat)
        else:
            add_channel("chamber_temperatures_k", np.empty((num_frames, 0), dtype="<f4"))

        max_press = max((len(f.chamber_pressures_pa) for f in telemetry.frames), default=0)
        if max_press > 0:
            press_mat = np.zeros((num_frames, max_press), dtype="<f4")
            for i, f in enumerate(telemetry.frames):
                if f.chamber_pressures_pa:
                    press_mat[i, : len(f.chamber_pressures_pa)] = f.chamber_pressures_pa
            add_channel("chamber_pressures_pa", press_mat)
        else:
            add_channel("chamber_pressures_pa", np.empty((num_frames, 0), dtype="<f4"))

        max_flux = max((len(f.shaft_poynting_flux_w_m2) for f in telemetry.frames), default=0)
        if max_flux > 0:
            flux_mat = np.zeros((num_frames, max_flux), dtype="<f4")
            for i, f in enumerate(telemetry.frames):
                if f.shaft_poynting_flux_w_m2:
                    flux_mat[i, : len(f.shaft_poynting_flux_w_m2)] = f.shaft_poynting_flux_w_m2
            add_channel("shaft_poynting_flux_w_m2", flux_mat)
        else:
            add_channel("shaft_poynting_flux_w_m2", np.empty((num_frames, 0), dtype="<f4"))

        if maser_population_keys:
            pop_mat = np.zeros((num_frames, len(maser_population_keys)), dtype="<f4")
            for i, f in enumerate(telemetry.frames):
                for k_idx, k in enumerate(maser_population_keys):
                    pop_mat[i, k_idx] = f.maser_state_populations.get(k, 0.0)
            add_channel("maser_state_populations", pop_mat)
        else:
            add_channel("maser_state_populations", np.empty((num_frames, 0), dtype="<f4"))

        if gallery_z:
            add_channel("gallery_z", np.array(gallery_z, dtype="<f4"))
        if fft_frequencies_hz:
            add_channel("fft_frequencies_hz", np.array(fft_frequencies_hz, dtype="<f4"))

        nz_p = len(gallery_z) if gallery_z else max((len(f.spatial.acoustic_pressure_profile) for f in telemetry.frames), default=0)
        if nz_p > 0:
            p_mat = np.zeros((num_frames, nz_p), dtype="<f4")
            for i, f in enumerate(telemetry.frames):
                if f.spatial.acoustic_pressure_profile:
                    p_mat[i, : len(f.spatial.acoustic_pressure_profile)] = f.spatial.acoustic_pressure_profile
            add_channel("acoustic_pressure_profile", p_mat)

            v_mat = np.zeros((num_frames, nz_p), dtype="<f4")
            for i, f in enumerate(telemetry.frames):
                if f.spatial.acoustic_velocity_profile:
                    v_mat[i, : len(f.spatial.acoustic_velocity_profile)] = f.spatial.acoustic_velocity_profile
            add_channel("acoustic_velocity_profile", v_mat)

            e_mat = np.zeros((num_frames, nz_p), dtype="<f4")
            for i, f in enumerate(telemetry.frames):
                if f.spatial.acoustic_energy_density:
                    e_mat[i, : len(f.spatial.acoustic_energy_density)] = f.spatial.acoustic_energy_density
            add_channel("acoustic_energy_density", e_mat)

        ngas = len(gas_nodes) if gas_nodes else max((len(f.spatial.gas_h2_mole_fractions) for f in telemetry.frames), default=0)
        if ngas > 0:
            gh2_mat = np.zeros((num_frames, ngas), dtype="<f4")
            for i, f in enumerate(telemetry.frames):
                if f.spatial.gas_h2_mole_fractions:
                    gh2_mat[i, : len(f.spatial.gas_h2_mole_fractions)] = f.spatial.gas_h2_mole_fractions
            add_channel("gas_h2_mole_fractions", gh2_mat)

            gc_mat = np.zeros((num_frames, ngas), dtype="<f4")
            for i, f in enumerate(telemetry.frames):
                if f.spatial.gas_sound_speeds:
                    gc_mat[i, : len(f.spatial.gas_sound_speeds)] = f.spatial.gas_sound_speeds
            add_channel("gas_sound_speeds", gc_mat)

            grho_mat = np.zeros((num_frames, ngas), dtype="<f4")
            for i, f in enumerate(telemetry.frames):
                if f.spatial.gas_densities:
                    grho_mat[i, : len(f.spatial.gas_densities)] = f.spatial.gas_densities
            add_channel("gas_densities", grho_mat)

        ntiers = max((len(f.spatial.tier_voltages) for f in telemetry.frames), default=0)
        if ntiers > 0:
            tv_mat = np.zeros((num_frames, ntiers), dtype="<f4")
            for i, f in enumerate(telemetry.frames):
                if f.spatial.tier_voltages:
                    tv_mat[i, : len(f.spatial.tier_voltages)] = f.spatial.tier_voltages
            add_channel("tier_voltages", tv_mat)

        nbeams = max((len(f.spatial.all_beam_stresses_mpa) for f in telemetry.frames), default=0)
        if nbeams > 0:
            bs_mat = np.zeros((num_frames, nbeams), dtype="<f4")
            for i, f in enumerate(telemetry.frames):
                if f.spatial.all_beam_stresses_mpa:
                    bs_mat[i, : len(f.spatial.all_beam_stresses_mpa)] = f.spatial.all_beam_stresses_mpa
            add_channel("all_beam_stresses_mpa", bs_mat)

            bv_mat = np.zeros((num_frames, nbeams), dtype="<f4")
            for i, f in enumerate(telemetry.frames):
                if f.spatial.all_beam_voltages_v:
                    bv_mat[i, : len(f.spatial.all_beam_voltages_v)] = f.spatial.all_beam_voltages_v
            add_channel("all_beam_voltages_v", bv_mat)

        nfft = len(fft_frequencies_hz) if fft_frequencies_hz else max((len(f.spatial.fft_power_spectral_density_db) for f in telemetry.frames), default=0)
        if nfft > 0:
            psd_mat = np.zeros((num_frames, nfft), dtype="<f4")
            for i, f in enumerate(telemetry.frames):
                if f.spatial.fft_power_spectral_density_db:
                    psd_mat[i, : len(f.spatial.fft_power_spectral_density_db)] = f.spatial.fft_power_spectral_density_db
            add_channel("fft_power_spectral_density_db", psd_mat)

        add_channel("north_shaft_power", np.array([float(f.spatial.north_shaft_power) for f in telemetry.frames], dtype="<f4"))
        add_channel("south_shaft_power", np.array([float(f.spatial.south_shaft_power) for f in telemetry.frames], dtype="<f4"))

    payload = b"".join(payload_byte_list)

    header = {
        "format": "pyramid_telemetry_bin_v1",
        "simulation_id": telemetry.simulation_id,
        "version": telemetry.version,
        "scenario_name": telemetry.scenario_name,
        "duration": float(telemetry.duration),
        "dt_macro": float(telemetry.dt_macro),
        "dt_micro": float(telemetry.dt_micro),
        "total_frames": int(num_frames),
        "metadata": _to_serializable(telemetry.metadata),
        "summary": _to_serializable(telemetry.summary),
        "gas_nodes": gas_nodes,
        "maser_population_keys": maser_population_keys,
        "channels": channels_meta,
    }

    header_json = json.dumps(header, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")
    padding_len = (4 - (len(header_bytes) % 4)) % 4
    if padding_len > 0:
        header_bytes += b" " * padding_len

    header_len = len(header_bytes)
    header_len_bytes = struct.pack("<I", header_len)

    with open(p, "wb") as f:
        f.write(header_len_bytes)
        f.write(header_bytes)
        f.write(payload)

    return p


def load_binary(input_path: Union[str, Path]) -> SimulationTelemetry:
    """Load and deserialize simulation telemetry from binary (.bin) file."""
    p = Path(input_path)
    with open(p, "rb") as f:
        file_bytes = f.read()

    if len(file_bytes) < 4:
        raise ValueError(f"Binary telemetry file is truncated: {p}")

    header_len = struct.unpack("<I", file_bytes[:4])[0]
    if len(file_bytes) < 4 + header_len:
        raise ValueError(f"Binary telemetry header corrupted or truncated: {p}")

    header_raw = file_bytes[4 : 4 + header_len].decode("utf-8").strip()
    header = json.loads(header_raw)
    payload_bytes = file_bytes[4 + header_len :]

    num_frames = int(header.get("total_frames", 0))
    channels_meta = header.get("channels", {})
    gas_nodes = header.get("gas_nodes", [])
    maser_pop_keys = header.get("maser_population_keys", [])

    channels: Dict[str, np.ndarray] = {}
    for name, meta in channels_meta.items():
        offset = meta["offset_bytes"]
        shape = tuple(meta["shape"])
        count = meta.get("count", math.prod(shape) if shape else 0)
        if count > 0:
            arr = np.frombuffer(payload_bytes, dtype="<f4", count=count, offset=offset).reshape(shape)
        else:
            arr = np.empty(shape, dtype=np.float32)
        channels[name] = arr

    frames: List[TelemetryFrame] = []
    if num_frames > 0:
        def get_mat_list(name: str) -> Optional[List[List[float]]]:
            if name in channels and channels[name].size > 0:
                return channels[name].tolist()
            return None

        chamber_temps_list = get_mat_list("chamber_temperatures_k")
        chamber_press_list = get_mat_list("chamber_pressures_pa")
        shaft_flux_list = get_mat_list("shaft_poynting_flux_w_m2")
        maser_pop_mat = channels.get("maser_state_populations")

        gallery_z_list = channels["gallery_z"].tolist() if "gallery_z" in channels and channels["gallery_z"].size > 0 else []
        fft_freqs_list = channels["fft_frequencies_hz"].tolist() if "fft_frequencies_hz" in channels and channels["fft_frequencies_hz"].size > 0 else []

        press_prof_list = get_mat_list("acoustic_pressure_profile")
        vel_prof_list = get_mat_list("acoustic_velocity_profile")
        eng_prof_list = get_mat_list("acoustic_energy_density")
        gas_h2_list = get_mat_list("gas_h2_mole_fractions")
        gas_c_list = get_mat_list("gas_sound_speeds")
        gas_rho_list = get_mat_list("gas_densities")
        tier_v_list = get_mat_list("tier_voltages")
        beam_s_list = get_mat_list("all_beam_stresses_mpa")
        beam_v_list = get_mat_list("all_beam_voltages_v")
        fft_psd_list = get_mat_list("fft_power_spectral_density_db")

        north_p_col = channels.get("north_shaft_power")
        south_p_col = channels.get("south_shaft_power")

        scalar_cols: Dict[str, np.ndarray] = {}
        for ch in TELEMETRY_SCALAR_CHANNELS:
            if ch in channels:
                scalar_cols[ch] = channels[ch]

        for i in range(num_frames):
            spatial = SpatialFieldSlice(
                gallery_z=gallery_z_list,
                acoustic_pressure_profile=press_prof_list[i] if press_prof_list is not None else [],
                acoustic_velocity_profile=vel_prof_list[i] if vel_prof_list is not None else [],
                acoustic_energy_density=eng_prof_list[i] if eng_prof_list is not None else [],
                gas_nodes=list(gas_nodes),
                gas_h2_mole_fractions=gas_h2_list[i] if gas_h2_list is not None else [],
                gas_sound_speeds=gas_c_list[i] if gas_c_list is not None else [],
                gas_densities=gas_rho_list[i] if gas_rho_list is not None else [],
                tier_voltages=tier_v_list[i] if tier_v_list is not None else [],
                all_beam_stresses_mpa=beam_s_list[i] if beam_s_list is not None else [],
                all_beam_voltages_v=beam_v_list[i] if beam_v_list is not None else [],
                fft_frequencies_hz=fft_freqs_list,
                fft_power_spectral_density_db=fft_psd_list[i] if fft_psd_list is not None else [],
                north_shaft_power=float(north_p_col[i]) if north_p_col is not None else 0.0,
                south_shaft_power=float(south_p_col[i]) if south_p_col is not None else 0.0,
            )

            maser_pops: Dict[str, float] = {}
            if maser_pop_mat is not None and maser_pop_mat.size > 0 and maser_pop_keys:
                for k_idx, k in enumerate(maser_pop_keys):
                    if k_idx < maser_pop_mat.shape[1]:
                        maser_pops[k] = float(maser_pop_mat[i, k_idx])

            frame = TelemetryFrame(
                time=float(scalar_cols["time"][i]) if "time" in scalar_cols else 0.0,
                step_index=int(round(float(scalar_cols["step_index"][i]))) if "step_index" in scalar_cols else 0,
                bedrock_displacement=float(scalar_cols["bedrock_displacement"][i]) if "bedrock_displacement" in scalar_cols else 0.0,
                bedrock_velocity=float(scalar_cols["bedrock_velocity"][i]) if "bedrock_velocity" in scalar_cols else 0.0,
                bedrock_acceleration=float(scalar_cols["bedrock_acceleration"][i]) if "bedrock_acceleration" in scalar_cols else 0.0,
                water_hammer_pressure=float(scalar_cols["water_hammer_pressure"][i]) if "water_hammer_pressure" in scalar_cols else 0.0,
                seismic_force=float(scalar_cols["seismic_force"][i]) if "seismic_force" in scalar_cols else 0.0,
                hydraulic_force=float(scalar_cols["hydraulic_force"][i]) if "hydraulic_force" in scalar_cols else 0.0,
                schumann_excitation=float(scalar_cols["schumann_excitation"][i]) if "schumann_excitation" in scalar_cols else 0.0,
                acoustic_pressure_sub=float(scalar_cols["acoustic_pressure_sub"][i]) if "acoustic_pressure_sub" in scalar_cols else 0.0,
                h2_mole_fraction_qc=float(scalar_cols["h2_mole_fraction_qc"][i]) if "h2_mole_fraction_qc" in scalar_cols else 0.0,
                h2_mole_fraction_kc=float(scalar_cols["h2_mole_fraction_kc"][i]) if "h2_mole_fraction_kc" in scalar_cols else 0.0,
                chemical_reaction_rate=float(scalar_cols["chemical_reaction_rate"][i]) if "chemical_reaction_rate" in scalar_cols else 0.0,
                qc_chamber_temperature_k=float(scalar_cols["qc_chamber_temperature_k"][i]) if "qc_chamber_temperature_k" in scalar_cols else 293.15,
                cumulative_h2_moles=float(scalar_cols["cumulative_h2_moles"][i]) if "cumulative_h2_moles" in scalar_cols else 0.0,
                qc_heat_release_w=float(scalar_cols["qc_heat_release_w"][i]) if "qc_heat_release_w" in scalar_cols else 0.0,
                chamber_temperatures_k=chamber_temps_list[i] if chamber_temps_list is not None else [],
                chamber_pressures_pa=chamber_press_list[i] if chamber_press_list is not None else [],
                gallery_peak_pressure=float(scalar_cols["gallery_peak_pressure"][i]) if "gallery_peak_pressure" in scalar_cols else 0.0,
                gallery_rms_pressure=float(scalar_cols["gallery_rms_pressure"][i]) if "gallery_rms_pressure" in scalar_cols else 0.0,
                gallery_sound_speed_avg=float(scalar_cols["gallery_sound_speed_avg"][i]) if "gallery_sound_speed_avg" in scalar_cols else 343.2,
                gallery_total_acoustic_energy=float(scalar_cols["gallery_total_acoustic_energy"][i]) if "gallery_total_acoustic_energy" in scalar_cols else 0.0,
                f_sharp_spectral_purity=float(scalar_cols["f_sharp_spectral_purity"][i]) if "f_sharp_spectral_purity" in scalar_cols else 0.0,
                top_pressure_kc_entry=float(scalar_cols["top_pressure_kc_entry"][i]) if "top_pressure_kc_entry" in scalar_cols else 0.0,
                antechamber_p_in=float(scalar_cols["antechamber_p_in"][i]) if "antechamber_p_in" in scalar_cols else 0.0,
                antechamber_p_out=float(scalar_cols["antechamber_p_out"][i]) if "antechamber_p_out" in scalar_cols else 0.0,
                antechamber_transmission_loss_db=float(scalar_cols["antechamber_transmission_loss_db"][i]) if "antechamber_transmission_loss_db" in scalar_cols else 0.0,
                antechamber_p_trans=float(scalar_cols["antechamber_p_trans"][i]) if "antechamber_p_trans" in scalar_cols else 0.0,
                total_piezo_voltage=float(scalar_cols["total_piezo_voltage"][i]) if "total_piezo_voltage" in scalar_cols else 0.0,
                total_piezo_charge=float(scalar_cols["total_piezo_charge"][i]) if "total_piezo_charge" in scalar_cols else 0.0,
                displacement_current_a=float(scalar_cols["displacement_current_a"][i]) if "displacement_current_a" in scalar_cols else 0.0,
                beam_array_impedance_ohms=float(scalar_cols["beam_array_impedance_ohms"][i]) if "beam_array_impedance_ohms" in scalar_cols else 0.0,
                total_mechanical_energy=float(scalar_cols["total_mechanical_energy"][i]) if "total_mechanical_energy" in scalar_cols else 0.0,
                total_electrostatic_energy=float(scalar_cols["total_electrostatic_energy"][i]) if "total_electrostatic_energy" in scalar_cols else 0.0,
                max_beam_stress_pa=float(scalar_cols["max_beam_stress_pa"][i]) if "max_beam_stress_pa" in scalar_cols else 0.0,
                spark_triggered=bool(float(scalar_cols["spark_triggered"][i]) > 0.5) if "spark_triggered" in scalar_cols else False,
                spark_count=int(round(float(scalar_cols["spark_count"][i]))) if "spark_count" in scalar_cols else 0,
                ion_density=float(scalar_cols["ion_density"][i]) if "ion_density" in scalar_cols else 0.0,
                maser_total_radiated_power=float(scalar_cols["maser_total_radiated_power"][i]) if "maser_total_radiated_power" in scalar_cols else 0.0,
                effective_radiated_power_w=float(scalar_cols["effective_radiated_power_w"][i]) if "effective_radiated_power_w" in scalar_cols else 0.0,
                maser_population_inversion=float(scalar_cols["maser_population_inversion"][i]) if "maser_population_inversion" in scalar_cols else 0.0,
                maser_photon_energy_density=float(scalar_cols["maser_photon_energy_density"][i]) if "maser_photon_energy_density" in scalar_cols else 0.0,
                maser_pumping_rate=float(scalar_cols["maser_pumping_rate"][i]) if "maser_pumping_rate" in scalar_cols else 0.0,
                maser_is_above_threshold=bool(float(scalar_cols["maser_is_above_threshold"][i]) > 0.5) if "maser_is_above_threshold" in scalar_cols else False,
                maser_north_beam_power=float(scalar_cols["maser_north_beam_power"][i]) if "maser_north_beam_power" in scalar_cols else 0.0,
                maser_south_beam_power=float(scalar_cols["maser_south_beam_power"][i]) if "maser_south_beam_power" in scalar_cols else 0.0,
                shaft_poynting_flux_w_m2=shaft_flux_list[i] if shaft_flux_list is not None else [],
                maser_state_populations=maser_pops,
                maser_cumulative_radiated_energy=float(scalar_cols["maser_cumulative_radiated_energy"][i]) if "maser_cumulative_radiated_energy" in scalar_cols else 0.0,
                p_total_in=float(scalar_cols["p_total_in"][i]) if "p_total_in" in scalar_cols else 0.0,
                p_total_out=float(scalar_cols["p_total_out"][i]) if "p_total_out" in scalar_cols else 0.0,
                p_total_loss=float(scalar_cols["p_total_loss"][i]) if "p_total_loss" in scalar_cols else 0.0,
                cumulative_energy_in=float(scalar_cols["cumulative_energy_in"][i]) if "cumulative_energy_in" in scalar_cols else 0.0,
                cumulative_energy_out=float(scalar_cols["cumulative_energy_out"][i]) if "cumulative_energy_out" in scalar_cols else 0.0,
                cumulative_energy_loss=float(scalar_cols["cumulative_energy_loss"][i]) if "cumulative_energy_loss" in scalar_cols else 0.0,
                total_stored_energy=float(scalar_cols["total_stored_energy"][i]) if "total_stored_energy" in scalar_cols else 0.0,
                delta_stored_energy=float(scalar_cols["delta_stored_energy"][i]) if "delta_stored_energy" in scalar_cols else 0.0,
                net_work=float(scalar_cols["net_work"][i]) if "net_work" in scalar_cols else 0.0,
                energy_balance_error=float(scalar_cols["energy_balance_error"][i]) if "energy_balance_error" in scalar_cols else 0.0,
                relative_energy_error=float(scalar_cols["relative_energy_error"][i]) if "relative_energy_error" in scalar_cols else 0.0,
                is_energy_conserved=bool(float(scalar_cols["is_energy_conserved"][i]) > 0.5) if "is_energy_conserved" in scalar_cols else True,
                spatial=spatial,
            )
            frames.append(frame)

    telemetry = SimulationTelemetry(
        simulation_id=header.get("simulation_id", "pyramid_powerplant_sim"),
        version=header.get("version", "1.0.0"),
        scenario_name=header.get("scenario_name", "baseline"),
        duration=float(header.get("duration", 0.0)),
        dt_macro=float(header.get("dt_macro", 0.01)),
        dt_micro=float(header.get("dt_micro", 0.0001)),
        total_frames=num_frames,
        metadata=dict(header.get("metadata", {})),
        summary=dict(header.get("summary", {})),
        frames=frames,
    )
    return telemetry


class TelemetryExporter:
    """Exporter orchestrating recording intervals, compression, and file saving."""

    def __init__(
        self,
        output_rate_hz: float = 60.0,
        enable_spatial_slices: bool = True,
        spatial_decimation: int = 1,
    ) -> None:
        self.output_rate_hz = max(0.1, float(output_rate_hz))
        self.frame_interval = 1.0 / self.output_rate_hz
        self.enable_spatial_slices = enable_spatial_slices
        self.spatial_decimation = max(1, int(spatial_decimation))
        self._last_record_time: float = -1.0e9

    def should_record_frame(self, current_time: float) -> bool:
        """Determine if a frame should be recorded at current_time."""
        if current_time - self._last_record_time >= (self.frame_interval - 1.0e-9):
            self._last_record_time = current_time
            return True
        return False

    def export(
        self,
        telemetry: SimulationTelemetry,
        filepath: Union[str, Path],
        compress: bool = False,
        indent: Optional[int] = None,
    ) -> Path:
        """Export telemetry to JSON, NPZ, or BIN depending on file extension."""
        p = Path(filepath)
        telemetry.compute_summary()
        if p.suffix == ".npz":
            return telemetry.save_npz(p)
        if p.suffix == ".bin":
            return telemetry.save_binary(p)
        return telemetry.save_json(p, compress=compress, indent=indent)

    def export_binary(
        self,
        telemetry: SimulationTelemetry,
        filepath: Union[str, Path],
    ) -> Path:
        """Export telemetry directly to packed binary (.bin) format."""
        return export_binary(telemetry, filepath)
