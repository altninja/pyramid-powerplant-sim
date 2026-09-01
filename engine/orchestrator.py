"""Coupled Multi-Scale Simulation Orchestrator for Giza Power Plant.

Coordinates all 6 physical modules into a unified, time-synchronous pipeline:
1. Subterranean Hydraulics (Schumann seismic drive & water hammer pulses)
2. Queen's Chamber Chemistry (Zn + HCl reaction kinetics & 5-node gas transport)
3. Grand Gallery Acoustics (F# Helmholtz resonator array & 1D wave solver)
4. Antechamber Acoustic Filter (TMM acoustic gating & harmonic bandpass)
5. King's Chamber Piezoelectric Beams (43 rose granite beams & spark discharge)
6. King's Chamber Microwave Maser (atomic H population inversion & shaft horn beaming)
7. Master Energy Balance Accountant (continuous First Law audit & power flows)

Features multi-rate time stepping:
- Slow outer loop (Delta t_macro ~ 5-10 ms) for chemistry, gas diffusion, thermodynamics
- Fast inner loop (Delta t_micro ~ 0.1-0.2 ms) for acoustic wave propagation & beam dynamics
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from engine.config import SimulationConfig
from engine.geometry import PyramidGeometry
from engine.physics.antechamber_filter import AntechamberFilter, AntechamberState
from engine.physics.chemical_gas_transport import ChemicalGasTransport, GasTransportState
from engine.physics.energy_accountant import (
    EnergyAccountant,
    EnergyBalanceSnapshot,
    PowerFlowState,
)
from engine.physics.grand_gallery_acoustics import (
    GalleryAcousticState,
    GrandGalleryAcoustics,
)
from engine.physics.microwave_maser import MaserState, MicrowaveMaser
from engine.physics.piezoelectric_beams import (
    PiezoelectricBeams,
    PiezoelectricState,
)
from engine.physics.schumann_hydraulics import (
    HydraulicState,
    SubterraneanHydraulics,
)
from engine.telemetry import (
    SimulationTelemetry,
    SpatialFieldSlice,
    TelemetryExporter,
    TelemetryFrame,
    export_binary,
)


@dataclass
class OrchestratorConfig:
    """Configuration parameters for coupled multi-scale simulation."""

    scenario_name: str = "baseline"
    duration_s: float = 10.0
    dt_macro: float = 0.01
    dt_micro: float = 0.0001
    telemetry_fps: float = 60.0
    energy_tolerance: float = 1.0e-3
    enable_spatial_slices: bool = True
    spatial_decimation: int = 1


class SimulationOrchestrator:
    """Master multi-physics simulation orchestrator synchronizing all 6 subsystems."""

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        geometry: Optional[PyramidGeometry] = None,
        orchestrator_config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self.geometry = geometry or PyramidGeometry()
        self.orch_cfg = orchestrator_config or OrchestratorConfig()

        self._init_subsystems()
        self.reset()

    def _init_subsystems(self) -> None:
        """Initialize all 6 physical sub-modules and the energy accountant."""
        self.hydraulics = SubterraneanHydraulics(
            config=self.config,
            geometry=self.geometry,
        )
        self.gas_transport = ChemicalGasTransport(
            config=self.config,
            geometry=self.geometry,
        )
        self.gallery_acoustics = GrandGalleryAcoustics(
            config=self.config,
            geometry=self.geometry,
        )
        self.antechamber_filter = AntechamberFilter(
            config=self.config,
            geometry=self.geometry,
        )
        self.piezo_beams = PiezoelectricBeams(
            config=self.config,
            geometry=self.geometry,
        )
        self.microwave_maser = MicrowaveMaser(
            config=self.config,
            geometry=self.geometry,
        )
        self.energy_accountant = EnergyAccountant(
            tolerance=self.orch_cfg.energy_tolerance,
        )
        self.exporter = TelemetryExporter(
            output_rate_hz=self.orch_cfg.telemetry_fps,
            enable_spatial_slices=self.orch_cfg.enable_spatial_slices,
            spatial_decimation=self.orch_cfg.spatial_decimation,
        )
        self.max_fft_buffer_size: int = 4096
        self._acoustic_buffer: List[float] = []

    def reset(self) -> None:
        """Reset simulation state and all subsystems to initial baseline."""
        self.time: float = 0.0
        self.step_index: int = 0
        self._acoustic_buffer = []

        self.hydraulics.reset() if hasattr(self.hydraulics, "reset") else None
        self.gas_transport.reset()
        self.gallery_acoustics.reset()
        self.antechamber_filter.reset()
        self.piezo_beams.reset()
        self.microwave_maser.reset()

        initial_stored = self._compute_total_stored_energy()
        self.energy_accountant.reset(initial_stored_energy=initial_stored)

        self.latest_hydraulic_state: Optional[HydraulicState] = None
        self.latest_gas_state: Optional[GasTransportState] = None
        self.latest_gallery_state: Optional[GalleryAcousticState] = None
        self.latest_antechamber_state: Optional[AntechamberState] = None
        self.latest_piezo_state: Optional[PiezoelectricState] = None
        self.latest_maser_state: Optional[MaserState] = None
        self.latest_energy_snapshot: Optional[EnergyBalanceSnapshot] = None
        self.latest_frame: Optional[TelemetryFrame] = None

    def _compute_fft_spectrum(
        self,
        num_bins: int = 128,
        max_freq_hz: float = 2000.0,
    ) -> Tuple[List[float], List[float]]:
        target_freqs = np.linspace(0.0, max_freq_hz, num_bins)
        if len(self._acoustic_buffer) < 8:
            return target_freqs.tolist(), [0.0] * num_bins

        data = np.array(self._acoustic_buffer, dtype=np.float64)
        n_pts = min(len(data), self.max_fft_buffer_size)
        segment = data[-n_pts:]

        segment = segment - np.mean(segment)
        window = np.hanning(len(segment))
        windowed_segment = segment * window

        dt = self.orch_cfg.dt_micro if self.orch_cfg.dt_micro > 0.0 else 0.0001
        fs = 1.0 / dt

        n_fft = max(2048, len(windowed_segment))
        fft_vals = np.fft.rfft(windowed_segment, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=dt)

        win_energy = float(np.sum(window ** 2))
        if win_energy <= 0.0:
            win_energy = 1.0
        psd = (np.abs(fft_vals) ** 2) / (fs * win_energy)
        psd_db = 10.0 * np.log10(np.maximum(psd, 1.0e-12))
        interpolated_psd_db = np.interp(target_freqs, freqs, psd_db)

        clean_psd_db = [
            0.0 if (math.isnan(v) or math.isinf(v)) else float(v)
            for v in interpolated_psd_db
        ]

        return target_freqs.tolist(), clean_psd_db

    def _compute_total_stored_energy(self) -> float:
        """Compute current sum of stored energies across all physical domains."""
        m_bed = self.hydraulics.bedrock_mass
        k_bed = self.hydraulics.stiffness_k
        v_bed = self.hydraulics.velocity
        x_bed = self.hydraulics.displacement
        e_hyd = 0.5 * m_bed * (v_bed ** 2) + 0.5 * k_bed * (x_bed ** 2)

        c_th = self.gas_transport.chamber_thermal_capacity
        e_chem = c_th * max(0.0, self.gas_transport.qc_temperature_k - self.gas_transport.T_amb)

        e_ac = self.gallery_acoustics.compute_total_acoustic_energy()

        e_pz = sum(b.mechanical_energy() for b in self.piezo_beams.all_beams) + (
            0.5 * self.piezo_beams.total_capacitance * (self.piezo_beams.compute_total_voltage() ** 2)
        )

        e_ms = self.microwave_maser.photon_energy_density * self.microwave_maser.chamber_volume

        return float(e_hyd + e_chem + e_ac + e_pz + e_ms)

    @classmethod
    def create_scenario(
        cls,
        scenario_name: str,
        config: Optional[SimulationConfig] = None,
        duration: float = 10.0,
        dt_macro: float = 0.01,
        dt_micro: float = 0.0001,
        telemetry_fps: float = 60.0,
    ) -> SimulationOrchestrator:
        """Factory method to configure specialized scenario presets."""
        cfg = config or SimulationConfig()
        orch_cfg = OrchestratorConfig(
            scenario_name=scenario_name,
            duration_s=duration,
            dt_macro=dt_macro,
            dt_micro=dt_micro,
            telemetry_fps=telemetry_fps,
        )
        orch = cls(config=cfg, orchestrator_config=orch_cfg)

        name = scenario_name.lower().strip()
        if name in ("baseline", "standard"):
            pass
        elif name == "acoustic_peak":
            orch.gallery_acoustics.coupling_gain = 1.5
            for res in orch.gallery_acoustics.resonator_bank.resonators:
                res.quality_factor = 100.0
            orch.hydraulics.seismic_force_amplitude = 2.0e5
        elif name == "full_maser_power":
            orch.hydraulics.seismic_force_amplitude = 3.0e5
            orch.gas_transport.initial_zn_moles = 5000.0
            orch.gas_transport.initial_hcl_moles = 10000.0
            orch.gas_transport.reset(initial_zn_moles=5000.0, initial_hcl_moles=10000.0)
            orch.microwave_maser.coupling_kappa_elec = 25.0
            orch.microwave_maser.coupling_kappa_acoust = 15.0
        elif name == "dry_run_no_gas":
            orch.gas_transport.initial_zn_moles = 0.0
            orch.gas_transport.initial_hcl_moles = 0.0
            orch.gas_transport.reset(initial_zn_moles=0.0, initial_hcl_moles=0.0)
        elif name == "high_seismic":
            orch.hydraulics.seismic_force_amplitude = 5.0e5
        elif name == "resonance_sweep":
            orch.hydraulics.f0 = 8.14
            orch.hydraulics.omega0 = 2.0 * math.pi * 8.14
            orch.hydraulics.stiffness_k = orch.hydraulics.bedrock_mass * (orch.hydraulics.omega0 ** 2)
            orch.hydraulics.damping_c = 2.0 * orch.hydraulics.zeta * orch.hydraulics.bedrock_mass * orch.hydraulics.omega0
        elif name == "transient_shock":
            orch.gallery_acoustics.inject_pulse(amplitude=500.0, center_z=orch.gallery_acoustics.length * 0.5, width=2.0)
        else:
            raise ValueError(f"Unknown scenario preset: '{scenario_name}'")

        initial_e0 = orch._compute_total_stored_energy()
        orch.energy_accountant.reset(initial_stored_energy=initial_e0)

        return orch

    def step(self, dt_macro: Optional[float] = None) -> TelemetryFrame:
        """Advance the coupled multi-physics simulation by one macro time step."""
        dt = float(dt_macro if dt_macro is not None else self.orch_cfg.dt_macro)
        if dt <= 0.0:
            raise ValueError(f"dt_macro must be positive, got {dt}")

        dt_micro = min(self.orch_cfg.dt_micro, dt)
        num_micro_steps = max(1, int(math.ceil(dt / dt_micro)))
        delta_t = dt / num_micro_steps

        e_chem_prev = self.gas_transport.chamber_thermal_capacity * max(
            0.0, self.gas_transport.qc_temperature_k - self.gas_transport.T_amb
        )
        gas_state = self.gas_transport.step(dt)
        self.latest_gas_state = gas_state
        e_chem_curr = self.gas_transport.chamber_thermal_capacity * max(
            0.0, gas_state.reaction.chamber_temperature_k - self.gas_transport.T_amb
        )

        gg_node = gas_state.get_node("grand_gallery")
        kc_node = gas_state.get_node("kings_chamber")
        qc_node = gas_state.get_node("queens_chamber")

        self.gallery_acoustics.set_gas_properties(
            sound_speed=gg_node.sound_speed_m_per_s,
            gas_density=gg_node.gas_density_kg_per_m3,
        )

        c_gg = gg_node.sound_speed_m_per_s
        rho_gg = gg_node.gas_density_kg_per_m3
        x_h2_gg = gg_node.h2_mole_fraction

        z_rock = self.hydraulics.limestone_acoustic_impedance
        z_gas = rho_gg * c_gg
        t_trans = (2.0 * z_gas) / (z_rock + z_gas)

        x_h2_kc = kc_node.h2_mole_fraction
        t_kc = kc_node.temperature_k
        p_kc_ambient = kc_node.pressure_pa

        for micro_idx in range(1, num_micro_steps + 1):
            frac = micro_idx / num_micro_steps
            e_chem_interp = e_chem_prev + frac * (e_chem_curr - e_chem_prev)

            t_curr = self.hydraulics.time
            _, f_seis_prev, _, f_hyd_prev, _ = self.hydraulics.compute_driving_forces(t_curr)
            v_bed_prev = self.hydraulics.velocity

            hyd_state = self.hydraulics.step(delta_t)
            v_bed_curr = self.hydraulics.velocity
            f_seis_curr = hyd_state.seismic_force
            f_hyd_curr = hyd_state.hydraulic_force

            p_seis_in = 0.5 * (f_seis_prev * v_bed_prev + f_seis_curr * v_bed_curr)
            p_hyd_in = 0.5 * (f_hyd_prev * v_bed_prev + f_hyd_curr * v_bed_curr)
            p_loss_hyd = 0.5 * self.hydraulics.damping_c * (v_bed_prev ** 2 + v_bed_curr ** 2)

            p_asc = hyd_state.acoustic_pressure_ascending_passage * t_trans
            e_ac_prev = self.gallery_acoustics.compute_total_acoustic_energy()
            gg_state = self.gallery_acoustics.step(delta_t, bottom_pressure_drive=p_asc)
            e_ac_curr = gg_state.total_acoustic_energy
            p_gg_top = gg_state.top_pressure
            p_ac_work = (e_ac_curr - e_ac_prev) / delta_t

            self._acoustic_buffer.append(float(p_gg_top + p_asc))

            ac_state = self.antechamber_filter.step(
                p_gg=p_gg_top,
                dt=delta_t,
                sound_speed=c_gg,
                density=rho_gg,
                h2_fraction=x_h2_gg,
            )
            p_kc_acoustic = ac_state.p_out

            pz_state = self.piezo_beams.step(delta_t, p_kc_acoustic=p_kc_acoustic)
            v_piezo = pz_state.total_voltage

            ms_state = self.microwave_maser.step(
                dt=delta_t,
                piezo_voltage=v_piezo,
                acoustic_pressure=p_kc_acoustic,
                h2_concentration=x_h2_kc,
                temperature_k=t_kc,
                pressure_pa=p_kc_ambient,
            )

            p_chem_in = gas_state.reaction.heat_release_rate_watts
            p_maser_rad = ms_state.total_radiated_power
            p_loss_pz = pz_state.damping_power_loss
            p_loss_spark = (
                (pz_state.spark_energy / delta_t)
                if (pz_state.spark_triggered and delta_t > 0)
                else 0.0
            )
            p_loss_cavity = ms_state.cavity_loss_power
            p_loss_shaft = max(0.0, ms_state.shaft_extracted_power - ms_state.total_radiated_power)
            p_loss_thermal = self.gas_transport.chamber_heat_loss_coeff * max(
                0.0, gas_state.reaction.chamber_temperature_k - self.gas_transport.T_amb
            )

            m_bed = self.hydraulics.bedrock_mass
            k_bed = self.hydraulics.stiffness_k
            x_bed = hyd_state.bedrock_displacement
            e_hyd = 0.5 * m_bed * (v_bed_curr ** 2) + 0.5 * k_bed * (x_bed ** 2)
            e_chem = e_chem_interp
            e_ac = e_ac_curr
            e_pz = pz_state.total_mechanical_energy + pz_state.stored_electrical_energy
            e_ms = ms_state.photon_energy_density * self.microwave_maser.chamber_volume

            energy_snap = self.energy_accountant.step(
                dt=delta_t,
                p_seismic=p_seis_in,
                p_hydraulic=p_hyd_in,
                p_chemical=p_chem_in,
                p_acoustic_in=p_ac_work,
                p_piezo_in=pz_state.acoustic_input_power,
                p_maser_in=ms_state.stimulated_power_total + ms_state.spontaneous_power_total,
                p_acoustic_transfer=gg_state.exit_power_flux,
                p_piezo_transfer=pz_state.acoustic_input_power,
                p_maser_radiated=p_maser_rad,
                p_hydraulic_loss=p_loss_hyd,
                p_acoustic_loss=0.0,
                p_beam_damping_loss=p_loss_pz,
                p_spark_loss=p_loss_spark,
                p_cavity_loss=p_loss_cavity,
                p_shaft_loss=p_loss_shaft,
                p_thermal_loss=p_loss_thermal,
                e_stored_hydraulic=e_hyd,
                e_stored_chemical_thermal=e_chem,
                e_stored_acoustic=e_ac,
                e_stored_beams=e_pz,
                e_stored_maser=e_ms,
            )

            self.latest_hydraulic_state = hyd_state
            self.latest_gallery_state = gg_state
            self.latest_antechamber_state = ac_state
            self.latest_piezo_state = pz_state
            self.latest_maser_state = ms_state
            self.latest_energy_snapshot = energy_snap

        if len(self._acoustic_buffer) > self.max_fft_buffer_size:
            self._acoustic_buffer = self._acoustic_buffer[-self.max_fft_buffer_size:]

        self.time += dt
        self.step_index += 1

        spatial = SpatialFieldSlice()
        if self.orch_cfg.enable_spatial_slices:
            if self.latest_gallery_state is not None:
                dec = self.orch_cfg.spatial_decimation
                spatial.gallery_z = self.latest_gallery_state.grid_z[::dec].tolist()
                spatial.acoustic_pressure_profile = self.latest_gallery_state.pressure[::dec].tolist()
                spatial.acoustic_velocity_profile = self.latest_gallery_state.velocity[::dec].tolist()
                spatial.acoustic_energy_density = self.latest_gallery_state.energy_density[::dec].tolist()
            if gas_state is not None:
                spatial.gas_nodes = [n.name for n in gas_state.nodes]
                spatial.gas_h2_mole_fractions = [float(n.h2_mole_fraction) for n in gas_state.nodes]
                spatial.gas_sound_speeds = [float(n.sound_speed_m_per_s) for n in gas_state.nodes]
                spatial.gas_densities = [float(n.gas_density_kg_per_m3) for n in gas_state.nodes]
            if self.latest_piezo_state is not None:
                spatial.tier_voltages = list(self.latest_piezo_state.tier_voltages)
                spatial.all_beam_stresses_mpa = [float(x) for x in self.latest_piezo_state.all_beam_stresses_mpa]
                spatial.all_beam_voltages_v = [float(x) for x in self.latest_piezo_state.all_beam_voltages_v]
            if self.latest_maser_state is not None:
                spatial.north_shaft_power = float(self.latest_maser_state.north_shaft_beam_power)
                spatial.south_shaft_power = float(self.latest_maser_state.south_shaft_beam_power)

            fft_freqs, fft_psd = self._compute_fft_spectrum()
            spatial.fft_frequencies_hz = fft_freqs
            spatial.fft_power_spectral_density_db = fft_psd

        hyd = self.latest_hydraulic_state
        gg = self.latest_gallery_state
        ac = self.latest_antechamber_state
        pz = self.latest_piezo_state
        ms = self.latest_maser_state
        eng = self.latest_energy_snapshot

        node_names = ("queens_chamber", "horizontal_passage", "ascending_passage", "grand_gallery", "kings_chamber")
        chamber_temps = [float(gas_state.get_node(n).temperature_k) for n in node_names]
        chamber_pressures = [float(gas_state.get_node(n).pressure_pa) for n in node_names]

        n1 = float(self.microwave_maser.n1_population)
        n2 = float(self.microwave_maser.n2_population)
        delta_n = float(self.microwave_maser.population_inversion)
        n_total = float(self.microwave_maser.total_h_density)
        maser_pops = {
            "n1": n1,
            "n2": n2,
            "delta_n": delta_n,
            "n_total": n_total,
        }

        a_north = self.microwave_maser.north_shaft.cross_section_area
        a_south = self.microwave_maser.south_shaft.cross_section_area
        p_north = float(ms.north_shaft_beam_power) if ms else 0.0
        p_south = float(ms.south_shaft_beam_power) if ms else 0.0
        s_north = (p_north / a_north) if a_north > 0.0 else 0.0
        s_south = (p_south / a_south) if a_south > 0.0 else 0.0
        shaft_flux = [float(s_north), float(s_south)]

        erp_w = float(ms.total_erp_watts) if ms else 0.0
        disp_curr = float(pz.total_displacement_current_a) if pz else 0.0
        array_z = float(pz.array_impedance_ohms) if pz else 0.0

        frame = TelemetryFrame(
            time=self.time,
            step_index=self.step_index,
            bedrock_displacement=hyd.bedrock_displacement if hyd else 0.0,
            bedrock_velocity=hyd.bedrock_velocity if hyd else 0.0,
            bedrock_acceleration=hyd.bedrock_acceleration if hyd else 0.0,
            water_hammer_pressure=hyd.water_hammer_pressure if hyd else 0.0,
            seismic_force=hyd.seismic_force if hyd else 0.0,
            hydraulic_force=hyd.hydraulic_force if hyd else 0.0,
            schumann_excitation=hyd.schumann_excitation if hyd else 0.0,
            acoustic_pressure_sub=hyd.acoustic_pressure_ascending_passage if hyd else 0.0,
            h2_mole_fraction_qc=qc_node.h2_mole_fraction,
            h2_mole_fraction_kc=kc_node.h2_mole_fraction,
            chemical_reaction_rate=gas_state.reaction.reaction_rate_mol_per_s,
            qc_chamber_temperature_k=gas_state.reaction.chamber_temperature_k,
            cumulative_h2_moles=gas_state.reaction.h2_moles_generated_total,
            qc_heat_release_w=gas_state.reaction.heat_release_rate_watts,
            chamber_temperatures_k=chamber_temps,
            chamber_pressures_pa=chamber_pressures,
            gallery_peak_pressure=gg.peak_pressure if gg else 0.0,
            gallery_rms_pressure=gg.rms_pressure if gg else 0.0,
            gallery_sound_speed_avg=float(np.mean(gg.sound_speed)) if gg else 343.2,
            gallery_total_acoustic_energy=gg.total_acoustic_energy if gg else 0.0,
            f_sharp_spectral_purity=min(1.0, float(np.sum(gg.resonator_energies) / max(gg.total_acoustic_energy, 1.0e-9))) if (gg and gg.total_acoustic_energy > 0) else 0.0,
            top_pressure_kc_entry=gg.top_pressure if gg else 0.0,
            antechamber_p_in=ac.p_in if ac else 0.0,
            antechamber_p_out=ac.p_out if ac else 0.0,
            antechamber_transmission_loss_db=ac.transmission_loss_db_438 if ac else 0.0,
            antechamber_p_trans=ac.p_trans if ac else 0.0,
            total_piezo_voltage=pz.total_voltage if pz else 0.0,
            total_piezo_charge=pz.total_charge if pz else 0.0,
            displacement_current_a=disp_curr,
            beam_array_impedance_ohms=array_z,
            total_mechanical_energy=pz.total_mechanical_energy if pz else 0.0,
            total_electrostatic_energy=pz.stored_electrical_energy if pz else 0.0,
            max_beam_stress_pa=pz.max_fiber_stress if pz else 0.0,
            spark_triggered=pz.spark_triggered if pz else False,
            spark_count=pz.spark_count if pz else 0,
            ion_density=pz.ion_density if pz else 0.0,
            maser_total_radiated_power=ms.total_radiated_power if ms else 0.0,
            effective_radiated_power_w=erp_w,
            maser_population_inversion=ms.population_inversion if ms else 0.0,
            maser_photon_energy_density=ms.photon_energy_density if ms else 0.0,
            maser_pumping_rate=ms.pumping_rate if ms else 0.0,
            maser_is_above_threshold=ms.is_above_threshold if ms else False,
            maser_north_beam_power=ms.north_shaft_beam_power if ms else 0.0,
            maser_south_beam_power=ms.south_shaft_beam_power if ms else 0.0,
            shaft_poynting_flux_w_m2=shaft_flux,
            maser_state_populations=maser_pops,
            maser_cumulative_radiated_energy=ms.cumulative_radiated_energy if ms else 0.0,
            p_total_in=eng.power_flow.p_total_in if eng else 0.0,
            p_total_out=eng.power_flow.p_total_out if eng else 0.0,
            p_total_loss=eng.power_flow.p_total_loss if eng else 0.0,
            cumulative_energy_in=eng.cumulative_energy_in if eng else 0.0,
            cumulative_energy_out=eng.cumulative_energy_out if eng else 0.0,
            cumulative_energy_loss=eng.cumulative_energy_loss if eng else 0.0,
            total_stored_energy=eng.total_stored_energy if eng else 0.0,
            delta_stored_energy=eng.delta_stored_energy if eng else 0.0,
            net_work=eng.net_work if eng else 0.0,
            energy_balance_error=eng.energy_balance_error if eng else 0.0,
            relative_energy_error=eng.relative_error if eng else 0.0,
            is_energy_conserved=eng.is_conserved if eng else True,
            spatial=spatial,
        )

        self.latest_frame = frame
        return frame

    def run(
        self,
        duration: Optional[float] = None,
        dt_macro: Optional[float] = None,
        dt_micro: Optional[float] = None,
        progress_callback: Optional[Callable[[float, float], None]] = None,
    ) -> SimulationTelemetry:
        """Run full continuous multi-physics simulation and collect telemetry."""
        total_duration = float(duration if duration is not None else self.orch_cfg.duration_s)
        macro_dt = float(dt_macro if dt_macro is not None else self.orch_cfg.dt_macro)
        micro_dt = float(dt_micro if dt_micro is not None else self.orch_cfg.dt_micro)

        self.orch_cfg.dt_macro = macro_dt
        self.orch_cfg.dt_micro = micro_dt
        self.orch_cfg.duration_s = total_duration

        if self.time == 0.0:
            e0 = self._compute_total_stored_energy()
            self.energy_accountant.set_initial_stored_energy(e0)

        telemetry = SimulationTelemetry(
            simulation_id=f"pyramid_sim_{self.orch_cfg.scenario_name}",
            scenario_name=self.orch_cfg.scenario_name,
            duration=total_duration,
            dt_macro=macro_dt,
            dt_micro=micro_dt,
            metadata={
                "scenario": self.orch_cfg.scenario_name,
                "pyramid_base_side_m": self.geometry.mean_base_side,
                "pyramid_height_m": self.geometry.height,
                "schumann_f1_hz": self.config.schumann.mode1_frequency,
                "hyperfine_freq_hz": self.microwave_maser.hyperfine_frequency,
            },
        )

        num_steps = max(1, int(math.ceil(total_duration / macro_dt)))

        for step_idx in range(num_steps):
            frame = self.step(macro_dt)
            if self.exporter.should_record_frame(frame.time):
                telemetry.add_frame(frame)

            if progress_callback is not None and step_idx % 10 == 0:
                progress_callback(frame.time, total_duration)

        telemetry.compute_summary()
        return telemetry

    def export_binary(self, filepath: Union[str, Path], telemetry: Optional[SimulationTelemetry] = None) -> Path:
        """Export simulation telemetry to packed little-endian binary (.bin) format."""
        if telemetry is None:
            telemetry = self.run()
        return export_binary(telemetry, filepath)

    def export_json(
        self,
        filepath: Union[str, Path],
        telemetry: Optional[SimulationTelemetry] = None,
        compress: bool = False,
        indent: Optional[int] = None,
    ) -> Path:
        """Export simulation telemetry to JSON format."""
        if telemetry is None:
            telemetry = self.run()
        return self.exporter.export(telemetry, filepath, compress=compress, indent=indent)
