"""Grand Gallery Acoustic Wave Equation & Helmholtz Resonator Arrays.

This module implements:
1. 1D spatial discretized acoustic wave propagation in the Grand Gallery
   coupled to variable gas composition (sound speed c(z) and density rho(z)).
2. 27-station Helmholtz resonator bank tuned to the F# harmonic series
   (438 Hz, 876 Hz, 1314 Hz, 1752 Hz).
3. Coupled oscillator-wave interaction where resonators amplify and focus
   acoustic power into the standing wave field.
4. Boundary condition management (infrasonic driving at Ascending Passage junction,
   impedance matching / radiation at Antechamber portal).
5. Comprehensive energy density and exit power flux diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from engine.config import SimulationConfig
from engine.geometry import PyramidGeometry


@dataclass
class HelmholtzResonator:
    """Represents a single Helmholtz resonator located along the Grand Gallery ramp slots.
    
    The governing oscillator equation is:
        d^2 U/dt^2 + (omega_r / Q) dU/dt + omega_r^2 U = (A_neck / (rho * L_eff)) * p(z_m, t)
    
    where U is acoustic volume displacement (m^3), dU/dt is volume velocity (m^3/s).
    """

    slot_index: int
    position_z: float
    target_frequency_hz: float = 438.0
    quality_factor: float = 50.0
    neck_area: float = 0.015  # m^2 (approx slot neck area)
    neck_length: float = 0.12  # m (effective neck length)
    cavity_volume: float = 0.08  # m^3 (resonator cavity volume)
    num_units_per_slot: int = 2  # Pair of resonators (east + west ramp slots)
    
    # Dynamic state variables
    displacement: float = 0.0  # U (m^3)
    velocity: float = 0.0  # dU/dt (m^3/s)
    acceleration: float = 0.0  # d^2U/dt^2 (m^3/s^2)

    def __post_init__(self) -> None:
        """If cavity_volume is not explicitly calibrated, calibrate it to target_frequency_hz in air."""
        if self.cavity_volume <= 0.0:
            c_air = 343.2
            # f_r = (c / (2*pi)) * sqrt(A_neck / (V * L_eff))
            # => V = (A_neck / L_eff) * (c / (2*pi*f_r))^2
            omega = 2.0 * math.pi * self.target_frequency_hz
            self.cavity_volume = (self.neck_area / self.neck_length) * ((c_air / omega) ** 2)

    def frequency(self, sound_speed: float) -> float:
        """Calculate natural resonant frequency f_r (Hz) at a given local sound speed."""
        if self.cavity_volume <= 0 or self.neck_length <= 0 or self.neck_area <= 0:
            return self.target_frequency_hz * (sound_speed / 343.2)
        return (sound_speed / (2.0 * math.pi)) * math.sqrt(
            self.neck_area / (self.cavity_volume * self.neck_length)
        )

    def angular_frequency(self, sound_speed: float) -> float:
        """Calculate natural resonant angular frequency omega_r (rad/s)."""
        return 2.0 * math.pi * self.frequency(sound_speed)

    def decay_time_constant(self, sound_speed: float) -> float:
        """Calculate ring-down exponential decay time constant tau = 2 * Q / omega_r (seconds)."""
        omega_r = self.angular_frequency(sound_speed)
        if omega_r <= 0.0:
            return float("inf")
        return (2.0 * self.quality_factor) / omega_r

    def acoustic_mass(self, gas_density: float) -> float:
        """Acoustic mass M_a = rho * L_eff / A_neck (kg/m^4)."""
        return (gas_density * self.neck_length) / self.neck_area

    def acoustic_compliance(self, sound_speed: float, gas_density: float) -> float:
        """Acoustic compliance C_a = V / (rho * c^2) (m^5/N)."""
        return self.cavity_volume / (gas_density * (sound_speed**2))

    def energy(self, sound_speed: float, gas_density: float) -> float:
        """Total instantaneous stored acoustic energy in resonator (Joules)."""
        m_a = self.acoustic_mass(gas_density)
        c_a = self.acoustic_compliance(sound_speed, gas_density)
        if c_a <= 0:
            return 0.0
        # E = (1/2) * M_a * (dU/dt)^2 + (1/2) * (1/C_a) * U^2
        # scaled by number of units in the slot pair
        e_single = 0.5 * m_a * (self.velocity**2) + 0.5 * (1.0 / c_a) * (self.displacement**2)
        return float(e_single * self.num_units_per_slot)

    def step_rk4(
        self,
        p_driving: float,
        sound_speed: float,
        gas_density: float,
        dt: float,
    ) -> Tuple[float, float, float]:
        """Advance resonator state by dt using 4th-order Runge-Kutta integration.
        
        Equation:
            dU/dt = w
            dw/dt = - (omega_r / Q) * w - omega_r^2 * U + (A_neck / (rho * L_eff)) * p
        """
        omega_r = self.angular_frequency(sound_speed)
        q = max(self.quality_factor, 0.1)
        gamma = omega_r / q
        omega_sq = omega_r * omega_r
        
        eff_density = max(gas_density, 1e-4)
        coupling_force = (self.neck_area / (eff_density * self.neck_length)) * p_driving

        def derivatives(u: float, w: float) -> Tuple[float, float]:
            du = w
            dw = -gamma * w - omega_sq * u + coupling_force
            return du, dw

        u0 = self.displacement
        w0 = self.velocity

        du1, dw1 = derivatives(u0, w0)
        du2, dw2 = derivatives(u0 + 0.5 * dt * du1, w0 + 0.5 * dt * dw1)
        du3, dw3 = derivatives(u0 + 0.5 * dt * du2, w0 + 0.5 * dt * dw2)
        du4, dw4 = derivatives(u0 + dt * du3, w0 + dt * dw3)

        u_next = u0 + (dt / 6.0) * (du1 + 2.0 * du2 + 2.0 * du3 + du4)
        w_next = w0 + (dt / 6.0) * (dw1 + 2.0 * dw2 + 2.0 * dw3 + dw4)
        
        # Acceleration at end of step
        _, a_next = derivatives(u_next, w_next)

        self.displacement = u_next
        self.velocity = w_next
        self.acceleration = a_next

        return self.displacement, self.velocity, self.acceleration

    def reset(self) -> None:
        """Reset resonator state to rest."""
        self.displacement = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0


class ResonatorBank:
    """Collection of 27 Helmholtz resonator pairs along the Grand Gallery ramp slots.
    
    Tuned to the F# harmonic series:
        f0 = 438.0 Hz (Fundamental)
        2f0 = 876.0 Hz (2nd harmonic / Octave)
        3f0 = 1314.0 Hz (3rd harmonic / Perfect Fifth above octave)
        4f0 = 1752.0 Hz (4th harmonic / Double octave)
    """

    def __init__(
        self,
        num_stations: int = 27,
        gallery_length: float = 46.61,
        slot_positions: Optional[Sequence[float]] = None,
        harmonic_frequencies: Sequence[float] = (438.0, 876.0, 1314.0, 1752.0),
        quality_factor: float = 50.0,
        neck_area: float = 0.015,
        neck_length: float = 0.12,
        units_per_slot: int = 2,
    ) -> None:
        self.num_stations = int(num_stations)
        self.gallery_length = float(gallery_length)
        self.harmonic_frequencies = tuple(float(f) for f in harmonic_frequencies)
        self.default_q = float(quality_factor)
        self.neck_area = float(neck_area)
        self.neck_length = float(neck_length)
        self.units_per_slot = int(units_per_slot)

        if slot_positions is not None:
            positions = [float(p) for p in slot_positions]
        else:
            # Grand Gallery slots: 28 pairs total, spaced at 1.68m starting from offset
            # 27 active resonator stations placed at slots 1..27 or evenly distributed
            slot_spacing = 1.68
            positions = [(i + 1) * slot_spacing for i in range(self.num_stations)]
            # Clamp positions within gallery length
            positions = [min(p, self.gallery_length - 0.5) for p in positions]

        self.resonators: List[HelmholtzResonator] = []
        c_air = 343.2
        for idx, pos in enumerate(positions):
            target_f = self.harmonic_frequencies[idx % len(self.harmonic_frequencies)]
            omega = 2.0 * math.pi * target_f
            # Calibrate cavity volume so that at 343.2 m/s resonance is exactly target_f
            v_cav = (self.neck_area / self.neck_length) * ((c_air / omega) ** 2)

            res = HelmholtzResonator(
                slot_index=idx,
                position_z=pos,
                target_frequency_hz=target_f,
                quality_factor=self.default_q,
                neck_area=self.neck_area,
                neck_length=self.neck_length,
                cavity_volume=v_cav,
                num_units_per_slot=self.units_per_slot,
            )
            self.resonators.append(res)

    def get_positions(self) -> np.ndarray:
        """Return 1D array of resonator positions along gallery incline (m)."""
        return np.array([r.position_z for r in self.resonators], dtype=np.float64)

    def get_frequencies(self, sound_speed: float = 343.2) -> np.ndarray:
        """Return array of current resonant frequencies (Hz) for given sound speed."""
        return np.array([r.frequency(sound_speed) for r in self.resonators], dtype=np.float64)

    def get_displacements(self) -> np.ndarray:
        """Return array of volume displacements U_m (m^3)."""
        return np.array([r.displacement for r in self.resonators], dtype=np.float64)

    def get_velocities(self) -> np.ndarray:
        """Return array of volume velocities dU_m/dt (m^3/s)."""
        return np.array([r.velocity for r in self.resonators], dtype=np.float64)

    def get_accelerations(self) -> np.ndarray:
        """Return array of volume accelerations d^2U_m/dt^2 (m^3/s^2)."""
        return np.array([r.acceleration for r in self.resonators], dtype=np.float64)

    def get_energies(
        self,
        sound_speeds: Union[float, np.ndarray] = 343.2,
        densities: Union[float, np.ndarray] = 1.204,
    ) -> np.ndarray:
        """Return array of stored acoustic energies (Joules) per resonator station."""
        if np.isscalar(sound_speeds):
            c_arr = np.full(len(self.resonators), float(sound_speeds))
        else:
            c_arr = np.asarray(sound_speeds, dtype=np.float64)
        if np.isscalar(densities):
            rho_arr = np.full(len(self.resonators), float(densities))
        else:
            rho_arr = np.asarray(densities, dtype=np.float64)

        energies = np.zeros(len(self.resonators), dtype=np.float64)
        for i, res in enumerate(self.resonators):
            energies[i] = res.energy(c_arr[i], rho_arr[i])
        return energies

    def step(
        self,
        p_grid: np.ndarray,
        z_grid: np.ndarray,
        c_grid: np.ndarray,
        rho_grid: np.ndarray,
        dt: float,
    ) -> None:
        """Advance all resonators by dt using local pressure, sound speed, and density."""
        for res in self.resonators:
            # Interpolate pressure, sound speed, density at resonator position z_m
            p_local = float(np.interp(res.position_z, z_grid, p_grid))
            c_local = float(np.interp(res.position_z, z_grid, c_grid))
            rho_local = float(np.interp(res.position_z, z_grid, rho_grid))
            res.step_rk4(p_local, c_local, rho_local, dt)

    def compute_coupling_source(
        self,
        z_grid: np.ndarray,
        gallery_area: Union[float, Sequence[float], np.ndarray],
        rho_grid: np.ndarray,
    ) -> np.ndarray:
        """Compute the spatial coupling source term S_res(z, t) for the acoustic wave equation.
        
        The source term represents the collective volume acceleration of the resonator bank,
        normalized by local duct cross-sectional area S(z_m) and spatial step dz:
            S_res(z, t) = sum_{m=1}^M (w_m(z) / (S(z_m) * dz)) * rho(z_m) * d^2 U_m/dt^2
        where w_m(z) is the spatial interpolation weight (sum_i w_m(z_i) = 1) ensuring volumetric
        acceleration d^2 U_m/dt^2 is dimensionally and physically balanced with acoustic pressure in Pa/m.
        """
        s_grid = np.zeros_like(z_grid, dtype=np.float64)
        if len(z_grid) < 2 or len(self.resonators) == 0:
            return s_grid

        dz = z_grid[1] - z_grid[0]
        n_grid = len(z_grid)

        area_arr = np.asarray(gallery_area, dtype=np.float64)
        is_scalar_area = area_arr.ndim == 0 or len(area_arr) == 1

        for res in self.resonators:
            zm = res.position_z
            if zm < z_grid[0] or zm > z_grid[-1]:
                continue
            
            # Linear interpolation / distribution of delta function
            idx = int(math.floor((zm - z_grid[0]) / dz))
            idx = max(0, min(idx, n_grid - 2))
            xi = (zm - z_grid[idx]) / dz
            
            total_vol_accel = res.acceleration * res.num_units_per_slot
            
            if is_scalar_area:
                s_local = float(area_arr) if area_arr.ndim == 0 else float(area_arr[0])
            else:
                s_local = float((1.0 - xi) * area_arr[idx] + xi * area_arr[idx + 1])
            s_local = max(s_local, 1.0e-6)

            rho_local = float((1.0 - xi) * rho_grid[idx] + xi * rho_grid[idx + 1])
            source_val = (rho_local / (s_local * dz)) * total_vol_accel
            
            s_grid[idx] += (1.0 - xi) * source_val
            s_grid[idx + 1] += xi * source_val

        return s_grid

    def reset(self) -> None:
        """Reset all resonators to rest."""
        for res in self.resonators:
            res.reset()


@dataclass
class GalleryAcousticState:
    """Complete diagnostic snapshot of the Grand Gallery acoustic wave field and resonator bank."""

    time: float
    grid_z: np.ndarray
    pressure: np.ndarray
    velocity: np.ndarray
    sound_speed: np.ndarray
    gas_density: np.ndarray
    energy_density: np.ndarray
    total_acoustic_energy: float
    exit_power_flux: float
    input_power_flux: float
    resonator_displacements: np.ndarray
    resonator_velocities: np.ndarray
    resonator_energies: np.ndarray
    bottom_pressure: float
    top_pressure: float
    rms_pressure: float
    peak_pressure: float
    area_profile: np.ndarray = field(default_factory=lambda: np.zeros(0))


class GrandGalleryAcoustics:
    """1D Finite-Difference Acoustic Wave Equation Solver for the Grand Gallery.
    
    Solves the lossy, inhomogeneous, variable-sound-speed acoustic wave equation:
        (1 / c(z)^2) * d^2p/dt^2 - (1 / S(z)) * d/dz(S(z) * dp/dz) + (2*alpha / c(z)) * dp/dt = S_res(z, t)
    
    coupled to 27 Helmholtz resonator pairs and variable H2-air gas dynamics.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        geometry: Optional[PyramidGeometry] = None,
        num_grid_points: int = 101,
        gallery_length: Optional[float] = None,
        cross_section_area: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        resonator_bank: Optional[ResonatorBank] = None,
        attenuation_coeff: Optional[float] = None,
        cfl_safety_factor: float = 0.8,
        top_boundary_type: str = "matched",
        bottom_boundary_type: str = "driven",
        top_impedance: Optional[float] = None,
        enable_resonators: bool = True,
        coupling_gain: float = 1.0,
    ) -> None:
        self.config = config or SimulationConfig()
        self.geometry = geometry or PyramidGeometry()

        self.num_grid = max(int(num_grid_points), 10)
        self.length = float(
            gallery_length
            if gallery_length is not None
            else self.geometry.grand_gallery.length_along_incline
        )
        if self.length <= 0.0:
            raise ValueError("gallery_length must be positive")

        self.dz = self.length / (self.num_grid - 1)
        self.z_grid = np.linspace(0.0, self.length, self.num_grid, dtype=np.float64)

        # Cross-sectional area profile S(z) along incline
        self.area_grid = np.zeros(self.num_grid, dtype=np.float64)
        if cross_section_area is not None:
            self.set_cross_section_area(cross_section_area)
        else:
            default_profile = self.compute_default_area_profile()
            self.area_grid[:] = default_profile

        # Attenuation coefficient alpha (Np/m) from limestone properties
        if attenuation_coeff is not None:
            self.alpha = float(attenuation_coeff)
        else:
            # Limestone attenuation: e.g. 0.45 dB/m => alpha = 0.45 / (20 * log10(e)) = 0.0518 Np/m
            db_per_m = self.config.limestone.acoustic_attenuation_db_per_m
            self.alpha = float(db_per_m / (20.0 * math.log10(math.e)))

        self.cfl_safety_factor = min(max(float(cfl_safety_factor), 0.1), 0.95)
        self.top_boundary_type = top_boundary_type.lower()
        self.bottom_boundary_type = bottom_boundary_type.lower()
        self.top_impedance = float(top_impedance) if top_impedance is not None else None
        self.enable_resonators = bool(enable_resonators)
        self.coupling_gain = float(coupling_gain)

        # Initialize resonator bank if not provided
        if resonator_bank is not None:
            self.resonator_bank = resonator_bank
        else:
            # Slot positions from GrandGallery geometry
            slot_pairs = self.geometry.grand_gallery.get_slot_positions()
            # 28 slot pairs total; take 27 active stations
            slot_positions = [
                i * self.geometry.grand_gallery.slot_spacing
                for i in range(1, min(28, len(slot_pairs) + 1))
            ]
            self.resonator_bank = ResonatorBank(
                num_stations=27,
                gallery_length=self.length,
                slot_positions=slot_positions[:27],
                harmonic_frequencies=self.config.acoustic.harmonic_modes_hz[:4],
                quality_factor=50.0,
            )

        # Field arrays
        self.p_prev = np.zeros(self.num_grid, dtype=np.float64)
        self.p_curr = np.zeros(self.num_grid, dtype=np.float64)
        self.p_next = np.zeros(self.num_grid, dtype=np.float64)
        self.velocity = np.zeros(self.num_grid, dtype=np.float64)

        # Calibrate gamma_h2 to match config target 1290 m/s at 20 deg C exactly
        target_c_h2 = float(self.config.gas.sound_speed_h2_20c)
        r = 8.314462618
        t_std = 293.15
        self.gamma_h2 = (target_c_h2**2 * float(self.config.gas.molar_mass_h2)) / (r * t_std)

        # Gas properties on grid
        self.c_grid = np.full(
            self.num_grid, self.config.gas.sound_speed_air_20c, dtype=np.float64
        )
        # Standard air density at 20 deg C
        rho_air = (
            self.config.gas.molar_mass_air * 101325.0 / (8.314462618 * 293.15)
        )
        self.rho_grid = np.full(self.num_grid, rho_air, dtype=np.float64)

        self.time: float = 0.0
        self.current_bottom_pressure: float = 0.0

    def compute_corbelled_cross_section_area(self, num_courses: int = 7) -> float:
        """Compute cross-sectional area accounting for 7-step corbel narrowing."""
        w_base = self.geometry.grand_gallery.width_base
        w_roof = self.geometry.grand_gallery.width_roof
        h = self.geometry.grand_gallery.vertical_height
        n = max(int(num_courses), 1)
        dh = h / n
        dw = (w_base - w_roof) / n
        total_area = sum((w_base - k * dw) * dh for k in range(n))
        return float(total_area)

    def compute_default_area_profile(self) -> np.ndarray:
        """Compute default cross-sectional area profile S(z) along Grand Gallery incline."""
        s_corbel = self.compute_corbelled_cross_section_area(num_courses=7)
        return np.full(self.num_grid, s_corbel, dtype=np.float64)

    def set_cross_section_area(
        self,
        area: Union[float, Sequence[float], np.ndarray],
    ) -> None:
        """Update spatial distribution of cross-sectional area S(z) (m^2)."""
        if np.isscalar(area):
            self.area_grid.fill(max(float(area), 1.0e-4))
        else:
            arr = np.asarray(area, dtype=np.float64)
            if len(arr) == self.num_grid:
                self.area_grid[:] = np.maximum(arr, 1.0e-4)
            else:
                x_in = np.linspace(0.0, self.length, len(arr))
                self.area_grid[:] = np.maximum(np.interp(self.z_grid, x_in, arr), 1.0e-4)

    @property
    def cross_section_area(self) -> float:
        """Mean cross-sectional area of the Grand Gallery (m^2)."""
        return float(np.mean(self.area_grid))

    @cross_section_area.setter
    def cross_section_area(self, value: Union[float, Sequence[float], np.ndarray]) -> None:
        self.set_cross_section_area(value)

    def set_gas_properties(
        self,
        sound_speed: Union[float, Sequence[float], np.ndarray],
        gas_density: Union[float, Sequence[float], np.ndarray],
    ) -> None:
        """Update spatial distribution of sound speed (m/s) and gas density (kg/m^3)."""
        if np.isscalar(sound_speed):
            self.c_grid.fill(float(sound_speed))
        else:
            arr = np.asarray(sound_speed, dtype=np.float64)
            if len(arr) == self.num_grid:
                self.c_grid[:] = arr
            else:
                # Interpolate if dimension does not match grid
                x_in = np.linspace(0.0, self.length, len(arr))
                self.c_grid[:] = np.interp(self.z_grid, x_in, arr)

        if np.isscalar(gas_density):
            self.rho_grid.fill(float(gas_density))
        else:
            arr = np.asarray(gas_density, dtype=np.float64)
            if len(arr) == self.num_grid:
                self.rho_grid[:] = arr
            else:
                x_in = np.linspace(0.0, self.length, len(arr))
                self.rho_grid[:] = np.interp(self.z_grid, x_in, arr)

    def set_gas_from_hydrogen_fraction(
        self,
        h2_mole_fraction: Union[float, Sequence[float], np.ndarray],
        temperature_k: float = 293.15,
        pressure_pa: float = 101325.0,
    ) -> None:
        """Compute and set c(z) and rho(z) dynamically from local H2 mole fraction."""
        r = 8.314462618
        m_h2 = float(self.config.gas.molar_mass_h2)
        m_air = float(self.config.gas.molar_mass_air)
        gamma_h2 = self.gamma_h2
        gamma_air = float(self.config.gas.gamma_air)

        if np.isscalar(h2_mole_fraction):
            x = float(h2_mole_fraction)
            x_arr = np.full(self.num_grid, x, dtype=np.float64)
        else:
            arr = np.asarray(h2_mole_fraction, dtype=np.float64)
            if len(arr) == self.num_grid:
                x_arr = arr
            else:
                x_in = np.linspace(0.0, self.length, len(arr))
                x_arr = np.interp(self.z_grid, x_in, arr)

        x_arr = np.clip(x_arr, 0.0, 1.0)
        m_mix = x_arr * m_h2 + (1.0 - x_arr) * m_air
        gamma_mix = x_arr * gamma_h2 + (1.0 - x_arr) * gamma_air
        c_mix = np.sqrt(gamma_mix * r * temperature_k / m_mix)
        rho_mix = pressure_pa * m_mix / (r * temperature_k)

        self.c_grid[:] = c_mix
        self.rho_grid[:] = rho_mix

    def update_from_gas_transport(
        self,
        transport_or_state: object,
    ) -> None:
        """Update acoustic gas properties from a ChemicalGasTransport or GasTransportState instance."""
        if hasattr(transport_or_state, "get_state"):
            state = transport_or_state.get_state()
        else:
            state = transport_or_state

        if hasattr(state, "get_node"):
            try:
                node = state.get_node("grand_gallery")
                self.set_gas_properties(
                    sound_speed=node.sound_speed_m_per_s,
                    gas_density=node.gas_density_kg_per_m3,
                )
                return
            except (KeyError, IndexError):
                pass

        if hasattr(state, "sound_speeds") and hasattr(state, "nodes"):
            for node in state.nodes:
                if "gallery" in node.name.lower():
                    self.set_gas_properties(
                        sound_speed=node.sound_speed_m_per_s,
                        gas_density=node.gas_density_kg_per_m3,
                    )
                    return

    def get_max_cfl_time_step(self) -> float:
        """Compute maximum allowable time step satisfying CFL condition."""
        c_max = float(np.max(self.c_grid))
        if c_max <= 0.0:
            c_max = 343.2
        return self.cfl_safety_factor * (self.dz / c_max)

    def inject_pulse(
        self,
        amplitude: float = 100.0,
        center_z: float = 0.0,
        width: float = 1.0,
    ) -> None:
        """Inject a smooth Gaussian acoustic pressure pulse into the spatial grid."""
        gaussian = amplitude * np.exp(-0.5 * ((self.z_grid - center_z) / max(width, 1e-3)) ** 2)
        self.p_curr += gaussian
        self.p_prev += gaussian

    def _step_substep(self, dt_sub: float, bottom_drive_pressure: float) -> None:
        """Perform a single CFL-compliant sub-step of the coupled wave-resonator system."""
        # 1. Advance resonators
        if self.enable_resonators:
            self.resonator_bank.step(
                self.p_curr, self.z_grid, self.c_grid, self.rho_grid, dt_sub
            )
            s_res = self.resonator_bank.compute_coupling_source(
                self.z_grid, self.area_grid, self.rho_grid
            ) * self.coupling_gain
        else:
            s_res = np.zeros_like(self.z_grid)

        # 2. Wave equation finite difference coefficients
        r = self.c_grid * dt_sub / self.dz
        r_sq = r * r
        gamma = self.alpha * dt_sub / self.c_grid

        # Area-normalized spatial divergence (conservative Webster horn formulation)
        s_half_plus = 0.5 * (self.area_grid[1:] + self.area_grid[:-1])
        s_int = self.area_grid[1:-1]
        
        spatial_flux = (
            s_half_plus[1:] * (self.p_curr[2:] - self.p_curr[1:-1])
            - s_half_plus[:-1] * (self.p_curr[1:-1] - self.p_curr[:-2])
        )
        spatial_term_int = (r_sq[1:-1] / s_int) * spatial_flux

        # Interior nodes update
        # p^{n+1} * (1 + gamma) = 2*p^n - (1 - gamma)*p^{n-1} + (r^2/S) * div(S * grad(p)) + c^2*dt^2*S_res
        c_sq_dt_sq = (self.c_grid * dt_sub) ** 2
        inv_damping = 1.0 / (1.0 + gamma)

        self.p_next[1:-1] = inv_damping[1:-1] * (
            2.0 * self.p_curr[1:-1]
            - (1.0 - gamma[1:-1]) * self.p_prev[1:-1]
            + spatial_term_int
            + c_sq_dt_sq[1:-1] * s_res[1:-1]
        )

        # 3. Bottom boundary condition (z = 0, i = 0)
        if self.bottom_boundary_type == "driven":
            self.p_next[0] = bottom_drive_pressure
        elif self.bottom_boundary_type == "rigid":
            # Neumann dp/dz = 0 => p_{-1} = p_1
            spatial_term_0 = (2.0 * r_sq[0] / self.area_grid[0]) * s_half_plus[0] * (self.p_curr[1] - self.p_curr[0])
            self.p_next[0] = inv_damping[0] * (
                2.0 * self.p_curr[0]
                - (1.0 - gamma[0]) * self.p_prev[0]
                + spatial_term_0
                + c_sq_dt_sq[0] * s_res[0]
            )
        elif self.bottom_boundary_type == "open":
            self.p_next[0] = 0.0
        elif self.bottom_boundary_type in ("absorbing", "matched"):
            # 1st-order Mur absorbing boundary condition: dp/dt - c*dp/dz = 0
            self.p_next[0] = self.p_curr[0] + r[0] * (self.p_curr[1] - self.p_curr[0])
        else:
            self.p_next[0] = bottom_drive_pressure

        # 4. Top boundary condition (z = L, i = N-1)
        if self.top_boundary_type == "rigid":
            # Neumann dp/dz = 0 => p_{N} = p_{N-2}
            spatial_term_end = (2.0 * r_sq[-1] / self.area_grid[-1]) * s_half_plus[-1] * (self.p_curr[-2] - self.p_curr[-1])
            self.p_next[-1] = inv_damping[-1] * (
                2.0 * self.p_curr[-1]
                - (1.0 - gamma[-1]) * self.p_prev[-1]
                + spatial_term_end
                + c_sq_dt_sq[-1] * s_res[-1]
            )
        elif self.top_boundary_type == "open":
            self.p_next[-1] = 0.0
        elif self.top_boundary_type in ("absorbing", "matched"):
            # 1st-order Mur radiation boundary condition: dp/dt + c*dp/dz = 0
            self.p_next[-1] = self.p_curr[-1] - r[-1] * (self.p_curr[-1] - self.p_curr[-2])
        elif self.top_boundary_type == "impedance" and self.top_impedance is not None:
            # Specific acoustic impedance matching
            z0 = self.rho_grid[-1] * self.c_grid[-1] / self.area_grid[-1]
            z_ratio = max(self.top_impedance / max(z0, 1e-6), 1e-3)
            self.p_next[-1] = self.p_curr[-1] - (r[-1] / z_ratio) * (
                self.p_curr[-1] - self.p_curr[-2]
            )
        else:
            # Default to matched absorbing boundary
            self.p_next[-1] = self.p_curr[-1] - r[-1] * (self.p_curr[-1] - self.p_curr[-2])

        # 5. Acoustic velocity update via momentum equation: rho * dv/dt = -dp/dz
        dp_dz = np.zeros_like(self.p_curr)
        dp_dz[1:-1] = (self.p_next[2:] - self.p_next[:-2]) / (2.0 * self.dz)
        dp_dz[0] = (self.p_next[1] - self.p_next[0]) / self.dz
        dp_dz[-1] = (self.p_next[-1] - self.p_next[-2]) / self.dz

        self.velocity -= (dt_sub / self.rho_grid) * dp_dz

        # Cycle pressure arrays
        self.p_prev[:] = self.p_curr
        self.p_curr[:] = self.p_next

        self.time += dt_sub

    def step(
        self,
        dt: float,
        bottom_pressure_drive: float = 0.0,
    ) -> GalleryAcousticState:
        """Advance the acoustic wave simulation by dt, automatically sub-stepping for CFL stability."""
        if dt <= 0.0:
            return self.get_state()

        dt_cfl = self.get_max_cfl_time_step()
        num_substeps = max(1, int(math.ceil(dt / dt_cfl)))
        dt_sub = dt / num_substeps

        self.current_bottom_pressure = float(bottom_pressure_drive)

        for _ in range(num_substeps):
            self._step_substep(dt_sub, self.current_bottom_pressure)

        return self.get_state()

    def simulate(
        self,
        duration: float,
        dt: float = 1.0e-4,
        drive_func: Optional[Callable[[float], float]] = None,
    ) -> List[GalleryAcousticState]:
        """Run acoustic wave simulation for total duration (seconds) with optional driving signal function."""
        num_steps = max(1, int(math.ceil(duration / dt)))
        states: List[GalleryAcousticState] = []
        states.append(self.get_state())

        for step_idx in range(num_steps):
            t_curr = self.time
            p_drive = drive_func(t_curr) if drive_func is not None else 0.0
            st = self.step(dt, bottom_pressure_drive=p_drive)
            states.append(st)

        return states

    def compute_energy_density(self) -> np.ndarray:
        """Compute local acoustic energy density w(z) = p^2 / (2*rho*c^2) + (1/2)*rho*v^2 (J/m^3)."""
        w_pot = (self.p_curr**2) / (2.0 * self.rho_grid * (self.c_grid**2))
        w_kin = 0.5 * self.rho_grid * (self.velocity**2)
        return w_pot + w_kin

    def compute_total_acoustic_energy(self) -> float:
        """Compute integrated acoustic field and resonator energy in the Grand Gallery (Joules)."""
        w = self.compute_energy_density()
        e_field = float(np.trapz(w * self.area_grid, self.z_grid))
        if self.enable_resonators and len(self.resonator_bank.resonators) > 0:
            e_res = float(np.sum(self.resonator_bank.get_energies(self.c_grid, self.rho_grid)))
            return e_field + e_res
        return e_field

    def compute_exit_power_flux(self) -> float:
        """Compute instantaneous acoustic power flux exiting the top portal (Watts).
        
        P_exit = p(L) * v(L) * A_exit
        """
        p_exit = float(self.p_curr[-1])
        v_exit = float(self.velocity[-1])
        return float(p_exit * v_exit * self.area_grid[-1])

    def compute_input_power_flux(self) -> float:
        """Compute instantaneous acoustic power flux entering from the bottom junction (Watts).
        
        P_in = p(0) * v(0) * A_in
        """
        p_in = float(self.p_curr[0])
        v_in = float(self.velocity[0])
        return float(p_in * v_in * self.area_grid[0])

    def get_state(self) -> GalleryAcousticState:
        """Return a snapshot of current acoustic fields and resonator states."""
        w = self.compute_energy_density()
        e_acoustic = self.compute_total_acoustic_energy()
        p_exit_flux = self.compute_exit_power_flux()
        p_in_flux = self.compute_input_power_flux()

        res_disp = self.resonator_bank.get_displacements()
        res_vel = self.resonator_bank.get_velocities()
        res_energies = self.resonator_bank.get_energies(self.c_grid, self.rho_grid)

        p_rms = float(np.sqrt(np.mean(self.p_curr**2)))
        p_peak = float(np.max(np.abs(self.p_curr)))

        return GalleryAcousticState(
            time=self.time,
            grid_z=self.z_grid.copy(),
            pressure=self.p_curr.copy(),
            velocity=self.velocity.copy(),
            sound_speed=self.c_grid.copy(),
            gas_density=self.rho_grid.copy(),
            energy_density=w,
            total_acoustic_energy=e_acoustic,
            exit_power_flux=p_exit_flux,
            input_power_flux=p_in_flux,
            resonator_displacements=res_disp,
            resonator_velocities=res_vel,
            resonator_energies=res_energies,
            bottom_pressure=float(self.p_curr[0]),
            top_pressure=float(self.p_curr[-1]),
            rms_pressure=p_rms,
            peak_pressure=p_peak,
            area_profile=self.area_grid.copy(),
        )

    def reset(self) -> None:
        """Reset all acoustic fields and resonators to zero state."""
        self.p_prev.fill(0.0)
        self.p_curr.fill(0.0)
        self.p_next.fill(0.0)
        self.velocity.fill(0.0)
        self.time = 0.0
        self.current_bottom_pressure = 0.0
        self.resonator_bank.reset()
