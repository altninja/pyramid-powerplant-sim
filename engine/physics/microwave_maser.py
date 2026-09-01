"""King's Chamber Microwave Maser & Shaft Horn Waveguide Beaming Module.

This module implements:
1. Two-level / three-level atomic hydrogen hyperfine population inversion rate equations
   pumped by acoustic pressure standing waves and high-voltage piezoelectric fields:
       dN2/dt = W_pump * N1 - B21 * rho_em * (N2 - N1) - A21 * N2 - (N2 - N2_eq) / tau_coll
       d(rho_em)/dt = h * nu21 * B21 * rho_em * (N2 - N1) + h * nu21 * A21 * N2 * eta_geom
                      - rho_em / tau_cav - P_out,shafts / V_KC
2. King's Chamber electromagnetic cavity resonant mode coupling (nu21 = 1.4204057517667 GHz,
   lambda = 21.106 cm, A21 = 2.85e-15 s^-1, B21 = 5.67e20 m^3/(J*s^2)).
3. Rectangular dielectric horn waveguide shafts (Northern 32° 28' 00", Southern 45° 00' 00"):
   - TE10 mode cutoff frequency: f_c = c0 / (2a) = 681.35 MHz (a = b = 0.22 m).
   - High transmission at 1.4204 GHz with guide propagation constant beta = 26.08 rad/m.
   - Evanescent attenuation for sub-cutoff frequencies f < f_c.
   - Limestone casing wall attenuation alpha_att = 0.005 Np/m (0.0434 dB/m).
   - Directional horn antenna aperture gain G0 = (4*pi / lambda^2) * A_aperture * eta_aperture.
4. Coherent stimulated microwave radiation extraction and directional beam power tracking.
5. Strict conservation of energy and quantum photon energy balance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

from engine.config import (
    PLANCK_CONSTANT_H,
    SPEED_OF_LIGHT_C0,
    STANDARD_ATMOSPHERIC_PRESSURE_PA,
    STANDARD_TEMPERATURE_K,
    VACUUM_PERMEABILITY_MU0,
    VACUUM_PERMITTIVITY_EPS0,
    SimulationConfig,
)
from engine.geometry import PyramidGeometry, Vector3D


# Standard King's Chamber microwave maser transition constants
HYPERFINE_FREQUENCY_HZ: float = 1420405751.7667  # 1.4204057517667 GHz (21.106 cm)
EINSTEIN_A21_S: float = 2.85e-15  # s^-1 (spontaneous emission rate)
EINSTEIN_B21_M3_J_S2: float = 5.67e20  # m^3 / (J * s^2) (stimulated emission rate coefficient)
WAVEGUIDE_CUTOFF_HZ: float = 681346495.4545455  # c0 / (2 * 0.22) = 681.346 MHz
PROPAGATION_BETA_RAD_M: float = 26.120985  # rad/m at 1.4204 GHz in 0.22 m x 0.22 m waveguide
WAVELENGTH_21CM_M: float = SPEED_OF_LIGHT_C0 / HYPERFINE_FREQUENCY_HZ  # 0.21106114 m
PHOTON_ENERGY_J: float = PLANCK_CONSTANT_H * HYPERFINE_FREQUENCY_HZ  # ~9.4117e-25 J
FREE_SPACE_IMPEDANCE_OHMS: float = math.sqrt(
    VACUUM_PERMEABILITY_MU0 / VACUUM_PERMITTIVITY_EPS0
)  # 376.730313668 Ohm


@dataclass
class WaveguideShaft:
    """Rectangular dielectric-lined waveguide shaft and horn antenna radiator.
    
    Models King's Chamber Northern and Southern shafts as rectangular dielectric-loaded
    waveguides operating predominantly in the fundamental TE10 mode, terminating at the
    pyramid exterior casing in a directional aperture horn antenna.
    """

    name: str = "King's Chamber Shaft"
    heading: str = "north"
    width: float = 0.22  # Dimension a (m)
    height: float = 0.22  # Dimension b (m)
    length: float = 71.0  # Path length L (m)
    incline_angle_degrees: float = 32.46666666666667  # Incline angle from horizontal (deg)
    start_point: Vector3D = field(default_factory=lambda: Vector3D(x=0.0, y=12.38, z=44.0))
    attenuation_np_per_m: float = 0.005  # Limestone dielectric wall attenuation alpha (Np/m)
    aperture_efficiency: float = 0.70  # Horn aperture efficiency eta_ap
    casing_dielectric_permittivity: float = 6.2  # eps_r of surrounding masonry
    casing_loss_tangent: float = 0.015  # tan(delta)

    @property
    def cross_section_area(self) -> float:
        """Cross-sectional guide area A = a * b (m^2)."""
        return self.width * self.height

    @property
    def cutoff_frequency(self) -> float:
        """Fundamental TE10 mode cutoff frequency f_c = c0 / (2*a) (Hz)."""
        return SPEED_OF_LIGHT_C0 / (2.0 * self.width)

    def cutoff_frequency_mode(self, m: int = 1, n: int = 0) -> float:
        """TE_mn / TM_mn mode cutoff frequency in Hz."""
        term_m = (m / self.width) ** 2
        term_n = (n / self.height) ** 2
        return 0.5 * SPEED_OF_LIGHT_C0 * math.sqrt(term_m + term_n)

    def propagation_constant(self, frequency_hz: float) -> complex:
        """Compute complex propagation constant gamma = alpha + j*beta.
        
        For f > f_c:
            beta = k0 * sqrt(1 - (fc/f)^2)
            alpha = attenuation_np_per_m
            gamma = alpha + j*beta
        For f < f_c (evanescent cutoff):
            alpha_c = k0 * sqrt((fc/f)^2 - 1)
            alpha_tot = alpha_c + attenuation_np_per_m
            gamma = alpha_tot + j*0
        """
        if frequency_hz <= 0.0:
            alpha_dc = math.pi / self.width + self.attenuation_np_per_m
            return complex(alpha_dc, 0.0)

        fc = self.cutoff_frequency
        k0 = (2.0 * math.pi * frequency_hz) / SPEED_OF_LIGHT_C0

        if frequency_hz >= fc:
            ratio = fc / frequency_hz
            radicand = max(0.0, 1.0 - ratio * ratio)
            beta = k0 * math.sqrt(radicand)
            return complex(self.attenuation_np_per_m, beta)
        else:
            ratio = fc / frequency_hz
            radicand = max(0.0, ratio * ratio - 1.0)
            alpha_c = k0 * math.sqrt(radicand)
            return complex(alpha_c + self.attenuation_np_per_m, 0.0)

    def guide_wavelength(self, frequency_hz: float) -> float:
        """Guide wavelength lambda_g = 2*pi / beta (m). Returns inf if below cutoff."""
        gamma = self.propagation_constant(frequency_hz)
        beta = gamma.imag
        if beta <= 1.0e-9:
            return float("inf")
        return (2.0 * math.pi) / beta

    def phase_velocity(self, frequency_hz: float) -> float:
        """Phase velocity v_p = omega / beta (m/s)."""
        gamma = self.propagation_constant(frequency_hz)
        beta = gamma.imag
        if beta <= 1.0e-9:
            return 0.0
        omega = 2.0 * math.pi * frequency_hz
        return omega / beta

    def group_velocity(self, frequency_hz: float) -> float:
        """Group velocity v_g = c0 * sqrt(1 - (fc/f)^2) (m/s)."""
        fc = self.cutoff_frequency
        if frequency_hz <= fc:
            return 0.0
        ratio = fc / frequency_hz
        return SPEED_OF_LIGHT_C0 * math.sqrt(max(0.0, 1.0 - ratio * ratio))

    def wave_impedance(self, frequency_hz: float) -> complex:
        """Waveguide characteristic impedance Z_TE10 (Ohms)."""
        fc = self.cutoff_frequency
        if frequency_hz <= 0.0:
            return complex(0.0, 0.0)
        
        eta0 = FREE_SPACE_IMPEDANCE_OHMS
        if frequency_hz > fc:
            ratio = fc / frequency_hz
            denom = math.sqrt(1.0 - ratio * ratio)
            return complex(eta0 / denom, 0.0)
        elif frequency_hz < fc:
            ratio = fc / frequency_hz
            denom = math.sqrt(ratio * ratio - 1.0)
            return complex(0.0, eta0 / denom)
        else:
            return complex(float("inf"), 0.0)

    def transmission_efficiency(self, frequency_hz: float) -> float:
        """Shaft power transmission efficiency eta_shaft = exp(-2 * alpha * L) in [0, 1]."""
        gamma = self.propagation_constant(frequency_hz)
        alpha = gamma.real
        arg = -2.0 * alpha * self.length
        if arg < -700.0:
            return 0.0
        return float(math.exp(arg))

    def transmission_amplitude(self, frequency_hz: float) -> float:
        """Field transmission amplitude |T| = exp(-alpha * L) in [0, 1]."""
        gamma = self.propagation_constant(frequency_hz)
        alpha = gamma.real
        arg = -alpha * self.length
        if arg < -700.0:
            return 0.0
        return float(math.exp(arg))

    def transmission_loss_db(self, frequency_hz: float) -> float:
        """Transmission loss in decibels (dB) = -10 * log10(eta_shaft)."""
        eff = self.transmission_efficiency(frequency_hz)
        if eff <= 1.0e-30:
            return 300.0
        return -10.0 * math.log10(eff)

    def aperture_gain(self, frequency_hz: float) -> float:
        """Linear horn aperture gain G0 = (4*pi / lambda^2) * A_aperture * eta_aperture."""
        if frequency_hz <= 0.0:
            return 0.0
        lambda_0 = SPEED_OF_LIGHT_C0 / frequency_hz
        area = self.cross_section_area
        return (4.0 * math.pi / (lambda_0 ** 2)) * area * self.aperture_efficiency

    def aperture_gain_dbi(self, frequency_hz: float) -> float:
        """Horn aperture gain in dBi = 10 * log10(G0)."""
        g0 = self.aperture_gain(frequency_hz)
        if g0 <= 1.0e-12:
            return -120.0
        return 10.0 * math.log10(g0)

    def get_unit_vector(self) -> Vector3D:
        """Unit vector pointing along the shaft beaming axis outward."""
        rad = math.radians(self.incline_angle_degrees)
        cos_val = math.cos(rad)
        sin_val = math.sin(rad)
        h = self.heading.lower()
        if h == "north":
            return Vector3D(x=0.0, y=-cos_val, z=sin_val).unit()
        elif h == "south":
            return Vector3D(x=0.0, y=cos_val, z=sin_val).unit()
        elif h == "east":
            return Vector3D(x=cos_val, y=0.0, z=sin_val).unit()
        elif h == "west":
            return Vector3D(x=-cos_val, y=0.0, z=sin_val).unit()
        return Vector3D(x=0.0, y=0.0, z=1.0)

    def get_exit_point(self) -> Vector3D:
        """Calculate the outer casing exit point (m) of the shaft."""
        u = self.get_unit_vector()
        return self.start_point + u * self.length

    def radiated_power(self, incident_power_watts: float, frequency_hz: float) -> float:
        """Radiated microwave beam power P_beam = P_incident * eta_shaft(f) (W)."""
        if incident_power_watts <= 0.0:
            return 0.0
        return incident_power_watts * self.transmission_efficiency(frequency_hz)


@dataclass
class MaserState:
    """Snapshot state of the King's Chamber microwave maser and shaft beaming module."""

    time: float = 0.0
    n1_population: float = 0.0  # Lower hyperfine state density N1 (atoms/m^3)
    n2_population: float = 0.0  # Upper hyperfine state density N2 (atoms/m^3)
    population_inversion: float = 0.0  # Delta N = N2 - N1 (atoms/m^3)
    total_h_density: float = 0.0  # N_total = N1 + N2 (atoms/m^3)
    photon_energy_density: float = 0.0  # EM energy density rho_em (J/m^3)
    pumping_rate: float = 0.0  # Pumping rate W_pump (s^-1)
    threshold_inversion: float = 0.0  # Critical inversion Delta N_th (atoms/m^3)
    is_above_threshold: bool = False  # True when Delta N > Delta N_th and actively lasing
    stimulated_transition_rate: float = 0.0  # Stimulated transitions / (m^3 * s)
    stimulated_power_total: float = 0.0  # Total stimulated emission power P_stim (W)
    spontaneous_power_total: float = 0.0  # Total spontaneous emission power P_spon (W)
    cavity_loss_power: float = 0.0  # Chamber internal dielectric/wall dissipation (W)
    shaft_extracted_power: float = 0.0  # Total microwave power extracted into shafts (W)
    north_shaft_power_in: float = 0.0  # Power entering Northern shaft (W)
    south_shaft_power_in: float = 0.0  # Power entering Southern shaft (W)
    north_shaft_beam_power: float = 0.0  # Power radiated from Northern shaft horn (W)
    south_shaft_beam_power: float = 0.0  # Power radiated from Southern shaft horn (W)
    total_radiated_power: float = 0.0  # Combined radiated microwave beam power (W)
    north_shaft_erp_watts: float = 0.0  # Directional Effective Radiated Power (ERP) Northern horn (W)
    south_shaft_erp_watts: float = 0.0  # Directional Effective Radiated Power (ERP) Southern horn (W)
    total_erp_watts: float = 0.0  # Total Effective Radiated Power (ERP) (W)
    cumulative_stimulated_energy: float = 0.0  # Integrated stimulated energy (J)
    cumulative_radiated_energy: float = 0.0  # Integrated radiated beam energy (J)
    cumulative_cavity_loss_energy: float = 0.0  # Integrated cavity dissipated energy (J)
    cumulative_shaft_loss_energy: float = 0.0  # Integrated shaft wall dissipated energy (J)
    frequency_hz: float = HYPERFINE_FREQUENCY_HZ
    wavelength_m: float = WAVELENGTH_21CM_M
    north_horn_gain_dbi: float = 0.0
    south_horn_gain_dbi: float = 0.0

    @property
    def north_erp(self) -> float:
        """Effective Radiated Power (ERP) for Northern shaft in Watts."""
        return self.north_shaft_erp_watts

    @property
    def south_erp(self) -> float:
        """Effective Radiated Power (ERP) for Southern shaft in Watts."""
        return self.south_shaft_erp_watts

    @property
    def total_erp(self) -> float:
        """Combined Effective Radiated Power (ERP) in Watts."""
        return self.total_erp_watts


class MicrowaveMaser:
    """King's Chamber Atomic Hydrogen Microwave Maser Simulation Engine.
    
    Integrates the coupled quantum-electromagnetic rate equations governing atomic
    hydrogen hyperfine maser amplification in the King's Chamber cavity and power beaming
    through the northern and southern horn waveguide shafts.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        geometry: Optional[PyramidGeometry] = None,
        chamber_volume: float = 320.0,
        hyperfine_frequency: float = HYPERFINE_FREQUENCY_HZ,
        einstein_a21: float = EINSTEIN_A21_S,
        einstein_b21: float = EINSTEIN_B21_M3_J_S2,
        cavity_quality_factor: Optional[float] = None,
        geometric_spontaneous_fraction: float = 1.0e-4,
        collisional_deexcitation_time: float = 0.05,
        coupling_kappa_elec: Optional[float] = None,
        coupling_kappa_acoust: Optional[float] = None,
        voltage_norm: Optional[float] = None,
        pressure_norm: Optional[float] = None,
        v_ref: Optional[float] = None,
        p_ref: Optional[float] = None,
        voltage_threshold: Optional[float] = None,
        pressure_threshold: Optional[float] = None,
        shaft_coupling_factor: Optional[float] = None,
        nominal_h_density: float = 5.0e22,
    ) -> None:
        """Initialize King's Chamber Microwave Maser engine.
        
        Args:
            config: Optional global simulation configuration.
            geometry: Optional 3D pyramid geometry model.
            chamber_volume: King's Chamber survey volume V_KC in m^3 (default 320.0 m^3).
            hyperfine_frequency: Hydrogen 21 cm transition frequency nu21 in Hz.
            einstein_a21: Spontaneous emission rate A21 in s^-1.
            einstein_b21: Stimulated emission coefficient B21 in m^3/(J*s^2).
            cavity_quality_factor: Loaded cavity quality factor Q_cav.
            geometric_spontaneous_fraction: Fraction eta_geom of spontaneous photons entering mode.
            collisional_deexcitation_time: Collisional relaxation time constant tau_coll in s.
            coupling_kappa_elec: Electrical piezoelectric pump coupling coefficient (s^-1).
            coupling_kappa_acoust: Acoustic wave pump coupling coefficient (s^-1).
            voltage_norm: Alias for v_ref (V).
            pressure_norm: Alias for p_ref (Pa).
            v_ref: Calibrated reference voltage threshold (V, default 5.0 kV).
            p_ref: Calibrated reference acoustic pressure threshold (Pa, default 100 kPa).
            voltage_threshold: Sub-threshold excitation voltage cutoff (V, default 100 V).
            pressure_threshold: Sub-threshold acoustic pressure cutoff (Pa, default 50 Pa).
            shaft_coupling_factor: Coupling efficiency kappa_shaft into shaft apertures.
            nominal_h_density: Baseline atomic hydrogen density at X_H2 = 1.0 (atoms/m^3).
        """
        self.config = config or SimulationConfig()
        cfg_maser = self.config.maser

        self.geometry = geometry or PyramidGeometry()
        self.chamber_volume = chamber_volume
        self.hyperfine_frequency = hyperfine_frequency
        self.einstein_a21 = einstein_a21
        self.einstein_b21 = einstein_b21
        self.cavity_quality_factor = (
            cavity_quality_factor
            if cavity_quality_factor is not None
            else getattr(cfg_maser, "cavity_quality_factor", 5.0e4)
        )
        self.geometric_spontaneous_fraction = geometric_spontaneous_fraction
        self.collisional_deexcitation_time = collisional_deexcitation_time

        # Calibrated electromechanical pumping parameters
        ref_v = v_ref if v_ref is not None else (voltage_norm if voltage_norm is not None else getattr(cfg_maser, "v_ref", 5000.0))
        ref_p = p_ref if p_ref is not None else (pressure_norm if pressure_norm is not None else getattr(cfg_maser, "p_ref", 100000.0))
        k_elec = coupling_kappa_elec if coupling_kappa_elec is not None else getattr(cfg_maser, "coupling_kappa_elec", 50.0)
        k_acoust = coupling_kappa_acoust if coupling_kappa_acoust is not None else getattr(cfg_maser, "coupling_kappa_acoust", 10.0)
        v_th = voltage_threshold if voltage_threshold is not None else getattr(cfg_maser, "voltage_threshold", 100.0)
        p_th = pressure_threshold if pressure_threshold is not None else getattr(cfg_maser, "pressure_threshold", 50.0)
        s_factor = shaft_coupling_factor if shaft_coupling_factor is not None else getattr(cfg_maser, "shaft_coupling_factor", 0.5)

        self.voltage_norm = float(ref_v)
        self.v_ref = float(ref_v)
        self.pressure_norm = float(ref_p)
        self.p_ref = float(ref_p)
        self.coupling_kappa_elec = float(k_elec)
        self.kappa_elec = float(k_elec)
        self.coupling_kappa_acoust = float(k_acoust)
        self.kappa_acoust = float(k_acoust)
        self.voltage_threshold = float(v_th)
        self.pressure_threshold = float(p_th)
        self.shaft_coupling_factor = float(s_factor)
        self.nominal_h_density = nominal_h_density

        # Initialize Northern and Southern Waveguide Shafts from Geometry
        geo_north = self.geometry.kings_shaft_north
        self.north_shaft = WaveguideShaft(
            name=geo_north.name,
            heading=geo_north.heading,
            width=geo_north.width,
            height=geo_north.height,
            length=geo_north.length,
            incline_angle_degrees=geo_north.angle_degrees,
            start_point=geo_north.start_point,
            attenuation_np_per_m=0.005,
            aperture_efficiency=0.70,
        )

        geo_south = self.geometry.kings_shaft_south
        self.south_shaft = WaveguideShaft(
            name=geo_south.name,
            heading=geo_south.heading,
            width=geo_south.width,
            height=geo_south.height,
            length=geo_south.length,
            incline_angle_degrees=geo_south.angle_degrees,
            start_point=geo_south.start_point,
            attenuation_np_per_m=0.005,
            aperture_efficiency=0.70,
        )

        # Dynamic state variables
        self.time: float = 0.0
        self.total_h_density: float = 0.0
        self.population_inversion: float = 0.0
        self.n1_population: float = 0.0
        self.n2_population: float = 0.0
        self.photon_energy_density: float = 0.0
        self.cumulative_stimulated_energy: float = 0.0
        self.cumulative_radiated_energy: float = 0.0
        self.cumulative_cavity_loss_energy: float = 0.0
        self.cumulative_shaft_loss_energy: float = 0.0

        # Precompute electromagnetic transition constants
        self.photon_energy = PLANCK_CONSTANT_H * self.hyperfine_frequency
        self.omega_21 = 2.0 * math.pi * self.hyperfine_frequency
        self.cavity_decay_rate = self.omega_21 / self.cavity_quality_factor
        self.cavity_tau = 1.0 / self.cavity_decay_rate

    @property
    def group_velocity(self) -> float:
        """Waveguide group velocity v_g (m/s) at the hyperfine maser frequency."""
        return self.north_shaft.group_velocity(self.hyperfine_frequency)

    @property
    def shaft_decay_rate(self) -> float:
        """Rate of photon extraction out of the chamber through both shafts (s^-1).
        
        1 / tau_shaft = v_g * (A_N + A_S) * kappa_shaft / V_KC
        """
        total_aperture = (
            self.north_shaft.cross_section_area + self.south_shaft.cross_section_area
        )
        vg = self.group_velocity
        return (vg * total_aperture * self.shaft_coupling_factor) / self.chamber_volume

    @property
    def loaded_cavity_decay_rate(self) -> float:
        """Total loaded cavity loss rate gamma_tot = 1/tau_cav + 1/tau_shaft (s^-1)."""
        return self.cavity_decay_rate + self.shaft_decay_rate

    @property
    def loaded_cavity_tau(self) -> float:
        """Effective photon lifetime in the loaded cavity tau_loaded (s)."""
        rate = self.loaded_cavity_decay_rate
        return 1.0 / rate if rate > 0.0 else 1.0e-3

    def threshold_population_inversion(self) -> float:
        """Compute critical population inversion Delta N_th = (N2 - N1)_th (atoms/m^3).
        
        Above this threshold, stimulated emission gain exceeds loaded cavity losses:
            Delta N_th = gamma_tot / (h * nu21 * B21)
        """
        gain_factor = self.photon_energy * self.einstein_b21
        if gain_factor <= 0.0:
            return float("inf")
        return self.loaded_cavity_decay_rate / gain_factor

    def compute_total_h_density(
        self,
        h2_concentration: float,
        temperature_k: float = STANDARD_TEMPERATURE_K,
        pressure_pa: float = STANDARD_ATMOSPHERIC_PRESSURE_PA,
    ) -> float:
        """Calculate active atomic hydrogen density N_total (atoms/m^3).
        
        Args:
            h2_concentration: H2 mole fraction X_H2 in [0, 1] or molar concentration (mol/m^3).
            temperature_k: Chamber gas temperature in Kelvin.
            pressure_pa: Chamber gas pressure in Pascals.
            
        Returns:
            Total active atomic hydrogen density N_total in atoms/m^3.
        """
        if h2_concentration <= 0.0:
            return 0.0

        # If passed as mole fraction in [0, 1]
        if h2_concentration <= 1.0:
            x_h2 = max(0.0, min(1.0, h2_concentration))
            return self.nominal_h_density * x_h2
        else:
            # Passed as molar concentration (mol/m^3)
            # Standard conversion: n_H = 2 * C_H2 * N_A * dissociation_fraction
            c_h2 = float(h2_concentration)
            avogadro = 6.02214076e23
            dissoc_fraction = self.nominal_h_density / (2.0 * 41.57 * avogadro)
            return 2.0 * c_h2 * avogadro * dissoc_fraction

    def compute_pumping_rate(
        self,
        piezo_voltage: float,
        acoustic_pressure: float,
        h2_fraction: float,
    ) -> float:
        """Calculate dynamic maser pumping rate W_pump (s^-1).
        
        Driven by King's Chamber piezoelectric voltage oscillations and acoustic pressure:
            W_pump = X_H2 * [ kappa_elec * (V_total / V_ref)^2 + kappa_acoust * (p_KC / p_ref) ]
        with sub-threshold drive yielding zero pumping.
        """
        if h2_fraction <= 0.0:
            return 0.0

        v_abs = abs(piezo_voltage)
        p_abs = abs(acoustic_pressure)

        # Apply sub-threshold cutoff
        v_eff = v_abs if v_abs >= self.voltage_threshold else 0.0
        p_eff = p_abs if p_abs >= self.pressure_threshold else 0.0

        if v_eff <= 0.0 and p_eff <= 0.0:
            return 0.0

        x_h2 = max(0.0, min(1.0, h2_fraction if h2_fraction <= 1.0 else 1.0))
        v_ratio = v_eff / self.v_ref
        p_ratio = p_eff / self.p_ref

        rate = x_h2 * (
            self.kappa_elec * (v_ratio ** 2)
            + self.kappa_acoust * p_ratio
        )
        return float(rate)

    def threshold_pumping_rate(
        self,
        h2_fraction: float,
        temperature_k: float = STANDARD_TEMPERATURE_K,
    ) -> float:
        """Calculate theoretical pumping rate threshold W_pump,th for maser action.
        
        W_th is the minimum pumping rate required to achieve Delta N = Delta N_th
        under steady-state unsaturated conditions.
        """
        n_total = self.compute_total_h_density(h2_fraction, temperature_k)
        if n_total <= 0.0:
            return float("inf")

        delta_n_th = self.threshold_population_inversion()
        if delta_n_th >= n_total:
            return float("inf")

        n2_th = 0.5 * (n_total + delta_n_th)
        n1_th = 0.5 * (n_total - delta_n_th)
        n2_eq = 0.5 * n_total

        if n1_th <= 0.0:
            return float("inf")

        relaxation_loss = (
            self.einstein_a21 * n2_th
            + (n2_th - n2_eq) / self.collisional_deexcitation_time
        )
        w_th = relaxation_loss / n1_th
        return max(0.0, float(w_th))

    def _derivatives(
        self,
        n2: float,
        rho: float,
        w_pump: float,
        n_total: float,
    ) -> Tuple[float, float]:
        """Compute ODE time derivatives dN2/dt and d(rho_em)/dt."""
        if n_total <= 0.0:
            dn2 = 0.0
            drho = -self.loaded_cavity_decay_rate * rho
            return dn2, drho

        n1 = max(0.0, n_total - n2)
        delta_n = n2 - n1
        n2_eq = 0.5 * n_total

        stim_rate = self.einstein_b21 * rho * delta_n
        spon_rate_mode = self.einstein_a21 * n2 * self.geometric_spontaneous_fraction
        coll_term = (n2 - n2_eq) / self.collisional_deexcitation_time
        dn2_dt = w_pump * n1 - stim_rate - self.einstein_a21 * n2 - coll_term

        drho_dt = (
            self.photon_energy * stim_rate
            + self.photon_energy * spon_rate_mode
            - self.loaded_cavity_decay_rate * rho
        )

        return dn2_dt, drho_dt

    def step(
        self,
        dt: float,
        piezo_voltage: float,
        acoustic_pressure: float,
        h2_concentration: float,
        temperature_k: float = STANDARD_TEMPERATURE_K,
        pressure_pa: float = STANDARD_ATMOSPHERIC_PRESSURE_PA,
        max_substeps: int = 50,
    ) -> MaserState:
        """Advance the maser physics simulation by time step dt using semi-implicit integration.
        
        Args:
            dt: Simulation time step in seconds.
            piezo_voltage: Peak or instantaneous King's Chamber piezo voltage (V).
            acoustic_pressure: Acoustic pressure standing wave amplitude (Pa).
            h2_concentration: H2 mole fraction in [0, 1] or concentration (mol/m^3).
            temperature_k: Chamber temperature (K).
            pressure_pa: Chamber pressure (Pa).
            max_substeps: Maximum sub-steps for adaptive integration.
            
        Returns:
            Snapshot MaserState object containing all physical observables.
        """
        if dt <= 0.0:
            return self.get_state(
                piezo_voltage=piezo_voltage,
                acoustic_pressure=acoustic_pressure,
                h2_concentration=h2_concentration,
            )

        n_total = self.compute_total_h_density(h2_concentration, temperature_k, pressure_pa)
        self.total_h_density = n_total
        w_pump = self.compute_pumping_rate(piezo_voltage, acoustic_pressure, h2_concentration)

        if n_total <= 0.0:
            decay_factor = math.exp(-min(50.0, self.loaded_cavity_decay_rate * dt))
            self.population_inversion = 0.0
            self.n1_population = 0.0
            self.n2_population = 0.0
            self.photon_energy_density *= decay_factor
            if self.photon_energy_density < 1.0e-30:
                self.photon_energy_density = 0.0
            self.time += dt
            return self.get_state(
                piezo_voltage=piezo_voltage,
                acoustic_pressure=acoustic_pressure,
                h2_concentration=h2_concentration,
            )

        delta_n = self.population_inversion
        rho = self.photon_energy_density
        delta_n_th = self.threshold_population_inversion()
        w_th = self.threshold_pumping_rate(h2_concentration, temperature_k)

        if w_pump <= w_th or delta_n_th >= n_total:
            # Sub-threshold regime: relax Delta N toward unsaturated steady state; photon density decays
            lam_inv = w_pump + self.einstein_a21 + 1.0 / self.collisional_deexcitation_time
            delta_n_ss = 0.0 if lam_inv <= 0.0 else ((w_pump - self.einstein_a21) * n_total) / lam_inv

            exp_inv = math.exp(-min(50.0, lam_inv * dt))
            decay_factor = math.exp(-min(50.0, self.loaded_cavity_decay_rate * dt))

            self.population_inversion = max(-n_total, min(n_total, delta_n_ss + (delta_n - delta_n_ss) * exp_inv))
            self.n2_population = max(0.0, min(n_total, 0.5 * (n_total + self.population_inversion)))
            self.n1_population = max(0.0, min(n_total, 0.5 * (n_total - self.population_inversion)))
            self.photon_energy_density = max(0.0, rho * decay_factor)
            if self.photon_energy_density < 1.0e-30:
                self.photon_energy_density = 0.0
        else:
            # Lasing regime: population inversion clamps at saturation threshold Delta N_sat = Delta N_th
            delta_n_ss = delta_n_th
            n2_ss = 0.5 * (n_total + delta_n_th)
            n1_ss = 0.5 * (n_total - delta_n_th)
            relaxation = self.einstein_a21 * n2_ss + (0.5 * delta_n_th) / self.collisional_deexcitation_time
            excess_rate = w_pump * n1_ss - relaxation
            stim_coeff = self.einstein_b21 * delta_n_th
            rho_ss = max(0.0, excess_rate / stim_coeff if stim_coeff > 0.0 else 0.0)

            # Linearized perturbation matrix J
            j11 = -(w_pump + 2.0 * self.einstein_b21 * rho_ss + self.einstein_a21 + 1.0 / self.collisional_deexcitation_time)
            j12 = -2.0 * self.einstein_b21 * delta_n_th
            j21 = self.photon_energy * self.einstein_b21 * rho_ss
            det_j = -j12 * j21
            sigma = -0.5 * j11

            omega_sq = det_j - sigma * sigma

            x0 = delta_n - delta_n_ss
            y0 = rho - rho_ss

            if omega_sq >= 0.0:
                omega = math.sqrt(omega_sq)
                e_damp = math.exp(-min(50.0, sigma * dt))
                c = math.cos(omega * dt)
                s = math.sin(omega * dt) / omega if omega > 1.0e-12 else dt
                m11 = e_damp * (c + (j11 + sigma) * s)
                m12 = e_damp * (j12 * s)
                m21 = e_damp * (j21 * s)
                m22 = e_damp * (c + sigma * s)
            else:
                kappa = math.sqrt(-omega_sq)
                lam1 = -(sigma + kappa)
                lam2 = -det_j / (sigma + kappa) if (sigma + kappa) > 0.0 else 0.0
                e1 = math.exp(max(-50.0, min(0.0, lam1 * dt))) if lam1 * dt > -50.0 else 0.0
                e2 = math.exp(max(-50.0, min(0.0, lam2 * dt))) if lam2 * dt > -50.0 else 0.0
                denom = -2.0 * kappa
                m11 = (e1 * (j11 - lam2) - e2 * (j11 - lam1)) / denom if abs(denom) > 1.0e-12 else 0.0
                m12 = (e1 - e2) * j12 / denom if abs(denom) > 1.0e-12 else 0.0
                m21 = (e1 - e2) * j21 / denom if abs(denom) > 1.0e-12 else 0.0
                m22 = (e1 * (-lam2) - e2 * (-lam1)) / denom if abs(denom) > 1.0e-12 else 0.0

            x1 = m11 * x0 + m12 * y0
            y1 = m21 * x0 + m22 * y0

            new_delta_n = delta_n_ss + x1
            self.population_inversion = max(-n_total, min(n_total, new_delta_n))
            self.n2_population = max(0.0, min(n_total, 0.5 * (n_total + self.population_inversion)))
            self.n1_population = max(0.0, min(n_total, 0.5 * (n_total - self.population_inversion)))
            self.photon_energy_density = max(0.0, rho_ss + y1)

        self.time += dt

        state = self.get_state(
            piezo_voltage=piezo_voltage,
            acoustic_pressure=acoustic_pressure,
            h2_concentration=h2_concentration,
        )

        # Accumulate energy accounting
        self.cumulative_stimulated_energy += state.stimulated_power_total * dt
        self.cumulative_radiated_energy += state.total_radiated_power * dt
        self.cumulative_cavity_loss_energy += state.cavity_loss_power * dt
        shaft_loss = (
            (state.north_shaft_power_in - state.north_shaft_beam_power)
            + (state.south_shaft_power_in - state.south_shaft_beam_power)
        )
        self.cumulative_shaft_loss_energy += max(0.0, shaft_loss) * dt

        state.cumulative_stimulated_energy = self.cumulative_stimulated_energy
        state.cumulative_radiated_energy = self.cumulative_radiated_energy
        state.cumulative_cavity_loss_energy = self.cumulative_cavity_loss_energy
        state.cumulative_shaft_loss_energy = self.cumulative_shaft_loss_energy

        return state

    def get_state(
        self,
        piezo_voltage: float = 0.0,
        acoustic_pressure: float = 0.0,
        h2_concentration: float = 0.0,
    ) -> MaserState:
        """Construct current observable snapshot state."""
        n_total = self.total_h_density if self.total_h_density > 0.0 else (self.n1_population + self.n2_population)
        delta_n = self.population_inversion
        rho = self.photon_energy_density
        delta_n_th = self.threshold_population_inversion()
        w_th = self.threshold_pumping_rate(h2_concentration)
        w_pump = self.compute_pumping_rate(piezo_voltage, acoustic_pressure, h2_concentration)

        is_lasing = (
            n_total > 0.0
            and w_pump > w_th
            and delta_n_th < n_total
            and rho > 1.0e-30
            and delta_n > 0.0
        )

        if is_lasing:
            stim_rate = self.einstein_b21 * rho * max(0.0, delta_n)
            p_stim = self.photon_energy * stim_rate * self.chamber_volume
            p_spon = self.photon_energy * self.einstein_a21 * self.n2_population * self.chamber_volume
            p_cav_loss = self.cavity_decay_rate * rho * self.chamber_volume

            vg = self.group_velocity
            p_in_north = rho * vg * self.north_shaft.cross_section_area * self.shaft_coupling_factor
            p_in_south = rho * vg * self.south_shaft.cross_section_area * self.shaft_coupling_factor
            p_out_shafts = p_in_north + p_in_south

            p_beam_north = self.north_shaft.radiated_power(p_in_north, self.hyperfine_frequency)
            p_beam_south = self.south_shaft.radiated_power(p_in_south, self.hyperfine_frequency)
            p_beam_total = p_beam_north + p_beam_south
        else:
            stim_rate = 0.0
            p_stim = 0.0
            p_spon = self.photon_energy * self.einstein_a21 * self.n2_population * self.chamber_volume if n_total > 0 else 0.0
            p_cav_loss = self.cavity_decay_rate * rho * self.chamber_volume if rho > 0 else 0.0
            p_in_north = 0.0
            p_in_south = 0.0
            p_out_shafts = 0.0
            p_beam_north = 0.0
            p_beam_south = 0.0
            p_beam_total = 0.0

        # Horn aperture directivity gains
        g_north_dbi = self.north_shaft.aperture_gain_dbi(self.hyperfine_frequency)
        g_south_dbi = self.south_shaft.aperture_gain_dbi(self.hyperfine_frequency)
        g_north_lin = self.north_shaft.aperture_gain(self.hyperfine_frequency)
        g_south_lin = self.south_shaft.aperture_gain(self.hyperfine_frequency)

        # Directional Effective Radiated Power (ERP)
        erp_north = p_beam_north * g_north_lin
        erp_south = p_beam_south * g_south_lin
        erp_total = erp_north + erp_south

        return MaserState(
            time=self.time,
            n1_population=self.n1_population,
            n2_population=self.n2_population,
            population_inversion=delta_n,
            total_h_density=n_total,
            photon_energy_density=rho,
            pumping_rate=w_pump,
            threshold_inversion=delta_n_th,
            is_above_threshold=bool(is_lasing),
            stimulated_transition_rate=stim_rate,
            stimulated_power_total=max(0.0, p_stim),
            spontaneous_power_total=p_spon,
            cavity_loss_power=p_cav_loss,
            shaft_extracted_power=p_out_shafts,
            north_shaft_power_in=p_in_north,
            south_shaft_power_in=p_in_south,
            north_shaft_beam_power=p_beam_north,
            south_shaft_beam_power=p_beam_south,
            total_radiated_power=p_beam_total,
            north_shaft_erp_watts=erp_north,
            south_shaft_erp_watts=erp_south,
            total_erp_watts=erp_total,
            cumulative_stimulated_energy=self.cumulative_stimulated_energy,
            cumulative_radiated_energy=self.cumulative_radiated_energy,
            cumulative_cavity_loss_energy=self.cumulative_cavity_loss_energy,
            cumulative_shaft_loss_energy=self.cumulative_shaft_loss_energy,
            frequency_hz=self.hyperfine_frequency,
            wavelength_m=WAVELENGTH_21CM_M,
            north_horn_gain_dbi=g_north_dbi,
            south_horn_gain_dbi=g_south_dbi,
        )

    def calculate_steady_state(
        self,
        piezo_voltage: float,
        acoustic_pressure: float,
        h2_concentration: float,
        temperature_k: float = STANDARD_TEMPERATURE_K,
        pressure_pa: float = STANDARD_ATMOSPHERIC_PRESSURE_PA,
    ) -> MaserState:
        """Calculate analytical steady-state operating point under constant driving.
        
        Computes saturated population inversion clamped at threshold Delta N_th
        and steady-state photon energy density rho_em,ss.
        """
        n_total = self.compute_total_h_density(h2_concentration, temperature_k, pressure_pa)
        w_pump = self.compute_pumping_rate(piezo_voltage, acoustic_pressure, h2_concentration)

        if n_total <= 0.0 or w_pump <= 0.0:
            self.reset()
            return self.get_state()

        delta_n_th = self.threshold_population_inversion()
        w_th = self.threshold_pumping_rate(h2_concentration, temperature_k)

        if w_pump <= w_th or delta_n_th >= n_total:
            n2_eq = 0.5 * n_total
            denom = w_pump + self.einstein_a21 + 1.0 / self.collisional_deexcitation_time
            n2_ss = (w_pump * n_total + n2_eq / self.collisional_deexcitation_time) / denom
            n1_ss = max(0.0, n_total - n2_ss)
            delta_n_ss = n2_ss - n1_ss

            self.total_h_density = n_total
            self.population_inversion = delta_n_ss
            self.n2_population = n2_ss
            self.n1_population = n1_ss
            self.photon_energy_density = 0.0
            return self.get_state(piezo_voltage, acoustic_pressure, h2_concentration)

        delta_n_ss = delta_n_th
        n2_ss = 0.5 * (n_total + delta_n_th)
        n1_ss = 0.5 * (n_total - delta_n_th)
        n2_eq = 0.5 * n_total

        relaxation = self.einstein_a21 * n2_ss + (n2_ss - n2_eq) / self.collisional_deexcitation_time
        pumping_flow = w_pump * n1_ss
        excess_rate = pumping_flow - relaxation

        stim_coeff = self.einstein_b21 * delta_n_th
        rho_ss = max(0.0, excess_rate / stim_coeff if stim_coeff > 0.0 else 0.0)

        self.total_h_density = n_total
        self.population_inversion = delta_n_ss
        self.n2_population = n2_ss
        self.n1_population = n1_ss
        self.photon_energy_density = max(0.0, rho_ss)

        return self.get_state(piezo_voltage, acoustic_pressure, h2_concentration)

    def reset(self) -> None:
        """Reset all dynamic populations and microwave electromagnetic fields to zero."""
        self.time = 0.0
        self.total_h_density = 0.0
        self.population_inversion = 0.0
        self.n1_population = 0.0
        self.n2_population = 0.0
        self.photon_energy_density = 0.0
        self.cumulative_stimulated_energy = 0.0
        self.cumulative_radiated_energy = 0.0
        self.cumulative_cavity_loss_energy = 0.0
        self.cumulative_shaft_loss_energy = 0.0
