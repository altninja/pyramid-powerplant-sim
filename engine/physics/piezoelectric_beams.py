"""King's Chamber Rose Granite Piezoelectric Transduction Module.

Implements:
1. Monolithic rose granite beam modal dynamics under Euler-Bernoulli beam theory
   for the 43 beams across 5 relieving tiers (Davison, Wellington, Nelson,
   Lady Arbuthnot, Campbell).
2. Distributed acoustic-to-mechanical force coupling driven by King's Chamber
   acoustic standing waves.
3. Quartz dipole piezoelectric polarization and direct open-circuit voltage /
   charge generation.
4. Collective 5-tier array voltage stacking producing multi-kilovolt oscillating
   potentials.
5. Coffer spark gap dielectric breakdown and ionization discharge modeling.
6. Rigorous electromechanical energy balance and conservation accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, List, Optional, Tuple, Union
import numpy as np

from engine.config import (
    SimulationConfig,
    VACUUM_PERMITTIVITY_EPS0,
)
from engine.geometry import PyramidGeometry


# Clamped-Clamped Euler-Bernoulli beam transcendental roots (beta_n * L)
CLAMPED_CLAMPED_BETA_L: Tuple[float, ...] = (
    4.730040744862704,
    7.853204624095839,
    10.995607838001671,
    14.137165491257464,
    17.278759657399480,
    20.420352245626060,
)

# Number of beams per tier in the King's Chamber relieving chambers
TIER_BEAM_COUNTS: Tuple[int, ...] = (9, 9, 9, 9, 7)
TIER_NAMES: Tuple[str, ...] = (
    "Davison's Chamber",
    "Wellington's Chamber",
    "Nelson's Chamber",
    "Lady Arbuthnot's Chamber",
    "Campbell's Chamber",
)


@dataclass
class GraniteBeam:
    """Individual monolithic rose granite beam in a relieving tier.
    
    Governing Euler-Bernoulli modal dynamic equation for mode n:
        d2q_n/dt2 + 2*zeta_n*omega_n*dq_n/dt + omega_n^2*q_n = F_modal,n(t) / M_modal,n
    """

    tier_index: int = 0
    beam_index: int = 0
    tier_name: str = "Davison's Chamber"
    length: float = 6.50  # Span L (m)
    width: float = 1.20   # Width b (m)
    depth: float = 1.50   # Thickness/height h (m)
    density: float = 2650.0  # Granite density rho (kg/m^3)
    youngs_modulus: float = 55.0e9  # Young's modulus E (Pa)
    poisson_ratio: float = 0.24
    quartz_fraction: float = 0.285  # Quartz fraction (28.5%)
    dielectric_permittivity: float = 6.2  # eps_r
    piezo_d33_eff: float = 0.35e-12  # C/N
    piezo_g33_eff: float = 0.012  # V*m/N
    quality_factor: float = 350.0  # Acoustic Q factor
    num_modes: int = 4
    position_y: float = 0.0  # Long-axis position inside King's Chamber (m)

    # Dynamic modal state
    q: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    dq_dt: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    d2q_dt2: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))

    # Dynamic electromechanical state
    voltage: float = 0.0  # Open-circuit voltage V_b (V)
    prev_voltage: float = 0.0  # Previous time-step voltage for dV/dt (V)
    charge: float = 0.0   # Piezoelectric charge Q_b (C)
    polarization: float = 0.0  # Piezoelectric polarization P_z (C/m^2)
    electric_field: float = 0.0  # Internal electric field E_z (V/m)
    max_fiber_stress: float = 0.0  # Peak fiber stress sigma_max (Pa)
    mean_fiber_stress: float = 0.0  # Effective spatial mean stress sigma_bar (Pa)
    shear_stress: float = 0.0  # Transverse shear stress tau_xz (Pa)
    displacement_current: float = 0.0  # Displacement current I_disp = C_b * dV/dt (A)

    # Modal parameters precomputed on init
    betas: np.ndarray = field(init=False)
    sigmas: np.ndarray = field(init=False)
    omegas: np.ndarray = field(init=False)
    modal_masses: np.ndarray = field(init=False)
    zeta: float = field(init=False)

    def __post_init__(self) -> None:
        """Initialize modal constants, frequencies, and matrices."""
        if len(self.q) != self.num_modes:
            self.q = np.zeros(self.num_modes, dtype=float)
            self.dq_dt = np.zeros(self.num_modes, dtype=float)
            self.d2q_dt2 = np.zeros(self.num_modes, dtype=float)

        self._compute_modal_parameters()

    def _compute_modal_parameters(self) -> None:
        """Compute Euler-Bernoulli clamped-clamped modal eigenvalues and frequencies."""
        betas_list: List[float] = []
        sigmas_list: List[float] = []
        omegas_list: List[float] = []

        area = self.cross_section_area
        i_moment = self.second_moment_area
        mass_per_unit_len = self.density * area
        wave_coeff = math.sqrt(self.youngs_modulus * i_moment / mass_per_unit_len)

        for n in range(self.num_modes):
            if n < len(CLAMPED_CLAMPED_BETA_L):
                beta_l = CLAMPED_CLAMPED_BETA_L[n]
            else:
                beta_l = (2 * (n + 1) + 1) * math.pi / 2.0
            
            beta = beta_l / self.length
            betas_list.append(beta)

            cosh_bl = math.cosh(beta_l)
            cos_bl = math.cos(beta_l)
            sinh_bl = math.sinh(beta_l)
            sin_bl = math.sin(beta_l)

            # Avoid division by zero
            denom = sinh_bl - sin_bl
            sigma_val = (cosh_bl - cos_bl) / denom if abs(denom) > 1.0e-12 else 1.0
            sigmas_list.append(sigma_val)

            omega = (beta ** 2) * wave_coeff
            omegas_list.append(omega)

        self.betas = np.array(betas_list, dtype=float)
        self.sigmas = np.array(sigmas_list, dtype=float)
        self.omegas = np.array(omegas_list, dtype=float)
        
        # For normalized clamped-clamped mode shapes where int_0^L phi_n^2 dx = L:
        # Modal mass M_modal,n = rho * A * L = total beam mass M
        total_mass = self.mass
        self.modal_masses = np.full(self.num_modes, total_mass, dtype=float)
        self.zeta = 1.0 / (2.0 * max(self.quality_factor, 1.0))

        uniform_factors: List[float] = []
        for n in range(self.num_modes):
            beta_l = self.betas[n] * self.length
            beta = self.betas[n]
            sigma_val = self.sigmas[n]
            integral = (
                math.sinh(beta_l)
                - math.sin(beta_l)
                - sigma_val * (math.cosh(beta_l) - 1.0 - (1.0 - math.cos(beta_l)))
            ) / beta
            uniform_factors.append(self.width * integral)
        self.uniform_force_factors = np.array(uniform_factors, dtype=float)

    @property
    def cross_section_area(self) -> float:
        """Cross-sectional area A = b * h (m^2)."""
        return self.width * self.depth

    @property
    def second_moment_area(self) -> float:
        """Second moment of area I = b * h^3 / 12 (m^4)."""
        return self.width * (self.depth ** 3) / 12.0

    @property
    def mass(self) -> float:
        """Total monolithic mass of the beam M = rho * A * L (kg)."""
        return self.density * self.cross_section_area * self.length

    @property
    def capacitance(self) -> float:
        """Electrical capacitance C_b = eps_r * eps_0 * (L * b) / h (Farads)."""
        top_area = self.length * self.width
        eps = self.dielectric_permittivity * VACUUM_PERMITTIVITY_EPS0
        return eps * top_area / self.depth

    def natural_frequencies(self) -> np.ndarray:
        """Return natural frequencies f_n = omega_n / (2*pi) in Hz."""
        return self.omegas / (2.0 * math.pi)

    def mode_shape(self, n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the n-th clamped-clamped mode shape phi_n(x) at position x in [0, L]."""
        if n < 0 or n >= self.num_modes:
            raise ValueError(f"Mode index {n} out of range (0 to {self.num_modes - 1})")

        beta = self.betas[n]
        sigma_val = self.sigmas[n]
        bx = beta * x

        if isinstance(x, np.ndarray):
            return np.cosh(bx) - np.cos(bx) - sigma_val * (np.sinh(bx) - np.sin(bx))
        else:
            return math.cosh(bx) - math.cos(bx) - sigma_val * (math.sinh(bx) - math.sin(bx))

    @property
    def max_fiber_stress_mpa(self) -> float:
        """Peak fiber bending stress sigma_max in MPa."""
        return self.max_fiber_stress / 1.0e6

    @property
    def mean_fiber_stress_mpa(self) -> float:
        """Mean fiber stress sigma_bar in MPa."""
        return self.mean_fiber_stress / 1.0e6

    @property
    def shear_stress_mpa(self) -> float:
        """Transverse shear stress tau_xz in MPa."""
        return self.shear_stress / 1.0e6

    def mode_shape_curvature(self, n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the second derivative d^2 phi_n / dx^2 at position x."""
        if n < 0 or n >= self.num_modes:
            raise ValueError(f"Mode index {n} out of range (0 to {self.num_modes - 1})")

        beta = self.betas[n]
        sigma_val = self.sigmas[n]
        bx = beta * x
        factor = beta * beta

        if isinstance(x, np.ndarray):
            return factor * (np.cosh(bx) + np.cos(bx) - sigma_val * (np.sinh(bx) + np.sin(bx)))
        else:
            return factor * (math.cosh(bx) + math.cos(bx) - sigma_val * (math.sinh(bx) + math.sin(bx)))

    def mode_shape_third_derivative(self, n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the third derivative d^3 phi_n / dx^3 at position x."""
        if n < 0 or n >= self.num_modes:
            raise ValueError(f"Mode index {n} out of range (0 to {self.num_modes - 1})")

        beta = self.betas[n]
        sigma_val = self.sigmas[n]
        bx = beta * x
        factor = beta ** 3

        if isinstance(x, np.ndarray):
            return factor * (np.sinh(bx) - np.sin(bx) - sigma_val * (np.cosh(bx) + np.cos(bx)))
        else:
            return factor * (math.sinh(bx) - math.sin(bx) - sigma_val * (math.cosh(bx) + math.cos(bx)))

    def modal_force(
        self,
        pressure_func_or_const: Union[float, Callable[[float], float]],
        num_quad_points: int = 32,
    ) -> np.ndarray:
        """Compute generalized modal force vector F_modal,n = int_0^L p(x) * phi_n(x) * b dx."""
        if isinstance(pressure_func_or_const, (int, float)):
            p_val = float(pressure_func_or_const)
            if abs(p_val) < 1.0e-15:
                return np.zeros(self.num_modes, dtype=float)
            return self.uniform_force_factors * p_val

        # Gauss-Legendre quadrature for spatially varying pressure p(x)
        nodes, weights = np.polynomial.legendre.leggauss(num_quad_points)
        # Map from [-1, 1] to [0, L]
        x_pts = 0.5 * self.length * (nodes + 1.0)
        w_scaled = 0.5 * self.length * weights

        p_vals = np.array([pressure_func_or_const(float(xi)) for xi in x_pts])

        for n in range(self.num_modes):
            phi_vals = self.mode_shape(n, x_pts)
            forces[n] = np.sum(w_scaled * p_vals * phi_vals * self.width)

        return forces

    def update_stress_and_voltage(self, dt: float = 0.0) -> None:
        """Update fiber stress, shear stress, quartz polarization, voltage, and displacement current."""
        # Calculate RMS and peak fiber stress from modal curvatures
        # Curvature: kappa(x) = sum_n q_n * phi_n''(x)
        # For clamped-clamped beam, int_0^L (phi_n'')^2 dx = beta_n^4 * L (due to orthogonality)
        # Mean curvature RMS: sqrt( sum_n (q_n * beta_n^2)^2 )
        rms_curvature = math.sqrt(float(np.sum((self.q * (self.betas ** 2)) ** 2)))
        
        # Spatial mean surface fiber stress (at z = h/2):
        self.mean_fiber_stress = self.youngs_modulus * (self.depth * 0.5) * rms_curvature

        # Center-span peak curvature estimate: kappa(L/2)
        center_x = 0.5 * self.length
        center_curvature = 0.0
        for n in range(self.num_modes):
            center_curvature += self.q[n] * self.mode_shape_curvature(n, center_x)
        
        # Max fiber stress across beam span (at clamped ends or center)
        end_curvature = 0.0
        for n in range(self.num_modes):
            end_curvature += self.q[n] * self.mode_shape_curvature(n, 0.0)

        peak_curvature = max(abs(center_curvature), abs(end_curvature), rms_curvature * 1.414)
        self.max_fiber_stress = self.youngs_modulus * (self.depth * 0.5) * peak_curvature

        # Transverse shear stress tau_xz at clamped end x=0:
        end_shear_deriv = 0.0
        for n in range(self.num_modes):
            end_shear_deriv += self.q[n] * self.mode_shape_third_derivative(n, 0.0)
        self.shear_stress = (self.youngs_modulus * (self.depth ** 2) / 8.0) * abs(end_shear_deriv)

        # Quartz dipole polarization P_z = d33_eff * sigma_bar
        self.polarization = self.piezo_d33_eff * self.mean_fiber_stress

        # Direct open-circuit voltage:
        # V_b(t) = (g33_eff * sigma_bar * h) / (eps_r * eps_0)
        # Sign is aligned with fundamental mode displacement q_0
        sign_q = 1.0 if self.q[0] >= 0.0 else -1.0
        v_next = sign_q * self.piezo_g33_eff * self.mean_fiber_stress * self.depth

        # Displacement current I_disp,b = C_b * dV/dt
        if dt > 0.0:
            self.displacement_current = self.capacitance * (v_next - self.prev_voltage) / dt
        else:
            self.displacement_current = 0.0
        
        self.prev_voltage = self.voltage
        self.voltage = v_next
        
        # Charge Q_b = C_b * |V_b|
        self.charge = self.capacitance * abs(self.voltage)

        # Internal electric field E_z = V_b / h
        self.electric_field = self.voltage / self.depth

    def kinetic_energy(self) -> float:
        """Mechanical kinetic energy E_kin = 0.5 * sum(M_n * (dq_n/dt)^2) in Joules."""
        return 0.5 * float(np.sum(self.modal_masses * (self.dq_dt ** 2)))

    def strain_energy(self) -> float:
        """Mechanical elastic strain energy E_strain = 0.5 * sum(M_n * omega_n^2 * q_n^2) in Joules."""
        return 0.5 * float(np.sum(self.modal_masses * (self.omegas ** 2) * (self.q ** 2)))

    def mechanical_energy(self) -> float:
        """Total mechanical energy E_mech = E_kin + E_strain in Joules."""
        return self.kinetic_energy() + self.strain_energy()

    def electrical_energy(self) -> float:
        """Stored electrostatic energy E_elec = 0.5 * C_b * V_b^2 in Joules."""
        return 0.5 * self.capacitance * (self.voltage ** 2)

    def damping_power_loss(self) -> float:
        """Instantaneous power dissipation rate P_diss = sum(2 * zeta * omega_n * M_n * (dq_n/dt)^2) in Watts."""
        return float(np.sum(2.0 * self.zeta * self.omegas * self.modal_masses * (self.dq_dt ** 2)))

    def step_verlet(
        self,
        dt: float,
        modal_forces: np.ndarray,
    ) -> None:
        """Advance modal state by dt using Velocity Verlet integration."""
        m = self.modal_masses
        w = self.omegas
        z = self.zeta

        # Current acceleration: a_0 = F_0/m - 2*zeta*w*v_0 - w^2*q_0
        a0 = modal_forces / m - 2.0 * z * w * self.dq_dt - (w ** 2) * self.q
        self.d2q_dt2 = a0

        # Position update: q_1 = q_0 + v_0*dt + 0.5*a_0*dt^2
        q_next = self.q + self.dq_dt * dt + 0.5 * a0 * (dt ** 2)

        damping_factor = 1.0 + z * w * dt
        v_tentative = self.dq_dt + 0.5 * dt * (a0 + modal_forces / m - (w ** 2) * q_next)
        v_next = v_tentative / damping_factor

        self.q = q_next
        self.dq_dt = v_next
        self.d2q_dt2 = modal_forces / m - 2.0 * z * w * self.dq_dt - (w ** 2) * self.q

        self.update_stress_and_voltage(dt)

    def step_rk4(
        self,
        dt: float,
        modal_forces: np.ndarray,
    ) -> None:
        """Advance modal state by dt using 4th-order Runge-Kutta integration."""
        m = self.modal_masses
        w = self.omegas
        z = self.zeta

        def deriv(q_vec: np.ndarray, v_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            dq = v_vec
            dv = modal_forces / m - 2.0 * z * w * v_vec - (w ** 2) * q_vec
            return dq, dv

        k1_q, k1_v = deriv(self.q, self.dq_dt)
        k2_q, k2_v = deriv(self.q + 0.5 * dt * k1_q, self.dq_dt + 0.5 * dt * k1_v)
        k3_q, k3_v = deriv(self.q + 0.5 * dt * k2_q, self.dq_dt + 0.5 * dt * k2_v)
        k4_q, k4_v = deriv(self.q + dt * k3_q, self.dq_dt + dt * k3_v)

        self.q += (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        self.dq_dt += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        self.d2q_dt2 = modal_forces / m - 2.0 * z * w * self.dq_dt - (w ** 2) * self.q

        self.update_stress_and_voltage(dt)

    def reset(self) -> None:
        """Reset all modal vibrations and electrical states to rest."""
        self.q.fill(0.0)
        self.dq_dt.fill(0.0)
        self.d2q_dt2.fill(0.0)
        self.voltage = 0.0
        self.prev_voltage = 0.0
        self.charge = 0.0
        self.polarization = 0.0
        self.electric_field = 0.0
        self.max_fiber_stress = 0.0
        self.mean_fiber_stress = 0.0
        self.shear_stress = 0.0
        self.displacement_current = 0.0


@dataclass
class PiezoelectricState:
    """Snapshot state of the 5-tier King's Chamber piezoelectric beam array."""

    time: float = 0.0
    total_voltage: float = 0.0  # Aggregated stacked voltage across 5 tiers (V)
    tier_voltages: Tuple[float, ...] = ()  # Individual tier voltages (V)
    total_capacitance: float = 0.0  # Equivalent 5-tier array capacitance (F)
    total_charge: float = 0.0  # Stored charge (C)
    max_fiber_stress: float = 0.0  # Maximum mechanical fiber stress across all beams (Pa)
    mean_fiber_stress: float = 0.0  # Spatial average fiber stress (Pa)
    mean_polarization: float = 0.0  # Average quartz polarization (C/m^2)
    mean_electric_field: float = 0.0  # Average internal electric field (V/m)
    kinetic_energy: float = 0.0  # Total mechanical kinetic energy (J)
    strain_energy: float = 0.0  # Total mechanical strain energy (J)
    total_mechanical_energy: float = 0.0  # Total mechanical energy (J)
    stored_electrical_energy: float = 0.0  # Stored electrostatic energy (J)
    damping_power_loss: float = 0.0  # Instantaneous acoustic damping dissipation rate (W)
    cumulative_loss_energy: float = 0.0  # Cumulative dissipated damping energy (J)
    acoustic_input_power: float = 0.0  # Instantaneous acoustic driving power (W)
    cumulative_input_work: float = 0.0  # Cumulative acoustic input work (J)
    spark_triggered: bool = False  # True if coffer dielectric breakdown occurred this step
    spark_energy: float = 0.0  # Energy discharged in spark event this step (J)
    cumulative_spark_energy: float = 0.0  # Cumulative energy discharged by sparks (J)
    spark_count: int = 0  # Total number of spark discharges
    ion_density: float = 0.0  # Free ion density generated by breakdown (m^-3)
    modal_displacements_tier0: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=float)
    )
    all_beam_stresses_mpa: List[float] = field(default_factory=list)  # 43 individual beam peak fiber stresses (MPa)
    all_beam_voltages_v: List[float] = field(default_factory=list)  # 43 individual beam open-circuit voltages (V)
    all_beam_displacement_currents_a: List[float] = field(default_factory=list)  # 43 individual beam displacement currents (A)
    total_displacement_current_a: float = 0.0  # Total aggregate displacement current sum(|I_disp,b|) (A)
    array_impedance_ohms: float = 0.0  # Equivalent 5-tier array AC impedance Z_array(f) (Ohms)

    @property
    def beam_stresses_mpa(self) -> List[float]:
        """Alias for all_beam_stresses_mpa."""
        return self.all_beam_stresses_mpa

    @property
    def beam_voltages_v(self) -> List[float]:
        """Alias for all_beam_voltages_v."""
        return self.all_beam_voltages_v

    @property
    def beam_displacement_currents_a(self) -> List[float]:
        """Alias for all_beam_displacement_currents_a."""
        return self.all_beam_displacement_currents_a

    @property
    def displacement_current_a(self) -> float:
        """Alias for total_displacement_current_a."""
        return self.total_displacement_current_a


class PiezoelectricBeams:
    """Multi-tier Rose Granite Piezoelectric Transduction Module.
    
    Models 43 monolithic granite beams spanning 5 relieving tiers above King's Chamber:
    - Tier 0: Davison's Chamber (9 beams) - directly adjacent to King's Chamber ceiling
    - Tier 1: Wellington's Chamber (9 beams)
    - Tier 2: Nelson's Chamber (9 beams)
    - Tier 3: Lady Arbuthnot's Chamber (9 beams)
    - Tier 4: Campbell's Chamber (7 beams)
    
    Coupled to King's Chamber acoustic standing waves and coffer spark breakdown.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        geometry: Optional[PyramidGeometry] = None,
        num_modes: int = 4,
        integrator: str = "verlet",
        breakdown_voltage: float = 30000.0,
        spark_gap_distance: float = 0.01,
        tier_coupling_factor: float = 0.90,
    ) -> None:
        self.config = config or SimulationConfig()
        self.geometry = geometry or PyramidGeometry()
        self.num_modes = num_modes
        self.integrator = integrator.lower()
        self.breakdown_voltage = breakdown_voltage
        self.spark_gap_distance = spark_gap_distance
        self.tier_coupling_factor = tier_coupling_factor

        granite_cfg = self.config.granite
        tiers_geom = self.geometry.kings_relieving_tiers

        self.tiers: List[List[GraniteBeam]] = []
        self.all_beams: List[GraniteBeam] = []

        # King's Chamber length along y-axis (10.47 m)
        kc_length = self.geometry.kings_chamber.dimensions().x if hasattr(self.geometry.kings_chamber, "dimensions") else 10.470
        self.kc_length: float = float(kc_length)

        for tier_idx, count in enumerate(TIER_BEAM_COUNTS):
            tier_list: List[GraniteBeam] = []
            tier_name = TIER_NAMES[tier_idx]
            
            y_spacing = self.kc_length / max(count, 1)

            for b_idx in range(count):
                pos_y = (b_idx + 0.5) * y_spacing
                beam = GraniteBeam(
                    tier_index=tier_idx,
                    beam_index=b_idx,
                    tier_name=tier_name,
                    length=tiers_geom.mean_beam_span,
                    width=tiers_geom.mean_beam_width,
                    depth=tiers_geom.mean_beam_depth,
                    density=granite_cfg.density,
                    youngs_modulus=granite_cfg.youngs_modulus,
                    poisson_ratio=granite_cfg.poisson_ratio,
                    quartz_fraction=granite_cfg.quartz_fraction,
                    dielectric_permittivity=granite_cfg.dielectric_permittivity,
                    piezo_d33_eff=granite_cfg.piezo_d33_eff,
                    piezo_g33_eff=granite_cfg.piezo_g33_eff,
                    quality_factor=granite_cfg.acoustic_quality_factor,
                    num_modes=self.num_modes,
                    position_y=pos_y,
                )
                tier_list.append(beam)
                self.all_beams.append(beam)
            self.tiers.append(tier_list)

        # Simulation history tracking
        self.time: float = 0.0
        self.cumulative_input_work: float = 0.0
        self.cumulative_loss_energy: float = 0.0
        self.cumulative_spark_energy: float = 0.0
        self.spark_count: int = 0
        self.ion_density: float = 0.0

        # Precompute array capacitance
        self._compute_array_capacitance()
        self._init_matrix_params()

    def _init_matrix_params(self) -> None:
        self._mat_masses = np.array([b.modal_masses for b in self.all_beams], dtype=np.float64)
        self._mat_omegas = np.array([b.omegas for b in self.all_beams], dtype=np.float64)
        self._vec_zeta = np.array([b.zeta for b in self.all_beams], dtype=np.float64)[:, None]
        self._vec_couplings = np.array([self.tier_coupling_factor ** b.tier_index for b in self.all_beams], dtype=np.float64)[:, None]
        self._vec_spatial_cos = np.array([math.cos(math.pi * b.position_y / self.kc_length) for b in self.all_beams], dtype=np.float64)[:, None]
        self._mat_factors = np.array([b.uniform_force_factors for b in self.all_beams], dtype=np.float64)
        self._mat_betas_sq = np.array([b.betas ** 2 for b in self.all_beams], dtype=np.float64)
        self._mat_youngs = np.array([b.youngs_modulus for b in self.all_beams], dtype=np.float64)
        self._mat_depths = np.array([b.depth for b in self.all_beams], dtype=np.float64)
        self._mat_d33 = np.array([b.piezo_d33_eff for b in self.all_beams], dtype=np.float64)
        self._mat_g33 = np.array([b.piezo_g33_eff for b in self.all_beams], dtype=np.float64)
        self._mat_capacitances = np.array([b.capacitance for b in self.all_beams], dtype=np.float64)
        self._mat_q = np.array([b.q for b in self.all_beams], dtype=np.float64)
        self._mat_v = np.array([b.dq_dt for b in self.all_beams], dtype=np.float64)
        self._mat_prev_v = np.zeros(len(self.all_beams), dtype=np.float64)

        # Precompute curvature / derivative matrices at center and clamped end
        self._mat_phi_d2_center = np.array([[b.mode_shape_curvature(n, 0.5 * b.length) for n in range(self.num_modes)] for b in self.all_beams], dtype=np.float64)
        self._mat_phi_d2_end = np.array([[b.mode_shape_curvature(n, 0.0) for n in range(self.num_modes)] for b in self.all_beams], dtype=np.float64)
        self._mat_phi_d3_end = np.array([[b.mode_shape_third_derivative(n, 0.0) for n in range(self.num_modes)] for b in self.all_beams], dtype=np.float64)

        for i, b in enumerate(self.all_beams):
            b.q = self._mat_q[i]
            b.dq_dt = self._mat_v[i]

    def _compute_array_capacitance(self) -> None:
        """Compute equivalent 5-tier series-stacked capacitance."""
        # Single beam capacitance
        sample_beam = self.all_beams[0]
        c_beam = sample_beam.capacitance

        # For 5 tiers with N_k beams in parallel within tier k:
        # C_tier,k = N_k * C_beam
        # 1 / C_total = sum_k (1 / C_tier,k)
        inv_c_sum = 0.0
        self.tier_capacitances: List[float] = []
        for count in TIER_BEAM_COUNTS:
            c_tier = count * c_beam
            self.tier_capacitances.append(c_tier)
            inv_c_sum += 1.0 / c_tier

        self.total_capacitance = 1.0 / inv_c_sum

    @property
    def total_beam_count(self) -> int:
        """Total number of granite beams across all 5 tiers (43)."""
        return len(self.all_beams)

    def compute_tier_capacitances(self) -> List[float]:
        """Compute capacitance of each tier C_tier = sum_{b in tier} C_b (Farads)."""
        return [float(sum(b.capacitance for b in tier)) for tier in self.tiers]

    def compute_array_impedance(self, f: float = 438.0) -> float:
        """Compute equivalent 5-tier array AC capacitive impedance Z_array(f) in Ohms."""
        if f <= 0.0 or self.total_capacitance <= 0.0:
            return float("inf")
        return 1.0 / (2.0 * math.pi * f * self.total_capacitance)

    def compute_tier_impedance(self, tier_idx: int, f: float = 438.0) -> float:
        """Compute AC capacitive impedance of a single tier Z_tier(f) in Ohms."""
        if tier_idx < 0 or tier_idx >= len(self.tiers) or f <= 0.0:
            return float("inf")
        c_tier = sum(b.capacitance for b in self.tiers[tier_idx])
        if c_tier <= 0.0:
            return float("inf")
        return 1.0 / (2.0 * math.pi * f * c_tier)

    def compute_tier_voltages(self) -> Tuple[float, ...]:
        """Compute equivalent open-circuit voltage for each tier."""
        voltages: List[float] = []
        for tier in self.tiers:
            sign_val = 1.0 if (tier[0].q[0] >= 0.0 if len(tier[0].q) > 0 else True) else -1.0
            v_mag = float(np.mean([abs(b.voltage) for b in tier]))
            voltages.append(sign_val * v_mag)
        return tuple(voltages)

    def compute_total_voltage(self) -> float:
        """Compute total stacked voltage across 5 tiers (series across tiers)."""
        tier_v = self.compute_tier_voltages()
        return float(sum(tier_v))

    def step(
        self,
        dt: float,
        p_kc_acoustic: Union[float, Callable[[float, float], float]] = 0.0,
    ) -> PiezoelectricState:
        """Step the piezoelectric beam array by dt under King's Chamber acoustic drive.
        
        Args:
            dt: Time step in seconds.
            p_kc_acoustic: Instantaneous acoustic pressure in King's Chamber (Pa).
                           Can be a scalar or a function p(x, y) where x is along beam span,
                           y is along chamber length.
        """
        instantaneous_input_power = 0.0
        instantaneous_damping_loss = 0.0

        if isinstance(p_kc_acoustic, (int, float)) and self.integrator == "verlet":
            p_val = float(p_kc_acoustic)
            forces = self._mat_factors * (self._vec_couplings * self._vec_spatial_cos * p_val)
            v_prev = self._mat_v.copy()
            damping_factor = 1.0 + self._vec_zeta * self._mat_omegas * dt
            w_sq = self._mat_omegas ** 2
            a0 = forces / self._mat_masses - 2.0 * self._vec_zeta * self._mat_omegas * self._mat_v - w_sq * self._mat_q
            q_next = self._mat_q + self._mat_v * dt + 0.5 * a0 * (dt ** 2)
            v_tentative = self._mat_v + 0.5 * dt * (a0 + forces / self._mat_masses - w_sq * q_next)
            v_next = v_tentative / damping_factor

            self._mat_q = q_next
            self._mat_v = v_next
            v_avg = 0.5 * (v_prev + v_next)
            v2_avg = 0.5 * (v_prev ** 2 + v_next ** 2)

            instantaneous_input_power = float(np.sum(forces * v_avg))
            instantaneous_damping_loss = float(np.sum(2.0 * self._vec_zeta * self._mat_omegas * self._mat_masses * v2_avg))

            rms_curv = np.sqrt(np.sum((self._mat_q * self._mat_betas_sq) ** 2, axis=1))
            mean_stress = (self._mat_youngs * self._mat_depths * 0.5) * rms_curv

            center_curv = np.abs(np.sum(self._mat_q * self._mat_phi_d2_center, axis=1))
            end_curv = np.abs(np.sum(self._mat_q * self._mat_phi_d2_end, axis=1))
            peak_curv = np.maximum(np.maximum(center_curv, end_curv), rms_curv * 1.414)
            max_stress = self._mat_youngs * (self._mat_depths * 0.5) * peak_curv

            shear_stress = (self._mat_youngs * (self._mat_depths ** 2) / 8.0) * np.abs(np.sum(self._mat_q * self._mat_phi_d3_end, axis=1))

            pols = self._mat_d33 * mean_stress
            sign_q = np.where(self._mat_q[:, 0] >= 0.0, 1.0, -1.0)
            volts = sign_q * self._mat_g33 * mean_stress * self._mat_depths

            if dt > 0.0:
                disp_currents = self._mat_capacitances * (volts - self._mat_prev_v) / dt
            else:
                disp_currents = np.zeros(len(self.all_beams), dtype=np.float64)
            self._mat_prev_v = volts.copy()

            for i, beam in enumerate(self.all_beams):
                beam.q[:] = self._mat_q[i]
                beam.dq_dt[:] = self._mat_v[i]
                beam.prev_voltage = beam.voltage
                beam.voltage = float(volts[i])
                beam.mean_fiber_stress = float(mean_stress[i])
                beam.max_fiber_stress = float(max_stress[i])
                beam.shear_stress = float(shear_stress[i])
                beam.polarization = float(pols[i])
                beam.charge = float(self._mat_capacitances[i] * abs(volts[i]))
                beam.electric_field = float(volts[i] / beam.depth)
                beam.displacement_current = float(disp_currents[i])
        else:
            for tier_idx, tier in enumerate(self.tiers):
                coupling = self.tier_coupling_factor ** tier_idx

                for beam in tier:
                    beam_y = beam.position_y
                    cos_y = math.cos(math.pi * beam_y / self.kc_length)
                    if callable(p_kc_acoustic):
                        def p_func(x: float, c: float = coupling, y_pos: float = beam_y, cy: float = cos_y) -> float:
                            try:
                                return c * p_kc_acoustic(x, y_pos)
                            except TypeError:
                                return c * cy * p_kc_acoustic(x)

                        forces = beam.modal_force(p_func)
                    else:
                        forces = beam.modal_force(coupling * cos_y * float(p_kc_acoustic))

                    v_prev = beam.dq_dt.copy()

                    if self.integrator == "rk4":
                        beam.step_rk4(dt, forces)
                    else:
                        beam.step_verlet(dt, forces)

                    v_curr = beam.dq_dt
                    v_avg = 0.5 * (v_prev + v_curr)
                    v2_avg = 0.5 * (v_prev ** 2 + v_curr ** 2)

                    p_in_beam = float(np.sum(forces * v_avg))
                    instantaneous_input_power += p_in_beam

                    p_loss_beam = float(np.sum(2.0 * beam.zeta * beam.omegas * beam.modal_masses * v2_avg))
                    instantaneous_damping_loss += p_loss_beam

        self.time += dt
        self.cumulative_input_work += instantaneous_input_power * dt
        self.cumulative_loss_energy += instantaneous_damping_loss * dt

        # Compute collective array voltage and state
        tier_voltages = self.compute_tier_voltages()
        total_v = sum(tier_voltages)

        # Check for coffer dielectric breakdown / spark discharge
        spark_triggered = False
        spark_energy_step = 0.0

        if abs(total_v) >= self.breakdown_voltage:
            spark_triggered = True
            self.spark_count += 1
            # Discharged electrostatic pulse energy
            spark_energy_step = 0.5 * self.total_capacitance * (total_v ** 2)
            self.cumulative_spark_energy += spark_energy_step

            # Ionization density surge in H2 gas (13.6 eV = 2.179e-18 J per pair)
            h2_ion_energy = 13.6 * 1.602176634e-19
            ion_eff = 0.60
            gap_vol = max(self.spark_gap_distance * 0.1 * 0.1, 1.0e-5)
            delta_ions = (spark_energy_step * ion_eff) / (h2_ion_energy * gap_vol)
            self.ion_density += delta_ions

            # Discharge beam voltages towards zero
            for b in self.all_beams:
                b.voltage *= 0.1
                b.charge *= 0.1
            total_v *= 0.1
            tier_voltages = self.compute_tier_voltages()

        # Ion recombination decay
        self.ion_density = max(0.0, self.ion_density * math.exp(-dt / 1.0e-3))

        # Aggregate physical quantities
        e_kin = sum(b.kinetic_energy() for b in self.all_beams)
        e_strain = sum(b.strain_energy() for b in self.all_beams)
        e_mech = e_kin + e_strain
        e_elec = 0.5 * self.total_capacitance * (total_v ** 2)

        max_stress = max(b.max_fiber_stress for b in self.all_beams)
        mean_stress = float(np.mean([b.mean_fiber_stress for b in self.all_beams]))
        mean_pol = float(np.mean([b.polarization for b in self.all_beams]))
        mean_efield = float(np.mean([b.electric_field for b in self.all_beams]))

        q_tier0 = self.tiers[0][0].q.copy()

        all_beam_stresses = [b.max_fiber_stress / 1.0e6 for b in self.all_beams]
        all_beam_voltages = [b.voltage for b in self.all_beams]
        all_beam_disp_currents = [b.displacement_current for b in self.all_beams]
        total_disp_current = float(sum(abs(i) for i in all_beam_disp_currents))
        array_impedance = self.compute_array_impedance(438.0)

        return PiezoelectricState(
            time=self.time,
            total_voltage=total_v,
            tier_voltages=tier_voltages,
            total_capacitance=self.total_capacitance,
            total_charge=self.total_capacitance * abs(total_v),
            max_fiber_stress=max_stress,
            mean_fiber_stress=mean_stress,
            mean_polarization=mean_pol,
            mean_electric_field=mean_efield,
            kinetic_energy=e_kin,
            strain_energy=e_strain,
            total_mechanical_energy=e_mech,
            stored_electrical_energy=e_elec,
            damping_power_loss=instantaneous_damping_loss,
            cumulative_loss_energy=self.cumulative_loss_energy,
            acoustic_input_power=instantaneous_input_power,
            cumulative_input_work=self.cumulative_input_work,
            spark_triggered=spark_triggered,
            spark_energy=spark_energy_step,
            cumulative_spark_energy=self.cumulative_spark_energy,
            spark_count=self.spark_count,
            ion_density=self.ion_density,
            modal_displacements_tier0=q_tier0,
            all_beam_stresses_mpa=all_beam_stresses,
            all_beam_voltages_v=all_beam_voltages,
            all_beam_displacement_currents_a=all_beam_disp_currents,
            total_displacement_current_a=total_disp_current,
            array_impedance_ohms=array_impedance,
        )

    def reset(self) -> None:
        """Reset all beams and simulation state to equilibrium rest."""
        for beam in self.all_beams:
            beam.reset()
        if hasattr(self, "_mat_q"):
            self._mat_q.fill(0.0)
            self._mat_v.fill(0.0)
        if hasattr(self, "_mat_prev_v"):
            self._mat_prev_v.fill(0.0)
        self.time = 0.0
        self.cumulative_input_work = 0.0
        self.cumulative_loss_energy = 0.0
        self.cumulative_spark_energy = 0.0
        self.spark_count = 0
        self.ion_density = 0.0
