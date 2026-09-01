"""Queen's Chamber Chemical Reaction Kinetics & Multi-Chamber Gas Transport.

Implements:
1. Chemical reaction kinetics: Zn(s) + 2HCl(aq) -> ZnCl2(aq) + H2(g)
   with Arrhenius temperature dependence, stoichiometric mass balance,
   and exothermic heat release.
2. 5-node interconnected gas transport network (Queen's Chamber,
   Horizontal Passage, Ascending Passage, Grand Gallery, King's Chamber)
   modeling multi-chamber advection-diffusion ODEs.
3. Dynamic gas mixture thermodynamics: local molar mass M_mix(X_H2),
   adiabatic index gamma_mix(X_H2), density rho_mix, and acoustic
   sound speed c_mix(X_H2, T) transitioning from 343.2 m/s to 1290.0 m/s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from engine.config import (
    GAS_CONSTANT_R,
    STANDARD_ATMOSPHERIC_PRESSURE_PA,
    STANDARD_TEMPERATURE_K,
    SimulationConfig,
)
from engine.geometry import PyramidGeometry


# Standard 5-node chamber network names
DEFAULT_NODE_NAMES: Tuple[str, ...] = (
    "queens_chamber",
    "horizontal_passage",
    "ascending_passage",
    "grand_gallery",
    "kings_chamber",
)

# Standard node volumes (m^3)
DEFAULT_NODE_VOLUMES: Tuple[float, ...] = (
    160.0,  # Queen's Chamber
    40.0,   # Horizontal Passage
    49.5,   # Ascending Passage
    350.0,  # Grand Gallery
    320.0,  # King's Chamber
)


@dataclass
class ReactionState:
    """State of the chemical reaction in Queen's Chamber."""

    time: float
    zn_mass_kg: float
    zn_moles: float
    hcl_moles: float
    hcl_concentration_mol_per_m3: float
    zncl2_moles: float
    h2_moles_generated_total: float
    reaction_rate_mol_per_s: float
    heat_release_rate_watts: float
    chamber_temperature_k: float
    cumulative_heat_joules: float
    is_active: bool


@dataclass
class GasNodeState:
    """Thermodynamic and transport state of a single gas network node."""

    node_id: int
    name: str
    volume_m3: float
    h2_concentration_mol_per_m3: float
    h2_moles: float
    total_gas_concentration_mol_per_m3: float
    h2_mole_fraction: float
    temperature_k: float
    pressure_pa: float
    molar_mass_kg_per_mol: float
    adiabatic_index: float
    sound_speed_m_per_s: float
    gas_density_kg_per_m3: float


@dataclass
class GasTransportState:
    """Snapshot of the full multi-node chemical and gas transport system."""

    time: float
    reaction: ReactionState
    nodes: List[GasNodeState]
    node_names: List[str]
    h2_concentrations: np.ndarray
    h2_mole_fractions: np.ndarray
    sound_speeds: np.ndarray
    temperatures: np.ndarray
    total_h2_moles_system: float
    mass_conservation_error_moles: float

    def get_node(self, name_or_id: Union[str, int]) -> GasNodeState:
        """Retrieve node state by name or integer index."""
        if isinstance(name_or_id, int):
            if 0 <= name_or_id < len(self.nodes):
                return self.nodes[name_or_id]
            raise IndexError(f"Node index out of range: {name_or_id}")
        name_clean = name_or_id.lower().strip()
        for node in self.nodes:
            if node.name.lower() == name_clean:
                return node
        # Partial match
        for node in self.nodes:
            if name_clean in node.name.lower():
                return node
        raise KeyError(f"Unknown node name: {name_or_id}")


class ChemicalGasTransport:
    """Chemical kinetics and multi-chamber gas transport simulation engine.

    Models:
    - Queen's Chamber zinc-acid reaction:
        Zn(s) + 2 HCl(aq) -> ZnCl2(aq) + H2(g) ^
        rate = k(T) * [HCl] * A_zn
        k(T) = k0 * exp(-Ea / (R * T))
    - Thermal balance:
        dT/dt = (Q_rxn - h_loss * (T - T_amb)) / C_thermal
    - 5-node interconnected advection-diffusion network:
        dC_i/dt = (1 / V_i) * [ n_dot_gen,i + sum_j(Q_ji * C_j) - sum_k(Q_ik * C_i)
                               + sum_j (D * A_ij / L_ij) * (C_j - C_i) ]
    - Dynamic mixture thermodynamics & sound speed:
        M_mix = X_H2 * M_H2 + (1 - X_H2) * M_air
        gamma_mix = X_H2 * gamma_H2 + (1 - X_H2) * gamma_air
        c_mix = sqrt(gamma_mix * R * T / M_mix)
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        geometry: Optional[PyramidGeometry] = None,
        initial_zn_moles: float = 1000.0,
        initial_hcl_moles: float = 2000.0,
        liquid_volume_m3: float = 1.0,
        zinc_surface_area_m2: float = 5.0,
        chamber_thermal_capacity: float = 5.0e5,
        chamber_heat_loss_coeff: float = 50.0,
        ambient_temperature_k: float = STANDARD_TEMPERATURE_K,
        ambient_pressure_pa: float = STANDARD_ATMOSPHERIC_PRESSURE_PA,
        diffusivity_h2: float = 6.1e-5,
        node_names: Optional[Sequence[str]] = None,
        node_volumes: Optional[Union[Sequence[float], Dict[str, float]]] = None,
        connections: Optional[Sequence[Tuple[int, int, float, float]]] = None,
        enable_reaction: bool = True,
        enable_diffusion: bool = True,
        enable_thermal_feedback: bool = True,
        calibrated_gamma_h2: Optional[float] = None,
    ) -> None:
        """Initialize chemical kinetics and gas transport solver."""
        self.config = config or SimulationConfig()
        self.geometry = geometry or PyramidGeometry()

        # Thermodynamic and physical constants
        self.R = GAS_CONSTANT_R
        self.T_amb = float(ambient_temperature_k)
        self.P_amb = float(ambient_pressure_pa)
        self.M_h2 = float(self.config.gas.molar_mass_h2)
        self.M_air = float(self.config.gas.molar_mass_air)
        self.gamma_air = float(self.config.gas.gamma_air)

        # Calibrate gamma_H2 to match target 1290.0 m/s at 20 deg C exactly
        if calibrated_gamma_h2 is not None:
            self.gamma_h2 = float(calibrated_gamma_h2)
        else:
            target_c_h2 = float(self.config.gas.sound_speed_h2_20c)
            # c = sqrt(gamma * R * T0 / M) => gamma = c^2 * M / (R * T0)
            self.gamma_h2 = (target_c_h2**2 * self.M_h2) / (self.R * STANDARD_TEMPERATURE_K)

        # Chemical kinetics parameters
        self.delta_H_rxn = float(self.config.hydrogen_rxn.reaction_enthalpy)  # -153.89 kJ/mol
        self.E_a = float(self.config.hydrogen_rxn.activation_energy)         # 38.5 kJ/mol
        self.k_0 = float(self.config.hydrogen_rxn.rate_pre_exponential)      # 1.25e4 m/(mol*s)
        self.molar_mass_zn = 0.06538  # kg/mol

        self.zinc_surface_area_m2 = float(zinc_surface_area_m2)
        self.liquid_volume_m3 = float(liquid_volume_m3)
        self.chamber_thermal_capacity = float(chamber_thermal_capacity)
        self.chamber_heat_loss_coeff = float(chamber_heat_loss_coeff)
        self.diffusivity_h2 = float(diffusivity_h2)

        self.enable_reaction = bool(enable_reaction)
        self.enable_diffusion = bool(enable_diffusion)
        self.enable_thermal_feedback = bool(enable_thermal_feedback)

        # Setup nodes
        if node_names is None:
            self._node_names: List[str] = list(DEFAULT_NODE_NAMES)
        else:
            self._node_names = list(node_names)

        num_nodes = len(self._node_names)

        if node_volumes is None:
            # Match survey volumes from geometry if default 5 nodes
            qc_vol = float(self.geometry.queens_chamber.survey_volume)
            ap_vol = float(self.geometry.ascending_passage.volume())
            kc_vol = float(self.geometry.kings_chamber.survey_volume)
            vols = [qc_vol, 40.0, ap_vol, 350.0, kc_vol]
            if len(vols) == num_nodes:
                self._node_volumes = np.array(vols, dtype=np.float64)
            else:
                self._node_volumes = np.array(DEFAULT_NODE_VOLUMES[:num_nodes], dtype=np.float64)
        elif isinstance(node_volumes, dict):
            self._node_volumes = np.array(
                [float(node_volumes.get(name, DEFAULT_NODE_VOLUMES[i % len(DEFAULT_NODE_VOLUMES)]))
                 for i, name in enumerate(self._node_names)],
                dtype=np.float64,
            )
        else:
            self._node_volumes = np.array([float(v) for v in node_volumes], dtype=np.float64)

        if len(self._node_volumes) != num_nodes:
            raise ValueError(f"Length of node_volumes ({len(self._node_volumes)}) must match node_names ({num_nodes})")

        if np.any(self._node_volumes <= 0):
            raise ValueError("All node volumes must be strictly positive")

        # Setup connections: List of (node_i, node_j, cross_section_area_m2, distance_m)
        if connections is None:
            # Default chain topology: QC (0) <-> HP (1) <-> AP (2) <-> GG (3) <-> KC (4)
            self._connections: List[Tuple[int, int, float, float]] = [
                (0, 1, 1.05 * 1.15, 20.0),  # Queen's Chamber <-> Horizontal Passage
                (1, 2, 1.05 * 1.20, 20.0),  # Horizontal Passage <-> Ascending Passage
                (2, 3, 1.05 * 1.20, 25.0),  # Ascending Passage <-> Grand Gallery
                (3, 4, 1.05 * 1.10, 10.0),  # Grand Gallery <-> King's Chamber
            ]
        else:
            self._connections = list(connections)

        # Validate connections
        for u, v, area, length in self._connections:
            if not (0 <= u < num_nodes and 0 <= v < num_nodes):
                raise ValueError(f"Connection indices ({u}, {v}) out of range for {num_nodes} nodes")
            if area <= 0 or length <= 0:
                raise ValueError(f"Connection area ({area}) and length ({length}) must be positive")

        # Initial state initialization
        self.time = 0.0
        self.initial_zn_moles = float(initial_zn_moles)
        self.initial_hcl_moles = float(initial_hcl_moles)

        self.reset()

    def reset(
        self,
        initial_zn_moles: Optional[float] = None,
        initial_hcl_moles: Optional[float] = None,
        initial_h2_concentrations: Optional[Union[np.ndarray, Sequence[float]]] = None,
        initial_qc_temperature: Optional[float] = None,
    ) -> None:
        """Reset simulation state to initial conditions."""
        self.time = 0.0
        if initial_zn_moles is not None:
            self.initial_zn_moles = max(0.0, float(initial_zn_moles))
        if initial_hcl_moles is not None:
            self.initial_hcl_moles = max(0.0, float(initial_hcl_moles))

        self.zn_moles = self.initial_zn_moles
        self.hcl_moles = self.initial_hcl_moles
        self.zncl2_moles = 0.0
        self.h2_moles_generated_total = 0.0
        self.cumulative_heat_joules = 0.0

        self.qc_temperature_k = (
            float(initial_qc_temperature)
            if initial_qc_temperature is not None
            else self.T_amb
        )

        num_nodes = len(self._node_names)
        if initial_h2_concentrations is not None:
            c_init = np.array(initial_h2_concentrations, dtype=np.float64)
            if len(c_init) != num_nodes:
                raise ValueError(f"initial_h2_concentrations length ({len(c_init)}) must match num_nodes ({num_nodes})")
            self.h2_concentrations = np.maximum(0.0, c_init)
        else:
            self.h2_concentrations = np.zeros(num_nodes, dtype=np.float64)

        # Node temperatures (QC tracks reaction temperature, others default to T_amb)
        self.node_temperatures_k = np.full(num_nodes, self.T_amb, dtype=np.float64)
        self.node_temperatures_k[0] = self.qc_temperature_k

        # Node pressures
        self.node_pressures_pa = np.full(num_nodes, self.P_amb, dtype=np.float64)

    @property
    def node_names(self) -> List[str]:
        """List of node names."""
        return list(self._node_names)

    @property
    def node_volumes(self) -> np.ndarray:
        """Array of node volumes in m^3."""
        return self._node_volumes.copy()

    @property
    def zn_mass_kg(self) -> float:
        """Current zinc mass in kg."""
        return self.zn_moles * self.molar_mass_zn

    @property
    def hcl_concentration_mol_per_m3(self) -> float:
        """Current HCl concentration in mol/m^3."""
        if self.liquid_volume_m3 <= 0:
            return 0.0
        return max(0.0, self.hcl_moles / self.liquid_volume_m3)

    @property
    def total_h2_moles(self) -> float:
        """Total moles of H2 currently inside all chambers of the network."""
        return float(np.sum(self.h2_concentrations * self._node_volumes))

    def compute_reaction_rate(self, temp_k: float, hcl_moles: float, zn_moles: float) -> float:
        """Compute instantaneous chemical reaction rate in mol/s."""
        if not self.enable_reaction or zn_moles <= 0.0 or hcl_moles <= 0.0:
            return 0.0
        if temp_k <= 0.0:
            return 0.0

        # Arrhenius rate constant k(T) = k0 * exp(-Ea / (R * T))
        k_t = self.k_0 * math.exp(-self.E_a / (self.R * temp_k))

        # Concentration [HCl] = n_HCl / V_liquid (mol/m^3)
        conc_hcl = hcl_moles / self.liquid_volume_m3

        # Rate d n_H2 / dt = k(T) * [HCl] * A_zn
        rate = k_t * conc_hcl * self.zinc_surface_area_m2
        return max(0.0, rate)

    def compute_mixture_molar_mass(self, x_h2: float) -> float:
        """Compute local gas mixture molar mass in kg/mol."""
        x = min(1.0, max(0.0, float(x_h2)))
        return x * self.M_h2 + (1.0 - x) * self.M_air

    def compute_adiabatic_index(self, x_h2: float) -> float:
        """Compute local gas mixture heat capacity ratio gamma."""
        x = min(1.0, max(0.0, float(x_h2)))
        return x * self.gamma_h2 + (1.0 - x) * self.gamma_air

    def compute_sound_speed(self, x_h2: float, temperature_k: Optional[float] = None) -> float:
        """Compute dynamic speed of sound in m/s for given H2 mole fraction and temperature."""
        x = min(1.0, max(0.0, float(x_h2)))
        t_k = float(temperature_k) if temperature_k is not None else self.T_amb
        if t_k <= 0.0:
            return 0.0

        m_mix = self.compute_mixture_molar_mass(x)
        gamma_mix = self.compute_adiabatic_index(x)

        return math.sqrt(gamma_mix * self.R * t_k / m_mix)

    def compute_mixture_density(
        self,
        x_h2: float,
        temperature_k: Optional[float] = None,
        pressure_pa: Optional[float] = None,
    ) -> float:
        """Compute local gas mixture mass density in kg/m^3."""
        x = min(1.0, max(0.0, float(x_h2)))
        t_k = float(temperature_k) if temperature_k is not None else self.T_amb
        p_pa = float(pressure_pa) if pressure_pa is not None else self.P_amb
        if t_k <= 0.0:
            return 0.0

        m_mix = self.compute_mixture_molar_mass(x)
        return (p_pa * m_mix) / (self.R * t_k)

    def _compute_derivatives(
        self,
        c_vec: np.ndarray,
        gen_rate_qc: float,
        advection_matrix: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute spatial concentration derivatives dC/dt for all nodes."""
        num_nodes = len(c_vec)
        d_flux = np.zeros(num_nodes, dtype=np.float64)

        # Generation term in Queen's Chamber (node 0)
        d_flux[0] += gen_rate_qc

        # Diffusion terms between connected nodes
        if self.enable_diffusion:
            for u, v, area, length in self._connections:
                conductance = self.diffusivity_h2 * area / length
                diff_flux = conductance * (c_vec[v] - c_vec[u])
                d_flux[u] += diff_flux
                d_flux[v] -= diff_flux

        # Advection terms if provided
        if advection_matrix is not None:
            # advection_matrix[j, i] = volumetric flow from j to i (m^3/s)
            inflow = np.sum(advection_matrix * c_vec[:, None], axis=0)
            outflow = np.sum(advection_matrix, axis=1) * c_vec
            d_flux += (inflow - outflow)

        return d_flux / self._node_volumes

    def step(
        self,
        dt: float,
        advection_flows: Optional[Union[np.ndarray, Dict[Tuple[int, int], float]]] = None,
    ) -> GasTransportState:
        """Advance the chemical reaction and gas transport state by time step dt (seconds)."""
        if dt <= 0.0:
            return self.get_state()

        num_nodes = len(self._node_names)
        adv_mat: Optional[np.ndarray] = None
        if advection_flows is not None:
            if isinstance(advection_flows, np.ndarray):
                if advection_flows.shape != (num_nodes, num_nodes):
                    raise ValueError(f"advection_flows matrix must have shape ({num_nodes}, {num_nodes})")
                adv_mat = np.maximum(0.0, advection_flows)
            elif isinstance(advection_flows, dict):
                adv_mat = np.zeros((num_nodes, num_nodes), dtype=np.float64)
                for (u, v), flow in advection_flows.items():
                    if 0 <= u < num_nodes and 0 <= v < num_nodes:
                        adv_mat[u, v] = max(0.0, float(flow))

        current_t = self.qc_temperature_k if self.enable_thermal_feedback else self.T_amb
        ideal_rate = self.compute_reaction_rate(current_t, self.hcl_moles, self.zn_moles)

        max_delta_h2_zn = self.zn_moles
        max_delta_h2_hcl = self.hcl_moles / 2.0
        ideal_delta_h2 = ideal_rate * dt

        actual_delta_h2 = min(ideal_delta_h2, max_delta_h2_zn, max_delta_h2_hcl)
        actual_delta_h2 = max(0.0, actual_delta_h2)

        actual_rate = actual_delta_h2 / dt if dt > 0 else 0.0

        self.zn_moles = max(0.0, self.zn_moles - actual_delta_h2)
        self.hcl_moles = max(0.0, self.hcl_moles - 2.0 * actual_delta_h2)
        self.zncl2_moles += actual_delta_h2
        self.h2_moles_generated_total += actual_delta_h2

        heat_release_rate_w = actual_rate * (-self.delta_H_rxn)
        delta_heat_j = heat_release_rate_w * dt
        self.cumulative_heat_joules += delta_heat_j

        if self.enable_thermal_feedback and self.chamber_thermal_capacity > 0:
            heat_loss_w = self.chamber_heat_loss_coeff * (self.qc_temperature_k - self.T_amb)
            net_heat_flow_w = heat_release_rate_w - heat_loss_w
            delta_t_k = (net_heat_flow_w / self.chamber_thermal_capacity) * dt
            self.qc_temperature_k = max(1.0, self.qc_temperature_k + delta_t_k)
        else:
            self.qc_temperature_k = self.T_amb

        self.node_temperatures_k[0] = self.qc_temperature_k

        max_rate = 0.0
        if self.enable_diffusion:
            for u, v, area, length in self._connections:
                cond = self.diffusivity_h2 * area / length
                max_rate = max(max_rate, cond / self._node_volumes[u], cond / self._node_volumes[v])
        if adv_mat is not None:
            max_adv_rate = np.max(np.sum(adv_mat, axis=1) / self._node_volumes)
            max_rate = max(max_rate, float(max_adv_rate))

        n_substeps = 1
        if max_rate > 0.0:
            required_substeps = int(math.ceil(dt * max_rate / 0.2))
            n_substeps = max(1, min(required_substeps, 1000))

        sub_dt = dt / n_substeps
        sub_gen_rate = actual_rate

        c_curr = self.h2_concentrations.copy()
        for _ in range(n_substeps):
            k1 = self._compute_derivatives(c_curr, sub_gen_rate, adv_mat)
            k2 = self._compute_derivatives(np.maximum(0.0, c_curr + 0.5 * sub_dt * k1), sub_gen_rate, adv_mat)
            k3 = self._compute_derivatives(np.maximum(0.0, c_curr + 0.5 * sub_dt * k2), sub_gen_rate, adv_mat)
            k4 = self._compute_derivatives(np.maximum(0.0, c_curr + sub_dt * k3), sub_gen_rate, adv_mat)

            c_curr = c_curr + (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            c_curr = np.maximum(0.0, c_curr)

        self.h2_concentrations = c_curr
        self.time += dt

        return self.get_state(actual_rate, heat_release_rate_w)

    def run(
        self,
        duration: float,
        dt: Optional[float] = None,
        advection_flows: Optional[Union[np.ndarray, Dict[Tuple[int, int], float]]] = None,
    ) -> List[GasTransportState]:
        """Run the simulation for the given duration (seconds) and return history of states."""
        step_dt = float(dt) if dt is not None else float(self.config.time_step_thermal_gas)
        if step_dt <= 0.0:
            step_dt = 0.01

        num_steps = int(math.ceil(duration / step_dt))
        history: List[GasTransportState] = [self.get_state()]

        for _ in range(num_steps):
            state = self.step(step_dt, advection_flows)
            history.append(state)

        return history

    def get_state(
        self,
        current_reaction_rate: Optional[float] = None,
        current_heat_release_w: Optional[float] = None,
    ) -> GasTransportState:
        """Construct the complete GasTransportState at the current simulation time."""
        num_nodes = len(self._node_names)

        # Compute reaction state
        if current_reaction_rate is None:
            current_t = self.qc_temperature_k if self.enable_thermal_feedback else self.T_amb
            current_reaction_rate = self.compute_reaction_rate(current_t, self.hcl_moles, self.zn_moles)
        if current_heat_release_w is None:
            current_heat_release_w = current_reaction_rate * (-self.delta_H_rxn)

        rxn_state = ReactionState(
            time=self.time,
            zn_mass_kg=self.zn_mass_kg,
            zn_moles=self.zn_moles,
            hcl_moles=self.hcl_moles,
            hcl_concentration_mol_per_m3=self.hcl_concentration_mol_per_m3,
            zncl2_moles=self.zncl2_moles,
            h2_moles_generated_total=self.h2_moles_generated_total,
            reaction_rate_mol_per_s=current_reaction_rate,
            heat_release_rate_watts=current_heat_release_w,
            chamber_temperature_k=self.qc_temperature_k,
            cumulative_heat_joules=self.cumulative_heat_joules,
            is_active=(self.zn_moles > 1e-6 and self.hcl_moles > 1e-6 and self.enable_reaction),
        )

        # Compute node states
        node_states: List[GasNodeState] = []
        mole_fractions = np.zeros(num_nodes, dtype=np.float64)
        sound_speeds = np.zeros(num_nodes, dtype=np.float64)

        for i in range(num_nodes):
            name = self._node_names[i]
            vol = self._node_volumes[i]
            c_h2 = self.h2_concentrations[i]
            n_h2 = c_h2 * vol
            t_node = self.node_temperatures_k[i]
            p_node = self.node_pressures_pa[i]

            # Total gas molar density at (P, T) under ideal gas law
            c_total = p_node / (self.R * t_node)
            x_h2 = float(np.clip(c_h2 / c_total, 0.0, 1.0)) if c_total > 0.0 else 0.0
            mole_fractions[i] = x_h2

            m_mix = self.compute_mixture_molar_mass(x_h2)
            gamma_mix = self.compute_adiabatic_index(x_h2)
            c_sound = self.compute_sound_speed(x_h2, t_node)
            sound_speeds[i] = c_sound
            rho_gas = self.compute_mixture_density(x_h2, t_node, p_node)

            node_states.append(
                GasNodeState(
                    node_id=i,
                    name=name,
                    volume_m3=vol,
                    h2_concentration_mol_per_m3=c_h2,
                    h2_moles=n_h2,
                    total_gas_concentration_mol_per_m3=c_total,
                    h2_mole_fraction=x_h2,
                    temperature_k=t_node,
                    pressure_pa=p_node,
                    molar_mass_kg_per_mol=m_mix,
                    adiabatic_index=gamma_mix,
                    sound_speed_m_per_s=c_sound,
                    gas_density_kg_per_m3=rho_gas,
                )
            )

        total_system_h2 = float(np.sum(self.h2_concentrations * self._node_volumes))
        mass_error = total_system_h2 - self.h2_moles_generated_total

        return GasTransportState(
            time=self.time,
            reaction=rxn_state,
            nodes=node_states,
            node_names=list(self._node_names),
            h2_concentrations=self.h2_concentrations.copy(),
            h2_mole_fractions=mole_fractions,
            sound_speeds=sound_speeds,
            temperatures=self.node_temperatures_k.copy(),
            total_h2_moles_system=total_system_h2,
            mass_conservation_error_moles=mass_error,
        )

    def get_node_index(self, name_or_id: Union[str, int]) -> int:
        """Get integer index for a node name or id."""
        if isinstance(name_or_id, int):
            if 0 <= name_or_id < len(self._node_names):
                return name_or_id
            raise IndexError(f"Node index out of range: {name_or_id}")
        name_clean = name_or_id.lower().strip()
        for idx, name in enumerate(self._node_names):
            if name.lower() == name_clean or name_clean in name.lower():
                return idx
        raise KeyError(f"Unknown node name: {name_or_id}")

    def get_node_sound_speed(self, name_or_id: Union[str, int]) -> float:
        """Get current speed of sound (m/s) in the specified node."""
        idx = self.get_node_index(name_or_id)
        x_h2 = self.get_node_h2_fraction(idx)
        return self.compute_sound_speed(x_h2, self.node_temperatures_k[idx])

    def get_node_h2_fraction(self, name_or_id: Union[str, int]) -> float:
        """Get current H2 mole fraction X_H2 in the specified node."""
        idx = self.get_node_index(name_or_id)
        c_h2 = self.h2_concentrations[idx]
        t_node = self.node_temperatures_k[idx]
        p_node = self.node_pressures_pa[idx]
        c_total = p_node / (self.R * t_node)
        return float(np.clip(c_h2 / c_total, 0.0, 1.0)) if c_total > 0.0 else 0.0

    def get_node_h2_concentration(self, name_or_id: Union[str, int]) -> float:
        """Get current H2 concentration (mol/m^3) in the specified node."""
        idx = self.get_node_index(name_or_id)
        return float(self.h2_concentrations[idx])

    def get_node_temperature(self, name_or_id: Union[str, int]) -> float:
        """Get current temperature (K) in the specified node."""
        idx = self.get_node_index(name_or_id)
        return float(self.node_temperatures_k[idx])
