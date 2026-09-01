"""Master Energy Balance Accountant for Coupled Multi-Physics Simulation.

Tracks instantaneous power fluxes (boundary inputs, internal couplings, dissipations,
and radiated outputs) and integrates cumulative energy balances across all 6
subsystems:
1. Subterranean Hydraulics (seismic + water hammer mechanical work)
2. Queen's Chamber Chemistry (exothermic enthalpy release & gas heating)
3. Grand Gallery Acoustics (acoustic field energy & Helmholtz resonator dynamics)
4. Antechamber Acoustic Filter (gated acoustic transmission and reflection)
5. King's Chamber Piezoelectric Beams (granite modal strain, kinetic & electrostatic)
6. King's Chamber Microwave Maser (quantum population inversion, cavity RF & beamed microwaves)

Enforces the First Law of Thermodynamics on the total pyramid boundary:
    P_in = P_seismic + P_hydraulic + P_chemical
    P_out = P_maser_radiated
    P_loss = P_hydraulic_loss + P_acoustic_loss + P_beam_damping_loss + P_spark_loss
             + P_cavity_loss + P_shaft_loss + P_thermal_loss
    P_net = P_in - P_out - P_loss
    Delta E_stored(t) = E_sys(t) - E_sys(0)
    W_net(t) = Integral_0^t P_net(tau) dtau
    RelError(t) = |Delta E_stored(t) - W_net(t)| / max(E_scale, E_sys(t)) <= 1.0e-3
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PowerFlowState:
    """Instantaneous power fluxes across all simulation domains (in Watts)."""

    time: float = 0.0

    p_seismic_in: float = 0.0
    p_hydraulic_in: float = 0.0
    p_chemical_in: float = 0.0
    p_acoustic_in: float = 0.0
    p_piezo_in: float = 0.0
    p_maser_in: float = 0.0
    p_total_in: float = 0.0

    p_acoustic_transfer: float = 0.0
    p_piezo_transfer: float = 0.0

    p_maser_radiated: float = 0.0
    p_total_out: float = 0.0

    p_hydraulic_loss: float = 0.0
    p_acoustic_loss: float = 0.0
    p_beam_damping_loss: float = 0.0
    p_spark_loss: float = 0.0
    p_cavity_loss: float = 0.0
    p_shaft_loss: float = 0.0
    p_thermal_loss: float = 0.0
    p_total_loss: float = 0.0

    net_power_flux: float = 0.0

    def compute_totals(self) -> None:
        """Compute aggregated power totals and net power flux on boundary."""
        self.p_total_in = (
            self.p_seismic_in
            + self.p_hydraulic_in
            + self.p_chemical_in
        )
        self.p_total_out = self.p_maser_radiated
        self.p_total_loss = (
            self.p_hydraulic_loss
            + self.p_acoustic_loss
            + self.p_beam_damping_loss
            + self.p_spark_loss
            + self.p_cavity_loss
            + self.p_shaft_loss
            + self.p_thermal_loss
        )
        self.net_power_flux = self.p_total_in - self.p_total_out - self.p_total_loss


@dataclass
class EnergyBalanceSnapshot:
    """Cumulative energy accounting snapshot and conservation verification."""

    time: float = 0.0

    e_stored_hydraulic: float = 0.0
    e_stored_chemical_thermal: float = 0.0
    e_stored_acoustic: float = 0.0
    e_stored_beams: float = 0.0
    e_stored_maser: float = 0.0
    total_stored_energy: float = 0.0
    initial_stored_energy: float = 0.0
    delta_stored_energy: float = 0.0

    cumulative_energy_in: float = 0.0
    cumulative_energy_out: float = 0.0
    cumulative_energy_loss: float = 0.0
    net_work: float = 0.0

    energy_balance_error: float = 0.0
    relative_error: float = 0.0
    is_conserved: bool = True

    power_flow: PowerFlowState = field(default_factory=PowerFlowState)


class EnergyAccountant:
    """Master energy balance auditor enforcing the First Law of Thermodynamics.
    
    Tracks cumulative energy inputs, useful output beams, dissipated losses,
    and instantaneous stored energies across all physical subsystems.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-3,
        initial_stored_energy: float = 0.0,
        energy_scale: float = 1.0e6,
    ) -> None:
        self.tolerance = float(tolerance)
        self.energy_scale = float(energy_scale)
        self._default_initial_stored_energy = float(initial_stored_energy)
        self.reset(initial_stored_energy=self._default_initial_stored_energy)

    def reset(self, initial_stored_energy: Optional[float] = None) -> None:
        """Reset all integrated energy accumulators and history."""
        self.time: float = 0.0
        self.initial_stored_energy = (
            float(initial_stored_energy)
            if initial_stored_energy is not None
            else self._default_initial_stored_energy
        )

        # Cumulative energy integrals (Joules)
        self.cumulative_seismic_in: float = 0.0
        self.cumulative_hydraulic_in: float = 0.0
        self.cumulative_chemical_in: float = 0.0
        self.cumulative_acoustic_in: float = 0.0
        self.cumulative_piezo_in: float = 0.0
        self.cumulative_maser_in: float = 0.0
        self.cumulative_acoustic_transfer: float = 0.0
        self.cumulative_piezo_transfer: float = 0.0
        self.cumulative_energy_in: float = 0.0

        self.cumulative_maser_out: float = 0.0
        self.cumulative_energy_out: float = 0.0

        self.cumulative_hydraulic_loss: float = 0.0
        self.cumulative_acoustic_loss: float = 0.0
        self.cumulative_beam_damping_loss: float = 0.0
        self.cumulative_spark_loss: float = 0.0
        self.cumulative_cavity_loss: float = 0.0
        self.cumulative_shaft_loss: float = 0.0
        self.cumulative_thermal_loss: float = 0.0
        self.cumulative_energy_loss: float = 0.0

        self.latest_power_flow = PowerFlowState()
        self.latest_snapshot = EnergyBalanceSnapshot(
            time=0.0,
            total_stored_energy=self.initial_stored_energy,
            initial_stored_energy=self.initial_stored_energy,
            delta_stored_energy=0.0,
            is_conserved=True,
        )

    def set_initial_stored_energy(self, energy: float) -> None:
        """Set baseline stored energy at t=0."""
        self.initial_stored_energy = float(energy)
        self._default_initial_stored_energy = float(energy)

    def step(
        self,
        dt: float,
        p_seismic: float = 0.0,
        p_hydraulic: float = 0.0,
        p_chemical: float = 0.0,
        p_acoustic_in: float = 0.0,
        p_piezo_in: float = 0.0,
        p_maser_in: float = 0.0,
        p_acoustic_transfer: float = 0.0,
        p_piezo_transfer: float = 0.0,
        p_maser_radiated: float = 0.0,
        p_hydraulic_loss: float = 0.0,
        p_acoustic_loss: float = 0.0,
        p_beam_damping_loss: float = 0.0,
        p_spark_loss: float = 0.0,
        p_cavity_loss: float = 0.0,
        p_shaft_loss: float = 0.0,
        p_thermal_loss: float = 0.0,
        e_stored_hydraulic: float = 0.0,
        e_stored_chemical_thermal: float = 0.0,
        e_stored_acoustic: float = 0.0,
        e_stored_beams: float = 0.0,
        e_stored_maser: float = 0.0,
    ) -> EnergyBalanceSnapshot:
        """Advance energy accounting by dt and record instantaneous snapshot."""
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt}")

        power = PowerFlowState(
            time=self.time + dt,
            p_seismic_in=float(p_seismic),
            p_hydraulic_in=float(p_hydraulic),
            p_chemical_in=float(p_chemical),
            p_acoustic_in=float(p_acoustic_in),
            p_piezo_in=float(p_piezo_in),
            p_maser_in=float(p_maser_in),
            p_acoustic_transfer=float(p_acoustic_transfer),
            p_piezo_transfer=float(p_piezo_transfer),
            p_maser_radiated=max(0.0, float(p_maser_radiated)),
            p_hydraulic_loss=max(0.0, float(p_hydraulic_loss)),
            p_acoustic_loss=max(0.0, float(p_acoustic_loss)),
            p_beam_damping_loss=max(0.0, float(p_beam_damping_loss)),
            p_spark_loss=max(0.0, float(p_spark_loss)),
            p_cavity_loss=max(0.0, float(p_cavity_loss)),
            p_shaft_loss=max(0.0, float(p_shaft_loss)),
            p_thermal_loss=max(0.0, float(p_thermal_loss)),
        )
        power.compute_totals()

        self.cumulative_seismic_in += power.p_seismic_in * dt
        self.cumulative_hydraulic_in += power.p_hydraulic_in * dt
        self.cumulative_chemical_in += power.p_chemical_in * dt
        self.cumulative_energy_in = (
            self.cumulative_seismic_in
            + self.cumulative_hydraulic_in
            + self.cumulative_chemical_in
        )

        self.cumulative_acoustic_in += power.p_acoustic_in * dt
        self.cumulative_piezo_in += power.p_piezo_in * dt
        self.cumulative_maser_in += power.p_maser_in * dt
        self.cumulative_acoustic_transfer += power.p_acoustic_transfer * dt
        self.cumulative_piezo_transfer += power.p_piezo_transfer * dt

        self.cumulative_maser_out += power.p_maser_radiated * dt
        self.cumulative_energy_out = self.cumulative_maser_out

        self.cumulative_hydraulic_loss += power.p_hydraulic_loss * dt
        self.cumulative_acoustic_loss += power.p_acoustic_loss * dt
        self.cumulative_beam_damping_loss += power.p_beam_damping_loss * dt
        self.cumulative_spark_loss += power.p_spark_loss * dt
        self.cumulative_cavity_loss += power.p_cavity_loss * dt
        self.cumulative_shaft_loss += power.p_shaft_loss * dt
        self.cumulative_thermal_loss += power.p_thermal_loss * dt
        self.cumulative_energy_loss = (
            self.cumulative_hydraulic_loss
            + self.cumulative_acoustic_loss
            + self.cumulative_beam_damping_loss
            + self.cumulative_spark_loss
            + self.cumulative_cavity_loss
            + self.cumulative_shaft_loss
            + self.cumulative_thermal_loss
        )

        total_stored = (
            float(e_stored_hydraulic)
            + float(e_stored_chemical_thermal)
            + float(e_stored_acoustic)
            + float(e_stored_beams)
            + float(e_stored_maser)
        )

        if self.initial_stored_energy is None:
            self.initial_stored_energy = total_stored

        delta_stored = total_stored - self.initial_stored_energy
        net_work = (
            self.cumulative_energy_in
            - self.cumulative_energy_out
            - self.cumulative_energy_loss
        )
        balance_error = abs(delta_stored - net_work)

        scale = max(self.energy_scale, total_stored)
        rel_error = balance_error / max(scale, 1.0e-9)
        is_conserved = rel_error <= self.tolerance

        self.time += dt
        self.latest_power_flow = power
        self.latest_snapshot = EnergyBalanceSnapshot(
            time=self.time,
            e_stored_hydraulic=float(e_stored_hydraulic),
            e_stored_chemical_thermal=float(e_stored_chemical_thermal),
            e_stored_acoustic=float(e_stored_acoustic),
            e_stored_beams=float(e_stored_beams),
            e_stored_maser=float(e_stored_maser),
            total_stored_energy=total_stored,
            initial_stored_energy=self.initial_stored_energy,
            delta_stored_energy=delta_stored,
            cumulative_energy_in=self.cumulative_energy_in,
            cumulative_energy_out=self.cumulative_energy_out,
            cumulative_energy_loss=self.cumulative_energy_loss,
            net_work=net_work,
            energy_balance_error=balance_error,
            relative_error=rel_error,
            is_conserved=is_conserved,
            power_flow=power,
        )

        return self.latest_snapshot

    def get_snapshot(self) -> EnergyBalanceSnapshot:
        """Return the most recent energy balance snapshot."""
        return self.latest_snapshot

    def overall_efficiency(self) -> float:
        """Return overall system conversion efficiency W_out / max(W_in, 1e-9)."""
        if self.cumulative_energy_in <= 1.0e-9:
            return 0.0
        return self.cumulative_energy_out / self.cumulative_energy_in

    def check_conservation(self, tolerance: Optional[float] = None) -> bool:
        """Verify that cumulative energy balance satisfies conservation criterion."""
        tol = self.tolerance if tolerance is None else float(tolerance)
        return self.latest_snapshot.relative_error <= tol
