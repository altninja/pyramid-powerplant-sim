from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import numpy as np

from engine.config import SimulationConfig
from engine.geometry import PyramidGeometry


@dataclass
class HydraulicState:
    time: float
    valve_position: float
    flow_velocity: float
    delta_v: float
    water_hammer_pressure: float
    cavitation_active: bool
    schumann_excitation: float
    seismic_force: float
    hydraulic_force: float
    total_driving_force: float
    bedrock_displacement: float
    bedrock_velocity: float
    bedrock_acceleration: float
    mechanical_power: float
    cumulative_work: float
    acoustic_pressure_ascending_passage: float


class SubterraneanHydraulics:
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        geometry: Optional[PyramidGeometry] = None,
        bedrock_mass: float = 5.0e7,
        bedrock_natural_frequency: Optional[float] = None,
        bedrock_damping_ratio: float = 0.05,
        chamber_pulse_area: Optional[float] = None,
        nominal_flow_velocity: float = 2.0,
        pulse_frequency: Optional[float] = None,
        pulse_duty_cycle: float = 0.5,
        schumann_amplitudes: Tuple[float, float, float, float] = (1.0, 0.5, 0.25, 0.125),
        schumann_phases: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        seismic_force_amplitude: float = 1.0e5,
        enable_water_hammer: bool = True,
        enable_schumann: bool = True,
        vapor_pressure_threshold: float = -1.0e5,
    ) -> None:
        if bedrock_mass <= 0:
            raise ValueError("bedrock_mass must be positive")
        if bedrock_damping_ratio < 0:
            raise ValueError("bedrock_damping_ratio must be non-negative")

        self.config = config or SimulationConfig()
        self.geometry = geometry or PyramidGeometry()

        self.bedrock_mass = float(bedrock_mass)
        self.f0 = (
            float(bedrock_natural_frequency)
            if bedrock_natural_frequency is not None
            else float(self.config.schumann.mode1_frequency)
        )
        if self.f0 <= 0:
            raise ValueError("bedrock_natural_frequency must be positive")

        self.omega0 = 2.0 * math.pi * self.f0
        self.zeta = float(bedrock_damping_ratio)
        self.stiffness_k = self.bedrock_mass * (self.omega0**2)
        self.damping_c = 2.0 * self.zeta * self.bedrock_mass * self.omega0

        if chamber_pulse_area is not None:
            if chamber_pulse_area <= 0:
                raise ValueError("chamber_pulse_area must be positive")
            self.chamber_pulse_area = float(chamber_pulse_area)
        else:
            sub_box = self.geometry.subterranean_chamber.bounding_box
            dim = sub_box.dimensions()
            self.chamber_pulse_area = float(dim.x * dim.y)

        self.nominal_flow_velocity = float(nominal_flow_velocity)
        self.pulse_frequency = (
            float(pulse_frequency)
            if pulse_frequency is not None
            else float(self.config.schumann.mode1_frequency)
        )
        if self.pulse_frequency <= 0:
            raise ValueError("pulse_frequency must be positive")

        self.pulse_duty_cycle = float(pulse_duty_cycle)
        self.schumann_amplitudes = tuple(float(a) for a in schumann_amplitudes)
        self.schumann_phases = tuple(float(p) for p in schumann_phases)
        self.seismic_force_amplitude = float(seismic_force_amplitude)

        self.enable_water_hammer = bool(enable_water_hammer)
        self.enable_schumann = bool(enable_schumann)
        self.vapor_pressure_threshold = float(vapor_pressure_threshold)

        self.water_density = float(self.config.hydraulic.water_density)
        self.water_sound_speed = float(self.config.hydraulic.water_sound_speed)
        self.limestone_density = float(self.config.limestone.density)
        self.limestone_sound_speed = float(self.config.limestone.sound_speed_longitudinal)
        self.limestone_acoustic_impedance = (
            self.limestone_density * self.limestone_sound_speed
        )

        self.time = 0.0
        self.displacement = 0.0
        self.velocity = 0.0
        self.cumulative_work = 0.0

    def reset(self, x0: float = 0.0, v0: float = 0.0, t0: float = 0.0) -> None:
        self.time = float(t0)
        self.displacement = float(x0)
        self.velocity = float(v0)
        self.cumulative_work = 0.0

    def schumann_signal(self, t: float) -> float:
        freqs = self.config.schumann.frequencies
        signal = 0.0
        for k in range(min(4, len(freqs), len(self.schumann_amplitudes))):
            f_k = freqs[k]
            a_k = self.schumann_amplitudes[k]
            phi_k = self.schumann_phases[k] if k < len(self.schumann_phases) else 0.0
            signal += a_k * math.sin(2.0 * math.pi * f_k * t + phi_k)
        return signal

    def valve_position(self, t: float) -> float:
        if not self.enable_water_hammer or self.nominal_flow_velocity == 0.0:
            return 1.0
        phase = 2.0 * math.pi * self.pulse_frequency * t
        return 0.5 * (1.0 + math.cos(phase))

    def compute_joukowsky_pressure(self, delta_v: float) -> float:
        return self.water_density * self.water_sound_speed * delta_v

    def compute_flow_and_pressure(self, t: float) -> Tuple[float, float, float]:
        if not self.enable_water_hammer or self.nominal_flow_velocity == 0.0:
            return (0.0, 0.0, 0.0)

        tau = self.valve_position(t)
        v_flow = self.nominal_flow_velocity * tau
        delta_v = self.nominal_flow_velocity - v_flow
        p_hammer = self.compute_joukowsky_pressure(delta_v)
        return (v_flow, delta_v, p_hammer)

    def compute_driving_forces(self, t: float) -> Tuple[float, float, float, float, float]:
        schumann_val = self.schumann_signal(t)
        f_seismic = (
            self.seismic_force_amplitude * schumann_val if self.enable_schumann else 0.0
        )

        _, delta_v, p_hammer = self.compute_flow_and_pressure(t)
        f_hydraulic = self.chamber_pulse_area * p_hammer if self.enable_water_hammer else 0.0
        f_total = f_hydraulic + f_seismic

        return (schumann_val, f_seismic, p_hammer, f_hydraulic, f_total)

    def _compute_acceleration(self, t: float, x: float, v: float) -> float:
        _, _, _, _, f_total = self.compute_driving_forces(t)
        net_force = f_total - self.damping_c * v - self.stiffness_k * x
        return net_force / self.bedrock_mass

    def get_state(self) -> HydraulicState:
        v_flow, delta_v, p_hammer = self.compute_flow_and_pressure(self.time)
        schumann_val, f_seismic, _, f_hydraulic, f_total = self.compute_driving_forces(
            self.time
        )
        accel = self._compute_acceleration(self.time, self.displacement, self.velocity)
        p_mech = f_total * self.velocity
        p_acoustic = self.limestone_acoustic_impedance * self.velocity
        tau = self.valve_position(self.time)
        cavitation = p_hammer < self.vapor_pressure_threshold

        return HydraulicState(
            time=self.time,
            valve_position=tau,
            flow_velocity=v_flow,
            delta_v=delta_v,
            water_hammer_pressure=p_hammer,
            cavitation_active=cavitation,
            schumann_excitation=schumann_val,
            seismic_force=f_seismic,
            hydraulic_force=f_hydraulic,
            total_driving_force=f_total,
            bedrock_displacement=self.displacement,
            bedrock_velocity=self.velocity,
            bedrock_acceleration=accel,
            mechanical_power=p_mech,
            cumulative_work=self.cumulative_work,
            acoustic_pressure_ascending_passage=p_acoustic,
        )

    def step(self, dt: float) -> HydraulicState:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        t = self.time
        x = self.displacement
        v = self.velocity

        _, _, _, _, f_total_start = self.compute_driving_forces(t)
        p_mech_start = f_total_start * v

        k1_x = v
        k1_v = self._compute_acceleration(t, x, v)

        x_mid1 = x + 0.5 * dt * k1_x
        v_mid1 = v + 0.5 * dt * k1_v
        k2_x = v_mid1
        k2_v = self._compute_acceleration(t + 0.5 * dt, x_mid1, v_mid1)

        x_mid2 = x + 0.5 * dt * k2_x
        v_mid2 = v + 0.5 * dt * k2_v
        k3_x = v_mid2
        k3_v = self._compute_acceleration(t + 0.5 * dt, x_mid2, v_mid2)

        x_end = x + dt * k3_x
        v_end = v + dt * k3_v
        k4_x = v_end
        k4_v = self._compute_acceleration(t + dt, x_end, v_end)

        x_new = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        v_new = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        t_new = t + dt

        _, _, _, _, f_total_end = self.compute_driving_forces(t_new)
        p_mech_end = f_total_end * v_new
        delta_work = 0.5 * (p_mech_start + p_mech_end) * dt

        self.time = t_new
        self.displacement = x_new
        self.velocity = v_new
        self.cumulative_work += delta_work

        return self.get_state()

    def simulate(self, duration: float, dt: float) -> List[HydraulicState]:
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        n_steps = int(math.ceil(duration / dt))
        states = [self.get_state()]
        for _ in range(n_steps):
            states.append(self.step(dt))
        return states
