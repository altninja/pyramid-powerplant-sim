import math
import numpy as np
import pytest

from engine.config import SimulationConfig
from engine.geometry import PyramidGeometry
from engine.physics.schumann_hydraulics import (
    HydraulicState,
    SubterraneanHydraulics,
)


def test_initial_state_defaults():
    sim = SubterraneanHydraulics()
    state = sim.get_state()
    assert isinstance(state, HydraulicState)
    assert state.time == 0.0
    assert state.bedrock_displacement == 0.0
    assert state.bedrock_velocity == 0.0
    assert state.cumulative_work == 0.0
    assert state.cavitation_active is False


def test_pure_schumann_fft_peak():
    sim = SubterraneanHydraulics(enable_water_hammer=False, enable_schumann=True)
    dt = 0.001
    duration = 40.0
    t_vals = np.arange(0, duration, dt)
    signal = np.array([sim.schumann_signal(t) for t in t_vals])

    freqs = np.fft.rfftfreq(len(t_vals), dt)
    fft_vals = np.abs(np.fft.rfft(signal))
    peak_freq = freqs[np.argmax(fft_vals)]

    assert abs(peak_freq - 7.83) <= 0.05

    peak1_idx = np.argmin(np.abs(freqs - 7.83))
    peak2_idx = np.argmin(np.abs(freqs - 14.30))
    peak3_idx = np.argmin(np.abs(freqs - 20.80))
    peak4_idx = np.argmin(np.abs(freqs - 27.30))

    assert fft_vals[peak1_idx] > fft_vals[peak2_idx]
    assert fft_vals[peak2_idx] > fft_vals[peak3_idx]
    assert fft_vals[peak3_idx] > fft_vals[peak4_idx]


def test_joukowsky_surge_pressure_amplitude():
    velocities = [1.0, 2.0, 3.5]
    config = SimulationConfig()
    rho = config.hydraulic.water_density
    c = config.hydraulic.water_sound_speed

    for v0 in velocities:
        sim = SubterraneanHydraulics(nominal_flow_velocity=v0)
        theor_max_pressure = rho * c * v0

        states = sim.simulate(duration=1.0, dt=0.0005)
        p_max = max(s.water_hammer_pressure for s in states)

        rel_error = abs(p_max - theor_max_pressure) / theor_max_pressure
        assert rel_error <= 0.01


def test_joukowsky_pressure_disabled():
    sim = SubterraneanHydraulics(enable_water_hammer=False, nominal_flow_velocity=2.0)
    states = sim.simulate(duration=0.5, dt=0.001)
    for s in states:
        assert s.water_hammer_pressure == 0.0
        assert s.hydraulic_force == 0.0
        assert s.flow_velocity == 0.0
        assert s.delta_v == 0.0


def test_energy_audit_conservation():
    sim = SubterraneanHydraulics()
    dt = 0.0005
    duration = 5.0
    states = sim.simulate(duration=duration, dt=dt)

    final_state = states[-1]
    assert final_state.cumulative_work >= 0.0

    m = sim.bedrock_mass
    k = sim.stiffness_k
    c = sim.damping_c

    e_mech_final = 0.5 * m * (final_state.bedrock_velocity**2) + 0.5 * k * (
        final_state.bedrock_displacement**2
    )
    v_arr = np.array([s.bedrock_velocity for s in states])
    e_dissipated = float(np.sum(c * (v_arr**2) * dt))

    total_accounted = e_mech_final + e_dissipated
    rel_diff = abs(final_state.cumulative_work - total_accounted) / total_accounted
    assert rel_diff < 0.01


def test_numerical_stability_10s_high_res():
    sim = SubterraneanHydraulics()
    dt = 0.0001
    duration = 10.0
    n_steps = int(duration / dt)

    final_state = None
    for _ in range(n_steps):
        final_state = sim.step(dt)

    assert final_state is not None
    assert not np.isnan(final_state.bedrock_displacement)
    assert not np.isinf(final_state.bedrock_displacement)
    assert not np.isnan(final_state.bedrock_velocity)
    assert not np.isinf(final_state.bedrock_velocity)
    assert abs(final_state.bedrock_displacement) < 1.0
    assert abs(final_state.bedrock_velocity) < 50.0


def test_acoustic_pressure_transmission():
    sim = SubterraneanHydraulics()
    config = SimulationConfig()
    z_limestone = config.limestone.density * config.limestone.sound_speed_longitudinal

    states = sim.simulate(duration=0.5, dt=0.001)
    for s in states:
        expected_acoustic_p = z_limestone * s.bedrock_velocity
        assert s.acoustic_pressure_ascending_passage == pytest.approx(
            expected_acoustic_p, rel=1e-6
        )


def test_free_decay_damping():
    sim = SubterraneanHydraulics(enable_water_hammer=False, enable_schumann=False)
    x0 = 0.05
    sim.reset(x0=x0, v0=0.0)

    dt = 0.0005
    duration = 2.0
    states = sim.simulate(duration=duration, dt=dt)

    decay_factor = math.exp(-sim.zeta * sim.omega0 * duration)
    envelope_bound = x0 * decay_factor

    assert abs(states[-1].bedrock_displacement) <= envelope_bound * 1.05


def test_cavitation_detection_flag():
    sim = SubterraneanHydraulics(vapor_pressure_threshold=1.0e6)
    states = sim.simulate(duration=0.5, dt=0.001)
    has_cavitation = any(s.cavitation_active for s in states)
    assert has_cavitation is True


def test_invalid_parameters_raise():
    with pytest.raises(ValueError, match="bedrock_mass must be positive"):
        SubterraneanHydraulics(bedrock_mass=-10.0)

    with pytest.raises(ValueError, match="bedrock_damping_ratio must be non-negative"):
        SubterraneanHydraulics(bedrock_damping_ratio=-0.1)

    with pytest.raises(ValueError, match="bedrock_natural_frequency must be positive"):
        SubterraneanHydraulics(bedrock_natural_frequency=0.0)

    with pytest.raises(ValueError, match="chamber_pulse_area must be positive"):
        SubterraneanHydraulics(chamber_pulse_area=-5.0)

    with pytest.raises(ValueError, match="pulse_frequency must be positive"):
        SubterraneanHydraulics(pulse_frequency=-1.0)

    sim = SubterraneanHydraulics()
    with pytest.raises(ValueError, match="dt must be positive"):
        sim.step(0.0)

    with pytest.raises(ValueError, match="duration must be positive"):
        sim.simulate(0.0, 0.01)

    with pytest.raises(ValueError, match="dt must be positive"):
        sim.simulate(1.0, -0.01)
