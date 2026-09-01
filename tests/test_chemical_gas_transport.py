import math
import numpy as np
import pytest

from engine.config import (
    GAS_CONSTANT_R,
    STANDARD_ATMOSPHERIC_PRESSURE_PA,
    STANDARD_TEMPERATURE_K,
    SimulationConfig,
)
from engine.geometry import PyramidGeometry
from engine.physics.chemical_gas_transport import (
    DEFAULT_NODE_NAMES,
    DEFAULT_NODE_VOLUMES,
    ChemicalGasTransport,
    GasNodeState,
    GasTransportState,
    ReactionState,
)


def test_initial_state_defaults():
    sim = ChemicalGasTransport()
    state = sim.get_state()
    assert isinstance(state, GasTransportState)
    assert state.time == 0.0
    assert len(state.nodes) == 5
    assert state.reaction.is_active is True
    assert state.reaction.zn_moles == 1000.0
    assert state.reaction.hcl_moles == 2000.0
    assert state.reaction.zncl2_moles == 0.0
    assert state.reaction.h2_moles_generated_total == 0.0
    assert state.total_h2_moles_system == 0.0
    assert state.mass_conservation_error_moles == 0.0
    for node in state.nodes:
        assert node.h2_concentration_mol_per_m3 == 0.0
        assert node.h2_moles == 0.0
        assert node.h2_mole_fraction == 0.0
        assert abs(node.sound_speed_m_per_s - 343.2) < 0.5


def test_reaction_stoichiometry_and_mass_conservation():
    sim = ChemicalGasTransport(
        initial_zn_moles=500.0,
        initial_hcl_moles=1000.0,
        liquid_volume_m3=1.0,
        zinc_surface_area_m2=2.0,
    )
    dt = 0.01
    duration = 5.0
    history = sim.run(duration=duration, dt=dt)
    final_state = history[-1]

    delta_zn = sim.initial_zn_moles - final_state.reaction.zn_moles
    delta_hcl = sim.initial_hcl_moles - final_state.reaction.hcl_moles
    produced_zncl2 = final_state.reaction.zncl2_moles
    generated_h2 = final_state.reaction.h2_moles_generated_total
    system_h2 = final_state.total_h2_moles_system

    assert delta_zn > 0.0
    assert abs(delta_zn - generated_h2) < 1e-10
    assert abs(delta_hcl - 2.0 * generated_h2) < 1e-10
    assert abs(produced_zncl2 - generated_h2) < 1e-10
    assert abs(system_h2 - generated_h2) < 1e-10
    assert abs(final_state.mass_conservation_error_moles) < 1e-10


def test_reaction_limiting_reagent_zinc():
    sim = ChemicalGasTransport(
        initial_zn_moles=10.0,
        initial_hcl_moles=1000.0,
        zinc_surface_area_m2=10.0,
    )
    history = sim.run(duration=10.0, dt=0.01)
    final_state = history[-1]

    assert abs(final_state.reaction.zn_moles) < 1e-12
    assert abs(final_state.reaction.h2_moles_generated_total - 10.0) < 1e-10
    assert abs(final_state.reaction.hcl_moles - (1000.0 - 20.0)) < 1e-10
    assert abs(final_state.reaction.zncl2_moles - 10.0) < 1e-10
    assert final_state.reaction.reaction_rate_mol_per_s == 0.0
    assert final_state.reaction.is_active is False


def test_reaction_limiting_reagent_hcl():
    sim = ChemicalGasTransport(
        initial_zn_moles=1000.0,
        initial_hcl_moles=20.0,
        zinc_surface_area_m2=500.0,
    )
    history = sim.run(duration=10.0, dt=0.01)
    final_state = history[-1]

    assert abs(final_state.reaction.hcl_moles) < 1e-6
    assert abs(final_state.reaction.h2_moles_generated_total - 10.0) < 1e-5
    assert abs(final_state.reaction.zn_moles - (1000.0 - 10.0)) < 1e-5
    assert abs(final_state.reaction.zncl2_moles - 10.0) < 1e-5
    assert final_state.reaction.reaction_rate_mol_per_s < 1e-6
    assert final_state.reaction.is_active is False


def test_diffusion_equilibrium_variance_decay():
    c_init = [10.0, 0.0, 0.0, 0.0, 0.0]
    sim = ChemicalGasTransport(
        enable_reaction=False,
        diffusivity_h2=25.0,
    )
    sim.reset(initial_h2_concentrations=c_init)

    volumes = sim.node_volumes
    total_moles_initial = sum(c * v for c, v in zip(c_init, volumes))
    total_volume = sum(volumes)
    c_expected_eq = total_moles_initial / total_volume

    dt = 1.0
    var_initial = float(np.var(c_init))
    variances = [var_initial]

    for step_idx in range(5000):
        state = sim.step(dt)
        concs = state.h2_concentrations
        var_curr = float(np.var(concs))
        variances.append(var_curr)

        total_moles_curr = float(np.sum(concs * volumes))
        assert abs(total_moles_curr - total_moles_initial) < 1e-8

    final_concs = sim.h2_concentrations
    for c in final_concs:
        assert abs(c - c_expected_eq) < 1e-3

    assert variances[-1] < variances[0] * 1e-4
    for i in range(len(variances) - 1):
        assert variances[i + 1] <= variances[i] + 1e-12


def test_diffusion_total_mole_conservation():
    c_init = [5.0, 2.0, 8.0, 1.0, 4.0]
    sim = ChemicalGasTransport(
        enable_reaction=False,
        diffusivity_h2=0.05,
    )
    sim.reset(initial_h2_concentrations=c_init)

    vols = sim.node_volumes
    initial_moles = float(np.sum(np.array(c_init) * vols))

    for _ in range(200):
        state = sim.step(0.5)
        current_moles = state.total_h2_moles_system
        assert abs(current_moles - initial_moles) < 1e-10


def test_dynamic_sound_speed_air_and_pure_h2():
    sim = ChemicalGasTransport()

    c_air = sim.compute_sound_speed(x_h2=0.0, temperature_k=STANDARD_TEMPERATURE_K)
    assert abs(c_air - 343.2) <= 0.5

    c_pure_h2 = sim.compute_sound_speed(x_h2=1.0, temperature_k=STANDARD_TEMPERATURE_K)
    assert abs(c_pure_h2 - 1290.0) <= 1.0


def test_dynamic_sound_speed_strict_monotonicity():
    sim = ChemicalGasTransport()
    fractions = np.linspace(0.0, 1.0, 1001)
    speeds = [sim.compute_sound_speed(x, STANDARD_TEMPERATURE_K) for x in fractions]

    assert abs(speeds[0] - 343.2) <= 0.5
    assert abs(speeds[-1] - 1290.0) <= 1.0

    diffs = np.diff(speeds)
    assert np.all(diffs > 0), "Speed of sound must be strictly increasing with H2 mole fraction"


def test_gas_mixture_density_and_molar_mass():
    sim = ChemicalGasTransport()

    m_air = sim.compute_mixture_molar_mass(0.0)
    m_h2 = sim.compute_mixture_molar_mass(1.0)
    m_half = sim.compute_mixture_molar_mass(0.5)

    assert abs(m_air - sim.M_air) < 1e-12
    assert abs(m_h2 - sim.M_h2) < 1e-12
    assert abs(m_half - 0.5 * (sim.M_air + sim.M_h2)) < 1e-12

    rho_air = sim.compute_mixture_density(0.0, STANDARD_TEMPERATURE_K, STANDARD_ATMOSPHERIC_PRESSURE_PA)
    rho_h2 = sim.compute_mixture_density(1.0, STANDARD_TEMPERATURE_K, STANDARD_ATMOSPHERIC_PRESSURE_PA)
    assert 1.15 < rho_air < 1.25
    assert 0.075 < rho_h2 < 0.095
    assert rho_h2 < rho_air


def test_exothermic_thermal_balance_and_temperature_rise():
    sim = ChemicalGasTransport(
        initial_zn_moles=1000.0,
        initial_hcl_moles=2000.0,
        liquid_volume_m3=1.0,
        zinc_surface_area_m2=2.0,
        chamber_thermal_capacity=5.0e5,
        chamber_heat_loss_coeff=20.0,
        enable_thermal_feedback=True,
    )
    t_start = sim.qc_temperature_k
    assert t_start == STANDARD_TEMPERATURE_K

    history = sim.run(duration=5.0, dt=0.01)
    mid_state = history[len(history) // 2]
    final_state = history[-1]

    assert mid_state.reaction.heat_release_rate_watts > 0.0
    assert final_state.reaction.cumulative_heat_joules > 0.0
    assert final_state.reaction.chamber_temperature_k > t_start
    assert final_state.nodes[0].temperature_k > t_start


def test_arrhenius_temperature_acceleration():
    sim = ChemicalGasTransport(
        initial_zn_moles=1000.0,
        initial_hcl_moles=2000.0,
    )
    rate_293k = sim.compute_reaction_rate(293.15, sim.hcl_moles, sim.zn_moles)
    rate_323k = sim.compute_reaction_rate(323.15, sim.hcl_moles, sim.zn_moles)
    rate_353k = sim.compute_reaction_rate(353.15, sim.hcl_moles, sim.zn_moles)

    assert rate_323k > rate_293k
    assert rate_353k > rate_323k

    expected_ratio = math.exp(-sim.E_a / (sim.R * 323.15)) / math.exp(-sim.E_a / (sim.R * 293.15))
    actual_ratio = rate_323k / rate_293k
    assert abs(actual_ratio - expected_ratio) / expected_ratio < 1e-10


def test_numerical_stability_long_run():
    sim = ChemicalGasTransport(
        initial_zn_moles=1000.0,
        initial_hcl_moles=2000.0,
        zinc_surface_area_m2=2.0,
        diffusivity_h2=1.0e-4,
    )
    dt = 0.01
    total_steps = 10000

    for step_i in range(total_steps):
        state = sim.step(dt)

        assert not np.any(np.isnan(state.h2_concentrations))
        assert not np.any(np.isinf(state.h2_concentrations))
        assert np.all(state.h2_concentrations >= 0.0)
        assert np.all(state.h2_mole_fractions >= 0.0)
        assert np.all(state.h2_mole_fractions <= 1.0)
        assert np.all(state.sound_speeds >= 340.0)
        assert np.all(state.sound_speeds <= 1400.0)
        assert not math.isnan(state.reaction.chamber_temperature_k)
        assert state.reaction.chamber_temperature_k > 0.0

    assert sim.time == pytest.approx(100.0, abs=1e-6)


def test_various_time_steps():
    time_steps = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    duration = 2.0

    for dt in time_steps:
        sim = ChemicalGasTransport(
            initial_zn_moles=200.0,
            initial_hcl_moles=400.0,
            diffusivity_h2=1e-4,
        )
        states = sim.run(duration=duration, dt=dt)
        final_state = states[-1]

        assert np.all(final_state.h2_concentrations >= 0.0)
        assert np.all(final_state.h2_mole_fractions >= 0.0)
        assert np.all(final_state.h2_mole_fractions <= 1.0)
        assert abs(final_state.mass_conservation_error_moles) < 1e-6


def test_advection_transport_between_nodes():
    sim = ChemicalGasTransport(
        enable_reaction=False,
        diffusivity_h2=0.0,
    )
    sim.reset(initial_h2_concentrations=[10.0, 0.0, 0.0, 0.0, 0.0])

    adv_matrix = np.zeros((5, 5))
    adv_matrix[0, 1] = 0.5
    adv_matrix[1, 2] = 0.5
    adv_matrix[2, 3] = 0.5
    adv_matrix[3, 4] = 0.5

    for _ in range(50):
        state = sim.step(dt=1.0, advection_flows=adv_matrix)

    assert sim.h2_concentrations[0] < 10.0
    assert sim.h2_concentrations[1] > 0.0


def test_node_queries_and_state_api():
    sim = ChemicalGasTransport()
    sim.reset(initial_h2_concentrations=[5.0, 2.0, 1.0, 0.5, 0.1])

    qc_idx = sim.get_node_index("queens_chamber")
    kc_idx = sim.get_node_index("kings_chamber")
    assert qc_idx == 0
    assert kc_idx == 4

    assert sim.get_node_h2_concentration("queens_chamber") == 5.0
    assert sim.get_node_h2_concentration("kings_chamber") == 0.1

    qc_speed = sim.get_node_sound_speed("queens_chamber")
    kc_speed = sim.get_node_sound_speed("kings_chamber")
    assert qc_speed > kc_speed
    assert qc_speed > 343.2

    state = sim.get_state()
    qc_node = state.get_node("queens_chamber")
    assert qc_node.name == "queens_chamber"
    assert qc_node.h2_concentration_mol_per_m3 == 5.0

    with pytest.raises(KeyError):
        sim.get_node_index("nonexistent_chamber")


def test_custom_nodes_and_connections():
    custom_names = ["chamber_a", "chamber_b", "chamber_c"]
    custom_vols = [100.0, 50.0, 200.0]
    custom_conns = [
        (0, 1, 2.0, 10.0),
        (1, 2, 1.5, 15.0),
    ]

    sim = ChemicalGasTransport(
        node_names=custom_names,
        node_volumes=custom_vols,
        connections=custom_conns,
        enable_reaction=False,
    )
    sim.reset(initial_h2_concentrations=[10.0, 0.0, 0.0])

    assert len(sim.node_names) == 3
    assert sim.node_names[0] == "chamber_a"
    assert sim.node_names[1] == "chamber_b"
    assert sim.node_names[2] == "chamber_c"

    state = sim.step(1.0)
    assert len(state.nodes) == 3
    assert state.nodes[0].volume_m3 == 100.0
    assert state.nodes[1].volume_m3 == 50.0
    assert state.nodes[2].volume_m3 == 200.0
