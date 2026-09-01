import math
import pytest

from engine.config import (
    GAS_CONSTANT_R,
    PLANCK_CONSTANT_H,
    SPEED_OF_LIGHT_C0,
    VACUUM_PERMITTIVITY_EPS0,
    VACUUM_PERMEABILITY_MU0,
    ROYAL_CUBIT_TO_METERS,
    GraniteProperties,
    LimestoneProperties,
    HydrogenReactionProperties,
    GasProperties,
    MaserProperties,
    SchumannProperties,
    SimulationConfig,
)
from engine.geometry import (
    Vector3D,
    BoundingBox3D,
    GraniteCofferGeometry,
    PyramidGeometry,
    get_chamber_volume,
    get_shaft_unit_vector,
    get_all_nodes,
    get_grand_gallery_slot_positions,
)


def test_physical_constants_exact_values():
    assert GAS_CONSTANT_R == pytest.approx(8.314462618, rel=1e-6)
    assert PLANCK_CONSTANT_H == pytest.approx(6.62607015e-34, rel=1e-6)
    assert SPEED_OF_LIGHT_C0 == pytest.approx(299792458.0, rel=1e-6)
    assert VACUUM_PERMITTIVITY_EPS0 == pytest.approx(8.8541878128e-12, rel=1e-6)
    assert VACUUM_PERMEABILITY_MU0 == pytest.approx(1.25663706212e-6, rel=1e-6)
    assert ROYAL_CUBIT_TO_METERS == pytest.approx(0.52360, rel=1e-6)


def test_granite_properties():
    granite = GraniteProperties()
    assert granite.density == pytest.approx(2650.0, rel=1e-3)
    assert granite.youngs_modulus == pytest.approx(55.0e9, rel=1e-3)
    assert granite.poisson_ratio == pytest.approx(0.24, rel=1e-3)
    assert granite.sound_speed_longitudinal == pytest.approx(4850.0, rel=1e-3)
    assert granite.sound_speed_shear == pytest.approx(2850.0, rel=1e-3)
    assert granite.quartz_fraction == pytest.approx(0.285, rel=1e-3)
    assert granite.dielectric_permittivity == pytest.approx(6.2, rel=1e-3)
    assert granite.loss_tangent == pytest.approx(0.015, rel=1e-3)
    assert granite.piezo_d33_eff == pytest.approx(0.35e-12, rel=1e-3)
    assert granite.piezo_d11_quartz == pytest.approx(2.3e-12, rel=1e-3)
    assert granite.piezo_g33_eff == pytest.approx(0.012, rel=1e-3)
    assert granite.acoustic_quality_factor == pytest.approx(350.0, rel=1e-3)


def test_limestone_properties():
    limestone = LimestoneProperties()
    assert limestone.density == pytest.approx(2450.0, rel=1e-3)
    assert limestone.sound_speed_longitudinal == pytest.approx(3200.0, rel=1e-3)
    assert limestone.youngs_modulus == pytest.approx(32.0e9, rel=1e-3)
    assert limestone.acoustic_attenuation_db_per_m == pytest.approx(0.45, rel=1e-3)


def test_hydrogen_and_gas_properties():
    rxn = HydrogenReactionProperties()
    assert rxn.reaction_enthalpy == pytest.approx(-153.89e3, rel=1e-3)
    assert rxn.gibbs_free_energy == pytest.approx(-147.16e3, rel=1e-3)
    assert rxn.activation_energy == pytest.approx(38.5e3, rel=1e-3)
    assert rxn.rate_pre_exponential == pytest.approx(1.25e4, rel=1e-3)

    gas = GasProperties()
    assert gas.molar_mass_h2 == pytest.approx(2.01588e-3, rel=1e-3)
    assert gas.molar_mass_air == pytest.approx(28.9647e-3, rel=1e-3)
    assert gas.sound_speed_h2_20c == pytest.approx(1290.0, rel=1e-3)
    assert gas.gamma_h2 == pytest.approx(1.405, rel=1e-3)
    assert gas.dynamic_viscosity_h2 == pytest.approx(8.82e-6, rel=1e-3)
    assert gas.sound_speed_air_20c == pytest.approx(343.2, rel=1e-3)
    assert gas.gamma_air == pytest.approx(1.400, rel=1e-3)
    assert gas.dynamic_viscosity_air == pytest.approx(1.81e-5, rel=1e-3)


def test_maser_and_schumann_properties():
    maser = MaserProperties()
    assert maser.hyperfine_frequency == pytest.approx(1420405751.7667, rel=1e-6)
    assert maser.einstein_a21 == pytest.approx(2.85e-15, rel=1e-3)
    assert maser.einstein_b21 == pytest.approx(5.67e20, rel=1e-3)
    assert maser.waveguide_cutoff_frequency == pytest.approx(681.35e6, rel=1e-3)
    assert maser.propagation_constant_beta == pytest.approx(26.08, rel=1e-3)

    schumann = SchumannProperties()
    assert schumann.mode1_frequency == pytest.approx(7.83, rel=1e-3)
    assert schumann.mode1_q == pytest.approx(5.0, rel=1e-3)
    assert schumann.mode2_frequency == pytest.approx(14.30, rel=1e-3)
    assert schumann.mode3_frequency == pytest.approx(20.80, rel=1e-3)
    assert schumann.mode4_frequency == pytest.approx(27.30, rel=1e-3)
    assert schumann.frequencies == (7.83, 14.30, 20.80, 27.30)


def test_simulation_config_initialization():
    config = SimulationConfig()
    assert config.granite.density == 2650.0
    assert config.time_step_acoustic == 1.0e-4
    assert config.total_duration == 10.0


def test_vector3d_algebra():
    v1 = Vector3D(x=1.0, y=2.0, z=3.0)
    v2 = Vector3D(x=4.0, y=-5.0, z=6.0)

    assert v1.magnitude() == pytest.approx(math.sqrt(14.0), rel=1e-5)
    u1 = v1.unit()
    assert u1.magnitude() == pytest.approx(1.0, rel=1e-5)
    assert v1.dot(v2) == pytest.approx(1.0 * 4.0 + 2.0 * (-5.0) + 3.0 * 6.0, rel=1e-5)
    vc = v1.cross(v2)
    assert vc.x == pytest.approx(2.0 * 6.0 - 3.0 * (-5.0))
    assert vc.y == pytest.approx(3.0 * 4.0 - 1.0 * 6.0)
    assert vc.z == pytest.approx(1.0 * (-5.0) - 2.0 * 4.0)
    assert v1.distance_to(v2) == pytest.approx(
        math.sqrt((1 - 4) ** 2 + (2 + 5) ** 2 + (3 - 6) ** 2)
    )
    add_v = v1 + v2
    assert add_v.to_tuple() == (5.0, -3.0, 9.0)
    sub_v = v2 - v1
    assert sub_v.to_tuple() == (3.0, -7.0, 3.0)
    mul_v = v1 * 2.5
    assert mul_v.to_tuple() == (2.5, 5.0, 7.5)
    rmul_v = 2.5 * v1
    assert rmul_v.to_tuple() == (2.5, 5.0, 7.5)
    div_v = v1 / 2.0
    assert div_v.to_tuple() == (0.5, 1.0, 1.5)


def test_bounding_box_3d():
    bb = BoundingBox3D(
        min_point=Vector3D(x=-5.0, y=-10.0, z=-15.0),
        max_point=Vector3D(x=5.0, y=10.0, z=15.0),
    )
    center = bb.center()
    assert center.to_tuple() == (0.0, 0.0, 0.0)
    dims = bb.dimensions()
    assert dims.to_tuple() == (10.0, 20.0, 30.0)
    assert bb.volume() == pytest.approx(6000.0)
    assert bb.contains_point(Vector3D(x=0.0, y=0.0, z=0.0)) is True
    assert bb.contains_point(Vector3D(x=6.0, y=0.0, z=0.0)) is False


def test_pyramid_base_geometry():
    pyr = PyramidGeometry()
    assert pyr.mean_base_side == pytest.approx(230.364, rel=1e-4)
    assert pyr.height == pytest.approx(146.580, rel=1e-4)
    assert pyr.slope_angle_degrees == pytest.approx(51.8444, rel=1e-3)
    assert (pyr.mean_base_side / ROYAL_CUBIT_TO_METERS) == pytest.approx(
        440.0, rel=1e-3
    )
    assert (pyr.height / ROYAL_CUBIT_TO_METERS) == pytest.approx(280.0, rel=1e-3)
    solid_vol = pyr.total_solid_volume()
    expected_vol = (1.0 / 3.0) * (230.364**2) * 146.580
    assert solid_vol == pytest.approx(expected_vol, rel=1e-4)
    assert solid_vol == pytest.approx(2593283.0, rel=1e-3)


def test_subterranean_chamber_geometry():
    pyr = PyramidGeometry()
    sub = pyr.subterranean_chamber
    assert sub.floor_datum == -30.00
    assert sub.center.z == -30.00
    assert sub.survey_volume == pytest.approx(280.0, rel=1e-3)
    dims = sub.bounding_box.dimensions()
    assert dims.x == pytest.approx(14.07, rel=1e-3)
    assert dims.y == pytest.approx(8.35, rel=1e-3)
    assert dims.z == pytest.approx(3.52, rel=1e-3)
    assert pyr.subterranean_pit_depth == 3.20
    assert pyr.subterranean_blind_passage_length == 16.38
    assert pyr.subterranean_blind_passage_cross_section == (0.74, 0.74)


def test_queens_chamber_and_shafts_geometry():
    pyr = PyramidGeometry()
    qc = pyr.queens_chamber
    assert qc.floor_datum == 21.20
    assert qc.survey_volume == pytest.approx(160.0, rel=1e-3)
    dims = qc.bounding_box.dimensions()
    assert dims.x == pytest.approx(5.75, rel=1e-3)
    assert dims.y == pytest.approx(5.23, rel=1e-3)
    assert dims.z == pytest.approx(6.23, rel=1e-3)

    assert pyr.queens_niche_height == 4.67
    assert pyr.queens_niche_base_width == 1.57
    assert pyr.queens_niche_depth == 1.04

    qn = pyr.queens_shaft_north
    assert qn.angle_degrees == pytest.approx(39.1167, rel=1e-3)
    assert qn.length == pytest.approx(65.0, rel=1e-3)
    assert qn.width == 0.21
    assert qn.height == 0.21
    qn_u = qn.get_unit_vector()
    assert qn_u.x == pytest.approx(0.0, abs=1e-5)
    assert qn_u.y == pytest.approx(-math.cos(math.radians(39.116667)), rel=1e-3)
    assert qn_u.z == pytest.approx(math.sin(math.radians(39.116667)), rel=1e-3)
    assert qn_u.magnitude() == pytest.approx(1.0, rel=1e-5)

    qs = pyr.queens_shaft_south
    assert qs.angle_degrees == pytest.approx(39.6078, rel=1e-3)
    assert qs.length == pytest.approx(63.60, rel=1e-3)
    assert qs.width == 0.21
    assert qs.height == 0.21
    qs_u = qs.get_unit_vector()
    assert qs_u.x == pytest.approx(0.0, abs=1e-5)
    assert qs_u.y == pytest.approx(math.cos(math.radians(39.607778)), rel=1e-3)
    assert qs_u.z == pytest.approx(math.sin(math.radians(39.607778)), rel=1e-3)
    assert qs_u.magnitude() == pytest.approx(1.0, rel=1e-5)


def test_grand_gallery_geometry():
    pyr = PyramidGeometry()
    gg = pyr.grand_gallery
    assert gg.length_along_incline == pytest.approx(46.61, rel=1e-3)
    assert gg.slope_angle_degrees == pytest.approx(26.04167, rel=1e-3)
    assert gg.vertical_height == pytest.approx(8.60, rel=1e-3)
    assert gg.width_base == pytest.approx(2.09, rel=1e-3)
    assert gg.width_roof == pytest.approx(1.05, rel=1e-3)
    assert gg.central_trench_width == pytest.approx(1.05, rel=1e-3)
    assert gg.side_ramps_width == pytest.approx(0.52, rel=1e-3)
    assert gg.num_slot_pairs == 28
    assert gg.slot_spacing == 1.68
    assert gg.slot_length == 0.54
    assert gg.slot_width == 0.16
    assert gg.slot_depth == 0.28
    assert gg.cavity_volume == pytest.approx(550.0, rel=1e-3)

    slot_positions = gg.get_slot_positions()
    assert len(slot_positions) == 28
    for west_slot, east_slot in slot_positions:
        assert west_slot.x < 0.0
        assert east_slot.x > 0.0
        assert west_slot.y == pytest.approx(east_slot.y, abs=1e-5)
        assert west_slot.z == pytest.approx(east_slot.z, abs=1e-5)


def test_antechamber_geometry():
    pyr = PyramidGeometry()
    ac = pyr.antechamber
    assert ac.floor_datum == 43.03
    assert ac.survey_volume == pytest.approx(19.67, rel=1e-3)
    dims = ac.bounding_box.dimensions()
    assert dims.x == pytest.approx(1.75, rel=1e-3)
    assert dims.y == pytest.approx(2.95, rel=1e-3)
    assert dims.z == pytest.approx(3.81, rel=1e-3)
    assert pyr.antechamber_granite_leaf_thickness == 0.41


def test_kings_chamber_and_shafts_geometry():
    pyr = PyramidGeometry()
    kc = pyr.kings_chamber
    assert kc.floor_datum == 43.03
    dims = kc.bounding_box.dimensions()
    assert dims.x == pytest.approx(10.470, rel=1e-3)
    assert dims.y == pytest.approx(5.235, rel=1e-3)
    assert dims.z == pytest.approx(5.840, rel=1e-3)
    raw_vol = dims.x * dims.y * dims.z
    assert raw_vol == pytest.approx(320.09, rel=1e-3)
    assert kc.survey_volume == pytest.approx(320.0, rel=1e-3)

    tiers = pyr.kings_relieving_tiers
    assert tiers.num_tiers == 5
    assert len(tiers.tier_names) == 5
    assert tiers.total_granite_beams == 43
    assert tiers.mean_beam_span == 6.50
    assert tiers.mean_beam_width == 1.20
    assert tiers.mean_beam_depth == 1.50

    kn = pyr.kings_shaft_north
    assert kn.angle_degrees == pytest.approx(32.4667, rel=1e-3)
    assert kn.length == pytest.approx(71.0, rel=1e-3)
    assert kn.width == 0.22
    assert kn.height == 0.22
    kn_u = kn.get_unit_vector()
    assert kn_u.x == pytest.approx(0.0, abs=1e-5)
    assert kn_u.y == pytest.approx(-math.cos(math.radians(32.466667)), rel=1e-3)
    assert kn_u.z == pytest.approx(math.sin(math.radians(32.466667)), rel=1e-3)
    assert kn_u.magnitude() == pytest.approx(1.0, rel=1e-5)

    ks = pyr.kings_shaft_south
    assert ks.angle_degrees == pytest.approx(45.0, rel=1e-3)
    assert ks.length == pytest.approx(53.0, rel=1e-3)
    assert ks.width == 0.22
    assert ks.height == 0.22
    ks_u = ks.get_unit_vector()
    assert ks_u.x == pytest.approx(0.0, abs=1e-5)
    assert ks_u.y == pytest.approx(math.cos(math.radians(45.0)), rel=1e-3)
    assert ks_u.z == pytest.approx(math.sin(math.radians(45.0)), rel=1e-3)
    assert ks_u.magnitude() == pytest.approx(1.0, rel=1e-5)


def test_granite_coffer():
    coffer = GraniteCofferGeometry()
    assert coffer.external_length == 2.278
    assert coffer.external_width == 0.977
    assert coffer.external_height == 1.048
    assert coffer.internal_length == 1.977
    assert coffer.internal_width == 0.677
    assert coffer.internal_height == 0.872

    int_vol = coffer.internal_volume()
    assert int_vol == pytest.approx(1.16715, rel=1e-3)
    assert int_vol == pytest.approx(1.166, rel=1e-2)

    solid_mass = coffer.mass_kg(granite_density=2650.0)
    assert solid_mass == pytest.approx(3088.0, rel=1e-2)


def test_passages_geometry():
    pyr = PyramidGeometry()
    desc = pyr.descending_passage
    assert desc.length == pytest.approx(105.23, rel=1e-3)
    assert desc.incline_angle_degrees == pytest.approx(26.52306, rel=1e-3)
    assert desc.width == 1.05
    assert desc.height == 1.20
    assert desc.volume() == pytest.approx(105.23 * 1.05 * 1.20, rel=1e-3)

    asc = pyr.ascending_passage
    assert asc.length == pytest.approx(39.28, rel=1e-3)
    assert asc.incline_angle_degrees == pytest.approx(26.04167, rel=1e-3)
    assert asc.width == 1.05
    assert asc.height == 1.20
    assert asc.volume() == pytest.approx(39.28 * 1.05 * 1.20, rel=1e-3)


def test_helper_functions():
    assert get_chamber_volume("subterranean") == pytest.approx(280.0)
    assert get_chamber_volume("queens") == pytest.approx(160.0)
    assert get_chamber_volume("kings") == pytest.approx(320.0)
    assert get_chamber_volume("antechamber") == pytest.approx(19.67)
    assert get_chamber_volume("grand_gallery") == pytest.approx(550.0)
    assert get_chamber_volume("coffer") == pytest.approx(1.16715, rel=1e-3)

    with pytest.raises(ValueError):
        get_chamber_volume("unknown_void")

    kc_n_u = get_shaft_unit_vector("kings_north")
    assert kc_n_u.magnitude() == pytest.approx(1.0)
    assert kc_n_u.y < 0.0
    assert kc_n_u.z > 0.0

    kc_s_u = get_shaft_unit_vector("kings_south")
    assert kc_s_u.magnitude() == pytest.approx(1.0)
    assert kc_s_u.y > 0.0
    assert kc_s_u.z > 0.0

    nodes = get_all_nodes()
    assert "base_center" in nodes
    assert "pyramid_apex" in nodes
    assert "subterranean_chamber" in nodes
    assert "queens_chamber" in nodes
    assert "grand_gallery_start" in nodes
    assert "grand_gallery_end" in nodes
    assert "antechamber" in nodes
    assert "kings_chamber" in nodes
    assert "kings_coffer" in nodes
    assert "kings_relieving_tier5" in nodes

    slots = get_grand_gallery_slot_positions()
    assert len(slots) == 28
