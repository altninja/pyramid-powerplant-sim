from typing import Tuple
from pydantic import BaseModel, Field


GAS_CONSTANT_R: float = 8.314462618
PLANCK_CONSTANT_H: float = 6.62607015e-34
SPEED_OF_LIGHT_C0: float = 299792458.0
VACUUM_PERMITTIVITY_EPS0: float = 8.8541878128e-12
VACUUM_PERMEABILITY_MU0: float = 1.25663706212e-6
BOLTZMANN_CONSTANT_KB: float = 1.380649e-23
STANDARD_ATMOSPHERIC_PRESSURE_PA: float = 101325.0
STANDARD_TEMPERATURE_K: float = 293.15
STANDARD_GRAVITY_G: float = 9.80665

ROYAL_CUBIT_TO_METERS: float = 0.52360


class GraniteProperties(BaseModel):
    density: float = Field(
        default=2650.0, description="Density in kg/m^3 (Aswan Rose Granite)"
    )
    youngs_modulus: float = Field(
        default=55.0e9, description="Young's modulus in Pa (55.0 GPa)"
    )
    poisson_ratio: float = Field(
        default=0.24, description="Poisson's ratio (dimensionless)"
    )
    sound_speed_longitudinal: float = Field(
        default=4850.0, description="P-wave speed in m/s"
    )
    sound_speed_shear: float = Field(default=2850.0, description="S-wave speed in m/s")
    quartz_fraction: float = Field(
        default=0.285, description="Quartz SiO2 volumetric fraction (28.5%)"
    )
    dielectric_permittivity: float = Field(
        default=6.2, description="Relative dielectric permittivity eps_r"
    )
    loss_tangent: float = Field(
        default=0.015, description="Dielectric loss tangent tan(delta)"
    )
    piezo_d33_eff: float = Field(
        default=0.35e-12,
        description="Effective polycrystalline d33 piezo tensor in C/N",
    )
    piezo_d11_quartz: float = Field(
        default=2.3e-12, description="Single-crystal quartz d11 piezo tensor in C/N"
    )
    piezo_g33_eff: float = Field(
        default=0.012, description="Effective piezo voltage coefficient in V*m/N"
    )
    piezo_g33_peak: float = Field(
        default=0.05, description="Peak open-circuit piezo voltage coefficient in V*m/N"
    )
    acoustic_quality_factor: float = Field(
        default=350.0, description="Resonant acoustic quality factor Q_gr"
    )


class LimestoneProperties(BaseModel):
    density: float = Field(
        default=2450.0, description="Density in kg/m^3 (Mokattam Limestone)"
    )
    sound_speed_longitudinal: float = Field(
        default=3200.0, description="P-wave speed in m/s"
    )
    youngs_modulus: float = Field(
        default=32.0e9, description="Young's modulus in Pa (32.0 GPa)"
    )
    acoustic_attenuation_db_per_m: float = Field(
        default=0.45, description="Attenuation at 438 Hz in dB/m"
    )


class HydrogenReactionProperties(BaseModel):
    reaction_enthalpy: float = Field(
        default=-153.89e3, description="Standard enthalpy of reaction in J/mol"
    )
    gibbs_free_energy: float = Field(
        default=-147.16e3, description="Standard Gibbs free energy in J/mol"
    )
    activation_energy: float = Field(
        default=38.5e3, description="Arrhenius activation energy in J/mol"
    )
    rate_pre_exponential: float = Field(
        default=1.25e4, description="Kinetic rate coefficient k0 in m/(mol*s)"
    )


class GasProperties(BaseModel):
    molar_mass_h2: float = Field(
        default=2.01588e-3, description="Molar mass of H2 in kg/mol"
    )
    molar_mass_air: float = Field(
        default=28.9647e-3, description="Molar mass of dry air in kg/mol"
    )
    sound_speed_h2_20c: float = Field(
        default=1290.0, description="Sound speed in pure H2 at 20 deg C in m/s"
    )
    gamma_h2: float = Field(
        default=1.405, description="Heat capacity ratio gamma for H2"
    )
    dynamic_viscosity_h2: float = Field(
        default=8.82e-6, description="Dynamic viscosity of H2 in Pa*s"
    )
    sound_speed_air_20c: float = Field(
        default=343.2, description="Sound speed in air at 20 deg C in m/s"
    )
    gamma_air: float = Field(
        default=1.400, description="Heat capacity ratio gamma for air"
    )
    dynamic_viscosity_air: float = Field(
        default=1.81e-5, description="Dynamic viscosity of air in Pa*s"
    )


class MaserProperties(BaseModel):
    hyperfine_frequency: float = Field(
        default=1420405751.7667, description="Hydrogen 21 cm line frequency in Hz"
    )
    einstein_a21: float = Field(
        default=2.85e-15,
        description="Einstein spontaneous emission coefficient in s^-1",
    )
    einstein_b21: float = Field(
        default=5.67e20,
        description="Einstein stimulated emission coefficient in m^3/(J*s^2)",
    )
    waveguide_cutoff_frequency: float = Field(
        default=681.35e6, description="TE10 waveguide cutoff frequency in Hz"
    )
    propagation_constant_beta: float = Field(
        default=26.08, description="Propagation constant beta at 1.42 GHz in rad/m"
    )
    v_ref: float = Field(
        default=5000.0, description="Reference piezoelectric pumping voltage in V (5.0 kV)"
    )
    p_ref: float = Field(
        default=100000.0, description="Reference acoustic pumping pressure in Pa (100 kPa)"
    )
    coupling_kappa_elec: float = Field(
        default=50.0, description="Electrical piezoelectric pump coupling coefficient in s^-1"
    )
    coupling_kappa_acoust: float = Field(
        default=10.0, description="Acoustic wave pump coupling coefficient in s^-1"
    )
    voltage_threshold: float = Field(
        default=100.0, description="Sub-threshold cutoff voltage in V"
    )
    pressure_threshold: float = Field(
        default=50.0, description="Sub-threshold cutoff acoustic pressure in Pa"
    )
    cavity_quality_factor: float = Field(
        default=5.0e4, description="Resonant King's Chamber cavity quality factor Q"
    )
    shaft_coupling_factor: float = Field(
        default=0.5, description="Shaft aperture power coupling factor"
    )


class SchumannProperties(BaseModel):
    mode1_frequency: float = Field(
        default=7.83, description="Fundamental Earth cavity mode in Hz"
    )
    mode1_q: float = Field(default=5.0, description="Quality factor Q for mode 1")
    mode2_frequency: float = Field(
        default=14.30, description="First harmonic mode in Hz"
    )
    mode2_q: float = Field(default=5.5, description="Quality factor Q for mode 2")
    mode3_frequency: float = Field(
        default=20.80, description="Second harmonic mode in Hz"
    )
    mode3_q: float = Field(default=6.0, description="Quality factor Q for mode 3")
    mode4_frequency: float = Field(
        default=27.30, description="Third harmonic mode in Hz"
    )
    mode4_q: float = Field(default=6.5, description="Quality factor Q for mode 4")

    @property
    def frequencies(self) -> Tuple[float, float, float, float]:
        return (
            self.mode1_frequency,
            self.mode2_frequency,
            self.mode3_frequency,
            self.mode4_frequency,
        )

    @property
    def q_factors(self) -> Tuple[float, float, float, float]:
        return (self.mode1_q, self.mode2_q, self.mode3_q, self.mode4_q)


class HydraulicProperties(BaseModel):
    water_density: float = Field(
        default=998.2, description="Water density at 20 deg C in kg/m^3"
    )
    water_sound_speed: float = Field(
        default=1482.0, description="Sound speed in water at 20 deg C in m/s"
    )
    water_bulk_modulus: float = Field(
        default=2.19e9, description="Bulk modulus of water in Pa"
    )


class AcousticTargetProperties(BaseModel):
    f_sharp_fundamental_hz: float = Field(
        default=438.0, description="Target F# fundamental resonance in Hz"
    )
    harmonic_modes_hz: Tuple[float, ...] = Field(
        default=(438.0, 876.0, 1314.0, 1752.0, 2190.0),
        description="F# harmonic overtone series in Hz",
    )


class SimulationConfig(BaseModel):
    granite: GraniteProperties = Field(default_factory=GraniteProperties)
    limestone: LimestoneProperties = Field(default_factory=LimestoneProperties)
    hydrogen_rxn: HydrogenReactionProperties = Field(
        default_factory=HydrogenReactionProperties
    )
    gas: GasProperties = Field(default_factory=GasProperties)
    maser: MaserProperties = Field(default_factory=MaserProperties)
    schumann: SchumannProperties = Field(default_factory=SchumannProperties)
    hydraulic: HydraulicProperties = Field(default_factory=HydraulicProperties)
    acoustic: AcousticTargetProperties = Field(default_factory=AcousticTargetProperties)

    time_step_acoustic: float = Field(
        default=1.0e-4, description="Acoustic wave solver time step in s"
    )
    time_step_thermal_gas: float = Field(
        default=1.0e-2, description="Gas diffusion / chemical solver time step in s"
    )
    time_step_schumann: float = Field(
        default=1.0e-3, description="Schumann / hydraulic solver time step in s"
    )
    total_duration: float = Field(
        default=10.0, description="Total simulation time in s"
    )
