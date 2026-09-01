"""Antechamber Acoustic Filter & Impedance Transfer Matrix Module.

Implements the Acoustic Transfer Matrix Method (TMM) modeling the Great Pyramid's
Antechamber granite leaves, wainscoting slots, and portal transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from engine.config import SimulationConfig
from engine.geometry import PyramidGeometry


@dataclass
class FilterSegment:
    """Discrete acoustic element in the Antechamber TMM cascade."""

    name: str
    length: float = 0.0
    area: float = 1.1655
    attenuation_coeff: float = 0.001
    segment_type: str = "duct"
    branch_inertance_len: float = 0.06
    branch_area: float = 0.10
    branch_resistance: float = 1.0
    series_resistance: float = 0.0
    series_inertance: float = 0.0

    def transfer_matrix(
        self,
        f: float,
        sound_speed: float = 343.2,
        density: float = 1.204,
    ) -> np.ndarray:
        """Compute the 2x2 complex Transfer Matrix M for this segment at frequency f (Hz)."""
        if self.segment_type in ("shunt_wainscot", "shunt", "shunt_resonator"):
            if f <= 0.0:
                z_b = complex(max(self.branch_resistance, 1.0e-9), 0.0)
            else:
                omega = 2.0 * math.pi * f
                m_b = density * self.branch_inertance_len / max(self.branch_area, 1.0e-6)
                z_b = complex(self.branch_resistance, omega * m_b)
            y_b = 1.0 / z_b
            return np.array([[1.0 + 0.0j, 0.0 + 0.0j], [y_b, 1.0 + 0.0j]], dtype=complex)

        if self.segment_type in ("series_impedance", "series_constriction"):
            omega = 2.0 * math.pi * f if f > 0.0 else 0.0
            z_s = complex(self.series_resistance, omega * self.series_inertance)
            return np.array([[1.0 + 0.0j, z_s], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=complex)

        if self.length <= 0.0 or self.area <= 0.0:
            return np.eye(2, dtype=complex)

        k_c = (2.0 * math.pi * f / max(sound_speed, 1.0)) - 1j * self.attenuation_coeff
        z_k = density * sound_speed / max(self.area, 1.0e-6)
        k_l = k_c * self.length

        cos_kl = np.cos(k_l)
        sin_kl = np.sin(k_l)

        return np.array(
            [
                [cos_kl, 1j * z_k * sin_kl],
                [1j / z_k * sin_kl, cos_kl],
            ],
            dtype=complex,
        )


@dataclass
class AntechamberState:
    """Dynamic operational state of the Antechamber Acoustic Filter."""

    time: float = 0.0
    p_in: float = 0.0
    p_out: float = 0.0
    u_in: float = 0.0
    u_out: float = 0.0
    p_trans: float = 0.0
    p_refl: float = 0.0
    p_inc: float = 0.0
    gas_sound_speed: float = 343.2
    gas_density: float = 1.204
    h2_fraction: float = 0.0
    transmission_loss_db_438: float = 0.0
    dominant_freq_hz: float = 438.0


class AntechamberFilter:
    """Acoustic Transfer Matrix Method (TMM) filter representing the Antechamber."""

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        geometry: Optional[PyramidGeometry] = None,
        inlet_area: Optional[float] = None,
        outlet_area: Optional[float] = None,
        num_leaves: int = 4,
        leaf_thickness: float = 0.41,
        chamber_length: float = 2.95,
        chamber_width: float = 1.75,
        chamber_height: float = 3.81,
        segments: Optional[List[FilterSegment]] = None,
        use_kings_chamber_load: bool = True,
        kings_chamber_length: Optional[float] = None,
        kings_chamber_attenuation: float = 0.002,
    ) -> None:
        self.config = config or SimulationConfig()
        self.geometry = geometry or PyramidGeometry()

        self.num_leaves = num_leaves
        self.leaf_thickness = leaf_thickness
        self.chamber_length = chamber_length
        self.chamber_width = chamber_width
        self.chamber_height = chamber_height

        portal_area = 1.05 * 1.11
        self.inlet_area = inlet_area if inlet_area is not None else portal_area
        self.outlet_area = outlet_area if outlet_area is not None else portal_area

        self.use_kings_chamber_load = bool(use_kings_chamber_load)
        if kings_chamber_length is not None:
            self.kings_chamber_length = float(kings_chamber_length)
        else:
            self.kings_chamber_length = float(self.geometry.kings_chamber.bounding_box.dimensions().x)
        self.kings_chamber_attenuation = float(kings_chamber_attenuation)

        if segments is not None:
            self.segments = list(segments)
        else:
            self.segments = self.build_default_segments()

        self.state = AntechamberState()

        self._fir_length = 512
        self._fir_dt: Optional[float] = None
        self._fir_sound_speed: Optional[float] = None
        self._fir_density: Optional[float] = None
        self._fir_kernel: np.ndarray = np.zeros(self._fir_length)
        self._input_buffer: np.ndarray = np.zeros(self._fir_length)

    def compute_kings_chamber_impedance(
        self,
        f: float,
        sound_speed: float = 343.2,
        density: float = 1.204,
        chamber_length: Optional[float] = None,
        attenuation_coeff: Optional[float] = None,
    ) -> complex:
        """Compute King's Chamber modal reactive termination impedance Z_L(f).
        
        Z_L(f) = -j * (rho * c / S_portal) * cot(k_c * L_KC)
        where k_c = (2*pi*f / c) - j * alpha.
        """
        l_kc = chamber_length if chamber_length is not None else self.kings_chamber_length
        alpha = attenuation_coeff if attenuation_coeff is not None else self.kings_chamber_attenuation
        s_portal = max(self.outlet_area, 1.0e-6)
        c = max(sound_speed, 1.0)
        z0 = density * c / s_portal

        if f <= 1.0e-6:
            omega = 2.0 * math.pi * max(f, 1.0e-9)
            c_a = s_portal * l_kc / (density * (c ** 2))
            return complex(0.0, -1.0 / (omega * max(c_a, 1.0e-12)))

        k_c = (2.0 * math.pi * f / c) - 1j * alpha
        k_l = k_c * l_kc
        sin_kl = np.sin(k_l)
        cos_kl = np.cos(k_l)
        if abs(sin_kl) < 1.0e-14:
            return complex(1.0e9, 0.0)

        cot_kl = cos_kl / sin_kl
        return -1j * z0 * cot_kl

    def resolve_load_impedance(
        self,
        f: float,
        sound_speed: float = 343.2,
        density: float = 1.204,
        load_impedance: Optional[Union[complex, float, str]] = None,
    ) -> complex:
        """Resolve acoustic load impedance for given frequency."""
        if isinstance(load_impedance, (complex, float, int)):
            return complex(load_impedance)
        if isinstance(load_impedance, str):
            load_str = load_impedance.lower().strip()
            if load_str in ("reactive", "kings_chamber", "kc", "cavity"):
                return self.compute_kings_chamber_impedance(f, sound_speed=sound_speed, density=density)
            if load_str in ("anechoic", "matched", "infinite", "open"):
                return complex(density * sound_speed / max(self.outlet_area, 1.0e-6), 0.0)
        if load_impedance is None:
            if self.use_kings_chamber_load and len(self.segments) > 0:
                return self.compute_kings_chamber_impedance(f, sound_speed=sound_speed, density=density)
            return complex(density * sound_speed / max(self.outlet_area, 1.0e-6), 0.0)
        return complex(density * sound_speed / max(self.outlet_area, 1.0e-6), 0.0)

    def build_default_segments(
        self,
        sound_speed: float = 343.2,
        density: float = 1.204,
    ) -> List[FilterSegment]:
        """Construct the cascade of acoustic segments for the Antechamber."""
        segments: List[FilterSegment] = []

        segments.append(
            FilterSegment(
                name="inlet_portal",
                length=1.05,
                area=self.inlet_area,
                attenuation_coeff=0.001,
                segment_type="duct",
            )
        )

        inter_len = max(
            0.0,
            (self.chamber_length - self.num_leaves * self.leaf_thickness)
            / max(self.num_leaves, 1),
        )

        for i in range(self.num_leaves):
            segments.append(
                FilterSegment(
                    name=f"wainscot_slot_{i+1}",
                    length=0.0,
                    area=self.inlet_area,
                    attenuation_coeff=0.001,
                    segment_type="shunt_wainscot",
                    branch_inertance_len=0.06,
                    branch_area=0.10,
                    branch_resistance=1.0,
                )
            )
            segments.append(
                FilterSegment(
                    name=f"granite_leaf_{i+1}",
                    length=self.leaf_thickness,
                    area=self.inlet_area,
                    attenuation_coeff=0.001,
                    segment_type="leaf",
                )
            )
            if inter_len > 0.0:
                segments.append(
                    FilterSegment(
                        name=f"inter_space_{i+1}",
                        length=inter_len,
                        area=self.inlet_area,
                        attenuation_coeff=0.001,
                        segment_type="duct",
                    )
                )

        segments.append(
            FilterSegment(
                name="exit_portal",
                length=2.58,
                area=self.outlet_area,
                attenuation_coeff=0.001,
                segment_type="duct",
            )
        )

        return segments

    def add_segment(self, segment: FilterSegment) -> None:
        """Add a segment to the filter cascade."""
        self.segments.append(segment)
        self._invalidate_fir()

    def clear_segments(self) -> None:
        """Clear all segments."""
        self.segments.clear()
        self._invalidate_fir()

    def transfer_matrix(
        self,
        f: float,
        sound_speed: float = 343.2,
        density: float = 1.204,
    ) -> np.ndarray:
        """Compute total cascaded Transfer Matrix M_total(f) = M_1 * M_2 * ... * M_N."""
        if not self.segments:
            return np.eye(2, dtype=complex)

        m_total = np.eye(2, dtype=complex)
        for seg in self.segments:
            m_k = seg.transfer_matrix(f, sound_speed=sound_speed, density=density)
            m_total = m_total @ m_k
        return m_total

    def input_impedance(
        self,
        f: float,
        sound_speed: float = 343.2,
        density: float = 1.204,
        load_impedance: Optional[Union[complex, float, str]] = None,
    ) -> complex:
        """Calculate input acoustic impedance Zin(f) = p_in / U_in (Pa*s/m^3)."""
        z_l = self.resolve_load_impedance(
            f, sound_speed=sound_speed, density=density, load_impedance=load_impedance
        )
        m = self.transfer_matrix(f, sound_speed=sound_speed, density=density)
        a, b = m[0, 0], m[0, 1]
        c, d = m[1, 0], m[1, 1]

        denom = c * z_l + d
        if abs(denom) < 1.0e-14:
            return complex(1.0e12, 0.0)
        return (a * z_l + b) / denom

    def reflection_coefficient(
        self,
        f: float,
        sound_speed: float = 343.2,
        density: float = 1.204,
        load_impedance: Optional[Union[complex, float, str]] = None,
    ) -> complex:
        """Calculate acoustic pressure reflection coefficient R(f) = (Zin - Z0) / (Zin + Z0)."""
        z0 = density * sound_speed / max(self.inlet_area, 1.0e-6)
        z_in = self.input_impedance(
            f, sound_speed=sound_speed, density=density, load_impedance=load_impedance
        )
        denom = z_in + z0
        if abs(denom) < 1.0e-14:
            return complex(1.0, 0.0)
        return (z_in - z0) / denom

    def transmission_coefficient(
        self,
        f: float,
        sound_speed: float = 343.2,
        density: float = 1.204,
        load_impedance: Optional[Union[complex, float, str]] = None,
    ) -> complex:
        """Calculate pressure transmission coefficient T(f) = p_out / p_inc."""
        z0 = density * sound_speed / max(self.inlet_area, 1.0e-6)
        z_l = self.resolve_load_impedance(
            f, sound_speed=sound_speed, density=density, load_impedance=load_impedance
        )

        m = self.transfer_matrix(f, sound_speed=sound_speed, density=density)
        a, b = m[0, 0], m[0, 1]
        c, d = m[1, 0], m[1, 1]

        denom = a + (b / z_l) + (c * z0) + (d * (z0 / z_l))
        if abs(denom) < 1.0e-14:
            return complex(0.0, 0.0)
        return 2.0 / denom

    def transmission_loss(
        self,
        f: float,
        sound_speed: float = 343.2,
        density: float = 1.204,
        load_impedance: Optional[Union[complex, float, str]] = None,
    ) -> float:
        """Calculate acoustic Transmission Loss TL(f) in dB."""
        z0 = density * sound_speed / max(self.inlet_area, 1.0e-6)
        z_l = self.resolve_load_impedance(
            f, sound_speed=sound_speed, density=density, load_impedance=load_impedance
        )

        m = self.transfer_matrix(f, sound_speed=sound_speed, density=density)
        a, b = m[0, 0], m[0, 1]
        c, d = m[1, 0], m[1, 1]

        sq_zl_z0 = np.sqrt(z_l / z0)
        sq_z0_zl = 1.0 / sq_zl_z0
        sq_prod = np.sqrt(z0 * z_l)

        term = 0.5 * (a * sq_zl_z0 + (b / sq_prod) + (c * sq_prod) + (d * sq_z0_zl))
        mag = abs(term)
        if mag < 1.0e-12:
            return 0.0
        return float(20.0 * math.log10(mag))

    def frequency_response(
        self,
        freq_array: np.ndarray,
        sound_speed: float = 343.2,
        density: float = 1.204,
        load_impedance: Optional[Union[complex, float, str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute frequency response metrics over an array of frequencies."""
        freqs = np.asarray(freq_array, dtype=float)
        n_pts = len(freqs)

        tl_db = np.zeros(n_pts, dtype=float)
        t_complex = np.zeros(n_pts, dtype=complex)
        r_complex = np.zeros(n_pts, dtype=complex)
        z_in = np.zeros(n_pts, dtype=complex)
        z_l = np.zeros(n_pts, dtype=complex)

        for i, f in enumerate(freqs):
            tl_db[i] = self.transmission_loss(
                f, sound_speed=sound_speed, density=density, load_impedance=load_impedance
            )
            t_complex[i] = self.transmission_coefficient(
                f, sound_speed=sound_speed, density=density, load_impedance=load_impedance
            )
            r_complex[i] = self.reflection_coefficient(
                f, sound_speed=sound_speed, density=density, load_impedance=load_impedance
            )
            z_in[i] = self.input_impedance(
                f, sound_speed=sound_speed, density=density, load_impedance=load_impedance
            )
            z_l[i] = self.resolve_load_impedance(
                f, sound_speed=sound_speed, density=density, load_impedance=load_impedance
            )

        transmittance = np.abs(t_complex) ** 2
        reflectance = np.abs(r_complex) ** 2

        return {
            "frequencies": freqs,
            "TL_dB": tl_db,
            "T_complex": t_complex,
            "R_complex": r_complex,
            "Z_in": z_in,
            "Z_L": z_l,
            "transmittance": transmittance,
            "reflectance": reflectance,
        }

    def filter_signal(
        self,
        time_series: np.ndarray,
        dt: float,
        sound_speed: float = 343.2,
        density: float = 1.204,
        load_impedance: Optional[Union[complex, float, str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter an input pressure time series p_GG(t) via exact TMM spectral transfer."""
        p_in = np.asarray(time_series, dtype=float)
        n_samples = len(p_in)
        if n_samples == 0:
            return np.zeros(0), np.zeros(0)

        freqs = np.fft.rfftfreq(n_samples, d=dt)
        p_fft = np.fft.rfft(p_in)

        t_arr = np.zeros(len(freqs), dtype=complex)
        zl_arr = np.zeros(len(freqs), dtype=complex)

        for i, f in enumerate(freqs):
            zl_f = self.resolve_load_impedance(
                f, sound_speed=sound_speed, density=density, load_impedance=load_impedance
            )
            zl_arr[i] = zl_f
            t_arr[i] = self.transmission_coefficient(
                f, sound_speed=sound_speed, density=density, load_impedance=zl_f
            )

        p_kc_fft = p_fft * t_arr
        p_kc = np.fft.irfft(p_kc_fft, n=n_samples)

        denom_zl = np.where(np.abs(zl_arr) < 1.0e-3, 1.0e12, zl_arr)
        u_kc_fft = p_kc_fft / denom_zl
        u_kc = np.fft.irfft(u_kc_fft, n=n_samples)

        return p_kc, u_kc

    def _invalidate_fir(self) -> None:
        """Invalidate the streaming FIR filter cache."""
        self._fir_dt = None
        self._fir_sound_speed = None
        self._fir_density = None

    def _update_fir(self, dt: float, sound_speed: float, density: float) -> None:
        """Design or update causal FIR filter kernel matching T(f) for real-time streaming."""
        if (
            self._fir_dt == dt
            and self._fir_sound_speed == sound_speed
            and self._fir_density == density
        ):
            return

        self._fir_dt = dt
        self._fir_sound_speed = sound_speed
        self._fir_density = density

        n_fir = self._fir_length
        freqs = np.fft.rfftfreq(n_fir, d=dt)
        tau = (n_fir - 1) * dt / 2.0

        h_freq = np.zeros(len(freqs), dtype=complex)
        for i, f in enumerate(freqs):
            t_val = self.transmission_coefficient(f, sound_speed=sound_speed, density=density)
            h_freq[i] = t_val * np.exp(-1j * 2.0 * math.pi * f * tau)

        h_time = np.fft.irfft(h_freq, n=n_fir)
        window = np.hamming(n_fir)
        self._fir_kernel = h_time * window

    def step(
        self,
        p_gg: float,
        dt: float,
        sound_speed: float = 343.2,
        density: float = 1.204,
        h2_fraction: float = 0.0,
        load_impedance: Optional[Union[complex, float, str]] = None,
    ) -> AntechamberState:
        """Execute a single streaming time step."""
        self._update_fir(dt, sound_speed, density)

        self._input_buffer = np.roll(self._input_buffer, 1)
        self._input_buffer[0] = p_gg

        p_out = float(np.dot(self._input_buffer, self._fir_kernel))

        z0 = density * sound_speed / max(self.inlet_area, 1.0e-6)
        zl = self.resolve_load_impedance(
            self.state.dominant_freq_hz, sound_speed=sound_speed, density=density, load_impedance=load_impedance
        )
        zl_mag = max(abs(zl), 1.0e-3)

        u_out = p_out / zl_mag
        p_refl_est = p_gg - p_out
        u_in = (p_gg - p_refl_est) / z0

        p_trans = p_out * u_out
        p_refl = (p_refl_est ** 2) / z0
        p_inc = (p_gg ** 2) / z0

        tl_438 = self.transmission_loss(
            438.0, sound_speed=sound_speed, density=density, load_impedance=load_impedance
        )

        self.state.time += dt
        self.state.p_in = p_gg
        self.state.p_out = p_out
        self.state.u_in = u_in
        self.state.u_out = u_out
        self.state.p_trans = p_trans
        self.state.p_refl = p_refl
        self.state.p_inc = p_inc
        self.state.gas_sound_speed = sound_speed
        self.state.gas_density = density
        self.state.h2_fraction = h2_fraction
        self.state.transmission_loss_db_438 = tl_438

        return self.state

    def reset(self) -> None:
        """Reset internal filter state and buffers."""
        self.state = AntechamberState()
        self._input_buffer.fill(0.0)
        self._invalidate_fir()
