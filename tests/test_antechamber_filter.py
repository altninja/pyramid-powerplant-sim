"""Comprehensive test suite for the Antechamber Acoustic Filter and TMM Module."""

import math
import numpy as np
import pytest

from engine.config import SimulationConfig
from engine.geometry import PyramidGeometry
from engine.physics.antechamber_filter import (
    AntechamberFilter,
    AntechamberState,
    FilterSegment,
)


def test_transfer_matrix_identity_zero_length():
    """Verify zero-length or zero-area segment yields identity matrix."""
    seg_zero = FilterSegment(name="zero_len", length=0.0, area=1.1655)
    f_test = 438.0
    c_sound = 343.2
    rho = 1.204

    m_zero = seg_zero.transfer_matrix(f_test, sound_speed=c_sound, density=rho)
    assert np.allclose(m_zero, np.eye(2, dtype=complex))

    seg_empty_area = FilterSegment(name="zero_area", length=1.0, area=0.0)
    m_empty = seg_empty_area.transfer_matrix(f_test, sound_speed=c_sound, density=rho)
    assert np.allclose(m_empty, np.eye(2, dtype=complex))

    filt = AntechamberFilter()
    filt.clear_segments()
    assert len(filt.segments) == 0
    m_tot_empty = filt.transfer_matrix(f_test, sound_speed=c_sound, density=rho)
    assert np.allclose(m_tot_empty, np.eye(2, dtype=complex))
    assert abs(filt.transmission_loss(f_test, sound_speed=c_sound, density=rho)) < 1.0e-6
    assert np.isclose(filt.transmission_coefficient(f_test, sound_speed=c_sound, density=rho), 1.0)


def test_transfer_matrix_unitarity_reciprocity():
    """Verify all acoustic segments maintain determinant of 1 (reciprocity)."""
    c_sound = 343.2
    rho = 1.204
    test_freqs = [7.83, 14.3, 30.0, 100.0, 438.0, 876.0, 1314.0]

    filt = AntechamberFilter()
    for f in test_freqs:
        for seg in filt.segments:
            m_seg = seg.transfer_matrix(f, sound_speed=c_sound, density=rho)
            det_seg = m_seg[0, 0] * m_seg[1, 1] - m_seg[0, 1] * m_seg[1, 0]
            assert np.isclose(det_seg, 1.0, atol=1.0e-5)

        m_tot = filt.transfer_matrix(f, sound_speed=c_sound, density=rho)
        det_tot = m_tot[0, 0] * m_tot[1, 1] - m_tot[0, 1] * m_tot[1, 0]
        assert np.isclose(det_tot, 1.0, atol=1.0e-5)


def test_passband_transmission_at_438hz_and_harmonics():
    """Verify passband transmission loss at 438 Hz is low (TL < 2.0 dB) and harmonics pass cleanly."""
    filt = AntechamberFilter()
    c_air = 343.2
    rho_air = 1.204

    # King's Chamber reactive termination at 438 Hz (fundamental)
    tl_438 = filt.transmission_loss(438.0, sound_speed=c_air, density=rho_air)
    t_438 = filt.transmission_coefficient(438.0, sound_speed=c_air, density=rho_air)

    assert tl_438 < 1.0
    assert np.abs(t_438) > 1.0  # Cavity resonance pressure amplification

    tl_876 = filt.transmission_loss(876.0, sound_speed=c_air, density=rho_air)
    assert tl_876 < 3.5

    tl_1314 = filt.transmission_loss(1314.0, sound_speed=c_air, density=rho_air)
    assert tl_1314 < 1.0

    # Verify anechoic baseline passband
    tl_438_anechoic = filt.transmission_loss(438.0, sound_speed=c_air, density=rho_air, load_impedance="anechoic")
    t_438_anechoic = filt.transmission_coefficient(438.0, sound_speed=c_air, density=rho_air, load_impedance="anechoic")
    assert tl_438_anechoic < 1.0
    assert np.abs(t_438_anechoic) > 0.90


def test_stopband_infrasonic_rejection():
    """Verify strong attenuation for low infrasonic bedrock frequencies (TL > 25.0 dB for seismic modes)."""
    filt = AntechamberFilter()
    c_air = 343.2
    rho_air = 1.204

    infrasonic_freqs = [7.83, 14.3, 20.8, 27.3]
    for f_infra in infrasonic_freqs:
        tl = filt.transmission_loss(f_infra, sound_speed=c_air, density=rho_air)
        t_coeff = filt.transmission_coefficient(f_infra, sound_speed=c_air, density=rho_air)
        assert tl > 25.0
        assert np.abs(t_coeff) < 0.05


def test_kings_chamber_reactive_cavity_load_and_matching():
    """Verify King's Chamber modal reactive termination impedance formula and matching at 438 Hz."""
    filt = AntechamberFilter()
    c_sound = 343.2
    rho = 1.204
    s_portal = 1.05 * 1.11
    l_kc = 10.47
    z0 = rho * c_sound / s_portal

    # 1. DC / low-frequency capacitive compliance (lossless limit)
    f_low = 1.0e-4
    zl_low = filt.compute_kings_chamber_impedance(
        f_low, sound_speed=c_sound, density=rho, attenuation_coeff=0.0
    )
    c_a_theor = s_portal * l_kc / (rho * (c_sound ** 2))
    zl_theor_low = -1j / (2.0 * np.pi * f_low * c_a_theor)
    assert np.isclose(zl_low.imag, zl_theor_low.imag, rtol=1.0e-3)

    # 2. Formula verification at 438 Hz: Z_L = -j * Z0 * cot(k * L_KC)
    k_c = (2.0 * np.pi * 438.0 / c_sound) - 1j * 0.002
    zl_theor_438 = -1j * z0 * (np.cos(k_c * l_kc) / np.sin(k_c * l_kc))
    zl_438 = filt.compute_kings_chamber_impedance(438.0, sound_speed=c_sound, density=rho)
    assert np.isclose(zl_438, zl_theor_438, rtol=1.0e-4)

    # 3. Transmission coefficient magnitude and impedance coupling
    t_kc = filt.transmission_coefficient(438.0, sound_speed=c_sound, density=rho, load_impedance="reactive")
    assert np.abs(t_kc) > 1.10
    tl_kc = filt.transmission_loss(438.0, sound_speed=c_sound, density=rho, load_impedance="reactive")
    assert tl_kc < 1.0


def test_frequency_response_dictionary_output():
    """Verify frequency_response returns complete metrics with correct keys and shape."""
    filt = AntechamberFilter()
    freqs = np.array([10.0, 50.0, 100.0, 438.0, 876.0])
    resp = filt.frequency_response(freqs)

    assert "frequencies" in resp
    assert "TL_dB" in resp
    assert "T_complex" in resp
    assert "R_complex" in resp
    assert "Z_in" in resp
    assert "Z_L" in resp
    assert "transmittance" in resp
    assert "reflectance" in resp

    assert len(resp["TL_dB"]) == len(freqs)
    assert len(resp["Z_L"]) == len(freqs)
    assert len(resp["transmittance"]) == len(freqs)
    assert np.all(resp["transmittance"] >= 0.0)
    assert np.all(resp["reflectance"] >= 0.0)

    # Anechoic baseline verifies non-negative passive TL
    resp_anechoic = filt.frequency_response(freqs, load_impedance="anechoic")
    assert np.all(resp_anechoic["TL_dB"] >= 0.0)
