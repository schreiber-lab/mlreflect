import numpy as np
from numpy import ndarray
from scipy.special import erf


def apply_footprint_correction(intensity: ndarray, scattering_angle: ndarray, beam_width: float, sample_length:
float, beam_shape: str = 'gauss'):
    factors = _get_factors_by_beam_shape(scattering_angle, beam_width, sample_length, beam_shape)
    return intensity.copy() * factors


def remove_footprint_correction(intensity: ndarray, scattering_angle: ndarray, beam_width: float, sample_length:
float, beam_shape: str = 'gauss'):
    factors = _get_factors_by_beam_shape(scattering_angle, beam_width, sample_length, beam_shape)
    return intensity.copy() / factors


def _get_factors_by_beam_shape(scattering_angle: ndarray, beam_width: float, sample_length: float, beam_shape: str):
    if beam_shape == 'gauss':
        return gaussian_factors(scattering_angle, beam_width, sample_length)
    elif beam_shape == 'box':
        return box_factors(scattering_angle, beam_width, sample_length)
    else:
        raise ValueError('invalid beam shape')


def box_factors(scattering_angle, beam_width, sample_length):
    max_angle = 2 * np.arcsin(beam_width / sample_length) / np.pi * 180
    ratios = beam_footprint_ratio(scattering_angle, beam_width, sample_length)
    ones = np.ones_like(scattering_angle)
    return np.where(scattering_angle < max_angle, ones * ratios, ones)


def gaussian_factors(scattering_angle, beam_width, sample_length):
    ratio = beam_footprint_ratio(scattering_angle, beam_width, sample_length)
    return 1 / erf(np.sqrt(np.log(2)) / ratio)


def beam_footprint_ratio(scattering_angle, beam_width, sample_length):
    return beam_width / sample_length / np.sin(scattering_angle / 2 * np.pi / 180)
