import numpy as np
from numpy import ndarray


def angle_to_q(scattering_angle: ndarray, wavelength: float):
    """Conversion from full scattering angle (degrees) to scattering vector (inverse angstroms)"""
    return 4 * np.pi / wavelength * np.sin(scattering_angle / 2 * np.pi / 180)


def q_to_angle(q: ndarray, wavelength: float):
    """Conversion from scattering vector (inverse angstroms) to full scattering angle (degrees)"""
    return 2 * np.arcsin(q * wavelength / (4 * np.pi)) / np.pi * 180


def energy_to_wavelength(energy: float):
    """Conversion from photon energy (eV) to photon wavelength (angstroms)"""
    return 1.2398 / energy * 1e4


def wavelength_to_energy(wavelength: float):
    """Conversion from photon wavelength (angstroms) to photon energy (eV)"""
    return 1.2398 / wavelength * 1e4
