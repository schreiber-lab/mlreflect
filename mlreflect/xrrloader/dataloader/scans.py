from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from ..footprint import apply_footprint_correction
from ..footprint import normalize_to_first, normalize_to_max


class ReflectivityScan:
    """Store information about a loaded scan and perform a basic footprint correction.

    :param scan_number: Scan number for identification (usually derived from SPEC scan number).
    :param scattering_angle: Full scattering angle (2theta) in degrees for each intensity value.
    :param intensity: Continuous raw intensity values (including attenuator correction).
    :param wavelength: Wavelength in units of angstroms.
    :param beam_width: Beam width for a rectangular approximation or FWHM of a Gaussian approximation in units of mm.
    :param sample_length: Sample length in beam direction in units of mm.
    :param beam_shape: Beam shape approximation. Either `'box'` or `'gauss'` (default).
    :param normalize_to: To what intensity value the curve is normalized to after footprint correction. Either
    `'first'` or `'max'` (default).
    :param timestamp: Timestamp of when the scan was taken in the SPEC format (e.g. "Wed Apr 28 09:32:45 2010").
    """

    def __init__(self, scan_number: int, scattering_angle: ndarray, intensity: ndarray, wavelength: float,
                 beam_width: float, sample_length: float, beam_shape: str = 'gauss', normalize_to: str = 'max',
                 timestamp: str = None):
        self.scan_number = scan_number
        self.timestamp = timestamp

        self.scattering_angle = self._replace_zeros(scattering_angle)
        self.raw_intensity = intensity
        self.wavelength = wavelength

        self.footprint_params = {
            'beam_width': beam_width,
            'sample_length': sample_length,
            'beam_shape': beam_shape
        }

        self.normalize_to = normalize_to

    def __len__(self):
        return len(self.scattering_angle)

    @property
    def q(self):
        return 4 * np.pi / self.wavelength * np.sin(self.scattering_angle / 2 * np.pi / 180)

    @property
    def corrected_intensity(self):
        try:
            intensity = self._correct_footprint(self.raw_intensity, self.scattering_angle)
        except ValueError:
            intensity = self.raw_intensity
        intensity = self._normalize(intensity)
        return intensity

    def get_raw_intensity_range(self, q_min: float = None, q_max: float = None):
        """Get raw intensity between `q_min` and `q_max`."""
        q_min_idx = self._get_closest_index(q_min, self.q)
        q_max_idx = self._get_closest_index(q_max, self.q)
        return self.raw_intensity[q_min_idx:q_max_idx]

    def get_corrected_intensity_range(self, q_min: float = None, q_max: float = None):
        """Get corrected intensity between `q_min` and `q_max`."""
        q_min_idx = self._get_closest_index(q_min, self.q)
        q_max_idx = self._get_closest_index(q_max, self.q)
        return self.corrected_intensity[q_min_idx:q_max_idx]

    def get_q_range(self, q_min: float = None, q_max: float = None):
        """Get q values between `q_min` and `q_max`."""
        q_min_idx = self._get_closest_index(q_min, self.q)
        q_max_idx = self._get_closest_index(q_max, self.q)
        return self.q[q_min_idx:q_max_idx]

    def get_interpolated_intensity(self, new_q: ndarray):
        """Interpolate log10 values of the corrected intensity to new values `new_q`."""
        return 10 ** (np.interp(new_q, self.q, np.log10(self.corrected_intensity)))

    def plot_raw_intensity(self, q_min: float = None, q_max: float = None):
        """Plot raw intensity within given q range."""
        plt.semilogy(self.get_q_range(q_min, q_max), self.get_raw_intensity_range(q_min, q_max), '.')
        plt.xlabel('Scattering vector [1/Å]')
        plt.ylabel('Raw intensity')
        plt.show()

    def plot_corrected_intensity(self, q_min: float = None, q_max: float = None):
        """Plot corrected intensity within given q range."""
        plt.semilogy(self.get_q_range(q_min, q_max), self.get_corrected_intensity_range(q_min, q_max), '.')
        plt.xlabel('Scattering vector [1/Å]')
        plt.ylabel('Corrected intensity')
        plt.show()

    def _correct_footprint(self, intensity, angle):
        if None in self.footprint_params.values():
            raise ValueError('one or multiple footprint parameters are None')
        else:
            return apply_footprint_correction(intensity, angle, **self.footprint_params)

    def _normalize(self, intensity: ndarray):
        if self.normalize_to == 'max':
            return normalize_to_max(intensity)
        elif self.normalize_to == 'first':
            return normalize_to_first(intensity)
        elif self.normalize_to is None:
            return intensity
        else:
            raise ValueError('invalid normalization')

    @staticmethod
    def _get_closest_index(value, array):
        if value is None:
            return None
        else:
            return np.argmin(abs(array - value))

    @staticmethod
    def _replace_zeros(array, value=1e-6):
        copy = array.copy()
        copy[array == 0] = value
        return copy


class ScanSeries(list):
    """Store a series of Scan objects in a list-like object."""

    @property
    def stats(self):
        length = np.empty_like(self, dtype=int)
        q_min = np.empty_like(self, dtype=float)
        q_max = np.empty_like(self, dtype=float)
        for i, scan in enumerate(self):
            length[i] = len(scan)
            q_min[i] = np.min(scan.q)
            q_max[i] = np.max(scan.q)
        return {'length': length, 'q_min': q_min, 'q_max': q_max}

    def append(self, scan: ReflectivityScan):
        if isinstance(scan, ReflectivityScan):
            super().append(scan)
        else:
            raise TypeError('Series can only contain ReflectivityScan objects')

    def to_array(self):
        """Convert corrected intensity to numpy array with each scan as a row."""
        return np.array([scan.corrected_intensity for scan in self])

    def plot_series(self, scan_range: Iterable = None, q_min: float = None, q_max: float = None, legend: bool = True):
        """Plot all or a subset of scans of the series within the given q range."""
        if not self:
            return
        if scan_range is None:
            for scan in self:
                plt.semilogy(scan.get_q_range(q_min, q_max), scan.get_corrected_intensity_range(q_min, q_max),
                             label=scan.scan_number)
        else:
            for scan_idx in scan_range:
                plt.semilogy(self[scan_idx].get_q_range(q_min, q_max),
                             self[scan_idx].get_corrected_intensity_range(q_min, q_max),
                             label=self[scan_idx].scan_number)
        plt.xlabel('Scattering vector [1/Å]')
        plt.ylabel('Corrected intensity')
        if legend:
            plt.legend()
        plt.show()
