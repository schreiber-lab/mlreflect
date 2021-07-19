from typing import Iterator

import numpy as np

from .scans import ReflectivityScan, ScanSeries


class ReflectivityLoader:
    """Abstract class for loading reflectivity scans from various data sources."""

    def __init__(self, file_path: str, beam_width: float, sample_length: float, beam_shape='gauss', normalize_to='max'):
        self.parser = self._load_parser(file_path)

        self.footprint_params = {
            'beam_width': beam_width,
            'sample_length': sample_length,
            'beam_shape': beam_shape
        }

        self.normalize_to = normalize_to

    def load_scan(self, scan_number: int, trim_front: int = None, trim_back: int = None) -> ReflectivityScan:
        raise NotImplementedError

    def load_scans(self, scan_numbers: Iterator, trim_front: int = None, trim_back: int = None) -> ScanSeries:
        raise NotImplementedError

    @staticmethod
    def apply_attenuation(intensity, attenuation):
        intensity = intensity.copy()
        intensity = intensity * attenuation
        return intensity

    @staticmethod
    def correct_discontinuities(intensity, scattering_angle):
        intensity = intensity.copy()
        diff_angle = np.diff(scattering_angle)
        for i in range(len(diff_angle)):
            if diff_angle[i] == 0:
                factor = intensity[i] / intensity[i + 1]
                intensity[(i + 1):] *= factor
        return intensity

    @staticmethod
    def apply_normalization(intensity, normalization):
        intensity = intensity.copy()
        intensity /= np.array(normalization)
        return intensity

    @staticmethod
    def _load_parser(file_path: str):
        raise NotImplementedError

    @staticmethod
    def _trim_back(scattering_angle, intensity, cutoff_back):
        return scattering_angle[:-cutoff_back], intensity[:-cutoff_back]

    @staticmethod
    def _trim_front(scattering_angle, intensity, cutoff_front):
        return scattering_angle[cutoff_front:], intensity[cutoff_front:]
