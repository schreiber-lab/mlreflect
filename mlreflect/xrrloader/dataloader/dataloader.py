from typing import Iterator

import numpy as np

from .scans import ReflectivityScan, ScanSeries
from ..parser import SpecParser
from ..preprocessing import apply_attenuation, correct_discontinuities


class SpecLoader:
    """
    Read SPEC file and extract reflectivity scans.

    :param file_path: Full file path of the SPEC file.
    :param angle_columns: List of column names of the angles in the SPEC file. If multiple are given, their absolute
    values are summed up to form the full scattering angle 2theta.
    :param intensity_column: Column name of the reflected intensity. Attenuator and monitor correction must already
    be performed.
    :param wavelength: Photon wavelength in units of angstroms.
    :param beam_width: Beam width for a rectangular approximation or FWHM of a Gaussian approximation in units of mm.
    :param sample_length: Sample length in beam direction in units of mm.
    :param beam_shape: Beam shape approximation. Either `'box'` or `'gauss'` (default).
    :param normalize_to: To what intensity value the curve is normalized to after footprint correction. Either
    `'first'` or `'max'` (default).
    :param attenuator_column: Optional column name of the attenuator values for intensity correction.
    :param division_column: Optional name of a column that the intensity is divided by after attenuator correction.
    """

    def __init__(self, file_path: str, angle_columns: list, intensity_column: str, wavelength: float, beam_width: float,
                 sample_length: float, beam_shape='gauss', normalize_to='max', attenuator_column=None,
                 division_column=None):
        self.parser = self._load_parser(file_path)

        self.angle_columns = angle_columns
        self.intensity_column = intensity_column

        self.attenuator_column = attenuator_column
        self.division_column = division_column

        self.wavelength = wavelength

        self.footprint_params = {
            'beam_width': beam_width,
            'sample_length': sample_length,
            'beam_shape': beam_shape
        }

        self.normalize_to = normalize_to

    @staticmethod
    def _load_parser(file_path: str):
        return SpecParser(file_path)

    def load_scan(self, scan_number: int, trim_front: int = None, trim_back: int = None) -> ReflectivityScan:
        """Read reflectivity scan from SPEC file, trim it and return it in a ReflectivityScan object."""
        scan = self.parser.extract_scan(scan_number)
        scan_time = self.parser.scan_info[str(scan_number)]['time']
        angles = np.array([np.array(abs(scan[column_name])) for column_name in self.angle_columns])
        scattering_angle = np.sum(np.atleast_2d(angles), axis=0)
        intensity = np.array(scan[self.intensity_column])

        if self.attenuator_column is not None:
            intensity = apply_attenuation(intensity, scan[self.attenuator_column])
            intensity = correct_discontinuities(intensity, scattering_angle)
        if self.division_column is not None:
            intensity /= np.array(scan[self.division_column])

        if trim_back is not None:
            scattering_angle, intensity = self._trim_back(scattering_angle, intensity, trim_back)
        if trim_front is not None:
            scattering_angle, intensity = self._trim_front(scattering_angle, intensity, trim_front)

        return ReflectivityScan(scan_number, scattering_angle, intensity, self.wavelength, **self.footprint_params,
                                normalize_to=self.normalize_to,
                                timestamp=scan_time)

    def load_scans(self, scan_numbers: Iterator, trim_front: int = None, trim_back: int = None) -> ScanSeries:
        """Read several reflectivity scans from SPEC file, trim them and return them in a ScanSeries object."""
        scans = ScanSeries()
        for scan_number in scan_numbers:
            scans.append(self.load_scan(scan_number, trim_front, trim_back))
        return scans

    @staticmethod
    def _trim_back(scattering_angle, intensity, cutoff_back):
        return scattering_angle[:-cutoff_back], intensity[:-cutoff_back]

    @staticmethod
    def _trim_front(scattering_angle, intensity, cutoff_front):
        return scattering_angle[cutoff_front:], intensity[cutoff_front:]
