from typing import Iterator

import numpy as np

from .exceptions import NotReflectivityScanError
from .reflectivity_loader import ReflectivityLoader
from .scans import ReflectivityScan, ScanSeries
from ..parser import SpecParser


class SpecLoader(ReflectivityLoader):
    """Read SPEC file and extract reflectivity scans.

    Args:
        file_path: Full file path of the SPEC file.
        angle_columns: List of column names of the angles in the SPEC file. If multiple are given, their absolute
            values are summed up to form the full scattering angle 2theta.
        intensity_column: Column name of the reflected intensity. Attenuator and monitor correction must already
            be performed.
        wavelength: Photon wavelength in units of angstroms.
        beam_width: Beam width for a rectangular approximation or FWHM of a Gaussian approximation in units of mm.
        sample_length: Sample length in beam direction in units of mm.
        beam_shape: Beam shape approximation. Either ``'box'`` or ``'gauss'`` (default).
        normalize_to: To what intensity value the curve is normalized to after footprint correction. Either
            ``'first'`` or ``'max'`` (default).
        attenuator_column: Optional column name of the attenuator values for intensity correction.
        division_column: Optional name of a column that the intensity is divided by after attenuator correction.
    """

    def __init__(self, file_path: str, angle_columns: list, intensity_column: str, wavelength: float, beam_width: float,
                 sample_length: float, beam_shape='gauss', normalize_to='max', attenuator_column=None,
                 division_column=None):
        super().__init__(file_path=file_path, beam_width=beam_width, sample_length=sample_length,
                         beam_shape=beam_shape, normalize_to=normalize_to)

        self.angle_columns = angle_columns
        self.intensity_column = intensity_column

        self.attenuator_column = attenuator_column
        self.division_column = division_column

        self.wavelength = wavelength

    @staticmethod
    def _load_parser(file_path: str):
        return SpecParser(file_path)

    def load_scan(self, scan_number: int, trim_front: int = None, trim_back: int = None) -> ReflectivityScan:
        """Read reflectivity scan from SPEC file, trim it and return it in a :class:`ReflectivityScan` object."""
        try:
            scan = self.parser.extract_scan(scan_number)
        except (KeyError, ValueError):
            raise NotReflectivityScanError(f'scan {scan_number} could not be found in {self.parser.file_path}')

        if not self._contains_counters(scan.columns):
            raise NotReflectivityScanError(
                f'scan counters must contain motor names {self.angle_columns} '
                f'(scan {scan_number} is "{self.parser.scan_info[scan_number]["spec_command"]}")')

        scan_time = self.parser.scan_info[scan_number]['time']
        angles = np.array([np.array(abs(scan[column_name])) for column_name in self.angle_columns])
        scattering_angle = np.sum(np.atleast_2d(angles), axis=0)
        intensity = np.array(scan[self.intensity_column])

        if self.attenuator_column is not None:
            intensity = self.apply_attenuation(intensity, scan[self.attenuator_column])
            intensity = self.correct_discontinuities(intensity, scattering_angle)
        if self.division_column is not None:
            intensity /= np.array(scan[self.division_column])

        if trim_back is not None:
            scattering_angle, intensity = self._trim_back(scattering_angle, intensity, trim_back)
        if trim_front is not None:
            scattering_angle, intensity = self._trim_front(scattering_angle, intensity, trim_front)

        return ReflectivityScan(scan_number, np.array(scattering_angle), np.array(intensity), self.wavelength,
                                **self.footprint_params,
                                normalize_to=self.normalize_to,
                                timestamp=scan_time)

    def load_scans(self, scan_numbers: Iterator, trim_front: int = None, trim_back: int = None) -> ScanSeries:
        """Read several reflectivity scans from SPEC file, trim them and return them in a :class:`ScanSeries` object."""
        scans = ScanSeries()
        for scan_number in scan_numbers:
            try:
                scans.append(self.load_scan(scan_number, trim_front, trim_back))
            except NotReflectivityScanError:
                pass
        return scans

    def _contains_counters(self, counter_list):
        return all([angle_name in counter_list for angle_name in self.angle_columns])
