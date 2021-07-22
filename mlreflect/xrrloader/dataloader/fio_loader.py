from typing import Iterator

import numpy as np
from numpy import ndarray

from .reflectivity_loader import ReflectivityLoader
from .scans import ReflectivityScan, ScanSeries
from .exceptions import NotReflectivityScanError
from ..p08tools import Scan as FioScan
from ..p08tools import ScanAnalyzer
from ..parser import FioParser


class FioLoader(ReflectivityLoader):
    """Read .fio files and associated detector images and extract reflectivity scans.

    Args:
        file_stem: The name of the experiment including the preceding folder structure. E.g. if you're experiment
            name is ``'my_data'`` and in the folder ``'user/data/'``, the file stem would be ``'user/data/my_data'``. This
            will look for all scans in the folder ``'user/data/'`` that begin with ``'my_data'``.
        two_theta_counter: Name of the counter that contains the full scattering angle (default: ``'tt'``).
        default_roi_name: Counter name of the default region of interest that is extracted as reflectivity (default:
        ``'p100k'``).
        beam_width: Beam width for a rectangular approximation or FWHM of a Gaussian approximation in units of mm.
        sample_length: Sample length in beam direction in units of mm.
        beam_shape: Beam shape approximation. Either ``'box'`` or ``'gauss'`` (default).
        normalize_to: To what intensity value the curve is normalized to after footprint correction. Either
            ``'first'`` or ``'max'`` (default).
        attenuator_counter: Optional column name of the attenuator values for intensity correction.
        division_counter: Optional name of a column that the intensity is divided by after attenuator correction.
    """

    def __init__(self, file_stem: str, beam_width: float, sample_length: float, two_theta_counter='tt',
                 default_roi_name='p100k_roi1', beam_shape='gauss', normalize_to='max',
                 attenuator_counter='atten_position', division_counter=None):
        super().__init__(file_path=file_stem, beam_width=beam_width, sample_length=sample_length,
                         beam_shape=beam_shape, normalize_to=normalize_to)

        self.two_theta_counter = two_theta_counter
        self.attenuator_counter = attenuator_counter
        self.division_counter = division_counter
        self.default_roi_name = default_roi_name

    def load_scan(self, scan_number: int, trim_front: int = None, trim_back: int = None,
                  roi: list = None, detector_name='p100k') -> ReflectivityScan:
        """Read .fio files and associated detector images, trim it and return it in a :class:`ReflectivityScan` object.

            Args:
                scan_number: Number of the scan to extract.
                trim_front: Number of measurement points that are cut off at the beginning of the scan (e.g. to
                    remove the direct beam).
                trim_back: Number of measurement points that are cut off at the end of the scan.
                roi: Alternative region of interest in the raw detector image that will be converted to a
                    reflectivity curve. The roi specifications must be a list of integers that specify the pixel
                    boundaries in the format ``[left, bottom, right, top]``, e.g. ``roi=[241, 106, 247, 109]``.
                    This will override the default roi counter.
                detector_name: Name of the detector from which the roi is extracted.

            Returns: :class:`ReflectivityScan`
        """
        scan = self.parser.extract_scan(scan_number)
        if not scan.is_theta2theta_scan:
            raise NotReflectivityScanError(
                f'must be a theta-2theta/reflectivity scan (scan {scan_number} is "{scan.scan_cmd}")')

        wavelength = 12380 / scan.motor_positions['energyfmb']
        scattering_angle = np.array(scan.data[self.two_theta_counter])

        if roi is None:
            intensity = scan.data[self.default_roi_name]
        else:
            intensity = self._redraw_region_of_interest(scan, roi, detector_name)

        if self.attenuator_counter is not None:
            intensity = self.apply_attenuation(intensity, np.array(scan.data[self.attenuator_counter]))
            intensity = self.correct_discontinuities(intensity, scattering_angle)

        if self.division_counter is not None:
            intensity /= np.array(scan[self.division_counter])

        if trim_back is not None:
            scattering_angle, intensity = self._trim_back(scattering_angle, intensity, trim_back)
        if trim_front is not None:
            scattering_angle, intensity = self._trim_front(scattering_angle, intensity, trim_front)

        return ReflectivityScan(scan_number=scan_number, scattering_angle=scattering_angle, intensity=intensity,
                                wavelength=wavelength, **self.footprint_params, normalize_to=self.normalize_to)

    def load_scans(self, scan_numbers: Iterator, trim_front: int = None, trim_back: int = None,
                   roi: list = None) -> ScanSeries:
        """Read several reflectivity scans and return them in a ``ScanSeries`` object."""
        scans = ScanSeries()
        for scan_number in scan_numbers:
            try:
                scans.append(self.load_scan(scan_number, trim_front, trim_back, roi))
            except NotReflectivityScanError:
                pass
        return scans

    @staticmethod
    def _redraw_region_of_interest(scan: FioScan, roi_coordinates: list, detector_name='p100k') -> ndarray:
        roi_name = 'custom_roi'
        analyzer = ScanAnalyzer()
        return analyzer.extract_rois(scan, {roi_name: roi_coordinates})[detector_name][roi_name]

    @staticmethod
    def _load_parser(file_stem: str):
        return FioParser(file_stem)
