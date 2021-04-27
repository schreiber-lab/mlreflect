import numpy as np
from numpy import ndarray


class FootprintRescaler:
    """Returns reflectivity curves with rescaled ("incorrect") footprint for standard (specular) XRR geometries
    Args:
        reflectivity: Reflected intensity values of one curve (1D) or several curves (2D).
        true_ratio: Assumed "true" ratio of beam width to sample length used to produce the fictional footprint on
            the curves. E.g. for beam width of 200 microns and sample length of 10 mm ``true_ratio = 0.02``.
        errors: List of error factors that the ``true_ratio`` can be multiplied by to produce the new ("incorrect")
            footprint corrections.
        q: ndarray of q-values corresponding the provided intensity matrix. If this value is provided, the wavelength
            has to be specified as well.
        wavelength: Fictional wavelength in Angstroms that is used to calculate the scattering angle.
        theta: ndarray angle values corresponding to the provided intensity matrix. If this is provided, ``q`` and
            ``wavelength`` do not need to be provided.

    """

    def __init__(self, reflectivity: ndarray, true_ratio: float, errors: list, q: ndarray = None, wavelength:
    float = None, theta: ndarray = None):
        self.reflectivity = np.atleast_2d(reflectivity)
        self.true_ratio = true_ratio
        self.errors = np.atleast_1d(errors)
        if q is None and theta is None:
            raise ValueError('q or theta values must be provided')
        elif q is not None and theta is not None:
            raise ValueError('cannot provide both q and theta together')
        elif q is not None:
            if wavelength is None:
                raise ValueError('wavelength must be provided together with q values')
            self.theta = self.q_to_angle(q, wavelength)
        elif theta is not None:
            self.theta = theta

    @property
    def rescaled_reflectivity(self):
        num_curves = self.reflectivity.shape[0]
        num_points = self.reflectivity.shape[1]
        rescaled_reflectivity = np.empty((num_curves, num_points))
        new_ratios = np.empty(num_curves)
        for i in range(num_curves):
            new_ratio = self.true_ratio * np.random.choice(self.errors)
            new_ratios[i] = new_ratio
            rescaled_reflectivity[i] = self._swap_footprints(self.reflectivity[i], self.theta, self.true_ratio,
                                                             new_ratio)

        return {'rescaled_reflectivity': rescaled_reflectivity, 'ratios': new_ratios}

    def _swap_footprints(self, reflectivity, theta, true_ratio, new_ratio):
        footprint_reflectivity = self.apply_footprint(reflectivity, theta, true_ratio)
        return self.normalize_to_max(self.correct_footprint(footprint_reflectivity, theta, new_ratio), reflectivity)

    @staticmethod
    def apply_footprint(intensity: ndarray, scattering_angle: ndarray, ratio: float) -> ndarray:
        max_angle = 2 * np.arcsin(ratio) / np.pi * 180
        corrected_intensity = intensity.copy()
        below_max_angle_index = scattering_angle < max_angle
        corrected_intensity[below_max_angle_index] = intensity[below_max_angle_index] * np.sin(
            scattering_angle[below_max_angle_index] / 2 * np.pi / 180) / ratio

        return corrected_intensity

    @staticmethod
    def correct_footprint(intensity: ndarray, scattering_angle: ndarray, ratio: float) -> ndarray:
        max_angle = 2 * np.arcsin(ratio) / np.pi * 180
        corrected_intensity = intensity.copy()
        below_max_angle_index = scattering_angle < max_angle
        corrected_intensity[below_max_angle_index] = intensity[below_max_angle_index] / np.sin(
            scattering_angle[below_max_angle_index] / 2 * np.pi / 180) * ratio

        return corrected_intensity

    @staticmethod
    def normalize_to_max(rescaled_intensity: ndarray, original_intensity: ndarray):
        return rescaled_intensity / np.max(rescaled_intensity) * np.max(original_intensity)

    @staticmethod
    def normalize_to_first(rescaled_intensity: ndarray, original_intensity: ndarray):
        return rescaled_intensity / rescaled_intensity[0] * original_intensity[0]

    @staticmethod
    def angle_to_q(scattering_angle, wavelength):
        return 4 * np.pi / wavelength * np.sin(scattering_angle / 2 * np.pi / 180)

    @staticmethod
    def q_to_angle(q, wavelength):
        return 2 * np.arcsin(q * wavelength / (4 * np.pi)) / np.pi * 180
