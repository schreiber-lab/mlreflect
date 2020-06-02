import numpy as np
import functools

from numpy import ndarray
from refl1d.reflectivity import convolve as refl1d_convolve


def iterate_over_curves(func):
    @functools.wraps(func)
    def wrapper(reflectivity_curves, *args, **kwargs):
        number_of_dimensions = len(reflectivity_curves.shape)
        if number_of_dimensions not in (1, 2):
            raise ValueError('number of dimensions mus be 1 or 2')
        if number_of_dimensions == 1:
            return func(reflectivity_curves, *args, **kwargs)
        else:
            return np.array([func(reflectivity_curve, *args, **kwargs) for reflectivity_curve in reflectivity_curves])

    return wrapper


@iterate_over_curves
def apply_shot_noise(reflectivity_curves: ndarray, shot_noise_spread: float) -> ndarray:
    """Returns reflectivity curves with applied shot noise based on `shot_noise_spread`.

        Args:
            reflectivity_curves: Array of normalized reflectivity curves
            shot_noise_spread: Scaling factor c for the standard deviation sqrt(I*c) of the shot noise around
                the intensity of simulated reflectivity curves. Since the intensity is normalized, this is
                equivalent to setting the direct beam intensity I_0 = 1/c.

        Returns:
            noisy_reflectivity
    """
    noisy_reflectivity = np.clip(
        np.random.normal(reflectivity_curves, np.sqrt(reflectivity_curves * shot_noise_spread)), 1e-8, None)
    return noisy_reflectivity


def generate_background(number_of_curves: int, number_of_q_values: int, background_base_level: float,
                        background_spread: float) -> ndarray:
    """Returns a background with a normal distribution that can be added to a reflectivity curve.

        Args:
            number_of_curves: Number of curves for which a background is generated
            number_of_q_values: Length of the generated array (should be same length as reflectivity curve)
            background_base_level: Mean value of the normal distribution
            background_spread: Standard deviation of the normal distribution

        Returns:
            background with dimensions (number_of_curves, number_of_q_values)
    """

    return np.random.normal(background_base_level, background_spread, (number_of_curves, number_of_q_values))


@iterate_over_curves
def apply_gaussian_convolution(reflectivity_curves: ndarray, q_before: ndarray, q_after: ndarray,
                               width: ndarray) -> ndarray:
    """Returns convolved reflectivity curves at q-values given in `q_before`.

    Args:
        reflectivity_curves: Array of normalized reflectivity curves
        q_before: q-values that correspond to the given reflectivity curve
        q_after: q-values for which the convolved curve is evaluated (shorter than `q_before`)
        width: width of the gaussian convolution at each q-value

    Returns:
        convolved reflectivity curves
    """
    if width.all() == 0:
        return reflectivity_curves
    else:
        return refl1d_convolve(q_before, reflectivity_curves, q_after, width)
