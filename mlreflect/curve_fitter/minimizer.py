import numpy as np
from scipy.optimize import curve_fit

from ..data_generation import interp_reflectivity, ReflectivityGenerator


def q_shift_variants(q_values_prediction, q_values_input, corrected_reflectivity, n_variants, scale=0.001):
    """Create ``n_variants`` interpolated reflectivity curve variants with randomly distributed q shifts."""
    shift = np.random.normal(loc=0, size=n_variants, scale=scale).reshape(n_variants, 1)
    shifted_qs = np.tile(q_values_input, (n_variants, 1)) + shift

    interpolated_curves = np.zeros((n_variants, len(q_values_prediction)))
    for i in range(n_variants):
        interpolated_curves[i] = interp_reflectivity(q_values_prediction, shifted_qs[i], corrected_reflectivity)
    return interpolated_curves, shift


def curve_scaling_variants(corrected_reflectivity, n_variants, scale=0.1):
    """Create ``n_variants`` reflectivity curve variants with randomly distributed scaling factors."""
    scalings = np.random.normal(loc=1, size=n_variants, scale=scale).reshape(n_variants, 1)
    scaled_curves = np.zeros((n_variants, len(corrected_reflectivity)))
    for i in range(n_variants):
        scaled_curves[i] = corrected_reflectivity.copy() * scalings[i]
    return scaled_curves, scalings


def curve_variant_log_mse(curve, variant_curves):
    """Calculate the log MSE of a curve and a :class:`ndarray` of curves"""
    errors = np.log10(curve) - np.log10(variant_curves)
    return np.mean(errors ** 2, axis=1)


def least_log_mean_squares_fit(q_values, data, predicted_labels, sample, output_preprocessor,
                               fraction_bounds=(0.5, 0.5, 0.1)):
    """Fits the data with a model curve with ``scipy.optimize.curve_fit`` using ``predicted_labels`` as start values."""
    prep_labels = output_preprocessor.apply_preprocessing(predicted_labels)[0]
    start_values = np.array(prep_labels)[0]
    bounds = ([val - bound * abs(val) for val, bound in zip(start_values, fraction_bounds)],
              [val + bound * abs(val) for val, bound in zip(start_values, fraction_bounds)])
    fit_result = curve_fit(fitting_model(q_values, sample, output_preprocessor), q_values, np.log10(data),
                           p0=start_values, bounds=bounds)
    return output_preprocessor.restore_labels(np.atleast_2d(fit_result[0]))


def fitting_model(q_values, sample, output_preprocessor):
    def log_refl_curve(q, *prep_labels):
        generator = ReflectivityGenerator(q_values, sample)
        restored_labels = output_preprocessor.restore_labels(np.atleast_2d(prep_labels))
        model = generator.simulate_reflectivity(restored_labels, progress_bar=False)[0]
        return np.log10(model)

    return log_refl_curve


def log_mse_loss(prep_labels, data, generator, output_preprocessor):
    """MSE loss between a reflectivity curve and a model curve generated with the given normalized labels."""
    restored_labels = output_preprocessor.restore_labels(np.atleast_2d(prep_labels))
    model = generator.simulate_reflectivity(restored_labels,
                                            progress_bar=False)[0]
    loss = mean_squared_error(np.log10(data), np.log10(model))
    return loss


def mean_squared_error(array1, array2):
    """Returns element-wise mean squared error between two arrays."""
    if len(array1) != len(array2):
        raise ValueError(f'array1 and array2 must be of same length ({len(array1)} != {len(array2)})')
    else:
        error = np.asarray(array1) - np.asarray(array2)
        return np.mean(np.atleast_2d(error ** 2), axis=1)
