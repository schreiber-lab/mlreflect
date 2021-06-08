import numpy as np
from scipy.optimize import minimize


def least_log_mean_squares_fit(data, predicted_labels, generator, output_preprocessor):
    """Fits the data with a model curve with `scipy.optimize.minimize` using `predicted_labels` as start values."""
    prep_labels = output_preprocessor.apply_preprocessing(predicted_labels)[0]
    start_values = np.array(prep_labels)[0]
    bounds = [(val - .5 * abs(val), val + .5 * abs(val)) for val in start_values]
    fit_params = minimize(log_mse_loss, start_values, args=(data, generator, output_preprocessor), bounds=bounds)
    return output_preprocessor.restore_labels(np.atleast_2d(fit_params.x))


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
        mse = np.zeros(len(array1))
        for i in range(len(array1)):
            mse[i] = (array1[i] - array2[i]) ** 2
        return np.mean(mse)
