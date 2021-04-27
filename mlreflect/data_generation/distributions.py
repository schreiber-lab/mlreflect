from typing import Union

import numpy as np
from numpy import ndarray


def sample_distribution(number_of_values: int, **kwargs) -> Union[float, ndarray]:
    """Sample from the specified distribution

    Args:
        number_of_values: number of samples values
        **kwargs:
            distribution_type: 'uniform', 'logarithmic' or 'bolstered'
            min_value, max_value: float limits of the distribution
            bolster_fraction: float fraction of samples that are redistributed to the limits
            bolster_width: width of the normal distribution of the redistributed samples

    Returns:
        samples: ndarray
    """
    if kwargs['distribution_type'] == 'bolstered':
        return bolstered_uniform_distribution(kwargs['min_value'], kwargs['max_value'], number_of_values,
                                              kwargs['bolster_fraction'], kwargs['bolster_width'])
    elif kwargs['distribution_type'] == 'uniform':
        return np.random.uniform(kwargs['min_value'], kwargs['max_value'], number_of_values)
    elif kwargs['distribution_type'] == 'logarithmic':
        return random_logarithmic_distribution(kwargs['min_value'], kwargs['max_value'], number_of_values)
    else:
        raise ValueError('not a valid distribution')


def bolstered_uniform_distribution(min_value: float, max_value: float, n_samples: int, bolster_fraction: float,
                                   bolster_width: float) -> Union[float, ndarray]:
    """Sample from a modified uniform distribution that has a higher sampling density at the limits

    Args:
        min_value: lower limit of the distribution
        max_value: upper limit of the distribution
        n_samples: number of samples drawn from the distribution
        bolster_fraction: fraction of samples that are drawn around the limits (instead of uniformly)
        bolster_width: width of the normal distribution that is sampled from around both limits

    Returns:
        samples: ndarray
    """
    span = max_value - min_value

    n_bolster = int(np.floor(n_samples * bolster_fraction / 2))
    n_uniform = n_samples - 2 * n_bolster

    uniform = np.random.uniform(min_value, max_value, n_uniform)

    bolster_min = np.random.normal(min_value, span * bolster_width, n_bolster)
    bolster_min = _fold_distribution(bolster_min, min_value, max_value)
    bolster_max = np.random.normal(max_value, span * bolster_width, n_bolster)
    bolster_max = _fold_distribution(bolster_max, min_value, max_value)

    total_distribution = np.concatenate((bolster_min, uniform, bolster_max))
    np.random.shuffle(total_distribution)

    return total_distribution


def _fold_distribution(values: ndarray, min_value: float, max_value: float) -> ndarray:
    """Fold all values in `values` inside the range (min_value, max_value)."""
    num_values = len(values)
    for i in range(num_values):
        if values[i] < min_value:
            values[i] += 2 * (min_value - values[i])
        elif values[i] > max_value:
            values[i] += 2 * (max_value - values[i])
    return values


def random_logarithmic_distribution(min_value: float, max_value: float, n_values: float) -> Union[float, ndarray]:
    """Sample uniformly from a logarithmic distribution."""
    log_ranges = (np.log10(min_value), np.log10(max_value))
    return 10 ** np.random.uniform(*log_ranges, n_values)
