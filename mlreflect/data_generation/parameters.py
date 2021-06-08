from copy import copy

import numpy as np

from .distributions import sample_distribution


class Parameter:
    """Defines a parameter range that can be sampled to generate random training data.

    Args:
        min_value: minimum value of the parameter range
        max_value: maximum value of the parameter range
        name: optional name for the parameter (usually 'thickness', 'roughness' or 'sld')
    """

    def __init__(self, min_value, max_value, name: str = None):
        if min_value >= max_value:
            raise ValueError('`min_value` must be lower than `max_value`')
        self.min = min_value
        self.max = max_value
        self.name = name

    @property
    def value(self):
        return self.min, self.max

    def sample(self, number_of_values: int, distribution_type: str = 'uniform', **distribution_params):
        """Samples random values in the given parameter range with the specified distribution.

        Args:
            number_of_values: number of sampled values
            distribution_type: distribution type
                'uniform' is numpy.random.uniform().
                'logarithmic' is a uniformly sampled on a logarithmic scale.
                'bolstered' is a uniform distribution where the sampling density is shifted towards both ends of the
                range.
            **distribution_params: catch-all for all parameters concerning the specified distribution
                min_value, max_value: float limits of the distribution
                bolster_fraction: float fraction of samples that are redistributed to the limits
                bolster_width: width of the normal distribution of the redistributed samples

        Returns:
            samples: ndarray
        """

        if np.all(np.isreal(self.value)):
            return sample_distribution(number_of_values, min_value=self.min, max_value=self.max,
                                       distribution_type=distribution_type, **distribution_params)
        else:
            real_part = sample_distribution(number_of_values, min_value=self.min.real, max_value=self.max.real,
                                            distribution_type=distribution_type, **distribution_params)

            imag_part = sample_distribution(number_of_values, min_value=self.min.imag, max_value=self.max.imag,
                                            distribution_type=distribution_type, **distribution_params)

            return real_part + 1j * imag_part

    def copy(self):
        return copy(self)

    def __copy__(self):
        return Parameter(self.min, self.max, self.name)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return repr(f'<Parameter({self.name} {str((self.min, self.max))})>')


class ConstantParameter(Parameter):
    """Defines a constant parameter that can be sampled to generate random training data.

    Args:
        value: value of the parameter
        name: optional name for the parameter (usually 'thickness', 'roughness' or 'sld')
    """

    def __init__(self, value, name: str = None):
        self._value = value
        self.name = name

    @property
    def value(self):
        return self._value

    @property
    def min(self):
        return self._value

    @min.setter
    def min(self, value):
        raise AttributeError('ConstantParameter cannot set min/max (set value instead)')

    @property
    def max(self):
        return self._value

    @max.setter
    def max(self, value):
        raise AttributeError('ConstantParameter cannot set min/max (set value instead)')

    def sample(self, number_of_values: int, **distribution_params):
        """Returns ndarray with `number_of_values` repetitions of the parameter value."""
        return np.repeat(self.value, number_of_values)

    def copy(self):
        return copy(self)

    def __copy__(self):
        return ConstantParameter(self.value, self.name)

    def __eq__(self, other):
        return float(self) == float(other)

    def __ne__(self, other):
        return float(self) != float(other)

    def __lt__(self, other):
        return float(self) < float(other)

    def __le__(self, other):
        return float(self) <= float(other)

    def __gt__(self, other):
        return float(self) > float(other)

    def __ge__(self, other):
        return float(self) >= float(other)

    def __str__(self):
        return str(self.value)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __repr__(self):
        return repr(f'<ConstantParameter({self.name} {str(self.value)})>')
