from copy import copy
from typing import Union, Iterable

import numpy as np

from .parameters import Parameter, ConstantParameter


class Layer:
    """Defines the name and parameter ranges of a single sample layer.

    Args:
        name: User defined name of this layer.
        thickness: Thickness of this layer in units of Å. Can be a float for a constant vale or a tuple for a range.
        roughness: Roughness of this layer in units of Å. Can be a float for a constant vale or a tuple for a range.
        sld: Scattering length density (SLD) of this layer in units of 1e-6 1/Å^2. Can be a float for a constant vale or
            a tuple for a range.

    Returns:
        Layer object
    """

    def __init__(self, name: str,
                 thickness: Union[float, Iterable, Parameter],
                 roughness: Union[float, Iterable, Parameter],
                 sld: Union[float, complex, Iterable, Parameter]):
        self.name = name

        self.thickness = self._set_parameter(thickness, 'thickness')
        self.roughness = self._set_parameter(roughness, 'roughness')
        self.sld = self._set_parameter(sld, 'sld')

        for param in (self.thickness, self.roughness):
            if param.min < 0:
                raise ValueError(f'min for {name} cannot be negative')

    @property
    def ranges(self):
        return {
            'min_thickness': self.thickness.min,
            'max_thickness': self.thickness.max,
            'min_roughness': self.roughness.min,
            'max_roughness': self.roughness.max,
            'min_sld': self.sld.min,
            'max_sld': self.sld.max,
        }

    def to_dict(self):
        layer_dict = {'name': self.name}
        if hasattr(self, 'thickness'):
            layer_dict['thickness'] = self._parse_parameter(self.thickness)
        if hasattr(self, 'roughness'):
            layer_dict['roughness'] = self._parse_parameter(self.roughness)
        if hasattr(self, 'sld'):
            layer_dict['sld'] = self._parse_parameter(self.sld)
        return layer_dict

    def copy(self):
        return copy(self)

    def _parse_parameter(self, parameter):
        if isinstance(parameter.value, tuple):
            return self._if_complex_to_dict(parameter.value[0]), self._if_complex_to_dict(parameter.value[1])
        else:
            return self._if_complex_to_dict(parameter.value)

    @staticmethod
    def _if_complex_to_dict(value):
        if np.iscomplex(value):
            return {'re': np.real(value), 'im': np.imag(value)}
        else:
            return value

    def _set_parameter(self, param, name=None):
        if isinstance(param, Parameter):
            return param.copy()
        elif isinstance(param, Iterable) and not isinstance(param, dict):
            return Parameter(self._if_dict_to_complex(param[0]), self._if_dict_to_complex(param[1]), name)
        else:
            return ConstantParameter(self._if_dict_to_complex(param), name)

    @staticmethod
    def _if_dict_to_complex(value):
        if isinstance(value, dict):
            return value['re'] + value['im'] * 1j
        else:
            return value

    def __copy__(self):
        return Layer(self.name, self.thickness.copy(), self.roughness.copy(), self.sld.copy())

    def __str__(self):
        return f'{self.name}:\n' \
               f'\tthickness: {self.thickness} [Å]\n' \
               f'\troughness: {self.roughness} [Å]\n' \
               f'\tsld: {self.sld} [1e-6 1/Å^2]'

    def __repr__(self):
        return repr(f'<Layer({str(self.name)})>')


class ConstantLayer(Layer):
    """Defines the name and constant parameter values of a single sample layer.

    Args:
        name: User defined name of this layer.
        thickness: Thickness of this layer in units of Å. Must be a float or int.
        roughness: Roughness of this layer in units of Å. Must be a float or int.
        sld: Scattering length density (SLD) of this layer in units of 1e-6 1/Å^2. Must be a float or int.

    Returns:
        Layer object
    """

    def __init__(self, name: str,
                 thickness: Union[float, ConstantParameter],
                 roughness: Union[float, ConstantParameter],
                 sld: Union[float, complex, ConstantParameter]):
        self.name = name

        self.thickness = self._set_parameter(thickness, 'thickness')
        self.roughness = self._set_parameter(roughness, 'roughness')
        self.sld = self._set_parameter(sld, 'sld')

        for param in (self.thickness, self.roughness):
            if param.min < 0:
                raise ValueError(f'min for {param.name} cannot be negative')

    @property
    def ranges(self):
        return {
            'min_thickness': self.thickness.min,
            'max_thickness': self.thickness.max,
            'min_roughness': self.roughness.min,
            'max_roughness': self.roughness.max,
            'min_sld': self.sld.min,
            'max_sld': self.sld.max,
        }

    def _set_parameter(self, param, name=None):
        if isinstance(param, Parameter):
            return param.copy()
        elif isinstance(param, Iterable) and not isinstance(param, dict):
            ValueError(f'parameter {name} needs to be float or int')
        else:
            return ConstantParameter(self._if_dict_to_complex(param), name)

    def __copy__(self):
        return ConstantLayer(self.name, self.thickness.copy(), self.roughness.copy(), self.sld.copy())

    def __repr__(self):
        return repr(f'<ConstantLayer({str(self.name)})>')


class Substrate(ConstantLayer):
    def __init__(self, name: str,
                 roughness: Union[float, ConstantParameter],
                 sld: Union[float, complex, ConstantParameter]):
        self.name = name

        self.roughness = self._set_parameter(roughness, 'roughness')
        self.sld = self._set_parameter(sld, 'sld')

        if self.roughness.min < 0:
            raise ValueError('min for roughness cannot be negative')

    @property
    def ranges(self):
        return {
            'min_roughness': self.roughness.min,
            'max_roughness': self.roughness.max,
            'min_sld': self.sld.min,
            'max_sld': self.sld.max,
        }

    def __copy__(self):
        return Substrate(self.name, self.roughness.copy(), self.sld.copy())

    def __str__(self):
        return f'{self.name} (substrate):\n' \
               f'\troughness: {self.roughness} [Å]\n' \
               f'\tsld: {self.sld} [1e-6 1/Å^2]'

    def __repr__(self):
        return repr(f'<Substrate({str(self.name)})>')


class AmbientLayer(ConstantLayer):
    def __init__(self, name: str,
                 sld: Union[float, complex, ConstantParameter]):
        self.name = name
        self.sld = self._set_parameter(sld, 'sld')

    @property
    def ranges(self):
        return {
            'min_sld': self.sld.min,
            'max_sld': self.sld.max,
        }

    def __copy__(self):
        return AmbientLayer(self.name, self.sld.copy())

    def __str__(self):
        return f'{self.name} (ambient):\n' \
               f'\tsld: {self.sld} [1e-6 1/Å^2]'

    def __repr__(self):
        return repr(f'<AmbientLayer({str(self.name)})>')
