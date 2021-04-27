from copy import copy
from typing import Union, List

from .layers import Layer, ConstantLayer, AmbientLayer, Substrate


class MultilayerStructure:
    """Defines the structure of a multilayer sample through one or multiple :class:`Layer` objects and an ambient SLD.

    Initializes with a default ambient and substrate layer.

    Returns:
        MultilayerStructure
    """

    def __init__(self):
        self._ambient_layer = AmbientLayer('undefined', 0)
        self._substrate = Substrate('undefined', 0, 0)
        self.layers = []

    @property
    def ambient_layer(self):
        return self._ambient_layer

    @property
    def substrate(self):
        return self._substrate

    def set_ambient_layer(self, ambient_layer: AmbientLayer):
        """Set the ambient layer."""
        self._check_layer_type(ambient_layer)
        self._ambient_layer = ambient_layer

    def set_substrate(self, substrate: Substrate):
        """Set the substrate layer."""
        self._check_layer_type(substrate)
        self._substrate = substrate

    def add_layer(self, layer: Layer, index: Union[str, int] = 'next'):
        """Add layer at given index position. If ``index='next'`` (default) layer is appended (added on top)."""
        self._check_layer_type(layer)

        if index == 'next':
            self.layers.append(layer)
        elif type(index) is int:
            self.layers.insert(index, layer)
        else:
            raise ValueError('position must be an integer index or "next"')

    def rename_layer(self, layer_index, name):
        """Renames layer at given index to ``name``."""
        self.layers[layer_index].name = name

    def swap_layers(self, from_index: int, to_index: int):
        """Swaps the position of the two layers given by ``from_index`` and ``to_index``."""
        if type(from_index) is not int or type(to_index) is not int:
            raise ValueError('Indices must be integers')

        swapped_value = self.layers[to_index]
        self.layers[to_index] = self.layers[from_index]
        self.layers[from_index] = swapped_value

    def move_layer(self, from_index: int, to_index: int):
        """Moves layer with given index to new index, shifting all layers with equal or higher index up."""
        if type(from_index) is not int or type(to_index) is not int:
            raise ValueError('Indices must be integers')

        if from_index < to_index:
            self.layers.insert(to_index + 1, self.layers[from_index])
            self.layers.remove(self.layers[from_index])
        elif from_index > to_index:
            moved_item = self.layers[from_index]
            self.layers.remove(moved_item)
            self.layers.insert(to_index, moved_item)

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the MultilayerStructure"""
        sample_dict = {
            'ambient_layer': self.ambient_layer.to_dict(),
            'layers': [],
            'substrate': self.substrate.to_dict()
        }
        for i in reversed(range(len(self.layers))):
            sample_dict['layers'] = sample_dict['layers'] + [self.layers[i].to_dict()]
        return sample_dict

    def from_dict(self, sample_dict: dict):
        """Set the layers in the MultilayerStructure according to the dictionary representation."""
        self.set_ambient_layer(AmbientLayer(**sample_dict['ambient_layer']))
        self.set_substrate(Substrate(**sample_dict['substrate']))
        for layer in reversed(sample_dict['layers']):
            self.add_layer(Layer(**layer))

    def copy(self):
        return copy(self)

    @property
    def thicknesses(self) -> list:
        thickness_list = []
        for layer in self:
            if hasattr(layer, 'thickness'):
                thickness = layer.thickness.copy()
                thickness.name = f'{layer.name}_{thickness.name}'
                thickness_list.append(thickness)
        return thickness_list

    @property
    def roughnesses(self) -> list:
        roughness_list = []
        for layer in self:
            if hasattr(layer, 'roughness'):
                roughness = layer.roughness.copy()
                roughness.name = f'{layer.name}_{roughness.name}'
                roughness_list.append(roughness)
        return roughness_list

    @property
    def slds(self) -> list:
        sld_list = []
        for layer in self:
            sld = layer.sld.copy()
            sld.name = f'{layer.name}_{sld.name}'
            sld_list.append(sld)
        return sld_list

    @property
    def label_names(self) -> List[str]:
        """Get list of all layer names in order."""

        label_names = []

        for thickness in self.thicknesses:
            label_names.append(thickness.name)

        for roughness in self.roughnesses:
            label_names.append(roughness.name)

        for sld in self.slds:
            label_names.append(sld.name)

        return label_names

    @staticmethod
    def _check_layer_type(layer: Layer):
        if not isinstance(layer, Layer):
            raise ValueError('not of type layer')

    @property
    def _sample_structure(self):
        return [self.substrate] + self.layers + [self.ambient_layer]

    def __str__(self):
        output = f'{self.ambient_layer}\n'
        for i in reversed(range(len(self.layers))):
            output += f'[{i}] {self.layers[i]}\n'
        output += f'{self.substrate}'
        return output

    def __repr__(self):
        return repr(f'<MultilayerStructure({[layer for layer in self]})>')

    def __copy__(self):
        this_copy = MultilayerStructure()
        this_copy.set_ambient_layer(self._ambient_layer.copy())
        this_copy.set_substrate(self._substrate.copy())
        for layer in self.layers:
            this_copy.add_layer(layer.copy())
        return this_copy

    def __getitem__(self, item):
        return self._sample_structure[item]

    def __len__(self):
        return len(self._sample_structure)

    def __list__(self):
        return list(self._sample_structure)

    def __iter__(self):
        return (layer for layer in self._sample_structure)


class LayerOnSubstrate(MultilayerStructure):
    """Defines the structure of a multilayer sample through one or multiple :class:`Layer` objects and an ambient SLD.

    Can only contain one non-constant layer of class Layer, all others must be of class ConstantLayer.

    Initializes with a default ambient and substrate layer.

    Returns:
        MultilayerStructure
    """

    def add_layer(self, layer: Union[Layer, ConstantLayer], index: Union[str, int] = 'next'):
        """Add layer at given index position. If ``index='next'`` (default) layer is appended (added on top)."""
        if isinstance(layer, ConstantLayer):
            self._add_layer(layer, index)
        elif isinstance(layer, Layer):
            if self.has_variable_layer:
                raise ValueError('sample already contains a non-constant layer')
            else:
                self._add_layer(layer, index)

    def _add_layer(self, layer: Union[Layer, ConstantLayer], index: Union[str, int] = 'next'):
        if index == 'next':
            self.layers.append(layer)
        elif type(index) is int:
            self.layers.insert(index, layer)
        else:
            raise ValueError('position must be an integer index or "next"')

    @property
    def has_variable_layer(self):
        for layer in self.layers:
            if isinstance(layer, Layer) and not isinstance(layer, ConstantLayer):
                return True
            else:
                return False

    @staticmethod
    def _check_constant_layer_type(constant_layer: ConstantLayer):
        if not isinstance(constant_layer, ConstantLayer):
            raise ValueError('not of type ConstantLayer')
