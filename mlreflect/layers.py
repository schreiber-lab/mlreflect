from typing import Tuple, Union, List

import numpy as np
from numpy import ndarray


class Layer:
    """Defines the name and parameter ranges of a single sample layer.

    Args:
        name: User defined name of this layer.
        thickness_range: Tuple that contains min and max thickness for this layer in units of Å. The thickness of the
        bottom most layer (substrate) is not relevant for the simulation, but some value must be provided, e.g. (1, 1).
        roughness_range: Tuple that contains the min and max roughness for this layer in units of Å.
        sld_range: Tuple that contains a the min and max scattering length density (SLD) for this in units of 1e-6 1/Å^2
        layer.

    Returns:
        Layer object
    """

    def __init__(self, name: str, thickness_range: Tuple, roughness_range: Tuple, sld_range: Tuple):
        self.name = name

        self.min_thickness = thickness_range[0]
        self.max_thickness = thickness_range[1]

        self.min_roughness = roughness_range[0]
        self.max_roughness = roughness_range[1]

        self.min_sld = sld_range[0]
        self.max_sld = sld_range[1]

    def __str__(self):
        return f'{self.name}:\n' \
               f'\tthickness: {self.min_thickness} -- {self.max_thickness} [Å]\n' \
               f'\troughness: {self.min_roughness} -- {self.max_roughness} [Å]\n' \
               f'\tsld: {self.min_sld} -- {self.max_sld} [1e-6 1/Å^2]'

    def __repr__(self):
        return self.name


class MultilayerStructure:
    """Defines the structure of a multilayer sample through one or multiple Layer objects and an ambient SLD.

    Args:
        ambient_sld: Scattering length density of the ambient environment above the top most layer in units of 1e-6
        1/Å^2, e.g. ~0 for X-rays in air.

    Methods:
        add_layer()
        rename_layer()
        swap_layers()
        move_layers()
        get_thickness_ranges()
        get_roughness_ranges()
        get_sld_ranges()
        get_label_names()

    Returns:
        MultilayerStructure object
    """

    def __init__(self, ambient_sld: float):
        self.ambient_sld = ambient_sld
        self.layers = []

    def add_layer(self, layer: Layer, index: Union[str, int] = 'next'):
        """Add layer at given index position. If index='next' (default) layer is appended (added on top)."""
        if index == 'next':
            self.layers.append(layer)
        elif type(index) is int:
            self.layers.insert(index, layer)
        else:
            raise ValueError('position must be an integer index or "next"')

    def rename_layer(self, layer_index, name):
        """Renames layer at given index to `name`."""
        self.layers[layer_index].name = name

    def swap_layers(self, from_index: int, to_index: int):
        """Swaps the position of the two layers given by `from_index` and `to_index`."""
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

    def get_thickness_ranges(self) -> ndarray:
        """Get ndarray of tuples with min and max values of each layer thickness."""
        number_of_layers = len(self.layers)

        thickness_ranges = np.zeros((number_of_layers, 2))
        for i in reversed(range(number_of_layers)):
            thickness_ranges[i, :] = np.asarray((self.layers[i].min_thickness, self.layers[i].max_thickness))

        thickness_ranges[0] = (1, 1)

        return thickness_ranges

    def get_roughness_ranges(self) -> ndarray:
        """Get ndarray of tuples with min and max values of each layer roughness."""
        number_of_layers = len(self.layers)

        roughness_ranges = np.zeros((number_of_layers, 2))
        for i in reversed(range(number_of_layers)):
            roughness_ranges[i, :] = np.asarray((self.layers[i].min_roughness, self.layers[i].max_roughness))

        return roughness_ranges

    def get_sld_ranges(self) -> ndarray:
        """Get ndarray of tuples with min and max values of each layer SLD."""
        number_of_layers = len(self.layers)

        sld_ranges = np.zeros((number_of_layers, 2))
        for i in reversed(range(number_of_layers)):
            sld_ranges[i, :] = np.asarray((self.layers[i].min_sld, self.layers[i].max_sld))

        return sld_ranges

    def get_label_names(self) -> List[str]:
        """Get list of all layer names in order."""
        layer_names = [layer.name for layer in self.layers]
        number_of_layers = len(self.layers)

        label_names = ['' for i in range(number_of_layers * 3)]

        for layer_index in range(number_of_layers):
            label_names[layer_index] = f'{layer_names[layer_index]}_thickness'
            label_names[
                layer_index + number_of_layers] = f'{layer_names[layer_index]}_roughness'
            label_names[
                layer_index + 2 * number_of_layers] = f'{layer_names[layer_index]}_sld'

        return label_names

    def __str__(self):
        output = f'ambient_sld: {self.ambient_sld} [1e-6 1/Å^2]\n'
        for i in reversed(range(len(self.layers))):
            output += f'[{i}] {self.layers[i]}\n'
        return output

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return (layer for layer in self.layers)
