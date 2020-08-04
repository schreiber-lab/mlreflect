import os
from typing import Any

import h5py
import numpy as np
import pandas as pd
from h5py import File
from numpy import ndarray
from pandas import DataFrame


def save_data_as_h5(file_name: str, q_values: ndarray, reflectivity: ndarray, labels: DataFrame, number_of_layers: int):
    """Saves ``q_values``, ``reflectivity`` and ``labels`` in the .5h file ``file_name``. Labels are saved as pandas `DataFrame`.

        Args:
            file_name: Name or path of the .h5 file
            q_values: `ndarray` of q values in units 1/A
            reflectivity: `n-by-m` `ndarray` of reflectivity curves where `n` is the number of curves and m the number
            of q-values
            labels: pandas `DataFrame` of labels
            number_of_layers: Number of thin film layers that were simulated (excluding ambient layer)

    """

    file_name = ensure_h5_extension(file_name)

    number_of_curves = labels.shape[0]

    with h5py.File(file_name, 'a') as data_file:
        create_dataset_with_override(data_file, 'q_values', q_values)

        info = data_file.require_group('info')
        info.attrs['number_of_layers'] = number_of_layers
        info.attrs['num_curves'] = number_of_curves

        info.attrs['q_unit'] = '1/A'
        info.attrs['thickness_unit'] = 'A'
        info.attrs['roughness_uni'] = 'A'
        info.attrs['sld_unit'] = '1e-6 1/A^2'

        for label_name in labels.keys():
            info.attrs[label_name + '_min'] = labels[label_name].min()
            info.attrs[label_name + '_max'] = labels[label_name].max()

        create_dataset_with_override(data_file, 'reflectivity', reflectivity)

    labels.to_hdf(file_name, 'labels')


def save_noise(file_name: str, noise_array: ndarray, noise_levels: ndarray):
    file_name = ensure_h5_extension(file_name)
    with h5py.File(file_name, 'a') as data_file:
        create_dataset_with_override(data_file, 'shot_noise', noise_array)
        create_dataset_with_override(data_file, 'shot_noise_levels', noise_levels)


def save_background(file_name: str, bg_array: ndarray, bg_levels: ndarray):
    file_name = ensure_h5_extension(file_name)
    with h5py.File(file_name, 'a') as data_file:
        create_dataset_with_override(data_file, 'background', bg_array)
        create_dataset_with_override(data_file, 'background_levels', bg_levels)


def load_data(file_name: str) -> dict:
    """Reads all data in h5 file ``file_name`` and returns them as a `dict`."""
    with h5py.File(file_name, 'r') as data_file:
        q_values = np.array(data_file.get('q_values'))

        reflectivity = np.array(data_file.get('reflectivity'))

        labels = pd.read_hdf(file_name, 'labels')

        info = {}
        for key in data_file.get('info').attrs.keys():
            info[key] = np.array(data_file.get('info').attrs[key])

    return {'q_values': q_values, 'reflectivity': reflectivity, 'labels': labels, 'info': info}


def create_dataset_with_override(file: File, name: Any, data: Any):
    """Create dataset and delete already existing one with the same name."""
    if name in file:
        del file[name]

    file.create_dataset(name, data=data)


def ensure_h5_extension(file_name: str):
    if not (file_name.endswith('.h5') or file_name.endswith('.hdf5')):
        file_name += '.h5'

    return file_name


def strip_file_extension(file_name: str):
    return os.path.splitext(file_name)[0]
