import os
from typing import Any

import h5py
import numpy as np
import pandas as pd
from h5py import File
from numpy import ndarray
from pandas import DataFrame


def save_data_as_h5(file_name: str, group_name: str, q_values: ndarray, reflectivity: ndarray, labels: DataFrame):
    """Saves `q_values`, `reflectivity` and `labels` in the .5h file `file_name` in the group `group name`."""

    file_name = ensure_h5_extension(file_name)

    number_of_layers = labels.shape[1] * 3
    number_of_curves = labels.shape[0]

    with h5py.File(file_name, 'a') as data_file:
        data_file.attrs['q_unit'] = '1/A'
        create_dataset_with_override(data_file, 'q_values', q_values)

        data_file.attrs['number_of_layers'] = number_of_layers

        main_group = data_file.require_group(group_name)

        info = main_group.require_group('info')

        info.attrs['num_curves'] = number_of_curves

        for label_name in labels.keys():
            info.attrs[label_name + '_min'] = labels[label_name].min()
            info.attrs[label_name + '_max'] = labels[label_name].max()

        data_group = main_group.require_group('data')

        create_dataset_with_override(data_group, 'reflectivity', reflectivity)

    labels.to_hdf(file_name, group_name + '/data/labels')


def read_from_h5(file_name: str, group_name: str):
    """Reads reflectivity and labels from group `group_name` in h5 file `file_name` and returns them in a dict."""
    with h5py.File(file_name, 'r') as data_file:
        q_values = np.array(data_file.get('q_values'))

        reflectivity = np.array(data_file.get(group_name + '/data/reflectivity'))

        labels = pd.read_hdf(file_name, group_name + '/data/labels')

        info = {}
        for key in data_file.get(group_name + '/info').attrs.keys():
            info[key] = np.array(data_file.get(group_name + '/info').attrs[key])

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
