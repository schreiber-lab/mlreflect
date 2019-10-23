from typing import Iterable, Callable, Union, List, Tuple
from warnings import warn

import numpy as np
from numpy import ndarray

Jobfunc = Callable[[Iterable[float]], Iterable[float]]


class InputPreprocessor:
    """Class that stores a list of preprocessing functions ('jobs') that can be executed on reflectivity input data.

    Args: None

    Returns:
        InputPreprocessor object.

    Attributes:
        job_list: List of all preprocessing jobs that will be applied in that order.

    Methods:
        append_to_job_list()
        reset_list()
        preprocess_data()
        log()
        standardize()
        reset_standardization()
    """

    def __init__(self):
        self._job_list = []
        self._standard_mean = None
        self._standard_std = None

    @property
    def job_list(self):
        return self._job_list

    def append_to_job_list(self, function: Union[Jobfunc, List[Jobfunc]]):
        """Adds a function or list of functions to the job list for preprocessing.

        Args:
            function: Any function or list of functions that can be applied to the input data. The function must take
            no arguments except an ndarray of the data and only return the modified data with the same shape and type.

        Returns: None
        """
        if type(function) is list:
            for entry in function:
                if not callable(entry):
                    raise TypeError(f'List entry {entry} not callable.')
            self._job_list += function
        elif callable(function):
            self._job_list += [function]
        else:
            raise TypeError('Provided argument not callable.')

    def reset_list(self):
        """Removes all current functions from the job list."""
        self._job_list = []

    def preprocess_data(self, data: ndarray) -> ndarray:
        """Executes all functions on the job list. Subsequent functions take the previous return value as argument.

        Args:
            data: numpy.ndarray input data with dimensions [number_of_curves, number_of_q_values].

        Returns:
              Preprocessed data.
        """
        for job in self.job_list:
            data = job(data)
        return data

    @staticmethod
    def log(data: ndarray) -> ndarray:
        """Shortcut for numpy.log function."""
        return np.log(data)

    def standardize(self, data: ndarray, axis: int = 0) -> ndarray:
        """Applies standardization along specified axis and returns standardized data. Mean and std will be reused."""
        if self._standard_mean and self._standard_std is not None:
            mean = self._standard_mean
            std = self._standard_std
            data_centered = data - mean
        else:
            mean = np.mean(data, axis=axis)
            data_centered = data - mean
            std = np.std(data_centered, axis=axis)

            self._standard_mean = mean
            self._standard_std = std

        standardized_data = data_centered / std
        return standardized_data

    def reset_standardization(self):
        """Resets previously stored mean and standard deviation for standardization."""
        self._standard_mean = None
        self._standard_std = None


class OutputPreprocessor:
    """Class for preprocessing reflectivity labels for training and validation.

    Args:
        thickness_limits: An array-like object (list, tuple, ndarray, etc.) that contains a tuple with the min and max
            thickness in units of Å for each sample layer in order from top to bottom. The thickness of the bottom most
            layer (substrate) is not relevant for the simulation, but some value must be provided, e.g. (1, 1).
        roughness_limits: An array-like object (list, tuple, ndarray, etc.) that contains a tuple with the min and max
            roughness in units of Å for each sample interface in order from top (ambient/top layer) to bottom (bottom
            layer/substrate).
        sld_limits: An array-like object (list, tuple, ndarray, etc.) that contains a tuple with the min and max
            scattering length density (SLD) in units of 1e+14 1/Å^2 for each sample layer in order from top to bottom
            (excluding the ambient SLD).

    Returns:
        OutputPreprocessor object

    Attributes:
        label_names: List of names of all labels.
        removed_label_names: List of names of all labels that will be removed by `apply_preprocessing`.

    Methods:
        apply_preprocessing()
        add_to_removal_list()
        restore_labels()
    """

    def __init__(self, thickness_limits: Iterable[Tuple[float]], roughness_limits: Iterable[Tuple[float]],
                 sld_limits: Iterable[Tuple[float]]):

        self._thickness_limits = np.asarray(thickness_limits)
        self._roughness_limits = np.asarray(roughness_limits)
        self._sld_limits = np.asarray(sld_limits)

        self._label_limits = np.concatenate((self._thickness_limits, self._roughness_limits, self._sld_limits), axis=0)

        self._labels_min = self._label_limits[:, 0]
        self._labels_max = self._label_limits[:, 1]

        self.label_names = ['' for i in range(self._number_of_labels)]
        self.normalized_label_names = []

        self._number_of_layers = len(self._thickness_limits)
        self._number_of_labels = len(self._label_limits)

        for layer_index in range(self._number_of_layers):
            self.label_names[layer_index] = f'layer{self._number_of_layers - layer_index}_thickness'
            self.label_names[
                layer_index + self._number_of_layers] = f'layer{self._number_of_layers - layer_index}_roughness'
            self.label_names[
                layer_index + 2 * self._number_of_layers] = f'layer{self._number_of_layers - layer_index}_sld'

        self.removed_label_names = []

        for name in self.label_names:
            label_min = self._label_limits_dict[name][0]
            label_max = self._label_limits_dict[name][1]
            if label_min == label_max:
                self.removed_label_names += name

        self._label_limits_dict = {}
        for label_index in range(self._number_of_labels):
            self._label_limits_dict[self.label_names[label_index]] = self._label_limits[label_index, :]

    def apply_preprocessing(self, labels: ndarray) -> ndarray:
        """Returns `labels` after normalizing and removing all labels defined in `removed_label_names`."""
        label_dict = self._make_label_dict(labels)
        label_dict = self._remove_labels(label_dict)
        label_dict = self._normalize_labels(label_dict)
        preprocessed_labels = self._unpack_label_dict(label_dict)

        return preprocessed_labels

    def add_to_removal_list(self, label_name: Union[str, Iterable[str]]):
        """Adds `label_name` to the list of removed labels. The names must be contained in `label_names`."""
        if type(label_name) is str:
            added_list = [label_name]
        elif type(label_name) is Iterable:
            added_list = list(label_name)
        else:
            raise TypeError('Wrong type for `label_names`.')

        for entry in added_list:
            if entry not in self.label_names:
                warn(f'{entry} not in `label_names` and it will be skipped.')
            elif entry in self.removed_label_names:
                warn(f'{entry} already in `removed_label_names` and it will be skipped.')
            else:
                self.removed_label_names.append(entry)

    def _make_label_dict(self, labels: ndarray) -> dict:
        """Takes an ndarray of labels and returns a dictionary with a key-value pair for each label."""
        label_dict = {}
        for label_index in range(self._number_of_labels):
            label_dict[self.label_names[label_index]] = labels[:, label_index]

        return label_dict

    def _unpack_label_dict(self, label_dict: dict) -> ndarray:
        """Takes a dictionary with key-value pairs of labels and returns an ndarray of labels."""
        num_labels = len(label_dict.keys())
        num_samples = len(list(label_dict.values())[0])

        labels = np.zeros((num_samples, num_labels))

        label_names = self.label_names.copy()

        for name in self.removed_label_names:
            label_names.remove(name)

        for index, name in enumerate(label_names):
            labels[:, index] = label_dict[name]

        return labels

    def _normalize_labels(self, label_dict: dict) -> dict:
        """Normalizes all labels contained in `normalized_label_names` by their minimum and maximum values."""
        self.normalized_label_names = label_dict.keys()

        for name in self.normalized_label_names:
            label_min = self._label_limits_dict[name][0]
            label_max = self._label_limits_dict[name][1]
            if label_max != label_min:
                label_dict[name] = (label_dict[name] - label_min) / (label_max - label_min)

        return label_dict

    def _remove_labels(self, label_dict: dict, ) -> dict:
        """Removes all labels that are contained in `removed_label_names` from `label_dict` and returns new dict."""
        for name in self.removed_label_names:
            if name not in label_dict.keys():
                warn(f'Label "{name}" not in the list of labels (maybe already removed). Skipping "{name}".')
            else:
                del label_dict[name]

        return label_dict

    def restore_labels(self, predicted_labels: ndarray, training_labels: ndarray) -> ndarray:
        """Takes the predicted labels, reverts normalization and adds removed labels and returns those as ndarray."""
        predicted_labels_dict = self._make_label_dict(predicted_labels)
        training_labels_dict = self._make_label_dict(training_labels)

        restored_labels_dict = self._renormalize_labels(predicted_labels_dict)
        restored_labels_dict = self._add_removed_labels(restored_labels_dict, training_labels_dict)

        restored_labels = self._unpack_label_dict(restored_labels_dict)

        return restored_labels

    def _renormalize_labels(self, label_dict: dict) -> dict:
        """Removes min-max normalization from all labels in `label_dict` which are `normalized_label_names`."""
        if not self.normalized_label_names:
            raise ValueError('No normalized labels. `_normalize_labels` must be called first.')

        for name in self.normalized_label_names:
            label_min = self._label_limits_dict[name][0]
            label_max = self._label_limits_dict[name][1]
            if label_max != label_min:
                label_dict[name] = label_dict[name] * (label_max - label_min) + label_min

        return label_dict

    def _add_removed_labels(self, predicted_labels_dict: dict, training_labels_dict: dict) -> dict:
        """Adds all labels in `removed_label_names` from `training_labels_dict` to `predicted_labels_dict`."""
        for name in self.removed_label_names:
            predicted_labels_dict[name] = training_labels_dict[name]

        return predicted_labels_dict
