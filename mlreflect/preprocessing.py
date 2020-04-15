from typing import Iterable, Union, Tuple
from warnings import warn

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from .label_helpers import convert_to_dataframe
from .layers import MultilayerStructure


class InputPreprocessor:
    """Allows standardization while storing mean and standard deviation for later use.

    Args: None

    Returns:
        InputPreprocessor object.
    """

    def __init__(self):
        self._standard_mean = None
        self._standard_std = None

        self._has_saved_standardization = False

    def standardize(self, data: ndarray) -> ndarray:
        """Applies standardization along axis 0 and returns standardized data. Mean and std will be reused."""
        if self._has_saved_standardization is True:
            mean = self._standard_mean
            std = self._standard_std
            data_centered = data - mean
        else:
            mean = np.mean(data, axis=0)
            data_centered = data - mean
            std = np.std(data_centered, axis=0)

            self._standard_mean = mean
            self._standard_std = std
            self._has_saved_standardization = True

        standardized_data = data_centered / std
        return standardized_data

    def revert_standardization(self, standardized_data: ndarray):
        if self._has_saved_standardization is False:
            raise ValueError('There are no saved mean and std to revert!')
        else:
            reverted_data = standardized_data * self._standard_std
            reverted_data += self._standard_mean
        return reverted_data

    def reset_mean_and_std(self):
        """Resets previously stored mean and standard deviation for standardization."""
        self._standard_mean = None
        self._standard_std = None
        self._has_saved_standardization = False


class OutputPreprocessor:
    """Class for preprocessing reflectivity labels for training and validation.

    Args:
        sample: MultilayerStructure object where the sample layers and their names and parameter ranges are defined.
        normalization: Defines how the output labels are normalized.
            "min_to_zero" (default): shifts minimum value to 0 and scales maximum value to 1 (= range [0, 1]).
            "absolute_max": scales absolute maximum value to 1 (= range [-1, 1]).

    Returns:
        OutputPreprocessor object
    """

    def __init__(self, sample: MultilayerStructure, normalization: str = 'min_to_zero'):
        allowed_normalizations = ['min_to_zero', 'absolute_max']
        if normalization in allowed_normalizations:
            self.normalization = normalization
        else:
            raise ValueError(f'normalization type "{normalization}" not supported')

        self._thickness_ranges = sample.get_thickness_ranges()
        self._roughness_ranges = sample.get_roughness_ranges()
        self.layer_sld_ranges = sample.get_layer_sld_ranges()
        self.ambient_sld_range = sample.get_ambient_sld_ranges()

        self._label_ranges = np.concatenate((self._thickness_ranges, self._roughness_ranges, self.layer_sld_ranges,
                                             self.ambient_sld_range), axis=0)

        self._labels_min = self._label_ranges[:, 0]
        self._labels_max = self._label_ranges[:, 1]

        self._number_of_layers = len(self._thickness_ranges)
        self._number_of_labels = len(self._label_ranges)

        self.all_label_names = sample.get_label_names()
        self.normalized_label_names = []
        self.removed_label_names = []
        self.constant_label_names = []
        self.used_label_names = []

        self._label_ranges_dict = {}
        for label_index in range(self._number_of_labels):
            self._label_ranges_dict[self.all_label_names[label_index]] = self._label_ranges[label_index, :]

        for name in self.all_label_names:
            label_min = self._label_ranges_dict[name][0]
            label_max = self._label_ranges_dict[name][1]
            if label_min == label_max:
                self.constant_label_names += [name]

    def apply_preprocessing(self, labels: Union[DataFrame, ndarray]) -> Tuple[DataFrame, DataFrame]:
        """Returns DataFrame of labels after normalization. Removes all constant labels and labels defined in
        `removed_label_names` and returns them separately."""
        label_df = convert_to_dataframe(labels, self.all_label_names)

        preprocessed_labels = self._remove_labels(label_df.copy())
        preprocessed_labels = self._normalize_labels(preprocessed_labels)

        removed_labels = label_df[self.removed_label_names + self.constant_label_names]

        return preprocessed_labels, removed_labels

    def add_to_removal_list(self, label_name: Union[str, Iterable[str]]):
        """Adds `label_name` to the list of labels that are removed during preprocessing. The names must be contained in
        `label_names`."""
        if type(label_name) is str:
            added_list = [label_name]
        elif type(label_name) is Iterable:
            added_list = list(label_name)
        else:
            raise TypeError('Wrong type for `label_names`.')

        for entry in added_list:
            if entry not in self.all_label_names:
                warn(f'{entry} not in `label_names` and it will be ignored.')
            elif entry in self.removed_label_names:
                warn(f'{entry} already in `removed_label_names` and it will be ignored.')
            else:
                self.removed_label_names.append(entry)

    def _normalize_labels(self, label_df: DataFrame) -> DataFrame:
        """Normalizes all labels contained in `normalized_label_names`."""

        for name in label_df.columns:
            label_min = self._label_ranges_dict[name][0]
            label_max = self._label_ranges_dict[name][1]
            if label_max != label_min:
                if self.normalization is 'min_to_zero':
                    label_df[name] = (label_df[name] - label_min) / (label_max - label_min)
                elif self.normalization is 'absolute_max':
                    label_df[name] = label_df[name] / np.abs(label_max)

        self.normalized_label_names = list(label_df.columns)

        return label_df

    def _remove_labels(self, label_df: DataFrame) -> DataFrame:
        """Removes labels in `removed_label_names` and `constant_label_names` from `label_df` and returns DataFrame."""

        removal_list = self.removed_label_names + self.constant_label_names
        for name in removal_list:
            if name not in label_df.columns:
                warn(f'Label "{name}" not in the list of labels (maybe already removed). Skipping "{name}".')
            else:
                del label_df[name]

        self.used_label_names = self._make_used_label_names()

        return label_df

    def _make_used_label_names(self):
        used_label_names = self.all_label_names.copy()
        removed_list = self.constant_label_names + self.removed_label_names
        for name in removed_list:
            used_label_names.remove(name)

        return used_label_names

    def restore_labels(self, predicted_labels: Union[DataFrame, ndarray],
                       removed_labels: Union[ndarray, DataFrame]) -> DataFrame:
        """Takes the predicted labels, reverts normalization and adds constant labels and returns those as DataFrame."""

        predicted_labels_df = convert_to_dataframe(predicted_labels, self.used_label_names)
        removed_labels_df = convert_to_dataframe(removed_labels, self.removed_label_names)

        restored_labels_df = self._renormalize_labels(predicted_labels_df)
        restored_labels_df = self._add_constant_labels(restored_labels_df, removed_labels_df)

        reordered_labels_df = restored_labels_df[self.all_label_names]

        return reordered_labels_df

    def _renormalize_labels(self, label_df: DataFrame) -> DataFrame:
        """Removes normalization from all labels in `label_df` which are in `normalized_label_names`."""
        if not self.normalized_label_names:
            raise ValueError('No normalized labels. `_normalize_labels` must be called first.')

        for name in self.normalized_label_names:
            label_min = self._label_ranges_dict[name][0]
            label_max = self._label_ranges_dict[name][1]
            if name not in self.constant_label_names:
                if self.normalization is 'min_to_zero':
                    label_df[name] = label_df[name] * (label_max - label_min) + label_min
                elif self.normalization is 'absolute_max':
                    label_df[name] = label_df[name] * np.abs(label_max)

        return label_df

    def _add_constant_labels(self, predicted_labels_df: DataFrame, removed_labels_df: DataFrame) -> DataFrame:
        """Adds all labels in `constant_label_names` from `removed_labels_df` to `predicted_labels_df`."""
        constant_label_list = self.constant_label_names
        for name in constant_label_list:
            predicted_labels_df[name] = removed_labels_df[name].reset_index(drop=True)

        return predicted_labels_df
