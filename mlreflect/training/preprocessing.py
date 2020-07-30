from typing import Iterable, Union, Tuple
from warnings import warn

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from ..data_generation.layers import MultilayerStructure
from ..utils.label_helpers import convert_to_dataframe


class InputPreprocessor:
    """Allows standardization while storing mean and standard deviation for later use.

    Args: None

    Returns:
        InputPreprocessor object.
    """

    def __init__(self):
        self._standard_mean = None
        self._standard_std = None

    @property
    def has_saved_standardization(self):
        if self._standard_std is None and self._standard_mean is None:
            return False
        elif self._standard_std is not None and self._standard_mean is not None:
            return True
        else:
            raise ValueError('Saved state different for mean and std. Try resetting to clear states.')

    @property
    def standard_mean(self):
        return self._standard_mean

    @property
    def standard_std(self):
        return self._standard_std

    def standardize(self, data: ndarray) -> ndarray:
        """Applies standardization along axis 0 and returns standardized data. Mean and std will be reused."""
        if self.has_saved_standardization is True:
            mean = self._standard_mean
            std = self._standard_std
            data_centered = data - mean
        else:
            mean = np.mean(data, axis=0)
            data_centered = data - mean
            std = np.std(data_centered, axis=0)
            if 0 in std:
                raise ValueError('std must not be 0')

            self._standard_mean = mean
            self._standard_std = std

        standardized_data = data_centered / std
        return standardized_data

    def revert_standardization(self, standardized_data: ndarray):
        if self.has_saved_standardization is False:
            raise ValueError('There are no saved mean and std to revert!')
        else:
            reverted_data = standardized_data * self._standard_std
            reverted_data += self._standard_mean
        return reverted_data

    def reset_mean_and_std(self):
        """Resets previously stored mean and standard deviation for standardization."""
        self._standard_mean = None
        self._standard_std = None


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

        self.sample = sample
        self._labels_removal_list = []

    @property
    def all_label_names(self):
        return self.sample.label_names

    @property
    def all_labels_min(self):
        return self.all_label_ranges[:, 0]

    @property
    def all_labels_max(self):
        return self.all_label_ranges[:, 1]

    @property
    def all_label_ranges(self):
        return np.concatenate(
            (self.thickness_ranges, self.roughness_ranges, self.layer_sld_ranges, self.ambient_sld_range), axis=0)

    @property
    def number_of_labels(self):
        return len(self.all_label_ranges)

    @property
    def number_of_layers(self):
        return len(self.thickness_ranges)

    @property
    def label_ranges_dict(self):
        label_ranges_dict = {}
        for label_index in range(self.number_of_labels):
            label_ranges_dict[self.all_label_names[label_index]] = self.all_label_ranges[label_index, :]
        return label_ranges_dict

    @property
    def constant_label_names(self):
        constant_label_names = []
        for name in self.all_label_names:
            label_min = self.label_ranges_dict[name][0]
            label_max = self.label_ranges_dict[name][1]
            if label_min == label_max:
                constant_label_names.append(name)
        return constant_label_names

    @property
    def used_label_names(self):
        not_used_list = self.constant_label_names + self.labels_removal_list
        return list(np.setdiff1d(self.all_label_names, not_used_list))

    @property
    def labels_removal_list(self):
        return self._labels_removal_list

    def add_to_removal_list(self, label_name: Union[str, Iterable[str]]):
        """Adds `label_name` to the list of labels that are removed during preprocessing. The names must be contained in
        `label_names`."""
        if type(label_name) is str:
            added_list = [label_name]
        elif isinstance(label_name, Iterable):
            added_list = list(label_name)
        else:
            raise TypeError('Wrong type for `label_names`.')

        for entry in added_list:
            if entry not in self.all_label_names:
                warn(f'{entry} not in `label_names` and it will be ignored.')
            elif entry in self.constant_label_names:
                warn(f'{entry} already in `constant_label_names` and it will be ignored.')
            elif entry in self.labels_removal_list:
                warn(f'{entry} already in `labels_removal_list` and it will be ignored.')
            else:
                self.labels_removal_list.append(entry)

    def remove_from_removal_list(self, label_name: Union[str, Iterable[str]]):
        if type(label_name) is str:
            added_list = [label_name]
        elif isinstance(label_name, Iterable):
            added_list = list(label_name)
        else:
            raise TypeError('Wrong type for `label_names`.')

        for entry in added_list:
            if entry in self.labels_removal_list:
                self._labels_removal_list.remove(entry)
            else:
                warn(f'{entry} not in `removed_label_names` and it will be ignored.')

    @property
    def thickness_ranges(self):
        return self.sample.thickness_ranges

    @property
    def roughness_ranges(self):
        return self.sample.roughness_ranges

    @property
    def layer_sld_ranges(self):
        return self.sample.layer_sld_ranges

    @property
    def ambient_sld_range(self):
        return self.sample.ambient_sld_ranges

    def apply_preprocessing(self, labels: Union[DataFrame, ndarray]) -> Tuple[DataFrame, DataFrame]:
        """Returns DataFrame of labels after normalization. Removes all constant labels and labels defined in
        `removed_label_names` and returns them separately."""
        label_df = convert_to_dataframe(labels, self.all_label_names)

        preprocessed_labels = self._remove_labels(label_df.copy())
        preprocessed_labels = self._normalize_labels(preprocessed_labels)

        removed_labels_df = label_df[self.labels_removal_list]

        return preprocessed_labels, removed_labels_df

    def _normalize_labels(self, label_df: DataFrame) -> DataFrame:
        """Normalizes all labels contained in `normalized_label_names`."""

        for name in label_df.columns:
            label_min = self.label_ranges_dict[name][0]
            label_max = self.label_ranges_dict[name][1]
            if label_max != label_min:
                if self.normalization is 'min_to_zero':
                    label_df[name] = (label_df[name] - label_min) / (label_max - label_min)
                elif self.normalization is 'absolute_max':
                    label_df[name] = label_df[name] / np.abs(label_max)
        return label_df

    def _remove_labels(self, label_df: DataFrame) -> DataFrame:
        """Removes labels in `removed_label_names` and `constant_label_names` from `label_df` and returns DataFrame."""

        removal_list = self.labels_removal_list + self.constant_label_names
        for name in removal_list:
            if name not in label_df.columns:
                warn(f'Label "{name}" not in the list of labels (maybe already removed). Skipping "{name}".')
            else:
                del label_df[name]
        return label_df

    def restore_labels(self, predicted_labels: Union[DataFrame, ndarray]) -> DataFrame:
        """Takes the predicted labels, reverts normalization and adds constant labels and returns those as DataFrame."""

        predicted_labels_df = convert_to_dataframe(predicted_labels, self.used_label_names)

        restored_labels_df = self._renormalize_labels(predicted_labels_df)
        restored_labels_df = self._add_constant_labels(restored_labels_df)

        remaining_label_names = self.all_label_names
        for name in self.labels_removal_list:
            remaining_label_names.remove(name)
        reordered_labels_df = restored_labels_df[remaining_label_names]

        return reordered_labels_df

    def _renormalize_labels(self, label_df: DataFrame) -> DataFrame:
        """Removes normalization from all labels in `label_df` which are in `normalized_label_names`."""

        for name in self.used_label_names:
            label_min = self.label_ranges_dict[name][0]
            label_max = self.label_ranges_dict[name][1]
            if name not in self.constant_label_names:
                if self.normalization is 'min_to_zero':
                    label_df[name] = label_df[name] * (label_max - label_min) + label_min
                elif self.normalization is 'absolute_max':
                    label_df[name] = label_df[name] * np.abs(label_max)
        return label_df

    def _add_constant_labels(self, predicted_labels_df: DataFrame) -> DataFrame:
        """Adds all labels in `constant_label_names` from `removed_labels_df` to `predicted_labels_df`."""
        for name in self.constant_label_names:
            predicted_labels_df[name] = self.label_ranges_dict[name][0]
        return predicted_labels_df
