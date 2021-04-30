from typing import Union, Tuple
from warnings import warn

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from ..data_generation import MultilayerStructure, ConstantParameter, Parameter
from ..utils.label_helpers import convert_to_dataframe


class InputPreprocessor:
    """Allows standardization while storing mean and standard deviation for later use.

    Returns:
        InputPreprocessor
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
        """Applies standardization along ``axis=0`` and returns standardized data. Mean and std will be reused."""
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
        sample: :class:`MultilayerStructure` object where the sample layers and their names and parameter ranges are
            defined.
        normalization: Defines how the output labels are normalized.
            "min_to_zero" (default): shifts minimum value to ``0`` and scales maximum value to ``1``).
            "absolute_max": scales absolute maximum value to ``1``).

    Returns:
        OutputPreprocessor
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
    def all_label_parameters(self):
        return np.concatenate((self.sample.thicknesses, self.sample.roughnesses, self.sample.slds), axis=0)

    @property
    def number_of_labels(self):
        return len(self.all_label_parameters)

    @property
    def number_of_layers(self):
        return len(self.sample.thicknesses)

    @property
    def constant_labels(self):
        constant_labels = []
        for parameter in self.all_label_parameters:
            if isinstance(parameter, ConstantParameter):
                constant_labels.append(parameter)
        return constant_labels

    @property
    def used_labels(self):
        used_labels = []
        for parameter in self.all_label_parameters:
            if isinstance(parameter, Parameter) and not isinstance(parameter, ConstantParameter):
                used_labels.append(parameter)
        return used_labels

    def apply_preprocessing(self, labels: Union[DataFrame, ndarray]) -> Tuple[DataFrame, DataFrame]:
        """Removes all constant labels and applies normalization to the non-constant labels.

        Args:
            labels: Pandas DataFrame or ndarray of randomly generated labels.

        Returns:
            normalized_labels: DataFrame
            constant_labels: DataFrame
        """
        label_df = convert_to_dataframe(labels, self.all_label_names)

        preprocessed_labels = self._remove_labels(label_df.copy())
        preprocessed_labels = self._normalize_labels(preprocessed_labels)

        removed_labels_df = label_df[[param.name for param in self.constant_labels]]

        return preprocessed_labels, removed_labels_df

    def _normalize_labels(self, label_df: DataFrame) -> DataFrame:
        """Normalizes all constant labels and returns normalized DataFrame."""

        for parameter in self.all_label_parameters:
            if not isinstance(parameter, ConstantParameter):
                if self.normalization is 'min_to_zero':
                    label_df[parameter.name] = (label_df[parameter.name] - parameter.min) / (parameter.max -
                                                                                             parameter.min)
                elif self.normalization is 'absolute_max':
                    label_df[parameter.name] = label_df[parameter.name] / np.abs(parameter.max)
        return label_df

    def _remove_labels(self, label_df: DataFrame) -> DataFrame:
        """Removes labels in `constant_labels` from `label_df` and returns DataFrame."""

        for param in self.constant_labels:
            if param.name not in label_df.columns:
                warn(f'Label "{param.name}" not in the list of labels (maybe already removed). Skipping "'
                     f'{param.name}".')
            else:
                del label_df[param.name]
        return label_df

    def restore_labels(self, predicted_labels: Union[DataFrame, ndarray]) -> DataFrame:
        """Takes the predicted labels, reverts normalization and adds constant labels and returns those as DataFrame."""

        predicted_labels_df = convert_to_dataframe(predicted_labels, [param.name for param in self.used_labels])

        restored_labels_df = self._renormalize_labels(predicted_labels_df)
        restored_labels_df = self._add_constant_labels(restored_labels_df)

        return restored_labels_df[self.all_label_names]

    def _renormalize_labels(self, label_df: DataFrame) -> DataFrame:
        """Removes normalization from all labels in ``label_df``."""

        for param in self.used_labels:
            if isinstance(param, Parameter) and not isinstance(param, ConstantParameter):
                if self.normalization is 'min_to_zero':
                    label_df[param.name] = label_df[param.name] * (param.max - param.min) + param.min
                elif self.normalization is 'absolute_max':
                    label_df[param.name] = label_df[param.name] * np.abs(param.max)
        return label_df

    def _add_constant_labels(self, predicted_labels_df: DataFrame) -> DataFrame:
        """Adds all labels in ``constant_labels`` to ``predicted_labels_df``."""
        for param in self.constant_labels:
            predicted_labels_df[param.name] = param.value
        return predicted_labels_df
