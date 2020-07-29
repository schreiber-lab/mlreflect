from typing import Union, List

import numpy as np
import tensorflow.keras as keras
from numpy import ndarray
from pandas import DataFrame
from tensorflow.keras import Model

from ..utils.label_helpers import convert_to_dataframe
from ..utils.performance_tools import timer


class Prediction:
    def __init__(self, model_path: str, label_names: List[str]):
        self.model_path = model_path
        self.model = self._load_model_from_file(model_path)

        self.label_names = label_names

    @timer
    def predict_labels(self, test_input: ndarray):
        try:
            test_input = np.asarray(test_input)
        except TypeError:
            raise TypeError('test_input must be castable to ndarray')

        test_input = np.atleast_2d(test_input)

        predicted_labels = self.model.predict(test_input)
        predicted_labels = convert_to_dataframe(predicted_labels, self.label_names)

        return predicted_labels

    def mean_absolute_percentage_error(self, predicted_labels: Union[DataFrame, ndarray],
                                       test_labels: Union[DataFrame, ndarray]):

        test_labels = convert_to_dataframe(test_labels, self.label_names)
        predicted_labels = convert_to_dataframe(predicted_labels, self.label_names)

        absolute_percentage_error = abs(test_labels.reset_index() - predicted_labels.reset_index()) / abs(
            test_labels.reset_index())
        del absolute_percentage_error['index']
        mean_absolute_percentage_error = absolute_percentage_error.mean()

        return mean_absolute_percentage_error

    def mean_absolute_error(self, predicted_labels: Union[DataFrame, ndarray], test_labels: Union[DataFrame, ndarray]):

        test_labels = convert_to_dataframe(test_labels, self.label_names)
        predicted_labels = convert_to_dataframe(predicted_labels, self.label_names)

        absolute_error = abs(test_labels.reset_index() - predicted_labels.reset_index())
        del absolute_error['index']
        mean_absolute_error = absolute_error.mean()

        return mean_absolute_error

    @staticmethod
    def _load_model_from_file(model_path: str) -> Model:
        return keras.models.load_model(model_path)

    @staticmethod
    def _wrap_ndarray_in_list(test_input: Union[ndarray, List[ndarray]]):
        if type(test_input) is ndarray:
            return [test_input]
        elif type(test_input) is list:
            return test_input
        else:
            raise TypeError('test_input must be ndarray or list of ndarrays.')
