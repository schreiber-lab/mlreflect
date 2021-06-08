import unittest
import os.path
import tempfile

import numpy as np
from mlreflect.curve_fitter import CurveFitter
from mlreflect.models import DefaultTrainedModel, TrainedModel
from numpy import ndarray
from tensorflow.keras.models import Model
from pandas import DataFrame


class TestTrainedModelMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.default_trained_model = DefaultTrainedModel()
        cls.tempdir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_save_model(self):
        model_path = os.path.join(self.tempdir.name, 'model.h5')
        self.default_trained_model.save_model(model_path)
        loaded_model = TrainedModel()
        loaded_model.from_file(model_path)
        self.assertIsInstance(loaded_model.keras_model, Model)
        self.assertEqual(str(loaded_model.sample), str(self.default_trained_model.sample))
        np.testing.assert_array_equal(loaded_model.q_values, self.default_trained_model.q_values)
        np.testing.assert_array_equal(loaded_model.ip_std, self.default_trained_model.ip_std)
        np.testing.assert_array_equal(loaded_model.ip_mean, self.default_trained_model.ip_mean)

    def test_from_variable(self):
        loaded_model = TrainedModel()
        loaded_model.from_variable(self.default_trained_model.keras_model,
                                   self.default_trained_model.sample,
                                   self.default_trained_model.q_values,
                                   self.default_trained_model.ip_mean,
                                   self.default_trained_model.ip_std)

        self.assertIsInstance(loaded_model.keras_model, Model)
        self.assertEqual(str(loaded_model.sample), str(self.default_trained_model.sample))
        np.testing.assert_array_equal(loaded_model.q_values, self.default_trained_model.q_values)
        np.testing.assert_array_equal(loaded_model.ip_std, self.default_trained_model.ip_std)
        np.testing.assert_array_equal(loaded_model.ip_mean, self.default_trained_model.ip_mean)
