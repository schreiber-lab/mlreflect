import os.path
import tempfile
import unittest

import numpy as np
from tensorflow.keras.models import Model

from mlreflect.data_generation import MultilayerStructure, Layer, ConstantLayer, AmbientLayer, Substrate
from mlreflect.models import DefaultTrainedModel, TrainedModel


class TestTrainedModelMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.default_trained_model = DefaultTrainedModel()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_save_default_model(self):
        model_path = os.path.join(self.tempdir.name, 'default_model.h5')
        self.default_trained_model.save_model(model_path)
        loaded_model = TrainedModel()
        loaded_model.from_file(model_path)
        self.assertIsInstance(loaded_model.keras_model, Model)
        self.assertEqual(str(loaded_model.sample), str(self.default_trained_model.sample))
        np.testing.assert_array_equal(loaded_model.q_values, self.default_trained_model.q_values)
        np.testing.assert_array_equal(loaded_model.ip_std, self.default_trained_model.ip_std)
        np.testing.assert_array_equal(loaded_model.ip_mean, self.default_trained_model.ip_mean)

    def test_save_model(self):
        sub = Substrate('Si', 1, 20.07008928480288 + 0.4571087851738205j)
        ambient = AmbientLayer('Air', 0)
        layer1 = ConstantLayer('SiOx', 10.0, 2.5, 17.773511903954788 + 0.4048028047807962j)
        layer2 = Layer('Film', (20, 1000), (0, 100), (1, 14))
        sample = MultilayerStructure()
        sample.set_ambient_layer(ambient)
        sample.set_substrate(sub)
        sample.add_layer(layer1)
        sample.add_layer(layer2)

        modified_trained_model = TrainedModel()
        modified_trained_model.from_variable(self.default_trained_model.keras_model,
                                             sample,
                                             self.default_trained_model.q_values,
                                             self.default_trained_model.ip_mean,
                                             self.default_trained_model.ip_std)
        model_path = os.path.join(self.tempdir.name, 'model.h5')
        modified_trained_model.save_model(model_path)

        loaded_model = TrainedModel()
        loaded_model.from_file(model_path)
        self.assertIsInstance(loaded_model.keras_model, Model)
        self.assertEqual(str(loaded_model.sample), str(modified_trained_model.sample))
        np.testing.assert_array_equal(loaded_model.q_values, modified_trained_model.q_values)
        np.testing.assert_array_equal(loaded_model.ip_std, modified_trained_model.ip_std)
        np.testing.assert_array_equal(loaded_model.ip_mean, modified_trained_model.ip_mean)

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
