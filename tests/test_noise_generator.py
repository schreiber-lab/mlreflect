import unittest

import numpy as np
from numpy.testing import assert_equal

from mlreflect.training import InputPreprocessor
from mlreflect.data_generation import ReflectivityGenerator, Layer, MultilayerStructure, AmbientLayer, Substrate
from mlreflect.training.noise_generator import NoiseGenerator


class TestNoiseGeneratorMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ambient = AmbientLayer('ambient', 0)
        cls.layer1 = Substrate('first_layer', 10, 20)
        cls.layer2 = Layer('second_layer', (50, 150), 1, (-10, 10))

        cls.multilayer = MultilayerStructure()
        cls.multilayer.set_ambient_layer(cls.ambient)
        cls.multilayer.set_substrate(cls.layer1)
        cls.multilayer.add_layer(cls.layer2)

        cls.q = np.linspace(0.01, 0.14, 100)
        cls.generator = ReflectivityGenerator(cls.q, cls.multilayer)

        cls.labels = cls.generator.generate_random_labels(10)
        cls.reflectivity = cls.generator.simulate_reflectivity(cls.labels)

        cls.ip = InputPreprocessor()
        cls.ip.standardize(cls.reflectivity)

    def test_batch_size(self):
        batch_size = 11
        with self.assertRaises(ValueError):
            noise_generator = NoiseGenerator(self.reflectivity, self.labels, self.ip, batch_size=batch_size,
                                             shuffle=False, mode='single', noise_range=None, background_range=None)

        batch_size = 3
        noise_generator = NoiseGenerator(self.reflectivity, self.labels, self.ip, batch_size=batch_size, shuffle=False,
                                         mode='single', noise_range=None, background_range=None)
        assert_equal(noise_generator[0][0], self.ip.standardize(self.reflectivity[:batch_size]))
        assert_equal(noise_generator[0][1], np.array(self.labels)[:batch_size])

        batch_size = 1
        noise_generator = NoiseGenerator(self.reflectivity, self.labels, self.ip, batch_size=batch_size, shuffle=False,
                                         mode='single', noise_range=None, background_range=None)
        assert_equal(noise_generator[0][0], self.ip.standardize(self.reflectivity[:batch_size]))
        assert_equal(noise_generator[0][1], np.array(self.labels)[:batch_size])

    def test_shuffle(self):
        noise_generator = NoiseGenerator(self.reflectivity, self.labels, self.ip, batch_size=10, shuffle=False,
                                         mode='single', noise_range=None, background_range=None)

        assert_equal(noise_generator[0][0], self.ip.standardize(self.reflectivity))
        assert_equal(noise_generator[0][1], np.array(self.labels))

        noise_generator = NoiseGenerator(self.reflectivity, self.labels, self.ip, batch_size=10, shuffle=True,
                                         mode='single', noise_range=None, background_range=None)

        with self.assertRaises(AssertionError):
            assert_equal(noise_generator[0][0], self.ip.standardize(self.reflectivity))
            assert_equal(noise_generator[0][1], np.array(self.labels))

    def test_noise(self):
        noise_generator = NoiseGenerator(self.reflectivity, self.labels, self.ip, batch_size=10, shuffle=False,
                                         mode='single', noise_range=(1e-7, 1e-4), background_range=(1e-6, 1e-3),
                                         relative_background_spread=0.1)
        first_draw = noise_generator[0]
        second_draw = noise_generator[0]
        assert_equal(first_draw[1], second_draw[1])
        with self.assertRaises(AssertionError):
            assert_equal(first_draw[0], second_draw[0])


if __name__ == '__main__':
    unittest.main()
