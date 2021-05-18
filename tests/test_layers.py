import unittest

from mlreflect.data_generation import Layer


class TestLayerMethods(unittest.TestCase):
    def test_layer_name(self):
        name = 'layer name'
        layer = Layer(name, (50, 100), (1, 10), (1, 10))
        self.assertEqual(name, layer.name)

    def test_max_greater_min(self):
        with self.assertRaises(ValueError):
            Layer('layer', (100, 50), (1, 10), (1, 10))
        with self.assertRaises(ValueError):
            Layer('layer', (50, 100), (10, 1), (1, 10))
        with self.assertRaises(ValueError):
            Layer('layer', (50, 100), (1, 10), (10, 1))

    def test_min_not_negative(self):
        with self.assertRaises(ValueError):
            Layer('layer', (-1, 50), (1, 10), (1, 10))
        with self.assertRaises(ValueError):
            Layer('layer', (50, 100), (-1, 10), (1, 10))


if __name__ == '__main__':
    unittest.main()
