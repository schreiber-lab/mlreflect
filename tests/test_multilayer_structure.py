import unittest

from mlreflect.data_generation import Layer, MultilayerStructure, Substrate, AmbientLayer


class TestReflectivityGeneratorMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ambient = AmbientLayer('ambient', 0)
        cls.layer1 = Substrate('first_layer', 10, 20 + 1j)
        cls.layer2 = Layer('second_layer', (50, 150), 1, (-10, 10))

        cls.multilayer = MultilayerStructure()
        cls.multilayer.set_ambient_layer(cls.ambient)
        cls.multilayer.set_substrate(cls.layer1)
        cls.multilayer.add_layer(cls.layer2)

        cls.expected_dict = {
            'ambient_layer': {
                'name': 'ambient',
                'sld': 0},
            'layers': [{
                'name': 'second_layer',
                'thickness': (50, 150),
                'roughness': 1,
                'sld': (-10, 10)}],
            'substrate': {
                'name': 'first_layer',
                'roughness': 10,
                'sld': {'re': 20.0, 'im': 1.0}}}

    def test_to_dict(self):
        mutlilayer_dict = self.multilayer.to_dict()
        self.assertIsInstance(mutlilayer_dict, dict)
        self.assertEqual(self.expected_dict, mutlilayer_dict)

    def test_from_dict(self):
        from_dict_multilayer = MultilayerStructure()
        from_dict_multilayer.from_dict(self.expected_dict)
        self.assertEqual(str(from_dict_multilayer), str(self.multilayer))