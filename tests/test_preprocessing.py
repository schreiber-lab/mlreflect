import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from mlreflect import InputPreprocessor, OutputPreprocessor
from mlreflect import Layer, MultilayerStructure
from mlreflect import ReflectivityGenerator


class TestInputPreprocessorMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layer1 = Layer('first_layer', (0, 100), (1, 10), (10, 20))
        cls.layer2 = Layer('second_layer', (50, 150), (1, 1), (-10, 10))

        cls.multilayer = MultilayerStructure((1, 5))
        cls.multilayer.add_layer(cls.layer1)
        cls.multilayer.add_layer(cls.layer2)

        cls.q = np.linspace(0.01, 0.14, 100)
        cls.generator = ReflectivityGenerator(cls.q, cls.multilayer)

        cls.labels = cls.generator.generate_random_labels(10)
        cls.reflectivity = cls.generator.simulate_reflectivity(cls.labels)

    def setUp(self) -> None:
        self.ip = InputPreprocessor()

    def test_initialize(self):
        self.assertFalse(self.ip.has_saved_standardization)
        self.assertIsNone(self.ip.standard_mean)
        self.assertIsNone(self.ip.standard_std)

    def test_standardize(self):
        self.ip.standardize(self.reflectivity)
        self.assertTrue(self.ip.has_saved_standardization)
        self.assertIsNotNone(self.ip.standard_mean)
        self.assertIsNotNone(self.ip.standard_std)

        saved_mean = self.ip.standard_mean
        saved_std = self.ip.standard_std
        other_reflectivity = self.reflectivity * 10 + 10
        self.ip.standardize(other_reflectivity)
        assert_equal(self.ip.standard_mean, saved_mean)
        assert_equal(self.ip.standard_std, saved_std)

    def test_revert_standardization(self):
        standardized_reflectivity = self.ip.standardize(self.reflectivity)
        assert_allclose(self.ip.revert_standardization(standardized_reflectivity), self.reflectivity)

    def test_reset_mean_and_std(self):
        self.ip.standardize(self.reflectivity)
        self.ip.reset_mean_and_std()
        self.assertFalse(self.ip.has_saved_standardization)
        self.assertIsNone(self.ip.standard_mean)
        self.assertIsNone(self.ip.standard_std)


class TestOutputPreprocessorMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layer1 = Layer('first_layer', (0, 100), (1, 10), (10, 20))
        cls.layer2 = Layer('second_layer', (50, 150), (1, 1), (-10, 10))

        cls.multilayer = MultilayerStructure((1, 5))
        cls.multilayer.add_layer(cls.layer1)
        cls.multilayer.add_layer(cls.layer2)

        cls.q = np.linspace(0.01, 0.14, 100)
        cls.generator = ReflectivityGenerator(cls.q, cls.multilayer)

        cls.labels = cls.generator.generate_random_labels(10)
        cls.reflectivity = cls.generator.simulate_reflectivity(cls.labels)

        cls.non_constant_label_names = [
            'second_layer_thickness',
            'first_layer_roughness',
            'first_layer_sld',
            'second_layer_sld',
            'ambient_sld'
        ]
        cls.constant_label_name = 'second_layer_roughness'

    def setUp(self) -> None:
        self.op_abs_max = OutputPreprocessor(self.multilayer, 'absolute_max')

        self.op_min_zero = OutputPreprocessor(self.multilayer, 'min_to_zero')

    def test_apply_preprocessing(self):
        prep_labels_abs_max, removed_labels_abs_max = self.op_abs_max.apply_preprocessing(self.labels)
        prep_labels_min_zero, removed_labels_min_zero = self.op_min_zero.apply_preprocessing(self.labels)

        self.assertEqual((10, 5), prep_labels_abs_max.shape)
        self.assertEqual((10, 5), prep_labels_min_zero.shape)

        self.assertTrue(all(label_name in prep_labels_abs_max.columns for label_name in
                            self.non_constant_label_names))
        self.assertTrue(all(label_name in prep_labels_min_zero.columns for label_name in
                            self.non_constant_label_names))

        self.assertNotIn(self.constant_label_name, prep_labels_abs_max.columns)
        self.assertNotIn(self.constant_label_name, prep_labels_min_zero.columns)

        for max_label in prep_labels_abs_max.max():
            self.assertLessEqual(max_label, 1)
        for min_label in prep_labels_abs_max.min():
            self.assertGreaterEqual(min_label, -1)

        for max_label in prep_labels_min_zero.max():
            self.assertLessEqual(max_label, 1)
        for min_label in prep_labels_min_zero.min():
            self.assertGreaterEqual(min_label, 0)

    def test_restore_labels(self):
        prep_labels_abs_max, removed_labels_abs_max = self.op_abs_max.apply_preprocessing(self.labels)
        prep_labels_min_zero, removed_labels_min_zero = self.op_min_zero.apply_preprocessing(self.labels)

        mock_predicted_labels_min_zero = np.array(prep_labels_min_zero)
        mock_predicted_labels_abs_max = np.array(prep_labels_abs_max)

        restored_labels_abs_max = self.op_abs_max.restore_labels(mock_predicted_labels_abs_max)
        restored_labels_min_zero = self.op_min_zero.restore_labels(mock_predicted_labels_min_zero)

        self.assertTrue(all(self.labels.columns == restored_labels_abs_max.columns))
        self.assertTrue(all(self.labels.columns == restored_labels_min_zero.columns))

        self.assertTrue(all(self.labels == restored_labels_abs_max))
        self.assertTrue(all(self.labels == restored_labels_min_zero))

    def test_add_to_removal_list(self):
        self.op_abs_max.add_to_removal_list('second_layer_thickness')
        self.assertEqual(1, self.op_abs_max.labels_removal_list.count('second_layer_thickness'))

        self.op_abs_max.add_to_removal_list(['second_layer_thickness', 'first_layer_roughness', 'ambient_sld'])
        self.assertEqual(1, self.op_abs_max.labels_removal_list.count('second_layer_thickness'))
        self.assertEqual(1, self.op_abs_max.labels_removal_list.count('first_layer_roughness'))

        self.op_abs_max.add_to_removal_list('second_layer_roughness')

        self.op_abs_max.remove_from_removal_list('second_layer_thickness')
        self.assertNotIn('second_layer_thickness', self.op_abs_max.labels_removal_list)

        prep_labels_abs_max, removed_labels_abs_max = self.op_abs_max.apply_preprocessing(self.labels)
        self.assertEqual((10, 3), prep_labels_abs_max.shape)
        removed_label_names = [
            'first_layer_roughness',
            'ambient_sld'
        ]
        self.assertTrue(all(label_name in removed_labels_abs_max.columns for label_name in removed_label_names))

        remaining_labels = [
            'second_layer_thickness',
            'first_layer_sld',
            'second_layer_sld'
        ]
        self.assertTrue(all(label_name in prep_labels_abs_max.columns for label_name in remaining_labels))

        mock_predicted_labels_abs_max = np.array(prep_labels_abs_max)
        restored_labels_abs_max = self.op_abs_max.restore_labels(mock_predicted_labels_abs_max)
        self.assertTrue(all(label_name in restored_labels_abs_max.columns for label_name in remaining_labels + [
            self.constant_label_name]))


if __name__ == '__main__':
    unittest.main()
