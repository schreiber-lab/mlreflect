import unittest

import numpy as np
from mlreflect.curve_fitter import CurveFitter
from mlreflect.models import DefaultTrainedModel
from numpy import ndarray
from pandas import DataFrame


class TestCurveFitterMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trained_model = DefaultTrainedModel()
        data = np.loadtxt('test_curve.dat')
        cls.q = data[:, 0]
        cls.reflectivity = data[:, 1]

    def test_data_import(self):
        self.assertIsInstance(self.q, ndarray)
        self.assertIsInstance(self.reflectivity, ndarray)
        self.assertEqual(len(self.q), 295)
        self.assertEqual(len(self.reflectivity), 295)

    def test_fit_curve(self):
        curve_fitter = CurveFitter(self.trained_model)
        pred_refl, labels = curve_fitter.fit_curve(self.reflectivity, self.q)
        self.assertIsInstance(pred_refl, ndarray)
        self.assertIsInstance(labels, DataFrame)

        model_q = self.trained_model.q_values
        self.assertEqual(len(model_q), len(pred_refl))


if __name__ == '__main__':
    unittest.main()