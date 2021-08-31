import os.path
import tempfile
import unittest

import numpy as np
import pandas as pd
from mlreflect.curve_fitter import DefaultSpecFitter, SpecFitter
from mlreflect.data_generation import MultilayerStructure
from mlreflect.models import DefaultTrainedModel
from numpy import ndarray
from pandas import DataFrame


class TestFitResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        file_name = 'test_spec_file.spec'

        import_params = {
            'angle_columns': ["Theta", "Two Theta"],
            'intensity_column': "Speccorr"
        }
        footprint_params = home_machine_footprint_params = {
            'wavelength': 1.5406,
            'beam_width': 0.165,
            'sample_length': 10
        }

        cls.default_spec_fitter = DefaultSpecFitter()
        cls.default_spec_fitter.set_import_params(**import_params)
        cls.default_spec_fitter.set_footprint_params(**footprint_params)
        cls.default_spec_fitter.set_file(file_name)

        cls.spec_fitter = SpecFitter()
        cls.spec_fitter.set_trained_model(DefaultTrainedModel())
        cls.spec_fitter.set_import_params(**import_params)
        cls.spec_fitter.set_footprint_params(**footprint_params)
        cls.spec_fitter.set_file(file_name)

        cls.default_fitter_results = cls.default_spec_fitter.fit(18, trim_front=3, trim_back=3, plot=False,
                                                                 polish=False)
        cls.spec_fitter_results = cls.spec_fitter.fit(18, trim_front=3, trim_back=3, plot=False, polish=False)

        cls.default_fitter_series = cls.default_spec_fitter.fit_range(range(20, 22), trim_front=5, plot=False,
                                                                      polish=False)
        cls.spec_fitter_series = cls.spec_fitter.fit_range(range(20, 22), trim_front=5, plot=False, polish=False)

        cls.tempdir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_fit_result(self):
        self._test_fit_results(self.default_fitter_results)
        self._test_fit_results(self.spec_fitter_results)

    def _test_fit_results(self, fit_results):
        self.assertEqual(fit_results.scan_number, 18)
        self.assertEqual(fit_results.timestamp, 'Tue Jan 26 12:23:23 2021')

        self.assertIsInstance(fit_results.corrected_reflectivity, ndarray)
        self.assertIsInstance(fit_results.q_values_input, ndarray)
        self.assertIsInstance(fit_results.predicted_reflectivity, ndarray)
        self.assertIsInstance(fit_results.q_values_prediction, ndarray)

        self.assertEqual(len(fit_results.q_values_input), 44)
        self.assertEqual(len(fit_results.corrected_reflectivity), 44)
        self.assertEqual(len(fit_results.q_values_prediction), 109)
        self.assertEqual(len(fit_results.predicted_reflectivity), 44)

        self.assertIsInstance(fit_results.predicted_parameters, DataFrame)
        self.assertIsInstance(fit_results.sample, MultilayerStructure)

    def test_fit_series(self):
        self._test_fit_results_series(self.default_fitter_series)
        self._test_fit_results_series(self.spec_fitter_series)

    def _test_fit_results_series(self, fit_results_series):
        self.assertEqual(fit_results_series.scan_number, [20, 21])
        self.assertEqual(fit_results_series.timestamp, ['Tue Jan 26 12:29:17 2021', 'Tue Jan 26 12:32:14 2021'])

        self.assertIsInstance(fit_results_series.corrected_reflectivity, ndarray)
        self.assertIsInstance(fit_results_series.q_values_input, ndarray)
        self.assertIsInstance(fit_results_series.predicted_reflectivity, ndarray)
        self.assertIsInstance(fit_results_series.q_values_prediction, ndarray)

        self.assertEqual(len(fit_results_series.q_values_input), 2)
        self.assertEqual(len(fit_results_series.corrected_reflectivity), 2)
        self.assertEqual(len(fit_results_series.q_values_prediction), 2)
        self.assertEqual(len(fit_results_series.predicted_reflectivity), 2)
        self.assertEqual(len(fit_results_series.predicted_parameters), 2)

        self.assertIsInstance(fit_results_series.predicted_parameters, DataFrame)
        self.assertIsInstance(fit_results_series.sample, MultilayerStructure)

    def test_save_corrected_reflectivity(self):
        file_name1 = os.path.join(self.tempdir.name, 'corrected_reflectivity1.dat')
        file_name2 = os.path.join(self.tempdir.name, 'corrected_reflectivity2.dat')
        self.default_fitter_results.save_corrected_reflectivity(file_name1)
        self.spec_fitter_results.save_corrected_reflectivity(file_name2)

        data1 = np.loadtxt(file_name1)
        data2 = np.loadtxt(file_name2)

        np.testing.assert_almost_equal(data1[:, 0], self.default_fitter_results.q_values_input, decimal=4)
        np.testing.assert_almost_equal(data1[:, 1], self.default_fitter_results.corrected_reflectivity, decimal=4)

        np.testing.assert_almost_equal(data2[:, 0], self.spec_fitter_results.q_values_input, decimal=4)
        np.testing.assert_almost_equal(data2[:, 1], self.spec_fitter_results.corrected_reflectivity, decimal=4)

    def test_save_predicted_reflectivity(self):
        file_name1 = os.path.join(self.tempdir.name, 'predicted_reflectivity1.dat')
        file_name2 = os.path.join(self.tempdir.name, 'predicted_reflectivity2.dat')
        self.default_fitter_results.save_predicted_reflectivity(file_name1)
        self.spec_fitter_results.save_predicted_reflectivity(file_name2)

        data1 = np.loadtxt(file_name1)
        data2 = np.loadtxt(file_name2)

        np.testing.assert_almost_equal(data1[:, 0], self.default_fitter_results.q_values_input, decimal=4)
        np.testing.assert_almost_equal(data1[:, 1], self.default_fitter_results.predicted_reflectivity, decimal=4)

        np.testing.assert_almost_equal(data2[:, 0], self.spec_fitter_results.q_values_input, decimal=4)
        np.testing.assert_almost_equal(data2[:, 1], self.spec_fitter_results.predicted_reflectivity, decimal=4)

    def test_save_predicted_parameters(self):
        file_name1 = os.path.join(self.tempdir.name, 'predicted_parameters1.dat')
        file_name2 = os.path.join(self.tempdir.name, 'predicted_parameters2.dat')
        self.default_fitter_results.save_predicted_parameters(file_name1)
        self.spec_fitter_results.save_predicted_parameters(file_name2)

        data1 = pd.read_csv(file_name1, sep='\t', index_col='scan')
        data2 = pd.read_csv(file_name2, sep='\t', index_col='scan')

        self.assertTrue(all(data1 == self.default_fitter_results.predicted_parameters))

        self.assertTrue(all(data2 == self.spec_fitter_results.predicted_parameters))

    def test_save_corrected_reflectivity_range(self):
        file_name1 = os.path.join(self.tempdir.name, 'corrected_reflectivity1.dat')
        file_name2 = os.path.join(self.tempdir.name, 'corrected_reflectivity2.dat')
        self.default_fitter_series.save_corrected_reflectivity(file_name1)
        self.spec_fitter_series.save_corrected_reflectivity(file_name2)

        for scan_no, default_results, results in zip(['scan20', 'scan21'], self.default_fitter_series.fit_results_list,
                                                     self.spec_fitter_series.fit_results_list):

            scan_name1 = os.path.join(os.path.dirname(file_name1), f'{scan_no}_{os.path.basename(file_name1)}')
            scan_name2 = os.path.join(os.path.dirname(file_name2), f'{scan_no}_{os.path.basename(file_name2)}')
            data1 = np.loadtxt(scan_name1)
            data2 = np.loadtxt(scan_name2)

            np.testing.assert_almost_equal(data1[:, 0], default_results.q_values_input, decimal=4)
            np.testing.assert_almost_equal(data1[:, 1], default_results.corrected_reflectivity, decimal=4)

            np.testing.assert_almost_equal(data2[:, 0], results.q_values_input, decimal=4)
            np.testing.assert_almost_equal(data2[:, 1], results.corrected_reflectivity, decimal=4)

    def test_save_predicted_reflectivity_range(self):
        file_name1 = os.path.join(self.tempdir.name, 'predicted_reflectivity1.dat')
        file_name2 = os.path.join(self.tempdir.name, 'predicted_reflectivity2.dat')
        self.default_fitter_series.save_predicted_reflectivity(file_name1)
        self.spec_fitter_series.save_predicted_reflectivity(file_name2)

        for scan_no, default_results, results in zip(['scan20', 'scan21'], self.default_fitter_series.fit_results_list,
                                                     self.spec_fitter_series.fit_results_list):

            scan_name1 = os.path.join(os.path.dirname(file_name1), f'{scan_no}_{os.path.basename(file_name1)}')
            scan_name2 = os.path.join(os.path.dirname(file_name2), f'{scan_no}_{os.path.basename(file_name2)}')
            data1 = np.loadtxt(scan_name1)
            data2 = np.loadtxt(scan_name2)

            np.testing.assert_almost_equal(data1[:, 0], default_results.q_values_input, decimal=4)
            np.testing.assert_almost_equal(data1[:, 1], default_results.predicted_reflectivity, decimal=4)

            np.testing.assert_almost_equal(data2[:, 0], results.q_values_input, decimal=4)
            np.testing.assert_almost_equal(data2[:, 1], results.predicted_reflectivity, decimal=4)

    def test_save_predicted_parameters_range(self):
        file_name1 = self.tempdir.name + 'predicted_parameters1.dat'
        file_name2 = self.tempdir.name + 'predicted_parameters2.dat'
        self.default_fitter_series.save_predicted_parameters(file_name1)
        self.spec_fitter_series.save_predicted_parameters(file_name2)

        data1 = pd.read_csv(file_name1, sep='\t', index_col='scan')
        data2 = pd.read_csv(file_name2, sep='\t', index_col='scan')

        self.assertTrue(all(data1 == self.default_fitter_series.predicted_parameters))

        self.assertTrue(all(data2 == self.spec_fitter_series.predicted_parameters))


if __name__ == '__main__':
    unittest.main()
