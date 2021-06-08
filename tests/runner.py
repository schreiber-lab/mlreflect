import unittest

import test_data_generator
import test_layers
import test_noise_generator
import test_preprocessing
import test_curve_fitter
import test_spec_fitter
import test_fit_results
import test_multilayer_structure
import test_trained_model

loader = unittest.TestLoader()
suite = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(test_data_generator))
suite.addTests(loader.loadTestsFromModule(test_layers))
suite.addTests(loader.loadTestsFromModule(test_noise_generator))
suite.addTests(loader.loadTestsFromModule(test_preprocessing))
suite.addTests(loader.loadTestsFromModule(test_curve_fitter))
suite.addTests(loader.loadTestsFromModule(test_spec_fitter))
suite.addTests(loader.loadTestsFromModule(test_fit_results))
suite.addTests(loader.loadTestsFromModule(test_multilayer_structure))
suite.addTests(loader.loadTestsFromModule(test_trained_model))

runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)
