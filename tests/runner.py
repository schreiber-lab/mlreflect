import unittest

import test_data_generator
import test_layers
import test_preprocessing
import test_noise_generator

loader = unittest.TestLoader()
suite = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(test_data_generator))
suite.addTests(loader.loadTestsFromModule(test_layers))
suite.addTests(loader.loadTestsFromModule(test_noise_generator))
suite.addTests(loader.loadTestsFromModule(test_preprocessing))

runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)
