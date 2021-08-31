import unittest

from mlreflect.curve_fitter import DefaultSpecFitter, SpecFitter
from mlreflect.models import DefaultTrainedModel


class TestCurveFitterMethods(unittest.TestCase):
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

    def test_fit(self):
        self.default_spec_fitter.fit(18, trim_front=3, trim_back=3, plot=False, polish=False)
        self.spec_fitter.fit(18, trim_front=3, trim_back=3, plot=False, polish=False)

        self.default_spec_fitter.fit(18, trim_front=None, trim_back=None, plot=False, polish=False)
        self.spec_fitter.fit(18, trim_front=None, trim_back=None, plot=False, polish=False)

    def test_fit_polish(self):
        self.default_spec_fitter.fit(18, trim_front=3, trim_back=3, plot=False, polish=True)
        self.spec_fitter.fit(18, trim_front=3, trim_back=3, plot=False, polish=True)

        self.default_spec_fitter.fit(18, trim_front=3, trim_back=None, plot=False, polish=True)
        self.spec_fitter.fit(18, trim_front=3, trim_back=None, plot=False, polish=True)

    def test_fit_range(self):
        self.default_spec_fitter.fit_range(range(20, 22), trim_front=5, plot=False, polish=False)
        self.spec_fitter.fit_range(range(20, 22), trim_front=5, plot=False, polish=False)


if __name__ == '__main__':
    unittest.main()
