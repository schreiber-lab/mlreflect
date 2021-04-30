import matplotlib.pyplot as plt
from xrrloader import SpecLoader

from . import CurveFitter
from ..models import TrainedModel, DefaultTrainedModel


class SpecFitter:
    def __init__(self):
        self._trained_model = None
        self._curve_fitter = None

        self._spec_file = None
        self._import_params = {}
        self._footprint_params = {}
        self._spec_loader = None

    def fit(self, scan_number: int, trim_front: int = 0, dq: float = 0.0, factor: float = 1.0, plot=False):
        try:
            scan = self._spec_loader.load_scan(scan_number, trim_front)
        except KeyError:
            print(f'scan {scan_number} could not be found in {self._spec_file}')
            return

        predicted_refl, predicted_labels = self._curve_fitter.fit_curve(scan.corrected_intensity, scan.q, dq, factor)

        output = {
            'corrected_intensity': scan.corrected_intensity,
            'q_values_input': scan.q,
            'predicted_intensity': predicted_refl,
            'q_values_prediction': self._trained_model.q_values,
            'predicted_labels': predicted_labels
        }
        if plot:
            self._plot_prediction(**output)
        return output

    @staticmethod
    def _plot_prediction(q_values_input, corrected_intensity, q_values_prediction, predicted_intensity,
                         predicted_labels):
        plt.semilogy(q_values_input, corrected_intensity, 'o', label='data')
        plt.semilogy(q_values_prediction, predicted_intensity, label='prediction')
        plt.title('Prediction')
        plt.xlabel('q [1/A]')
        plt.ylabel('Intensity [a.u.]')
        plt.legend()
        plt.show()

    def load_spec_file(self, spec_file_path: str):
        self._spec_loader = SpecLoader(spec_file_path, **self._import_params, **self._footprint_params)
        self._spec_file = spec_file_path

    @property
    def trained_model(self):
        return self._trained_model

    def set_trained_model(self, trained_model: TrainedModel = None, model_path: str = None):
        input_error = ValueError('must provide either `trained_model` or `model_path`')
        if trained_model is None and model_path is None:
            raise input_error
        elif trained_model is not None and model_path is not None:
            raise input_error

        if trained_model is None:
            trained_model = TrainedModel()
            trained_model.from_file(model_path)
        self._trained_model = trained_model
        self._curve_fitter = CurveFitter(trained_model)

    @property
    def spec_file(self):
        return self._spec_file

    @property
    def import_params(self):
        return self._import_params

    def set_import_params(self, **params):
        self._import_params.update(params)

    @property
    def footprint_params(self):
        return self._footprint_params

    def set_footprint_params(self, **params):
        self._footprint_params.update(params)


class DefaultSpecFitter(SpecFitter):
    def __init__(self):
        super().__init__()
        self.set_trained_model(DefaultTrainedModel())
