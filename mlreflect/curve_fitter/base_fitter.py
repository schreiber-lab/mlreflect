from . import CurveFitter
from .results import FitResult, FitResultSeries
from ..models import TrainedModel


class BaseFitter:
    """Semi-abstract class for fitting directly from data files."""

    def __init__(self):
        self._trained_model = None
        self._curve_fitter = None

        self._import_params = {}
        self._footprint_params = {}
        self._loader = None
        self._file_name = None

    @property
    def trained_model(self):
        return self._trained_model

    @property
    def import_params(self):
        return self._import_params

    @property
    def footprint_params(self):
        return self._footprint_params

    def fit(self, scan_number: int, trim_front: int = None, trim_back: int = None, theta_offset: float = 0.0,
            dq: float = 0.0, factor: float = 1.0, plot=False, polish=True) -> FitResult:
        raise NotImplementedError

    def fit_range(self, scan_range: range, trim_front: int = None, trim_back: int = None, theta_offset: float = 0.0,
                  dq: float = 0.0, factor: float = 1.0, plot=False, polish=True) -> FitResultSeries:
        raise NotImplementedError

    def show_scans(self, min_scan: int = None, max_scan: int = None):
        """Show information about all scans from `min_scan` to `max_scan`."""
        raise NotImplementedError

    def set_file(self, file_path: str):
        """Define the full path of the file from which the scans are read."""
        raise NotImplementedError

    def set_trained_model(self, trained_model: TrainedModel = None, model_path: str = None):
        """Set the trained Keras model either as an object or as a path to a saved hdf5 file."""
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

    def set_import_params(self, *args, **kwargs):
        raise NotImplementedError

    def set_footprint_params(self, *args, **kwargs):
        raise NotImplementedError