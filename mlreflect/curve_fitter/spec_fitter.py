import numpy as np

from . import CurveFitter
from .results import FitResult, FitResultSeries
from ..models import TrainedModel, DefaultTrainedModel
from ..xrrloader import SpecLoader


class SpecFitter:
    """Load reflectivity scans from a SPEC file and fit them using a trained neural network model.

    Before use:
        - A neural network model has to be set via the set_trained_model() method.
        - Import parameters have to be defined via the set_import_params() method.
        - Parameters for footprint correction have to be defined via set_footprint_params() method.
        - The input SPEC file has to be specified via the set_spec_file() method.
    """

    def __init__(self):
        self._trained_model = None
        self._curve_fitter = None

        self._spec_file = None
        self._import_params = {}
        self._footprint_params = {}
        self._spec_loader = None

    @property
    def trained_model(self):
        return self._trained_model

    @property
    def spec_file(self):
        return self._spec_file

    @property
    def import_params(self):
        return self._import_params

    @property
    def footprint_params(self):
        return self._footprint_params

    def fit(self, scan_number: int, trim_front: int = None, trim_back: int = None, theta_offset: float = 0.0,
            dq: float = 0.0, factor: float = 1.0, plot=False, polish=True) -> FitResult:
        """Extract scan from SPEC file and predict thin film parameters.

        Args:
            scan_number: SPEC scan number of the scan that is to be fitted.
            trim_front: How many intensity points are cropped from the beginning.
            trim_back: How many intensity points are cropped from the end.
            theta_offset: Angular correction that is added before transformation to q space.
            dq: Q-shift that is applied before interpolation of the data to the trained q values. Can sometimes
                improve the results if the total reflection edge is not perfectly aligned.
            factor: Multiplicative factor that is applied to the data after interpolation. Can sometimes
                improve the results if the total reflection edge is not perfectly aligned.
            plot: If set to `True`, the intensity prediction is shown in a plot.
            polish: If `True`, the predictions will be refined with a simple least log mean squares minimization via
                `scipy.optimize.minimize`. This can often improve the "fit" of the model curve to the data at the
                expense of higher prediction times.

        Returns:
            FitResult
        """
        try:
            scan = self._spec_loader.load_scan(scan_number=scan_number, trim_front=trim_front, trim_back=trim_back)
        except KeyError:
            print(f'scan {scan_number} could not be found in {self._spec_file}')
            return
        scan.scattering_angle += theta_offset
        predicted_refl, predicted_parameters = self._curve_fitter.fit_curve(scan.corrected_intensity, scan.q,
                                                                            dq, factor, polish)

        fit_result = FitResult(scan_number=scan_number,
                               timestamp=scan.timestamp,
                               corrected_reflectivity=scan.corrected_intensity,
                               q_values_input=scan.q,
                               predicted_reflectivity=predicted_refl[0],
                               q_values_prediction=self._trained_model.q_values - dq,
                               predicted_parameters=predicted_parameters,
                               sample=self._trained_model.sample)
        if plot:
            parameters = [self.trained_model.sample.layers[-1].name + param for param in ('_thickness', '_roughness',
                                                                                          '_sld')]
            fit_result.plot_prediction(parameters)
            fit_result.plot_sld_profile()
        return fit_result

    def fit_range(self, scan_range: range, trim_front: int = None, trim_back: int = None, theta_offset: float = 0.0,
                  dq: float = 0.0, factor: float = 1.0, plot=False, polish=True) -> FitResultSeries:
        """Iterate fit method over a range of scans."""

        fit_results = []
        for i in scan_range:
            fit_results.append(self.fit(i, trim_front=trim_front, trim_back=trim_back, theta_offset=theta_offset, dq=dq,
                                        factor=factor, plot=False, polish=polish))

        fit_result_series = FitResultSeries(fit_results)

        if plot:
            fit_result_series.plot_sld_profiles()
            parameters = [self.trained_model.sample.layers[-1].name + param for param in ('_thickness', '_roughness',
                                                                                          '_sld')]
            fit_result_series.plot_predicted_parameter_range(parameters)

        return fit_result_series

    def show_scans(self, min_scan: int = None, max_scan: int = None):
        """Show information about all scans from `min_scan` to `max_scan`."""
        parser = self._spec_loader.parser
        if max_scan is None:
            max_scan = np.max(np.asarray(list(parser.scan_info.keys()), dtype=int))
        else:
            max_scan = np.min((max_scan, np.max(np.asarray(list(parser.scan_info.keys()), dtype=int))))
        if min_scan is None:
            min_scan = np.min(np.asarray(list(parser.scan_info.keys()), dtype=int))
        else:
            min_scan = np.max((min_scan, np.min(np.asarray(list(parser.scan_info.keys()), dtype=int))))
        for i in range(min_scan, max_scan + 1):
            try:
                out = f'scan #{i}\n' \
                      f'\tcommand: {parser.scan_info[str(i)]["spec_command"]}\n' \
                      f'\ttime: {parser.scan_info[str(i)]["time"]}'
                print(out)
            except KeyError:
                print(f'scan #{i}\n\tnot found')

    def set_spec_file(self, spec_file_path: str):
        """Define the full path of the SPEC file from which the scans are read."""
        self._spec_loader = SpecLoader(spec_file_path, **self._import_params, **self._footprint_params)
        self._spec_file = spec_file_path

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

    def set_import_params(self, angle_columns: list, intensity_column: str, attenuator_column: str = None,
                          division_column: str = None):
        """Set the parameters necessary to correctly import the scans from the SPEC file.

        Args:
            angle_columns: List of SPEC counters that are summed up to form the full scattering angle (2theta).
            intensity_column: SPEC counter from which the intensity is extracted from.
            attenuator_column: SPEC counter of the applied attenuator used to correct possible kinks in the data.
            division_column: Optional SPEC counter that is used to divide the intensity counter by.
        """
        params = {
            'angle_columns': angle_columns,
            'intensity_column': intensity_column,
            'attenuator_column': attenuator_column,
            'division_column': division_column
        }
        self._import_params.update(params)

    def set_footprint_params(self, wavelength: float, sample_length: float, beam_width: float,
                             beam_shape: str = 'gauss', normalize_to: str = 'max'):
        """Set the parameters necessary to apply footprint correction.

        Args:
            wavelength: Photon wavelength in Angstroms.
            sample_length: Sample length along the beam direction in mm.
            beam_width: Beam width along the beam direction (height). For a gaussian beam profile this is the full
                width at half maximum.
            beam_shape:
                'gauss' (default) for a gaussian beam profile
                'box' for a box profile
            normalize_to:
                'max' (default): normalize data by the highest intensity value
                'first': normalize data by the first intensity value
        """

        params = {
            'wavelength': wavelength,
            'beam_width': beam_width,
            'sample_length': sample_length,
            'beam_shape': beam_shape,
            'normalize_to': normalize_to
        }

        self._footprint_params.update(params)


class DefaultSpecFitter(SpecFitter):
    """SpecFitter that is initialized with a pre-trained model for reflectivity on single-layer systems on Si/SiOx."""

    def __init__(self):
        super().__init__()
        self.set_trained_model(DefaultTrainedModel())
