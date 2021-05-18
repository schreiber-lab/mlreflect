import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xrrloader import SpecLoader

from . import CurveFitter
from ..models import TrainedModel, DefaultTrainedModel


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

    def fit(self, scan_number: int, trim_front: int = 0, dq: float = 0.0, factor: float = 1.0, plot=False):
        """Extract scan from SPEC file and predict thin film parameters.

        Args:
            scan_number: SPEC scan number of the scan that is to be fitted.
            trim_front: How many intensity points are cropped from the beginning.
            dq: Q-shift that is applied before interpolation of the data to the trained q values. Can sometimes
                improve the results if the total reflection edge is not perfectly aligned.
            factor: Multiplicative factor that is applied to the data after interpolation. Can sometimes
                improve the results if the total reflection edge is not perfectly aligned.
            plot: If set to `True`, the intensity prediction is shown in a plot.

        Returns:
            output_dict = {
                'corrected_intensity': Extracted scan intensity with footprint correction
                'q_values_input': Corresponding q values for the extracted scan in 1/A,
                'predicted_intensity': Predicted simulated intensity,
                'q_values_prediction': Corresponding q values for the predicted intensity in 1/A,
                'predicted_labels': Predicted thin film parameters as Pandas DataFrame
            }
        """
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
            'q_values_prediction': self._trained_model.q_values - dq,
            'predicted_labels': predicted_labels
        }
        if plot:
            self._plot_prediction(**output)
        return output

    def fit_range(self, scan_range: range, trim_front: int = 0, dq: float = 0.0, factor: float = 1.0):
        """Iterate fit method over a range of scans."""
        total_output = {
            'corrected_intensity': [],
            'q_values_input': [],
            'predicted_intensity': [],
            'q_values_prediction': [],
            'predicted_labels': []
        }
        for i in scan_range:
            output = self.fit(i, trim_front=trim_front, dq=dq, factor=factor, plot=False)
            for key in output.keys():
                total_output[key].append(output[key])

        for key in ['corrected_intensity', 'q_values_input', 'predicted_intensity', 'q_values_prediction']:
            total_output[key] = np.vstack(total_output[key])
        total_output['predicted_labels'] = pd.concat(total_output['predicted_labels']).reset_index(drop=True)

        return total_output

    @staticmethod
    def _plot_prediction(q_values_input, corrected_intensity, q_values_prediction, predicted_intensity,
                         predicted_labels):
        min_q_idx = np.argmin(q_values_prediction)
        max_q_idx = np.argmax(q_values_prediction)

        plt.semilogy(q_values_input[min_q_idx:max_q_idx], corrected_intensity[min_q_idx:max_q_idx], 'o', label='data')
        plt.semilogy(q_values_prediction, predicted_intensity, label='prediction')
        plt.title('Prediction')
        plt.xlabel('q [1/A]')
        plt.ylabel('Intensity [a.u.]')
        plt.legend()

        labels = ['Film_thickness', 'Film_roughness', 'Film_sld', 'SiOx_thickness']
        units = {
            'Film_thickness': 'Å',
            'Film_roughness': 'Å',
            'Film_sld': 'x 10$^{-6}$ Å$^{-2}$',
            'SiOx_thickness': 'Å'
        }
        predictions = ''
        for label in labels:
            predictions += f'{label}: {float(predicted_labels[label]):0.1f} {units[label]}\n'
        plt.annotate(predictions, (0.6, 0.75), xycoords='axes fraction', va='top', ha='left')

        plt.show()

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
