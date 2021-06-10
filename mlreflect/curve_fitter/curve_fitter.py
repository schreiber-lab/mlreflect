import warnings

import numpy as np
import pandas as pd
from numpy import ndarray

from .minimizer import least_log_mean_squares_fit
from ..data_generation import ReflectivityGenerator
from ..models import TrainedModel
from ..training import InputPreprocessor, OutputPreprocessor


class CurveFitter:
    """Make a prediction on specular reflectivity data based on the trained model.

    Args:
        trained_model: TrainedModel object that contains the trained Keras model, the trained q values, the
            standardization values and the sample structure.
    """

    def __init__(self, trained_model: TrainedModel):
        self.trained_model = trained_model

        self.generator = ReflectivityGenerator(trained_model.q_values, trained_model.sample)

        self.ip = InputPreprocessor()
        self.ip._standard_std = trained_model.ip_std
        self.ip._standard_mean = trained_model.ip_mean

        self.op = OutputPreprocessor(trained_model.sample, 'min_to_zero')

    def fit_curve(self, corrected_curve: ndarray, q_values: ndarray, dq: float = 0, factor: float = 1, polish=False):
        """Return predicted reflectivity and thin film properties based footprint-corrected data.

        Args:
            corrected_curve: "Ideal" reflectivity curve that has already been treated with footprint correction and
                other intensity corrections and is normalized to 1.
            q_values: Corresponding q values for each of the intensity values in units of 1/A.
            dq: Q-shift that is applied before interpolation of the data to the trained q values. Can sometimes
                improve the results if the total reflection edge is not perfectly aligned.
            factor: Multiplicative factor that is applied to the data after interpolation. Can sometimes
                improve the results if the total reflection edge is not perfectly aligned.
            polish: If `True`, the predictions will be refined with a simple least log mean squares minimization via
                `scipy.optimize.minimize`. This can often improve the "fit" of the model curve to the data at the
                expense of higher prediction times.

        Returns:
            predicted_reflectivity: Numpy array of the predicted intensity.
            predicted_parameters: Pandas DataFrame of the predicted thin film parameters.
        """
        corrected_curve = self._interpolate_intensity(corrected_curve * factor, q_values + dq)

        predicted_parameters = self.trained_model.keras_model.predict(
            self.ip.standardize(np.atleast_2d(corrected_curve)))

        restored_predicted_parameters = self.op.restore_labels(predicted_parameters)
        self._ensure_positive_parameters(restored_predicted_parameters)

        if polish:
            polished_parameters = []
            for i in range(len(corrected_curve)):
                polished_parameters.append(least_log_mean_squares_fit(corrected_curve[i],
                                                                      restored_predicted_parameters[i:(i+1)],
                                                                      self.generator, self.op))
            polished_parameters = pd.concat(polished_parameters).reset_index(drop=True)
            self._ensure_positive_parameters(polished_parameters)
            predicted_refl = self.generator.simulate_reflectivity(polished_parameters, progress_bar=False)
            return predicted_refl, polished_parameters
        else:
            predicted_refl = self.generator.simulate_reflectivity(restored_predicted_parameters, progress_bar=False)
            return predicted_refl, restored_predicted_parameters

    def _interpolate_intensity(self, intensity: ndarray, q_values: ndarray):
        warnings.filterwarnings('ignore')
        intensity = np.atleast_2d(intensity)
        interp_intensity = np.empty((len(intensity), len(self.trained_model.q_values)))
        for i in range(len(intensity)):
            interp_intensity[i] = 10 ** np.interp(self.trained_model.q_values, q_values, np.log10(intensity[i]))
        return interp_intensity

    @staticmethod
    def _ensure_positive_parameters(parameters):
        for parameter_name in parameters.columns:
            if 'thickness' in parameter_name or 'roughness' in parameter_name:
                parameters[parameter_name] = abs(parameters[parameter_name])
