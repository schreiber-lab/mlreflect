import warnings

import numpy as np
import pandas as pd
from numpy import ndarray

from .minimizer import least_log_mean_squares_fit, q_shift_variants, curve_variant_log_mse, curve_scaling_variants
from ..data_generation import ReflectivityGenerator, interp_reflectivity
from ..models import TrainedModel
from ..training import InputPreprocessor, OutputPreprocessor


class CurveFitter:
    """Make a prediction on specular reflectivity data based on the trained model.

    Args:
        trained_model: :class:`TrainedModel` object that contains the trained Keras model, the trained q values, the
            standardization values and the sample structure.
    """

    def __init__(self, trained_model: TrainedModel):
        self.trained_model = trained_model

        self.generator = None

        self.ip = InputPreprocessor()
        self.ip._standard_std = trained_model.ip_std
        self.ip._standard_mean = trained_model.ip_mean

        self.op = OutputPreprocessor(trained_model.sample, 'min_to_zero')

    def fit_curve(self, corrected_curve: ndarray, q_values: ndarray, dq: float = 0, factor: float = 1, polish=False,
                  fraction_bounds: tuple = (0.5, 0.5, 0.1), optimize_q=True, n_q_samples: int = 1000,
                  optimize_scaling=False, n_scale_samples: int = 300, simulate_reflectivity=True) -> dict:
        """Return predicted reflectivity and thin film properties based footprint-corrected data.

        Args:
            corrected_curve: "Ideal" reflectivity curve that has already been treated with footprint correction and
                other intensity corrections and is normalized to 1.
            q_values: Corresponding q values for each of the intensity values in units of 1/A.
            dq: Q-shift that is applied before interpolation of the data to the trained q values. Can sometimes
                improve the results if the total reflection edge is not perfectly aligned.
            factor: Multiplicative factor that is applied to the data after interpolation. Can sometimes
                improve the results if the total reflection edge is not perfectly aligned.
            polish: If ``True``, the predictions will be refined with a simple least log mean squares minimization via
                ``scipy.optimize.minimize``. This can often improve the "fit" of the model curve to the data at the
                expense of higher prediction times.
            fraction_bounds: The relative fitting bounds if the LMS for thickness, roughness and SLD, respectively.
                E.g. if the predicted thickness was 150 A, then a value of 0.5 would mean the fit bounds are
                ``(75, 225)``.
            optimize_q: If ``True``, the q interpolation will be resampled with small q shifts in a range of about
                +-0.003 1/A and the neural network prediction with the smallest MSE will be selected. If
                ``polish=True``, this step will happen before the LMS fit.
            n_q_samples: Number of q shift samples that will be generated. More samples can lead to a better result,
                but will increase the prediction time.
            optimize_scaling: If ``True``, the interpolated input curve is randomly rescaled by a factor between 0.9
                and 1.1 and the neural network prediction with the smallest MSE will be selected. If ``polish=True``,
                this step will happen before the LMS fit. If ``optimize_q=True``, this will step will happen after
                the q shift optimization.
            n_scale_samples: Number of curve scaling samples that will be generated. More samples can lead to a better
                result, but will increase the prediction time.
            simulate_reflectivity: If ``True`` (default), the reflectivity according to the predicted parameter
                values will be simulated as well. This might slow down the prediction times.

        Returns:
            :class:`dict`: A dictionary containing the fit results:
                ``'predicted_reflectivity'``: Numpy :class:`ndarray` of the predicted intensity.
                ``'predicted_parameters'``: Pandas :class:`DataFrame` of the predicted thin film parameters.
                ``'best_shift'``: Q shift that lead to the prediction with the lowest MSE. Is ``None``
                if ``optimize_q=False``.
                ``'best_scaling'``: Curve scaling factor that lead to the prediction with the lowest MSE. Is ``None``
                if ``optimize_scaling=False``.

        """
        max_q_idx = abs(q_values - self.trained_model.q_values.max()).argmin() + 1
        min_q_idx = abs(q_values - self.trained_model.q_values.min()).argmin()

        corrected_curve = np.atleast_2d(corrected_curve)
        interpolated_curve = self._interpolate_intensity(corrected_curve * factor, q_values + dq)
        generator = ReflectivityGenerator(q_values, self.trained_model.sample)

        n_curves = len(corrected_curve)
        if optimize_q:
            restored_predicted_parameters = []
            predicted_refl = np.empty_like(corrected_curve)
            best_q_shift = np.empty(n_curves)
            for i, curve in enumerate(corrected_curve):
                best_q_output = self._optimize_q(self.trained_model.q_values, q_values, curve,
                                                 generator, n_q_samples)
                restored_predicted_parameters.append(best_q_output['best_prediction'])
                predicted_refl[i] = best_q_output['best_predicted_curve']
                best_q_shift[i] = best_q_output['best_shift']
            restored_predicted_parameters = pd.concat(restored_predicted_parameters).reset_index(drop=True)
        else:
            best_q_shift = None

        if optimize_scaling:
            restored_predicted_parameters = []
            scaled_predicted_refl = np.empty(corrected_curve.shape)
            best_scaling = np.empty(n_curves)
            for i, curve in enumerate(corrected_curve):
                if best_q_shift is None:
                    dq = 0
                else:
                    dq = best_q_shift[i]
                best_scaling_output = self._optimize_scaling(q_values, min_q_idx, max_q_idx, curve, n_scale_samples,
                                                             dq=dq)
                restored_predicted_parameters.append(best_scaling_output['best_prediction'])
                scaled_predicted_refl[i] = best_scaling_output['best_predicted_curve']
                best_scaling[i] = best_scaling_output['best_scaling']
            restored_predicted_parameters = pd.concat(restored_predicted_parameters).reset_index(drop=True)
        else:
            best_scaling = None

        if not (optimize_scaling & optimize_q):
            predicted_parameters = self.trained_model.keras_model.predict(
                self.ip.standardize(np.atleast_2d(interpolated_curve)))

            restored_predicted_parameters = self.op.restore_labels(predicted_parameters)
            self._ensure_positive_parameters(restored_predicted_parameters)

        if polish:
            polished_parameters = []
            for i in range(len(interpolated_curve)):
                polished_parameters.append(least_log_mean_squares_fit(q_values[min_q_idx:max_q_idx],
                                                                      corrected_curve[i, min_q_idx:max_q_idx],
                                                                      restored_predicted_parameters[i:(i + 1)],
                                                                      self.trained_model.sample, self.op,
                                                                      fraction_bounds))
            polished_parameters = pd.concat(polished_parameters).reset_index(drop=True)
            self._ensure_positive_parameters(polished_parameters)
            if simulate_reflectivity:
                predicted_refl = generator.simulate_reflectivity(polished_parameters, progress_bar=False)
            else:
                predicted_refl = None
            return {'predicted_reflectivity': predicted_refl, 'predicted_parameters': polished_parameters,
                    'best_q_shift': best_q_shift, 'best_scaling': best_scaling}
        else:
            if simulate_reflectivity:
                predicted_refl = generator.simulate_reflectivity(restored_predicted_parameters, progress_bar=False)
            else:
                predicted_refl = None
            return {'predicted_reflectivity': predicted_refl, 'predicted_parameters': restored_predicted_parameters,
                    'best_q_shift': best_q_shift, 'best_scaling': best_scaling}

    def _optimize_q(self, q_values_prediction, q_values_input, corrected_reflectivity, generator,
                    n_variants=300, scale=0.001):
        q_shift_curves, shifts = q_shift_variants(q_values_prediction, q_values_input, corrected_reflectivity,
                                                  n_variants, scale=scale)

        shift_predictions = self.fit_curve(q_shift_curves, q_values_prediction, polish=False, optimize_q=False,
                                           optimize_scaling=False)

        interpolated_reflectivity = self._interpolate_intensity(corrected_reflectivity, q_values_input)

        shift_mse = curve_variant_log_mse(interpolated_reflectivity,
                                          shift_predictions['predicted_reflectivity'])

        min_mse_idx = shift_mse.argmin()

        best_prediction = shift_predictions['predicted_parameters'][min_mse_idx:min_mse_idx + 1]
        best_predicted_curve = generator.simulate_reflectivity(best_prediction, progress_bar=False)[0]

        return {'best_shift': shifts[min_mse_idx][0],
                'best_prediction': best_prediction,
                'best_predicted_curve': best_predicted_curve}

    def _optimize_scaling(self, q_values, min_q_idx, max_q_idx, corrected_intensity, n_variants=300, scale=0.1, dq=0):
        scaled_curve_variants, scalings = curve_scaling_variants(corrected_intensity, n_variants, scale)
        scaled_predictions = self.fit_curve(scaled_curve_variants, q_values, polish=False, optimize_q=False,
                                            optimize_scaling=False, dq=dq)
        scaling_mse = curve_variant_log_mse(corrected_intensity[min_q_idx:max_q_idx],
                                            scaled_predictions['predicted_reflectivity'][:, min_q_idx:max_q_idx])

        min_mse_idx = scaling_mse.argmin()
        return {'best_scaling': scalings[min_mse_idx][0],
                'best_prediction': scaled_predictions['predicted_parameters'][min_mse_idx:min_mse_idx + 1],
                'best_predicted_curve': scaled_predictions['predicted_reflectivity'][min_mse_idx]}

    def _interpolate_intensity(self, intensity: ndarray, q_values: ndarray):
        warnings.filterwarnings('ignore')
        intensity = np.atleast_2d(intensity)
        interp_intensity = np.empty((len(intensity), len(self.trained_model.q_values)))
        for i in range(len(intensity)):
            interp_intensity[i] = interp_reflectivity(self.trained_model.q_values, q_values, intensity[i])
        return interp_intensity

    @staticmethod
    def _ensure_positive_parameters(parameters):
        for parameter_name in parameters.columns:
            if 'thickness' in parameter_name or 'roughness' in parameter_name:
                parameters[parameter_name] = abs(parameters[parameter_name])
