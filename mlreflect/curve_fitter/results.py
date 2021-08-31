import datetime
import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .minimizer import mean_squared_error
from ..data_generation import ReflectivityGenerator, interp_reflectivity


class FitResult:
    FORMAT = '%.5e'

    def __init__(self, scan_number, corrected_reflectivity, q_values_input, predicted_reflectivity, q_values_prediction,
                 predicted_parameters, best_q_shift, sample, timestamp: str = None):
        """A class to store prediction results in one object and allow easy plotting and saving of the results."""
        self.scan_number = int(scan_number)
        self.timestamp = timestamp
        self.corrected_reflectivity = corrected_reflectivity
        self.q_values_input = q_values_input
        self.predicted_reflectivity = predicted_reflectivity
        self.q_values_prediction = q_values_prediction
        predicted_parameters.index = [scan_number]
        predicted_parameters.index.name = 'scan'
        self.predicted_parameters = predicted_parameters
        self.best_q_shift = best_q_shift
        self.sample = sample

    @property
    def sld_profile(self):
        generator = ReflectivityGenerator(self.q_values_prediction, self.sample)
        return generator.simulate_sld_profiles(self.predicted_parameters, progress_bar=False)[0]

    @property
    def interpolated_corrected_reflectivity(self):
        return interp_reflectivity(self.q_values_prediction, self.q_values_input, self.corrected_reflectivity)

    @property
    def curve_mse(self):
        return mean_squared_error(np.log10(self.predicted_reflectivity),
                                  np.log10(self.corrected_reflectivity)).round(decimals=6)[0]

    def save_predicted_parameters(self, path: str, delimiter='\t'):
        """Save all predicted parameters in a text file with the given delimiter."""
        self.predicted_parameters.to_csv(path, sep=delimiter)

    def save_predicted_reflectivity(self, path: str):
        """Save the predicted reflectivity in a text file with the first column being the q values."""
        output = np.zeros((len(self.q_values_input), 2))
        output[:, 0] = self.q_values_input
        output[:, 1] = self.predicted_reflectivity
        np.savetxt(path, output, delimiter='\t', fmt=self.FORMAT)

    def save_corrected_reflectivity(self, path: str):
        """Save the measured and corrected reflectivity in a text file with the first column being the q values."""
        output = np.zeros((len(self.q_values_input), 2))
        output[:, 0] = self.q_values_input
        output[:, 1] = self.corrected_reflectivity
        np.savetxt(path, output, delimiter='\t', fmt=self.FORMAT)

    def plot_prediction(self, parameters: list):
        """Plot the corrected data and the predicted reflectivity curve and print the predictions for ``parameters``."""

        plt.semilogy(self.q_values_input,
                     self.corrected_reflectivity, 'o', label='data')
        plt.semilogy(self.q_values_input, self.predicted_reflectivity, label='prediction')
        plt.title(f'Prediction of scan #{self.scan_number} ({self.timestamp})')
        plt.xlabel('q [1/A]')
        plt.ylabel('Intensity [a.u.]')
        plt.legend()

        self._annotate_plot(parameters)

        plt.show()

    def _annotate_plot(self, parameters):
        predictions = ''
        for param in parameters:
            if 'thickness' in param or 'roughness' in param:
                unit = 'Å'
            elif 'sld' in param:
                unit = 'x 10$^{-6}$ Å$^{-2}$'
            else:
                raise ValueError(f"couldn't determine unit for parameter {param} (must be thickness, roughness or sld)")
            predictions += f'{param}: {float(self.predicted_parameters[param]):0.1f} {unit}\n'

        predictions += f'log_mse: {self.curve_mse}'
        plt.annotate(predictions, (0.6, 0.75), xycoords='axes fraction', va='top', ha='left')

    def plot_sld_profile(self):
        """Plots the SLD profile of the predicted parameters."""
        profile = self.sld_profile

        plt.plot(profile[0], profile[1])
        plt.ylabel('SLD [10$^{-6}$ Å$^{-2}$]')
        plt.xlabel('Sample height [Å]')
        plt.title('Scattering length density profile')
        plt.show()


class FitResultSeries:
    def __init__(self, fit_results_list):
        self.fit_results_list = fit_results_list
        self.scan_number = [fit_result.scan_number for fit_result in fit_results_list]
        self.timestamp = [fit_result.timestamp for fit_result in fit_results_list]
        self.corrected_reflectivity = np.array([fit_result.corrected_reflectivity for fit_result in fit_results_list])
        self.q_values_input = np.array([fit_result.q_values_input for fit_result in fit_results_list])
        self.predicted_reflectivity = np.array([fit_result.predicted_reflectivity for fit_result in fit_results_list])
        self.q_values_prediction = np.array([fit_result.q_values_prediction for fit_result in fit_results_list])
        self.predicted_parameters = pd.concat(
            [fit_result.predicted_parameters for fit_result in fit_results_list])
        self.best_q_shift = np.array([fit_result.best_q_shift for fit_result in self.fit_results_list])
        self.sample = fit_results_list[0].sample

    @property
    def delta_t(self):
        if None in self.timestamp:
            return None
        else:
            datetime_list = [datetime.datetime.strptime(t, '%a %b %d %H:%M:%S %Y') for t in self.timestamp]
            return np.array([(dt - datetime_list[0]).total_seconds() / 60 for dt in datetime_list])

    @property
    def curve_mse(self):
        return np.array([fit_result.curve_variant_log_mse for fit_result in self.fit_results_list])

    @property
    def sld_profiles(self):
        generator = ReflectivityGenerator(self.q_values_prediction, self.sample)
        return generator.simulate_sld_profiles(self.predicted_parameters, progress_bar=False)

    def save_predicted_parameters(self, path: str, delimiter='\t'):
        """Save all predicted parameters in a text file with the given delimiter."""
        self.predicted_parameters.to_csv(path, sep=delimiter)

    def save_predicted_reflectivity(self, path: str):
        """Save the predicted reflectivity in a text file with the first column being the q values."""
        for fit_result in self.fit_results_list:
            save_path = os.path.join(os.path.dirname(path), f'scan{fit_result.scan_number}_{os.path.basename(path)}')
            fit_result.save_predicted_reflectivity(save_path)

    def save_corrected_reflectivity(self, path: str):
        """Save the measured and corrected reflectivity in a text file with the first column being the q values."""
        for fit_result in self.fit_results_list:
            save_path = os.path.join(os.path.dirname(path), f'scan{fit_result.scan_number}_{os.path.basename(path)}')
            fit_result.save_corrected_reflectivity(save_path)

    def plot_predicted_parameter_range(self, parameters: list, x_format='time'):
        """Plot predicted parameters in `parameters` against scan number or time (relative to the first scan).
        Args:
            parameters: List of strings of which parameters are plotted. Possible values are ``'thickness'``,
            ``'roughness'`` or ``'sld'``.
            x_format: If ``x_format='time'`` (default), the x axis will be formatted using the timestamps of each scan.
            If ``x_format='scan'``, the x axis will show the scan numbers instead. If no timestamps are available it
            will always use ``'scan'``.

        """
        n_labels = len(parameters)
        if x_format == 'time' and self.delta_t is not None:
            x = self.delta_t
            x_label = 'Time [min]'
        else:
            x = self.scan_number
            x_label = 'Scan'

        fig = plt.figure(figsize=(5, 10))

        for i, param in enumerate(parameters):
            if 'thickness' in param or 'roughness' in param:
                unit = 'Å'
            elif 'sld' in param:
                unit = '10$^{-6}$ Å$^{-2}$'
            else:
                raise ValueError(f"couldn't determine unit for parameter {param} (must be thickness, roughness or sld)")
            plt.subplot(n_labels, 1, i + 1)
            plt.plot(x, self.predicted_parameters[param])
            plt.xlabel(x_label)
            if x_label == 'Scan':
                ax = fig.gca()
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.ylabel(f'{param} [{unit}]')
            plt.tight_layout()

        plt.show()

    def plot_sld_profiles(self):
        """Plots the SLD profiles of the predicted parameters."""
        profiles = self.sld_profiles

        n_profiles = len(profiles)
        min_scan = self.predicted_parameters.index[0]
        max_scan = self.predicted_parameters.index[-1]

        colormap = plt.cm.get_cmap('Spectral')
        colors = colormap(np.linspace(0, 1, n_profiles))

        fig = plt.figure(figsize=(6, 3))

        for i in range(len(self.predicted_parameters)):
            plt.plot(profiles[i][0], profiles[i][1], color=colors[i])
        plt.ylabel('SLD [10$^{-6}$ Å$^{-2}$]')
        plt.xlabel('Sample height [Å]')
        plt.title('Scattering length density profile')
        colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=colormap,
                                                      norm=matplotlib.colors.Normalize(vmin=min_scan, vmax=max_scan)))
        colorbar.set_label('scan', rotation=-90, labelpad=20)
        plt.tight_layout()
        plt.show()
