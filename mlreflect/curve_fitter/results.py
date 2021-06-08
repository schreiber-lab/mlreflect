import datetime
import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data_generation import ReflectivityGenerator


class FitResult:
    FORMAT = '%.5e'

    def __init__(self, scan_number, corrected_reflectivity, q_values_input, predicted_reflectivity, q_values_prediction,
                 predicted_parameters, sample, timestamp: str = None):
        """A class to store prediction results in one object and allow easy plotting and saving of the results."""
        self.scan_number = scan_number
        self.timestamp = timestamp
        self.corrected_reflectivity = corrected_reflectivity
        self.q_values_input = q_values_input
        self.predicted_reflectivity = predicted_reflectivity
        self.q_values_prediction = q_values_prediction
        predicted_parameters.index = [scan_number]
        predicted_parameters.index.name = 'scan'
        self.predicted_parameters = predicted_parameters
        self.sample = sample

    def save_predicted_parameters(self, path: str, delimiter='\t'):
        """Save all predicted parameters in a text file with the given delimiter."""
        self.predicted_parameters.to_csv(path, sep=delimiter)

    def save_predicted_reflectivity(self, path: str):
        """Save the predicted reflectivity in a text file with the first column being the q values."""
        output = np.zeros((len(self.q_values_prediction), 2))
        output[:, 0] = self.q_values_prediction
        output[:, 1] = self.predicted_reflectivity
        np.savetxt(path, output, delimiter='\t', fmt=self.FORMAT)

    def save_corrected_reflectivity(self, path: str):
        """Save the measured and corrected reflectivity in a text file with the first column being the q values."""
        output = np.zeros((len(self.q_values_input), 2))
        output[:, 0] = self.q_values_input
        output[:, 1] = self.corrected_reflectivity
        np.savetxt(path, output, delimiter='\t', fmt=self.FORMAT)

    def plot_prediction(self, parameters: list):
        """Plot the corrected data and the predicted reflectivity curve and print the predictions for `parameters`."""
        min_q_idx = np.argmin(self.q_values_prediction)
        max_q_idx = np.argmax(self.q_values_prediction)

        plt.semilogy(self.q_values_input[min_q_idx:max_q_idx],
                     self.corrected_reflectivity[min_q_idx:max_q_idx], 'o', label='data')
        plt.semilogy(self.q_values_prediction, self.predicted_reflectivity, label='prediction')
        plt.title(f'Prediction of scan #{self.scan_number} ({self.timestamp})')
        plt.xlabel('q [1/A]')
        plt.ylabel('Intensity [a.u.]')
        plt.legend()

        predictions = ''
        for param in parameters:
            if 'thickness' in param or 'roughness' in param:
                unit = 'Å'
            elif 'sld' in param:
                unit = 'x 10$^{-6}$ Å$^{-2}$'
            else:
                raise ValueError(f"couldn't determine unit for parameter {param} (must be thickness, roughness or sld)")
            predictions += f'{param}: {float(self.predicted_parameters[param]):0.1f} {unit}\n'
        plt.annotate(predictions, (0.6, 0.75), xycoords='axes fraction', va='top', ha='left')

        plt.show()

    def plot_sld_profile(self):
        """Plots the SLD profile of the predicted parameters."""
        generator = ReflectivityGenerator(self.q_values_prediction, self.sample)
        profile = generator.simulate_sld_profiles(self.predicted_parameters, progress_bar=False)[0]

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
        self.sample = fit_results_list[0].sample

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

    def plot_predicted_parameter_range(self, parameters: list):
        """Plot predicted parameters in `parameters` against time (relative to the first scan)."""
        n_labels = len(parameters)

        datetime_list = [datetime.datetime.strptime(t, '%a %b %d %H:%M:%S %Y') for t in self.timestamp]
        delta_t = np.array([(dt - datetime_list[0]).total_seconds() / 60 for dt in datetime_list])

        fig = plt.figure(figsize=(5, 10))

        for i, param in enumerate(parameters):
            if 'thickness' in param or 'roughness' in param:
                unit = 'Å'
            elif 'sld' in param:
                unit = '10$^{-6}$ Å$^{-2}$'
            else:
                raise ValueError(f"couldn't determine unit for parameter {param} (must be thickness, roughness or sld)")
            plt.subplot(n_labels, 1, i + 1)
            plt.plot(delta_t, self.predicted_parameters[param])
            plt.xlabel('Time [min]')
            plt.ylabel(f'{param} [{unit}]')
            plt.tight_layout()

        plt.show()

    def plot_sld_profiles(self):
        """Plots the SLD profiles of the predicted parameters."""
        generator = ReflectivityGenerator(self.q_values_prediction, self.sample)
        profiles = generator.simulate_sld_profiles(self.predicted_parameters, progress_bar=False)

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
