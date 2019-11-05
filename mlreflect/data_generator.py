from typing import Tuple, Iterable, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from .reflectivity import multilayer_reflectivity as refl
from .label_names import make_label_names, convert_to_ndarray
from .performance_tools import timer


class ReflectivityGenerator:
    """Generation of simulated reflectivity data and labels for neural network training.

    Args:
        q_values: An array-like object (list, tuple, ndarray, etc.) that contains the q-values in units of
            1/Å at which the reflected intensity will be simulated.
        ambient_sld: Scattering length density of the ambient environment above the top most layer in units of 1e+14
            1/Å^2, e.g. ~0 for air.
        random_seed: Random seed for numpy.random.seed which affects the generation of the random labels.

    Methods:
        generate_random_labels()
        simulate_reflectivity()

    Returns:
        TrainingData object.
    """

    def __init__(self, q_values: ndarray, ambient_sld: float, random_seed: int = 1):

        np.random.seed(random_seed)
        self.q_values = np.asarray(q_values)
        self.ambient_sld = ambient_sld

        self.q_noise_spread = 0
        self.shot_noise_spread = 0
        self.background_noise_base_level = 0
        self.background_noise_spread = 0
        self.slit_width = 0

    @timer
    def generate_random_labels(self, thickness_ranges: Iterable[Tuple[float, float]],
                               roughness_ranges: Iterable[Tuple[float, float]],
                               sld_ranges: Iterable[Tuple[float, float]],
                               number_of_samples: int, bolster_fraction: float = 0.15, bolster_width: float = 0.1) -> \
            DataFrame:
        """Generates random labels in the given parameter ranges and returns them as pandas DataFrame.

        Args:
            thickness_ranges: An array-like object (list, tuple, ndarray, etc.) that contains a tuple with the min and max
                thickness in units of Å for each sample layer in order from top to bottom. The thickness of the bottom most
                layer (substrate) is not relevant for the simulation, but some value must be provided, e.g. (1, 1).
            roughness_ranges: An array-like object (list, tuple, ndarray, etc.) that contains a tuple with the min and max
                roughness in units of Å for each sample interface in order from top (ambient/top layer) to bottom (bottom
                layer/substrate).
            sld_ranges: An array-like object (list, tuple, ndarray, etc.) that contains a tuple with the min and max
                scattering length density (SLD) in units of 1e+14 1/Å^2 for each sample layer in order from top to bottom
                (excluding the ambient SLD).
            number_of_samples: Number of label sets that will be generated.
            bolster_fraction: Fraction of simulated samples that will be redistributed to the ends of the distribution.
            bolster_width: Width of the Gaussian distribution of the redistributed samples.

        Returns:
            labels: Pandas DataFrame with the randomly generated labels.
        """

        thickness_ranges = np.asarray(thickness_ranges)
        roughness_ranges = np.asarray(roughness_ranges)
        sld_ranges = np.asarray(sld_ranges)

        number_of_layers = len(thickness_ranges)

        if not number_of_samples > 0:
            raise ValueError('`number_of_samples` must be at least 1.')

        label_names = make_label_names(number_of_layers)

        randomized_slds = self._generate_random_values(sld_ranges, number_of_samples, bolster_fraction, bolster_width)
        randomized_thicknesses = self._generate_random_values(thickness_ranges, number_of_samples, bolster_fraction,
                                                              bolster_width)
        randomized_roughnesses = self._generate_random_roughness_from_thickness(roughness_ranges,
                                                                                randomized_thicknesses)

        labels = np.concatenate((randomized_thicknesses, randomized_roughnesses, randomized_slds), axis=1)
        labels = pd.DataFrame(data=labels, columns=label_names)

        return labels

    @timer
    def simulate_reflectivity(self, labels: Union[DataFrame, ndarray]) -> ndarray:
        """Simulates reflectivity curves for the given labels and returns them as ndarray.

        Args:
            labels: Must be ndarray or DataFrame with each column representing one label. The label order from left to
            right must be "thickness", "roughness" and "scattering length density" with decreasing layer number
            within each label.
                Example for 2 layers: ['thickness_layer2', 'thickness_layer1', 'roughness_layer2', 'roughness_layer1',
                'sld_layer2', 'sld_layer1']

        Returns:
            reflectivity_curves: Simulated reflectivity curves.
        """
        labels = convert_to_ndarray(labels)

        number_of_q_values = len(self.q_values)
        number_of_curves = labels.shape[0]
        number_of_labels = labels.shape[1]
        number_of_layers = int(number_of_labels / 3)

        thicknesses = labels[:, :number_of_layers]
        roughnesses = labels[:, number_of_layers:2 * number_of_layers]
        slds = labels[:, 2 * number_of_layers:3 * number_of_layers]

        thicknesses_si = thicknesses * 1e-10
        roughnesses_si = roughnesses * 1e-10
        slds_si = slds * 1e14
        ambient_sld_si = self.ambient_sld * 1e14

        q_values_si = self.q_values * 1e10

        reflectivity_curves = np.zeros([number_of_curves, number_of_q_values])

        noisy_q_values = self._make_noisy_q_values(q_values_si, number_of_curves)

        for curve in tqdm(range(number_of_curves)):
            reflectivity = refl(noisy_q_values[curve, :], thicknesses_si[curve, :], roughnesses_si[curve, :],
                                slds_si[curve, :], ambient_sld_si)

            reflectivity_noisy = self._apply_shot_noise(reflectivity)
            reflectivity_noisy = self._apply_background_noise(reflectivity_noisy)

            reflectivity_curves[curve, :] = reflectivity_noisy

        return reflectivity_curves

    def _make_noisy_q_values(self, q_values: ndarray, number_of_curves: int) -> ndarray:
        percentage_deviation = np.random.normal(1, self.q_noise_spread, (number_of_curves, len(q_values)))
        return q_values * percentage_deviation

    def _apply_shot_noise(self, reflectivity_curve: ndarray) -> ndarray:
        noisy_reflectivity = np.clip(np.random.normal(reflectivity_curve, self.shot_noise_spread * np.sqrt(
            reflectivity_curve)), 1e-8, None)

        return noisy_reflectivity

    def _apply_background_noise(self, reflectivity_curve: ndarray) -> ndarray:
        num_q_values = len(reflectivity_curve)
        background = np.random.normal(self.background_noise_base_level, self.background_noise_spread, num_q_values)

        return reflectivity_curve + background

    # TODO This method is not yet finished and should only be used with slit_width = 0.
    def _apply_slit_convolution(self, q_values: ndarray, reflectivity_curve: ndarray) -> ndarray:
        raise NotImplementedError('slit convolution not implemented yet')
        sigma = self.slit_width
        if sigma == 0:
            return reflectivity_curve

        conv_reflectivity = np.zeros_like(reflectivity_curve)
        q_values /= np.max(q_values)
        for i in range(len(conv_reflectivity)):
            q_pos = q_values[i]
            g = self._gauss(q_values, sigma, q_pos)
            g_norm = g / sum(g)

            weighted_reflectivity = g_norm * reflectivity_curve
            conv_reflectivity[i] = sum(weighted_reflectivity)
        return conv_reflectivity

    @staticmethod
    def _gauss(x: ndarray, sigma: float = 1.0, mu: float = 0.0) -> ndarray:
        g = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        return g

    def _generate_random_values(self, label_ranges: ndarray, number_of_values: int, bolster_fraction: float,
                                bolster_width: float) -> ndarray:

        number_of_layers = label_ranges.shape[0]

        randomized_labels = np.zeros((number_of_values, number_of_layers))
        for layer_index in range(number_of_layers):
            layer_ranges = label_ranges[layer_index]

            if np.all(np.isreal(layer_ranges)):
                if number_of_values > 10:
                    randomized_labels[:, layer_index] = self._bolstered_uniform_distribution(layer_ranges[0],
                                                                                             layer_ranges[1],
                                                                                             number_of_values,
                                                                                             bolster_fraction,
                                                                                             bolster_width)
                else:
                    randomized_labels[:, layer_index] = np.random.uniform(*layer_ranges, number_of_values)
            else:
                real_randomized_labels = self._generate_random_values(np.asarray([(layer_ranges[0].real, layer_ranges[
                    1].real)]), number_of_values, bolster_fraction, bolster_width)

                imag_randomized_labels = self._generate_random_values(
                    np.asarray([(layer_ranges[0].imag, layer_ranges[1].imag)]), number_of_values, bolster_fraction,
                    bolster_width)

                randomized_labels[:, layer_index] = real_randomized_labels + 1j * imag_randomized_labels

        return randomized_labels

    def _bolstered_uniform_distribution(self, value_min: float, value_max: float, n_samples: int, bolster_fraction:
    float, bolster_width: float) -> ndarray:
        span = value_max - value_min

        n_bolster = int(np.ceil(n_samples * bolster_fraction / 2))
        n_uniform = n_samples - 2 * n_bolster

        uniform = np.random.uniform(value_min, value_max, n_uniform)

        bolster_min = np.random.normal(value_min, span * bolster_width, n_bolster)
        bolster_min = self._fold_distribution(bolster_min, value_min, value_max)
        bolster_max = np.random.normal(value_max, span * bolster_width, n_bolster)
        bolster_max = self._fold_distribution(bolster_max, value_min, value_max)

        total_distribution = np.concatenate((bolster_min, uniform, bolster_max))
        np.random.shuffle(total_distribution)

        return total_distribution

    @staticmethod
    def _fold_distribution(values: ndarray, min_value: float, max_value: float) -> ndarray:
        num_values = len(values)
        for i in range(num_values):
            if values[i] < min_value:
                values[i] += 2 * (min_value - values[i])
            elif values[i] > max_value:
                values[i] += 2 * (max_value - values[i])
        return values

    def _generate_random_roughness_from_thickness(self, roughness_ranges, randomized_thicknesses: ndarray) -> ndarray:
        randomized_roughnesses = np.zeros_like(randomized_thicknesses)
        number_of_samples = randomized_thicknesses.shape[0]
        number_of_layers = roughness_ranges.shape[0]

        min_roughnesses = roughness_ranges[:, 0]
        max_roughnesses = roughness_ranges[:, 1]

        for sample in range(number_of_samples):
            for layer in range(number_of_layers):
                max_roughness_from_thickness = self._thickness_correlation(randomized_thicknesses[sample, layer])
                if max_roughness_from_thickness < min_roughnesses[layer]:
                    randomized_roughnesses[sample, layer] = min_roughnesses[layer]
                elif max_roughness_from_thickness > max_roughnesses[layer]:
                    randomized_roughnesses[sample, layer] = np.random.uniform(min_roughnesses[layer],
                                                                              max_roughnesses[layer])
                else:
                    randomized_roughnesses[sample, layer] = np.random.uniform(min_roughnesses[layer],
                                                                              max_roughness_from_thickness)
        return randomized_roughnesses

    @staticmethod
    def _thickness_correlation(thickness: float) -> float:
        roughness = thickness / 2
        return roughness
