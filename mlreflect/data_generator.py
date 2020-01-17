from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from scipy.special import erf
from tqdm import tqdm

from .label_helpers import convert_to_ndarray
from .layers import MultilayerStructure
from .performance_tools import timer
from .reflectivity import multilayer_reflectivity as refl


class ReflectivityGenerator:
    """Generation of simulated reflectivity data and labels for neural network training.

    Args:
        q_values: An array-like object (list, tuple, ndarray, etc.) that contains the q-values in units of
            1/Å at which the reflected intensity will be simulated.
        sample: MultilayerStructure object where the sample layers and their names and parameter ranges are defined.
        random_seed: Random seed for numpy.random.seed which affects the generation of the random labels (default 1).
        q_noise_spread: Standard deviation of the normal distribution of scaling factors (centered at 1) that are
            applied to each q-value during reflectivity simulation.
        shot_noise_spread: Scaling factor c for the standard deviation sqrt(I*c) of the shot noise around the intensity
            of simulated reflectivity curves. Since the intensity is normalized, this is equivalent to setting the
            number of photon counts N = 1/c.
        background_noise_base_level: Constant background intensity that is added to reflectivity simulations.
        background_noise_spread: Standard deviation of the background intensity centered at background_noise_base_level.
        blur_width: Standard deviation of the gaussian blur in units of 1/Å that is applied to simulated reflectivity
        curves to simulate the slit function. If blur_width = 0 (default) no blur is  applied.

    Methods:
        generate_random_labels()
        simulate_reflectivity()
        simulate_sld_profiles()
        make_sld_profile()
        separate_labels()

    Returns:
        TrainingData object.
    """

    def __init__(self, q_values: ndarray, sample: MultilayerStructure, q_noise_spread: float = 0,
                 shot_noise_spread: float = 0, background_noise_base_level: float = 0,
                 background_noise_spread: float = 0, blur_width: float = 0, random_seed: int = 1):

        np.random.seed(random_seed)
        self.q_values = np.asarray(q_values)
        self.sample = sample

        self.q_noise_spread = q_noise_spread
        self.shot_noise_spread = shot_noise_spread
        self.background_noise_base_level = background_noise_base_level
        self.background_noise_spread = background_noise_spread
        self.blur_width = blur_width

    @timer
    def generate_random_labels(self, number_of_samples: int, distribution_type: str = 'bolstered',
                               bolster_fraction: float = 0.15, bolster_width: float = 0.1) -> DataFrame:
        """Generates random labels in the parameter ranges defined by the sample and returns them as pandas DataFrame.

        Args:
            number_of_samples: Number of label sets that will be generated.
            distribution_type: Can be 'bolstered' (default) or 'uniform'.
            bolster_fraction: Fraction of simulated samples that will be redistributed to the sides of the distribution.
            bolster_width: Width of the Gaussian distribution of the redistributed samples.

        Returns:
            labels: Pandas DataFrame with the randomly generated labels.
        """

        thickness_ranges = self.sample.get_thickness_ranges()
        roughness_ranges = self.sample.get_roughness_ranges()
        sld_ranges = self.sample.get_sld_ranges()

        if not number_of_samples > 0:
            raise ValueError('`number_of_samples` must be at least 1.')

        label_names = self.sample.get_label_names()

        randomized_slds = self._generate_random_values(sld_ranges, number_of_samples, distribution_type,
                                                       bolster_fraction, bolster_width)
        randomized_thicknesses = self._generate_random_values(thickness_ranges, number_of_samples, distribution_type,
                                                              bolster_fraction, bolster_width)
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
            right must be "thickness", "roughness" and "scattering length density" with layers from bottom to top.
                Example for 2 layers: ['thickness_layer1', 'thickness_layer2', 'roughness_layer1', 'roughness_layer2',
                'sld_layer1', 'sld_layer2']

        Returns:
            reflectivity_curves: Simulated reflectivity curves.
        """
        labels = convert_to_ndarray(labels)

        number_of_q_values = len(self.q_values)
        number_of_curves = labels.shape[0]

        thicknesses, roughnesses, slds = self.separate_labels(labels)

        thicknesses_si = thicknesses * 1e-10
        roughnesses_si = roughnesses * 1e-10
        slds_si = slds * 1e14
        ambient_sld_si = self.sample.ambient_sld * 1e14

        q_values_si = self.q_values * 1e10

        reflectivity_curves = np.zeros([number_of_curves, number_of_q_values])

        noisy_q_values = self._make_noisy_q_values(q_values_si, number_of_curves)

        for curve in tqdm(range(number_of_curves)):
            reflectivity = refl(noisy_q_values[curve, :], thicknesses_si[curve, :], roughnesses_si[curve, :],
                                slds_si[curve, :], ambient_sld_si)

            reflectivity_noisy = self._apply_gaussian_blur(reflectivity)
            reflectivity_noisy = self._apply_shot_noise(reflectivity_noisy)
            reflectivity_noisy = self._apply_background_noise(reflectivity_noisy)

            reflectivity_curves[curve, :] = reflectivity_noisy

        return reflectivity_curves

    def _make_noisy_q_values(self, q_values: ndarray, number_of_curves: int) -> ndarray:
        percentage_deviation = np.random.normal(1, self.q_noise_spread, (number_of_curves, len(q_values)))
        return q_values * percentage_deviation

    def _apply_shot_noise(self, reflectivity_curve: ndarray) -> ndarray:
        noisy_reflectivity = np.clip(
            np.random.normal(reflectivity_curve, np.sqrt(reflectivity_curve * self.shot_noise_spread)), 1e-8, None)

        return noisy_reflectivity

    def _apply_background_noise(self, reflectivity_curve: ndarray) -> ndarray:
        num_q_values = len(reflectivity_curve)
        background = np.random.normal(self.background_noise_base_level, self.background_noise_spread, num_q_values)

        return reflectivity_curve + background

    def _apply_gaussian_blur(self, reflectivity_curve: ndarray) -> ndarray:
        sigma = self.blur_width
        if sigma == 0:
            return reflectivity_curve

        gaussian, _ = self._make_gaussian(self.q_values, self.blur_width)
        blurred_reflectivity = np.convolve(reflectivity_curve, gaussian, 'same')
        return blurred_reflectivity

    @staticmethod
    def _make_gaussian(x: ndarray, std: float, n_std: float = 5):
        center = np.min(x) + (np.max(x) - np.min(x)) / 2
        g = np.exp(- ((x - center) / std) ** 2 / 2) / (std * np.sqrt(2 * np.pi))
        g /= np.sum(g)
        gauss_range = ((x - center) >= -n_std * std) & ((x - center) <= n_std * std)
        g = g[gauss_range]
        x_red = x[gauss_range]
        return g, x_red

    def _generate_random_values(self, label_ranges: ndarray, number_of_values: int, distribution_type: str,
                                bolster_fraction: float, bolster_width: float) -> ndarray:

        number_of_layers = label_ranges.shape[0]

        randomized_labels = np.zeros((number_of_values, number_of_layers))
        for layer_index in range(number_of_layers):
            layer_ranges = label_ranges[layer_index]

            if np.all(np.isreal(layer_ranges)):
                if distribution_type == 'bolstered':
                    randomized_labels[:, layer_index] = self._bolstered_uniform_distribution(layer_ranges[0],
                                                                                             layer_ranges[1],
                                                                                             number_of_values,
                                                                                             bolster_fraction,
                                                                                             bolster_width)
                elif distribution_type == 'uniform':
                    randomized_labels[:, layer_index] = np.random.uniform(layer_ranges[0], layer_ranges[1],
                                                                          number_of_values)

            else:
                real_randomized_labels = self._generate_random_values(np.asarray([(layer_ranges[0].real, layer_ranges[
                    1].real)]), number_of_values, distribution_type, bolster_fraction, bolster_width)

                imag_randomized_labels = self._generate_random_values(
                    np.asarray([(layer_ranges[0].imag, layer_ranges[1].imag)]), number_of_values, distribution_type,
                    bolster_fraction, bolster_width)

                randomized_labels[:, layer_index] = real_randomized_labels + 1j * imag_randomized_labels

        return randomized_labels

    def _bolstered_uniform_distribution(self, value_min: float, value_max: float, n_samples: int, bolster_fraction:
    float, bolster_width: float) -> ndarray:
        span = value_max - value_min

        n_bolster = int(np.floor(n_samples * bolster_fraction / 2))
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

    @timer
    def simulate_sld_profiles(self, labels: Union[DataFrame, ndarray]) -> List[ndarray]:
        """Simulates real scattering length density profiles for the given labels and returns them as ndarray.

        Args:
            labels: Must be ndarray or DataFrame with each column representing one label. The label order from left to
            right must be "thickness", "roughness" and "scattering length density" with layers from bottom to top.
                Example for 2 layers: ['thickness_layer1', 'thickness_layer2', 'roughness_layer1', 'roughness_layer2',
                'sld_layer1', 'sld_layer2']

        Returns:
            sld_profiles: List of ndarrays of simulated scattering length density profiles (real part).
        """

        labels = convert_to_ndarray(labels)

        number_of_profiles = labels.shape[0]

        thicknesses, roughnesses, slds = self.separate_labels(labels)

        thicknesses = thicknesses[:, 1:]
        sld_substrate = slds[:, 0]
        slds = slds[:, 1:]
        roughnesses = roughnesses[:, :]

        sld_profiles = []

        for i in tqdm(range(number_of_profiles)):
            height, profile = self.make_sld_profile(thicknesses[i, :], slds[i, :], roughnesses[i, :], sld_substrate[i],
                                                    self.sample.ambient_sld)

            this_profile = np.zeros((2, len(height)))
            this_profile[0, :] = height
            this_profile[1, :] = profile
            sld_profiles += [this_profile]

        return sld_profiles

    def make_sld_profile(self, thickness: ndarray, sld: ndarray, roughness: ndarray, sld_substrate: float, sld_ambient:
    float) -> Tuple[ndarray, ndarray]:
        """Generate scattering length density profile in units 1/Å^-2 * 10^-6 with height in units Å.
        
        Args:
            thickness: ndarray of layer thicknesses in units Å from bottom to top. For no layers (only substrate) 
                provide empty tuple ().
            sld: ndarray of layer scattering length densities in units 1/Å^-2 * 10^-6 from bottom to top.  For no layers
                (only substrate) provide empty tuple ().
            roughness: ndarray of RMS interface roughnesses in units Å from bottom to top. At least one has to be given.
            sld_substrate: Scattering length density of the used substrate in units 1/Å^-2 * 10^-6.
            sld_ambient: Scattering length density of the ambient medium in units 1/Å^-2 * 10^-6.

        Returns:
            height, sld_profile: Tuple of ndarrays of sample height in units Å and the scattering length density profile
                in units 1/Å^-2 * 10^-6.
        """

        if not len(thickness) == len(sld) == (len(roughness) - 1):
            raise ValueError('Number of layers must be consistent')

        total_thickness = np.sum(thickness)
        cumulative_thickness = np.append(0, np.cumsum(thickness))

        sld = np.append(sld_substrate, sld)
        sld = np.append(sld, sld_ambient)

        sld = np.real(sld)

        dummy_sub_thickness = 100
        dummy_ambient_thickness = 100

        height = np.arange(-dummy_sub_thickness, total_thickness + dummy_ambient_thickness, 0.1)
        sld_profile = np.ones_like(height) * sld[0]
        for i in range(len(roughness)):
            center = cumulative_thickness[i]
            width = roughness[i]

            segment = self._smooth_step(height, center, width, sld[i], sld[i + 1])

            sld_profile += segment

        return height, sld_profile

    @staticmethod
    def _smooth_step(z: ndarray, center: float, stdev: float, left_value: float, right_value: float) -> ndarray:
        difference = abs(left_value - right_value)

        profile = erf((z - center) / (stdev * np.sqrt(2)))

        profile += 1

        if left_value > right_value:
            profile *= -1

        profile *= difference / 2

        return profile

    @staticmethod
    def separate_labels(labels: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        number_of_labels = labels.shape[1]
        number_of_layers = int(number_of_labels / 3)

        thicknesses = labels[:, :number_of_layers]
        roughnesses = labels[:, number_of_layers:2 * number_of_layers]
        slds = labels[:, 2 * number_of_layers:3 * number_of_layers]

        return thicknesses, roughnesses, slds
