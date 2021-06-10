import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from scipy.special import erf
from tqdm import tqdm

from .distributions import sample_distribution
from .multilayer import MultilayerStructure
from .reflectivity import multilayer_reflectivity as builtin_engine
from ..utils.performance_tools import timer


class ReflectivityGenerator:
    """Generation of simulated reflectivity data and labels for neural network training.

    Args:
        q_values: An array-like object (`list`, `tuple`, `ndarray`, etc.) that contains the q-values in units of
            1/Å at which the reflected intensity will be simulated.
        sample: :class:`MultilayerStructure` object where the sample layers and their names and parameter ranges are
            defined.
        random_seed: Random seed for numpy.random.seed which affects the generation of the random labels (default
            ``None`` means random seed).
    """

    def __init__(self, q_values: ndarray, sample: MultilayerStructure, random_seed: int = None):
        try:
            from refl1d.reflectivity import reflectivity as refl1d_engine
            self._refl1d_engine = refl1d_engine
        except ImportError as error:
            self._refl1d_engine = None
            warnings.filterwarnings('once', '.*cannot import refl1d package.*')
            warnings.warn(f'Cannot import refl1d package ({error}).\nUsing builtin Python method.', ImportWarning)

        if random_seed is not None:
            np.random.seed(random_seed)
        self.q_values = np.asarray(q_values)
        self.sample = sample

    @timer
    def generate_random_labels(self, number_of_samples: int, distribution_type: str = 'bolstered',
                               bolster_fraction: float = 0.15, bolster_width: float = 0.1) -> DataFrame:
        """Generates random labels in the parameter ranges defined by the sample and returns them as pandas DataFrame.

        Args:
            number_of_samples: Number of label sets that will be generated.
            distribution_type: Can be ``'bolstered'`` (default) or ``'uniform'``.
            bolster_fraction: Fraction of simulated samples that will be redistributed to the sides of the distribution.
            bolster_width: Width of the Gaussian distribution of the redistributed samples.

        Returns:
            labels: Pandas DataFrame with the randomly generated labels.
        """

        if not number_of_samples > 0:
            raise ValueError('`number_of_samples` must be at least 1.')

        distribution_params = {
            'distribution_type': distribution_type,
            'bolster_fraction': bolster_fraction,
            'bolster_width': bolster_width
        }

        label_names = self.sample.label_names

        label_dict = {}

        for layer in self.sample:
            label_dict[layer.name + '_sld'] = layer.sld.sample(number_of_samples, **distribution_params)
            if hasattr(layer, 'thickness'):
                label_dict[layer.name + '_thickness'] = layer.thickness.sample(number_of_samples, **distribution_params)
                label_dict[layer.name + '_roughness'] = self._generate_roughness_from_thickness(layer.roughness,
                                                                                                label_dict[
                                                                                                    layer.name + '_thickness'],
                                                                                                **distribution_params)
            elif hasattr(layer, 'roughness'):
                label_dict[layer.name + '_roughness'] = layer.roughness.sample(number_of_samples, **distribution_params)

        return pd.DataFrame(label_dict)[label_names]

    def _generate_roughness_from_thickness(self, roughness, random_thickness, distribution_type: str,
                                           bolster_fraction: float, bolster_width: float) -> ndarray:
        number_of_values = len(random_thickness)
        random_roughness = roughness.sample(number_of_values, distribution_type=distribution_type,
                                            bolster_fraction=bolster_fraction, bolster_width=bolster_width)
        for i in range(number_of_values):
            max_roughness = self._get_max_roughness(random_thickness[i])
            if max_roughness < roughness.min:
                random_roughness[i] = roughness.min
            elif max_roughness > roughness.max:
                pass
            else:
                random_roughness[i] = sample_distribution(1, min_value=roughness.min, max_value=max_roughness,
                                                          distribution_type=distribution_type,
                                                          bolster_fraction=bolster_fraction,
                                                          bolster_width=bolster_width)
        return random_roughness

    @staticmethod
    def _get_max_roughness(thickness: float) -> float:
        max_roughness = thickness / 2
        return max_roughness

    @timer
    def simulate_reflectivity(self, labels: DataFrame, q_noise_spread: float = 0, engine: str = 'refl1d',
                              progress_bar=True) -> ndarray:
        """Simulates reflectivity curves for the given labels and returns them as `ndarray`.

        Args:
            labels: Must a pandas `DataFrame` with each column representing one label. The label order from left to
                right must be "thickness", "roughness" and "scattering length density" with layers from bottom to top.
                Example for 2 layers: ``['thickness_layer1', 'thickness_layer2', 'roughness_layer1', 'roughness_layer2',
                'sld_layer1', 'sld_layer2']``
            q_noise_spread: Standard deviation of the normal distribution of scaling factors (centered at 1) that are
                applied to each q-value during reflectivity simulation.
            engine: ``'refl1d'`` (default): Uses C++-based simulation from the `refl1d` package.
                    ``'builtin'``: Uses the built-in python-based simulation (slower).
            progress_bar: If `True`, a `tqdm` progress bar will be displayed.

        Args:
            labels: Must a pandas `DataFrame` with each column representing one label.

        Returns:
            reflectivity_curves: Simulated reflectivity curves.
        """
        if type(labels) is not DataFrame:
            raise TypeError(f'`labels` must a pandas `DataFrame` with each column representing one label.')

        valid_engines = ('refl1d', 'builtin')
        if engine not in valid_engines:
            raise ValueError(f'"{engine}" not a valid engine')

        thicknesses, roughnesses, slds = self.separate_labels_by_category(labels)

        number_of_curves = len(thicknesses)

        noisy_q_values = self._make_noisy_q_values(self.q_values, q_noise_spread, number_of_curves)

        if self._refl1d_engine is None:
            engine = 'builtin'

        if engine == 'refl1d':
            return self._refl1d(thicknesses, roughnesses, slds, noisy_q_values, progress_bar)
        else:
            return self._builtin(thicknesses, roughnesses, slds, noisy_q_values, progress_bar)

    def _refl1d(self, thicknesses, roughnesses, slds, q_values, progress_bar=True):
        number_of_q_values = q_values.shape[1]
        number_of_curves = thicknesses.shape[0]

        reflectivity_curves = np.zeros([number_of_curves, number_of_q_values])
        depth = np.fliplr(thicknesses)
        depth = np.hstack((np.ones((number_of_curves, 1)), depth, np.ones((number_of_curves, 1))))
        rho = np.fliplr(slds)

        for curve in tqdm(range(number_of_curves), disable=not progress_bar):
            params = {'kz': q_values[curve, :] / 2, 'depth': depth[curve, :],
                      'sigma': np.flip(roughnesses[curve, :])}

            this_rho = rho[curve, :]
            if np.sum(np.iscomplex(this_rho)) > 0:
                irho = this_rho.imag
                this_rho = this_rho.real
                params['irho'] = irho
            params['rho'] = this_rho

            reflectivity = self._refl1d_engine(**params)
            del params
            reflectivity_curves[curve, :] = reflectivity
        return reflectivity_curves

    @staticmethod
    def _builtin(thicknesses, roughnesses, slds, q_values, progress_bar=True):
        number_of_q_values = q_values.shape[1]
        number_of_curves = thicknesses.shape[0]

        reflectivity_curves = np.zeros([number_of_curves, number_of_q_values])
        thicknesses_si = thicknesses * 1e-10
        roughnesses_si = roughnesses * 1e-10
        slds_si = slds * 1e14
        q_values_si = q_values * 1e10

        for curve in tqdm(range(number_of_curves), disable=not progress_bar):
            reflectivity = builtin_engine(q_values_si[curve, :], thicknesses_si[curve, :], roughnesses_si[curve, :],
                                          slds_si[curve, :-1], slds_si[curve, -1])
            reflectivity_curves[curve, :] = reflectivity

        return reflectivity_curves

    @staticmethod
    def _make_noisy_q_values(q_values: ndarray, q_noise_spread: float, number_of_curves: int) -> ndarray:
        percentage_deviation = np.random.normal(1, q_noise_spread, (number_of_curves, len(q_values)))
        return q_values * percentage_deviation

    @staticmethod
    def _make_gaussian(x: ndarray, std: float, n_std: float = 5):
        center = np.min(x) + (np.max(x) - np.min(x)) / 2
        g = np.exp(- ((x - center) / std) ** 2 / 2) / (std * np.sqrt(2 * np.pi))
        g /= np.sum(g)
        gauss_range = ((x - center) >= -n_std * std) & ((x - center) <= n_std * std)
        g = g[gauss_range]
        x_red = x[gauss_range]
        return g, x_red

    @staticmethod
    def separate_labels_by_category(labels: DataFrame) -> Tuple[ndarray, ndarray, ndarray]:
        thicknesses = []
        roughnesses = []
        slds = []

        reindexed_labels = labels.reset_index(drop=True)

        for name in reindexed_labels.columns:
            if 'thickness' in name:
                thicknesses.append(reindexed_labels[name])
            elif 'roughness' in name:
                roughnesses.append(reindexed_labels[name])
            elif 'sld' in name:
                slds.append(reindexed_labels[name])

        thicknesses = np.array(thicknesses).T
        roughnesses = np.array(roughnesses).T
        slds = np.array(slds).T

        return thicknesses, roughnesses, slds

    @timer
    def simulate_sld_profiles(self, labels: DataFrame, progress_bar=True) -> List[ndarray]:
        """Simulates real scattering length density profiles for the given labels and returns them as ndarray.

        Args:
            labels: Must be pandas `DataFrame` with each column representing one label. The label order from left to
                right must be "thickness", "roughness" and "scattering length density" with layers from bottom to top.
                Example for 2 layers: ``['thickness_layer1', 'thickness_layer2', 'roughness_layer1',
                'roughness_layer2', 'sld_layer1', 'sld_layer2']``
            progress_bar: If `True`, a `tqdm` progress bar will be displayed.

        Returns:
            sld_profiles: List of `ndarray` of simulated scattering length density profiles (real part).
        """
        if len(labels.shape) != 2:
            raise ValueError('labels dataframe must have 2 dimensions (#samples, #labels_per_sample)')

        number_of_profiles = labels.shape[0]

        thicknesses, roughnesses, slds = self.separate_labels_by_category(labels)
        sld_substrate = slds[:, 0]
        layer_slds = slds[:, 1:-1]
        sld_ambient = slds[:, -1]
        roughnesses = roughnesses[:, :]

        sld_profiles = []

        for i in tqdm(range(number_of_profiles), disable=not progress_bar):
            height, profile = self.make_sld_profile(thicknesses[i, :], layer_slds[i, :], roughnesses[i, :],
                                                    sld_substrate[i],
                                                    sld_ambient[i])

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
                provide empty tuple ``(,)``.
            sld: `ndarray` of layer scattering length densities in units 1/Å^-2 * 10^-6 from bottom to top.  For no
                layers (only substrate) provide empty tuple ``(,)``.
            roughness: `ndarray` of RMS interface roughnesses in units Å from bottom to top. At least one has to be
                given.
            sld_substrate: Scattering length density of the used substrate in units 1/Å^-2 * 10^-6.
            sld_ambient: Scattering length density of the ambient medium in units 1/Å^-2 * 10^-6.

        Returns:
            height, sld_profile: Tuple of `ndarrays` of sample height in units Å and the scattering length density 
            profile in units 1/Å^-2 * 10^-6.
        """

        if not len(thickness) == len(sld) == (len(roughness) - 1):
            raise ValueError('Number of layers must be consistent')

        thickness = abs(thickness)
        roughness = abs(roughness)

        total_thickness = np.sum(thickness)
        cumulative_thickness = np.append(0, np.cumsum(thickness))

        sld = np.append(sld_substrate, sld)
        sld = np.append(sld, sld_ambient)

        sld = np.real(sld)

        dummy_sub_thickness = 10 + 3 * roughness[0]
        dummy_ambient_thickness = 10 + 3 * roughness[-1]

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
