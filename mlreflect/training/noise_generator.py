import logging
from typing import Union, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tensorflow import keras

from .preprocessing import InputPreprocessor
from ..data_generation import noise
from ..data_generation.data_generator import ReflectivityGenerator
from ..utils import h5_tools


class NoiseGenerator(keras.utils.Sequence):

    def __init__(self, reflectivity, labels, input_preprocessor: InputPreprocessor, batch_size=32, mode='single',
                 shuffle=True, noise_range=None, background_range=None, relative_background_spread: float = 0.1):

        self.input_preprocessor = input_preprocessor
        self.n_input = reflectivity.shape[1]
        self.n_output = labels.shape[1]
        self.n_samples = len(reflectivity)
        self.batch_size = batch_size
        self.labels = labels
        self.reflectivity = reflectivity
        self.mode = mode
        self.shuffle = shuffle
        self.indexes = np.arange(self.n_samples)
        self.noise_range = noise_range
        self.background_range = background_range
        self.relative_background_spread = relative_background_spread
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__data_generation(indexes)

        return x, y, [None]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        if not self.input_preprocessor.has_saved_standardization:
            raise ValueError('input_preprocessor must have saved standardization')
        refl = self.reflectivity[indexes]
        if self.noise_range is not None:
            if self.mode is 'single':
                refl = noise.apply_shot_noise(refl, self.noise_range)[0]
            elif self.mode is 'batch':
                batch_noise_level = np.random.uniform(*self.noise_range)
                refl = noise.apply_shot_noise(refl, batch_noise_level)[0]
            else:
                raise ValueError('not a valid mode')
        if self.background_range is not None:
            if self.mode is 'single':
                refl += noise.generate_background(len(refl), self.n_input, self.background_range,
                                                  self.relative_background_spread)[0]
            elif self.mode is 'batch':
                batch_bg_level = np.random.uniform(*self.background_range)
                refl += noise.generate_background(len(refl), self.n_input, batch_bg_level,
                                                  self.relative_background_spread)[0]
            else:
                raise ValueError('not a valid mode')

        return self.input_preprocessor.standardize(refl), np.array(self.labels)[indexes]


def add_random_levels(file_name: str, labels: DataFrame, generator: ReflectivityGenerator, n_layers: int,
                      shot_noise_spread: Union[float, Tuple[float, float]],
                      background_base_level: Union[float, Tuple[float, float]],
                      relative_background_spread: float) -> Tuple[ndarray, ndarray, ndarray]:
    logging.info('Generating reflectivity curves ...')
    reflectivity = generator.simulate_reflectivity(labels)
    h5_tools.save_data_as_h5(file_name, generator.q_values, reflectivity, labels, n_layers)
    logging.info(f'Saved reflectivity data to {file_name}!')

    params = {'shot_noise_spread': shot_noise_spread,
              'background_base_level': background_base_level,
              'relative_background_spread': relative_background_spread}
    modified_data, noise_levels, bg_levels = apply_background(reflectivity, params, file_name)

    return modified_data, noise_levels, bg_levels


def add_discrete_levels(labels: DataFrame, generator: ReflectivityGenerator, n_layers: int,
                        shot_noise_spread: Tuple[float], background_base_level: Tuple[float],
                        relative_background_spread: float, file_name: str = None) -> Tuple[ndarray, DataFrame, ndarray,
                                                                                           ndarray]:
    multiplier = len(shot_noise_spread) * len(background_base_level)
    logging.info(f'The number of combinations is {multiplier}')
    level_combinations = [(noise_lvl, bg_lvl) for noise_lvl in shot_noise_spread for bg_lvl in background_base_level]

    logging.info('Simulating reflectivity ...')
    reflectivity = generator.simulate_reflectivity(labels)

    all_reflectivity = []
    all_noise_levels = []
    all_bg_levels = []

    for i, levels in enumerate(level_combinations):
        if file_name is None:
            this_file_name = None
        else:
            this_file_name = h5_tools.strip_file_extension(file_name) + f'_variant{i}.h5'
            h5_tools.save_data_as_h5(this_file_name, generator.q_values, reflectivity, labels, n_layers)
            logging.info(f'Saved reflectivity data to {this_file_name}!')

        params = {'shot_noise_spread': levels[0],
                  'background_base_level': levels[1],
                  'relative_background_spread': relative_background_spread}

        logging.info(f'Working on: noise_level {levels[0]}, background level {levels[1]} ...')
        this_reflectivity, this_noise_levels, this_bg_levels = apply_background(reflectivity, params, this_file_name)

        all_reflectivity.append(this_reflectivity)
        all_noise_levels.append(this_noise_levels)
        all_bg_levels.append(this_bg_levels)

    all_labels = pd.concat([labels] * multiplier).reset_index(drop=True)

    return np.concatenate(all_reflectivity), all_labels, np.concatenate(all_noise_levels), np.concatenate(all_bg_levels)


# TODO These functions are obsolete and should be removed
def apply_noise(reflectivity: ndarray, shot_noise_spread: Union[float, Tuple[float, float]], file_name: str = None):
    logging.info('Generating noise ...')
    noisy_reflectivity, noise_levels = noise.apply_shot_noise(reflectivity, shot_noise_spread)
    noise_array = noisy_reflectivity - reflectivity
    if file_name is not None:
        h5_tools.save_noise(file_name, noise_array, noise_levels)
        logging.info(f'Saved noise to {file_name}!')
    return noisy_reflectivity, noise_levels


def apply_background(reflectivity: ndarray, parameters: dict, file_name: str = None):
    noisy_reflectivity, noise_levels = apply_noise(reflectivity, parameters['shot_noise_spread'], file_name)
    logging.info('Generating background ...')
    background, bg_levels = noise.generate_background(*reflectivity.shape,
                                                      background_base_level=parameters['background_base_level'],
                                                      relative_background_spread=parameters[
                                                          'relative_background_spread'])
    if file_name is not None:
        h5_tools.save_background(file_name, background, bg_levels)
        logging.info(f'Saved background to {file_name}!')
    return noisy_reflectivity + background, noise_levels, bg_levels
