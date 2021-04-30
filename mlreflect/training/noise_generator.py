import numpy as np
from numpy import ndarray
from tensorflow import keras

from .preprocessing import InputPreprocessor
from ..data_generation import noise
from ..data_generation.distributions import random_logarithmic_distribution


class BaseGenerator(keras.utils.Sequence):
    def __init__(self, reflectivity, labels, batch_functions, batch_size=32, shuffle=True):
        self.n_samples = len(reflectivity)
        if batch_size > self.n_samples:
            raise ValueError('batch size cannot be larger than input length')
        self.batch_functions = batch_functions
        self.batch_size = batch_size
        self.n_input = reflectivity.shape[1]
        self.n_output = labels.shape[1]
        self.labels = labels
        self.reflectivity = reflectivity
        self.shuffle = shuffle
        self.indexes = np.arange(self.n_samples)
        self.__shuffle()

    def __len__(self):
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__data_generation(indexes)

        return x, y, [None]

    def on_epoch_end(self):
        self.__shuffle()

    def __shuffle(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        batch_input = self.reflectivity[indexes]
        batch_output = np.array(self.labels)[indexes]
        for function in self.batch_functions:
            batch_input, batch_output = function(batch_input, batch_output)
        return batch_input, batch_output


class NoiseGenerator(BaseGenerator):
    """Generator object that returns a standardized batch of reflectivity and labels with random noise and background.

    Args:
        reflectivity: Training reflectivity curves
        labels: Training labels on the same order as reflectivity
        input_preprocessor: :class:``InputPreprocessor`` object with or without stored standardization values
        batch_size: Number of samples per mini batch
        shuffle: If ``True``, shuffles reflectivity and labels after every epoch
        noise_range: Tuple ``(min, max)`` between which the shot noise levels are randomly generated
        background_range: Tuple ``(min, max)`` between which the background levels are randomly generated
        mode:
            'single': random noise and background levels are generated for every curve of a mini batch
            'batch': random noise and background levels are generated for each mini batch
        relative_background_spread: Relative standard deviation of the normal distribution (e.g. a value of ``0.1``
                means the standard deviation is 10% of the mean)

    """

    def __init__(self, reflectivity: ndarray, labels: ndarray, input_preprocessor: InputPreprocessor, batch_size=32,
                 shuffle=True, mode='single', noise_range=None, background_range=None,
                 relative_background_spread: float = 0.1):

        super().__init__(reflectivity, labels, None, batch_size, shuffle)

        self.input_preprocessor = input_preprocessor
        self.mode = mode
        self.noise_range = noise_range
        self.background_range = background_range
        self.relative_background_spread = relative_background_spread

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__data_generation(indexes)

        return x, y, [None]

    def __data_generation(self, indexes):
        if not self.input_preprocessor.has_saved_standardization:
            raise ValueError('input_preprocessor must have saved standardization')
        refl = self.reflectivity[indexes]
        if self.noise_range is not None:
            if self.mode is 'single':
                refl = noise.apply_shot_noise(refl, self.noise_range)[0]
            elif self.mode is 'batch':
                batch_noise_level = random_logarithmic_distribution(*self.noise_range, 1)
                refl = noise.apply_shot_noise(refl, batch_noise_level)[0]
            else:
                raise ValueError('not a valid mode')
        if self.background_range is not None:
            if self.mode is 'single':
                refl += noise.generate_background(len(refl), self.n_input, self.background_range,
                                                  self.relative_background_spread)[0]
            elif self.mode is 'batch':
                batch_bg_level = random_logarithmic_distribution(*self.background_range, 1)
                refl += noise.generate_background(len(refl), self.n_input, batch_bg_level,
                                                  self.relative_background_spread)[0]
            else:
                raise ValueError('not a valid mode')

        return self.input_preprocessor.standardize(refl), np.array(self.labels)[indexes]


class UniformNoiseGenerator(NoiseGenerator):
    def __init__(self, reflectivity, labels, ip, batch_size=32, mode='single', shuffle=True, uniform_noise_range=(1, 1),
                 scaling_range=(1, 1)):
        super().__init__(reflectivity, labels, ip, batch_size=batch_size, mode=mode, shuffle=shuffle,
                         noise_range=None, background_range=None,
                         relative_background_spread=0)
        self.uniform_noise_range = uniform_noise_range
        self.scaling_range = scaling_range
        self.ip = ip

    def __data_generation(self, indexes):
        refl = self.reflectivity[indexes]
        refl = noise.apply_scaling_factor(noise.apply_uniform_noise(refl, self.uniform_noise_range), self.scaling_range)

        return self.ip.standardize(refl), np.array(self.labels)[indexes]

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__data_generation(indexes)

        return x, y, [None]


class NoiseGeneratorLog(NoiseGenerator):
    def __init__(self, reflectivity, labels, batch_size=32, mode='single', shuffle=True, noise_range=None,
                 background_range=None, relative_background_spread: float = 0.1):
        super().__init__(reflectivity, labels, None, batch_size=batch_size, mode=mode, shuffle=shuffle,
                         noise_range=noise_range, background_range=background_range,
                         relative_background_spread=relative_background_spread)

    def __data_generation(self, indexes):
        refl = self.reflectivity[indexes]
        if self.noise_range is not None:
            if self.mode is 'single':
                refl = noise.apply_shot_noise(refl, self.noise_range)[0]
            elif self.mode is 'batch':
                batch_noise_level = random_logarithmic_distribution(*self.noise_range, 1)
                refl = noise.apply_shot_noise(refl, batch_noise_level)[0]
            else:
                raise ValueError('not a valid mode')
        if self.background_range is not None:
            if self.mode is 'single':
                refl += noise.generate_background(len(refl), self.n_input, self.background_range,
                                                  self.relative_background_spread)[0]
            elif self.mode is 'batch':
                batch_bg_level = random_logarithmic_distribution(*self.background_range, 1)
                refl += noise.generate_background(len(refl), self.n_input, batch_bg_level,
                                                  self.relative_background_spread)[0]
            else:
                raise ValueError('not a valid mode')

        return abs(np.log10(refl) / 10), np.array(self.labels)[indexes]

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__data_generation(indexes)

        return x, y, [None]
