import datetime
from typing import Tuple

from numpy import ndarray
from tensorflow import keras

from . import UniformNoiseGenerator
from ..data_generation import MultilayerStructure
from ..data_generation import noise, ReflectivityGenerator
from ..models import TrainedModel
from ..training import InputPreprocessor, OutputPreprocessor


class Trainer:
    """Train a neural network model for a given sample structure and q values

    Args:
        sample_structure: MultilayerStructure object that describes the sample. Should only have one non-constant layer.
        q_values: ndarray of the q values used for training. Should be similar to the experimental q values.
        random_seed: random seed for the training data generation. `None` means the seed is chosen randomly.
    """

    def __init__(self, sample_structure: MultilayerStructure, q_values: ndarray, random_seed=None):
        self._sample_structure = sample_structure
        self._q_values = q_values
        self._generator = ReflectivityGenerator(q_values, sample_structure, random_seed)

        self.training_data = None
        self._uniform_noise_range = (0.9, 1.1)
        self._scaling_range = (0.9, 1.1)

        self.input_preprocessor = None
        self.output_preprocessor = OutputPreprocessor(sample_structure, 'min_to_zero')

        self.keras_model = self._prepare_model(len(q_values), len(self.output_preprocessor.used_labels))

    def generate_training_data(self, training_samples: int = 2 ** 17):
        """Generate a training data set for training the neural network."""
        labels = self._generator.generate_random_labels(training_samples)
        reflectivity = self._generator.simulate_reflectivity(labels)

        self.training_data = {
            'labels': labels,
            'reflectivity': reflectivity
        }

        self.input_preprocessor = self._prepare_input_preprocessor()

    def train(self, n_epochs=175, batch_size=512, verbose=1, val_split=0.2) -> Tuple[TrainedModel, 'History']:
        """Train a fully-connected neural network with the generated training data.

        Args:
            n_epochs: Number of epochs to train for.
            batch_size: Number of curves per training batch. Must be smaller than `val_split` times the training set
                size.
            verbose: Determines the amount of text output during training (0, 1, 2).
            val_split: The fraction of the training set that is withheld for validation.

        Returns:
            trained_model: TrainedModel object that contains the trained keras model as well as other parameters
                necessary to predict test data.
            history: Training history output from keras model.fit().
        """
        if not self.has_training_data:
            raise ValueError('must first generate training data')
        prep_labels, removed_labels = self.output_preprocessor.apply_preprocessing(self.training_data['labels'])

        n_train = int((1 - val_split) * len(prep_labels))

        noise_gen_train = UniformNoiseGenerator(self.training_data['reflectivity'][:n_train], prep_labels[:n_train],
                                                self.input_preprocessor, batch_size,
                                                uniform_noise_range=self._uniform_noise_range,
                                                scaling_range=self._scaling_range)
        noise_gen_val = UniformNoiseGenerator(self.training_data['reflectivity'][n_train:], prep_labels[n_train:],
                                              self.input_preprocessor, batch_size,
                                              uniform_noise_range=self._uniform_noise_range,
                                              scaling_range=self._scaling_range)
        now = datetime.datetime.now()

        history = self.keras_model.fit(noise_gen_train, validation_data=noise_gen_val, epochs=n_epochs, verbose=verbose,
                                       callbacks=self._prepare_callbacks(verbose))

        then = datetime.datetime.now()
        print(f'Time needed for training: {then - now}')

        trained_model = TrainedModel()
        trained_model.from_variable(self.keras_model, self._sample_structure, self._q_values,
                                    self.input_preprocessor.standard_mean,
                                    self.input_preprocessor.standard_std)
        return trained_model, history

    @property
    def has_training_data(self):
        try:
            return 'reflectivity' in self.training_data.keys() and 'labels' in self.training_data.keys()
        except AttributeError:
            return False

    @staticmethod
    def _prepare_model(n_input: int, n_output: int):
        model = keras.models.Sequential()

        model.add(keras.layers.Dense(500, input_dim=n_input))
        model.add(keras.layers.Activation('relu'))

        for i in range(2):
            model.add(keras.layers.Dense(500))
            model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.Dense(n_output))

        adam_optimizer = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False)

        model.compile(loss='mean_squared_error', optimizer=adam_optimizer)

        return model

    def _prepare_input_preprocessor(self):
        noisy_reflectivity = noise.apply_scaling_factor(
            noise.apply_uniform_noise(self.training_data['reflectivity'], self._uniform_noise_range),
            self._scaling_range)
        ip = InputPreprocessor()
        ip.standardize(noisy_reflectivity)
        return ip

    @staticmethod
    def _prepare_callbacks(self, verbose=1):
        lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=verbose)

        # checkpoint = keras.callbacks.ModelCheckpoint(filepath='models/' + experiment_id + '_model.h5',
        # monitor='val_loss', verbose=1, save_best_only=True)

        return [lr_reduction]
