import numpy as np
from pandas import DataFrame

from ..data_generation.data_generator import ReflectivityGenerator


class ModelMixer:
    """Simulate reflectivity curves from given labels with different models with given probabilities.

    Args:
        q_values: An array-like object (`list`, `tuple`, `ndarray`, etc.) that contains the q-values in units of
            1/Å at which the reflected intensity will be simulated.
    """

    def __init__(self, q_values):
        self._sample_models = []
        self._model_weights = []
        self._generators = []
        self._q = q_values

    @property
    def sample_models(self):
        return self._sample_models

    @property
    def model_weights(self):
        return self._model_weights

    @property
    def q(self):
        return self._q

    def add_model(self, sample, weight=1):
        self._sample_models.append(sample)
        self._model_weights.append(weight)
        self._add_generator(sample)

    def _add_generator(self, sample):
        self._generators.append(ReflectivityGenerator(self.q, sample))

    def simulate_reflectivity(self, labels: DataFrame, engine: str = 'refl1d'):
        num_labels = labels.shape[0]
        reflectivity = np.zeros((num_labels, len(self.q)))
        normalized_weights = np.array(self.model_weights) / sum(self.model_weights)
        for i in range(num_labels):
            generator = np.random.choice(self._generators, p=normalized_weights)
            new_labels = generator.generate_random_labels(1)
            new_labels[labels.columns] = np.array(labels[i:(i + 1)])
            reflectivity[i] = generator.simulate_reflectivity(new_labels, engine)
        return reflectivity
