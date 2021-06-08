import numpy as np
from numpy import ndarray


def normalize_to_max(intensity: ndarray):
    return intensity / np.max(intensity)


def normalize_to_first(intensity: ndarray):
    return intensity / intensity[0]
