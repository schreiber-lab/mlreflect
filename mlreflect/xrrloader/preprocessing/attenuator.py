import numpy as np


def apply_attenuation(intensity, attenuator):
    return intensity * attenuator


def correct_discontinuities(intensity, angle):
    diff_angle = np.diff(angle)
    for i in range(len(diff_angle)):
        if diff_angle[i] == 0:
            factor = intensity[i] / intensity[i + 1]
            intensity[(i + 1):] *= factor
    return intensity
