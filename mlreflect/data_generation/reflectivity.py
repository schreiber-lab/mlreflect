from typing import Iterable

import numpy as np


def multilayer_reflectivity(q_values: Iterable, thickness: Iterable, roughness: Iterable,
                            scattering_length_density: Iterable, ambient_sld: float = 0.0):
    """Returns a normalized reflectivity curve for a set of stacked layers with the given parameters.

    Args:
        q_values: An array-like object (list, tuple, ndarray, etc.) that contains the q-values in SI units at which the
            reflected intensity will be simulated.
        thickness: An array-like object (list, tuple, ndarray, etc.) that contains the thicknesses of the sample layers
            in SI units in order from bottom to top excluding the bottom most layer (substrate).
        roughness: An array-like object (list, tuple, ndarray, etc.) that contains the roughnesses of the sample
            interfaces in SI units in order from bottom to top.
        scattering_length_density: An array-like object (list, tuple, ndarray, etc.) that contains the scattering
            length densities of the sample layers in SI units in order from bottom to top (excluding the ambient SLD).
        ambient_sld: Scattering length density of the ambient environment (above the top most layer).

    Returns:
        ndrray of simulated intensity values with same length as ``q_values``.
    """

    q_values = np.asarray(q_values)
    thickness = np.flip(np.asarray(thickness))
    roughness = np.flip(np.asarray(roughness))
    scattering_length_density = np.flip(np.asarray(scattering_length_density))
    ambient_sld = np.asarray(ambient_sld)

    if ambient_sld != 0:
        raise NotImplementedError('Ambient SLDs other than 0 not implemented')

    if (len(thickness) + 1) == len(roughness) == len(scattering_length_density):
        number_of_interfaces = len(roughness)
    else:
        raise ValueError('Inconsistent number of layers')

    k_z0 = q_values.astype(np.complex128) / 2

    thickness_air = 1

    for interface in range(number_of_interfaces):

        prev_layer_idx = interface - 1
        next_layer_idx = interface

        if interface == 0:
            thickness_prev_layer = thickness_air
            k_z_previous_layer = _get_relative_k_z(k_z0, ambient_sld)
        else:
            thickness_prev_layer = thickness[prev_layer_idx] * np.ones_like(q_values, 'd')
            k_z_previous_layer = _get_relative_k_z(k_z0, scattering_length_density[prev_layer_idx])

        k_z_next_layer = _get_relative_k_z(k_z0, scattering_length_density[next_layer_idx])

        current_roughness = roughness[interface] * np.ones_like(q_values, 'd')

        reflection_matrix = _make_reflection_matrix(k_z_previous_layer, k_z_next_layer, current_roughness)

        if interface == 0:
            total_reflectivity_matrix = reflection_matrix
        else:
            translation_matrix = _make_translation_matrix(k_z_previous_layer, thickness_prev_layer)

            for n in range(len(q_values)):
                total_reflectivity_matrix[:, :, n] = np.matmul(total_reflectivity_matrix[:, :, n],
                                                               translation_matrix[:, :, n])
                total_reflectivity_matrix[:, :, n] = np.matmul(total_reflectivity_matrix[:, :, n],
                                                               reflection_matrix[:, :, n])

    r = np.zeros_like(q_values, 'D')

    for n in range(len(r)):
        r[n] = total_reflectivity_matrix[0, 1, n] / total_reflectivity_matrix[1, 1, n]

    reflectivity = np.clip(abs(r) ** 2, None, 1)
    reflectivity.reshape(len(reflectivity), 1)

    return reflectivity


def _get_relative_k_z(k_z0, scattering_length_density):
    k_z_rel = np.sqrt(k_z0 ** 2 - 4 * np.pi * scattering_length_density)

    return k_z_rel


def _make_reflection_matrix(k_z_previous_layer, k_z_next_layer, interface_roughness):
    p = (_safe_div((k_z_previous_layer + k_z_next_layer), (2 * k_z_previous_layer))
         * np.exp(-(k_z_previous_layer - k_z_next_layer) ** 2 * 0.5 * interface_roughness ** 2))

    m = (_safe_div((k_z_previous_layer - k_z_next_layer), (2 * k_z_previous_layer))
         * np.exp(-(k_z_previous_layer + k_z_next_layer) ** 2 * 0.5 * interface_roughness ** 2))

    R = np.array([[p, m],
                  [m, p]])

    return R


def _make_translation_matrix(k_z, thickness):
    T = np.array([[np.exp(-1j * k_z * thickness), np.zeros_like(k_z)],
                  [np.zeros_like(k_z), np.exp(1j * k_z * thickness)]])

    return T


def _safe_div(numerator, denominator):
    result = np.zeros_like(numerator, 'D')
    length = len(numerator)
    for i in range(length):

        if numerator[i] == denominator[i]:
            result[i] = 1
        elif denominator[i] == 0:
            result[i] = numerator[i] / 1e-20
        else:
            result[i] = numerator[i] / denominator[i]

    return result
