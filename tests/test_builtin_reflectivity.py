import unittest

import numpy as np
from mlreflect.data_generation.reflectivity import multilayer_reflectivity


class TestBuiltinReflectivity(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        q = np.linspace(0.01, 0.05, 1001) * 1e10
        ambient_sld = 0
        thickness = np.array([1200]) * 1e-10
        roughness = np.array([3, 10]) * 1e-10
        sld = np.array([6.36 + 0j, 4.66 + 0.016j]) * 1e14

        cls.refnx_test_refl = np.loadtxt('refnx_test_refl.dat')
        cls.builtin_refl = multilayer_reflectivity(q_values=q, thickness=thickness, roughness=roughness,
                                               scattering_length_density=sld, ambient_sld=ambient_sld)

    def test_multilayer_reflectivity(self):
        np.testing.assert_almost_equal(self.refnx_test_refl, self.builtin_refl)

    if __name__ == '__main__':
        unittest.main()
