import numpy as np


def interp_reflectivity(q_interp, q, reflectivity):
    return 10 ** np.interp(q_interp, q, np.log10(reflectivity))
