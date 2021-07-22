.. mlreflect documentation master file, created by
   sphinx-quickstart on Tue Jul 13 17:43:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


mlreflect - Fast fitting of X-ray and neutron reflectivity data using AI
========================================================================
*mlreflect* is a Python package for training and using artificial neural networks
designed to analyze specular X-ray and neutron reflectivity data. The training and
usage of the neural network models is done via *Keras* as an API for *TensorFlow*.
If installed, the simulation of reflectivity curves is done via the C-based simulation
of the refl1D package (if it is not installed, a built-in Python-based simulation is
used.
Other data operations and optimizations are done via the *numpy* and *scipy.optimize*
packages.

The advantage of data analysis using neural networks is the speed with which results
are obtained, which is usually far exceeds any typical minimization algorithm. This
is particularly advantagous for fast screening of the data or on-the-fly analysis
during real-time experiments.

Currently, the package is optimized for single-layer problems. After training the
neural neutwork model on a particular box model sample layout for a given *q*
range, it can predict thickness, roughness and scattering length density within 
milliseconds per reflectivity curve.


.. toctree::
   :maxdepth: 2

   installation
   examples
   modules





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
