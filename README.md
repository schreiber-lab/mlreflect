# mlreflect

_mlreflect_ is a Python package for training and using artificial neural networks to analyze specular X-ray and 
neutron reflectivity data. The training and usage of the neural network models is done via Keras as an API for TensorFlow. 

## Installation
The mlreflect package can be installed directly from the command line using the python package manager pip:

`pip install mlreflect`

In case the newest version is not available on PyPI, the package can also be installed locally. Download the package, unzip it and navigate to the folder containing the downloaded mlreflect folder. Then use:

`pip install mlreflect/`

## Online documentation

Documentation and API reference can be found online on https://mlreflect.readthedocs.io/en/latest/

## Usage example
The package can then be imported in python using

`import mlreflect`

or

`from mlreflect import <module>`

An example of how to generate training data, train the model and test it on experimental data is shown in the 
_example/notebooks/training_example.ipynb_ Jupyter notebook.

An example of how to use the default pre-trained model for single layers on Si/SiOx substrates to fit data directly 
from a SPEC file is shown in _examples/notebooks/spec_usage_example.ipynb_ Jupyter notebook.

A detailed explanation as well as API info can be found in the documentation.

## Authors
#### Main author
- Alessandro Greco <alessandro.greco@uni-tuebingen.de> (Institut für Angewandte Physik, University of Tübingen)

#### Contributors
- Vladimir Starostin (Institut für Angewandte Physik, University of Tübingen)
- Christos Karapanagiotis (Institut für Physik, Humboldt Universität zu Berlin)
- Alexander Hinderhofer (Institut für Angewandte Physik, University of Tübingen)
- Alexander Gerlach (Institut für Angewandte Physik, University of Tübingen)
- Linus Pithan (ESRF The European Synchrotron)
- Sascha Liehr (Bundesanstalt für Materialforschung und -prüfung (BAM))
- Frank Schreiber (Institut für Angewandte Physik, University of Tübingen)
- Stefan Kowarik (Bundesanstalt für Materialforschung und -prüfung (BAM) and Institut für Physik, Humboldt Universität zu Berlin)
