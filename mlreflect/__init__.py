from . import h5_tools
from .data_generator import ReflectivityGenerator
from .layers import Layer, MultilayerStructure
from .prediction import Prediction
from .preprocessing import InputPreprocessor, OutputPreprocessor
from .reflectivity import multilayer_reflectivity
from . import noise

__version__ = '0.13.0'

__all__ = ['multilayer_reflectivity', 'Layer', 'MultilayerStructure', 'ReflectivityGenerator', 'InputPreprocessor',
           'OutputPreprocessor', 'Prediction', 'h5_tools', 'noise']

__author__ = "Alessandro Greco <alessandro.greco@uni-tuebingen.de>"
__credits__ = ["Vladimir Starostin", "Christos Karapanagiotis", "Linus Pithan", "Stefan Kowarik"]
