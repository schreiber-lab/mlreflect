from .data_generation import noise
from .data_generation.data_generator import ReflectivityGenerator
from .data_generation.layers import Layer, MultilayerStructure
from .data_generation.reflectivity import multilayer_reflectivity
from .training.prediction import Prediction
from .training.preprocessing import InputPreprocessor, OutputPreprocessor
from .utils import h5_tools

__version__ = '0.13.1'

__all__ = ['multilayer_reflectivity', 'Layer', 'MultilayerStructure', 'ReflectivityGenerator', 'InputPreprocessor',
           'OutputPreprocessor', 'Prediction', 'h5_tools', 'noise']

__author__ = "Alessandro Greco <alessandro.greco@uni-tuebingen.de>"
__credits__ = ["Vladimir Starostin", "Christos Karapanagiotis", "Linus Pithan", "Stefan Kowarik"]
