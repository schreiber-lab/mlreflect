from .data_generation import noise
from .data_generation.data_generator import ReflectivityGenerator
from .data_generation.layers import Layer, ConstantLayer, AmbientLayer, Substrate
from .data_generation.multilayer import MultilayerStructure, LayerOnSubstrate
from .data_generation.parameters import Parameter, ConstantParameter
from .data_generation.reflectivity import multilayer_reflectivity
from .training.noise_generator import NoiseGenerator, NoiseGeneratorLog
from .training.prediction import Prediction
from .training.preprocessing import InputPreprocessor, OutputPreprocessor
from .utils import h5_tools
from .utils.check_gpu import check_gpu

__version__ = '0.15.0'

__all__ = ['multilayer_reflectivity', 'Parameter', 'ConstantParameter', 'Layer', 'ConstantLayer', 'AmbientLayer',
           'Substrate', 'MultilayerStructure', 'LayerOnSubstrate', 'ReflectivityGenerator', 'InputPreprocessor',
           'OutputPreprocessor', 'NoiseGenerator', 'Prediction', 'h5_tools', 'noise', 'check_gpu', 'NoiseGeneratorLog']

__author__ = "Alessandro Greco <alessandro.greco@uni-tuebingen.de>"
__credits__ = ["Vladimir Starostin", "Christos Karapanagiotis", "Linus Pithan", "Stefan Kowarik"]
