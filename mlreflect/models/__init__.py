from pathlib import Path

from . import model_helpers
from . import simple_model
from .trained_model import TrainedModel, DefaultTrainedModel

default_model_path = Path(__file__).parents[1] / Path('resources', 'models', 'default_trained_model_with_absorption.h5')
alternative_model_path = Path(__file__).parents[1] / Path('resources', 'models', 'different_model.h5')

__all__ = ['model_helpers', 'simple_model', 'TrainedModel', 'DefaultTrainedModel', 'default_model_path',
           'alternative_model_path']
