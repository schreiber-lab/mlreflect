from typing import Union, List

import pandas as pd
from numpy import ndarray
from pandas import DataFrame


def make_label_names(number_of_layers: int) -> List[str]:
    label_names = ['' for i in range(number_of_layers * 3)]

    for layer_index in range(number_of_layers):
        label_names[layer_index] = f'layer{number_of_layers - layer_index}_thickness'
        label_names[
            layer_index + number_of_layers] = f'layer{number_of_layers - layer_index}_roughness'
        label_names[
            layer_index + 2 * number_of_layers] = f'layer{number_of_layers - layer_index}_sld'

    return label_names


def convert_to_dataframe(labels: Union[DataFrame, ndarray], label_names: List[str]) -> DataFrame:
    if type(labels) is ndarray:
        label_df = pd.DataFrame(data=labels, columns=label_names)
    elif type(labels) is DataFrame:
        label_df = labels
    else:
        raise TypeError('Labels type must be ndarray or DataFrame.')

    return label_df
