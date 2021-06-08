from typing import Union, List

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame


def convert_to_dataframe(labels: Union[DataFrame, ndarray], label_names: List[str]) -> DataFrame:
    if type(labels) is ndarray:
        label_df = pd.DataFrame(data=labels.copy(), columns=label_names)
    elif type(labels) is DataFrame:
        label_df = labels.copy()
    else:
        raise TypeError('Labels type must be ndarray or DataFrame.')

    return label_df


def convert_to_ndarray(labels: Union[DataFrame, ndarray]) -> ndarray:
    if type(labels) is ndarray:
        label_array = labels.copy()
    elif type(labels) is DataFrame:
        label_array = np.array(labels)
    else:
        raise TypeError('Labels type must be ndarray or DataFrame.')

    return label_array
