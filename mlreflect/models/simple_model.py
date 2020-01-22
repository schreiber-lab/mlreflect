from datetime import datetime
from typing import Union

import keras
import numpy as np
from numpy import ndarray
from pandas import DataFrame

from .model_helpers import make_tensorboard_callback, make_save_path


class SimpleModel:
    def __init__(self, directory_name: str, n_input: int, n_output: int):
        self.directory_name = directory_name

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(400, input_dim=n_input))
        self.model.add(keras.layers.Activation('relu'))

        self.model.add(keras.layers.Dense(800))
        self.model.add(keras.layers.Activation('relu'))

        self.model.add(keras.layers.Dense(400))
        self.model.add(keras.layers.Activation('relu'))

        self.model.add(keras.layers.Dense(300))
        self.model.add(keras.layers.Activation('relu'))

        self.model.add(keras.layers.Dense(200))
        self.model.add(keras.layers.Activation('relu'))

        self.model.add(keras.layers.Dense(100))
        self.model.add(keras.layers.Activation('relu'))

        self.model.add(keras.layers.Dense(n_output))
        self.model.add(keras.layers.Activation('relu'))

        self.model.summary()

        adam_optimizer = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0,
                                               amsgrad=False)

        self.model.compile(loss='mean_squared_error', optimizer=adam_optimizer)

    def train(self, input_train: ndarray, output_train: Union[DataFrame, ndarray], input_val: ndarray,
              output_val: Union[DataFrame, ndarray]):
        output_train = np.array(output_train)
        output_val = np.array(output_val)

        time_stamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')

        tb_callback = make_tensorboard_callback(self.directory_name, time_stamp)

        save_path = make_save_path(self.directory_name, time_stamp)

        checkpoint = keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss', verbose=1,
                                                     save_best_only=True)
        lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5)

        hist = self.model.fit(input_train, output_train, epochs=60, batch_size=256, verbose=1,
                              validation_data=(input_val, output_val),
                              callbacks=[checkpoint, tb_callback, lr_reduction])
        return hist, time_stamp
