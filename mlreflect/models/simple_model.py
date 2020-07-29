from typing import Union

import numpy as np
import tensorflow.keras as keras
from numpy import ndarray
from pandas import DataFrame

from .model_helpers import make_tensorboard_callback, make_save_path
from ..utils import naming


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
              output_val: Union[DataFrame, ndarray], epochs=60):
        *callbacks, time_stamp = self._setup_training()

        output_train = np.array(output_train)
        output_val = np.array(output_val)
        hist = self.model.fit(input_train, output_train, validation_data=(input_val, output_val), epochs=epochs,
                              batch_size=256, verbose=1, callbacks=callbacks)
        return hist, time_stamp

    def train_with_generator(self, data_generator_train, data_generator_val, epochs=60):
        *callbacks, time_stamp = self._setup_training()

        hist = self.model.fit(data_generator_train, validation_data=data_generator_val, epochs=epochs, verbose=1,
                              callbacks=callbacks)
        return hist, time_stamp

    def _setup_training(self):
        time_stamp = naming.make_timestamp()

        tb_callback = make_tensorboard_callback(self.directory_name, time_stamp)

        save_path = make_save_path(self.directory_name, time_stamp)

        checkpoint = keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss', verbose=1,
                                                     save_best_only=True)
        lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5)

        return tb_callback, checkpoint, lr_reduction, time_stamp
