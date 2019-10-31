import datetime
import os

import keras


def run_model(input_train, output_train, input_val, output_val):
    n_input = input_train.shape[1]
    n_output = output_train.shape[1]

    model_save_path = 'my_model.h5'

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(400, input_dim=n_input))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(800))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(400))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(300))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(200))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(100))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(n_output))
    model.add(keras.layers.Activation('relu'))

    model.summary()

    # compile model
    adam_optimizer = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False)

    model.compile(loss='mean_squared_error', optimizer=adam_optimizer)

    # Fit the model

    now = datetime.datetime.now()
    tb_logdir = os.path.join('graphs', now.strftime('%Y-%m-%d-%H%M%S'))

    tb_callback = keras.callbacks.TensorBoard(log_dir=tb_logdir, histogram_freq=0, write_graph=True, write_images=True)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_loss', verbose=1,
                                                 save_best_only=True)
    lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5)

    hist = model.fit(input_train, output_train, epochs=60, batch_size=256, verbose=1,
                     validation_data=(input_val, output_val),
                     callbacks=[checkpoint, tb_callback, lr_reduction])

    return hist
