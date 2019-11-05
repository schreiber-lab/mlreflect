import os
from datetime import datetime

from keras import callbacks as cb

from .. import h5_tools


def create_save_directory(folder_name: str):
    directory_name = h5_tools.strip_file_extension(folder_name)

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def make_save_path(model_folder: str):
    time_stamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')

    dirname = os.path.dirname(model_folder)
    model_name = os.path.basename(model_folder)
    model_name_wo_ext = h5_tools.strip_file_extension(model_name)

    path = os.path.join(dirname, model_name_wo_ext, model_name_wo_ext + '_' + time_stamp + '.h5')
    return path


def make_tensorboard_callback(model_folder: str, graphs_subfolder: str = 'tb_graphs'):
    time_stamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')

    dirname = os.path.dirname(model_folder)
    model_name = os.path.basename(model_folder)
    model_name_wo_ext = h5_tools.strip_file_extension(model_name)

    logdir = os.path.join(dirname, model_name_wo_ext, graphs_subfolder, time_stamp)

    create_save_directory(logdir)

    return cb.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)
