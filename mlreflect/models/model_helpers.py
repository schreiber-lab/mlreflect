import os

from tensorflow.keras import callbacks as cb

from ..utils import h5_tools


def create_save_directory(directory_name: str):
    directory_name = h5_tools.strip_file_extension(directory_name)

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def make_save_path(model_directory: str, time_stamp: str):
    dirname = os.path.dirname(model_directory)
    model_name = os.path.basename(model_directory)
    model_name_wo_ext = h5_tools.strip_file_extension(model_name)

    save_directory = os.path.join(dirname, model_name_wo_ext, 'models')

    create_save_directory(save_directory)

    path = os.path.join(save_directory, time_stamp + '.h5')
    return path


def make_tensorboard_callback(model_directory: str, time_stamp: str, graphs_subdirectory: str = 'tb_graphs'):
    dirname = os.path.dirname(model_directory)
    model_name = os.path.basename(model_directory)
    model_name_wo_ext = h5_tools.strip_file_extension(model_name)

    logdir = os.path.join(dirname, model_name_wo_ext, graphs_subdirectory, time_stamp)

    create_save_directory(logdir)

    return cb.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)
