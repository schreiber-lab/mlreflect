import tensorflow as tf
from packaging import version


def check_gpu():
    if version.parse(tf.__version__) < version.parse('2.1.0'):
        print(tf.test.is_gpu_available())
    else:
        print(tf.config.list_physical_devices('GPU'))


if __name__ == '__main__':
    check_gpu()
