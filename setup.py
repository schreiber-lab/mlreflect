from distutils.util import convert_path

from setuptools import setup, find_packages

main_ns = {}
with open(convert_path('mlreflect/version.py')) as ver_file:
    exec(ver_file.read(), main_ns)

with open('requirements.txt') as file:
    requirements = file.readlines()

with open('README.md') as file:
    README = file.read()

setup(
    name='mlreflect',
    description='mlreflect is a Python package for training and using artificial neural networks to analyze '
                'specular X-ray and neutron reflectivity data. The training and usage of the neural network models is '
                'done via Keras as an API for TensorFlow.',
    version=main_ns['__version__'],
    long_description=README,
    long_description_content_type='text/markdown',
    author='Alessandro Greco',
    author_email='alessandro.greco@uni-tuebingen.de',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    python_requires='>=3.6'
)
