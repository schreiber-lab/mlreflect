from setuptools import setup, find_packages
import mlreflect

setup(
    name='mlreflect',
    version=mlreflect.__version__,
    author='Alessandro Greco',
    author_email='alessandro.greco@uni-tuebingen.de',
    license='GPL3',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'pandas',
        'numpy',
        'keras',
        'h5py',
        'tqdm',
        'typing',
    ]
)