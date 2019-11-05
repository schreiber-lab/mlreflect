from setuptools import setup

setup(
    name='mlreflect',
    version='0.10.0',
    author='Alessandro Greco',
    author_email='alessandro.greco@uni-tuebingen.de',
    license='GPL3',
    packages=['mlreflect'],
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