from setuptools import setup, find_packages

with open('requirements.txt') as file:
    requirements = file.readlines()

setup(
    name='mlreflect',
    version='0.16.0',
    author='Alessandro Greco',
    author_email='alessandro.greco@uni-tuebingen.de',
    license='GPL3',
    packages=find_packages(),
    package_data={'mlreflect': ['models/default_trained_model.h5']},
    zip_safe=False,
    install_requires=requirements,
    python_requires='>=3.6'
)
