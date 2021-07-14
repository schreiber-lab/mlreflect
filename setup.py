from setuptools import setup, find_packages
from distutils.util import convert_path

main_ns = {}
with open(convert_path('mlreflect/version.py')) as ver_file:
    exec(ver_file.read(), main_ns)


with open('requirements.txt') as file:
    requirements = file.readlines()

setup(
    name='mlreflect',
    version=main_ns['__version__'],
    author='Alessandro Greco',
    author_email='alessandro.greco@uni-tuebingen.de',
    license='GPL3',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    python_requires='>=3.6'
)
