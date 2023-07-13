from setuptools import setup, find_packages
from deepquantum import __version__

requirements = [
    'torch==2.0',
    'numpy',
    'matplotlib',
    'qiskit',
    'pylatexenc',
    'pytest',
    'scipy',
    'thewalrus'
]

setup(
    name='deepquantum',
    version=__version__,
    packages=find_packages(),
    url='',
    license='',
    author='TuringQ',
    install_requires=requirements,
    description='DeepQuantum for quantum computing'
)