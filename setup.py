from setuptools import setup, find_packages

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
    version='0.2.3',
    packages=find_packages(),
    url='',
    license='',
    author='TuringQ',
    install_requires=requirements,
    description='DeepQuantum for quantum computing'
)