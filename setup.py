from setuptools import setup, find_packages

requirements = [
    'torch==2.0',
    'numpy',
    'matplotlib',
    'qiskit',
    'pylatexenc',
    'pytest',
    'scipy',
    'svgwrite'
]

setup(
    name='deepquantum',
    version='0.3.0',
    packages=find_packages(),
    url='',
    license='',
    author='TuringQ',
    install_requires=requirements,
    description='DeepQuantum for quantum computing'
)