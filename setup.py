from setuptools import setup, find_packages

requirements = [
    'torch>=2.0.0',
    'numpy',
    'matplotlib',
    'qiskit',
    'pylatexenc',
    'pytest',
    'scipy',
    'sympy',
    'svgwrite',
    'bayesian-optimization'
]

setup(
    name='deepquantum',
    version='0.3.3',
    packages=find_packages(),
    url='https://github.com/TuringQ/deepquantum',
    license='Apache License 2.0',
    author='TuringQ',
    install_requires=requirements,
    description='DeepQuantum for quantum computing'
)