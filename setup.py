from setuptools import setup, find_packages

requirements = [
    'torch==2.1.2',
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
    version='0.3.1',
    packages=find_packages(),
    url='',
    license='',
    author='TuringQ',
    install_requires=requirements,
    description='DeepQuantum for quantum computing'
)