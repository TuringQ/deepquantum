from setuptools import setup, find_packages

requirements = [
    "torch==2.0",
    "numpy",
    "qiskit"
]

setup(
    name='deepquantum',
    version='0.0.2',
    packages=find_packages(where="."),
    url='',
    license='',
    author='TuringQ',
    install_requires=requirements,
    description='DeepQuantum for quantum computing'
)