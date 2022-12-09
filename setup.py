from setuptools import setup, find_packages

requirements = [
    "torch",
    "numpy"
]

setup(
    name='deepquantum',
    version='0.0.1',
    packages=find_packages(where="."),
    url='',
    license='',
    author='TuringQ',
    install_requires=requirements,
    description='DeepQuantum for quantum computing'
)