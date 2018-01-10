#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy >= 1.11', 'statsmodels', 'numba']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='replay_identification',
    version='0.0.1.dev0',
    license='MIT',
    description=('Identify replay events'),
    author='Eric Denovellis',
    author_email='edeno@bu.edu',
    url='https://github.com/Eden-Kramer-Lab/replay_identification',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
