#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="pysparkplug",
    version="2.0.0.0",
    description="A package for estimating heterogeneous probability density functions.",
    author="Adam Walder",
    author_email="walder2@llnl.gov",
    url="N/A",
    packages=find_packages(),
    long_description="""\
    A package for estimating heterogeneous probability density functions.
    """,
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
    ],
    keywords="machine learning density estimation statistics heterogeneous data",
    license="BSD",
    install_requires=[
        "scipy",
        "matplotlib",
        "numpy",
        "numba",
        "mpmath",
        "pandas",
        "bokeh",
        "pyspark",
        "tbb",
    ],
    # ext_package='pysp',
    # ext_modules=[Extension('c_ext', ['extensions/lda.c'])],
)
