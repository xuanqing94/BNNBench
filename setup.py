#!/usr/bin/env python
from setuptools import find_packages, setup

with open("README.md", "r") as mdread:
    readme = mdread.read()

setup(
    name="BNNBench",
    version="0.0.1",
    description="A benchmark suite for Bayesian neural networks",
    author="Xuanqing Liu",
    author_email="xqliu@cs.ucla.edu",
    packages=find_packages(exclude=["tests"]),
    long_description=readme,
    long_description_content_type="text/markdown",
    setup_requires=["setuptools>=18.0"],
    install_requires=[],
    dependency_links=[],
    test_suite="tests",
    entry_points={"console_scripts": []},
    include_package_data=True,
)
