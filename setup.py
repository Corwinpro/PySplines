"""
Setup configuration for installation via pip
"""

from setuptools import setup, find_packages

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

# The find_packages function does a lot of the heavy lifting for us w.r.t.
# discovering any Python packages we ship.
setup(
    name="pysplines",
    version="0.2.0",
    author="Petr Kungurtsev",
    packages=find_packages(),
    # PyPI packages required for the *installation* and usual running of the
    # tools.
    install_requires=["numpy", "sympy", "scipy"],
    # Metadata for PyPI (https://pypi.python.org).
    description="Tool to create discrete b-splines with surface properties",
    url="https://github.com/Corwinpro/PySplines",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
