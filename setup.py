"""
Setup configuration for installation via pip
"""

from setuptools import setup, find_packages


# The find_packages function does a lot of the heavy lifting for us w.r.t.
# discovering any Python packages we ship.
setup(
    name="pysplines",
    version="0.1.0.dev2",
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
)
