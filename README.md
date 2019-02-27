# PySplines

Python / Sympy implementation of Uniform rational B-splines.

The supposed usage of the PySplines is CAD-style parameter-based shape generation for meshes and shape optimization.

The package focuses on:

    - Easy generation of B-splines with given control points and degree,
    - Access to the surface properties in analytical and numerical forms (normals, curvature, displacement fields)
    - Smoothness of the curve and uniformity of the points distribution along the curve
    - Fast access to the surface properties, especially when they are called multiple times

## Demos

Please refer to the ```demo``` folder for some use cases.

## Installation

With pip:

```bash
git clone git@github.com:Corwinpro/PySplines.git
cd bspline
pip install --user -e .
```

