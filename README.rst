=========
PySplines
=========

Python / Sympy implementation of Uniform rational B-splines. 

The supposed usage of the PySplines is CAD-style parameter-based shape generation for meshes and shape optimization.

The package focuses on:

- Easy generation of B-splines with given control points and degree,
- Access to the surface properties in analytical and numerical forms (normals, curvature, displacement fields)
- Smoothness of the curve and uniformity of the points distribution along the curve
- Fast access to the surface properties, especially when they are called multiple times

Sphinx documentation available here: `http://petrkungurtsev.me/PySplines`__.

.. __: http://petrkungurtsev.me/PySplines

Demos
-------------

Please refer to the ``demo`` folder for some use cases.

Installation
-------------

With pip:

.. code-block:: rst

    python -m pip install pysplines


