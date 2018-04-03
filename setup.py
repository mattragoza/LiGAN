from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='cgenerate',
	  ext_modules=cythonize('cgenerate.pyx'))

# python setup.py build_ext --inplace
