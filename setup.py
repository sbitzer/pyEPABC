# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize



setup(
    ext_modules = cythonize(Extension("testDDM_C", 
                                      ["examples/testDDM_C.pyx"],
                                      include_dirs=["src"],
                                      libraries=["DDMsampler"],
                                      library_dirs=["lib"], 
                                      runtime_library_dirs=["lib"]))
)