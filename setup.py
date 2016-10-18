# -*- coding: utf-8 -*-

import sys
from setuptools import setup

if len(sys.argv) > 2 and sys.argv[-1] == '-useC':
    del sys.argv[-1]
    from setuptools import Extension
    from Cython.Build import cythonize
    
    setup(
        name='pyEPABC',
        version='1.0.0',
        author='Sebastian Bitzer',
        author_email='sebastian.bitzer@tu-dresden.de',
        packages=['pyEPABC'],
        description='A Python implementation of EP-ABC for likelihood-free, probabilistic inference.',
        install_requires=['numpy', 'scipy'],
        classifiers=[
                    'Development Status :: 5 - Production/Stable',
                    'Environment :: Console',
                    'Operating System :: OS Independent',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: BSD License',
                    'Programming Language :: Python :: 3',
                    'Topic :: Scientific/Engineering',
                     ],
        ext_modules = cythonize(Extension("testDDM_C", 
                                          ["examples/testDDM_C.pyx"],
                                          include_dirs=["src"],
                                          libraries=["DDMsampler"],
                                          library_dirs=["lib"], 
                                          runtime_library_dirs=["lib"]))
    )
else:
    setup(
        name='pyEPABC',
        version='1.0.0',
        author='Sebastian Bitzer',
        author_email='sebastian.bitzer@tu-dresden.de',
        packages=['pyEPABC'],
        description='A Python implementation of EP-ABC for likelihood-free, probabilistic inference.',
        install_requires=['numpy', 'scipy'],
        classifiers=[
                    'Development Status :: 5 - Production/Stable',
                    'Environment :: Console',
                    'Operating System :: OS Independent',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: BSD License',
                    'Programming Language :: Python :: 3',
                    'Topic :: Scientific/Engineering',
                     ]
    )