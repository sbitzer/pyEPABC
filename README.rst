pyEPABC
=======
A Python implementation of EP-ABC for likelihood-free, probabilistic inference.

:Author: Sebastian Bitzer
:Contact: sebastian.bitzer@tu-dresden.de

This code is based on Simon Barthelmé's `Matlab implementation`_
of EP-ABC as described in the corresponding article_. See the documentation 
of ``run_EPABC`` for further details.

This implementation requires Python 3.

Examples
--------

The ``examples``-folder contains two examples demonstrating the usage of 
pyEPABC: ``testGauss`` and ``testDDM``. Instructions for how to run these 
examples follow.

Simple Gaussian example
.......................
This example compares EPABC to an analytic solution for the simplest case you 
can imagine: Estimating the mean of some data points. The simplicity of the
problem allows that the posterior distribution and log marginal likelihood can
be computed analytically so that the EP-ABC estimates can be compared to that.

I assume that you have downloaded (or cloned) the pyEPABC git-project into a
folder ``pyEPABC_git``. To run this example open an ipython shell and navigate 
to ``pyEPABC_git``. Then::

	import pyEPABC
	run examples/testGauss

You can change parameters of EP-ABC, or the problem, e.g., the dimensionality of
the data points, in ``testGauss.py`` to investigate the behaviour of EP-ABC.

This examples requires: numpy, scipy, pandas, matplotlib.

Drift-diffusion model example
.............................
This example fits a drift-diffusion model (DDM) to an example data set
consisting of reaction times and choices recorded in a psychophysics 
experiment. The results are compared to the results produced by HDDM_.

Additional to showing the working of EP-ABC on this kind of problem, this
example also demonstrates how pyEPABC can be integrated with simulation
functions implemented in C. The idea is that your simulation function is
provided by an externally compiled shared library. This example shows how you
can link this code for use in pyEPABC. Note that, if you are implementing a new
model, i.e., simulation function, from scratch you may want to consider
alternatives for this pure-C path. See the section on `Binary Extensions`_ in
the `Python Packaging User Guide`_ for further information.

Before you can run the example you need to compile a few things. I'll only
explain how to do this in Linux. First, compile the 'external' library which 
will provide super-fast simulations from a DDM. Do this by opening a shell and
navigating to the ``pyEPABC_git/src`` folder. There run::

	mkdir ../lib
	gcc -shared -fPIC -O3 -lc -o ../lib/libDDMsampler.so DDMsampler.c brownian_motion_simulation.c

which will place the resulting shared library into ``pyEPABC_git/lib``. Then,
compile the Python-C-interface::

	python setup.py build_ext -i

which will generate another shared library ``testDDM_C.xxx.so`` where ``xxx`` will
depend on your python setup. This library can be directly imported into python
as a module, as done in ``testDDM.py``. To run the example open an ipython shell,
navigate to the ``pyEPABC_git`` folder and execute::

	import pyEPABC
	run examples/testDDM

This example requires: numpy, scipy, pandas, matplotlib, seaborn, cython, hddm.


Narrow posteriors
-----------------
EP-ABC sometimes produces too narrow posteriors. See
``examples/narrow_posteriors.ipynb`` for an explanation, how to identify this
issue and how to prevent it.

.. _`Matlab implementation`: https://sites.google.com/site/simonbarthelme/software
.. _article: https://doi.org/10.1080/01621459.2013.864178
.. _HDDM: http://ski.clps.brown.edu/hddm_docs/
.. _`Binary Extensions`: https://packaging.python.org/en/latest/extensions/
.. _`Python Packaging User Guide`: https://packaging.python.org/en/latest/
