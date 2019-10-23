Gaussian process regression fit routine for IMAS
================================================

If you need help please contact: Mohamad Kozeiha (mohamad.kozeiha@cea.fr)
or Jorge Morales (jorge.morales2@cea.fr)

GPR1D
=====

Installing the GPR1D program
----------------------------

*Author: Aaron Ho (01/06/2018)*

Installation is **mandatory** for this package!

For first time users, it is strongly recommended to use the GUI
developed for this Python package. To obtain the Python package
dependencies needed to use this capability, install this package
by using the following on the command line::

    pip install [--user] GPR1D[guis]

Use the :code:`--user` flag if you do not have root access on the system
that you are working on. If you have already cloned the
repository, enter the top level of the repository directory and
use the following instead::

    pip install [--user] -e .[guis]

Removal of the :code:`[guis]` portion will no longer check for
the GUI generation and plotting packages needed for this
functionality. However, these packages are not crucial for the
base classes and algorithms.


Documentation
=============

Documentation of the equations used in the algorithm, along with
the available kernels and optimizers, can be found in docs/.
Documentation of the GPR1D module can be found on
`GitLab pages <https://aaronkho.gitlab.io/GPR1D>`_


Using the GPR1D program
-----------------------

For those who wish to include the functionality of this package
into their own Python scripts, a demo script is provided in
scripts/. The basic syntax used to create kernels, select
settings, and perform GPR fits are outlined there.

In addition, a simplified GPR1D class is available for those
wishing to distill the parameters into a subset of the most
crucial ones.

For any questions or to report bugs, please do so through the
proper channels in the GitLab repository.


*Important note for users!*

The following runtime warnings are common within this routine,
but they are filtered out by default::

    RuntimeWarning: overflow encountered in double_scalars
    RuntimeWarning: invalid value encountered in true_divide
    RuntimeWarning: invalid value encountered in sqrt


They normally occur when using the kernel restarts option (as
in the demo) and do not necessarily mean that the resulting
fit is poor.

Plotting the resulting fit and errors is the recommended way to
check its quality. The log-marginal-likelihood metric can also
be used, but is only valuable when comparing different fits of
the same data, ie. its absolute value is meaningless.

From v1.1.1, the adjusted R\ :sup:`2` and pseudo R\ :sup:`2`
metrics are now available. The adjusted R\ :sup:`2` metric provides
a measure of how close the fit is to the input data points. The
pseudo R\ :sup:`2` provides a measure of this closeness accounting
for the input data uncertainties.
