Gaussian process regression fit routine for IMAS
================================================
https://github.com/jmoralesFusion/imas_gpr1d

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




############################################################################################################
#### running the commands:
Running the fitting routine:

It is only needed to launch the following command from the terminal:

python imas_gpr1d.py 55564 -ids=interferometer -k=Gibbs_Kernel -wep

where :
1) imas_gpr1d.py is the script responsible for the studying of the data
2) 55564 is the shot number, you can replace it with whatever shot you want
3) -ids=interferometer : is the type of the diagnostics used to take the measurements./
   It should be one of the following: [reflectometer_profile, ece, interferometer]
4) -k=Gibbs_Kernel : is the type of the kernel used to fit the data, it should be either/
   Gibbs_Kernel or RQ_Kernel
5) -wep : refers to write edge profiles, this means if we want to save the reconstructed /
   profiles to the database or not. so removing the term -wep will not allow saving the /
   data to west db 

once choosing the interferometer diagnostics, it is worth noting that the work becomes 
complicated and therefore you will find a little description of the fitting procedure within 
the following lines. It is also worth mentiong that posing the optin interferometr enables 
the complete reconstruction of the profile density form rho =0 to rho = 1


The result will be located in the following directories that is also created by the fitting/
routine under a name that is combination between the shot number and the diagnostic name. 
The use of the fit_fucntion.py has occurred three times and each time results a different fit/
function that corresponds to the requested data by the routine.


0)A combination of the measurements from both interferometery and reflectometry diagnostics is 
    done and therefore a complete full reconstruction of the Line Integrated Density is calculated 
    and therefore passed to the fitting routine

1)fitting the Line Integrated density as a function of normalized rho 

2)a series of interpolations and transoprting the density to rho mid plane is then done, and another
    measurements are then taken for the Line integrated density asa function of raduis in meters.

3)The data are then passed to the second fit_function and the derivative of the results is taken 
      to be used as an input to the final fit function


4)the data taken are then mapped again from R-space to rho-space and passed to the final fit function
that gives the final profile density that we need in the following directory 
/55564_interferometer_data/GPPlots_final_FITS/GPPlots_final_FITS_0.png

############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

there exist also several other functions responsible fro plotting, printing and visualizing the 
data that we have.