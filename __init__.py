# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab
'''
    GPR1D fit_data
'''

import warnings
import traceback

try:
    from .fit_data import fit_data
    del warnings, traceback, GPR1D, \
        Optimization_function, plot_data, print_data
except Exception as err:
    msg = str(traceback.format_exc())
    msg += "\n\n    => the optional package is not usable\n"
    warnings.warn(msg)
    del msg, err

__all__ = ['fit_data']
