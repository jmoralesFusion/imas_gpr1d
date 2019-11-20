# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab
'''
    Main equimap
'''
#from __future__ import (unicode_literals, absolute_import,  \
#                        print_function, division)
import numpy as np
import os
import scipy.io
import sys

#print('path 1 =', sys.path)
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#print('path 2 =', sys.path)

# Local modules
try:
    from .fit_data import fit_data
except Exception as err:
    from fit_data import fit_data

# Set
#np.set_printoptions(precision=15, threshold=np.nan)

if __name__ == '__main__':

    # Parse input arguments
    fileRandomName = sys.argv[1]

    in_eq = scipy.io.loadmat(fileRandomName)

    x_coord = np.atleast_1d(np.squeeze(in_eq['x_coord'])).astype(float)
    y_coord = np.atleast_1d(np.squeeze(in_eq['y_coord'])).astype(float)
    x_coord_err = np.atleast_1d(np.squeeze(in_eq['x_coord_err'])).astype(float)
    y_coord_err = np.atleast_1d(np.squeeze(in_eq['y_coord_err'])).astype(float)
    kernel_method = str(np.squeeze(in_eq['kernel_method']))
    optimise_all_params = bool(np.squeeze(in_eq['optimise_all_params']))
    slices_optim_nbr = int(in_eq['slices_optim_nbr'])
    nbr_pts = int(in_eq['nbr_pts'])
    slices_nbr = int(in_eq['slices_nbr'])
    plot_fit = bool(np.squeeze(in_eq['plot_fit']))
    dx_data = np.atleast_1d(np.squeeze(in_eq['dx_data'])).astype(float)
    dy_data = np.atleast_1d(np.squeeze(in_eq['dy_data'])).astype(float)
    dy_err = np.atleast_1d(np.squeeze(in_eq['dy_err'])).astype(float)
   
    if (x_coord_err.size == 1 and x_coord_err[0] == 0):
        x_coord_err = None
    if (slices_nbr == 0):
        slices_nbr = None
    # CALL
    out = fit_data(X_coordinates=x_coord, \
                   Y_coordinates=y_coord, \
                   X_coordinates_errors=x_coord_err, \
                   Y_coordinates_errors=y_coord_err, \
                   kernel_method=kernel_method, \
                   optimise_all_params=optimise_all_params, \
                   slices_optim_nbr=slices_optim_nbr, \
                   nbr_pts=nbr_pts, \
                   slices_nbr=slices_nbr, \
                   plot_fit=plot_fit, \
                   dx_data=dx_data, dy_data=dy_data, dy_err=dy_err)

    scipy.io.savemat('out' + fileRandomName, out)
