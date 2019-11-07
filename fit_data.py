# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab

from __future__ import (unicode_literals, absolute_import,print_function, division, with_statement)
# Standard python modules
import numpy as np
import os
import re
import sys
import traceback
import warnings
import time

# Local modules
import GPR1D
import imas
import pywed as pw
try:
    from .Optimization_function import Optimization
except Exception as err:
    from Optimization_function import Optimization
try:
    from .plot_data import plot_data
except Exception as err:
    from plot_data import plot_data
try:
    from .print_data import print_data
except Exception as err:
    from print_data import print_data
'''
    GPR1D fit data function
'''

__author__ = ['Mohamad Kozeiha', 'Jorge Morales']
__date__ = '10/09/2019'
__version__ = '$Revision: 1.10 $'

__all__ = ('fit_data')




def fit_data(X_coordinates, Y_coordinates, X_coordinates_errors=None, Y_coordinates_errors=None, \
             kernel_method='RQ_Kernel', optimise_all_params=False, slices_optim_nbr=10, nbr_pts=100, \
             slices_nbr=None, plot_fit=False, dx_data=[0.0], dy_data=[0.0], dy_err=[0.0]):
    '''
    Fit Y profile as a function of X quantity

    Parameters
    ----------
    X_coordinate : 2D array size (points, time)
        X coordinate of profile

    Y_coordinate : 2D array size (points, time)
        Y coordinate of profile

    X_coordinate_errors : 2D array size (points, time)
        X profile errors

    Y_coordinate_errors : 2D array size (points, time)
        Y profile errors

    kernel_method : string (default='RQ_Kernel')
        which kernel use for fit. One of

          ``RQ_Kernel``
            quadratic kernel for general purposes

          ``Gibbs_Kernel``
            IMPORTANT: use this kernel if profile contains a pedestal
            with option optimise_all_params=True

    optimise_all_params : boolean (default=False)
        If True optimise hyperparameters for all time slices in input (slow computation)

    slices_optim_nbr: int (default=10)
        If optimise_all_params=False maximum number of slices equally spaced where to
        perform search for optimised hyperparameters

    nbr_pts: int (default=100)
        Number of points of fitting curve

    slices_nbr: int
        Number of equally spaced slices in X and Y coordinate input arrays where fit is computed

    plot_fit: boolean (default=False)
        FOR TESTS ONLY. If True saves in sub folders fitted curves images

    Returns
    -------
    result : dictionary
        with keys:

            fit_x : size (points, time)
                Fit x coordinate

            fit_y : size (points, time)
                Fit y coordinate

            fit_y_error : size (points, time)
                Fit y error

            fit_dydx : size (points, time)
                Fit derivative dy/dx

            fit_dydx_y_error : size (points, time)
                Fit y error of derivative dy/dx

            x : size (points, time)
                Original x coordinate data

            y : size (points, time)
                Original y coordinate data

            x_error : size (points, time)
                Original x errors data

            y_error : size (points, time)
                Original y errors data
    '''
    import matplotlib.pyplot as plt
    #####################################################################################################

    if (Y_coordinates_errors is None):
        raise RuntimeError('Input Y errors are mandatory')


    kernel_methodlist=['RQ_Kernel','Matern_HI_Kernel', 'NN_Kernel', 'Gibbs_Kernel']
    print('Printing the list of kernel methods that should be used: ')
    print(kernel_methodlist)
    if kernel_method in kernel_methodlist:
        print('The chosed method is : ',kernel_method)
       
    else:
        raise ValueError("The Fit method is not know, please provide an method from the List")
        return


    #grab the obtimized values and use them in the fitting routine:
    if X_coordinates_errors is not None:#hyde kmn zedta w arrbt yalle t7t 
        start_time = time.time()
        print(start_time)
        if not optimise_all_params:
            print('computing the time for optimisating the ' , slices_optim_nbr,  'slices')
            optimized_values = Optimization(X_coordinates, Y_coordinates, X_coordinates_errors, \
                                            Y_coordinates_errors,  kernel_method, slices_optim_nbr)
            if kernel_method == 'RQ_Kernel': 
                default_configuartion = {
                    'RQ_Kernel'        : GPR1D.RQ_Kernel(optimized_values['nigp_fit_regpar_optimized']['amp'], optimized_values['nigp_fit_regpar_optimized']['ls'], optimized_values['nigp_fit_regpar_optimized']['alpha'])
                    }

            if kernel_method == 'Matern_HI_Kernel': 
                default_configuartion = {
                    'Matern_HI_Kernel' : GPR1D.Matern_HI_Kernel(optimized_values['nigp_fit_regpar_optimized']['amp'], optimized_values['nigp_fit_regpar_optimized']['ls'], optimized_values['nigp_fit_regpar_optimized']['alpha'])
                    }
            if kernel_method == 'NN_Kernel':
                default_configuartion = {
                    'NN_Kernel'        : GPR1D.NN_Kernel(optimized_values['nigp_fit_regpar_optimized']['amp'], optimized_values['nigp_fit_regpar_optimized']['ls'], optimized_values['nigp_fit_regpar_optimized']['alpha']),
                    }
            if kernel_method == 'Gibbs_Kernel':
                default_configuartion = {
                    'Gibbs_Kernel'     : GPR1D.Gibbs_Kernel(optimized_values['nigp_fit_regpar_optimized']['amp'], optimized_values['nigp_fit_regpar_optimized']['alpha'])
                    }

        else :

            default_configuartion = {
                'RQ_Kernel'  : GPR1D.RQ_Kernel(),
                'Matern_HI_Kernel'   : GPR1D.Matern_HI_Kernel(),
                'NN_Kernel'  : GPR1D.NN_Kernel(),
                'Gibbs_Kernel'  : GPR1D.Gibbs_Kernel()
                }


        print(kernel_method)
        print("--- %s seconds ---" % (time.time() - start_time))
    #hon ane zdt hyde 
    else:
      
        default_configuartion = {
            'RQ_Kernel'  : GPR1D.RQ_Kernel(),
            'Matern_HI_Kernel'   : GPR1D.Matern_HI_Kernel(),
            'NN_Kernel'  : GPR1D.NN_Kernel(),
            'Gibbs_Kernel'  : GPR1D.Gibbs_Kernel()
            }
        


    nbr_time = Y_coordinates.shape[0]

    fit_data = {'fit_x': [np.nan]*nbr_time, \
                'fit_y': [np.nan]*nbr_time, \
                'fit_x_error': [np.nan]*nbr_time, \
                'fit_y_error': [np.nan]*nbr_time, \
                'fit_dydx': [np.nan]*nbr_time, \
                'fit_dydx_x_error': [np.nan]*nbr_time, \
                'fit_dydy_y_error': [np.nan]*nbr_time, \
                'x': X_coordinates, \
                'y': Y_coordinates, \
                'x_error': X_coordinates_errors , \
                'y_error': Y_coordinates_errors , \
               }

    if (slices_nbr is None):
        slices_nbr = Y_coordinates.shape[0]

    for i in range(0, Y_coordinates.shape[0], int((Y_coordinates.shape[0])/(slices_nbr))):
        print('slice number : ', i)
        Y_reduced = Y_coordinates[i]
        X_reduced = X_coordinates[i]
        Y_errors = Y_coordinates_errors[i]
        
        if X_coordinates_errors is not None:
            X_errors = X_coordinates_errors[i] 
        else:
            X_errors =  np.full(X_coordinates.shape, np.mean(X_coordinates)*0.05)

        minimum = X_reduced.min()
        maximum = X_reduced.max()

        fit_x_values = np.linspace(minimum, maximum, nbr_pts)
        # Define a kernel to fit the data itself
        #     Rational quadratic kernel is usually robust enough for general fitting
        kernel =  default_configuartion.get(kernel_method)
        # This is only necessary if using kernel restart option on the data fitting
        kernel_hyppar_bounds = np.atleast_2d()

        # Define a kernel to fit the given y-errors, needed for rigourous estimation of fit error including data error
        #     Typically a simple rational quadratic kernel is sufficient given a high regularization parameter (specified later)
        #     Here, the RQ kernel is summed with a noise kernel for extra robustness and to demonstrate how to use operator kernels
        error_kernel = default_configuartion.get(kernel_method)

        # Again, this is only necessary if using kernel restart option on the error fitting
        error_kernel_hyppar_bounds = np.atleast_2d()
                
        if (plot_fit):
            # GPR fit using y-errors only as weights
            #     Create class object to store raw data, kernels, and settings
            gpr_object = GPR1D.GaussianProcessRegression1D()

            #     Define the kernel and regularization parameter to be used in the data fitting routine
            gpr_object.set_kernel(kernel=kernel)
            #     Define the raw data and associated errors to be fitted
            gpr_object.set_raw_data(xdata=X_reduced,ydata=Y_reduced,yerr=Y_errors,xerr=X_errors, \
                                        dxdata=dx_data, dydata=dy_data, dyerr=dy_err )     # Example of applying derivative constraints

            #     Define the search criteria for data fitting routine and error fitting routine
            if optimise_all_params:
                gpr_object.set_search_parameters(epsilon=1.0e-2)
                gpr_object.set_error_search_parameters(epsilon=1.0e-1)
            #     Default optimizer is gradient ascent / descent - extremely robust but slow
            #     Uncomment any of the following lines to test the recommended optimizers
            else:
                gpr_object.set_search_parameters(epsilon='None')
                gpr_object.set_error_search_parameters(epsilon='None')
            
            #     Perform the fit with kernel restarts
            gpr_object.GPRFit(fit_x_values,hsgp_flag=False)#,nrestarts=5)

            #     Grab optimized kernel settings - easy way to minimize data storage requirements for fit reproduction
            (gp_kernel_name,gp_kernel_hyppars,gp_fit_regpar) = gpr_object.get_gp_kernel_details()

            #     Grab fit results
            (fit_y_values,fit_y_errors,fit_dydx_values,fit_dydx_errors) = gpr_object.get_gp_results()

            #     Grab the log-marginal-likelihood of fit
            fit_lml = gpr_object.get_gp_lml()
            
        if (X_coordinates_errors is None or plot_fit):
            # GPR fit rigourously accounting only for y-errors (this is the recommended option)
            #     Procedure is nearly identical to above, except for the addition of an error kernel
            if not optimise_all_params:
                if kernel_method == 'Gibbs_Kernel':
                    hsgpr_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['hsgp_fit_regpar_optimized']['amp'] ,\
                                                                 optimized_values['hsgp_fit_regpar_optimized']['alpha']],\
                                                                [optimized_values['hsgp_fit_regpar_optimized']['amp'] ,\
                                                                 optimized_values['hsgp_fit_regpar_optimized']['alpha']],\
                                                                 ])
                    hsgpr_error_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['hsgp_error_fit_regpar_optimized']['amp'] ,\
                                                                       optimized_values['hsgp_error_fit_regpar_optimized']['alpha']],\
                                                                      [optimized_values['hsgp_error_fit_regpar_optimized']['amp'] ,\
                                                                       optimized_values['hsgp_error_fit_regpar_optimized']['alpha']],\
                                                                       ])
                elif kernel_method == 'Matern_HI_Kernel':
                    hsgpr_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['hsgp_fit_regpar_optimized']['amp'] ,\
                                                                 optimized_values['hsgp_fit_regpar_optimized']['ls'] ,\
                                                                 optimized_values['hsgp_fit_regpar_optimized']['alpha']],\
                                                                [optimized_values['hsgp_fit_regpar_optimized']['amp'] ,\
                                                                 optimized_values['hsgp_fit_regpar_optimized']['ls'] ,\
                                                                 optimized_values['hsgp_fit_regpar_optimized']['alpha']]])
                    hsgpr_error_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['hsgp_error_fit_regpar_optimized']['amp'] ,\
                                                                       optimized_values['hsgp_error_fit_regpar_optimized']['ls'] ,\
                                                                       optimized_values['hsgp_error_fit_regpar_optimized']['alpha']],\
                                                                      [optimized_values['hsgp_error_fit_regpar_optimized']['amp'] ,\
                                                                       optimized_values['hsgp_error_fit_regpar_optimized']['ls'] ,\
                                                                       optimized_values['hsgp_error_fit_regpar_optimized']['alpha']],\
                                                                       ])
                else : 
                    hsgpr_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['hsgp_fit_regpar_optimized']['amp'] ,\
                                                                 optimized_values['hsgp_fit_regpar_optimized']['ls'] ,\
                                                                 optimized_values['hsgp_fit_regpar_optimized']['alpha']],\
                                                                [optimized_values['hsgp_fit_regpar_optimized']['amp'] ,\
                                                                 optimized_values['hsgp_fit_regpar_optimized']['ls'] ,\
                                                                 optimized_values['hsgp_fit_regpar_optimized']['alpha']]])
                    hsgpr_error_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['hsgp_error_fit_regpar_optimized']['amp'] ,\
                                                                       optimized_values['hsgp_error_fit_regpar_optimized']['ls'] ,\
                                                                       optimized_values['hsgp_error_fit_regpar_optimized']['alpha']],\
                                                                      [optimized_values['hsgp_error_fit_regpar_optimized']['amp'] ,\
                                                                       optimized_values['hsgp_error_fit_regpar_optimized']['ls'] ,\
                                                                       optimized_values['hsgp_error_fit_regpar_optimized']['alpha']],\
                                                                       ])




            hsgpr_object = GPR1D.GaussianProcessRegression1D()
            hsgpr_object.set_raw_data(xdata=X_reduced,ydata=Y_reduced,yerr=Y_errors,xerr=X_errors, \
                                      dxdata=dx_data, dydata=dy_data, dyerr=dy_err )
            
            if optimise_all_params:
            #     Default optimizer is gradient ascent / descent - extremely robust but slow
            #     Uncomment any of the following lines to test the recommended optimizers
                hsgpr_object.set_kernel(kernel=kernel)
                hsgpr_object.set_error_kernel(kernel=error_kernel)
                hsgpr_object.set_search_parameters(epsilon=1.0e-2)
                hsgpr_object.set_error_search_parameters(epsilon=1.0e-1)
            else:
                hsgpr_object.set_kernel(kernel=kernel,kbounds=hsgpr_kernel_hyppar_bounds,\
                     regpar=optimized_values['hsgp_fit_regpar_optimized']['regularaiztion'])
                hsgpr_object.set_error_kernel(kernel=error_kernel,kbounds=hsgpr_error_kernel_hyppar_bounds,\
                     regpar=optimized_values['hsgp_error_fit_regpar_optimized']['regularaiztion'])
                hsgpr_object.set_search_parameters(epsilon='None')
                hsgpr_object.set_error_search_parameters(epsilon='None')

            #     Perform the fit with kernel restarts
            hsgpr_object.GPRFit(fit_x_values,hsgp_flag=True)#,nrestarts=5)

            #     Grab optimized kernel settings - easy way to minimize data storage requirements for fit reproduction
            (hsgp_kernel_name,hsgp_kernel_hyppars,hsgp_fit_regpar) = hsgpr_object.get_gp_kernel_details()
            (hsgp_error_kernel_name,hsgp_error_kernel_hyppars,hsgp_error_fit_regpar) = hsgpr_object.get_gp_error_kernel_details()

            #     Grab fit results
            (hs_fit_y_values,hs_fit_y_errors,hs_fit_dydx_values,hs_fit_dydx_errors) = hsgpr_object.get_gp_results()
            (hs_zfit_y_values,hs_zfit_y_errors,hs_zfit_dydx_values,hs_zfit_dydx_errors) = hsgpr_object.get_gp_results(noise_flag=False)

            #     Grab the log-marginal-likelihood of fit
            hs_fit_lml = hsgpr_object.get_gp_lml()

            ### Sampling distribution (only done with HSGP option)

            num_samples = 1000

            # Samples the fit distribution - smooth noise representation
            sample_array = hsgpr_object.sample_GP(num_samples,actual_noise=False)

            # Calculates the derivatives of the sampled fit distributions
            dfit_x_values = (fit_x_values[1:] + fit_x_values[:-1]) / 2.0
            deriv_array = (sample_array[:,1:] - sample_array[:,:-1]) / (fit_x_values[1:] - fit_x_values[:-1])
            # Samples the derivative distribution - smooth noise representation
            dsample_array = hsgpr_object.sample_GP_derivative(num_samples,actual_noise=False)

            # Calculates the integrals of the sampled derivative distributions
            ifit_x_values = dfit_x_values.copy()
            integ_array = dsample_array[:,1] * (ifit_x_values[0] - fit_x_values[0]) # + raw_intercept
            if integ_array.ndim == 1:
                integ_array = np.transpose(np.atleast_2d(integ_array))
            for jj in np.arange(1,dsample_array.shape[1]-1):
                integ = integ_array[:,jj-1] + dsample_array[:,jj] * (ifit_x_values[jj] - ifit_x_values[jj-1])
                if integ.ndim == 1:
                    integ = np.transpose(np.atleast_2d(integ))
                integ_array = np.hstack((integ_array,integ))
            # Integrals require renormalization to the fit mean to define the constant of integration that is lost
            orig_mean = np.nanmean(hs_fit_y_values)
            for ii in np.arange(0,num_samples):
                sint_mean = np.nanmean(integ_array[ii,:])
                integ_array[ii,:] = integ_array[ii,:] - sint_mean + orig_mean

            # Samples the fit distribution - true noise representation
            nsample_array = hsgpr_object.sample_GP(num_samples,actual_noise=True)

            # Samples the derivative distribution - true noise representation
            ndsample_array = hsgpr_object.sample_GP_derivative(num_samples,actual_noise=True)

            # Samples the fit distribution - zero noise representation
            zsample_array = hsgpr_object.sample_GP(num_samples,without_noise=True)

            # Calculates the derivatives of the sampled fit distributions - zero noise representation
            zderiv_array = (zsample_array[:,1:] - zsample_array[:,:-1]) / (fit_x_values[1:] - fit_x_values[:-1])

            # Samples the derivative distribution - zero noise representation
            #    Note that zero noise is only different from smooth noise if an error kernel is used
            zdsample_array = hsgpr_object.sample_GP_derivative(num_samples,without_noise=True)

            # Calculates the integrals of the sampled derivative distributions - zero noise representation
            zinteg_array = zdsample_array[:,1] * (ifit_x_values[0] - fit_x_values[0]) # + raw_intercept
            if zinteg_array.ndim == 1:
                zinteg_array = np.transpose(np.atleast_2d(zinteg_array))
            for jj in np.arange(1,zdsample_array.shape[1]-1):
                zinteg = zinteg_array[:,jj-1] + zdsample_array[:,jj] * (ifit_x_values[jj] - ifit_x_values[jj-1])
                if zinteg.ndim == 1:
                    zinteg = np.transpose(np.atleast_2d(zinteg))
                zinteg_array = np.hstack((zinteg_array,zinteg))
            # Integrals require renormalization to the fit mean to define the constant of integration that is lost
            zorig_mean = np.nanmean(hs_zfit_y_values)
            for ii in np.arange(0,num_samples):
                zsint_mean = np.nanmean(zinteg_array[ii,:])
                zinteg_array[ii,:] = zinteg_array[ii,:] - zsint_mean + zorig_mean

            # Computing statistics of sampled profiles
            sample_mean = np.nanmean(sample_array,axis=0)
            deriv_mean = np.nanmean(deriv_array,axis=0)
            dsample_mean = np.nanmean(dsample_array,axis=0)
            integ_mean = np.nanmean(integ_array,axis=0)
            sample_std = np.nanstd(sample_array,axis=0)
            deriv_std = np.nanstd(deriv_array,axis=0)
            dsample_std = np.nanstd(dsample_array,axis=0)
            integ_std = np.nanstd(integ_array,axis=0)

            # Computing statistics of sampled profiles - zero noise representation
            zsample_mean = np.nanmean(zsample_array,axis=0)
            zderiv_mean = np.nanmean(zderiv_array,axis=0)
            zdsample_mean = np.nanmean(zdsample_array,axis=0)
            zinteg_mean = np.nanmean(zinteg_array,axis=0)
            zsample_std = np.nanstd(zsample_array,axis=0)
            zderiv_std = np.nanstd(zderiv_array,axis=0)
            zdsample_std = np.nanstd(zdsample_array,axis=0)
            zinteg_std = np.nanstd(zinteg_array,axis=0)



        

        if (X_coordinates_errors is not None or plot_fit):
            # GPR fit rigourously accounting for y-errors AND x-errors
            #     Procedure is nearly identical to above, except for the addition of an extra option
            if not optimise_all_params:
                if kernel_method == 'Gibbs_Kernel':

                    nigpr_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['nigp_fit_regpar_optimized']['amp'] ,\
                                                                 optimized_values['nigp_fit_regpar_optimized']['alpha']],\
                                                                 [optimized_values['nigp_fit_regpar_optimized']['amp'] ,\
                                                                  optimized_values['nigp_fit_regpar_optimized']['alpha']],\
                                                                  ])
                    nigpr_error_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['nigp_error_fit_regpar_optimized']['amp'] ,\
                                                                       optimized_values['nigp_error_fit_regpar_optimized']['alpha']],\
                                                                      [optimized_values['nigp_error_fit_regpar_optimized']['amp'] ,\
                                                                       optimized_values['nigp_error_fit_regpar_optimized']['alpha']],\
                                                                       ])
                elif kernel_method == 'Matern_HI_Kernel':
                    nigpr_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['nigp_fit_regpar_optimized']['amp'] ,\
                                                                 optimized_values['nigp_fit_regpar_optimized']['ls'] ,\
                                                                 optimized_values['nigp_fit_regpar_optimized']['alpha']],\
                                                                [optimized_values['nigp_fit_regpar_optimized']['amp'] ,\
                                                                 optimized_values['nigp_fit_regpar_optimized']['ls'] ,\
                                                                 optimized_values['nigp_fit_regpar_optimized']['alpha']]])
                    nigpr_error_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['nigp_error_fit_regpar_optimized']['amp'] ,\
                                                                       optimized_values['nigp_error_fit_regpar_optimized']['ls'] ,\
                                                                       optimized_values['nigp_error_fit_regpar_optimized']['alpha']],\
                                                                      [optimized_values['nigp_error_fit_regpar_optimized']['amp'] ,\
                                                                       optimized_values['nigp_error_fit_regpar_optimized']['ls'] ,\
                                                                       optimized_values['nigp_error_fit_regpar_optimized']['alpha']],\
                                                                       ])

                else : 
                    nigpr_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['nigp_fit_regpar_optimized']['amp'] ,\
                                                                 optimized_values['nigp_fit_regpar_optimized']['ls'] ,\
                                                                 optimized_values['nigp_fit_regpar_optimized']['alpha']],\
                                                                [optimized_values['nigp_fit_regpar_optimized']['amp'] ,\
                                                                 optimized_values['nigp_fit_regpar_optimized']['ls'] ,\
                                                                 optimized_values['nigp_fit_regpar_optimized']['alpha']]])
                    nigpr_error_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['nigp_error_fit_regpar_optimized']['amp'] ,\
                                                                       optimized_values['nigp_error_fit_regpar_optimized']['ls'] ,\
                                                                       optimized_values['nigp_error_fit_regpar_optimized']['alpha']],\
                                                                      [optimized_values['nigp_error_fit_regpar_optimized']['amp'] ,\
                                                                       optimized_values['nigp_error_fit_regpar_optimized']['ls'] ,\
                                                                       optimized_values['nigp_error_fit_regpar_optimized']['alpha']],\
                                                                       ])

            nigpr_object = GPR1D.GaussianProcessRegression1D()
            nigpr_object.set_raw_data(xdata=X_reduced,ydata=Y_reduced,yerr=Y_errors,xerr=X_errors, \
                                      dxdata=dx_data, dydata=dy_data, dyerr=dy_err )
            if optimise_all_params:
                nigpr_object.set_kernel(kernel=kernel)
                nigpr_object.set_error_kernel(kernel=error_kernel)
                nigpr_object.set_search_parameters(epsilon=1.0e-2)
                nigpr_object.set_error_search_parameters(epsilon=1.0e-1)
            else:
                nigpr_object.set_search_parameters(epsilon='None')
                nigpr_object.set_error_search_parameters(epsilon='None')
                nigpr_object.set_kernel(kernel=kernel,kbounds=nigpr_kernel_hyppar_bounds,\
                     regpar=optimized_values['nigp_fit_regpar_optimized']['regularaiztion'])
                nigpr_object.set_error_kernel(kernel=error_kernel,kbounds=nigpr_error_kernel_hyppar_bounds,\
                     regpar=optimized_values['nigp_error_fit_regpar_optimized']['regularaiztion'])


            #     Perform the fit with kernel restarts, here is the extra option to account for x-errors in fit
            nigpr_object.GPRFit(fit_x_values,hsgp_flag=True,nigp_flag=True)#,nrestarts=5)
                # Grab outputs
            (nigp_kernel_name,nigp_kernel_hyppars,nigp_fit_regpar) = nigpr_object.get_gp_kernel_details()
            (nigp_error_kernel_name,nigp_error_kernel_hyppars,nigp_error_fit_regpar) = nigpr_object.get_gp_error_kernel_details()
            (ni_fit_y_values,ni_fit_y_errors,ni_fit_dydx_values,ni_fit_dydx_errors) = nigpr_object.get_gp_results()
            ni_fit_lml = nigpr_object.get_gp_lml()
 


        if plot_fit:
            print_data(gp_kernel_name,gp_kernel_hyppars,gp_fit_regpar,fit_lml,\
                           hsgp_kernel_name,hsgp_fit_regpar,hsgp_kernel_hyppars,\
                           hsgp_error_kernel_name,hsgp_error_fit_regpar,hsgp_error_kernel_hyppars,hs_fit_lml,\
                           nigp_kernel_name,nigp_fit_regpar,nigp_kernel_hyppars,\
                           nigp_error_kernel_name,nigp_error_fit_regpar,nigp_error_kernel_hyppars,ni_fit_lml)
           
            plot_data(i, fit_x_values, minimum, maximum, \
                          X_reduced, Y_reduced, \
                          Y_errors, X_errors, \
                          hs_fit_y_values, hs_fit_y_errors,\
                          hs_fit_dydx_errors,hs_fit_dydx_values, \
                          ni_fit_y_values, ni_fit_dydx_values,  ni_fit_y_errors, ni_fit_dydx_errors,\
                          sample_array, sample_mean, sample_std, \
                          deriv_array, deriv_mean, deriv_std,\
                          dsample_array, dsample_mean, dsample_std,\
                          integ_array, integ_mean, integ_std, ifit_x_values,\
                          ndsample_array, dfit_x_values,  \
                          hs_zfit_y_values, hs_zfit_y_errors,\
                          zsample_array, zsample_mean, zsample_std,\
                          hs_zfit_dydx_values, hs_zfit_dydx_errors,\
                          zderiv_array, zderiv_mean, zderiv_std,\
                          zdsample_array, zdsample_mean, zdsample_std,\
                          zinteg_array, zinteg_mean, zinteg_std, \
                          nsample_array, \
                          fit_y_values, fit_y_errors, \
                          fit_dydx_values, fit_dydx_errors)
            
        # Results
        # -------
        if (X_coordinates_errors is None):
            fit_data['fit_x'][i]            = fit_x_values
            fit_data['fit_y'][i]            = hs_fit_y_values
            fit_data['fit_y_error'][i]      = hs_fit_y_errors
            fit_data['fit_dydx'][i]         = fit_dydx_values
            fit_data['fit_dydy_y_error'][i] = hs_fit_dydx_errors
        elif (X_coordinates_errors is not None):
            fit_data['fit_x'][i]            = fit_x_values
            fit_data['fit_y'][i]            = ni_fit_y_values
            fit_data['fit_y_error'][i]      = ni_fit_y_errors
            fit_data['fit_dydx'][i]         = ni_fit_dydx_values
            fit_data['fit_dydy_y_error'][i] = ni_fit_dydx_errors
        
    return fit_data
