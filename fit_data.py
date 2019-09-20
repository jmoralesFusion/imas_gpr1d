# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab

from __future__ import (unicode_literals, absolute_import,print_function, division)
# Standard python modules
import numpy as np
import os
import re
import sys
import warnings
import time
# Local modules
import GPR1D
import imas
import pywed as pw
from Optimization_function import Optimization
'''
    GPR1D fit data function
'''

__author__ = ['Mohamad Kozeiha', 'Jorge Morales']
__date__ = '10/09/2018'
__version__ = '$Revision: 1.10 $'

__all__ = ('fit_data',
           'default_config')

default_configuartion = {
    'Constant_Kernel' : GPR1D.Constant_Kernel(),
    'Noise_Kernel'  : GPR1D.Noise_Kernel(),
    'Linear_Kernel'   : GPR1D.Linear_Kernel(),
    'Poly_Order_Kernel'   : GPR1D.Poly_Order_Kernel(),
    'SE_Kernel'   : GPR1D.SE_Kernel(),
    'RQ_Kernel'  : GPR1D.RQ_Kernel(),
    'Matern_HI_Kernel'   : GPR1D.Matern_HI_Kernel(),
    'NN_Kernel'  : GPR1D.NN_Kernel(),
    'Gibbs_Kernel'  : GPR1D.Gibbs_Kernel()
    }





def fit_data(X_coordinates, Y_coordinates, kernel_method='RQ_Kernel', print_stat = False, plot_fit =False):
    '''
    Fit Y profile as a function of X quantity

    Parameters
    ----------
    X_coordinate : 2D array size (points, time)
        X coordinate of profile
    Y_coordinate : 2D array size (points, time)
        Y coordinate of profile
    kernel_method : string
        which kernel use for fit. One of

          ``RQ_Kernel``
            quadratic kernel...

          ``Gibbs_Kernel``
            ...

    Returns
    -------
    result : dictionary
        with keys:

            fit_x : size (points, time)
                Fit x coordinate

            fit_y : size (points, time)
                Fit y coordinate

            fit_x_error : size (points, time)
                Fit y error

            fit_y_error : size (points, time)
                Fit y error

            fit_dydx : size (points, time)
                Fit derivative dy/dx

            fit_dydx_x_error : size (points, time)
                Fit x error of derivative dy/dx

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



    kernel_methodlist=['Constant_Kernel', 'Noise_Kernel', 'Linear_Kernel', 'Poly_Order_Kernel', 'SE_Kernel', 'RQ_Kernel',
                       'Matern_HI_Kernel', 'NN_Kernel', 'Gibbs_Kernel']
    print('Printing the list of kernel methods that should be used: ')
    print(kernel_methodlist)
    if kernel_method in kernel_methodlist:
        print('The chosed method is : ',kernel_method)
    else:
        raise ValueError("The Fit method is not know, please provide an method from the List")
        return


    #np.savetxt('x_y_eror.txt',(X_coordinates[:,10],Y_coordinates[:,10],np.full(Y_coordinates[:,10].shape,100)))
    '''
    file = open("list.txt", "w")
    for index in range(len(X_coordinates[:,10])):
        file.write(str(X_coordinates[:,10][index]) + " " + str(Y_coordinates[:,10][index]) + " " + str(np.full(Y_coordinates[:,10].shape,100)[index])+ "\n")
    file.close()
    '''
    #grab the obtimized values and use them in the fitting routine:
    print('computing the time of 10 slices')
    start_time = time.time()
    optimized_values = Optimization(X_coordinates, Y_coordinates,  kernel_method='RQ_Kernel')
    print("--- %s seconds ---" % (time.time() - start_time))
    
    nbr_pts  = 100
    nbr_time = Y_coordinates.shape[1]

    fit_data = {'fit_x': np.full((nbr_pts, nbr_time), np.nan), \
                'fit_y': np.full((nbr_pts, nbr_time), np.nan), \
                'fit_x_error': np.full((nbr_pts, nbr_time), np.nan), \
                'fit_y_error': np.full((nbr_pts, nbr_time), np.nan), \
                'fit_dydx': np.full((nbr_pts, nbr_time), np.nan), \
                'fit_dydx_x_error': np.full((nbr_pts, nbr_time), np.nan), \
                'fit_dydy_y_error': np.full((nbr_pts, nbr_time), np.nan), \
                'x': X_coordinates, \
                'y': Y_coordinates, \
                'x_error': np.full((nbr_pts, nbr_time), np.nan), \
                'y_error': np.full((nbr_pts, nbr_time), np.nan), \
               }
    #for i in range(Y_coordinates.shape[1]):

    for i in range(1000):
        print('slice number : ', i)
        Y_reduced = Y_coordinates[ :,i]
        X_reduced = (X_coordinates)[:,i]

        Y_errors = np.full(Y_reduced.shape, np.mean(Y_reduced)*0.05)
        minimum = X_reduced.min()
        maximum = X_reduced.max()
        X_errors =  np.full(X_reduced.shape,0.0091)

        fit_x_values = np.linspace(minimum,maximum,100)
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

        

        
        

        # GPR fit using y-errors only as weights
        #     Create class object to store raw data, kernels, and settings
        gpr_object = GPR1D.GaussianProcessRegression1D()

        #     Define the kernel and regularization parameter to be used in the data fitting routine
        gpr_object.set_kernel(kernel=kernel)

        #     Define the raw data and associated errors to be fitted
        gpr_object.set_raw_data(xdata=X_reduced,ydata=Y_reduced,yerr=Y_errors,xerr=X_errors, \
                                    dxdata=[0.0],dydata=[0.0],dyerr=[0.0])     # Example of applying derivative constraints

        #     Define the search criteria for data fitting routine and error fitting routine
        gpr_object.set_search_parameters(epsilon=1.0e-2)
        gpr_object.set_error_search_parameters(epsilon=1.0e-1)
        #     Default optimizer is gradient ascent / descent - extremely robust but slow
        #     Uncomment any of the following lines to test the recommended optimizers
        # gpr_object.set_search_parameters(epsilon=1.0e-2,method='adam',spars=[1.0e-1,0.4,0.8])
        # gpr_object.set_error_search_parameters(epsilon=1.0e-1,method='adam',spars=[1.0e-1,0.4,0.8])

        #     Perform the fit with kernel restarts
        gpr_object.GPRFit(fit_x_values,hsgp_flag=False,nrestarts=5)

        #     Grab optimized kernel settings - easy way to minimize data storage requirements for fit reproduction
        (gp_kernel_name,gp_kernel_hyppars,gp_fit_regpar) = gpr_object.get_gp_kernel_details()

        #     Grab fit results
        (fit_y_values,fit_y_errors,fit_dydx_values,fit_dydx_errors) = gpr_object.get_gp_results()

        #     Grab the log-marginal-likelihood of fit
        fit_lml = gpr_object.get_gp_lml()




        

        # GPR fit rigourously accounting only for y-errors (this is the recommended option)
        #     Procedure is nearly identical to above, except for the addition of an error kernel
        hsgpr_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['hsgp_fit_regpar_optimized']['amp'] ,\
                                                         optimized_values['hsgp_fit_regpar_optimized']['ls'] ,\
                                                         optimized_values['hsgp_fit_regpar_optimized']['alpha'] ],\
                                                        [ optimized_values['hsgp_fit_regpar_optimized']['amp'],\
                                                              optimized_values['hsgp_fit_regpar_optimized']['ls'] ,\
                                                              optimized_values['hsgp_fit_regpar_optimized']['alpha'] ]])
        hsgpr_error_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['hsgp_error_fit_regpar_optimized']['amp'] ,\
                                                               optimized_values['hsgp_error_fit_regpar_optimized']['ls'] ,\
                                                               optimized_values['hsgp_error_fit_regpar_optimized']['alpha'] ],\
                                                              [ optimized_values['hsgp_error_fit_regpar_optimized']['amp'],\
                                                                    optimized_values['hsgp_error_fit_regpar_optimized']['ls'] ,\
                                                                    optimized_values['hsgp_error_fit_regpar_optimized']['alpha'] ]])
        
        hsgpr_object = GPR1D.GaussianProcessRegression1D()
        hsgpr_object.set_kernel(kernel=kernel,kbounds=hsgpr_kernel_hyppar_bounds,\
                                    regpar=optimized_values['hsgp_fit_regpar_optimized']['regularaiztion'])
        hsgpr_object.set_error_kernel(kernel=error_kernel,kbounds=hsgpr_error_kernel_hyppar_bounds,\
                                          regpar=optimized_values['hsgp_error_fit_regpar_optimized']['regularaiztion'])
        hsgpr_object.set_raw_data(xdata=X_reduced,ydata=Y_reduced,yerr=Y_errors,xerr=X_errors, \
                                      dxdata=[0.0],dydata=[0.0],dyerr=[0.0])
        hsgpr_object.set_search_parameters(epsilon=1.0e-2)
        hsgpr_object.set_error_search_parameters(epsilon=1.0e-1)
        
        #     Default optimizer is gradient ascent / descent - extremely robust but slow
        #     Uncomment any of the following lines to test the recommended optimizers
        #hsgpr_object.set_search_parameters(epsilon=1.0e-2,method='adam',spars=[1.0e-1,0.4,0.8])
        #hsgpr_object.set_error_search_parameters(epsilon=1.0e-1,method='adam',spars=[1.0e-1,0.4,0.8])

        #     Perform the fit with kernel restarts
        hsgpr_object.GPRFit(fit_x_values,hsgp_flag=True,nrestarts=5)

        #     Grab optimized kernel settings - easy way to minimize data storage requirements for fit reproduction
        (hsgp_kernel_name,hsgp_kernel_hyppars,hsgp_fit_regpar) = hsgpr_object.get_gp_kernel_details()
        (hsgp_error_kernel_name,hsgp_error_kernel_hyppars,hsgp_error_fit_regpar) = hsgpr_object.get_gp_error_kernel_details()

        #     Grab fit results
        (hs_fit_y_values,hs_fit_y_errors,hs_fit_dydx_values,hs_fit_dydx_errors) = hsgpr_object.get_gp_results()
        (hs_zfit_y_values,hs_zfit_y_errors,hs_zfit_dydx_values,hs_zfit_dydx_errors) = hsgpr_object.get_gp_results(noise_flag=False)

        #     Grab the log-marginal-likelihood of fit
        hs_fit_lml = hsgpr_object.get_gp_lml()


        

        # GPR fit rigourously accounting for y-errors AND x-errors
        #     Procedure is nearly identical to above, except for the addition of an extra option
        nigpr_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['nigp_fit_regpar_optimized']['amp'] ,\
                                                               optimized_values['nigp_fit_regpar_optimized']['ls'] ,\
                                                               optimized_values['nigp_fit_regpar_optimized']['alpha'] ],\
                                                              [ optimized_values['nigp_fit_regpar_optimized']['amp'],\
                                                                    optimized_values['nigp_fit_regpar_optimized']['ls'] ,\
                                                                    optimized_values['nigp_fit_regpar_optimized']['alpha'] ]])
        nigpr_error_kernel_hyppar_bounds = np.atleast_2d([[optimized_values['nigp_error_fit_regpar_optimized']['amp'] ,\
                                                               optimized_values['nigp_error_fit_regpar_optimized']['ls'] ,\
                                                               optimized_values['nigp_error_fit_regpar_optimized']['alpha'] ],\
                                                              [ optimized_values['nigp_error_fit_regpar_optimized']['amp'],\
                                                                    optimized_values['nigp_error_fit_regpar_optimized']['ls'] ,\
                                                                    optimized_values['nigp_error_fit_regpar_optimized']['alpha'] ]])
        
        nigpr_object = GPR1D.GaussianProcessRegression1D()
        nigpr_object.set_kernel(kernel=kernel,kbounds=nigpr_kernel_hyppar_bounds,\
                                    regpar=optimized_values['nigp_fit_regpar_optimized']['regularaiztion'])
        nigpr_object.set_error_kernel(kernel=error_kernel,kbounds=nigpr_error_kernel_hyppar_bounds,\
                                          regpar=optimized_values['nigp_error_fit_regpar_optimized']['regularaiztion'])
        nigpr_object.set_raw_data(xdata=X_reduced,ydata=Y_reduced,yerr=Y_errors,xerr=X_errors, \
                                      dxdata=[0.0],dydata=[0.0],dyerr=[0.0])
        nigpr_object.set_search_parameters(epsilon=1.0e-2)
        nigpr_object.set_error_search_parameters(epsilon=1.0e-1)

        #     Uncomment any of the following lines to test the recommended optimizers
        #nigpr_object.set_search_parameters(epsilon=1.0e-2,method='adam',spars=[1.0e-1,0.4,0.8])
        #nigpr_object.set_error_search_parameters(epsilon=1.0e-1,method='adam',spars=[1.0e-1,0.4,0.8])

        #     Perform the fit with kernel restarts, here is the extra option to account for x-errors in fit
        nigpr_object.GPRFit(fit_x_values,hsgp_flag=True,nigp_flag=True,nrestarts=5)

        # Grab outputs
        (nigp_kernel_name,nigp_kernel_hyppars,nigp_fit_regpar) = nigpr_object.get_gp_kernel_details()
        (nigp_error_kernel_name,nigp_error_kernel_hyppars,nigp_error_fit_regpar) = nigpr_object.get_gp_error_kernel_details()
        (ni_fit_y_values,ni_fit_y_errors,ni_fit_dydx_values,ni_fit_dydx_errors) = nigpr_object.get_gp_results()
        ni_fit_lml = nigpr_object.get_gp_lml()
        
        
        
        
        
        
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





        if print_stat == True:
            ### Printing

            gp_str = "\n--- GPR Fit ---\n\n"
            gp_str = gp_str + "Kernel name: %30s\n" % (gp_kernel_name)
            gp_str = gp_str + "Regularization parameter: %17.4f\n" % (gp_fit_regpar)
            gp_str = gp_str + "Optimized kernel hyperparameters:\n"
            for hh in np.arange(0,gp_kernel_hyppars.size):
                gp_str = gp_str + "%15.6e" % (gp_kernel_hyppars[hh])
            gp_str = gp_str + "\n\n"
            gp_str = gp_str + "Log-marginal-likelihood: %18.6f\n" % (fit_lml)

            print(gp_str)

            hsgp_str = "\n--- HSGPR Fit ---\n\n"
            hsgp_str = hsgp_str + "Kernel name: %30s\n" % (hsgp_kernel_name)
            hsgp_str = hsgp_str + "Regularization parameter: %17.4f\n" % (hsgp_fit_regpar)
            hsgp_str = hsgp_str + "Optimized kernel hyperparameters:\n"
            for hh in np.arange(0,hsgp_kernel_hyppars.size):
                hsgp_str = hsgp_str + "%15.6e" % (hsgp_kernel_hyppars[hh])
            hsgp_str = hsgp_str + "\n\n"
            hsgp_str = hsgp_str + "Error kernel name: %24s\n" % (hsgp_error_kernel_name)
            hsgp_str = hsgp_str + "Regularization parameter: %17.4f\n" % (hsgp_error_fit_regpar)
            hsgp_str = hsgp_str + "Optimized error kernel hyperparameters:\n"
            for hh in np.arange(0,hsgp_error_kernel_hyppars.size):
                hsgp_str = hsgp_str + "%15.6e" % (hsgp_error_kernel_hyppars[hh])
            hsgp_str = hsgp_str + "\n\n"
            hsgp_str = hsgp_str + "Log-marginal-likelihood: %18.6f\n" % (hs_fit_lml)

            print(hsgp_str)

            nigp_str = "--- NIGPR Fit ---\n\n"
            nigp_str = nigp_str + "Kernel name: %30s\n" % (nigp_kernel_name)
            nigp_str = nigp_str + "Regularization parameter: %17.4f\n" % (nigp_fit_regpar)
            nigp_str = nigp_str + "Optimized kernel hyperparameters:\n"
            for hh in np.arange(0,nigp_kernel_hyppars.size):
                nigp_str = nigp_str + "%15.6e" % (nigp_kernel_hyppars[hh])
            nigp_str = nigp_str + "\n\n"
            nigp_str = nigp_str + "Error kernel name: %24s\n" % (nigp_error_kernel_name)
            nigp_str = nigp_str + "Regularization parameter: %17.4f\n" % (nigp_error_fit_regpar)
            nigp_str = nigp_str + "Optimized error kernel hyperparameters:\n"
            for hh in np.arange(0,nigp_error_kernel_hyppars.size):
                nigp_str = nigp_str + "%15.6e" % (nigp_error_kernel_hyppars[hh])
            nigp_str = nigp_str + "\n\n"
            nigp_str = nigp_str + "Log-marginal-likelihood: %18.6f\n" % (ni_fit_lml)

            print(nigp_str)




        if plot_fit == True:
            ### Some basic setup

            plot_save_directory = './bebe' + str(i)
            if not plot_save_directory.endswith('/'):
                plot_save_directory = plot_save_directory+'/'
            if not os.path.isdir(plot_save_directory):
                os.makedirs(plot_save_directory)

            ### Fitting
            fig = plt.figure()
            fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
            ax = fig.add_subplot(111)
            plt.plot(X_reduced, Y_reduced)
            fig.savefig(plot_save_directory+'data.png')
            plt.close(fig)


            ### Plotting

            plt = None
            try:
                import matplotlib.pyplot as plt
            except:
                plt = None

            if plt is not None:

                plot_num_samples = 10
                plot_sigma = 2.0

                # Raw data with GPR fit and error, only accounting for y-errors
                plot_Y_errors = plot_sigma * Y_errors
                fig = plt.figure()
                fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.errorbar(X_reduced,Y_reduced,yerr=plot_Y_errors,ls='',marker='.',color='b')
                ax.plot(fit_x_values,hs_fit_y_values,color='r')
                plot_hs_fit_y_lower = hs_fit_y_values - plot_sigma * hs_fit_y_errors
                plot_hs_fit_y_upper = hs_fit_y_values + plot_sigma * hs_fit_y_errors
                ax.fill_between(fit_x_values,plot_hs_fit_y_lower,plot_hs_fit_y_upper,facecolor='r',edgecolor='None',alpha=0.2)
                ax.set_xlim(minimum,maximum)
                fig.savefig(plot_save_directory+'gp_data.png')
                plt.close(fig)

                # Derivative of GPR fit and error, only accounting for y-errors
                fig = plt.figure()
                fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.plot(fit_x_values,fit_dydx_values,color='r')
                plot_hs_fit_dydx_lower = hs_fit_dydx_values - plot_sigma * hs_fit_dydx_errors
                plot_hs_fit_dydx_upper = hs_fit_dydx_values + plot_sigma * hs_fit_dydx_errors
                ax.fill_between(fit_x_values,plot_hs_fit_dydx_lower,plot_hs_fit_dydx_upper,facecolor='r',edgecolor='None',alpha=0.2)
                ax.set_xlim(minimum,maximum)
                fig.savefig(plot_save_directory+'gp_derivative_data.png')
                plt.close(fig)

                # Raw data with GPR fit and error, comparison of using y-errors as weights, rigourously accounting for y-errors, and rigourously account for y-errors AND x-errors
                plot_X_errors = plot_sigma * X_errors
                fig = plt.figure()
                fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.errorbar(X_reduced,Y_reduced,xerr=plot_X_errors,yerr=plot_Y_errors,ls='',marker='.',color='k')
                ax.plot(fit_x_values,fit_y_values,color='g')
                plot_fit_y_lower = fit_y_values - plot_sigma * fit_y_errors
                plot_fit_y_upper = fit_y_values + plot_sigma * fit_y_errors
                ax.plot(fit_x_values,plot_fit_y_lower,color='g',ls='--')
                ax.plot(fit_x_values,plot_fit_y_upper,color='g',ls='--')
                ax.fill_between(fit_x_values,plot_fit_y_lower,plot_fit_y_upper,facecolor='g',edgecolor='None',alpha=0.2)
                ax.plot(fit_x_values,hs_fit_y_values,color='r')
                ax.plot(fit_x_values,plot_hs_fit_y_lower,color='r',ls='--')
                ax.plot(fit_x_values,plot_hs_fit_y_upper,color='r',ls='--')
                ax.fill_between(fit_x_values,plot_hs_fit_y_lower,plot_hs_fit_y_upper,facecolor='r',edgecolor='None',alpha=0.2)
                ax.plot(fit_x_values,ni_fit_y_values,color='b')
                plot_ni_fit_y_lower = ni_fit_y_values - plot_sigma * ni_fit_y_errors
                plot_ni_fit_y_upper = ni_fit_y_values + plot_sigma * ni_fit_y_errors
                ax.plot(fit_x_values,plot_ni_fit_y_lower,color='b',ls='--')
                ax.plot(fit_x_values,plot_ni_fit_y_upper,color='b',ls='--')
                ax.fill_between(fit_x_values,plot_ni_fit_y_lower,plot_ni_fit_y_upper,facecolor='b',edgecolor='None',alpha=0.2)
                ax.set_xlim(minimum,maximum)
                fig.savefig(plot_save_directory+'gp_options_test.png')
                plt.close(fig)

                # Derivative of GPR fit and error, comparison of using y-errors as weights, rigourously accounting for y-errors, and rigourously account for y-errors AND x-errors
                fig = plt.figure()
                fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.plot(fit_x_values,fit_dydx_values,color='g')
                plot_fit_dydx_lower = fit_dydx_values - plot_sigma * fit_dydx_errors
                plot_fit_dydx_upper = fit_dydx_values + plot_sigma * fit_dydx_errors
                ax.plot(fit_x_values,plot_fit_dydx_lower,color='g',ls='--')
                ax.plot(fit_x_values,plot_fit_dydx_upper,color='g',ls='--')
                ax.fill_between(fit_x_values,plot_fit_dydx_lower,plot_fit_dydx_upper,facecolor='g',edgecolor='None',alpha=0.2)
                ax.plot(fit_x_values,hs_fit_dydx_values,color='r')
                ax.plot(fit_x_values,plot_hs_fit_dydx_lower,color='r',ls='--')
                ax.plot(fit_x_values,plot_hs_fit_dydx_upper,color='r',ls='--')
                ax.fill_between(fit_x_values,plot_hs_fit_dydx_lower,plot_hs_fit_dydx_upper,facecolor='r',edgecolor='None',alpha=0.2)
                ax.plot(fit_x_values,ni_fit_dydx_values,color='b')
                plot_ni_fit_dydx_lower = ni_fit_dydx_values - plot_sigma * ni_fit_dydx_errors
                plot_ni_fit_dydx_upper = ni_fit_dydx_values + plot_sigma * ni_fit_dydx_errors
                ax.plot(fit_x_values,plot_ni_fit_dydx_lower,color='b',ls='--')
                ax.plot(fit_x_values,plot_ni_fit_dydx_upper,color='b',ls='--')
                ax.fill_between(fit_x_values,plot_ni_fit_dydx_lower,plot_ni_fit_dydx_upper,facecolor='b',edgecolor='None',alpha=0.2)
                ax.set_xlim(minimum,maximum)
                fig.savefig(plot_save_directory+'gp_options_dtest.png')
                plt.close(fig)

                # Sampled fit curves (smooth noise) against GPR fit distribution
                fig = plt.figure()
                fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.fill_between(fit_x_values,plot_hs_fit_y_lower,plot_hs_fit_y_upper,facecolor='r',edgecolor='None',alpha=0.2)
                for ii in np.arange(0,plot_num_samples):
                    ax.plot(fit_x_values,sample_array[ii,:],color='k',alpha=0.5)
                plot_hs_sample_y_lower = sample_mean - plot_sigma * sample_std
                plot_hs_sample_y_upper = sample_mean + plot_sigma * sample_std
                ax.fill_between(fit_x_values,plot_hs_sample_y_lower,plot_hs_sample_y_upper,facecolor='b',edgecolor='None',alpha=0.2)
                ax.set_xlim(minimum,maximum)
                fig.savefig(plot_save_directory+'sample_gp_test.png')
                plt.close(fig)

                # Derivatives of sampled fit curves (smooth noise) against GPR fit derivative distribution
                fig = plt.figure()
                fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.fill_between(fit_x_values,plot_hs_fit_dydx_lower,plot_hs_fit_dydx_upper,facecolor='r',edgecolor='None',alpha=0.2)
                for ii in np.arange(0,plot_num_samples):
                    ax.plot(dfit_x_values,deriv_array[ii,:],color='k',alpha=0.5)
                plot_hs_sample_dydx_lower = deriv_mean - plot_sigma * deriv_std
                plot_hs_sample_dydx_upper = deriv_mean + plot_sigma * deriv_std
                ax.fill_between(dfit_x_values,plot_hs_sample_dydx_lower,plot_hs_sample_dydx_upper,facecolor='b',edgecolor='None',alpha=0.2)
                ax.set_xlim(minimum,maximum)
                fig.savefig(plot_save_directory+'sample_gp_drv_test.png')
                plt.close(fig)

                # Sampled fit derivative curves (smooth noise) against GPR fit derivative distribution
                fig = plt.figure()
                fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.fill_between(fit_x_values,plot_hs_fit_dydx_lower,plot_hs_fit_dydx_upper,facecolor='r',edgecolor='None',alpha=0.2)
                for ii in np.arange(0,plot_num_samples):
                    ax.plot(fit_x_values,dsample_array[ii,:],color='k',alpha=0.5)
                plot_hs_dsample_dydx_lower = dsample_mean - plot_sigma * dsample_std
                plot_hs_dsample_dydx_upper = dsample_mean + plot_sigma * dsample_std
                ax.fill_between(fit_x_values,plot_hs_dsample_dydx_lower,plot_hs_dsample_dydx_upper,facecolor='b',edgecolor='None',alpha=0.2)
                ax.set_xlim(minimum,maximum)
                fig.savefig(plot_save_directory+'sample_gp_dtest.png')
                plt.close(fig)

                # Integrals of sampled fit derivative curves (smooth noise) against GPR fit distribution
                fig = plt.figure()
                fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.fill_between(fit_x_values,plot_hs_fit_y_lower,plot_hs_fit_y_upper,facecolor='r',edgecolor='None',alpha=0.2)
                for ii in np.arange(0,plot_num_samples):
                    ax.plot(ifit_x_values,integ_array[ii,:],color='k',alpha=0.5)
                plot_hs_dsample_y_lower = integ_mean - plot_sigma * integ_std
                plot_hs_dsample_y_upper = integ_mean + plot_sigma * integ_std
                ax.fill_between(ifit_x_values,plot_hs_dsample_y_lower,plot_hs_dsample_y_upper,facecolor='b',edgecolor='None',alpha=0.2)
                ax.set_xlim(minimum,maximum)
                fig.savefig(plot_save_directory+'sample_gp_itg_dtest.png')
                plt.close(fig)

                # Sampled fit curves (true noise) against GPR fit distribution
                fig = plt.figure()
                fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.fill_between(fit_x_values,plot_hs_fit_y_lower,plot_hs_fit_y_upper,facecolor='r',edgecolor='None',alpha=0.2)
                for ii in np.arange(0,plot_num_samples):
                    ax.plot(fit_x_values,nsample_array[ii,:],color='k',alpha=0.5)
                ax.set_xlim(minimum,maximum)
                fig.savefig(plot_save_directory+'sample_gp_noisy_test.png')
                plt.close(fig)

                # Sampled fit derivative curves (true noise) against GPR fit derivative distribution
                fig = plt.figure()
                fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.fill_between(fit_x_values,plot_hs_fit_dydx_lower,plot_hs_fit_dydx_upper,facecolor='r',edgecolor='None',alpha=0.2)
                for ii in np.arange(0,plot_num_samples):
                    ax.plot(fit_x_values,ndsample_array[ii,:],color='k',alpha=0.5)
                ax.set_xlim(minimum,maximum)
                fig.savefig(plot_save_directory+'sample_gp_noisy_dtest.png')
                plt.close(fig)

                # Sampled fit curves (zero noise) against GPR fit distribution
                fig = plt.figure()
                fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                plot_hs_zfit_y_lower = hs_zfit_y_values - plot_sigma * hs_zfit_y_errors
                plot_hs_zfit_y_upper = hs_zfit_y_values + plot_sigma * hs_zfit_y_errors
                ax.fill_between(fit_x_values,plot_hs_zfit_y_lower,plot_hs_zfit_y_upper,facecolor='r',edgecolor='None',alpha=0.2)
                for ii in np.arange(0,plot_num_samples):
                    ax.plot(fit_x_values,zsample_array[ii,:],color='k',alpha=0.5)
                plot_hs_zsample_y_lower = zsample_mean - plot_sigma * zsample_std
                plot_hs_zsample_y_upper = zsample_mean + plot_sigma * zsample_std
                ax.fill_between(fit_x_values,plot_hs_sample_y_lower,plot_hs_sample_y_upper,facecolor='b',edgecolor='None',alpha=0.2)
                ax.set_xlim(minimum,maximum)
                fig.savefig(plot_save_directory+'sample_gp_no_noise_test.png')
                plt.close(fig)

                # Derivatives of sampled fit curves (zero noise) against GPR fit derivative distribution
                fig = plt.figure()
                fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                plot_hs_zfit_dydx_lower = hs_zfit_dydx_values - plot_sigma * hs_zfit_dydx_errors
                plot_hs_zfit_dydx_upper = hs_zfit_dydx_values + plot_sigma * hs_zfit_dydx_errors
                ax.fill_between(fit_x_values,plot_hs_zfit_dydx_lower,plot_hs_zfit_dydx_upper,facecolor='r',edgecolor='None',alpha=0.2)
                for ii in np.arange(0,plot_num_samples):
                    ax.plot(dfit_x_values,zderiv_array[ii,:],color='k',alpha=0.5)
                plot_hs_zsample_dydx_lower = zderiv_mean - plot_sigma * zderiv_std
                plot_hs_zsample_dydx_upper = zderiv_mean + plot_sigma * zderiv_std
                ax.fill_between(dfit_x_values,plot_hs_zsample_dydx_lower,plot_hs_zsample_dydx_upper,facecolor='b',edgecolor='None',alpha=0.2)
                ax.set_xlim(minimum,maximum)
                fig.savefig(plot_save_directory+'sample_gp_drv_no_noise_test.png')
                plt.close(fig)

                # Sampled fit derivative curves (zero noise) against GPR fit derivative distribution
                fig = plt.figure()
                fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.fill_between(fit_x_values,plot_hs_zfit_dydx_lower,plot_hs_zfit_dydx_upper,facecolor='r',edgecolor='None',alpha=0.2)
                for ii in np.arange(0,plot_num_samples):
                    ax.plot(fit_x_values,zdsample_array[ii,:],color='k',alpha=0.5)
                plot_hs_zdsample_dydx_lower = zdsample_mean - plot_sigma * zdsample_std
                plot_hs_zdsample_dydx_upper = zdsample_mean + plot_sigma * zdsample_std
                ax.fill_between(fit_x_values,plot_hs_zdsample_dydx_lower,plot_hs_zdsample_dydx_upper,facecolor='b',edgecolor='None',alpha=0.2)
                ax.set_xlim(minimum,maximum)
                fig.savefig(plot_save_directory+'sample_gp_no_noise_dtest.png')
                plt.close(fig)

                # Integrals of sampled fit derivative curves (zero noise) against GPR fit distribution
                fig = plt.figure()
                fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.fill_between(fit_x_values,plot_hs_zfit_y_lower,plot_hs_zfit_y_upper,facecolor='r',edgecolor='None',alpha=0.2)
                for ii in np.arange(0,plot_num_samples):
                    ax.plot(ifit_x_values,zinteg_array[ii,:],color='k',alpha=0.5)
                plot_hs_zdsample_y_lower = zinteg_mean - plot_sigma * zinteg_std
                plot_hs_zdsample_y_upper = zinteg_mean + plot_sigma * zinteg_std
                ax.fill_between(ifit_x_values,plot_hs_zdsample_y_lower,plot_hs_zdsample_y_upper,facecolor='b',edgecolor='None',alpha=0.2)
                ax.set_xlim(minimum,maximum)
                fig.savefig(plot_save_directory+'sample_gp_itg_no_noise_dtest.png')
                plt.close(fig)

                print("Results of demonstration plotted in directory ./GPPlots/\n")

            else:

                print("   Module matplotlib not found. Skipping plotting of demonstration results.\n")

            print("Demonstration script successfully completed!\n")

        # Results
        # -------
        fit_data['fit_x'][:, i]            = fit_x_values
        fit_data['fit_y'][:, i]            = hs_fit_y_values
        fit_data['fit_x_error'][:, i]      = np.zeros(fit_x_values.size)
        fit_data['fit_y_error'][:, i]      = hs_fit_y_errors
        fit_data['fit_dydx'][:, i]         = fit_dydx_values
        fit_data['fit_dydx_x_error'][:, i] = np.zeros(fit_x_values.size)
        fit_data['fit_dydy_y_error'][:, i] = hs_fit_dydx_errors

    return fit_data
