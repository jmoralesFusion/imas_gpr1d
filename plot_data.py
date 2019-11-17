# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab

from __future__ import (unicode_literals, absolute_import,print_function, division)
# Standard python modules
import os
import re
import sys
import warnings
import time
#import pdb; pdb.set_trace()

import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap


def plot_data(fit_x_values, minimum, maximum, \
                  X_reduced, Y_reduced, \
                  Y_errors, X_errors, \
                  hs_fit_y_values, hs_fit_y_errors, \
                  hs_fit_dydx_errors, hs_fit_dydx_values, \
                  ni_fit_y_values, ni_fit_dydx_values,  ni_fit_y_errors, ni_fit_dydx_errors, \
                  sample_array, sample_mean, sample_std, \
                  deriv_array, deriv_mean, deriv_std, \
                  dsample_array, dsample_mean, dsample_std, \
                  integ_array, integ_mean, integ_std, ifit_x_values, \
                  ndsample_array, dfit_x_values,  \
                  hs_zfit_y_values, hs_zfit_y_errors, \
                  zsample_array, zsample_mean, zsample_std, \
                  hs_zfit_dydx_values, hs_zfit_dydx_errors, \
                  zderiv_array, zderiv_mean, zderiv_std, \
                  zdsample_array, zdsample_mean, zdsample_std, \
                  zinteg_array, zinteg_mean, zinteg_std, \
                  nsample_array, \
                  fit_y_values, fit_y_errors, \
                  fit_dydx_values, fit_dydx_errors,\
                  i=0): 
 
    ###Some basic setup
    plot_save_directory = './GPPlots' + str(i)
    if not plot_save_directory.endswith('/'):
        plot_save_directory = plot_save_directory+'/'
    if not os.path.isdir(plot_save_directory):
            os.makedirs(plot_save_directory)
            
            

    ### Plotting

    plt = None
    try:
        import matplotlib.pyplot as plt
    except:
        plt = None

    if plt is not None:


        plot_num_samples = 3
        plot_sigma = 2.0
        
        plot_hs_sample_y_lower = sample_mean - plot_sigma * sample_std
        plot_hs_sample_y_upper = sample_mean + plot_sigma * sample_std
               
        
        ###Fitting
        #plotting the data as it is 
        fig = plt.figure()
        fig.suptitle('Raw Data', fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax = fig.add_subplot(111)
        plt.plot(X_reduced, Y_reduced)
        plt.gca().legend(('Raw data'),loc = 'best')
        fig.savefig(plot_save_directory+'data.png')
        plt.close(fig)


        # Raw data with GPR fit and error, only accounting for y-errors
        plot_Y_errors = plot_sigma * Y_errors
        fig = plt.figure()
        fig.suptitle('Raw data with GPR fit and error, only accounting for y-error', fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax = fig.add_subplot(111)
        ax.errorbar(X_reduced, Y_reduced, yerr=plot_Y_errors, ls='', marker='.', color='b')
        ax.plot(fit_x_values, hs_fit_y_values, color='r')
        plot_hs_fit_y_lower = hs_fit_y_values - plot_sigma * hs_fit_y_errors
        plot_hs_fit_y_upper = hs_fit_y_values + plot_sigma * hs_fit_y_errors
        ax.fill_between(fit_x_values, plot_hs_fit_y_lower, plot_hs_fit_y_upper, facecolor='r', edgecolor='None', alpha=0.2)
        ax.set_xlim(minimum, maximum)
        plt.gca().legend(('gpr fit','2$\\sigma$ plot region','Raw data'), loc = 'best')
        fig.savefig(plot_save_directory+'gp_data.png')
        plt.close(fig)
        
        # Derivative of GPR fit and error, only accounting for y-errors
        fig = plt.figure()
        fig.suptitle('Derivative of GPR fit and error, only accounting for y-errors', fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax = fig.add_subplot(111)
        #if (fit_dydx_values is not None):
        ax.plot(fit_x_values, hs_fit_dydx_values, color='r')
        plot_hs_fit_dydx_lower = hs_fit_dydx_values - plot_sigma * hs_fit_dydx_errors
        plot_hs_fit_dydx_upper = hs_fit_dydx_values + plot_sigma * hs_fit_dydx_errors
        ax.fill_between(fit_x_values, plot_hs_fit_dydx_lower, plot_hs_fit_dydx_upper, facecolor='r', edgecolor='None', alpha=0.2)
        ax.set_xlim(minimum, maximum)
        plt.gca().legend(('derv gpr fit','2$\\sigma$ plot region ','Raw data'))
        fig.savefig(plot_save_directory+'gp_derivative_data.png')
        plt.close(fig)
        
        # Raw data with GPR fit and error, comparison of using y-errors as weights, rigourously accounting for y-errors, and rigourously account for y-errors AND x-errors
        plot_X_errors = plot_sigma * X_errors
        fig = plt.figure()
        fig.suptitle("\n".join(wrap('Raw data (GPR fit and error), comparison of using y-errors as weights, rigourously accounting for y-errors, and rigourously account for y-errors AND x-errors')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
        ax = fig.add_subplot(111)
        ax.errorbar(X_reduced, Y_reduced, xerr=plot_X_errors, yerr=plot_Y_errors, ls='', marker='.', color='k')
        #if (fit_y_values is not None):
        ax.plot(fit_x_values, fit_y_values, color='g', label = 'gpr fit')
        plot_fit_y_lower = fit_y_values - plot_sigma * fit_y_errors
        plot_fit_y_upper = fit_y_values + plot_sigma * fit_y_errors
        ax.plot(fit_x_values, plot_fit_y_lower, color='g', ls='--')
        ax.plot(fit_x_values, plot_fit_y_upper, color='g', ls='--')
        ax.fill_between(fit_x_values, plot_fit_y_lower, plot_fit_y_upper, facecolor='g', edgecolor='None', alpha=0.2, label = '2$\\sigma$ y-errors as weights')
        ax.plot(fit_x_values, hs_fit_y_values, color='r', label = 'gpr fit account for ey')
        ax.plot(fit_x_values, plot_hs_fit_y_lower, color='r', ls='--')
        ax.plot(fit_x_values, plot_hs_fit_y_upper, color='r', ls='--')
        ax.fill_between(fit_x_values, plot_hs_fit_y_lower, plot_hs_fit_y_upper, facecolor='r', edgecolor='None', alpha=0.2, label = '2$\\sigma$ account for ey')
        ax.plot(fit_x_values, ni_fit_y_values, color='b', label = 'gpr fit account for ey & ex')
        plot_ni_fit_y_lower = ni_fit_y_values - plot_sigma * ni_fit_y_errors
        plot_ni_fit_y_upper = ni_fit_y_values + plot_sigma * ni_fit_y_errors
        ax.plot(fit_x_values, plot_ni_fit_y_lower, color='b', ls='--')
        ax.plot(fit_x_values, plot_ni_fit_y_upper, color='b', ls='--')
        ax.fill_between(fit_x_values, plot_ni_fit_y_lower, plot_ni_fit_y_upper, facecolor='b', edgecolor='None', alpha=0.2, label = '2$\\sigma$ account for ey & ex')
        ax.set_xlim(minimum, maximum)
        plt.legend()
        fig.savefig(plot_save_directory+'gp_options_test.png')
        plt.close(fig)

        # Derivative of GPR fit and error, comparison of using y-errors as weights, rigourously accounting for y-errors, and rigourously account for y-errors AND x-errors
        fig = plt.figure()
        fig.suptitle("\n".join(wrap('Derivative of GPR fit and error, comparison of using y-errors as weights, rigourously accounting for y-errors, and rigourously account for y-errors AND x-errors')), fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax = fig.add_subplot(111)
        ax.plot(fit_x_values, fit_dydx_values, color='g', label = 'der gpr fit')
        plot_fit_dydx_lower = fit_dydx_values - plot_sigma * fit_dydx_errors
        plot_fit_dydx_upper = fit_dydx_values + plot_sigma * fit_dydx_errors
        ax.plot(fit_x_values, plot_fit_dydx_lower, color='g', ls='--')
        ax.plot(fit_x_values, plot_fit_dydx_upper, color='g', ls='--')
        ax.fill_between(fit_x_values, plot_fit_dydx_lower, plot_fit_dydx_upper, facecolor='g', edgecolor='None', alpha=0.2, label = '2$\\sigma$ y-errors as weights')
        ax.plot(fit_x_values, hs_fit_dydx_values, color='r', label = 'der gpr fit account for ey')
        ax.plot(fit_x_values, plot_hs_fit_dydx_lower, color='r', ls='--')
        ax.plot(fit_x_values, plot_hs_fit_dydx_upper, color='r', ls='--')
        ax.fill_between(fit_x_values, plot_hs_fit_dydx_lower, plot_hs_fit_dydx_upper, facecolor='r', edgecolor='None', alpha=0.2, label = '2$\\sigma$ account for ey')
        ax.plot(fit_x_values, ni_fit_dydx_values, color='b', label = 'der gpr fit account for ey & ex')
        plot_ni_fit_dydx_lower = ni_fit_dydx_values - plot_sigma * ni_fit_dydx_errors
        plot_ni_fit_dydx_upper = ni_fit_dydx_values + plot_sigma * ni_fit_dydx_errors
        ax.plot(fit_x_values, plot_ni_fit_dydx_lower, color='b', ls='--')
        ax.plot(fit_x_values, plot_ni_fit_dydx_upper, color='b', ls='--')
        ax.fill_between(fit_x_values, plot_ni_fit_dydx_lower, plot_ni_fit_dydx_upper, facecolor='b', edgecolor='None', alpha=0.2, label = '2$\\sigma$ account for ey & ex')
        ax.set_xlim(minimum, maximum)
        plt.legend()
        fig.savefig(plot_save_directory+'gp_options_dtest.png')
        plt.close(fig)
        
       

        # Sampled fit curves (true noise) against GPR fit distribution
        fig = plt.figure()
        fig.suptitle("\n".join(wrap('Sampled fit curves (true noise) against GPR fit distribution')), fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax = fig.add_subplot(111)
        ax.fill_between(fit_x_values, plot_hs_fit_y_lower, plot_hs_fit_y_upper, facecolor='r', edgecolor='None', alpha=0.2)
        for ii in np.arange(0, plot_num_samples):
            ax.plot(fit_x_values, nsample_array[ii, :], color='k', alpha=0.5)
        ax.set_xlim(minimum, maximum)
        fig.savefig(plot_save_directory+'sample_gp_noisy_test.png')
        plt.close(fig)

        # Sampled fit derivative curves (true noise) against GPR fit derivative distribution
        fig = plt.figure()
        fig.suptitle("\n".join(wrap('Sampled fit derivative curves (true noise) against GPR fit derivative distribution')), fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax = fig.add_subplot(111)
        ax.fill_between(fit_x_values, plot_hs_fit_dydx_lower, plot_hs_fit_dydx_upper, facecolor='r', edgecolor='None', alpha=0.2)
        for ii in np.arange(0, plot_num_samples):
            ax.plot(fit_x_values, ndsample_array[ii, :], color='k', alpha=0.5)
        ax.set_xlim(minimum, maximum)
        fig.savefig(plot_save_directory+'sample_gp_noisy_dtest.png')
        plt.close(fig)
        
        # Sampled fit curves (zero noise) against GPR fit distribution
        fig = plt.figure()
        fig.suptitle('Sampled fit curves (zero noise) against GPR fit distribution', fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax = fig.add_subplot(111)
        plot_hs_zfit_y_lower = hs_zfit_y_values - plot_sigma * hs_zfit_y_errors
        plot_hs_zfit_y_upper = hs_zfit_y_values + plot_sigma * hs_zfit_y_errors
        ax.fill_between(fit_x_values, plot_hs_zfit_y_lower, plot_hs_zfit_y_upper, facecolor='r', edgecolor='None', alpha=0.2)
        for ii in np.arange(0, plot_num_samples):
            ax.plot(fit_x_values, zsample_array[ii, :], color='k', alpha=0.5)
        plot_hs_zsample_y_lower = zsample_mean - plot_sigma * zsample_std
        plot_hs_zsample_y_upper = zsample_mean + plot_sigma * zsample_std
        ax.fill_between(fit_x_values, plot_hs_sample_y_lower, plot_hs_sample_y_upper, facecolor='b', edgecolor='None', alpha=0.2)
        ax.set_xlim(minimum, maximum)
        fig.savefig(plot_save_directory+'sample_gp_no_noise_test.png')
        plt.close(fig)

        # Derivatives of sampled fit curves (zero noise) against GPR fit derivative distribution
        fig = plt.figure()
        fig.suptitle("\n".join(wrap('Derivatives of sampled fit curves (zero noise) against GPR fit derivative distribution')), fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax = fig.add_subplot(111)
        plot_hs_zfit_dydx_lower = hs_zfit_dydx_values - plot_sigma * hs_zfit_dydx_errors
        plot_hs_zfit_dydx_upper = hs_zfit_dydx_values + plot_sigma * hs_zfit_dydx_errors
        ax.fill_between(fit_x_values, plot_hs_zfit_dydx_lower, plot_hs_zfit_dydx_upper, facecolor='r', edgecolor='None', alpha=0.2)
        for ii in np.arange(0, plot_num_samples):
            ax.plot(dfit_x_values, zderiv_array[ii, :], color='k', alpha=0.5)
        plot_hs_zsample_dydx_lower = zderiv_mean - plot_sigma * zderiv_std
        plot_hs_zsample_dydx_upper = zderiv_mean + plot_sigma * zderiv_std
        ax.fill_between(dfit_x_values, plot_hs_zsample_dydx_lower, plot_hs_zsample_dydx_upper, facecolor='b', edgecolor='None', alpha=0.2)
        ax.set_xlim(minimum, maximum)
        fig.savefig(plot_save_directory+'sample_gp_drv_no_noise_test.png')
        plt.close(fig)

        # Sampled fit derivative curves (zero noise) against GPR fit derivative distribution
        fig = plt.figure()
        fig.suptitle("\n".join(wrap('Sampled fit derivative curves (zero noise) against GPR fit derivative distribution')), fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax = fig.add_subplot(111)
        ax.fill_between(fit_x_values, plot_hs_zfit_dydx_lower, plot_hs_zfit_dydx_upper, facecolor='r', edgecolor='None', alpha=0.2)
        for ii in np.arange(0, plot_num_samples):
            ax.plot(fit_x_values, zdsample_array[ii, :], color='k', alpha=0.5)
        plot_hs_zdsample_dydx_lower = zdsample_mean - plot_sigma * zdsample_std
        plot_hs_zdsample_dydx_upper = zdsample_mean + plot_sigma * zdsample_std
        ax.fill_between(fit_x_values, plot_hs_zdsample_dydx_lower, plot_hs_zdsample_dydx_upper, facecolor='b', edgecolor='None', alpha=0.2)
        ax.set_xlim(minimum, maximum)
        fig.savefig(plot_save_directory+'sample_gp_no_noise_dtest.png')
        plt.close(fig)
        
        # Integrals of sampled fit derivative curves (zero noise) against GPR fit distribution
        fig = plt.figure()
        fig.suptitle("\n".join(wrap('Integrals of sampled fit derivative curves (zero noise) against GPR fit distribution')), fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax = fig.add_subplot(111)
        ax.fill_between(fit_x_values, plot_hs_zfit_y_lower, plot_hs_zfit_y_upper, facecolor='r', edgecolor='None', alpha=0.2)
        for ii in np.arange(0, plot_num_samples):
            ax.plot(ifit_x_values, zinteg_array[ii, :], color='k', alpha=0.5)
        plot_hs_zdsample_y_lower = zinteg_mean - plot_sigma * zinteg_std
        plot_hs_zdsample_y_upper = zinteg_mean + plot_sigma * zinteg_std
        ax.fill_between(ifit_x_values, plot_hs_zdsample_y_lower, plot_hs_zdsample_y_upper, facecolor='b', edgecolor='None', alpha=0.2)
        ax.set_xlim(minimum, maximum)
        fig.savefig(plot_save_directory+'sample_gp_itg_no_noise_dtest.png')
        plt.close(fig)
        
        print("Results of demonstration plotted in directory ./GPPlots/\n")

    else:

        print("   Module matplotlib not found. Skipping plotting of demonstration results.\n")

    print("Demonstration script successfully completed!\n")
