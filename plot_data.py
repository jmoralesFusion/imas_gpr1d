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

import getpass

import numpy as np
import matplotlib.pyplot as plt

import GPR1D
import imas

def plot_data(i,fit_x_values, minimum, maximum, \
                  X_reduced, Y_reduced, \
                  Y_errors, X_errors, \
                  fit_y_values, fit_y_errors,  \
                  hs_fit_y_values, hs_fit_y_errors,\
                  fit_dydx_values, fit_dydx_errors, hs_fit_dydx_errors,hs_fit_dydx_values, \
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
                  zinteg_array, zinteg_mean, zinteg_std):

 
    ###Some basic setup
    plot_save_directory = './bebe' + str(i)
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
        ###Fitting
        fig = plt.figure()
        fig.suptitle('My Title', fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax = fig.add_subplot(111)
        plt.plot(X_reduced, Y_reduced)
        fig.savefig(plot_save_directory+'data.png')
        plt.close(fig)


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
        print(dfit_x_values.shape,deriv_array.shape)
        #for ii in np.arange(0,plot_num_samples):
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
