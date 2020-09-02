# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab
'''
    IMAS GPR1D
'''
# Standard python modules
from __future__ import (unicode_literals, absolute_import,  \
                        print_function, division)
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
#start plots for checkup the results:

directory = os.getcwd()

for filename in os.listdir(directory):
    if filename.endswith("_data"):
        os.chdir(os.path.join(directory, filename))
        print('you are working in the following directory :')
        print(os.getcwd())
        os.chdir('./output_directory')

        with np.load('Time_file.npz') as Time_file_loaded:
            Time_index=Time_file_loaded['Time_index']
            time_slices_red_upper=Time_file_loaded['time_slices_red_upper']
            time_slices_red_lower=Time_file_loaded['time_slices_red_lower']



        with np.load('Error_files.npz') as Error_files_loaded:
            error_difference=Error_files_loaded['error_difference']
            error_difference_percent=Error_files_loaded['error_difference_percent']
            RMSE=Error_files_loaded['RMSE']
            RMS=Error_files_loaded['RMS']

        with np.load('Compare_density.npz') as Compare_density_loaded:
            R_meters_mask_lower=Compare_density_loaded['R_meters_mask_lower']
            R_meters_mask_upper=Compare_density_loaded['R_meters_mask_upper']
            average_der_density=Compare_density_loaded['average_der_density']
            derivative_density_upper=Compare_density_loaded['derivative_density_upper']
            derivative_density_lower=Compare_density_loaded['derivative_density_lower']

        with np.load('Rhos_vs_ne.npz') as Rhos_vs_ne_loaded:
            rho_pol_norm_base_min=Rhos_vs_ne_loaded['rho_pol_norm_base_min']
            rho_pol_norm_ref=Rhos_vs_ne_loaded['rho_pol_norm_ref']
            electron_density_ne=Rhos_vs_ne_loaded['electron_density_ne']
            integrale_density_ref=Rhos_vs_ne_loaded['integrale_density_ref']
            density_check=Rhos_vs_ne_loaded['density_check']
            integrale_density_final=Rhos_vs_ne_loaded['integrale_density_final']


        plot_test_figures = True
        if plot_test_figures:
            # to check 
            RMS_error = True
            if RMS_error:
                plot_save_directory = './RMS_figuers'
                if not plot_save_directory.endswith('/'):
                    plot_save_directory = plot_save_directory+'/'
                if not os.path.isdir(plot_save_directory):
                    os.makedirs(plot_save_directory)

                fig = plt.figure()
                fig.suptitle((('error diff btw inter and  integrated  data')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                #ax.plot(RMS, '-', color='r', label = 'rms')
                ax.plot(RMSE, '-', color='b', label = '% rmse')
                ax.set_ylabel('RMS')
                plt.legend()
                fig.savefig(plot_save_directory + 'RMS.png')
                plt.close(fig)
                print("Results of demonstration plotted in directory ./RMS_figuers/\n")

            ###Some basic setup
            #create the test directory to save all the testing files
            test_save_directory = './test_directory'
            if not test_save_directory.endswith('/'):
                test_save_directory = test_save_directory+'/'
            if not os.path.isdir(test_save_directory):
                os.makedirs(test_save_directory)
            os.chdir(test_save_directory)

            #create the testing files directories
            plot_save_directory = './comparison_figuers'
            if not plot_save_directory.endswith('/'):
                plot_save_directory = plot_save_directory+'/'
            if not os.path.isdir(plot_save_directory):
                os.makedirs(plot_save_directory)

            for ii in range(len(Time_index)):
                fig = plt.figure()
                fig.suptitle((('inter verses integrated  data')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.plot(integrale_density_final[:,Time_index[ii]], '-', color='r', label = 'ne from integration and fit procedure')
                ax.plot(density_check[:,Time_index[ii]] , '-', color='k', label = 'ne from inter')
                ax.set_ylabel('Density')
                plt.legend()
                fig.savefig(plot_save_directory + 'time_slice' + str(Time_index[ii]) +'.png')
                plt.close(fig)
            print("Results of demonstration plotted in directory ./comparison_figuers/\n")

            plot_upper_lower = False
            if plot_upper_lower:
                plot_save_directory = './upper_figuers'
                if not plot_save_directory.endswith('/'):
                    plot_save_directory = plot_save_directory+'/'
                if not os.path.isdir(plot_save_directory):
                    os.makedirs(plot_save_directory)

                for ii in range(len(time_slices_red_upper)):
                    fig = plt.figure()
                    fig.suptitle((('Raw data_upper')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
                    ax = fig.add_subplot(111)
                    ax.plot(R_meters_mask_upper[ii],-(np.asarray(out_put_R_upper['fit_dydx']))[ii] ,'.', color='b', label = 'deriv ne vers r fit_data_upper before masking ')
                    ax.plot(R_meters_mask_upper[ii],derivative_density_upper[ii] ,color='r', label = 'deriv ne vers r fit_data_upper after masking ')
                    ax.plot(R_meters_mask_upper[ii],average_der_density[ii] ,color='k', label = 'average fit_data')
                    ax.set_xlabel('radius')
                    ax.set_ylabel('Density')
                    plt.legend()
                    fig.savefig(plot_save_directory + 'time_slice' + str(time_slices_red_lower[ii]) +'.png')
                    plt.close(fig)
                print("Results of demonstration plotted in directory ./upper_figuers/\n")




                plot_save_directory = './lower_figuers'
                if not plot_save_directory.endswith('/'):
                    plot_save_directory = plot_save_directory+'/'
                if not os.path.isdir(plot_save_directory):
                    os.makedirs(plot_save_directory)

                for ii in range(len(time_slices_red_lower)):
                    fig = plt.figure()
                    fig.suptitle((('Raw data_lower')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
                    ax = fig.add_subplot(111)
                    ax.plot(R_meters_mask_lower[ii],-(np.asarray(out_put_R_lower['fit_dydx']))[ii] , '.',color='b', label = 'deriv ne vers r fit_data_lower before masking ')
                    ax.plot(R_meters_mask_lower[ii],derivative_density_lower[ii] ,color='r', label = 'deriv ne vers r fit_data_lower after masking')
                    ax.plot(R_meters_mask_lower[ii],average_der_density[ii] ,color='k', label = 'average fit_data')
                    ax.set_xlabel('radius')
                    ax.set_ylabel('Density')
                    plt.legend()
                    fig.savefig(plot_save_directory+ 'time_slice' + str(time_slices_red_lower[ii]) +'.png')
                    plt.close(fig)

                print("Results of demonstration plotted in directory ./lower_figuers/\n")



            plot_save_directory = './comparison_upper_average_lower_figuers'
            if not plot_save_directory.endswith('/'):
                plot_save_directory = plot_save_directory+'/'
            if not os.path.isdir(plot_save_directory):
                os.makedirs(plot_save_directory)

            for ii in range(len(time_slices_red_lower)):
                fig = plt.figure()
                fig.suptitle((('Raw data lower, upper and average')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.plot(R_meters_mask_lower[ii],derivative_density_lower[ii] , '.',color='b', label = 'fit_data_lower')
                ax.plot(R_meters_mask_upper[ii],derivative_density_upper[ii] ,color='r', label = 'fit_data_upper')
                ax.plot(R_meters_mask_lower[ii],average_der_density[ii] ,color='k', label = 'average fit_data')
                ax.set_xlabel('radius')
                ax.set_ylabel('Density')
                plt.legend()
                fig.savefig(plot_save_directory+ 'time_slice' + str(time_slices_red_lower[ii]) +'.png')
                plt.close(fig)

            print("Results of demonstration plotted in directory ./comparison_upper_average_lower_figuers/\n")


            plot_save_directory = './figuers_inter_refl_fits_rho'
            if not plot_save_directory.endswith('/'):
                plot_save_directory = plot_save_directory+'/'
            if not os.path.isdir(plot_save_directory):
                os.makedirs(plot_save_directory)

            for ii in range(len(time_slices_red_lower)):
                fig = plt.figure()
                fig.suptitle((('figures before and after fits rho')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.plot(rho_pol_norm_base_min[:,time_slices_red_lower[ii]],electron_density_ne[:,time_slices_red_lower[ii]], '.' , color='r', label = 'ne interferometry')
                ax.plot(rho_pol_norm_ref[:,time_slices_red_lower[ii]],integrale_density_ref[:,time_slices_red_lower[ii]] ,color='b', label = 'integrale density reflectometry')
                ax.set_xlabel('rho')
                ax.set_ylabel('Density')
                plt.legend()
                fig.savefig(plot_save_directory+ 'time_slice' + str(time_slices_red_lower[ii]) +'.png')
                plt.close(fig)

            print("Results of demonstration plotted in directory ./figuers_inter_refl_fits_rho/\n")

        os.chdir('../')
        os.chdir('../../')
        print('current working directory')
        print(os.getcwd())

    else:
        continue
    
print('Program finished successfuly') 


