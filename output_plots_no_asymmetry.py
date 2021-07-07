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

x = [0,1,2,3,4,5,6,7,8,9]

for filename in os.listdir(directory):
    if filename.endswith("_data"):
        os.chdir(os.path.join(directory, filename))

        print('----------> you are working in the following directory :',os.getcwd())
        print('----------> entering the output_directory : ')
        os.chdir('./output_directory')

        with np.load('Time_file.npz') as Time_file_loaded:
            Time_index=Time_file_loaded['Time_index']
            time_global = Time_file_loaded['time_global']

        with np.load('Error_files.npz') as Error_files_loaded:
            error_difference=Error_files_loaded['error_difference']
            error_difference_percent=Error_files_loaded['error_difference_percent']
            error_difference_1=Error_files_loaded['error_difference_1']
            error_difference_percent_1=Error_files_loaded['error_difference_percent_1']
            RMSE=Error_files_loaded['RMSE']
            RMS=Error_files_loaded['RMS']
            RMSE_1=Error_files_loaded['RMSE_1']
            electron_density_ne_error = Error_files_loaded['electron_density_ne_error']
            integrale_density_final_error = Error_files_loaded['integrale_density_final_error']
            #RMS_1=Error_files_loaded['RMS_1']



        with np.load('Rhos_vs_ne.npz') as Rhos_vs_ne_loaded:
            rho_pol_norm_base_min=Rhos_vs_ne_loaded['rho_pol_norm_base_min']
            #rho_pol_norm_ref=Rhos_vs_ne_loaded['rho_pol_norm_ref']
            #integrale_density_ref=Rhos_vs_ne_loaded['integrale_density_ref']
            electron_density_ne=Rhos_vs_ne_loaded['electron_density_ne']
            density_check=Rhos_vs_ne_loaded['density_check']
            integrale_density_final=Rhos_vs_ne_loaded['integrale_density_final']
            #electron_density_line = Rhos_vs_ne_loaded['electron_density_line']
            integral_density_final_corrected = Rhos_vs_ne_loaded['integral_density_final_corrected']


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
                fig.suptitle((('error diff btw measured and  reconstructed  data')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                #ax.plot(RMS, '-', color='r', label = 'rms')
                ax.plot(error_difference_percent, '-', color='b', label = '(mesured density - reconstructed density )/ mesured density')
                ax.set_ylabel('density_error')
                plt.legend()
                fig.savefig(plot_save_directory + 'density_error.png')
                plt.close(fig)
                print("Results of demonstration plotted in directory ./RMS_figuers/\n")

                fig = plt.figure()
                fig.suptitle((('error diff btw measured and  reconstructed  data')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                #ax.plot(RMS, '-', color='r', label = 'rms')
                ax.plot(error_difference_percent_1, '-', color='b', label = '(mesured density - reconstructed density )/ mesured density')
                ax.set_ylabel('density_error')
                plt.legend()
                fig.savefig(plot_save_directory + 'density_error_1.png')
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
                fig.suptitle((('inter verses integrated  data at T = ' +str(time_global[Time_index][ii]) + ' s ')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.plot(integrale_density_final[:,Time_index[ii]], '.', color='r', label = 'ne from integration and fit procedure')
                #ax.plot(electron_density_ne[:,Time_index[ii]] , '*', color='k', label = 'ne from inter')
                ax.plot(density_check[:,Time_index[ii]] , '-', color='g', label = 'density_check')
                #ax.plot(electron_density_line[:,Time_index[ii]] , '*', color='b', label = 'ne from inter')
                ax.set_ylabel('Density')
                plt.legend()
                fig.savefig(plot_save_directory + 'time_slice' + str(Time_index[ii]) +'.png')
                plt.close(fig)
            print("Results of demonstration plotted in directory ./comparison_figuers/\n")

            plot_save_directory = './comparison_figuers_new'
            if not plot_save_directory.endswith('/'):
                plot_save_directory = plot_save_directory+'/'
            if not os.path.isdir(plot_save_directory):
                os.makedirs(plot_save_directory)
            for ii in range(len(Time_index)):
                fig = plt.figure()
                fig.suptitle((('comparison of data at time = ' +str(time_global[Time_index][ii]) + ' s ')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                #ax.plot(integral_density_final_corrected[:,Time_index[ii]], '.', color='r', label = 'ne from integration and fit procedure')
                #ax.plot(density_check[:,Time_index[ii]] , '-', color='g', label = 'density_check')
                ax.errorbar(x, integral_density_final_corrected[:,Time_index[ii]],yerr=integrale_density_final_error[:,Time_index[ii]]  , label = 'integrated reconstructed density')
                ax.errorbar(x, density_check[:,Time_index[ii]] , yerr=electron_density_ne_error[:,Time_index[ii]], color='g', label = 'integrated measured density')
                ax.set_ylabel('integrated density (m^-2)')
                ax.set_xlabel('LOS')
                plt.grid(True)
                plt.legend()
                fig.savefig(plot_save_directory + 'time_slice' + str(Time_index[ii]) +'.png')
                plt.close(fig)
            print("Results of demonstration plotted in directory ./comparison_figuers_new/\n")
            
            plot_save_directory = './figuers_inter_refl_fits_rho'
            if not plot_save_directory.endswith('/'):
                plot_save_directory = plot_save_directory+'/'
            if not os.path.isdir(plot_save_directory):
                os.makedirs(plot_save_directory)
            '''
            for ii in range(len(Time_index)):
                fig = plt.figure()
                fig.suptitle((('figures before and after fits rho at T = ' +str(time_global[Time_index][ii]) + ' s ')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
                ax = fig.add_subplot(111)
                ax.plot(rho_pol_norm_base_min[:,Time_index[ii]],electron_density_ne[:,Time_index[ii]], '.' , color='r', label = 'ne interferometry')
                #ax.plot(rho_pol_norm_ref[:,Time_index[ii]],integrale_density_ref[:,Time_index[ii]] ,color='b', label = 'integrale density reflectometry')
                ax.set_xlabel('rho')
                ax.set_ylabel('Density')
                plt.legend()
                fig.savefig(plot_save_directory+ 'time_slice' + str(Time_index[ii]) +'.png')
                plt.close(fig)

            print("Results of demonstration plotted in directory ./figuers_inter_refl_fits_rho/\n")
            '''
        os.chdir('../')
        os.chdir('../../')
        print('current working directory :', os.getcwd())

    else:
        continue
    
print('Program finished successfuly') 


#x = [0,1,2,3,4,5,6,7,8,9]
#plt.errorbar(x, integral_density_final_corrected[:,2], yerr=integrale_density_final_error[:,2])
#plt.grid(True)
#t_igni = pw.tsbase(SHOT, 'RIGNITRON')
#import pywed as pw
