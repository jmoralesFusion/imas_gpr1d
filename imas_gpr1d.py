# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab
'''
    IMAS GPR1D
'''
# Standard python modules
from __future__ import (unicode_literals, absolute_import,  \
                        print_function, division)
import argparse
import getpass
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re

# Local libraries
import imas

# Project libraries
from fit_data import fit_data


def get_data(shot, run_out, occ_out, user_out, machine_out, run_in, occ_in, user_in, machine_in, \
             datatype, write_edge_profiles):
    Data_typelist = ['reflectometer_profile', 'ece', 'interferometer']

    print('Printing the list of data type that should be used: ')
    print(Data_typelist)
    if datatype in Data_typelist:
        print('The chosen type of data  is : ',datatype)
    else:
        raise ValueError("The data type is not know, please provide an valid type from the List")
        return

    
    idd_in = imas.ids(shot, run_in)
    idd_in.open_env(user_in, machine_in, '3')
    
    if datatype == 'reflectometer_profile':
        
        idd_in.reflectometer_profile.get()
        
        R_real = idd_in.reflectometer_profile.channel[0].position.r.data
        print(R_real.shape)
        
        electron_density = idd_in.reflectometer_profile.channel[0].n_e.data
        
        
        import matplotlib.pyplot as plt
        
        
        import equimap
        
        Time = idd_in.reflectometer_profile.time
        R_base = np.linspace(R_real.min(), R_real.max(), 1000)
        Phi = np.zeros(1000)
        Z = np.zeros(1000)
        
        
        
        rho_pol_norm_base = equimap.get(shot, idd_in.reflectometer_profile.time,R_base ,Phi, Z, 'rho_pol_norm')
        if rho_pol_norm_base.shape != electron_density.shape :
            rho_pol_norm_base = rho_pol_norm_base.T
        else :
            rho_pol_norm_base = rho_pol_norm_base
            
            
        rho_pol_norm = np.full(R_real.shape, np.nan)
        rho_pol_norm.shape
        for ii in range(rho_pol_norm_base.shape[1]):
            rho_pol_norm[:, ii] = np.interp(R_real[:, ii], R_base, rho_pol_norm_base[:, ii])
            
        print(' rho_pol_norm.shape : ' , rho_pol_norm.shape)
        print('density.shape : ', electron_density.shape)
        print('Time shape : ',Time.shape)
        print('R shape: ',R_base.shape)
        print('Phi shape : ' , Phi.shape)
        print('Z shape : ',Z.shape)

        if (write_edge_profiles):
            #####################################################################################################
            ### save the output to the edge profiles as a start
            #def put_data(shot, run_out, occ_out, user_out, machine_out):
            idd_out = imas.ids(shot,  run_out)
            idd_out.create_env(user_out, machine_out, '3')
            #print(len(idd_out.core_profiles.profiles_1d))
            idd_out.edge_profiles.profiles_1d.resize(100)
            print('rho_pol_norm =', rho_pol_norm)
            idd_out.edge_profiles.profiles_1d[0].grid.rho_tor_norm = rho_pol_norm[0, :]
            idd_out.edge_profiles.ids_properties.homogeneous_time = 0
            idd_out.edge_profiles.put()
            
            idd_out.close()

            #####################################################################################################

        return rho_pol_norm, electron_density

    if datatype == 'ece':

        idd_in.ece.get()
        idd_in.equilibrium.get()

        mask_eq = np.asarray(idd_in.equilibrium.code.output_flag) > -1

        nbr_channels = len(idd_in.ece.channel)
        nbr_pts =len(idd_in.ece.channel[0].position.r.data)
        nbr_temperature =  len(idd_in.ece.channel[0].t_e.data)
        
        matrix_position = np.full((nbr_pts, nbr_channels), np.nan)
        matrix_temperature = np.full((nbr_temperature, nbr_channels), np.nan)
        
        
        for channel in range(len(idd_in.ece.channel)):
            for raduis in range(len(idd_in.ece.channel[channel].position.r.data)):
                matrix_position[raduis][channel] = idd_in.ece.channel[channel].position.r.data[raduis]
            for temperature in range(len(idd_in.ece.channel[channel].t_e.data)):
                matrix_temperature[temperature][channel] = idd_in.ece.channel[channel].t_e.data[temperature]


        mask_eq_time = (idd_in.ece.time > idd_in.equilibrium.time[mask_eq][0]) \
                     & (idd_in.ece.time < idd_in.equilibrium.time[mask_eq][-1]) \

        Time                                            = idd_in.ece.time[mask_eq_time]
        R_real                                          = matrix_position[mask_eq_time]
        electron_temperature                            = matrix_temperature[mask_eq_time]
        electron_temperature[electron_temperature < 0]  = np.nan
        
        R_real[np.isnan(electron_temperature)] = np.nan
        R_base = np.linspace(np.nanmin(R_real), np.nanmax(R_real), 1000)
        
        Phi = np.zeros(1000)
        Z   = np.zeros(1000)
        
        import matplotlib.pyplot as plt
        import equimap
        rho_pol_norm_base = equimap.get(shot, Time, R_base , Phi, Z, 'rho_pol_norm')
        
        rho_pol_norm           = [None]*R_real.shape[0]
        electron_temperature_2 = [None]*R_real.shape[0]

        #import ipdb; ipdb.set_trace()
        for ii in range(rho_pol_norm_base.shape[0]):
            rho_pol_norm[ii] = np.interp(R_real[ii, :][~np.isnan(R_real[ii, :])], \
                                         R_base, rho_pol_norm_base[ii, :])
            electron_temperature_2[ii] = electron_temperature[ii, :][~np.isnan(electron_temperature[ii, :])]

        rho_pol_norm           = np.asarray(rho_pol_norm)
        electron_temperature_2 = np.asarray(electron_temperature_2)
        
        
        #plt.plot(rho_pol_norm[1000], electron_temperature_2[1000])
        #plt.plot(rho_pol_norm[500], electron_temperature_2[500])
        #plt.plot(rho_pol_norm[200], electron_temperature_2[200])
        #plt.show()
        
        return rho_pol_norm, electron_temperature_2





 


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description= \
             '''IMAS GPR1D
             ''')
    parser.add_argument('shot', type=int,  \
                        help='shot')
    parser.add_argument('run_out', type=int, nargs='?', default=0, \
                        help='run_out, default=0')
    parser.add_argument('occurrence_out', type=int, nargs='?', default=0, \
                        help='occurrence_out, default=0')
    parser.add_argument('user_out', type=str, nargs='?', default=getpass.getuser(), \
                        help='user_out, default=getpass.getuser()')
    parser.add_argument('machine_out', type=str, nargs='?', default='west', \
                        help='machine_out, default=west')
    parser.add_argument('run_in', type=int, nargs='?', default=0, \
                        help='run_out, default=0')
    parser.add_argument('occurrence_in', type=int, nargs='?', default=0, \
                        help='occurrence_in, default=0')
    parser.add_argument('user_in', type=str, nargs='?', default='imas_public', \
                        help='user_in, default=imas_public')
    parser.add_argument('machine_in', type=str, nargs='?', default='west', \
                        help='machine_in, default=west')
    parser.add_argument('--ids', type=str, default='reflectometer_profile', \
                        help='IDS source of data for profile fit, default=reflectometer_profile')
    parser.add_argument('-k', '--kernel', type=str, default='RQ_Kernel', \
                        help='Kernel to use for profile fit, default=RQ_Kernel')
    parser.add_argument('-wep', '--write-edge-profiles', action='store_true', \
                        help='Write IDS edge_profiles')

    args = parser.parse_args()

    # Call wrapper function
    x, y = get_data(args.shot, \
             args.run_out, args.occurrence_out, args.user_out, args.machine_out, \
             args.run_in, args.occurrence_in, args.user_in, args.machine_in, \
             args.ids, args.write_edge_profiles)

    fit_data(x, y, args.kernel)
