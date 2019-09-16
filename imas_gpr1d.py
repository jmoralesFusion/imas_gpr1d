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


def get_data(shot, run_out, occ_out, user_out, machine_out, run_in, occ_in, user_in, machine_in, datatype):
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
        #print(idd_in.reflectometer_profile.channel[0].position.r.data)
        print(R_real.shape)
        
        electron_density = idd_in.reflectometer_profile.channel[0].n_e.data
        
        #print(idd_in.reflectometer_profile.channel[0].n_e.data)
        #print(idd_in.reflectometer_profile.channel[0].n_e.data.shape) #(100,3177)
        
        import matplotlib.pyplot as plt
        
        #plt.plot(idd_in.reflectometer_profile.channel[0].position.r.data, idd_in.reflectometer_profile.channel[0].n_e.data)
        #plt.show()
        
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
        plt.plot(rho_pol_norm, idd_in.reflectometer_profile.channel[0].n_e.data)
        plt.show()
        plt.figure() 
        plt.plot(rho_pol_norm[:, 1000], idd_in.reflectometer_profile.channel[0].n_e.data[:, 1000])
        plt.show()
        
        
        print(' rho_pol_norm.shape : ' , rho_pol_norm.shape)
        print('density.shape : ', electron_density.shape)
        print('Time shape : ',Time.shape)
        print('R shape: ',R_base.shape)
        print('Phi shape : ' , Phi.shape)
        print('Z shape : ',Z.shape)


        return rho_pol_norm, electron_density
    
            
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
    #parser.add_argument('--fast', action='store_true', \
    #                    help='fast calculation')

    args = parser.parse_args()

    # Call wrapper function
    x, y = get_data(args.shot, \
             args.run_out, args.occurrence_out, args.user_out, args.machine_out, \
             args.run_in, args.occurrence_in, args.user_in, args.machine_in, \
             args.ids)

    fit_data(x, y, args.kernel)
