import imas
import os
import sys
import re
#import pdb; pdb.set_trace()
import time

import getpass

import numpy as np
import matplotlib.pyplot as plt

import GPR1D

# Local libraries
from fit_data import fit_data
from Optimization_function import Optimization
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
        
        R_real= idd_in.reflectometer_profile.channel[0].position.r.data
        electron_density = idd_in.reflectometer_profile.channel[0].n_e.data
        
        import matplotlib.pyplot as plt
        import equimap
        
        Time = idd_in.reflectometer_profile.time
        R_base = np.linspace(R_real.min(), R_real.max(), 1000)
        Phi = np.zeros(1000)
        Z = np.zeros(1000)
        
        
        
        rho_pol_norm_base = equimap.get(shot, Time, R_base, Phi, Z, 'rho_pol_norm')
        if rho_pol_norm_base.shape != electron_density.shape :
            rho_pol_norm_base = rho_pol_norm_base.T
        else :
            rho_pol_norm_base = rho_pol_norm_base
            
            
        rho_pol_norm = np.full(R_real.shape, np.nan)
        rho_pol_norm.shape
        for ii in range(rho_pol_norm_base.shape[1]):
            rho_pol_norm[:, ii] = np.interp(R_real[:, ii], R_base, rho_pol_norm_base[:, ii])
        print(rho_pol_norm.shape, electron_density.shape)
        return rho_pol_norm, electron_density
    


    
    if datatype == 'ece':
        
        idd_in.ece.get()
        
        idd_in.ece.channel[0].position.r.data
        print(idd_in.ece.channel[0].position.r.data)
        print(idd_in.ece.channel[0].position.r.data.shape)
        
        idd_in.ece.channel[0].t_e.data
        
        print(idd_in.ece.channel[0].t_e.data)
        print(idd_in.ece.channel[0].t_e.data.shape) #(100,3177)
        
        import matplotlib.pyplot as plt
        '''
        #from matplotlib import style
        #style.use('ggplot')
        for ii in range (len(idd_in.ece.channel)):
            for jj in range (len(idd_in.ece.channel[ii].position.r.data)):
                for kk in range(len(idd_in.ece.channel[ii].t_e.validity_timed)):
                    if (idd_in.ece.channel[ii].t_e.validity_timed[kk] > 0):
                        plt.plot(idd_in.ece.channel[ii].position.r.data[jj], idd_in.ece.channel[ii].t_e.data[jj],label='temperature versus raduis')#, linewidth=4,color='red')
                        plt.legend()
                        plt.show()
        '''
        print(type(idd_in.ece.channel[0].position.r.data[idd_in.ece.channel[0].t_e.data>0]))
        print(idd_in.ece.channel[0].t_e.data[idd_in.ece.channel[0].t_e.data>0])
        print(idd_in.ece.channel[0].t_e.validity_timed[idd_in.ece.channel[0].t_e.data>0])
        import collections
        print(collections.Counter(idd_in.ece.channel[0].t_e.validity_timed))
        
        for kk in range(len(idd_in.ece.channel)):
            plt.plot(idd_in.ece.channel[kk].position.r.data[idd_in.ece.channel[0].t_e.validity_timed==0], idd_in.ece.channel[kk].t_e.data[idd_in.ece.channel[0].t_e.validity_timed==0],label='temperature versus raduis')#, linewidth=4,color='red')
            plt.legend()
            plt.show()
        else : print('nothing positive')
        #for ii in range(len(idd_in.ece.channel[0].position.r.data)):
        #    plt.plot(idd_in.ece.channel[0].position.r.data[ii], idd_in.ece.channel[0].t_e.data[ii],label='temperature versus raduis', linewidth=4,color='red')
        #    plt.legend()
        #    plt.show()
            
        
        density = idd_in.ece.channel[0].t_e.data
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
    


def main():
  
    rho_pol_norm, electron_density = get_data(54095, 0, 0, 'MK260524', 'west', 0, 0, 'imas_public', 'west', 'reflectometer_profile')
    
    fit_data( rho_pol_norm, electron_density, 'RQ_Kernel')
main()
