import imas
import os
import sys
import re
#import pdb; pdb.set_trace()
from scipy.linalg import cholesky
import getpass

import numpy as np
import matplotlib.pyplot as plt

import GPR1D



def get_data(shot, run_in, occ_in, user_in, machine_in, datatype):

    idd_in = imas.ids(shot, run_in)
    idd_in.open_env(user_in, machine_in, '3')
        
    if datatype == 'ece':

        idd_in.ece.get()
        idd_in.ece.channel[0].position.r.data
        print(idd_in.ece.channel[0].position.r.data)
        print(idd_in.ece.channel[0].position.r.data.shape)
            
        idd_in.ece.channel[0].t_e.data
        print(idd_in.ece.channel[0].t_e.data)
        print(idd_in.ece.channel[0].t_e.data.shape) #(100,3177)

        nbr_channels = len(idd_in.ece.channel)
        nbr_pts =len(idd_in.ece.channel[0].position.r.data)
        nbr_temperature =  len(idd_in.ece.channel[0].t_e.data)
        
        matrix_position = np.full((nbr_pts, nbr_channels), np.nan)
        matrix_temperature = np.full((nbr_temperature, nbr_channels), np.nan)
        
        print(idd_in.ece.channel[0].t_e.data[1], idd_in.ece.channel[0].position.r.data[1])
        
        for channel in range(len(idd_in.ece.channel)):
            for raduis in range(len(idd_in.ece.channel[channel].position.r.data)):
                matrix_position[raduis][channel] = idd_in.ece.channel[channel].position.r.data[raduis]
        print(matrix_position.shape)
        print(matrix_position)
       
        for channel in range(len(idd_in.ece.channel)):
            for temperature in range(len(idd_in.ece.channel[channel].t_e.data)):
                matrix_temperature[temperature][channel] = idd_in.ece.channel[channel].t_e.data[temperature]
        print(matrix_temperature.shape)
        print(matrix_temperature)




        
        R_real = idd_in.ece.channel[0].position.r.data
        
        electron_temperature = idd_in.ece.channel[0].t_e.data
        
        import matplotlib.pyplot as plt
        import equimap
        
        Time = idd_in.ece.time
        R_base = np.linspace(R_real.min(), R_real.max(), 1000)
        Phi = np.zeros(1000)
        Z = np.zeros(1000)
        
        
        
        rho_pol_norm_base = equimap.get(shot, Time, R_base , Phi, Z, 'rho_pol_norm')
        if rho_pol_norm_base.shape != electron_temperature.shape :
            rho_pol_norm_base = rho_pol_norm_base.T
        else :
            rho_pol_norm_base = rho_pol_norm_base
            
            
        rho_pol_norm = np.full(R_real.shape, np.nan)
        rho_pol_norm.shape
        for ii in range(rho_pol_norm_base.shape[1]):
            rho_pol_norm[:, ii] = np.interp(R_real[:, ii], R_base, rho_pol_norm_base[:, ii])

        return rho_pol_norm, electron_temperature
    



        
def main():
    get_data(54095, 0, 0, 'imas_public', 'west', 'ece')

main()
   
