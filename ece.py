import imas
import os
import sys
import re
#import pdb; pdb.set_trace()
from scipy.linalg import cholesky
import getpass

import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import GPR1D



def get_data(shot, run_in, occ_in, user_in, machine_in, datatype):

    idd_in = imas.ids(shot, run_in)
    idd_in.open_env(user_in, machine_in, '3')
        
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
        
        
        plt.plot(rho_pol_norm[1000], electron_temperature_2[1000])
        #plt.plot(rho_pol_norm[500], electron_temperature_2[500])
        #plt.plot(rho_pol_norm[200], electron_temperature_2[200])
        plt.show()
        
        rho_pol_norm_file = TemporaryFile()
        np.save(rho_pol_norm_file,rho_pol_norm)
        electron_temperature_file = TemporaryFile()
        np.save(electron_temperature_file, electron_temperature_2)
        
        X_coordinates = rho_pol_norm
        Y_coordinates = electron_temperature_2
        Y_errors = np.full(Y_coordinates.shape, 100)
        
        '''
        file = open("list.txt", "w")
        file.write(str(X_coordinates[1000]) + " " + str(Y_coordinates[1000]) + " " + str(Y_errors[1000])+ "\n")
        file.close()
        '''
        return rho_pol_norm, electron_temperature_2
    



        
def main():
    rho_pol_norm, electron_temperature_2=get_data(54095, 0, 0, 'imas_public', 'west', 'ece')
main()
   
