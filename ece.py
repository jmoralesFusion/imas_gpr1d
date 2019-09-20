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
        nbr_time =  len(idd_in.ece.channel[0].t_e.data)
        
        matrix_position = np.full((nbr_pts, nbr_channels), np.nan)
        matrix_time = np.full((nbr_time, nbr_channels), np.nan)
        
        print(idd_in.ece.channel[0].t_e.data[1], idd_in.ece.channel[0].position.r.data[1])
        
        for channel in range(len(idd_in.ece.channel)):
            for raduis in range(len(idd_in.ece.channel[channel].position.r.data)):
                matrix_position[raduis][channel] = idd_in.ece.channel[channel].position.r.data[raduis]
        print(matrix_position.shape)
        print(matrix_position)
       
        for channel in range(len(idd_in.ece.channel)):
            for time in range(len(idd_in.ece.channel[channel].t_e.data)):
                matrix_time[time][channel] = idd_in.ece.channel[channel].t_e.data[time]
        print(matrix_time.shape)
        print(matrix_time)
                    
def main():
    get_data(54095, 0, 0, 'imas_public', 'west', 'ece')

main()
   
