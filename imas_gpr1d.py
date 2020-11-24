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
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
from scipy import integrate
from scipy import interpolate

# Local libraries
import equimap
import imas

# Project libraries
from fit_data import fit_data
#define global errors
absolute_error_ne = 6*1e17

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

    shot_directory = './'+str(shot)+'_'+datatype+'_data'
    if not shot_directory.endswith('/'):
        shot_directory = shot_directory+'/'
    if not os.path.isdir(shot_directory):
        os.makedirs(shot_directory)
    os.chdir(shot_directory)
    #-	55580 and 54560  abserrNe = 6.0e17  relerrNe = 0.1
    idd_in = imas.ids(shot, run_in)
    idd_in.open_env(user_in, machine_in, '3')

    if datatype == 'reflectometer_profile':

        idd_in.reflectometer_profile.get()
        R_real = idd_in.reflectometer_profile.channel[0].position.r.data
        electron_density = idd_in.reflectometer_profile.channel[0].n_e.data

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
        for ii in range(rho_pol_norm_base.shape[1]):
            rho_pol_norm[:, ii] = np.interp(R_real[:, ii], R_base, rho_pol_norm_base[:, ii])


        electron_density_errors = np.full(electron_density.shape, np.mean(electron_density)*0.05)
        rho_pol_norm_errors =  np.full(rho_pol_norm.shape, np.mean(rho_pol_norm)*0.005)
        if (write_edge_profiles):
            #####################################################################################################
            ### save the output to the edge profiles as a start

            # Create or open IDS
            # ------------------
            run_number = '{:04d}'.format(run_out)
            shot_file  = os.path.expanduser('~' + user_out + '/public/imasdb/' \
                                                + machine_out + '/3/0/' + 'ids_' + str(shot) \
                                                + run_number + '.datafile')

            idd_out = imas.ids(shot, run_out)

            if (os.path.isfile(shot_file)):
                print('open the IDS')
                idd_out.open_env(user_out, machine_out, '3')
            else:
                if (user_out == 'imas_public'):
                    print('ERROR IDS file does not exist, the IDS file must be')
                    print('created first for imas_public user_out')
                    raise FileNotFoundError
                else:
                    print('Create the IDS')
                    idd_out.create_env(user_out, machine_out, '3')

            # Write data
            # ----------
            print(' ')
            print('Write data')
            print('----------')
            idd_out.edge_profiles.profiles_1d.resize(100)
            print('rho_pol_norm =', rho_pol_norm)
            idd_out.edge_profiles.profiles_1d[0].grid.rho_tor_norm = rho_pol_norm[0, :]
            idd_out.edge_profiles.ids_properties.homogeneous_time = 0
            idd_out.edge_profiles.put()

            idd_out.close()



        return rho_pol_norm.T, electron_density.T, rho_pol_norm_errors.T, electron_density_errors.T



            #####################################################################################################
            #####################################################################################################

    if datatype == 'ece':

        idd_in.ece.get()
        idd_in.equilibrium.get()

        mask_eq = np.asarray(idd_in.equilibrium.code.output_flag) > -1

        mask_eq_time = (idd_in.ece.time > idd_in.equilibrium.time[mask_eq][0]) \
                     & (idd_in.ece.time < idd_in.equilibrium.time[mask_eq][-1]) \

        nbr_channels        = len(idd_in.ece.channel)
        nbr_pts             = len(idd_in.ece.channel[0].position.r.data)
        nbr_temperature     = len(idd_in.ece.channel[0].t_e.data)


        matrix_position = np.full((nbr_pts, nbr_channels), np.nan)
        matrix_temperature = np.full((nbr_temperature, nbr_channels), np.nan)

        for channel in range(len(idd_in.ece.channel)):
            for raduis in range(len(idd_in.ece.channel[channel].position.r.data)):
                matrix_position[raduis][channel] = idd_in.ece.channel[channel].position.r.data[raduis]
            for temperature in range(len(idd_in.ece.channel[channel].t_e.data)):
                matrix_temperature[temperature][channel] = idd_in.ece.channel[channel].t_e.data[temperature]


        Time                                            = idd_in.ece.time[mask_eq_time]
        R_real                                          = matrix_position[mask_eq_time]
        electron_temperature                            = matrix_temperature[mask_eq_time]
        electron_temperature[electron_temperature < 0]  = np.nan
        R_real[np.isnan(electron_temperature)]          = np.nan

        ## Get errors on the temperature
        nbr_errors_Up       = len(idd_in.ece.channel[0].t_e.data_error_upper)
        nbr_errors_Low      = len(idd_in.ece.channel[0].t_e.data_error_lower)

        error_upper_temp = np.full((nbr_errors_Up, nbr_channels), np.nan)
        error_lower_temp = np.full((nbr_errors_Low, nbr_channels), np.nan)
        error_max_temp   = np.full((nbr_errors_Low, nbr_channels), np.nan)

        for channel in range(len(idd_in.ece.channel)):
            for error_up in range(len(idd_in.ece.channel[channel].t_e.data_error_upper)):
                error_upper_temp[error_up][channel] = idd_in.ece.channel[channel].t_e.data_error_upper[error_up]
            for error_low in range(len(idd_in.ece.channel[channel].t_e.data_error_lower)):
                error_lower_temp[error_low][channel] = idd_in.ece.channel[channel].t_e.data_error_lower[error_low]
                #choose the max of the upper and lower
                error_max_temp[error_low][channel] = 2.0*(max(error_upper_temp[error_low][channel],error_lower_temp[error_low][channel]))

        #mask and filter error data according to equilibrium time:
        temperature_error  = error_max_temp[mask_eq_time]
        temperature_error[np.isnan(electron_temperature)] = np.nan

        R_base = np.linspace(np.nanmin(R_real), np.nanmax(R_real), 1000)
        Phi = np.zeros(1000)
        Z   = np.zeros(1000)
        rho_pol_norm_base = equimap.get(shot, Time, R_base , Phi, Z, 'rho_pol_norm')
        '''
        rho_pol_norm           = [None]*R_real.shape[0]
        electron_temperature_2 = [None]*R_real.shape[0]
        rho_pol_norm_error     = [None]*R_real.shape[0]
        temperature_error_2    = [None]*R_real.shape[0]

        for ii in range(rho_pol_norm_base.shape[0]):
            rho_pol_norm[ii] = np.interp(R_real[ii, :][~np.isnan(R_real[ii, :])], \
                                         R_base, rho_pol_norm_base[ii, :])
            electron_temperature_2[ii] = electron_temperature[ii, :][~np.isnan(electron_temperature[ii, :])]
            rho_pol_norm_error[ii]  = np.full(rho_pol_norm[ii].shape, 0.01)
            temperature_error_2[ii] = temperature_error[ii, :][~np.isnan(temperature_error[ii, :])]

        rho_pol_norm           = np.asarray(rho_pol_norm)
        electron_temperature_2 = np.asarray(electron_temperature_2)
        rho_pol_norm_error     = np.asarray(rho_pol_norm_error)
        temperature_error_2    = np.asarray(temperature_error_2)
        '''
        import ipdb;ipdb.set_trace()

        rho_pol_norm           = np.full((R_real.shape), np.nan)
        electron_temperature_2 = np.full((R_real.shape), np.nan)
        rho_pol_norm_error     = np.full((R_real.shape), np.nan)
        temperature_error_2    = np.full((R_real.shape), np.nan)

        for ii in range(rho_pol_norm_base.shape[1]):
            rho_pol_norm[ii,:] = np.interp(R_real[ii, :][~np.isnan(R_real[ii, :])], \
                                         R_base, rho_pol_norm_base[ii, :])
            electron_temperature_2[ii,:] = electron_temperature[ii, :][~np.isnan(electron_temperature[ii, :])]
            rho_pol_norm_error[ii,:]  = np.full(rho_pol_norm[ii:,].shape, 0.01)
            temperature_error_2[ii,:] = temperature_error[ii, :][~np.isnan(temperature_error[ii, :])]
        import ipdb;ipdb.set_trace()
        rho_pol_norm           = np.asarray(rho_pol_norm)
        electron_temperature_2 = np.asarray(electron_temperature_2)
        rho_pol_norm_error     = np.asarray(rho_pol_norm_error)
        temperature_error_2    = np.asarray(temperature_error_2)
        return rho_pol_norm.T, electron_temperature_2.T, rho_pol_norm_error.T, temperature_error_2.T

            #####################################################################################################
            #####################################################################################################
            #####################################################################################################

    if datatype == 'interferometer':

        idd_in.interferometer.get()
        idd_in.reflectometer_profile.get()
        idd_in.equilibrium.get(occurrence=1)

        Time_eq = idd_in.equilibrium.time
        Time_inter = idd_in.interferometer.time
        Time_ref   = idd_in.reflectometer_profile.time
        #if ((Time_ref.size == 0) or (Time_inter == 0) or (Time_eq == 0 )) :
        #    raise RuntimeError('choose another shot that have a valid time')

        mask_reflectometer_exist = (Time_ref.size != 0)
        #import pdb; pdb.set_trace()
        if mask_reflectometer_exist :
            time_min = np.asarray([Time_eq.min(),Time_inter.min(),Time_ref.min()]).max()
            time_max = np.asarray([Time_eq.max(),Time_inter.max(),Time_ref.max()]).min()

            #mask over the times of reflectometer and interferometer
            mask_time_reflec = (Time_ref > time_min) & (Time_ref < time_max)
            mask_time_interf = (Time_inter > time_min) & (Time_inter < time_max)
            mask_time_equi   = (Time_eq > time_min)&(Time_eq < time_max)

        else: 
            time_min = np.asarray([Time_eq.min(),Time_inter.min()]).max()
            time_max = np.asarray([Time_eq.max(),Time_inter.max()]).min()

            #mask over the times of reflectometer and interferometer
            mask_time_reflec = (Time_ref > time_min) & (Time_ref < time_max)
            mask_time_interf = (Time_inter > time_min) & (Time_inter < time_max)
            mask_time_equi   = (Time_eq > time_min)&(Time_eq < time_max)
            
        TimeReference = Time_eq[mask_time_equi]
            
        if not mask_reflectometer_exist :
            Phi_meters = np.zeros(1000)
            Z_meters = np.zeros(1000)
            R_meters = np.linspace(2, 3.5, 1000)
            rho_mid_plane = equimap.get(shot, TimeReference,R_meters,Phi_meters,Z_meters, 'rho_pol_norm')

            list_of_nans = []
            for ii in range(rho_mid_plane.shape[0]):
                if (np.isnan(rho_mid_plane[ii]).all()):
                    list_of_nans.append(ii)
            
            if list_of_nans :
                TimeReference = np.delete(TimeReference, list_of_nans)
                #Time_eq = np.delete(Time_eq, list_of_nans)
                #Time_inter = np.delete(Time_inter, list_of_nans)

        #mask_time_ref = (TimeReference > 32.2824614) & (TimeReference < 47.06669449)
        #TimeReference = TimeReference[mask_time_ref]
        #TimeReference[0] =  TimeReference[0] + 0.001
        #TimeReference[-1] =  TimeReference[-1] - 0.001

        if mask_reflectometer_exist :
            ##########################
            ##########################
            #grab the reflectometer data and mask them
            R_real_ref1       = idd_in.reflectometer_profile.channel[0].position.r
            electron_density1 = idd_in.reflectometer_profile.channel[0].n_e.data
            R_real_ref1       = R_real_ref1[:,mask_time_reflec]
            electron_density1 = electron_density1[:,mask_time_reflec]
            electron_density  = []
            R_real_ref  = []

            for ii in range(electron_density1.shape[0]):
                electron_density.append(np.interp(TimeReference,Time_ref[mask_time_reflec],electron_density1[ii]))
                R_real_ref.append(np.interp(TimeReference,Time_ref[mask_time_reflec],R_real_ref1[ii]))


            electron_density = np.asarray(electron_density)
            R_real_ref = np.asarray(R_real_ref)


            integrale_density_ref = np.full((electron_density.shape), np.nan)
            for jj in np.arange(0,integrale_density_ref.shape[1]):
                integrale_density_ref[:,jj] = (-1)*(integrate.cumtrapz(electron_density[:,jj],R_real_ref[:,jj], initial=0))

            R_base_ref = np.linspace(R_real_ref.min(), R_real_ref.max(), 1000)
            Phi_ref = np.zeros(1000)
            Z_ref = np.zeros(1000)

            rho_pol_norm_base_ref1 = equimap.get(shot, Time_ref, R_base_ref, Phi_ref, Z_ref, 'rho_pol_norm')
            rho_pol_norm_base_ref = []
            for ii in range(rho_pol_norm_base_ref1.shape[1]):
                rho_pol_norm_base_ref.append(np.interp(TimeReference,Time_ref, rho_pol_norm_base_ref1[:,ii]))


            rho_pol_norm_base_ref = np.asarray(rho_pol_norm_base_ref)
            rho_pol_norm_ref = np.full(R_real_ref.shape, np.nan)
            for ii in range(rho_pol_norm_base_ref.shape[1]):
                rho_pol_norm_ref[:, ii] = np.interp(R_real_ref[:, ii], R_base_ref, rho_pol_norm_base_ref[:,ii])

         ##################################################################################

        #list of r and z boundary positions
        nbr_channels = len(idd_in.interferometer.channel)
        boundary_r   = []
        boundary_z   = []
        for ii in range(0, len(idd_in.equilibrium.time_slice)):
            boundary_r.append(idd_in.equilibrium.time_slice[ii].boundary.outline.r)
            boundary_z.append(idd_in.equilibrium.time_slice[ii].boundary.outline.z)

        boundary_r = np.asarray(boundary_r)
        boundary_z = np.asarray(boundary_z)

        #extract the minimum and the maximum of the raduis
        maximum = (boundary_r[0]).max()
        minimum = (boundary_r[0]).min()

        for ii in range(0, len(boundary_r)):
            for jj in range(0, len(boundary_r[ii])):
                if (boundary_r[ii][jj] > maximum):
                    maximum = boundary_r[ii][jj]
                if (boundary_r[ii][jj] < minimum):
                    minimum = boundary_r[ii][jj]

        #list of first, second points of line of sight and their vectors(length)
        R_inter_first   = []
        Z_inter_first   = []
        R_inter_second  = []
        Z_inter_second  = []
        Z = []
        R = []

        for ii in range(0, nbr_channels):
            R_inter_first.append(idd_in.interferometer.channel[ii].line_of_sight.first_point.r)
            Z_inter_first.append(idd_in.interferometer.channel[ii].line_of_sight.first_point.z)
            R_inter_second.append(idd_in.interferometer.channel[ii].line_of_sight.second_point.r)
            Z_inter_second.append(idd_in.interferometer.channel[ii].line_of_sight.second_point.z)
            Z.append(np.linspace(Z_inter_second[ii], Z_inter_first[ii], 1000))
            R.append(np.linspace(R_inter_second[ii], R_inter_first[ii], 1000))

        Z = np.asarray(Z)
        R = np.asarray(R)

        r_outboard = []
        r_outboard_interpolated = []
        r_axis = np.full(len(idd_in.equilibrium.time_slice), np.nan)
        z_axis = np.full(len(idd_in.equilibrium.time_slice), np.nan)

        for ii in range(r_axis.shape[0]):
            r_axis[ii] = idd_in.equilibrium.time_slice[ii].global_quantities.magnetic_axis.r
            z_axis[ii] = idd_in.equilibrium.time_slice[ii].global_quantities.magnetic_axis.z
            r_outboard.append(idd_in.equilibrium.time_slice[ii].profiles_1d.r_outboard)

        r_outboard = np.asarray(r_outboard)

        #interpolation from the equilibruim time to the Time Reference we are using
        for ii in range(r_outboard.shape[1]):
            r_outboard_interpolated.append(np.interp(TimeReference, Time_eq, r_outboard[:,ii]))

        r_outboard_interpolated = np.asarray(r_outboard_interpolated)

        #start the equimap procedure:
        Phi = np.zeros(1000)
        rho_pol_norm_base = []
        rho_pol_norm_base_min = []
        electron_density_line=[]
        mask_inter=[]
        
        for zz in range(0,nbr_channels):
            mask_inter.append((idd_in.interferometer.channel[zz].n_e_line.validity) > -1)
        
        for zz in range(0,nbr_channels):
            if mask_inter[zz] == True :
                rho_pol_norm_base.append(equimap.get(shot, TimeReference, R[zz], Phi, Z[zz], 'rho_pol_norm', occ=1))
                electron_density_line.append(idd_in.interferometer.channel[zz].n_e_line.data)
            #else:
            #    raise RuntimeError('please be aware that data in channel' + zz + 'are not valid')

        rho_pol_norm_base     = np.asarray(rho_pol_norm_base)
        electron_density_line = 0.25*np.asarray(electron_density_line)

        for zz in range(0,rho_pol_norm_base.shape[0]):
            rho_pol_norm_base_min.append(np.nanmin(rho_pol_norm_base[zz], axis=1))
        
        rho_pol_norm_base_min = np.asarray(rho_pol_norm_base_min)

        if all(mask_inter) == True :
            print('################################################################################')
            print('################################################################################')
            print('############# Interferometer data are valid and ready to be used ###############')
            print('################################################################################')
            print('################################################################################')
        time.sleep(3) # Sleep for 3 seconds
        

        #####################################################################################################
        ##################################### equimaps in meters ############################################
        #####################################################################################################
        #####################################################################################################
        #The procedure here now is to calculate the desnity along line of sight (integrated density)
        #and also find the density along the rho_mid_plane i.e z=0
        #in order to normalise the desnity
        if mask_reflectometer_exist :
            R_meters = np.linspace(r_axis.min(), R_real_ref.max(), 1000)
        else:
            r_outboard_interpolated_max = np.nanmax(r_outboard_interpolated[0])
            r_outboard_interpolated_min = np.nanmin(r_outboard_interpolated[0])

            
            for ii in range(r_outboard_interpolated.shape[0]):
                if r_outboard_interpolated_max < np.nanmax(r_outboard_interpolated[ii]):
                    r_outboard_interpolated_max = np.nanmax(r_outboard_interpolated[ii])
                if r_outboard_interpolated_min < np.nanmin(r_outboard_interpolated[ii]):
                    r_outboard_interpolated_min = np.nanmin(r_outboard_interpolated[ii])
                

            R_meters = np.linspace(r_outboard_interpolated_min, r_outboard_interpolated_max, 1000)

        Phi_meters = np.zeros(1000)
        Z_meters = np.zeros(1000)
        rho_mid_plane = equimap.get(shot, TimeReference,R_meters,Phi_meters,Z_meters, 'rho_pol_norm')

        #import pdb; pdb.set_trace()
        index_rho_min = np.full(rho_pol_norm_base_min.shape, np.nan)
        index_separatrix = np.full(rho_pol_norm_base_min.shape, np.nan)

        for ii in range(rho_pol_norm_base_min.shape[0]):
            index_rho_min[ii, :] =np.nanargmin(np.abs(rho_pol_norm_base[ii, :, :] - rho_pol_norm_base_min[ii, :, np.newaxis]), axis=1)
            for jj in range(rho_pol_norm_base_min.shape[1]):
                index_separatrix[ii,:] = np.nanargmin(np.abs(rho_pol_norm_base[ii, jj, index_rho_min[ii, jj].astype(int):] - 1))

        R_separatrix = np.full(rho_pol_norm_base_min.shape, np.nan)
        Z_separatrix = np.full(rho_pol_norm_base_min.shape, np.nan)
        R_LOS = np.full(rho_pol_norm_base_min.shape, np.nan)
        Z_LOS = np.full(rho_pol_norm_base_min.shape, np.nan)

        for ii in range(rho_pol_norm_base_min.shape[0]):
            for jj in range(rho_pol_norm_base_min.shape[1]):
                R_separatrix[ii,jj] = R[ii, (index_rho_min[ii,jj].astype(int)):][index_separatrix[ii,jj].astype(int)]
                Z_separatrix[ii,jj] = Z[ii, (index_rho_min[ii,jj].astype(int)):][index_separatrix[ii,jj].astype(int)]
                R_LOS[ii,jj] = R[ii, index_rho_min[ii,jj].astype(int)]
                Z_LOS[ii,jj] = Z[ii, index_rho_min[ii,jj].astype(int)]


        length_1 = np.sqrt((R_separatrix-R_LOS)**2 + (Z_separatrix-Z_LOS)**2)
        #rho_mid_plane calculations

        index_rho_mid_plane_min = np.full(rho_pol_norm_base_min.shape, np.nan)
        index_sep_mid_plane_min = np.full(index_rho_mid_plane_min.shape[1], np.nan)
        R_mid_left = np.full(rho_pol_norm_base_min.shape, np.nan)
        R_mid_right = np.full(rho_pol_norm_base_min.shape[1], np.nan)
        
        for ii in range(index_rho_mid_plane_min.shape[0]):
            for jj in range(index_rho_mid_plane_min.shape[1]):
                if not (np.isnan(rho_mid_plane[jj]).all()):
                    index_rho_mid_plane_min[ii,jj]  =  np.nanargmin((np.abs(rho_mid_plane[jj] - rho_pol_norm_base_min[ii,jj])))
                    R_mid_left[ii,jj]               =  R_meters[index_rho_mid_plane_min[ii,jj].astype(int)]
                    index_sep_mid_plane_min[jj]     =  np.nanargmin((np.abs(rho_mid_plane[jj] - 1)))
                    R_mid_right[jj]                 =  R_meters[index_sep_mid_plane_min[jj].astype(int)]

        length_2 = np.sqrt((R_mid_right-R_mid_left)**2)

        #the normalization constant used uppon normalizing the electron density from interferometry
        if mask_reflectometer_exist :
            Normalization_constant  = length_2/length_1
        else:
            Normalization_constant = length_2/length_1#1
            Normalization_constant_prime = length_2/length_1
            Normalization_constant[np.isnan(Normalization_constant)] = 0
            Normalization_constant[np.isinf(Normalization_constant)] = 0

        #####################################################################################################
        #interpolate the reflectometer integrals and the electron density from the interferometer
        # to the same time window
        #import pdb; pdb.set_trace()
        electron_density_ne = np.full((len(electron_density_line),len(TimeReference)), np.nan)

        for jj in range(electron_density_line.shape[0]):
            electron_density_ne[jj] = np.interp(TimeReference, Time_inter, electron_density_line[jj,:])

        electron_density_ne = Normalization_constant*electron_density_ne
        #import pdb; pdb.set_trace()

        if mask_reflectometer_exist :
            #prepare for concatenation of density from interferometry and reflectometry:
            ne_line_total = np.concatenate((electron_density_ne,integrale_density_ref))
            #prepare for concatenation of rho from interferometry and reflectometry:
            rho_total = np.full((ne_line_total.shape) , np.nan)
            for jj in range(rho_total.shape[1]):
                rho_total[:,jj] = np.concatenate((rho_pol_norm_base_min[:,jj],rho_pol_norm_ref[:,jj]))


        else:
            #prepare for concatenation of density from interferometry and reflectometry:
            ne_line_total = ((electron_density_ne))
            #prepare for concatenation of rho from interferometry and reflectometry:
            rho_total = np.full((ne_line_total.shape) , np.nan)
            for jj in range(rho_total.shape[1]):
                rho_total[:,jj] = (rho_pol_norm_base_min[:,jj])
            

        #prepare for sorting arrays :
        index_sort = np.argsort(rho_total, axis=0)
        rho_total_sort = (np.array(list(map(lambda x, y: y[x], index_sort.T, rho_total.T))))
        ne_line_total_sort = np.array(list(map(lambda x, y: y[x], index_sort.T, ne_line_total.T)))
        '''
        if((np.any(np.isnan(ne_line_total_sort))) or (np.any(np.isinf(ne_line_total_sort)))):
            print('please check the following non finite data elements')
            NaNs_index = np.argwhere(np.isnan(ne_line_total_sort))
            Infs_index = np.argwhere(np.isinf(ne_line_total_sort))
            print('NaNs_index = '  ,NaNs_index)
            print('Infs_index = '  ,Infs_index)
            raise RuntimeError('array must not contain infs or NaNs')
            '''

        ne_line_total_sort_final_nonan = ne_line_total_sort
        rho_total_sort_final_nonan = rho_total_sort


        ne_line_total_sort_final_error_added = np.full(ne_line_total_sort_final_nonan.shape, (ne_line_total_sort_final_nonan)*0.03)
        rho_total_sort_final_error_added = np.full(rho_total_sort_final_nonan.shape,np.mean(rho_total_sort_final_nonan)*0.01)
        ne_line_total_sort_final_error_added= np.clip(ne_line_total_sort_final_error_added, absolute_error_ne, None)

        time_global= TimeReference



        #import pdb; pdb.set_trace()
        out_put_All_added12 = fit_data(rho_total_sort_final_nonan , ne_line_total_sort_final_nonan , rho_total_sort_final_error_added, \
                                ne_line_total_sort_final_error_added , kernel_method='Gibbs_Kernel', \
                                optimise_all_params=False, slices_nbr=30, plot_fit=True, x_fix_data=None, dy_fix_data=None, \
                                dy_fix_err=None, Time_real=time_global, file_name = 'GPPlots_Data_Rho')


        #extract the fit results:
        #ne_line_density_fit_All_added = (np.asarray(out_put_All_added12['hs_fit_y']))
        ne_line_density_fit_All_added = (np.asarray(out_put_All_added12['ni_fit_y']))
        rho_total_fit_x_All_added =  (np.asarray(out_put_All_added12['fit_x']))

        #add the time slices as an output for the data_fit function
        time_slices_red_All_added = out_put_All_added12['fit_time_slice']





        if mask_reflectometer_exist :
            mask_total_All_added = []
            ne_line_interpolated_R_All_added = []
            derivative_interp_array_All_added = []
            for ii in range(len(time_slices_red_All_added)):
                mask_diff_All_added = np.diff(rho_mid_plane[time_slices_red_All_added[ii],:]) > 0
                mask_rho_fit_All_added = ((rho_mid_plane[time_slices_red_All_added[ii],:]>rho_total_fit_x_All_added[ii].min()) \
                                   & (rho_mid_plane[time_slices_red_All_added[ii],:]<rho_total_fit_x_All_added[ii].max()))

                mask_diff_All_added = np.insert(mask_diff_All_added, False, 0)
                mask_total_All_added.append(mask_rho_fit_All_added & mask_diff_All_added)

            mask_total_All_added = np.asarray(mask_total_All_added)

            for ii in range(len(time_slices_red_All_added)):
                ne_line_interpolated_R_All_added.append(np.interp(rho_mid_plane[time_slices_red_All_added[ii],mask_total_All_added[ii]], \
                                                    rho_total_fit_x_All_added[ii,:], ne_line_density_fit_All_added[ii,:]))

            ne_line_interpolated_R_All_added = np.asarray(ne_line_interpolated_R_All_added)



            ######################################################################################################
            #derivative of the interpolated density over space
            for ii in range(len(time_slices_red_All_added)):
                derivative_interp_array_All_added.append(np.gradient(ne_line_interpolated_R_All_added[ii], R_meters[mask_total_All_added[ii]]))

            derivative_interp_array_All_added = np.asarray(derivative_interp_array_All_added)
            ######################################################################################################

            R_meters_mask_All_added = np.full((len(time_slices_red_All_added), 100),np.nan)
            ne_line_interpolated_R_2d_All_added = np.full(R_meters_mask_All_added.shape, np.nan)
            ne_derivative_interpolated_2d_All_added = np.full(R_meters_mask_All_added.shape, np.nan)

            for ii in range(len(time_slices_red_All_added)):
                R_meters_mask_All_added[ii]=np.linspace(np.nanmin(R_meters[mask_total_All_added[ii]]), np.nanmax(R_meters[mask_total_All_added[ii]]), 100)
                ne_line_interpolated_R_2d_All_added[ii] = np.interp(R_meters_mask_All_added[ii], (R_meters[mask_total_All_added[ii]]), \
                                                                ne_line_interpolated_R_All_added[ii])
                ne_derivative_interpolated_2d_All_added[ii] = np.interp(R_meters_mask_All_added[ii], (R_meters[mask_total_All_added[ii]]), \
                                                                    derivative_interp_array_All_added[ii])





        else:
            ######################################################################################################
            mask_total_All_added = []
            ne_line_interpolated_R_All_added = []
            derivative_interp_array_All_added = []
            for ii in range(len(time_slices_red_All_added)):
                mask_diff_All_added = np.diff(rho_mid_plane[time_slices_red_All_added[ii],:]) > 0
                mask_rho_fit_All_added = ((rho_mid_plane[time_slices_red_All_added[ii],:]>rho_total_fit_x_All_added[ii].min()) \
                                   & (rho_mid_plane[time_slices_red_All_added[ii],:]<rho_total_fit_x_All_added[ii].max()))

                mask_diff_All_added = np.insert(mask_diff_All_added, True, 0)
                mask_total_All_added.append(mask_rho_fit_All_added & mask_diff_All_added)

            mask_total_All_added = np.asarray(mask_total_All_added)

            for ii in range(len(time_slices_red_All_added)):
                ne_line_interpolated_R_All_added.append(np.interp(rho_mid_plane[time_slices_red_All_added[ii],mask_total_All_added[ii]], \
                                                    rho_total_fit_x_All_added[ii,:], ne_line_density_fit_All_added[ii,:]))

            ne_line_interpolated_R_All_added = np.asarray(ne_line_interpolated_R_All_added)



            #derivative of the interpolated density over space

            derivative_interp_array_All_added = []
            for ii in range(len(time_slices_red_All_added)):
                derivative_interp_array_All_added.append(np.gradient(ne_line_interpolated_R_All_added[ii], R_meters[mask_total_All_added[ii]]))

            derivative_interp_array_All_added = np.asarray(derivative_interp_array_All_added)
            ######################################################################################################

            R_meters_mask_All_added = np.full((len(time_slices_red_All_added), 100),np.nan)
            ne_line_interpolated_R_2d_All_added = np.full(R_meters_mask_All_added.shape, np.nan)
            ne_derivative_interpolated_2d_All_added = np.full(R_meters_mask_All_added.shape, np.nan)
            for ii in range(len(time_slices_red_All_added)):
                R_meters_mask_All_added[ii]=np.linspace(np.nanmin(R_meters), np.nanmax(R_meters), 100)

            ne_line_interpolated_R_2d_All_added = ne_line_density_fit_All_added
            ne_derivative_interpolated_2d_All_added = derivative_interp_array_All_added



        #create the errors assigned to each of density and space:
        ne_line_interpolated_R_2d_errors_All_added = np.full(ne_line_interpolated_R_2d_All_added.shape, ((ne_line_interpolated_R_2d_All_added))*0.1)
        R_meters_2d_errors_All_added =  np.full(R_meters_mask_All_added.shape, np.mean((R_meters_mask_All_added))*0.01)
        ne_line_interpolated_R_2d_errors_All_added      = np.clip(ne_line_interpolated_R_2d_errors_All_added, absolute_error_ne, None)

        print('------------------------------------------')
        print('------------------------------------------')
        print('----fit_data for the All_added in R_meters----')
        print('------------------------------------------')
        print('------------------------------------------')


        time_real_upp = TimeReference[time_slices_red_All_added]

        out_put_R_All_added = fit_data(R_meters_mask_All_added, ne_line_interpolated_R_2d_All_added, R_meters_2d_errors_All_added, \
                                   ne_line_interpolated_R_2d_errors_All_added , kernel_method=args.kernel, \
                                   optimise_all_params=False, slices_nbr=30, plot_fit=True, x_fix_data=None, \
                                   dy_fix_data=None, dy_fix_err=None, Time_real=time_real_upp, file_name = 'GPPlots_Data_Meters')
      



        print('---------------------------------------------------------------')
        print('---------------------------------------------------------------')
        print('----Transformation from R_space to rho_space for All_added ----')
        print('---------------------------------------------------------------')
        print('---------------------------------------------------------------')


        time_slices_real_All_added = np.asarray(TimeReference[time_slices_red_All_added])

        rho_mid_plane_All_added = np.full((R_meters_mask_All_added.shape), np.nan)
        Phi_meters_trans = np.zeros(100)
        Z_meters_trans = np.zeros(100)
        #loop over time in the time array
        for ii in range(R_meters_mask_All_added.shape[0]):
            rho_mid_plane_All_added[ii] = equimap.get(shot,time_slices_real_All_added[ii] , R_meters_mask_All_added[ii], \
                                                  Phi_meters_trans, Z_meters_trans, 'rho_pol_norm')


        mask_All_added_rho = rho_mid_plane_All_added<np.nanmax(rho_pol_norm_base_min)
        rho_mid_plane_All_added_1 = np.ma.array(rho_mid_plane_All_added, mask = ~mask_All_added_rho, fill_value=np.nan)
        rho_mid_plane_All_added_masked = rho_mid_plane_All_added_1.filled(np.nan)


        #apply the mask to both the All_added and lower densities
        #if mask_reflectometer_exist :
        #derivative_density_All_added_fit_output = -(np.asarray(out_put_R_All_added['hs_fit_dydx']))
        derivative_density_All_added_fit_output = -(np.asarray(out_put_R_All_added['ni_fit_dydx']))
        #else :
        #derivative_density_All_added_fit_output = -(np.asarray(out_put_R_All_added['fit_dydx_weighted']))
            

        derivative_density_All_added_masked = np.ma.array(derivative_density_All_added_fit_output, mask = ~mask_All_added_rho, fill_value=np.nan)
        derivative_density_All_added = derivative_density_All_added_masked.filled(np.nan)

        average_der_density = derivative_density_All_added
        rho_total_final = rho_mid_plane_All_added_masked


        if mask_reflectometer_exist :
            #Interpolate the electron density to the time slices that we are working in
            #Interpolate the rho_pol_norm_ref that is associated to the electron density to the same time slices
            electron_density_interpolated=[]
            rho_pol_norm_ref_interpolated = []
            for ii in range(electron_density.shape[0]):
                rho_pol_norm_ref_interpolated.append(np.interp(time_slices_real_All_added, TimeReference, rho_pol_norm_ref[ii,:]))
                electron_density_interpolated.append(np.interp(time_slices_real_All_added, TimeReference, electron_density[ii,:]))
            electron_density_interpolated = (np.asarray(electron_density_interpolated)).T # transpose from (space,time) to (time, space)  shape
            rho_pol_norm_ref_interpolated = (np.asarray(rho_pol_norm_ref_interpolated)).T # transpose from (space,time) to (time, space)  shape

            rho_pol_norm_ref_concat = (np.concatenate((rho_total_final,rho_pol_norm_ref_interpolated), axis=1))#concatenate along the second axis ======> in space
            electron_density_concat = (np.concatenate((average_der_density, electron_density_interpolated), axis=1))#concatenate along the second axis ======> in space
            #define the errors for the final results:
            electron_density_interpolated_error = np.full(electron_density_interpolated.shape, electron_density_interpolated*0.03)
            rho_pol_norm_ref_interpolated_error = np.full(rho_pol_norm_ref_interpolated.shape, rho_pol_norm_ref_interpolated*0.01)

            rho_total_final_error      = np.full(rho_total_final.shape, rho_total_final*0.01)
            average_der_density_error  = np.full(average_der_density.shape, average_der_density*0.1)

            #concatenate the errors
            rho_pol_norm_ref_concat_error = (np.concatenate((rho_total_final_error,rho_pol_norm_ref_interpolated_error), axis=1))
            electron_density_concat_error = (np.concatenate((average_der_density_error,electron_density_interpolated_error), axis=1))


        else:
            rho_pol_norm_ref_concat = rho_total_final
            electron_density_concat = average_der_density
            #define the errors for the final results:

            rho_total_final_error      = np.full(rho_total_final.shape, rho_total_final*0.01)
            average_der_density_error  = np.full(average_der_density.shape, average_der_density*0.1)

                #concatenate the errors
            rho_pol_norm_ref_concat_error = rho_total_final_error
            electron_density_concat_error = average_der_density_error



        #prepare for sorting arrays and the array errors:

        array_index = (np.argsort(rho_pol_norm_ref_concat, axis=1))
        rho_total_sort_final = np.asarray(list(map(lambda x, y: y[x], array_index, rho_pol_norm_ref_concat)))
        ne_line_total_sort_final = np.asarray(list(map(lambda x, y: y[x], array_index, electron_density_concat)))
        
        rho_total_sort_final_error = np.asarray(list(map(lambda x, y: y[x], array_index, rho_pol_norm_ref_concat_error)))
        ne_line_total_sort_final_error = np.asarray(list(map(lambda x, y: y[x], array_index, electron_density_concat_error)))


        #check for nans in density and mask over the values that corresponds to the nans in the rho
        if np.isnan(ne_line_total_sort_final).any():
            mask_rho_total_sort_final = np.ma.array(rho_total_sort_final, mask = np.isnan(ne_line_total_sort_final), fill_value=np.nan)
            rho_total_sort_final = mask_rho_total_sort_final.filled(np.nan)
            
        #check for nans in rho and mask over the values that corresponds to the nans in the ne_profile
        if np.isnan(rho_total_sort_final).any():
            mask_ne_line_total_sort = np.ma.array(ne_line_total_sort_final, mask = np.isnan(rho_total_sort_final), fill_value=np.nan)
            ne_line_total_sort_final = mask_ne_line_total_sort.filled(np.nan)



        #check for nans in density error and mask over the values that corresponds to the nans in the rho
        if np.isnan(ne_line_total_sort_final_error).any():
            mask_rho_total_sort_final_error  = np.ma.array(rho_total_sort_final_error , mask = np.isnan(ne_line_total_sort_final_error ), fill_value=np.nan)
            rho_total_sort_final_error  = mask_rho_total_sort_final_error.filled(np.nan)
            
        #check for nans in rho error and mask over the values that corresponds to the nans in the ne_profile
        if np.isnan(rho_total_sort_final_error).any():
            mask_ne_line_total_sort_error  = np.ma.array(ne_line_total_sort_final_error , mask = np.isnan(rho_total_sort_final_error ), fill_value=np.nan)
            ne_line_total_sort_final_error  = mask_ne_line_total_sort_error.filled(np.nan)
            print('1')


        ############################################################################
        ############################################################################
        ############################################################################

        ##check for the deleting the nans and times from the same matrices indices
        
        #start the new procedure concerning fisation of the grid
        fixed_grid = True
        if fixed_grid:
            from scipy import interpolate

            channel_dim = ne_line_total_sort_final.shape[0]
            time_dim = 100
            ne_line_final_fixed_grid = np.full((channel_dim, time_dim), np.nan)
            rho_final_fixed_grid = np.full((channel_dim, time_dim), np.nan)

            rho_total_sort_final_nonan_list = []
            ne_line_total_sort_final_nonan_list = []
            
            for ii in range(ne_line_total_sort_final.shape[0]):
                    rho_total_sort_final_nonan_list.append(rho_total_sort_final[ii][~np.isnan(rho_total_sort_final[ii])])
                    ne_line_total_sort_final_nonan_list.append(ne_line_total_sort_final[ii][~np.isnan(ne_line_total_sort_final[ii])])
                    min_in_rho = min(rho_total_sort_final_nonan_list[ii])
                    max_in_rho = max(rho_total_sort_final_nonan_list[ii])
                    array_to_interpolate_into = np.linspace(min_in_rho, max_in_rho, time_dim)
                    rho_total_sort_final_nonan_list[ii] = np.asarray(rho_total_sort_final_nonan_list[ii])
                    ne_line_total_sort_final_nonan_list[ii] = np.asarray(ne_line_total_sort_final_nonan_list[ii])
                    interpolation_func = interpolate.interp1d(rho_total_sort_final_nonan_list[ii], ne_line_total_sort_final_nonan_list[ii])
                    rho_final_fixed_grid[ii] = array_to_interpolate_into
                    ne_line_final_fixed_grid[ii] = interpolation_func(array_to_interpolate_into)


            rho_total_sort_nonans_final     = rho_final_fixed_grid 
            ne_line_total_sort_nonans_final = ne_line_final_fixed_grid

            ################
            ################
            #error fixed grid
            ne_line_final_fixed_grid_error = np.full((channel_dim, time_dim), np.nan)
            rho_final_fixed_grid_error = np.full((channel_dim, time_dim), np.nan)

            rho_total_sort_nonans_final_list_error = []
            ne_line_total_sort_nonans_final_list_error = []

            for ii in range(ne_line_total_sort_final_error.shape[0]):
                    rho_total_sort_nonans_final_list_error.append(rho_total_sort_final_error[ii][~np.isnan(rho_total_sort_final_error[ii])])
                    ne_line_total_sort_nonans_final_list_error.append(ne_line_total_sort_final_error[ii][~np.isnan(ne_line_total_sort_final_error[ii])])
                    min_in_rho_error = min(rho_total_sort_nonans_final_list_error[ii])
                    max_in_rho_error = max(rho_total_sort_nonans_final_list_error[ii])
                    array_to_interpolate_into_error  = np.linspace(min_in_rho_error, max_in_rho_error, time_dim)
                    rho_total_sort_nonans_final_list_error[ii] = np.asarray(rho_total_sort_nonans_final_list_error[ii])
                    ne_line_total_sort_nonans_final_list_error[ii] = np.asarray(ne_line_total_sort_nonans_final_list_error[ii])
                    interpolation_func_error = interpolate.interp1d(rho_total_sort_nonans_final_list_error[ii], ne_line_total_sort_nonans_final_list_error[ii])
                    rho_final_fixed_grid_error[ii] = array_to_interpolate_into_error
                    ne_line_final_fixed_grid_error[ii] = interpolation_func_error(array_to_interpolate_into_error)



            rho_total_sort_nonans_final_error     = rho_final_fixed_grid_error
            ne_line_total_sort_nonans_final_error = ne_line_final_fixed_grid_error




        ############################################################################
        ############################################################################
        ############################################################################
        #change the error to global parameter
        # for inter and reflectormetry and for the meters 
        ne_line_total_sort_nonans_final_error= np.clip(ne_line_total_sort_nonans_final_error, absolute_error_ne, None)

        
        #add an extra point to rho total
        minimuim_rho_elements_array = []
        for ii in range(rho_total_sort_nonans_final.shape[0]):
            minimuim_rho_elements_array.append(np.nanmin(rho_total_sort_nonans_final[ii]))
        minimuim_rho_elements_array = np.asarray(minimuim_rho_elements_array)

        rho_total_sort_nonans_final = np.concatenate((minimuim_rho_elements_array[:,None], rho_total_sort_nonans_final),axis=1)
        rho_total_sort_nonans_final_error = np.concatenate((minimuim_rho_elements_array[:,None]*0.01, rho_total_sort_nonans_final_error),axis=1)

        #rho_total_sort_nonans_final = np.insert(rho_total_sort_nonans_final, 0, 0, axis=1)#index, value
        #rho_total_sort_nonans_final_error = np.insert(rho_total_sort_nonans_final_error, 0, 0, axis=1)#index, value
        
        #the maximum should be by time slice and should
        maximum_elements_array = []
        for ii in range(ne_line_total_sort_nonans_final.shape[0]):
            maximum_elements_array.append(np.nanmax(ne_line_total_sort_nonans_final[ii]))
        maximum_elements_array = np.asarray(maximum_elements_array)

        #concatenate the array of the first elements with the total density elements
        ne_line_total_sort_nonans_final = np.concatenate((maximum_elements_array[:,None], ne_line_total_sort_nonans_final),axis=1)
        ne_line_total_sort_nonans_final_error = np.concatenate((maximum_elements_array[:,None]*0.2,ne_line_total_sort_nonans_final_error),axis=1)
        error_max_added = max(0.2*absolute_error_ne, np.max(maximum_elements_array[:,None]*0.2))
        
        ne_line_total_sort_nonans_final_error = np.clip(ne_line_total_sort_nonans_final_error, absolute_error_ne, error_max_added)




        out_put_final = fit_data(rho_total_sort_nonans_final , ne_line_total_sort_nonans_final , rho_total_sort_nonans_final_error, \
                               ne_line_total_sort_nonans_final_error , kernel_method='Gibbs_Kernel', \
                                optimise_all_params=False, slices_nbr=30, plot_fit=True, x_fix_data=None, dy_fix_data=None, \
                                dy_fix_err=None, Time_real=time_global, file_name = 'GPPlots_final_FITS')

        ne_density_fit = (np.asarray(out_put_final['ni_fit_y']))
        ne_density_fit_error = (np.asarray(out_put_final['ni_fit_y_error']))
        #ne_density_fit = (np.asarray(out_put_final['hs_fit_y']))
        #ne_density_fit_error = (np.asarray(out_put_final['hs_fit_y_error']))
        rho_total_fit =  (np.asarray(out_put_final['fit_x']))
        Time_index = np.asarray(out_put_final['fit_time_slice'])



        #interpolate rho_pol_norm_base along time and space to rho_total_fit

        density_pol_norm_base_interp = np.full((rho_pol_norm_base.shape[0],rho_total_fit.shape[0],rho_pol_norm_base.shape[2]),np.nan)

        for ii in range(density_pol_norm_base_interp.shape[0]):
            for jj in range(density_pol_norm_base_interp.shape[1]):
                density_pol_norm_base_interp[ii,jj] = np.interp(rho_pol_norm_base[ii, Time_index[jj]], \
                                                                rho_total_fit[jj],  ne_density_fit[jj])#, left=0, right=0)


        rho_pol_norm_base_sample = rho_pol_norm_base[:, Time_index, :] #will have the size of (line of sight, Time_index, space)
        mask_profile_rho = np.ones(rho_pol_norm_base_sample.shape, dtype=bool)
        for ii in range(mask_profile_rho.shape[1]):
            mask_profile_rho[:,ii,:] = rho_pol_norm_base_sample[:,ii,:]<np.nanmax(rho_total_fit[ii])#loop on time slices for the rho_total_fit

        R_0 = np.full((R.shape[0]), np.nan)
        Z_0 = np.full((R.shape[0]), np.nan)
        distance_length = np.full((R.shape), np.nan)

        #create a loop over the line of sight
        for ii in range(R.shape[0]):
            R_0[ii] = R[ii,0]
            Z_0[ii] = Z[ii,0]
            distance_length[ii] = np.sqrt((R[ii]-R_0[ii])**2 + (Z[ii]-Z_0[ii])**2)


        #the final step will be to integrate the density
        #over the distance in space, trapizodal integration
        integrale_density_final = np.full((density_pol_norm_base_interp.shape[0],density_pol_norm_base_interp.shape[1]),np.nan)
        density_pol_norm_base_interp[np.isnan(density_pol_norm_base_interp)]=0

        for ii in range(density_pol_norm_base_interp.shape[0]):
            for jj in range(density_pol_norm_base_interp.shape[1]):
                integrale_density_final[ii, jj] = 0.5*(integrate.trapz(density_pol_norm_base_interp[ii, jj],distance_length[ii]))


        density_check = (electron_density_ne/Normalization_constant)
        density_check[np.isnan(density_check)] = 0
       
        error_difference = integrale_density_final - density_check[:,Time_index]
        error_difference_percent  = (integrale_density_final - density_check[:,Time_index])/(density_check[:,Time_index])
        error_difference_percent[np.isinf(error_difference_percent)] = 0
        RMS = np.sqrt(np.mean((error_difference)**2, axis=0)) 
        RMSE = (np.sqrt(np.mean((error_difference_percent)**2, axis=0)))*100 # it is the mean along the time axis

        chi_sqaure = ((ne_line_total_sort_nonans_final[:,1:] - ne_density_fit)**2)/((ne_density_fit_error)**2)

        ###Some basic setup
        #create the test directory to save  output files
        #user can add as much as he wants here
        output_save_directory = './output_directory'
        if not output_save_directory.endswith('/'):
            test_save_directory = output_save_directory+'/'
        if not os.path.isdir(output_save_directory):
            os.makedirs(output_save_directory)
        os.chdir(output_save_directory)
        #save the outputs to files to be loaded and used in plots
        np.savez('statistics', RMSE_mean=np.nanmean(RMSE) , RMSE_median=np.nanmedian(RMSE), RMSE_min=np.nanmin(RMSE), RMSE_max=np.nanmax(RMSE) , Chi_sqaure=chi_sqaure)
        np.savez('Time_file', Time_index=Time_index, time_global=time_global)
        np.savez('Error_files', error_difference=error_difference, error_difference_percent=error_difference_percent, RMSE=RMSE, RMS=RMS)
        if mask_reflectometer_exist :
            np.savez('Rhos_vs_ne', rho_pol_norm_base_min=rho_pol_norm_base_min , rho_pol_norm_ref=rho_pol_norm_ref , electron_density_ne=electron_density_ne , integrale_density_ref=integrale_density_ref, density_check=density_check, integrale_density_final=integrale_density_final)
        else:
            np.savez('Rhos_vs_ne', rho_pol_norm_base_min=rho_pol_norm_base_min , electron_density_ne=electron_density_ne ,  density_check=density_check, integrale_density_final=integrale_density_final)
            
        os.chdir('../')
        os.chdir('../')
        print('current working directory')
        print(os.getcwd())
        import pdb; pdb.set_trace()


        if (write_edge_profiles):
            #####################################################################################################
            ### save the output to the edge profiles as a start

            # Create or open IDS
            # ------------------
            run_number = '{:04d}'.format(run_out)
            shot_file  = os.path.expanduser('~' + user_out + '/public/imasdb/' \
                                                + machine_out + '/3/0/' + 'ids_' + str(shot) \
                                                + run_number + '.datafile')

            idd_out = imas.ids(shot, run_out)

            if (os.path.isfile(shot_file)):
                print('open the IDS')
                idd_out.open_env(user_out, machine_out, '3')
            else:
                if (user_out == 'imas_public'):
                    print('ERROR IDS file does not exist, the IDS file must be')
                    print('created first for imas_public user_out')
                    raise FileNotFoundError
                else:
                    print('Create the IDS')
                    idd_out.create_env(user_out, machine_out, '3')
            # Write data
            # ----------
            print(' ')
            print('Write data')
            print('----------')

            idd_out.edge_profiles.profiles_1d.resize(rho_total_fit.shape[0])
            for ii in range(rho_total_fit.shape[0]):
                idd_out.edge_profiles.profiles_1d[ii].grid.rho_pol_norm = rho_total_fit[ii, :]
                idd_out.edge_profiles.profiles_1d[ii].grid.rho_pol_norm_error_upper =rho_total_sort_nonans_final_error[ii, :] 
                idd_out.edge_profiles.profiles_1d[ii].electrons.density = ne_density_fit[ii, :]
                idd_out.edge_profiles.profiles_1d[ii].electrons.density_error_upper = ne_density_fit_error[ii, :]
                idd_out.edge_profiles.profiles_1d[ii].electrons.density_fit.reconstructed = ne_density_fit[ii, :]
                idd_out.edge_profiles.profiles_1d[ii].electrons.density_fit.reconstructed_error_upper = ne_density_fit_error[ii, :]
                idd_out.edge_profiles.profiles_1d[ii].electrons.density_fit.measured = ne_line_total_sort_final_nonan[ii, :]
                idd_out.edge_profiles.profiles_1d[ii].electrons.density_fit.measured_error_upper = ne_line_total_sort_final_error_added[ii, :]
                idd_out.edge_profiles.profiles_1d[ii].electrons.density_fit.time_measurement = time_global[ii]
                idd_out.edge_profiles.profiles_1d[ii].electrons.density_fit.chi_squared = chi_sqaure[ii]

            idd_out.edge_profiles.profiles_1d[0].electrons.density_fit.source = ['reflectometer_profile.channel[0].n_e.data' , 'interferometer.channel[zz].n_e_line.data'] 
            idd_out.edge_profiles.profiles_1d[0].electrons.density_fit.time_measurement_slice_method.name = 'time_measurement_slice_method'
            idd_out.edge_profiles.profiles_1d[0].electrons.density_fit.time_measurement_slice_method.index = 1
            idd_out.edge_profiles.profiles_1d[0].electrons.density_fit.time_measurement_slice_method.description = 'linear interpolation'
            idd_out.edge_profiles.profiles_1d[0].electrons.density_fit.weight = 1
            #add the chi_square and save it 
            #add an put to interferomtry ids with input occurance (>4)
            #import pdb; pdb.set_trace()
            #add the time global 
            idd_out.edge_profiles.time = time_global 
            idd_out.edge_profiles.put()
            


            #add an put to interferomtry ids with input occurance (>4)
            #idd_out.interferometry.put(occ=4)
            
            idd_out.close()
            print('finish Writing data')



        return rho_total_fit, ne_density_fit, rho_total_sort_nonans_final_error, ne_density_fit_error


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
    parser.add_argument('-ids', '--ids', type=str, default='reflectometer_profile', \
                        help='IDS source of data for profile fit, default=reflectometer_profile')
    parser.add_argument('-k', '--kernel', type=str, default='RQ_Kernel', \
                        help='Kernel to use for profile fit, default=RQ_Kernel')
    parser.add_argument('-wep', '--write_edge_profiles', action='store_true', \
                        help='Write IDS edge_profiles')
    parser.add_argument('-plt', '--plot_fit', action='store_true', \
                        help='Save fit plots')

    args = parser.parse_args()

    # Call wrapper function
    x, y, ex, ey = get_data(args.shot, args.run_out, args.occurrence_out, args.user_out, args.machine_out, args.run_in, args.occurrence_in, args.user_in, args.machine_in, args.ids, args.write_edge_profiles)

