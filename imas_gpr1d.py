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
from scipy import integrate
# Local libraries
import equimap
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
        if ((Time_ref is None) or (Time_inter is None) or (Time_eq is None)) :
            raise RuntimeError('choose another shot that have a valid time')

        time_min = np.asarray([Time_eq.min(),Time_inter.min(),Time_ref.min()]).max()
        time_max = np.asarray([Time_eq.max(),Time_inter.max(),Time_ref.max()]).min()

        #mask over the times of reflectometer and interferometer
        mask_time_reflec = (Time_ref > time_min) & (Time_ref < time_max)
        mask_time_interf = (Time_inter > time_min) & (Time_inter < time_max)
        if (len(Time_ref[mask_time_reflec])< len(Time_inter[mask_time_interf])):
            TimeReference = Time_ref[mask_time_reflec]
        else:
            TimeReference = Time_inter[mask_time_interf]
        
        ##########################
        ##########################
        #grab the reflectometer data and mask them 
        R_real_ref       = idd_in.reflectometer_profile.channel[0].position.r.data
        electron_density = idd_in.reflectometer_profile.channel[0].n_e.data
        R_real_ref       = R_real_ref[:,mask_time_reflec]
        electron_density = electron_density[:,mask_time_reflec]

        integrale_density_ref = np.full((electron_density.shape), np.nan)
        for jj in np.arange(0,integrale_density_ref.shape[1]):
            integrale_density_ref[:,jj] = (-1)*(integrate.cumtrapz(electron_density[:,jj],R_real_ref[:,jj], initial=0))

        Phi_reflec = np.zeros(R_real_ref.shape[0])
        Z_reflec = np.zeros(R_real_ref.shape[0])
        rho_pol_norm_base_reflec = np.full((R_real_ref.shape), np.nan)
        #for ii in range(R_real_ref.shape[1]):
        for ii in range(20):
            rho_pol_norm_base_reflec[:,ii] = equimap.get(shot, Time_ref[ii], R_real_ref[:,ii], Phi_reflec, Z_reflec, 'rho_pol_norm')

        R_base_ref = np.linspace(R_real_ref.min(), R_real_ref.max(), 1000)
        Phi_ref = np.zeros(1000)
        Z_ref = np.zeros(1000)
        rho_pol_norm_base_ref = equimap.get(shot, Time_ref, R_base_ref, Phi_ref, Z_ref, 'rho_pol_norm')
        if rho_pol_norm_base_ref.shape != electron_density.shape :
            rho_pol_norm_base_ref = rho_pol_norm_base_ref.T
        else :
            rho_pol_norm_base_ref = rho_pol_norm_base_ref
            
        rho_pol_norm_base_ref = rho_pol_norm_base_ref[:,mask_time_reflec]
        rho_pol_norm_ref = np.full(R_real_ref.shape, np.nan)
        for ii in range(rho_pol_norm_base_ref.shape[1]):
            rho_pol_norm_ref[:, ii] = np.interp(R_real_ref[:, ii], R_base_ref, rho_pol_norm_base_ref[:, ii])

        electron_density_errors = np.full(electron_density.shape, np.mean(electron_density)*0.05)
        rho_pol_norm_errors =  np.full(rho_pol_norm_ref.shape, np.mean(rho_pol_norm_ref)*0.005)
        
         ##################################################################################

        #list of raduis and a list of z boundary positions
        nbr_channels      = len(idd_in.interferometer.channel)
        print(nbr_channels)
        boundary_r = []
        boundary_z = []
        for ii in range(0, len(idd_in.equilibrium.time_slice)):
            boundary_r.append(idd_in.equilibrium.time_slice[ii].boundary.outline.r)
            boundary_z.append(idd_in.equilibrium.time_slice[ii].boundary.outline.z)

        boundary_r = np.asarray(boundary_r)
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

        R_inter_second   = []
        Z_inter_second   = []

        R_inter_vector   = []
        Z_inter_vector   = []
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

        r_axis = np.full(len(idd_in.equilibrium.time_slice), None)
        z_axis = np.full(len(idd_in.equilibrium.time_slice), None)

        for ii in range(r_axis.shape[0]):
            r_axis[ii] = idd_in.equilibrium.time_slice[ii].global_quantities.magnetic_axis.r
            z_axis[ii] = idd_in.equilibrium.time_slice[ii].global_quantities.magnetic_axis.z




        #interpolation from the equilibruim time to the Time Reference we are using 
        r_axis_interp = np.interp(TimeReference, Time_eq, r_axis.astype(float))
        z_axis_interp = np.interp(TimeReference, Time_eq, z_axis.astype(float))


        index_r_axis = np.full((R.shape[0],r_axis_interp.shape[0]), np.nan)
        mask_upper_LOS = []
        mask_lower_LOS = []
        for ii in range(R.shape[0]):
            for jj in range(r_axis_interp.shape[0]):
                index_r_axis[ii,jj] =np.nanargmin(np.abs( R[ii] - r_axis_interp[jj]))



                
        #introduce the upper and lower masks:
        for ii in range(R.shape[0]):
            mask_upper_LOS.append(Z[ii, index_r_axis[ii,:].astype(int)]>z_axis_interp)
            mask_lower_LOS.append(Z[ii, index_r_axis[ii,:].astype(int)]<z_axis_interp)
            
        mask_upper_LOS = np.asarray(mask_upper_LOS) 
        mask_lower_LOS = np.asarray(mask_lower_LOS)

        #start the equimap procedure:
        Phi = np.zeros(1000)
        rho_pol_norm_base = []
        rho_pol_norm_base_min = []
        electron_density_line=[]

        for zz in range(0,nbr_channels):
            rho_pol_norm_base.append(equimap.get(shot, TimeReference, R[zz], Phi, Z[zz], 'rho_pol_norm', occ=1))
            rho_pol_norm_base_min.append(np.nanmin(rho_pol_norm_base[zz], axis=1))
            electron_density_line.append(idd_in.interferometer.channel[zz].n_e_line.data)

        rho_pol_norm_base             = np.asarray(rho_pol_norm_base)
        rho_pol_norm_base_min         = np.asarray(rho_pol_norm_base_min)
        electron_density_line         = 0.25*np.asarray(electron_density_line)



        #####################################################################################################
        ##################################### equimaps in meters ############################################
        #####################################################################################################
        #####################################################################################################

        R_meters = np.linspace(r_axis.min(), 3.03, 1000)
        Phi_meters = np.zeros(1000)
        Z_meters = np.zeros(1000)

        rho_mid_plane = equimap.get(shot, TimeReference,R_meters,Phi_meters,Z_meters, 'rho_pol_norm')



        index_rho_min = np.full(rho_pol_norm_base_min.shape, np.nan)
        index_separatrix = np.full(rho_pol_norm_base_min.shape, np.nan)

        for ii in range(rho_pol_norm_base_min.shape[0]):
            index_rho_min[ii, :] =np.nanargmin(np.abs(rho_pol_norm_base[ii, :, :] - rho_pol_norm_base_min[ii, :, np.newaxis]), axis=1)
            for jj in range(rho_pol_norm_base_min.shape[1]):
                index_separatrix[ii,:] = np.nanargmin(np.abs(rho_pol_norm_base[ii, jj, index_rho_min[ii, jj].astype(int):] - 1))


        #import ipdb; ipdb.set_trace()
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
                index_rho_mid_plane_min[ii,jj]  =  np.nanargmin((np.abs(rho_mid_plane[jj] - rho_pol_norm_base_min[ii,jj])))
                R_mid_left[ii,jj]               =  R_meters[index_rho_mid_plane_min[ii,jj].astype(int)]
                index_sep_mid_plane_min[jj]     =  np.nanargmin((np.abs(rho_mid_plane[jj] - 1)))
                R_mid_right[jj]                 =  R_meters[index_sep_mid_plane_min[jj].astype(int)]

        
        length_2 = np.sqrt((R_mid_right-R_mid_left)**2) 
        Normalization_constant  = length_2/length_1

        #####################################################################################################
        
      
        #interpolate the reflectometer integrals and the electron density from the interferometer
        # to the same time window
        electron_density_ne = np.full((len(electron_density_line),len(TimeReference)), np.nan)

        for jj in range(10):
            electron_density_ne[jj] = np.interp(TimeReference, Time_inter, electron_density_line[jj,:])

        electron_density_ne = Normalization_constant*electron_density_ne

        #prepare for concatenation :
        ne_line_total = np.concatenate((electron_density_ne,integrale_density_ref))
        rho_total = np.full((ne_line_total.shape) , np.nan)
        
        for jj in range(rho_total.shape[1]):
            rho_total[:,jj] = np.concatenate((rho_pol_norm_base_min[:,jj],rho_pol_norm_ref[:,jj]))

        #prepare for sorting arrays :
        index_sort = np.argsort(rho_total, axis=0)
        rho_total_sort = (np.array(list(map(lambda x, y: y[x], index_sort.T, rho_total.T)))).T
        ne_line_total_sort = np.array(list(map(lambda x, y: y[x], index_sort.T, ne_line_total.T))).T

        if((np.any(np.isnan(ne_line_total_sort))) or (np.any(np.isinf(ne_line_total_sort)))):
            print('please check the following non finite data elements')
            NaNs_index = np.argwhere(np.isnan(ne_line_total_sort))
            Infs_index = np.argwhere(np.isinf(ne_line_total_sort))
            print('NaNs_index = '  ,NaNs_index)
            print('Infs_index = '  ,Infss_index)
            raise RuntimeError('array must not contain infs or NaNs')






        #divide the elecron density into two parts, upper and lower using the upper and lower masks created above :
        
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################

        #######################################################################################################################

        rho_total_lower = np.full((ne_line_total.shape) , np.nan)
        electron_density_ne_lower = np.full(electron_density_ne.shape, np.nan)
        rho_pol_norm_base_min_lower = np.full((rho_pol_norm_base_min.shape) , np.nan)
        electron_density_ne_lower = np.ma.array(electron_density_ne, mask = ~mask_lower_LOS, fill_value=np.nan)
        rho_pol_norm_base_min_lower = np.ma.array(rho_pol_norm_base_min, mask = ~mask_lower_LOS, fill_value=np.nan)
        electron_density_ne_lower.filled(np.nan)
        rho_pol_norm_base_min_lower.filled(np.nan)


        #prepare for concatenation :
        ne_line_total_lower = np.concatenate((electron_density_ne_lower,integrale_density_ref))


        for jj in range(rho_total_lower.shape[1]):
            rho_total_lower[:,jj] = np.concatenate((rho_pol_norm_base_min_lower[:,jj],rho_pol_norm_ref[:,jj]))

        #prepare for sorting arrays :

        index_sort_lower = np.argsort(rho_total_lower, axis=0)
        rho_total_sort_lower = (np.array(list(map(lambda x, y: y[x], index_sort_lower.T, rho_total_lower.T)))).T
        ne_line_total_sort_lower = np.array(list(map(lambda x, y: y[x], index_sort_lower.T, ne_line_total_lower.T))).T


        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################

        #######################################################################################################################



        rho_total_upper = np.full((ne_line_total.shape) , np.nan)
        electron_density_ne_upper = np.full(electron_density_ne.shape, np.nan)
        rho_pol_norm_base_min_upper = np.full((rho_pol_norm_base_min.shape) , np.nan)
        electron_density_ne_upper = np.ma.array(electron_density_ne, mask = ~mask_upper_LOS, fill_value=np.nan)
        rho_pol_norm_base_min_upper = np.ma.array(rho_pol_norm_base_min, mask = ~mask_upper_LOS, fill_value=np.nan)
        electron_density_ne_upper.filled(np.nan)
        rho_pol_norm_base_min_upper.filled(np.nan)


        #prepare for concatenation :
        ne_line_total_upper = np.concatenate((electron_density_ne_upper,integrale_density_ref))


        for jj in range(rho_total_upper.shape[1]):
            rho_total_upper[:,jj] = np.concatenate((rho_pol_norm_base_min_upper[:,jj],rho_pol_norm_ref[:,jj]))

        #prepare for sorting arrays :

        index_sort_upper = np.argsort(rho_total_upper, axis=0)
        rho_total_sort_upper = (np.array(list(map(lambda x, y: y[x], index_sort_upper.T, rho_total_upper.T)))).T
        ne_line_total_sort_upper = np.array(list(map(lambda x, y: y[x], index_sort_upper.T, ne_line_total_upper.T))).T



        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################

        '''
        ne_line_total_errors_lower = np.full(ne_line_total_sort_lower.shape, np.mean(ne_line_total_sort_lower)*0.3)
        rho_total_errors_lower =  np.full(rho_total_sort_lower.shape, np.mean(rho_total_sort_lower)*0.01)
        
        out_put = fit_data(rho_total_sort_lower.T, (ne_line_total_sort_lower).T, rho_total_errors_lower.T, ne_line_total_errors_lower.T, kernel_method=args.kernel, \
                          optimise_all_params=True, slices_nbr=10, plot_fit=True, dx_data=None, dy_data=None, dy_err=None)

        import ipdb; ipdb.set_trace()
        '''
        
        ne_line_total_errors_upper = np.full(ne_line_total_sort_upper.shape, np.mean(ne_line_total_sort_upper)*0.3)
        rho_total_errors_upper =  np.full(rho_total_sort_upper.shape, np.mean(rho_total_sort_upper)*0.01)
        
        out_put = fit_data(rho_total_sort_upper.T, (ne_line_total_sort_upper).T, rho_total_errors_upper.T, ne_line_total_errors_upper.T, kernel_method=args.kernel, \
                          optimise_all_params=True, nbr_pts=100, slices_nbr=10, plot_fit=True, dx_data=None, dy_data=None, dy_err=None)

        import ipdb; ipdb.set_trace()
        



        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################

        #######################################################################################################################



        '''
        ne_line_total_errors = np.full(ne_line_total_sort.shape, np.mean(ne_line_total_sort)*0.3)
        rho_total_errors =  np.full(rho_total_sort.shape, np.mean(rho_total_sort)*0.01)
        
        out_put = fit_data(rho_total_sort.T, (ne_line_total_sort).T, rho_total_errors.T, ne_line_total_errors.T, kernel_method=args.kernel, \
                          optimise_all_params=True, slices_nbr=10, plot_fit=True, dx_data=None, dy_data=None, dy_err=None)
        '''
        ne_line_density_fit = np.asarray(out_put['fit_y'])
        rho_total_fit_x =  np.asarray(out_put['fit_x'])

        

        ne_line_interpolated_R = np.full((ne_line_density_fit.shape[0],rho_mid_plane.shape[0]),np.nan)
        for ii in range(ne_line_density_fit.shape[0]):
            ne_line_interpolated_R[ii,:]= np.interp(rho_mid_plane[:,ii], rho_total_fit_x[ii,:], ne_line_density_fit[ii,:])
        




        import ipdb; ipdb.set_trace()
        derivative_interp_array = np.full(ne_line_interpolated_R.shape, np.nan) 
        for ii in range(derivative_interp_array.shape[1]):
            derivative_interp_array[ii,:] = np.gradient(ne_line_interpolated_R[ii,:], R_meters[ii])






         
        import ipdb; ipdb.set_trace()
        return rho_total_sort_lower.T, ne_line_total_sort_lower.T, rho_total_errors_lower.T, ne_line_total_errors_lower.T


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
    parser.add_argument('-wep', '--write-edge-profiles', action='store_true', \
                        help='Write IDS edge_profiles')
    parser.add_argument('-plt', '--plot-fit', action='store_true', \
                        help='Save fit plots')

    args = parser.parse_args()

    # Call wrapper function
    x, y, ex, ey = get_data(args.shot, \
                                args.run_out, args.occurrence_out, args.user_out, args.machine_out, \
                                args.run_in, args.occurrence_in, args.user_in, args.machine_in, \
                                args.ids, args.write_edge_profiles)

    '''
    if x.ndim == 1:
        x = np.transpose(np.atleast_2d(x))

        
    if y.ndim == 1:
        y = np.transpose(np.atleast_2d(y))

    if ex.ndim == 1:
        ex = np.transpose(np.atleast_2d(ex))

    if ey.ndim == 1:
        ey = np.transpose(np.atleast_2d(ey))
    print(x.shape, y.shape)    
    
    import ipdb; ipdb.set_trace()
    '''
    out = fit_data(x, y, ex , ey, kernel_method=args.kernel, \
                       optimise_all_params=True, slices_nbr=10, plot_fit=True)
    #import pdb; pdb.set_trace()

