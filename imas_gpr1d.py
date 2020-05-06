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
        #take a small portion of the R_real to save time on the loop of equimaps
        #for ii in range(R_real_ref.shape[1]):
        for ii in range(10):
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
        
         ##################################################################################

        #list of raduis and a list of z boundary positions
        nbr_channels = len(idd_in.interferometer.channel)
        boundary_r   = []
        boundary_z   = []
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
        R_inter_second  = []
        Z_inter_second  = []
        R_inter_vector  = []
        Z_inter_vector  = []
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

        r_axis = np.full(len(idd_in.equilibrium.time_slice), np.nan)
        z_axis = np.full(len(idd_in.equilibrium.time_slice), np.nan)

        for ii in range(r_axis.shape[0]):
            r_axis[ii] = idd_in.equilibrium.time_slice[ii].global_quantities.magnetic_axis.r
            z_axis[ii] = idd_in.equilibrium.time_slice[ii].global_quantities.magnetic_axis.z


        #interpolation from the equilibruim time to the Time Reference we are using 
        r_axis_interp = np.interp(TimeReference, Time_eq, r_axis)
        z_axis_interp = np.interp(TimeReference, Time_eq, z_axis)


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

        R_meters = np.linspace(r_axis.min(), R_real_ref.max(), 1000)
        Phi_meters = np.zeros(1000)
        Z_meters = np.zeros(1000)
        
        rho_mid_plane = equimap.get(shot, TimeReference,R_meters,Phi_meters,Z_meters, 'rho_pol_norm')

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
        #upper electron density 
        rho_total_upper = np.full((ne_line_total.shape) , np.nan)
        ne_line_total_upper = np.full((ne_line_total.shape) , np.nan)
        electron_density_ne_upper = np.full(electron_density_ne.shape, np.nan)
        rho_pol_norm_base_min_upper = np.full((rho_pol_norm_base_min.shape) , np.nan)

        electron_density_ne_upper = np.ma.array(electron_density_ne, mask = ~mask_upper_LOS, fill_value=np.nan)
        rho_pol_norm_base_min_upper = np.ma.array(rho_pol_norm_base_min, mask = ~mask_upper_LOS, fill_value=np.nan)

        electron_density_ne_upper = electron_density_ne_upper.filled()
        rho_pol_norm_base_min_upper = rho_pol_norm_base_min_upper.filled()

        #prepare for concatenation :
        for jj in range(rho_total_upper.shape[1]):
            ne_line_total_upper[:,jj] = np.concatenate((electron_density_ne_upper[:,jj],integrale_density_ref[:,jj]))
            rho_total_upper[:,jj] = np.concatenate((rho_pol_norm_base_min_upper[:,jj],rho_pol_norm_ref[:,jj]))

        #prepare for sorting arrays :
        index_sort_upper = np.argsort(rho_total_upper, axis=0)
        rho_total_sort_upper = (np.array(list(map(lambda x, y: y[x], index_sort_upper.T, rho_total_upper.T)))).T
        ne_line_total_sort_upper = np.array(list(map(lambda x, y: y[x], index_sort_upper.T, ne_line_total_upper.T))).T
        
        ne_line_total_errors_upper = np.full((ne_line_total_sort_upper).shape, (ne_line_total_sort_upper)*0.05)
        rho_total_errors_upper =  np.full((rho_total_sort_upper).shape, (rho_total_sort_upper)*0.01)

        print('------------------------------')
        print('------------------------------')
        print('----fit_data for the upper----')
        print('------------------------------')
        print('------------------------------')
        out_put_upper = fit_data(rho_total_sort_upper.T, (ne_line_total_sort_upper).T, rho_total_errors_upper.T, ne_line_total_errors_upper.T, kernel_method=args.kernel, \
                          optimise_all_params=True, nbr_pts=100, slices_nbr=10, plot_fit=False, x_fix_data=None, dy_fix_data=None, dy_fix_err=None)

        ne_line_density_fit_upper = (np.asarray(out_put_upper['fit_y']))
        rho_total_fit_x_upper =  (np.asarray(out_put_upper['fit_x']))
  

        #add the time slices as an output for the data_fit function
        time_slices_red_upper = out_put_upper['fit_time_slice']
        mask_diff_upper = []
        ne_line_interpolated_R_upper = []
        derivative_interp_array_upper = []

        for ii in range(len(time_slices_red_upper)):
            mask_diff_upper.append((np.diff(rho_mid_plane[time_slices_red_upper[ii],:])>0))
            mask_rho_fit_upper= ((rho_mid_plane[time_slices_red_upper[ii],:]>rho_total_fit_x_upper[ii].min()) & (rho_mid_plane[time_slices_red_upper[ii],:]<rho_total_fit_x_upper[ii].max()))
        
        mask_diff_upper = np.asarray(mask_diff_upper)
        mask_diff_upper = np.insert(mask_diff_upper, False, 0, axis=1)
        mask_total_upper = mask_rho_fit_upper & mask_diff_upper

        for ii in range(len(time_slices_red_upper)):
            ne_line_interpolated_R_upper.append(np.interp(rho_mid_plane[time_slices_red_upper[ii],mask_total_upper[ii]], rho_total_fit_x_upper[ii,:], ne_line_density_fit_upper[ii,:]))
        
        ne_line_interpolated_R_upper = np.asarray(ne_line_interpolated_R_upper)
 
        ######################################################################################################
        #derivative of the interpolated density over space
        for ii in range(len(time_slices_red_upper)):
            derivative_interp_array_upper.append(np.gradient(ne_line_interpolated_R_upper[ii], R_meters[mask_total_upper[ii]]))
        
        derivative_interp_array_upper = np.asarray(derivative_interp_array_upper)
        ######################################################################################################

        R_meters_mask_upper = np.full((len(time_slices_red_upper), 100),np.nan)
        ne_line_interpolated_R_2d_upper = np.full(R_meters_mask_upper.shape, np.nan)
        ne_derivative_interpolated_2d_upper = np.full(R_meters_mask_upper.shape, np.nan)

        for ii in range(len(time_slices_red_upper)):
            R_meters_mask_upper[ii]=np.linspace(np.nanmin(R_meters[mask_total_upper[ii]]), np.nanmax(R_meters[mask_total_upper[ii]]), 100)
            ne_line_interpolated_R_2d_upper[ii] = np.interp(R_meters_mask_upper[ii], (R_meters[mask_total_upper[ii]]) ,ne_line_interpolated_R_upper[ii])
            ne_derivative_interpolated_2d_upper[ii] = np.interp(R_meters_mask_upper[ii], (R_meters[mask_total_upper[ii]]) ,derivative_interp_array_upper[ii])

        R_meters_mask_upper = R_meters_mask_upper.T
        ne_line_interpolated_R_2d_upper = ne_line_interpolated_R_2d_upper.T
        ne_line_interpolated_R_2d_errors_upper = np.full(ne_line_interpolated_R_2d_upper.shape, np.mean(ne_line_interpolated_R_2d_upper)*0.091)
        R_meters_2d_errors_upper =  np.full(R_meters_mask_upper.shape, np.mean(R_meters_mask_upper)*0.001)
 
        print('------------------------------------------')
        print('------------------------------------------')
        print('----fit_data for the upper in R_meters----')
        print('------------------------------------------')
        print('------------------------------------------')
        
        out_put_R_upper12 = fit_data(R_meters_mask_upper.T, (ne_line_interpolated_R_2d_upper).T, R_meters_2d_errors_upper.T, ne_line_interpolated_R_2d_errors_upper.T, kernel_method=args.kernel, \
                          optimise_all_params=True, slices_nbr=10, plot_fit=False, x_fix_data=None, dy_fix_data=None, dy_fix_err=None)


        # output the data necessary for the boundary conditions
        upper_maximum_R    = (np.asarray(out_put_R_upper12['fit_x'])).max(axis=1)
        upper_minimum_R    = (np.asarray(out_put_R_upper12['fit_x'])).min(axis=1)
        upper_derivative_R = (np.asarray(out_put_R_upper12['fit_dydx'])).min(axis=1)#take the maximum instead of the min


        out_put_R_upper = fit_data(R_meters_mask_upper.T, (ne_line_interpolated_R_2d_upper).T, R_meters_2d_errors_upper.T, ne_line_interpolated_R_2d_errors_upper.T, kernel_method=args.kernel, \
                          optimise_all_params=True, slices_nbr=10, plot_fit=False, x_fix_data=None, dy_fix_data=None, dy_fix_err=None, boundary_max=upper_maximum_R, boundary_min=upper_minimum_R, boundary_derv=upper_derivative_R)


        derivative_ne_line_interpolated_upper = np.asarray(out_put_R_upper['fit_dydx'])
        electron_density_from_ref_upper = electron_density
        
        #####################################################################################################################
        #####################################################################################################################
        #####################################################################################################################

        #lower electron density 
        rho_total_lower = np.full((ne_line_total.shape) , np.nan)
        ne_line_total_lower = np.full((ne_line_total.shape) , np.nan)
        electron_density_ne_lower = np.full(electron_density_ne.shape, np.nan)
        rho_pol_norm_base_min_lower = np.full((rho_pol_norm_base_min.shape) , np.nan)

        electron_density_ne_lower = np.ma.array(electron_density_ne, mask = ~mask_lower_LOS, fill_value=np.nan)
        rho_pol_norm_base_min_lower = np.ma.array(rho_pol_norm_base_min, mask = ~mask_lower_LOS, fill_value=np.nan)

        electron_density_ne_lower = electron_density_ne_lower.filled(np.nan)
        rho_pol_norm_base_min_lower = rho_pol_norm_base_min_lower.filled(np.nan)

        
        #prepare for concatenation :
        for jj in range(rho_total_lower.shape[1]):
            ne_line_total_lower[:,jj] = np.concatenate((electron_density_ne_lower[:,jj],integrale_density_ref[:,jj]))
            rho_total_lower[:,jj] = np.concatenate((rho_pol_norm_base_min_lower[:,jj],rho_pol_norm_ref[:,jj]))

        #prepare for sorting arrays :
        index_sort_lower = np.argsort(rho_total_lower, axis=0)
        rho_total_sort_lower = (np.array(list(map(lambda x, y: y[x], index_sort_lower.T, rho_total_lower.T)))).T
        ne_line_total_sort_lower = np.array(list(map(lambda x, y: y[x], index_sort_lower.T, ne_line_total_lower.T))).T
                
        ne_line_total_errors_lower = np.full(ne_line_total_sort_lower.shape, (ne_line_total_sort_lower)*0.05)
        rho_total_errors_lower =  np.full(rho_total_sort_lower.shape, (rho_total_sort_lower)*0.01)
    
        print('------------------------------')
        print('------------------------------')
        print('----fit_data for the lower----')
        print('------------------------------')
        print('------------------------------')


        out_put_lower = fit_data(rho_total_sort_lower.T, (ne_line_total_sort_lower).T, rho_total_errors_lower.T, ne_line_total_errors_lower.T, kernel_method=args.kernel, \
                          optimise_all_params=True, slices_nbr=10, plot_fit=False, x_fix_data=None, dy_fix_data=None, dy_fix_err=None)

        ne_line_density_fit_lower = (np.asarray(out_put_lower['fit_y']))
        rho_total_fit_x_lower =  (np.asarray(out_put_lower['fit_x']))
  

        #add the time slices as an output for the data_fit function
        
        time_slices_red_lower = out_put_lower['fit_time_slice']
        mask_diff_lower = []
        ne_line_interpolated_R_lower = []
        derivative_interp_array_lower = []

        for ii in range(len(time_slices_red_lower)):
            mask_diff_lower.append((np.diff(rho_mid_plane[time_slices_red_lower[ii],:])>0))
            mask_rho_fit_lower= ((rho_mid_plane[time_slices_red_lower[ii],:]>rho_total_fit_x_lower[ii].min()) & (rho_mid_plane[time_slices_red_lower[ii],:]<rho_total_fit_x_lower[ii].max()))
        
        mask_diff_lower = np.asarray(mask_diff_lower)
        mask_diff_lower = np.insert(mask_diff_lower, False, 0, axis=1)
        mask_total_lower = mask_rho_fit_lower & mask_diff_lower

        for ii in range(len(time_slices_red_lower)):
            ne_line_interpolated_R_lower.append(np.interp(rho_mid_plane[time_slices_red_lower[ii],mask_total_lower[ii]], rho_total_fit_x_lower[ii,:], ne_line_density_fit_lower[ii,:]))
        
        ne_line_interpolated_R_lower = np.asarray(ne_line_interpolated_R_lower)

        ######################################################################################################
        #derivative of the interpolated density over space for the lower 
        for ii in range(len(time_slices_red_lower)):
            derivative_interp_array_lower.append(np.gradient(ne_line_interpolated_R_lower[ii], R_meters[mask_total_lower[ii]]))
        
        derivative_interp_array_lower = np.asarray(derivative_interp_array_lower)
        ######################################################################################################

        R_meters_mask_lower = np.full((len(time_slices_red_lower), 100),np.nan)
        ne_line_interpolated_R_2d_lower = np.full(R_meters_mask_lower.shape, np.nan)
        ne_derivative_interpolated_2d_lower = np.full(R_meters_mask_lower.shape, np.nan)

        for ii in range(len(time_slices_red_lower)):
            R_meters_mask_lower[ii]=np.linspace(np.nanmin(R_meters[mask_total_lower[ii]]), np.nanmax(R_meters[mask_total_lower[ii]]), 100)
            ne_line_interpolated_R_2d_lower[ii] = np.interp(R_meters_mask_lower[ii], (R_meters[mask_total_lower[ii]]) ,ne_line_interpolated_R_lower[ii])
            ne_derivative_interpolated_2d_lower[ii] = np.interp(R_meters_mask_lower[ii], (R_meters[mask_total_lower[ii]]) ,derivative_interp_array_lower[ii])

        R_meters_mask_lower = R_meters_mask_lower.T
        ne_line_interpolated_R_2d_lower = ne_line_interpolated_R_2d_lower.T
        ne_line_interpolated_R_2d_errors_lower = np.full(ne_line_interpolated_R_2d_lower.shape, np.mean(ne_line_interpolated_R_2d_lower)*0.091)
        R_meters_2d_errors_lower =  np.full(R_meters_mask_lower.shape, np.mean(R_meters_mask_lower)*0.001)


        print('------------------------------------------')
        print('------------------------------------------')
        print('----fit_data for the lower in R_meters----')
        print('------------------------------------------')
        print('------------------------------------------')
        
        out_put_R_lower12 = fit_data(R_meters_mask_lower.T, (ne_line_interpolated_R_2d_lower).T, R_meters_2d_errors_lower.T, ne_line_interpolated_R_2d_errors_lower.T, kernel_method=args.kernel, \
                          optimise_all_params=True, slices_nbr=10, plot_fit=False,x_fix_data=None, dy_fix_data=None, dy_fix_err=None)

        # output the data necessary for the boundary conditions
        lower_maximum_R    = (np.asarray(out_put_R_lower12['fit_x'])).max(axis=1)
        lower_minimum_R    = (np.asarray(out_put_R_lower12['fit_x'])).min(axis=1)
        lower_derivative_R = (np.asarray(out_put_R_lower12['fit_dydx'])).min(axis=1)#take the max

        
        out_put_R_lower = fit_data(R_meters_mask_lower.T, (ne_line_interpolated_R_2d_lower).T, R_meters_2d_errors_lower.T, ne_line_interpolated_R_2d_errors_lower.T, kernel_method=args.kernel, \
                          optimise_all_params=True, slices_nbr=10, plot_fit=False,x_fix_data=None, dy_fix_data=None, dy_fix_err=None, boundary_max=lower_maximum_R, boundary_min=lower_minimum_R, boundary_derv=lower_derivative_R)

        print('-----------------------------------------------------------')
        print('-----------------------------------------------------------')
        print('----Transformation from R_space to rho_space for lower ----')
        print('----Transformation from R_space to rho_space for upper ----')
        print('-----------------------------------------------------------')
        print('-----------------------------------------------------------')
       


        time_slices_real_lower = np.asarray(TimeReference[time_slices_red_lower])
        time_slices_real_upper = np.asarray(TimeReference[time_slices_red_upper])


        rho_mid_plane_lower = np.full((R_meters_mask_lower.shape[1],R_meters_mask_lower.shape[0] ), np.nan)
        rho_mid_plane_upper = np.full((R_meters_mask_upper.shape[1],R_meters_mask_upper.shape[0]), np.nan)
        Phi_meters_trans = np.zeros(100)
        Z_meters_trans = np.zeros(100)

        #loop over time in the time array
        for ii in range(R_meters_mask_lower.shape[1]):
            rho_mid_plane_lower[ii] = equimap.get(shot,time_slices_real_lower[ii] , R_meters_mask_lower[:,ii], Phi_meters_trans, Z_meters_trans, 'rho_pol_norm')

        for ii in range(R_meters_mask_upper.shape[1]):
            rho_mid_plane_upper[ii] = equimap.get(shot,time_slices_real_upper[ii] , R_meters_mask_upper[:,ii], Phi_meters_trans, Z_meters_trans, 'rho_pol_norm')




        mask_lower_rho =  rho_mid_plane_lower<np.nanmax(rho_pol_norm_base_min_lower)
        rho_mid_plane_lower = np.ma.array(rho_mid_plane_lower, mask = ~mask_lower_rho, fill_value=np.nan)
        rho_mid_plane_lower_masked = rho_mid_plane_lower.filled(np.nan)

        mask_upper_rho = rho_mid_plane_upper<np.nanmax(rho_pol_norm_base_min_upper)
        rho_mid_plane_upper = np.ma.array(rho_mid_plane_upper, mask = ~mask_upper_rho, fill_value=np.nan)
        rho_mid_plane_upper_masked = rho_mid_plane_upper.filled(np.nan)



        derivative_density_upper = -(np.asarray(out_put_R_upper['fit_dydx']))
        derivative_density_lower = -(np.asarray(out_put_R_lower['fit_dydx']))
        electron_density_der_upper = np.full(rho_mid_plane_upper_masked.shape, np.nan)
        electron_density_der_lower = np.full(rho_mid_plane_lower_masked.shape, np.nan)


        #Check which on of the data sets have more nans and interpolate the one having more into the other in rho 
        if(np.isnan(rho_mid_plane_upper_masked).sum()<np.isnan(rho_mid_plane_lower_masked).sum()):
            for ii in range(electron_density_der_lower.shape[0]):
                electron_density_der_lower[ii]= np.interp(rho_mid_plane_upper_masked[ii,:], rho_mid_plane_lower_masked[ii,:], derivative_density_lower[ii], left=np.nan, right=np.nan)
            rho_total_final = rho_mid_plane_upper_masked
            average_der_density = np.nanmean((electron_density_der_lower, derivative_density_upper),axis=0)

        else:
            for ii in range(electron_density_der_upper.shape[0]):
                electron_density_der_upper[ii]= np.interp(rho_mid_plane_lower_masked[ii,:], rho_mid_plane_upper_masked[ii,:] ,derivative_density_upper[ii], left=np.nan, right=np.nan)
            rho_total_final = rho_mid_plane_lower_masked
            #Calculate the average density of the upper and lower data samples 
            average_der_density = np.nanmean((derivative_density_lower, electron_density_der_upper),axis=0)
            


        #Interpolate the electron density to the time slices that we are working in
        #Interpolate the rho_pol_norm_ref that is associated to the electron density to the same time slices
        electron_density_interpolated=[]
        rho_pol_norm_ref_interpolated = []
        for ii in range(electron_density.shape[0]):
            rho_pol_norm_ref_interpolated.append(np.interp(time_slices_real_lower, TimeReference, rho_pol_norm_ref[ii,:]))
            electron_density_interpolated.append(np.interp(time_slices_real_lower, TimeReference, electron_density[ii,:]))
        electron_density_interpolated = (np.asarray(electron_density_interpolated)).T # transpose from (space,time) to (time, space)  shape
        rho_pol_norm_ref_interpolated = (np.asarray(rho_pol_norm_ref_interpolated)).T # transpose from (space,time) to (time, space)  shape


        #concatenate the rho_pol_norm_ref_interpolated with rho_mid_plane_upper_masked or rho_mid_plane_lower_masked
        rho_pol_norm_ref_concat = (np.concatenate((rho_total_final,rho_pol_norm_ref_interpolated), axis=1))#concatenate along the second axis ======> in space
        electron_density_concat = (np.concatenate((average_der_density, electron_density_interpolated), axis=1))#concatenate along the second axis ======> in space
       

        #prepare for sorting arrays :

        array_index = (np.argsort(rho_pol_norm_ref_concat, axis=1))
        rho_total_sort_final = np.asarray(list(map(lambda x, y: y[x], array_index, rho_pol_norm_ref_concat)))
        ne_line_total_sort_final = np.asarray(list(map(lambda x, y: y[x], array_index, electron_density_concat)))
        
        #check for nans in density and mask over the values that corresponds to the nans in the rho
        if np.isnan(ne_line_total_sort_final).any(): 
            mask_rho_total_sort_final = np.ma.array(rho_total_sort_final, mask = np.isnan(ne_line_total_sort_final), fill_value=np.nan)
            rho_total_sort_final = mask_rho_total_sort_final.filled(np.nan)

        #check for nans in rho and mask over the values that corresponds to the nans in the ne_profile
        if np.isnan(rho_total_sort_final).any(): 
            mask_ne_line_total_sort = np.ma.array(ne_line_total_sort_final, mask = np.isnan(rho_total_sort_final), fill_value=np.nan)
            ne_line_total_sort_final = mask_ne_line_total_sort.filled(np.nan)

        #add an 2 extra point to rho total
        rho_total_sort_final = np.insert(rho_total_sort_final, 0, 0.02, axis=1)#index, value
        rho_total_sort_final = np.insert(rho_total_sort_final, 0, 0.01, axis=1)
        #the maximum should be by time slice and should 
        maximum_elements_array = []
        for ii in range(ne_line_total_sort_final.shape[0]):
            maximum_elements_array.append(np.nanmax(ne_line_total_sort_final[ii]))
        maximum_elements_array = np.asarray(maximum_elements_array)
                                          
        #concatenate the array of the first elements with the total density elements
        ne_line_total_sort_final = np.concatenate((maximum_elements_array[:,None], ne_line_total_sort_final),axis=1)
        ne_line_total_sort_final = np.concatenate((maximum_elements_array[:,None], ne_line_total_sort_final),axis=1)
        
        rho_total_sort_final_error = np.full(rho_total_sort_final.shape,(rho_total_sort_final)*0.001)
        ne_line_total_sort_final_error = np.full(ne_line_total_sort_final.shape, (ne_line_total_sort_final)*0.01)

        out_put_final12 = fit_data(rho_total_sort_final , ne_line_total_sort_final , rho_total_sort_final_error , ne_line_total_sort_final_error , kernel_method='Gibbs_Kernel', \
                                       optimise_all_params=True, slices_nbr=10, plot_fit=True, x_fix_data=None, dy_fix_data=None, dy_fix_err=None)



        # output the data necessary for the boundary conditions
        final_maximum_R    = (np.asarray(out_put_final12['fit_x'])).max(axis=1)
        final_minimum_R    = (np.asarray(out_put_final12['fit_x'])).min(axis=1)
        final_derivative_R = (np.asarray(out_put_final12['fit_dydx'])).min(axis=1)


        out_put_final = fit_data(rho_total_sort_final , ne_line_total_sort_final , rho_total_sort_final_error , ne_line_total_sort_final_error , kernel_method='Gibbs_Kernel', \
                                       optimise_all_params=True, slices_nbr=10, plot_fit=True, x_fix_data=None, dy_fix_data=None, dy_fix_err=None)#, boundary_max=final_maximum_R, boundary_min=final_minimum_R, boundary_derv=final_derivative_R)

        ne_density_fit = (np.asarray(out_put_final['fit_y']))
        rho_total_fit =  (np.asarray(out_put_final['fit_x']))
        Time_index = np.asarray(out_put_final['fit_time_slice'])

        #interpolate rho_pol_norm_base along time and space to rho_total_fit
        
        density_pol_norm_base_interp = np.full((rho_pol_norm_base.shape[0],rho_total_fit.shape[0],rho_pol_norm_base.shape[2]),np.nan)
        
        for ii in range(density_pol_norm_base_interp.shape[0]):
            for jj in range(density_pol_norm_base_interp.shape[1]):
                density_pol_norm_base_interp[ii,jj] = np.interp(rho_pol_norm_base[ii, Time_index[jj]], rho_total_fit[jj],  ne_density_fit[jj], left=0, right=0)
        

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

        integrale_density_final = np.full((density_pol_norm_base_interp.shape[0],density_pol_norm_base_interp.shape[1]),np.nan)
        density_pol_norm_base_interp[np.isnan(density_pol_norm_base_interp)]=0

        for ii in range(density_pol_norm_base_interp.shape[0]):
            for jj in range(density_pol_norm_base_interp.shape[1]):
                integrale_density_final[ii, jj] = (integrate.trapz(density_pol_norm_base_interp[ii, jj],distance_length[ii]))
        #import ipdb; ipdb.set_trace()









































        #try the comparison between the interferometer data befor and after integration 
        #prepare for sorting arrays :

        array_index = (np.argsort(rho_total_final, axis=1))
        rho_total_sort_final1 = np.asarray(list(map(lambda x, y: y[x], array_index, rho_total_final)))
        ne_line_total_sort_final1 = np.asarray(list(map(lambda x, y: y[x], array_index, average_der_density)))
        
        #check for nans in density and mask over the values that corresponds to the nans in the rho
        if np.isnan(ne_line_total_sort_final1).any(): 
            mask_rho_total_sort_final1 = np.ma.array(rho_total_sort_final1, mask = np.isnan(ne_line_total_sort_final1), fill_value=np.nan)
            rho_total_sort_final1 = mask_rho_total_sort_final1.filled(np.nan)

        #check for nans in rho and mask over the values that corresponds to the nans in the ne_profile
        if np.isnan(rho_total_sort_final1).any(): 
            mask_ne_line_total_sort1 = np.ma.array(ne_line_total_sort_final1, mask = np.isnan(rho_total_sort_final1), fill_value=np.nan)
            ne_line_total_sort_final1 = mask_ne_line_total_sort1.filled(np.nan)

        #add an 2 extra point to rho total
        rho_total_sort_final1 = np.insert(rho_total_sort_final1, 0, 0.02, axis=1)#index, value
        rho_total_sort_final1 = np.insert(rho_total_sort_final1, 0, 0.01, axis=1)
        #the maximum should be by time slice and should 
        maximum_elements_array1 = []
        for ii in range(ne_line_total_sort_final1.shape[0]):
            maximum_elements_array1.append(np.nanmax(ne_line_total_sort_final1[ii]))
        maximum_elements_array1 = np.asarray(maximum_elements_array1)
                                          
        #concatenate the array of the first elements with the total density elements
        ne_line_total_sort_final1 = np.concatenate((maximum_elements_array[:,None], ne_line_total_sort_final1),axis=1)
        ne_line_total_sort_final1 = np.concatenate((maximum_elements_array[:,None], ne_line_total_sort_final1),axis=1)
        
        rho_total_sort_final1_error = np.full(rho_total_sort_final1.shape,(rho_total_sort_final1)*0.001)
        ne_line_total_sort_final1_error = np.full(ne_line_total_sort_final1.shape, (ne_line_total_sort_final1)*0.01)


        out_put_final1 = fit_data(rho_total_sort_final1 , ne_line_total_sort_final1 , rho_total_sort_final1_error , ne_line_total_sort_final1_error , kernel_method='Gibbs_Kernel', \
                                       optimise_all_params=True, slices_nbr=10, plot_fit=False, x_fix_data=None, dy_fix_data=None, dy_fix_err=None)


        ne_density_fit1 = (np.asarray(out_put_final1['fit_y']))
        rho_total_fit1 =  (np.asarray(out_put_final1['fit_x']))
        Time_index1 = np.asarray(out_put_final1['fit_time_slice'])

        #interpolate rho_pol_norm_base along time and space to rho_total_fit
        
        density_pol_norm_base_interp1 = np.full((rho_pol_norm_base.shape[0],rho_total_fit1.shape[0],rho_pol_norm_base.shape[2]),np.nan)
        
        for ii in range(density_pol_norm_base_interp1.shape[0]):
            for jj in range(density_pol_norm_base_interp1.shape[1]):
                density_pol_norm_base_interp1[ii,jj] = np.interp(rho_pol_norm_base[ii, Time_index1[jj]], rho_total_fit1[jj],  ne_density_fit1[jj], left=0, right=0)
        

        rho_pol_norm_base_sample1 = rho_pol_norm_base[:, Time_index1, :] #will have the size of (line of sight, Time_index, space)

        R_01 = np.full((R.shape[0]), np.nan)
        Z_01 = np.full((R.shape[0]), np.nan)
        distance_length1 = np.full((R.shape), np.nan)

        #create a loop over the line of sight 
        for ii in range(R.shape[0]):
            R_01[ii] = R[ii,0]
            Z_01[ii] = Z[ii,0]
            distance_length1[ii] = np.sqrt((R[ii]-R_01[ii])**2 + (Z[ii]-Z_01[ii])**2)

        integrale_density_final1 = np.full((density_pol_norm_base_interp1.shape[0],density_pol_norm_base_interp1.shape[1]),np.nan)
        density_pol_norm_base_interp1[np.isnan(density_pol_norm_base_interp1)]=0

        for ii in range(density_pol_norm_base_interp1.shape[0]):
            for jj in range(density_pol_norm_base_interp1.shape[1]):
                integrale_density_final1[ii, jj] = (integrate.trapz(density_pol_norm_base_interp1[ii, jj],distance_length1[ii]))
        #import ipdb; ipdb.set_trace()
















        density4 = electron_density_ne/Normalization_constant
        ###Some basic setup
        plot_save_directory = './comparison_figuers'
        if not plot_save_directory.endswith('/'):
            plot_save_directory = plot_save_directory+'/'
        if not os.path.isdir(plot_save_directory):
            os.makedirs(plot_save_directory)
            
        for ii in range(len(Time_index)):
            fig = plt.figure()
            fig.suptitle((('inter verses integrated  data')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
            ax = fig.add_subplot(111)
            ax.plot(integrale_density_final[:,Time_index[ii]] , color='r', label = 'ne from integration and fit procedure')
            ax.plot(2*density4[:,Time_index][:,Time_index[ii]] , color='g', label = 'ne from interferometery without normalisation')
            ax.plot(2*electron_density_ne[:,Time_index][:,Time_index[ii]] , color='b', label = 'ne from inter using normalisaton')
            ax.plot(integrale_density_final1[:,Time_index1[ii]] , color='k', label = 'ne from inter without concatination')

            plt.legend()
            fig.savefig(plot_save_directory + 'time_slice' + str(Time_index[ii]) +'.png')
            plt.close(fig)
        print("Results of demonstration plotted in directory ./comparison_figuers/\n")

        

        
 
        ###Some basic setup
        plot_save_directory = './upper_figuers'
        if not plot_save_directory.endswith('/'):
            plot_save_directory = plot_save_directory+'/'
        if not os.path.isdir(plot_save_directory):
            os.makedirs(plot_save_directory)
            
        for ii in range(len(time_slices_red_upper)):
            fig = plt.figure()
            fig.suptitle((('Raw data_upper')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
            ax = fig.add_subplot(111)
            ax.plot(R_real_ref[:,time_slices_red_upper[ii]],electron_density[:,time_slices_red_upper[ii]] , color='g', label = 'ne vers r from reflec real')
            ax.plot(R_meters_mask_upper[:,time_slices_red_upper[ii]],-(np.asarray(out_put_R_upper['fit_dydx']))[time_slices_red_upper[ii]] , color='r', label = 'deriv ne vers r fit_data_upper')
            ax.plot(R_meters_mask_upper[:,time_slices_red_upper[ii]],average_der_density[time_slices_red_upper[ii]] , color='k', label = 'ne vers r from reflec real')
            plt.legend()
            fig.savefig(plot_save_directory + 'time_slice_upper' + str(time_slices_red_lower[ii]) +'.png')
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
            ax.plot(R_real_ref[:,time_slices_red_lower[ii]],electron_density[:,time_slices_red_lower[ii]] , color='g', label = 'ne vers r from reflec real')
            ax.plot(R_meters_mask_lower[:,time_slices_red_lower[ii]],-(np.asarray(out_put_R_lower['fit_dydx']))[time_slices_red_lower[ii]] , color='r', label = 'deriv ne vers r fit_data_lower')
            ax.plot(R_meters_mask_lower[:,time_slices_red_lower[ii]],average_der_density[time_slices_red_lower[ii]] , color='k', label = 'deriv ne vers r fit_data_lower')
            plt.legend()
            fig.savefig(plot_save_directory+ 'time_slice_upper' + str(time_slices_red_lower[ii]) +'.png')
            plt.close(fig)

        print("Results of demonstration plotted in directory ./lower_figuers/\n")

        #import ipdb; ipdb.set_trace()


        plot_save_directory = './comparison_upper_average_lower_figuers'
        if not plot_save_directory.endswith('/'):
            plot_save_directory = plot_save_directory+'/'
        if not os.path.isdir(plot_save_directory):
            os.makedirs(plot_save_directory)
   
        for ii in range(len(time_slices_red_lower)):
            fig = plt.figure()
            fig.suptitle((('Raw data_lower average, upper')), fontdict={'fontsize': 5, 'fontweight': 'medium'})
            ax = fig.add_subplot(111)
            ax.plot(R_real_ref[:,time_slices_red_lower[ii]],electron_density[:,time_slices_red_lower[ii]] , color='g', label = 'ne vers r from reflec real')
            ax.plot(R_meters_mask_upper[:,time_slices_red_lower[ii]],-(np.asarray(out_put_R_upper['fit_dydx']))[time_slices_red_lower[ii]] , color='r', label = 'deriv ne vers r fit_data_upper')
            ax.plot(R_meters_mask_lower[:,time_slices_red_lower[ii]],-(np.asarray(out_put_R_lower['fit_dydx']))[time_slices_red_lower[ii]] , color='b', label = 'deriv ne vers r fit_data_lower')
            ax.plot(R_meters_mask_lower[:,time_slices_red_lower[ii]],average_der_density[time_slices_red_lower[ii]] , color='k', label = 'average fit_data_lower')
            plt.legend()
            fig.savefig(plot_save_directory+ 'time_slice' + str(time_slices_red_lower[ii]) +'.png')
            plt.close(fig)

        print("Results of demonstration plotted in directory ./comparison_upper_average_lower_figuers/\n")


        

        import ipdb; ipdb.set_trace()
        return rho_total_sort_upper.T, ne_line_total_sort_upper.T, rho_total_errors_upper.T, ne_line_total_errors_upper.T


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
    #out = fit_data(x, y, ex , ey, kernel_method=args.kernel, \
                       #optimise_all_params=True, slices_nbr=10, plot_fit=True)
    #import pdb; pdb.set_trace()

