# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab

from __future__ import (unicode_literals, absolute_import,print_function, division)
# Standard python modules
import numpy as np
import os
import re
import sys
import warnings
import random
import itertools

# Project modules
try:
    from .GPR1D import GPR1D
except Exception as err:
    from GPR1D import GPR1D

'''
    GPR1D fit data function
'''

__author__ = ['Mohamad Kozeiha', 'Jorge Morales']
__date__ = '10/09/2019'
__version__ = '$Revision: 1.10 $'

__all__ = ('default_config')

default_configuartion = {
    'RQ_Kernel'  : GPR1D.RQ_Kernel(),
    'Gibbs_Kernel'  : GPR1D.Gibbs_Kernel()
    }


def Optimization(X_coordinates, Y_coordinates, X_coordinates_errors, Y_coordinates_errors, \
                 kernel_method='RQ_Kernel', slices_optim_nbr=10, \
                 dx_data=[0.0],dy_data=[0.0],dy_err=[0.0], nbr_pts=100):
    print('Please be reminded that you are in the new optimization code')
    if Y_coordinates.shape[0]<slices_optim_nbr:
        slices_optim_nbr = Y_coordinates.shape[0]
    print('slices_optim_nbr = ', slices_optim_nbr)
    if(kernel_method == 'Gibbs_Kernel'):
        optimized_config = {
            'hsgp_fit_regpar_optimized'    : {'regularaiztion'  : [],
                                              'amp' : [],
                                              'alpha' : []                                    
                                              },
            'hsgp_error_fit_regpar_optimized'      : {'regularaiztion'  : [],
                                                      'amp' : [],
                                                      'alpha' : []                                    
                                                      },
            'nigp_fit_regpar_optimized'      : {'regularaiztion'        : [],
                                                'amp'       : [],
                                                'alpha'       : []                                    
                                                },
            'nigp_error_fit_regpar_optimized'      : {'regularaiztion'  : [],
                                                      'amp' : [],
                                                      'alpha' : []                                    
                                                      }
            }
        optimized_values = {
            'hsgp_fit_regpar_optimized'    : {'regularaiztion'  : 0,
                                              'amp' : 0,
                                              'alpha' : 0                                    
                                              },
            'hsgp_error_fit_regpar_optimized'      : {'regularaiztion'  : 0,
                                                      'amp' : 0,
                                                      'alpha' : 0                                    
                                                      },
            'nigp_fit_regpar_optimized'      : {'regularaiztion'        : 0,
                                                'amp'       : 0,
                                                'alpha'       : 0                                    
                                                },
            'nigp_error_fit_regpar_optimized'      : {'regularaiztion'  : 0,
                                                      'amp' : 0,
                                                      'alpha' : 0                                    
                                                      }
            }
        
        
    else:
        optimized_config = {
            'hsgp_fit_regpar_optimized'    : {'regularaiztion'  : [],
                                              'amp' : [],
                                              'ls'   : [],
                                              'alpha' : []                                    
                                              },
            'nigp_fit_regpar_optimized'      : {'regularaiztion'        : [],
                                                'amp'       : [],
                                                'ls'         : [],
                                                'alpha'       : []                                    
                                                },
            'hsgp_error_fit_regpar_optimized'      : {'regularaiztion'  : [],
                                                      'amp' : [],
                                                      'ls'   : [],
                                                      'alpha' : []                                    
                                                      },
            'nigp_error_fit_regpar_optimized'      : {'regularaiztion'  : [],
                                                      'amp' : [],
                                                      'ls'   : [],
                                                      'alpha' : []                                    
                                                      }
            }
        optimized_values = {
            'hsgp_fit_regpar_optimized'    : {'regularaiztion'  : 0,
                                              'amp' : 0,
                                              'ls'   : 0,
                                              'alpha' : 0                                    
                                              },
            'nigp_fit_regpar_optimized'      : {'regularaiztion'        : 0,
                                                'amp'       : 0,
                                                'ls'         : 0,
                                                'alpha'       : 0                                    
                                                },
            'hsgp_error_fit_regpar_optimized'      : {'regularaiztion'  : 0,
                                                      'amp' : 0,
                                                      'ls'   : 0,
                                                      'alpha' : 0                                    
                                                      },
            'nigp_error_fit_regpar_optimized'      : {'regularaiztion'  : 0,
                                                      'amp' : 0,
                                                      'ls'   : 0,
                                                      'alpha' : 0                                    
                                                      }
            }
            
        

    for i in range(0,(Y_coordinates.shape[0]) , int((Y_coordinates.shape[0])/slices_optim_nbr)):
        print(i)
        Y_reduced = Y_coordinates[i]
        X_reduced = X_coordinates[i]
        Y_errors = Y_coordinates_errors[i]
        X_errors = X_coordinates_errors[i]

        minimum = (X_reduced).min()
        maximum = (X_reduced).max()
       
        fit_x_values = np.linspace(minimum,maximum,nbr_pts)
        kernel =  default_configuartion.get(kernel_method)
        kernel_hyppar_bounds = np.atleast_2d()
        error_kernel = default_configuartion.get(kernel_method)
        error_kernel_hyppar_bounds = np.atleast_2d()


        # GPR fit rigourously accounting only for y-errors (this is the recommended option)
        hsgpr_object = GPR1D.GaussianProcessRegression1D()
        hsgpr_object.set_kernel(kernel=kernel)
        hsgpr_object.set_error_kernel(kernel=error_kernel)
        hsgpr_object.set_raw_data(xdata=X_reduced,ydata=Y_reduced,yerr=Y_errors,xerr=X_errors, dxdata=dx_data,dydata=dy_data,dyerr=dy_err)     
        hsgpr_object.set_search_parameters(epsilon=1.0e-2)
        hsgpr_object.set_error_search_parameters(epsilon=1.0e-1)
        #     Perform the fit with kernel restarts
        hsgpr_object.GPRFit(fit_x_values,hsgp_flag=True)#,nrestarts=3)
        (hsgp_kernel_name,hsgp_kernel_hyppars,hsgp_fit_regpar) = hsgpr_object.get_gp_kernel_details()
        (hsgp_error_kernel_name,hsgp_error_kernel_hyppars,hsgp_error_fit_regpar) = hsgpr_object.get_gp_error_kernel_details()


        # GPR fit rigourously accounting for y-errors AND x-errors
        nigpr_object = GPR1D.GaussianProcessRegression1D()
        nigpr_object.set_kernel(kernel=kernel)
        nigpr_object.set_error_kernel(kernel=error_kernel)
        nigpr_object.set_raw_data(xdata=X_reduced,ydata=Y_reduced,yerr=Y_errors,xerr=X_errors, dxdata=dx_data,dydata=dy_data,dyerr=dy_err)
        nigpr_object.set_search_parameters(epsilon=1.0e-2)
        nigpr_object.set_error_search_parameters(epsilon=1.0e-1)
        #     Perform the fit with kernel restarts, here is the extra option to account for x-errors in fit
        nigpr_object.GPRFit(fit_x_values,hsgp_flag=False,nigp_flag=True,nrestarts=3)
        (nigp_kernel_name,nigp_kernel_hyppars,nigp_fit_regpar) = nigpr_object.get_gp_kernel_details()
        (nigp_error_kernel_name,nigp_error_kernel_hyppars,nigp_error_fit_regpar) = nigpr_object.get_gp_error_kernel_details()


        if (kernel_method == 'Gibbs_Kernel'):
            optimized_config['hsgp_fit_regpar_optimized']['regularaiztion'].append(hsgp_fit_regpar)
            optimized_config['hsgp_fit_regpar_optimized']['amp'].append(hsgp_kernel_hyppars[0])
            optimized_config['hsgp_fit_regpar_optimized']['alpha'].append(hsgp_kernel_hyppars[1])
            optimized_config['hsgp_error_fit_regpar_optimized']['regularaiztion'].append(hsgp_fit_regpar)
            optimized_config['hsgp_error_fit_regpar_optimized']['amp'].append(hsgp_kernel_hyppars[0])
            optimized_config['hsgp_error_fit_regpar_optimized']['alpha'].append(hsgp_kernel_hyppars[1])
            optimized_config['nigp_fit_regpar_optimized']['regularaiztion'].append(nigp_fit_regpar)
            optimized_config['nigp_fit_regpar_optimized']['amp'].append(nigp_kernel_hyppars[0])
            optimized_config['nigp_fit_regpar_optimized']['alpha'].append(nigp_kernel_hyppars[1])
            optimized_config['nigp_error_fit_regpar_optimized']['regularaiztion'].append(nigp_fit_regpar)
            optimized_config['nigp_error_fit_regpar_optimized']['amp'].append(nigp_kernel_hyppars[0])
            optimized_config['nigp_error_fit_regpar_optimized']['alpha'].append(nigp_kernel_hyppars[1])
        else:
            optimized_config['hsgp_fit_regpar_optimized']['regularaiztion'].append(hsgp_fit_regpar)
            optimized_config['hsgp_fit_regpar_optimized']['amp'].append(hsgp_kernel_hyppars[0])
            optimized_config['hsgp_fit_regpar_optimized']['ls'].append(hsgp_kernel_hyppars[1])
            optimized_config['hsgp_fit_regpar_optimized']['alpha'].append(hsgp_kernel_hyppars[2])
            optimized_config['hsgp_error_fit_regpar_optimized']['regularaiztion'].append(hsgp_fit_regpar)
            optimized_config['hsgp_error_fit_regpar_optimized']['amp'].append(hsgp_kernel_hyppars[0])
            optimized_config['hsgp_error_fit_regpar_optimized']['ls'].append(hsgp_kernel_hyppars[1])
            optimized_config['hsgp_error_fit_regpar_optimized']['alpha'].append(hsgp_kernel_hyppars[2])
            optimized_config['nigp_fit_regpar_optimized']['regularaiztion'].append(nigp_fit_regpar)
            optimized_config['nigp_fit_regpar_optimized']['amp'].append(nigp_kernel_hyppars[0])
            optimized_config['nigp_fit_regpar_optimized']['ls'].append(nigp_kernel_hyppars[1])
            optimized_config['nigp_fit_regpar_optimized']['alpha'].append(nigp_kernel_hyppars[2])
            optimized_config['nigp_error_fit_regpar_optimized']['regularaiztion'].append(nigp_fit_regpar)
            optimized_config['nigp_error_fit_regpar_optimized']['amp'].append(nigp_kernel_hyppars[0])
            optimized_config['nigp_error_fit_regpar_optimized']['ls'].append(nigp_kernel_hyppars[1])
            optimized_config['nigp_error_fit_regpar_optimized']['alpha'].append(nigp_kernel_hyppars[2])


    if (kernel_method == 'Gibbs_Kernel'):
        optimized_values['hsgp_fit_regpar_optimized']['regularaiztion']       = np.nanmedian(optimized_config['hsgp_fit_regpar_optimized']['regularaiztion'])
        optimized_values['hsgp_fit_regpar_optimized']['amp']                  = np.nanmedian(optimized_config['hsgp_fit_regpar_optimized']['amp'])
        optimized_values['hsgp_fit_regpar_optimized']['alpha']                = np.nanmedian(optimized_config['hsgp_fit_regpar_optimized']['alpha'])
        optimized_values['hsgp_error_fit_regpar_optimized']['regularaiztion'] = np.nanmedian(optimized_config['hsgp_error_fit_regpar_optimized']['regularaiztion'])
        optimized_values['hsgp_error_fit_regpar_optimized']['amp']            = np.nanmedian(optimized_config['hsgp_error_fit_regpar_optimized']['amp'])
        optimized_values['hsgp_error_fit_regpar_optimized']['alpha']          = np.nanmedian(optimized_config['hsgp_error_fit_regpar_optimized']['alpha'])
        optimized_values['nigp_fit_regpar_optimized']['regularaiztion']       = np.nanmedian(optimized_config['nigp_fit_regpar_optimized']['regularaiztion'])
        optimized_values['nigp_fit_regpar_optimized']['amp']                  = np.nanmedian(optimized_config['nigp_fit_regpar_optimized']['amp'])
        optimized_values['nigp_fit_regpar_optimized']['alpha']                = np.nanmedian(optimized_config['nigp_fit_regpar_optimized']['alpha'])
        optimized_values['nigp_error_fit_regpar_optimized']['regularaiztion'] = np.nanmedian(optimized_config['nigp_error_fit_regpar_optimized']['regularaiztion'])
        optimized_values['nigp_error_fit_regpar_optimized']['amp']            = np.nanmedian(optimized_config['nigp_error_fit_regpar_optimized']['amp'])
        optimized_values['nigp_error_fit_regpar_optimized']['alpha']          = np.nanmedian(optimized_config['nigp_error_fit_regpar_optimized']['alpha'])
        print('###########################################################################################')
        print('###########################################################################################')
        print('###########################################################################################')
        print('################################### optimaisation is done #################################')
        print('###########################################################################################')
        print('###########################################################################################')
        return optimized_values

    else:
        optimized_values['hsgp_fit_regpar_optimized']['regularaiztion']               = np.nanmedian(optimized_config['hsgp_fit_regpar_optimized']['regularaiztion'])
        optimized_values['hsgp_fit_regpar_optimized']['amp']                          = np.nanmedian(optimized_config['hsgp_fit_regpar_optimized']['amp'])
        optimized_values['hsgp_fit_regpar_optimized']['ls']                           = np.nanmedian(optimized_config['hsgp_fit_regpar_optimized']['ls'])
        optimized_values['hsgp_fit_regpar_optimized']['alpha']                        = np.nanmedian(optimized_config['hsgp_fit_regpar_optimized']['alpha'])
        optimized_values['hsgp_error_fit_regpar_optimized']['regularaiztion']         = np.nanmedian(optimized_config['hsgp_error_fit_regpar_optimized']['regularaiztion'])
        optimized_values['hsgp_error_fit_regpar_optimized']['amp']                    = np.nanmedian(optimized_config['hsgp_error_fit_regpar_optimized']['amp'])
        optimized_values['hsgp_error_fit_regpar_optimized']['ls']                     = np.nanmedian(optimized_config['hsgp_error_fit_regpar_optimized']['ls'])
        optimized_values['hsgp_error_fit_regpar_optimized']['alpha']                  = np.nanmedian(optimized_config['hsgp_error_fit_regpar_optimized']['alpha'])
        optimized_values['nigp_fit_regpar_optimized']['regularaiztion']               = np.nanmedian(optimized_config['nigp_fit_regpar_optimized']['regularaiztion'])
        optimized_values['nigp_fit_regpar_optimized']['amp']                          = np.nanmedian(optimized_config['nigp_fit_regpar_optimized']['amp'])
        optimized_values['nigp_fit_regpar_optimized']['ls']                           = np.nanmedian(optimized_config['nigp_fit_regpar_optimized']['ls'])
        optimized_values['nigp_fit_regpar_optimized']['alpha']                        = np.nanmedian(optimized_config['nigp_fit_regpar_optimized']['alpha'])
        optimized_values['nigp_error_fit_regpar_optimized']['regularaiztion']         = np.nanmedian(optimized_config['nigp_error_fit_regpar_optimized']['regularaiztion'])
        optimized_values['nigp_error_fit_regpar_optimized']['amp']                    = np.nanmedian(optimized_config['nigp_error_fit_regpar_optimized']['amp'])
        optimized_values['nigp_error_fit_regpar_optimized']['ls']                     = np.nanmedian(optimized_config['nigp_error_fit_regpar_optimized']['ls'])
        optimized_values['nigp_error_fit_regpar_optimized']['alpha']                  = np.nanmedian(optimized_config['nigp_error_fit_regpar_optimized']['alpha'])
        print('########  #########  ##########################################################################')
        print('#      #  ##     ##  ##########################################################################')
        print('#      #  #########  ##########################################################################')
        print('#      #  ##         ################## optimaisation is done #################################')
        print('#      #  ##         ##########################################################################')
        print('########  ##         ##########################################################################')
        return optimized_values

