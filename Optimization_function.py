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
# Local modules
import GPR1D
import imas
import pywed as pw

'''
    GPR1D fit data function
'''

__author__ = ['Mohamad Kozeiha', 'Jorge Morales']
__date__ = '10/09/2018'
__version__ = '$Revision: 1.10 $'

__all__ = ('fit_data1',
           'default_config')

default_configuartion = {
    'Constant_Kernel' : GPR1D.Constant_Kernel(),
    'Noise_Kernel'  : GPR1D.Noise_Kernel(),
    'Linear_Kernel'   : GPR1D.Linear_Kernel(),
    'Poly_Order_Kernel'   : GPR1D.Poly_Order_Kernel(),
    'SE_Kernel'   : GPR1D.SE_Kernel(),
    'RQ_Kernel'  : GPR1D.RQ_Kernel(),
    'Matern_HI_Kernel'   : GPR1D.Matern_HI_Kernel(),
    'NN_Kernel'  : GPR1D.NN_Kernel(),
    'Gibbs_Kernel'  : GPR1D.Gibbs_Kernel()
    }





def Optimization(X_coordinates, Y_coordinates, loop_over = 'even' , kernel_method='RQ_Kernel' ):


    optimized_config = {
        'gp_fit_regpar_optimized'      : {'regularaiztion'  : [],
                                          'amp' : [],
                                          'ls'   : [],
                                          'alpha' : []                                    
                                          },
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
        'gp_fit_regpar_optimized'      : {'regularaiztion'  : 0,
                                          'amp' : 0,
                                          'ls'   : 0,
                                          'alpha' : 0                                    
                                          },
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
    
    
    
    
    
    
    #import secrets
    #print(secrets.choice(Y_coordinates[1]))
    #print(secrets.randbelow(len(Y_coordinates[1])))
    ###################################################################################################
    
    
    #for i in range(0,len(Y_coordinates[1]), int(len(Y_coordinates[1])/500)):
    for i in range(3):
        even = 2*i
        odd = 2*i +1 
        if loop_over == odd :
            Y_reduced = Y_coordinates[ :,odd]
            X_reduced = (X_coordinates)[:,odd]
            
        else:
            Y_reduced = Y_coordinates[ :,even]
            X_reduced = (X_coordinates)[:,even]

        Y_errors = np.full(Y_reduced.shape, np.mean(Y_reduced)*0.05)
        minimum = X_reduced.min()
        maximum = X_reduced.max()
        X_errors =  np.full(X_reduced.shape,0.0091)
        
        
        fit_x_values = np.linspace(minimum,maximum,100)
        kernel =  default_configuartion.get(kernel_method)
        kernel_hyppar_bounds = np.atleast_2d()
        error_kernel = default_configuartion.get(kernel_method)
        error_kernel_hyppar_bounds = np.atleast_2d()
        
        
        gpr_object = GPR1D.GaussianProcessRegression1D()
        gpr_object.set_kernel(kernel=kernel)
        gpr_object.set_raw_data(xdata=X_reduced,ydata=Y_reduced,yerr=Y_errors,xerr=X_errors, dxdata=[0.0],dydata=[0.0],dyerr=[0.0])  
        gpr_object.set_search_parameters(epsilon=1.0e-2)
        gpr_object.set_error_search_parameters(epsilon=1.0e-1)
        #     Default optimizer is gradient ascent / descent - extremely robust but slow
        #     Uncomment any of the following lines to test the recommended optimizers
        # gpr_object.set_search_parameters(epsilon=1.0e-2,method='adam',spars=[1.0e-1,0.4,0.8])
        # gpr_object.set_error_search_parameters(epsilon=1.0e-1,method='adam',spars=[1.0e-1,0.4,0.8])
        gpr_object.GPRFit(fit_x_values,hsgp_flag=False,nrestarts=5)
        (gp_kernel_name,gp_kernel_hyppars,gp_fit_regpar) = gpr_object.get_gp_kernel_details()
        
        optimized_config['gp_fit_regpar_optimized']['regularaiztion'].append(gp_fit_regpar)
        optimized_config['gp_fit_regpar_optimized']['amp'].append(gp_kernel_hyppars[0])
        optimized_config['gp_fit_regpar_optimized']['ls'].append(gp_kernel_hyppars[1])
        optimized_config['gp_fit_regpar_optimized']['alpha'].append(gp_kernel_hyppars[2])

        





        # GPR fit rigourously accounting only for y-errors (this is the recommended option)
        hsgpr_object = GPR1D.GaussianProcessRegression1D()
        hsgpr_object.set_kernel(kernel=kernel)
        hsgpr_object.set_error_kernel(kernel=error_kernel)
        hsgpr_object.set_raw_data(xdata=X_reduced,ydata=Y_reduced,yerr=Y_errors,xerr=X_errors, dxdata=[0.0],dydata=[0.0],dyerr=[0.0])     
        hsgpr_object.set_search_parameters(epsilon=1.0e-2)
        hsgpr_object.set_error_search_parameters(epsilon=1.0e-1)
        #     Default optimizer is gradient ascent / descent - extremely robust but slow
        #     Uncomment any of the following lines to test the recommended optimizers
        #hsgpr_object.set_search_parameters(epsilon=1.0e-2,method='adam',spars=[1.0e-1,0.4,0.8])
        #hsgpr_object.set_error_search_parameters(epsilon=1.0e-1,method='adam',spars=[1.0e-1,0.4,0.8])
        #     Perform the fit with kernel restarts
        hsgpr_object.GPRFit(fit_x_values,hsgp_flag=True,nrestarts=5)
        (hsgp_kernel_name,hsgp_kernel_hyppars,hsgp_fit_regpar) = hsgpr_object.get_gp_kernel_details()
        (hsgp_error_kernel_name,hsgp_error_kernel_hyppars,hsgp_error_fit_regpar) = hsgpr_object.get_gp_error_kernel_details()


        optimized_config['hsgp_fit_regpar_optimized']['regularaiztion'].append(hsgp_fit_regpar)
        optimized_config['hsgp_fit_regpar_optimized']['amp'].append(hsgp_kernel_hyppars[0])
        optimized_config['hsgp_fit_regpar_optimized']['ls'].append(hsgp_kernel_hyppars[1])
        optimized_config['hsgp_fit_regpar_optimized']['alpha'].append(hsgp_kernel_hyppars[2])
        optimized_config['hsgp_error_fit_regpar_optimized']['regularaiztion'].append(hsgp_fit_regpar)
        optimized_config['hsgp_error_fit_regpar_optimized']['amp'].append(hsgp_kernel_hyppars[0])
        optimized_config['hsgp_error_fit_regpar_optimized']['ls'].append(hsgp_kernel_hyppars[1])
        optimized_config['hsgp_error_fit_regpar_optimized']['alpha'].append(hsgp_kernel_hyppars[2])





       
        # GPR fit rigourously accounting for y-errors AND x-errors
        nigpr_object = GPR1D.GaussianProcessRegression1D()
        nigpr_object.set_kernel(kernel=kernel)
        nigpr_object.set_error_kernel(kernel=error_kernel)
        nigpr_object.set_raw_data(xdata=X_reduced,ydata=Y_reduced,yerr=Y_errors,xerr=X_errors, dxdata=[0.0],dydata=[0.0],dyerr=[0.0])
        nigpr_object.set_search_parameters(epsilon=1.0e-2)
        nigpr_object.set_error_search_parameters(epsilon=1.0e-1)
        #     Uncomment any of the following lines to test the recommended optimizers
        #nigpr_object.set_search_parameters(epsilon=1.0e-2,method='adam',spars=[1.0e-1,0.4,0.8])
        #nigpr_object.set_error_search_parameters(epsilon=1.0e-1,method='adam',spars=[1.0e-1,0.4,0.8])
        #     Perform the fit with kernel restarts, here is the extra option to account for x-errors in fit
        nigpr_object.GPRFit(fit_x_values,hsgp_flag=True,nigp_flag=True,nrestarts=5)
        (nigp_kernel_name,nigp_kernel_hyppars,nigp_fit_regpar) = nigpr_object.get_gp_kernel_details()
        (nigp_error_kernel_name,nigp_error_kernel_hyppars,nigp_error_fit_regpar) = nigpr_object.get_gp_error_kernel_details()



        optimized_config['nigp_fit_regpar_optimized']['regularaiztion'].append(nigp_fit_regpar)
        optimized_config['nigp_fit_regpar_optimized']['amp'].append(nigp_kernel_hyppars[0])
        optimized_config['nigp_fit_regpar_optimized']['ls'].append(nigp_kernel_hyppars[1])
        optimized_config['nigp_fit_regpar_optimized']['alpha'].append(nigp_kernel_hyppars[2])
        optimized_config['nigp_error_fit_regpar_optimized']['regularaiztion'].append(nigp_fit_regpar)
        optimized_config['nigp_error_fit_regpar_optimized']['amp'].append(nigp_kernel_hyppars[0])
        optimized_config['nigp_error_fit_regpar_optimized']['ls'].append(nigp_kernel_hyppars[1])
        optimized_config['nigp_error_fit_regpar_optimized']['alpha'].append(nigp_kernel_hyppars[2])



    optimized_values['gp_fit_regpar_optimized']['regularaiztion']                 = np.mean(optimized_config['gp_fit_regpar_optimized']['regularaiztion'])
    optimized_values['gp_fit_regpar_optimized']['amp']                          = np.mean(optimized_config['gp_fit_regpar_optimized']['amp'])
    optimized_values['gp_fit_regpar_optimized']['ls']                            = np.mean(optimized_config['gp_fit_regpar_optimized']['ls'])
    optimized_values['gp_fit_regpar_optimized']['alpha']                          = np.mean(optimized_config['gp_fit_regpar_optimized']['alpha'])
    optimized_values['hsgp_fit_regpar_optimized']['regularaiztion']               = np.mean(optimized_config['hsgp_fit_regpar_optimized']['regularaiztion'])
    optimized_values['hsgp_fit_regpar_optimized']['amp']                        = np.mean(optimized_config['hsgp_fit_regpar_optimized']['amp'])
    optimized_values['hsgp_fit_regpar_optimized']['ls']                          = np.mean(optimized_config['hsgp_fit_regpar_optimized']['ls'])
    optimized_values['hsgp_fit_regpar_optimized']['alpha']                        = np.mean(optimized_config['hsgp_fit_regpar_optimized']['alpha'])
    optimized_values['hsgp_error_fit_regpar_optimized']['regularaiztion']         = np.mean(optimized_config['hsgp_error_fit_regpar_optimized']['regularaiztion'])
    optimized_values['hsgp_error_fit_regpar_optimized']['amp']                  = np.mean(optimized_config['hsgp_error_fit_regpar_optimized']['amp'])
    optimized_values['hsgp_error_fit_regpar_optimized']['ls']                    = np.mean(optimized_config['hsgp_error_fit_regpar_optimized']['ls'])
    optimized_values['hsgp_error_fit_regpar_optimized']['alpha']                  = np.mean(optimized_config['hsgp_error_fit_regpar_optimized']['alpha'])
    optimized_values['nigp_fit_regpar_optimized']['regularaiztion']               = np.mean(optimized_config['nigp_fit_regpar_optimized']['regularaiztion'])
    optimized_values['nigp_fit_regpar_optimized']['amp']                        = np.mean(optimized_config['nigp_fit_regpar_optimized']['amp'])
    optimized_values['nigp_fit_regpar_optimized']['ls']                          = np.mean(optimized_config['nigp_fit_regpar_optimized']['ls'])
    optimized_values['nigp_fit_regpar_optimized']['alpha']                        = np.mean(optimized_config['nigp_fit_regpar_optimized']['alpha'])
    optimized_values['nigp_error_fit_regpar_optimized']['regularaiztion']         = np.mean(optimized_config['nigp_error_fit_regpar_optimized']['regularaiztion'])
    optimized_values['nigp_error_fit_regpar_optimized']['amp']                  = np.mean(optimized_config['nigp_error_fit_regpar_optimized']['amp'])
    optimized_values['nigp_error_fit_regpar_optimized']['ls']                    = np.mean(optimized_config['nigp_error_fit_regpar_optimized']['ls'])
    optimized_values['nigp_error_fit_regpar_optimized']['alpha']                  = np.mean(optimized_config['nigp_error_fit_regpar_optimized']['alpha'])
    
    return optimized_values