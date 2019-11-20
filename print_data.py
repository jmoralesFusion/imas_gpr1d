# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab

from __future__ import (unicode_literals, absolute_import,print_function, division)
# Standard python modules
import os
import re
import sys
import warnings
import time
#import pdb; pdb.set_trace()

import getpass

import numpy as np
import matplotlib.pyplot as plt


def print_data(gp_kernel_name,gp_kernel_hyppars,gp_fit_regpar,fit_lml,\
                   hsgp_kernel_name,hsgp_fit_regpar,hsgp_kernel_hyppars,\
                   hsgp_error_kernel_name,hsgp_error_fit_regpar,hsgp_error_kernel_hyppars,hs_fit_lml,\
                   nigp_kernel_name,nigp_fit_regpar,nigp_kernel_hyppars,\
                   nigp_error_kernel_name,nigp_error_fit_regpar,nigp_error_kernel_hyppars,ni_fit_lml):
    ###Printing
    gp_str = "\n--- GPR Fit ---\n\n"
    gp_str = gp_str + "Kernel name: %30s\n" % (gp_kernel_name)
    gp_str = gp_str + "Regularization parameter: %17.4f\n" % (gp_fit_regpar)
    gp_str = gp_str + "Optimized kernel hyperparameters:\n"
    for hh in np.arange(0,gp_kernel_hyppars.size):
        gp_str = gp_str + "%15.6e" % (gp_kernel_hyppars[hh])
    gp_str = gp_str + "\n\n"
    gp_str = gp_str + "Log-marginal-likelihood: %18.6f\n" % (fit_lml)

    print(gp_str)

    hsgp_str = "\n--- HSGPR Fit ---\n\n"
    hsgp_str = hsgp_str + "Kernel name: %30s\n" % (hsgp_kernel_name)
    hsgp_str = hsgp_str + "Regularization parameter: %17.4f\n" % (hsgp_fit_regpar)
    hsgp_str = hsgp_str + "Optimized kernel hyperparameters:\n"
    for hh in np.arange(0,hsgp_kernel_hyppars.size):
        hsgp_str = hsgp_str + "%15.6e" % (hsgp_kernel_hyppars[hh])
    hsgp_str = hsgp_str + "\n\n"
    hsgp_str = hsgp_str + "Error kernel name: %24s\n" % (hsgp_error_kernel_name)
    hsgp_str = hsgp_str + "Regularization parameter: %17.4f\n" % (hsgp_error_fit_regpar)
    hsgp_str = hsgp_str + "Optimized error kernel hyperparameters:\n"
    for hh in np.arange(0,hsgp_error_kernel_hyppars.size):
        hsgp_str = hsgp_str + "%15.6e" % (hsgp_error_kernel_hyppars[hh])
    hsgp_str = hsgp_str + "\n\n"
    hsgp_str = hsgp_str + "Log-marginal-likelihood: %18.6f\n" % (hs_fit_lml)

    print(hsgp_str)

    nigp_str = "--- NIGPR Fit ---\n\n"
    nigp_str = nigp_str + "Kernel name: %30s\n" % (nigp_kernel_name)
    nigp_str = nigp_str + "Regularization parameter: %17.4f\n" % (nigp_fit_regpar)
    nigp_str = nigp_str + "Optimized kernel hyperparameters:\n"
    for hh in np.arange(0,nigp_kernel_hyppars.size):
        nigp_str = nigp_str + "%15.6e" % (nigp_kernel_hyppars[hh])
    nigp_str = nigp_str + "\n\n"
    nigp_str = nigp_str + "Error kernel name: %24s\n" % (nigp_error_kernel_name)
    nigp_str = nigp_str + "Regularization parameter: %17.4f\n" % (nigp_error_fit_regpar)
    nigp_str = nigp_str + "Optimized error kernel hyperparameters:\n"
    for hh in np.arange(0,nigp_error_kernel_hyppars.size):
        nigp_str = nigp_str + "%15.6e" % (nigp_error_kernel_hyppars[hh])
    nigp_str = nigp_str + "\n\n"
    nigp_str = nigp_str + "Log-marginal-likelihood: %18.6f\n" % (ni_fit_lml)

    print(nigp_str)
