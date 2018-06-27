#!/usr/bin/env python
import sys, os, ScopeTrace, pylandau, ScopeTrace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy import interpolate
from itertools import product
#import warnings
#-------------------------------------------------------------------------------


#determine full width at half maximum
        def fwhm(y_list ):
            y_closest_to_hm= min(y_list, key= lambda x: abs(x-.5*max(y_list)))
            idx_hm_left= np.where(y_array[:idx] == y_closest_to_hm)
            idx_hm_right = np.where(y_array[idx:]==y_closest_to_hm)
            x_hm_left = x[idx_hm_left]
            x_hm_right = x[idx_hm_right]
            fw =  x_hm_left - x_hm_right
            return fw
     
