#!/usr/bin/env python
import sys, os, ScopeTrace, pylandau, ScopeTrace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from scipy.optimize import curve_fit
from itertools import product
import csv
#-------------------------------------------------------------------------------
#loop through all files
mlp.rcParams['axes.linewidth'] = 2
jitter_stored = []
mpv_stored = []
eta_stored = []
A_stored = []
for file in sorted(os.listdir("/home/kpark1/Work/SLab/data/")) :

    
    with open('data/'+file, "r") as file1: 
        data= file1.read()
        print(str(file1) + ':\n')
    
# decode the scope trace
        trace = ScopeTrace.ScopeTrace(data,1)
# find baseline and jitter 
        baseline,jitter = trace.find_baseline_and_jitter(0,250)
        jitter_stored.append(jitter)
#set x and y
        inverted_yvalues = []
        for value in trace.yvalues:
            inverted_yvalue = -(value-baseline)
            inverted_yvalues.append(inverted_yvalue)
        x = np.array(trace.xvalues)
        y = np.array(inverted_yvalues)    
#x values at peaks
        y_array = np.array(y)
        idx = np.where(y_array == y_array.max())
#if multiple x values of the same max y values, select the first max
        idx = idx[0][0]
        x_values_peak = x[idx]

#GAUSSIAN
        mean = x_values_peak 
        a = max(y)
        g_par = []
        def gaus(x, a, x0, sigma):
            return a* np.exp(-(x-x0)**2/(2*sigma**2))
        def diff_sq_fn(parameter):
            return round(sum((y-gaus(x, *parameter))**2), 4)
        for sigma in range(1, 35):
            popt, pcov = curve_fit(gaus, x, y, p0 = [a,mean, sigma])
            if diff_sq_fn(popt) < .5:
                g_par.append(popt)
                break
            else:
                continue
        g_par = g_par[0]
        plt.plot(x, y, 'b', label = 'Oscilloscope Data')
        plt.plot(x, gaus(x, *g_par), 'g', label = 'Gaussian Fit')
        plt.show()
       
       
