#!/usr/bin/env python
import sys, os, ScopeTrace, pylandau, ScopeTrace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from scipy.optimize import curve_fit
from scipy import signal
from scipy import interpolate
from itertools import product

#-------------------------------------------------------------------------------
#loop through all files
mlp.rcParams['axes.linewidth'] = 2
#for file in sys.argv[1:]:
    #with open('data/'+file, "r") as file1: 
for file in sorted(os.listdir("/home/kpark1/Work/SLab/data")) :
    with open('data/'+file, "r") as file1: 
        print(file)
        data= file1.read()
# decode the scope trace
        trace = ScopeTrace.ScopeTrace(data,1)
        
# find baseline and jitter 
        baseline,jitter = trace.find_baseline_and_jitter(0,250)
        
#set x and y 
        inverted_yvalues = []
        for value in trace.yvalues:
            inverted_yvalue = -(value-baseline)
            inverted_yvalues.append(inverted_yvalue)
        #locate pulses
        x = trace.xvalues
        y = inverted_yvalues
        max_peak_width = 500
        peak_widths = np.arange(1, max_peak_width)
        peaks = signal.find_peaks_cwt(np.array(y), peak_widths)
        pulses = []
        plt.plot(x,y)
        plt.ylim(-.004, .3)
        plt.legend()
        plt.show()
        for each_peak in peaks:
            if inverted_yvalues[each_peak] > 
            #if inverted_yvalues[each_peak]> 3.5*np.mean(inverted_yvalues[800:]):
                pulses.append(each_peak)                   
        print(pulses)
        n_of_pulses = len(pulses)
        
        if n_of_pulses == 0:
            continue
        elif n_of_pulses ==1:
            continue
 #blah 
        elif n_of_pulses >1:
            #find ranges
            
            start  = [0]
            for i in range(n_of_pulses):
                print(start[0])
                halfwidth = peaks[i] - start[0]
                print('halfwidth: '+str(halfwidth))
                y_new = y[peaks[i]:(peaks[i] +halfwidth)]
                mean = np.mean(y_new)
                print(mean)
                y_closest_to = np.around(min(y_new, key = lambda y:abs(y-mean)), decimals = 5)
                idx = np.where(np.around(y_new, decimals =5)==y_closest_to) 
                idx = idx + peaks[i]
                true_idx =idx[0][0]
                
                print('trueidx'+str(true_idx))
                del start[0]
                start.append(true_idx)
                print('start' + str(start))

        
