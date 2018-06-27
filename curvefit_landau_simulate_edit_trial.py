#!/usr/bin/env python
import sys, os, ScopeTrace, pylandau
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from itertools import product
from random import choice
#-------------------------------------------------------------------------------
#simulate pulse
x = np.array(np.linspace(0, 2000, 1000))
mpv_x = np.loadtxt('mpv_data.csv', unpack = True, delimiter = ',')
mpv = choice(mpv_x)
mpv2= mpv/2
mpv3 = 1200
eta_x = np.loadtxt('eta_data.csv', unpack = True, delimiter = ',')
eta = choice(eta_x)
eta2 = eta/2

A_x = np.loadtxt('A_data.csv', unpack = True, delimiter = ',')
A = choice(A_x)
A2= A/2
jitter_x = np.loadtxt('jitter_data.csv', unpack = True, delimiter = ',')
jitter = choice(jitter_x)

parameter = [mpv, eta, A]
parameter2= [mpv2, eta2, A2]
parameter3 = [mpv3, eta, A2]

y = pylandau.landau(x, *parameter)
y2 = pylandau.landau(x, *parameter2)
y3 = pylandau.landau(x, *parameter3)

new_y = [value + np.random.normal(0, np.sqrt(jitter)) for value in y]
new_y2 = [y2[i] + new_y[i] for i in range(len(y))]
new_ym = [new_y2[i] + y3[i] for i in range(len(y))]

plt.xlabel('Time [nanoseconds]')
plt.ylabel('Voltage [V]')
plt.title('mpv, eta, A, jitter = '+str(parameter[0])+', '+ str(parameter[1])+', '+ str(parameter[2])+', '+ str(np.around(jitter, decimals = 6)))

#Interpolation and find local minima
new_x =  np.array(np.linspace(0, 2000, 30))
intp = np.interp(new_x, x, new_ym)
min_ind = np.ndarray.tolist(argrelextrema(intp, np.less, order  =1)[0])
print('Indices at local minima are ' + str(min_ind))
min_x =[int(new_x[i]) for i in min_ind]
min_y = [int(intp[i]) for i in min_ind]
#plt.plot(new_x,intp, 'm', label = 'Interpolation')
plt.plot(x, new_ym, 'b', alpha = .3)
plt.plot(new_x, intp, 'r')
plt.scatter([new_x[min_ind_v] for min_ind_v in min_ind], [intp[min_ind_v] for min_ind_v in min_ind])
plt.show()
