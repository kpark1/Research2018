#!/usr/bin/env python
import sys, os, ScopeTrace, pylandau
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from itertools import product
from random import choice
import numpy.polynomial.polynomial as poly

#-------------------------------------------------------------------------------
#simulate pulse
x = np.array(np.linspace(0, 2000, 1000))
mpv_x = np.loadtxt('mpv_data.csv', unpack = True, delimiter = ',')
mpv = choice(mpv_x)
mpv2= mpv/2 +60
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

y1 = pylandau.landau(x, *parameter)
y2 = pylandau.landau(x, *parameter2)
y3 = pylandau.landau(x, *parameter3)

new_y = [value + np.random.normal(0, np.sqrt(jitter)) for value in y1]
new_y2 = [y2[i] + new_y[i] for i in range(len(y1))]
#new_ym = [new_y2[i] + y3[i] for i in range(len(y1))]
y = new_y2

plt.title('mpv, eta, A, jitter = '+str(parameter[0])+', '+ str(parameter[1])+', '+ str(parameter[2])+', '+ str(np.around(jitter, decimals = 6)))

#Interpolation and find local minima
y_array = np.array(y)
idx = np.where(y_array == y_array.max())
#if multiple x values of the same max y values, select the first max
idx = idx[0][0]
x_values_peak = x[idx]

#Curvefit for Landau with parameters mpv,  eta, A 
mpv = x_values_peak
rmin= 1
       
try: 
    landau_par_rmin, pcov_rmin = curve_fit(pylandau.landau, x, y, p0= (mpv, rmin, rmin))
    array_1 = np.ndarray.tolist(np.around(landau_par_rmin, decimals = 3))
    param_list = [array_1]
    workinglandaupar = [array_1]
    def diff_sq_fn(parameter):
                return round(sum((y-pylandau.landau(x, *parameter))**2), 4)
        #print('The first diff is ' +str(diff_sq_fn(landau_par_rmin))+' for initial parameters ' +str(landau_par_rmin))
    initial_diff_sq = [diff_sq_fn(landau_par_rmin)]
except RuntimeError as message:
    print(str(message))
except TypeError as message:
    print(str(message) )


#plt.plot(x,y, 'r', alpha = .4)


#print(len(y), len(pylandau.landau(x, *workinglandaupar)))
y2 = [y[i]-landau[i] for i in range(len(y))]
#print(pylandau.landau(x, *workinglandaupar))
#plt.plot(x, y2, 'g', alpha= .6)



for i in range(len(y)):
    if landau[i] >0.001:
        break
yfirst = [0]*i

ymm =yfirst+ y[i:]
ym = [float(ymmm) for ymmm in ymm]


yf= [y[i] -ym[i] for i in range(len(y))]

y = yf

y_array = np.array(y)
idx = np.where(y_array == y_array.max())
#if multiple x values of the same max y values, select the first max
idx = idx[0][0]
x_values_peak = x[idx]

#Curvefit for Landau with parameters mpv,  eta, A 

    

plt.plot(x,yf, 'g', alpha = .6,label = 'final')

plt.plot(x, pylandau.landau(x, *workinglandaupar), 'b', label = 'final')
plt.legend()
plt.show()
