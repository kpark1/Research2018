#!/usr/bin/env python
import sys, os, ScopeTrace, pylandau, ScopeTrace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from itertools import product
import csv
import random
#-------------------------------------------------------------------------------
#mean jitter = .0000025
x = np.array(np.linspace(0, 1000, 1000))
mpv_x = np.loadtxt('mpv_data.csv', unpack = True, delimiter = ',')
mpv = random.choice(mpv_x)
mpv2= mpv/2
eta_x = np.loadtxt('eta_data.csv', unpack = True, delimiter = ',')
eta = random.choice(eta_x)
eta2 = eta/2
A_x = np.loadtxt('A_data.csv', unpack = True, delimiter = ',')
A = random.choice(A_x)
A2= A/2
jitter_x = np.loadtxt('jitter_data.csv', unpack = True, delimiter = ',')
jitter = random.choice(jitter_x)

mpv= 358.742
eta= 29.766
A=.208
jitter=2e-06

parameter = [mpv, eta, A]
parameter2= [mpv2, eta2, A2]

y = pylandau.landau(x, *parameter)
y2 = pylandau.landau(x, *parameter2)

new_y = [value + np.random.normal(0, np.sqrt(jitter)) for value in y]
new_y_2_pulses = [y2[i] + new_y[i] for i in range(len(y))]

#plt.plot(x, new_y,color = 'r')
plt.plot(x, new_y_2_pulses, color = 'b')
#plt.plot(x, y, color ='b')
plt.xlabel('Time [nanoseconds]')
plt.ylabel('Voltage [V]')
plt.ylim(-.01, .25)
plt.title('mpv, eta, A, jitter = '+str(parameter[0])+', '+ str(parameter[1])+', '+ str(parameter[2])+', '+ str(np.around(jitter, decimals = 6)))


#number of pulses
new_x =  np.array(np.linspace(0, 1000, 30))
y_intp = np.interp(new_x, x, new_y_2_pulses)
twominindex= argrelextrema(y_intp, np.less, order  =2)
print(twominindex)
mark_ind = np.ndarray.tolist(twominindex[0])
mark_x =[float(new_x[i]) for i in mark_ind]
mark_y = [float(y_intp[i]) for i in mark_ind]
print(mark_ind,mark_x, mark_y, mark_y[1])
plt.plot(new_x, y_intp, 'm', markevery = mark_x)

#find the first peak parameters
firstendidx= int(mark_x[1])
new_x_2=x[0:firstendidx]

new_y_2_pulses_edited = new_y_2_pulses[0:firstendidx]
n= len(new_y_2_pulses)-len(new_y_2_pulses_edited)
b= [new_y_2_pulses_edited[-1]]*n
new_y_2_pulses_edited_final = new_y_2_pulses_edited.extend(b)
#
y=new_y_2_pulses_edited
print(len(x), len(y))
plt.plot(new_x_2,new_y_2_pulses[0:firstendidx], 'r')
plt.plot(x, new_y_2_pulses_edited, 'g')

#
mpv = np.where(y == max(y))[0][0]
rmin= 1

       
landau_par_rmin, pcov_rmin = curve_fit(pylandau.landau, x, y, p0= (mpv, rmin, rmin))
array_1 = np.ndarray.tolist(np.around(landau_par_rmin, decimals = 3))
param_list = [array_1]
workinglandaupar = [array_1]
def diff_sq_fn(parameter):
    return round(sum((y-pylandau.landau(x, *parameter))**2), 4)   
initial_diff_sq = [diff_sq_fn(landau_par_rmin)]
    
#only one best possible parameter with the least diff_sq
for eta, A in product(np.linspace(rmin, 25,5 ), np.linspace(rmin, 25, 5)):
    try:
        landau_par, pcov = curve_fit(pylandau.landau, x, y, p0= (mpv, eta,A))
        landau_par = np.ndarray.tolist(np.around(landau_par, decimals =3))
        diff= diff_sq_fn(landau_par)
        par = param_list[0]
                
        if initial_diff_sq[0] < .01:
            break
        elif landau_par != par and diff < initial_diff_sq[0]:
            param_list.append(landau_par)
                    
            del workinglandaupar[0]
            del initial_diff_sq[0]
            initial_diff_sq.append(diff)
            workinglandaupar.append(landau_par)     
            break
        else:      
            continue
    except RuntimeError as message:
        print(str(message))
        continue
    except TypeError as message:
        print(str(message))
    except ValueError as message:
        print(str(message))
print(workinglandaupar)
plt.plot(x, pylandau.landau(x, *workinglandaupar[0]),'k', linewidth = 2)
plt.show()
